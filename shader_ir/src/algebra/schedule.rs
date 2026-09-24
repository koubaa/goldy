//! Schedules: how a region's outputs map onto a compute grid, and lowering to a kernel.
//!
//! Lowering first substitutes every temporary and every output into its readers, so
//! each output is one term over inputs. All outputs must have one rank once unit axes
//! are dropped, and the shared domain is their largest extent on each axis; element `t`
//! of it is one thread, or one lane group, which computes every output that spans `t`
//! and then stores them all. Inputs are read before any store, which keeps the
//! region's entry semantics for storage that an output overwrites in place.

use super::affine::{Affine, IndexParam, IndexVar, Sym};
use super::graph::Graph;
use super::region::{ParcelId, Region, RegionError, Role, Storage};
use super::term::{BinaryOp, CmpOp, ReduceOp, ReduceOrder, ScalarParam, Term, UnaryOp, ValueId};
use crate::{
    BinOp, BuiltinFn, BuiltinMask, ElementType, Expr, KernelParam, ScalarType, ShaderKernel, SourceMap, Stmt,
    UnaryOp as IrUnaryOp,
};
use std::collections::HashMap;
use std::fmt;

/// How a region's shared output domain maps onto workgroups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Schedule {
    /// One thread per element, `workgroup` threads per workgroup. Every reduction
    /// runs sequentially in its element's thread; sibling reductions over one extent
    /// share a loop.
    Threads { workgroup: u32 },
    /// One group of `lanes` threads per element, `elements` elements per workgroup.
    ///
    /// Every lane-ordered reduction outside a select is spread over the group in
    /// exactly its [`ReduceOrder::Lanes`] association. Sibling reductions with one
    /// extent and accumulator count share a strided loop and its loads, and all of
    /// them share one `exchange` tree. The rest of each output runs on the group's
    /// first lane.
    Lanes {
        lanes: u32,
        elements: u32,
        exchange: Exchange,
    },
    /// One subgroup of `subgroup` threads per [`MATRIX_TILE`](crate::MATRIX_TILE)-square tile of a
    /// two-dimensional domain.
    ///
    /// Each contraction with one summed index whose factors split into a row factor and
    /// a column factor runs on matrix units: both are rounded to f16, and their
    /// products are summed in f32 in an order the hardware chooses. Only
    /// [`ContractionPrecision::F16Factors`] admits it. Every other term of each output
    /// is evaluated per element as under [`Schedule::Threads`].
    Matrix { subgroup: u32 },
}

/// How a [`Schedule::Lanes`] group combines its lanes' partial results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Exchange {
    /// Through workgroup memory, one barrier per tree level.
    Workgroup,
    /// By reading other lanes of the subgroup. The tree is the same, so the result is
    /// too: for `lane < s`, `lane ^ s == lane + s`.
    Subgroup,
}

/// What a device offers lowering beyond portable compute; the default offers nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Target {
    /// Threads per subgroup, when that width is fixed and every subgroup of a
    /// one-dimensional workgroup is that many consecutive local invocations.
    pub subgroup: Option<u32>,
    /// Whether a subgroup multiplies [`MATRIX_TILE`](crate::MATRIX_TILE)-square f16 tiles into f32
    /// accumulators ([`crate::MatrixOp`]).
    pub matrix: bool,
}

/// The rounding a lowering may add to contractions beyond what their terms state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ContractionPrecision {
    /// Every product and sum as the region's terms and reduction orders state.
    #[default]
    Exact,
    /// A contraction's factors may be rounded to f16 and their products summed in f32
    /// in any order, which admits [`Schedule::Matrix`].
    F16Factors,
}

/// Threads per workgroup of the schedules [`lower`] chooses.
pub const WORKGROUP_THREADS: u32 = 256;
/// Threads per workgroup of a [`Schedule::Lanes`] that [`lower`] chooses.
pub const LANE_WORKGROUP_THREADS: u32 = 128;

/// Where the runtime value of an index parameter comes from: one integer element of
/// a parcel, read once per thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexSource {
    pub parcel: ParcelId,
    pub element: u32,
    /// [`ElementType::I32`] or [`ElementType::U32`].
    pub element_type: ElementType,
}

/// A region lowered to one kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct Lowered {
    pub kernel: ShaderKernel,
    pub schedule: Schedule,
    pub groups: [u32; 3],
    /// The parcel each resource parameter binds, in parameter order, and whether the
    /// kernel writes it.
    pub parcels: Vec<(ParcelId, bool)>,
    /// The scalar parameter each scalar kernel parameter binds, in order after the
    /// resources.
    pub scalars: Vec<ScalarParam>,
    pub workgroup_bytes: u32,
}

/// Why a region cannot run as one kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    Invalid(RegionError),
    NoOutputs,
    /// Two outputs have different ranks once unit axes are dropped.
    Domain {
        output: String,
        other: String,
    },
    /// The shared domain does not fit one grid.
    Grid {
        elements: u64,
    },
    /// An output's storage maps two of its elements to one location.
    NotInjective {
        output: String,
    },
    /// Two outputs can write one location.
    OutputOverlap {
        output: String,
        other: String,
    },
    /// A thread reads a location another thread's output writes.
    Race {
        input: String,
        output: String,
    },
    /// A reduction's order cannot be realized where it appears.
    Order {
        output: String,
    },
    /// An index expression can leave the 32-bit range.
    IndexRange,
    /// A parcel is used both as f32 data and as an index source, or an output writes
    /// an index source.
    IndexSource {
        param: String,
    },
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::Invalid(e) => e.fmt(f),
            LowerError::NoOutputs => f.write_str("the region has no outputs"),
            LowerError::Domain { output, other } => {
                write!(f, "`{output}` and `{other}` have different ranks")
            }
            LowerError::Grid { elements } => write!(f, "{elements} elements do not fit one grid"),
            LowerError::NotInjective { output } => write!(f, "`{output}` stores two elements in one location"),
            LowerError::OutputOverlap { output, other } => write!(f, "`{output}` and `{other}` can write one location"),
            LowerError::Race { input, output } => {
                write!(f, "`{input}` reads what another thread's `{output}` writes")
            }
            LowerError::Order { output } => write!(f, "a reduction order in `{output}` cannot be realized"),
            LowerError::IndexRange => f.write_str("an index expression can leave the 32-bit range"),
            LowerError::IndexSource { param } => {
                write!(f, "the source of index parameter `{param}` is also data or written")
            }
        }
    }
}

impl std::error::Error for LowerError {}

/// Lowers `region` for a device that offers nothing beyond portable compute; see
/// [`lower_on`].
pub fn lower(region: &Region, sources: &[IndexSource]) -> Result<Lowered, LowerError> {
    lower_on(region, sources, &Target::default())
}

/// Lowers `region` with the schedule that realizes every reduction order it names:
/// [`Schedule::Lanes`] when an output has a lane-ordered reduction outside a select,
/// exchanging through the subgroup when `target`'s holds whole lane groups, else
/// [`Schedule::Threads`]. `sources` gives each index parameter's source.
pub fn lower_on(region: &Region, sources: &[IndexSource], target: &Target) -> Result<Lowered, LowerError> {
    let prepared = Prepared::new(region)?;
    let mut lanes = None;
    for (_, body) in &prepared.bodies {
        for r in top_reductions(body) {
            if let Term::Reduce {
                order: ReduceOrder::Lanes { lanes: l, .. },
                ..
            } = r
            {
                match lanes {
                    None => lanes = Some(*l),
                    Some(other) if other != *l => return Err(prepared.order_error()),
                    Some(_) => {}
                }
            }
        }
    }
    let schedule = match lanes {
        Some(lanes) => {
            let elements = (LANE_WORKGROUP_THREADS / lanes).max(1);
            let whole = |w: u32| w >= lanes && w.is_multiple_of(lanes) && (lanes * elements).is_multiple_of(w);
            Schedule::Lanes {
                lanes,
                elements,
                exchange: match target.subgroup.is_some_and(whole) {
                    true => Exchange::Subgroup,
                    false => Exchange::Workgroup,
                },
            }
        }
        None => Schedule::Threads {
            workgroup: WORKGROUP_THREADS,
        },
    };
    prepared.lower(schedule, sources)
}

/// Lowers `graph`'s composition: on [`Schedule::Matrix`] when `precision` admits it,
/// `target` offers matrix units and the composition has a contraction it applies to,
/// else as [`lower_on`] does.
pub fn lower_graph(
    graph: &Graph,
    sources: &[IndexSource],
    target: &Target,
    precision: ContractionPrecision,
) -> Result<Lowered, LowerError> {
    if let (ContractionPrecision::F16Factors, true, Some(subgroup)) = (precision, target.matrix, target.subgroup) {
        if let Some(lowered) = super::matrix::lower(graph, sources, subgroup) {
            return Ok(lowered);
        }
    }
    lower_on(graph.region(), sources, target)
}

/// Lowers `region` with `schedule`. [`Schedule::Matrix`] needs the contractions of a
/// [`Graph`]; see [`lower_graph`].
pub fn lower_with(region: &Region, schedule: Schedule, sources: &[IndexSource]) -> Result<Lowered, LowerError> {
    let prepared = Prepared::new(region)?;
    match schedule {
        Schedule::Matrix { .. } => Err(prepared.order_error()),
        _ => prepared.lower(schedule, sources),
    }
}

/// Reductions outside any select and any other reduction, in evaluation order.
fn top_reductions(term: &Term) -> Vec<&Term> {
    fn collect<'a>(t: &'a Term, out: &mut Vec<&'a Term>) {
        match t {
            Term::Reduce { .. } => out.push(t),
            Term::Select { .. } => {}
            _ => t.for_each_child(&mut |c| collect(c, out)),
        }
    }
    let mut out = Vec::new();
    collect(term, &mut out);
    out
}

/// A region whose outputs are single terms over inputs, on shared coordinates.
pub(super) struct Prepared {
    region: Region,
    /// Extents of the shared domain.
    pub(super) extents: Vec<u32>,
    pub(super) coords: Vec<IndexVar>,
    /// Each output and its body on `coords`.
    bodies: Vec<(ValueId, Term)>,
    /// Each output's index on `coords`.
    at: Vec<Vec<Affine>>,
    /// Per output, the axes of `coords` it spans only a prefix of, and that prefix.
    limits: Vec<Vec<(usize, u32)>>,
}

impl Prepared {
    pub(super) fn new(region: &Region) -> Result<Self, LowerError> {
        region.validate().map_err(LowerError::Invalid)?;
        let mut region = region.clone();
        region.inline_temporaries();
        let outputs: Vec<ValueId> = region
            .values()
            .filter(|(_, v)| v.role == Role::Output)
            .map(|(id, _)| id)
            .collect();
        for &o in &outputs {
            region.substitute(o).expect("an output has a definition");
        }
        let &first = outputs.first().ok_or(LowerError::NoOutputs)?;
        let squeeze = |shape: &[u32]| shape.iter().copied().filter(|&e| e != 1).collect::<Vec<_>>();
        let mut extents = squeeze(&region.value(first).expect("live").shape);
        for &o in &outputs {
            let value = region.value(o).expect("live");
            let squeezed = squeeze(&value.shape);
            if squeezed.len() != extents.len() {
                return Err(LowerError::Domain {
                    output: value.name.clone(),
                    other: region.value(first).expect("live").name.clone(),
                });
            }
            for (e, s) in extents.iter_mut().zip(squeezed) {
                *e = (*e).max(s);
            }
        }
        let coords: Vec<IndexVar> = (0..extents.len()).map(|a| region.index(&format!("c{a}"))).collect();
        let mut bodies = Vec::new();
        let mut at = Vec::new();
        let mut limits = Vec::new();
        for &o in &outputs {
            let value = region.value(o).expect("live");
            limits.push(
                squeeze(&value.shape)
                    .into_iter()
                    .zip(&extents)
                    .enumerate()
                    .filter(|(_, (s, e))| s < e)
                    .map(|(a, (s, _))| (a, s))
                    .collect(),
            );
            let def = value.definition.as_ref().expect("an output has a definition");
            let mut next = coords.iter();
            let index: Vec<Affine> = value
                .shape
                .iter()
                .map(|&e| match e {
                    1 => Affine::constant(0),
                    _ => Affine::from(*next.next().expect("same squeezed rank")),
                })
                .collect();
            let map: HashMap<IndexVar, Affine> = def.domain.iter().copied().zip(index.iter().cloned()).collect();
            bodies.push((o, def.body.instantiate(&|v| map.get(&v).cloned(), &HashMap::new())));
            at.push(index);
        }
        let prepared = Self {
            region,
            extents,
            coords,
            bodies,
            at,
            limits,
        };
        prepared.check()?;
        Ok(prepared)
    }

    fn name(&self, id: ValueId) -> &str {
        &self.region.value(id).expect("live").name
    }

    fn storage(&self, id: ValueId) -> &Storage {
        self.region
            .value(id)
            .expect("live")
            .storage
            .as_ref()
            .expect("validated")
    }

    fn order_error(&self) -> LowerError {
        LowerError::Order {
            output: self.name(self.bodies[0].0).to_string(),
        }
    }

    fn param_range(&self, p: IndexParam) -> (i64, i64) {
        let r = self.region.param_range(p);
        (r.start, r.end - 1)
    }

    /// The obligations of one kernel writing every output: injective stores, no two
    /// outputs on one location, and no read of a location another thread writes.
    fn check(&self) -> Result<(), LowerError> {
        let shape = |id: ValueId| self.region.value(id).expect("live").shape.clone();
        let interval = |id: ValueId| self.storage(id).interval(&shape(id), &|p| self.param_range(p));
        for (k, &(o, _)) in self.bodies.iter().enumerate() {
            if !self.storage(o).injective(&shape(o)) {
                return Err(LowerError::NotInjective {
                    output: self.name(o).to_string(),
                });
            }
            for &(other, _) in &self.bodies[..k] {
                let overlap = match (interval(o), interval(other)) {
                    (Some((a, b)), Some((c, d))) => a <= d && c <= b,
                    _ => false,
                };
                if self.storage(o).parcel == self.storage(other).parcel && overlap {
                    return Err(LowerError::OutputOverlap {
                        output: self.name(o).to_string(),
                        other: self.name(other).to_string(),
                    });
                }
            }
        }
        let mut ranges: HashMap<Sym, (i64, i64)> = self
            .coords
            .iter()
            .zip(&self.extents)
            .map(|(&c, &e)| (Sym::Index(c), (0, i64::from(e) - 1)))
            .collect();
        for (_, body) in &self.bodies {
            let mut failure = None;
            body.walk_reads(&mut ranges, &mut |value, index, ranges| {
                if failure.is_some() {
                    return;
                }
                let storage = self.storage(value);
                let element = storage.element(index);
                let (lo, hi) = element
                    .bounds(&|s| match s {
                        Sym::Param(p) => Some(self.param_range(p)),
                        Sym::Index(_) => ranges.get(&s).copied(),
                    })
                    .expect("validated reads are bounded");
                for ((o, _), at) in self.bodies.iter().zip(&self.at) {
                    let written = self.storage(*o);
                    if written.parcel != storage.parcel || written.element(at) == element {
                        continue;
                    }
                    if interval(*o).is_some_and(|(a, b)| a <= hi && lo <= b) {
                        failure = Some(LowerError::Race {
                            input: self.name(value).to_string(),
                            output: self.name(*o).to_string(),
                        });
                        return;
                    }
                }
            });
            if let Some(failure) = failure {
                return Err(failure);
            }
        }
        Ok(())
    }

    /// The resource and scalar parameters: every parcel an input, output or index
    /// source names, then every scalar.
    pub(super) fn resources(&self, sources: &[IndexSource]) -> Result<Resources, LowerError> {
        if let Some(p) = self.region.index_params().nth(sources.len()) {
            return Err(LowerError::IndexSource {
                param: self.region.param_name(p).to_string(),
            });
        }
        let mut parcels: Vec<(ParcelId, bool)> = Vec::new();
        let mut types: HashMap<ParcelId, ElementType> = HashMap::new();
        let mut note = |parcel: ParcelId, written: bool, ty: ElementType| -> bool {
            if types.insert(parcel, ty).is_some_and(|t| t != ty) {
                return false;
            }
            match parcels.iter_mut().find(|(p, _)| *p == parcel) {
                Some(entry) => entry.1 |= written,
                None => parcels.push((parcel, written)),
            }
            true
        };
        for (id, value) in self.region.values() {
            if value.role == Role::Temporary || (value.role == Role::Input && self.region.readers(id) == 0) {
                continue;
            }
            note(
                value.storage.as_ref().expect("validated").parcel,
                value.role == Role::Output,
                ElementType::F32,
            );
        }
        for (p, source) in sources.iter().enumerate() {
            if !note(source.parcel, false, source.element_type) {
                return Err(LowerError::IndexSource {
                    param: self.region.param_name(IndexParam(p as u32)).to_string(),
                });
            }
        }
        for (p, source) in sources.iter().enumerate() {
            if parcels.iter().any(|&(q, written)| q == source.parcel && written) {
                return Err(LowerError::IndexSource {
                    param: self.region.param_name(IndexParam(p as u32)).to_string(),
                });
            }
        }
        let mut params: Vec<KernelParam> = parcels
            .iter()
            .map(|&(parcel, written)| {
                let name = parcel_name(parcel);
                let ty = types[&parcel];
                if written {
                    KernelParam::buffer_read_write(name, ty)
                } else {
                    KernelParam::buffer_read(name, ty)
                }
            })
            .collect();
        let scalars: Vec<ScalarParam> = self.region.scalars().collect();
        params.extend(
            scalars
                .iter()
                .map(|s| KernelParam::scalar_param(scalar_name(*s), ScalarType::F32)),
        );
        Ok(Resources {
            params,
            parcels,
            scalars,
        })
    }

    /// An emitter with each coordinate bound to its local `c{n}`.
    pub(super) fn emitter(&self) -> Emit<'_> {
        let mut emit = Emit {
            region: &self.region,
            names: HashMap::new(),
            ranges: HashMap::new(),
            fresh: 0,
            cache: Vec::new(),
            loads: None,
            depth: 0,
            output: self.name(self.bodies[0].0).to_string(),
        };
        for (&c, &e) in self.coords.iter().zip(&self.extents) {
            emit.bind(c, coord_name(c), e);
        }
        emit
    }

    /// `q{p}`: the value of each index parameter, read from its source.
    pub(super) fn index_params(&self, sources: &[IndexSource], body: &mut Vec<Stmt>) {
        for (p, source) in sources.iter().enumerate() {
            let at = Expr::Index {
                base: Box::new(var(&parcel_name(source.parcel))),
                index: Box::new(Expr::LitU32(source.element)),
            };
            body.push(let_int(&format!("q{p}"), cast(at, "int")));
        }
    }

    fn lower(&self, schedule: Schedule, sources: &[IndexSource]) -> Result<Lowered, LowerError> {
        let Resources {
            params,
            parcels,
            scalars,
        } = self.resources(sources)?;
        let elements: u64 = self.extents.iter().map(|&e| u64::from(e)).product();
        let per_group = match schedule {
            Schedule::Threads { workgroup } => workgroup,
            Schedule::Lanes { elements, .. } => elements,
            Schedule::Matrix { .. } => unreachable!("matrix schedules lower from a graph"),
        };
        let groups = elements.div_ceil(u64::from(per_group.max(1)));
        if elements > i32::MAX as u64 || groups > 65_535 {
            return Err(LowerError::Grid { elements });
        }

        let mut emit = self.emitter();
        let mut body = Vec::new();
        let builtins = match schedule {
            Schedule::Threads { .. } => {
                body.push(let_int("t", cast(field(BuiltinFn::GlobalId), "int")));
                BuiltinMask {
                    global_id: true,
                    ..BuiltinMask::NONE
                }
            }
            Schedule::Lanes { lanes, elements, .. } => {
                body.push(let_int("local", cast(field(BuiltinFn::LocalId), "int")));
                body.push(let_int("lane", bin(BinOp::Rem, var("local"), int(lanes as i64)?)));
                body.push(let_int(
                    "t",
                    bin(
                        BinOp::Add,
                        bin(
                            BinOp::Mul,
                            cast(field(BuiltinFn::WorkgroupId), "int"),
                            int(elements as i64)?,
                        ),
                        bin(BinOp::Div, var("local"), int(lanes as i64)?),
                    ),
                ));
                BuiltinMask {
                    local_id: true,
                    workgroup_id: true,
                    ..BuiltinMask::NONE
                }
            }
            Schedule::Matrix { .. } => unreachable!("matrix schedules lower from a graph"),
        };
        body.push(Stmt::Let {
            name: "valid".into(),
            mutable: false,
            ty: Some("bool".into()),
            init: bin(BinOp::Lt, var("t"), int(elements as i64)?),
        });
        let mut rest = var("t");
        for (a, (&c, &e)) in self.coords.iter().zip(&self.extents).enumerate().rev() {
            let name = coord_name(c);
            if a == 0 {
                body.push(let_int(&name, rest.clone()));
            } else {
                body.push(let_int(&name, bin(BinOp::Rem, rest.clone(), int(i64::from(e))?)));
                body.push(let_int(&format!("rest{a}"), bin(BinOp::Div, rest, int(i64::from(e))?)));
                rest = var(&format!("rest{a}"));
            }
        }
        self.index_params(sources, &mut body);

        let mut workgroup_bytes = 0u32;
        if let Schedule::Lanes {
            lanes,
            elements,
            exchange,
        } = schedule
        {
            let is_lanes = |r: &&Term| {
                matches!(
                    r,
                    Term::Reduce {
                        order: ReduceOrder::Lanes { .. },
                        ..
                    }
                )
            };
            // Lane groups reduce over the whole domain, so an output spanning part of it
            // may only reuse a reduction an output spanning all of it computes.
            let pending = self.top_reductions(true, is_lanes);
            for ((_, term), limits) in self.bodies.iter().zip(&self.limits) {
                if !limits.is_empty()
                    && top_reductions(term)
                        .into_iter()
                        .filter(is_lanes)
                        .any(|r| !pending.iter().any(|p| p.alpha_eq(r)))
                {
                    return Err(self.order_error());
                }
            }
            let mut work = Vec::new();
            let mut reduced = Vec::new();
            for group in siblings(pending) {
                let partials = emit.lanes(&group, lanes, &mut work)?;
                reduced.extend(group.into_iter().zip(partials));
            }
            let mut decls = Vec::new();
            workgroup_bytes = emit.exchange(&reduced, lanes, elements, exchange, &mut decls, &mut work)?;
            body.extend(decls);
            body.extend(work);
        }
        let values = self.outputs(&mut emit)?;
        let cond = match schedule {
            Schedule::Lanes { .. } => bin(BinOp::And, var("valid"), bin(BinOp::Eq, var("lane"), int(0)?)),
            _ => var("valid"),
        };
        body.push(Stmt::If {
            cond,
            then_body: values,
            else_body: None,
        });

        let workgroup_size = match schedule {
            Schedule::Lanes { lanes, elements, .. } => [lanes * elements, 1, 1],
            Schedule::Threads { workgroup } | Schedule::Matrix { subgroup: workgroup } => [workgroup, 1, 1],
        };
        Ok(Lowered {
            kernel: ShaderKernel {
                name: "tensor_region".into(),
                workgroup_size,
                params,
                builtins,
                body,
                source_map: SourceMap::default(),
                type_decls: Vec::new(),
            },
            schedule,
            groups: [groups as u32, 1, 1],
            parcels,
            scalars,
            workgroup_bytes,
        })
    }

    /// The distinct reductions `keep` selects outside any select and any other
    /// reduction of the outputs, in evaluation order; with `whole`, of only the
    /// outputs that span the whole domain.
    pub(super) fn top_reductions(&self, whole: bool, keep: impl Fn(&&Term) -> bool) -> Vec<&Term> {
        let mut found: Vec<&Term> = Vec::new();
        for ((_, term), limits) in self.bodies.iter().zip(&self.limits) {
            if whole && !limits.is_empty() {
                continue;
            }
            for r in top_reductions(term).into_iter().filter(&keep) {
                if !found.iter().any(|f| f.alpha_eq(r)) {
                    found.push(r);
                }
            }
        }
        found
    }

    /// Computes every output of element `c{n}` into a local and stores them all, the
    /// stores after every computation. Sequential sibling reductions of the outputs
    /// that span the whole domain share loops.
    pub(super) fn outputs(&self, emit: &mut Emit) -> Result<Vec<Stmt>, LowerError> {
        let mut values = Vec::new();
        let sequential = self.top_reductions(true, |r| {
            matches!(
                r,
                Term::Reduce {
                    order: ReduceOrder::Sequential,
                    ..
                }
            ) && emit.cached(r).is_none()
        });
        for group in siblings(sequential).into_iter().filter(|g| g.len() > 1) {
            emit.sequential(&group, &mut values)?;
        }
        let mut stores = Vec::new();
        for (((o, term), at), limits) in self.bodies.iter().zip(&self.at).zip(&self.limits) {
            emit.output = self.name(*o).to_string();
            let name = emit.fresh("o");
            let element = self.storage(*o).element(at);
            let store = Stmt::Assign {
                target: Expr::Index {
                    base: Box::new(var(&parcel_name(self.storage(*o).parcel))),
                    index: Box::new(cast(emit.affine(&element)?, "uint")),
                },
                value: var(&name),
            };
            let Some(within) = limits
                .iter()
                .map(|&(a, e)| Ok(bin(BinOp::Lt, var(&coord_name(self.coords[a])), int(i64::from(e))?)))
                .reduce(|a, b| Ok(bin(BinOp::And, a?, b?)))
                .transpose()?
            else {
                let x = emit.term(term, &mut values)?;
                values.push(let_float(&name, x));
                stores.push(store);
                continue;
            };
            // Outside its prefix the output's reads can leave their storage, and what
            // it computes inside the guard is local to it.
            values.push(let_float(&name, float(0.0)));
            let cached = emit.cache.len();
            let mut guarded = Vec::new();
            let x = emit.term(term, &mut guarded)?;
            emit.cache.truncate(cached);
            guarded.push(assign(&name, x));
            values.push(Stmt::If {
                cond: within.clone(),
                then_body: guarded,
                else_body: None,
            });
            stores.push(Stmt::If {
                cond: within,
                then_body: vec![store],
                else_body: None,
            });
        }
        values.extend(stores);
        Ok(values)
    }
}

/// A kernel's parameters and what each binds.
pub(super) struct Resources {
    pub(super) params: Vec<KernelParam>,
    pub(super) parcels: Vec<(ParcelId, bool)>,
    pub(super) scalars: Vec<ScalarParam>,
}

/// `reductions` grouped by extent and accumulator count, in first-appearance order.
fn siblings(reductions: Vec<&Term>) -> Vec<Vec<&Term>> {
    let shape = |t: &Term| match t {
        Term::Reduce { extent, order, .. } => (*extent, *order),
        _ => unreachable!("only reductions are grouped"),
    };
    let mut groups: Vec<Vec<&Term>> = Vec::new();
    for r in reductions {
        match groups.iter_mut().find(|g| shape(g[0]) == shape(r)) {
            Some(group) => group.push(r),
            None => groups.push(vec![r]),
        }
    }
    groups
}

pub(super) fn coord_name(c: IndexVar) -> String {
    format!("c{}", c.0)
}

pub(super) fn parcel_name(parcel: ParcelId) -> String {
    format!("p{}", parcel.0)
}

fn scalar_name(scalar: ScalarParam) -> String {
    format!("s{}", scalar.0)
}

/// A reduction folded into a sibling loop: its operator, index and term.
type Folded<'t> = (ReduceOp, IndexVar, &'t Term);

pub(super) struct Emit<'a> {
    region: &'a Region,
    /// The local each bound index variable lives in.
    names: HashMap<IndexVar, String>,
    ranges: HashMap<Sym, (i64, i64)>,
    fresh: u32,
    /// Terms computed where every later use can see them, by their term.
    pub(super) cache: Vec<(Term, String)>,
    /// Loads made at the depth of `.0`, by their expression: the step of a sibling
    /// loop reads each element its reductions share once.
    loads: Option<(u32, Vec<(Expr, String)>)>,
    /// Nesting inside selects and reductions; only depth 0 is cached.
    pub(super) depth: u32,
    /// The output being lowered, for errors.
    output: String,
}

impl Emit<'_> {
    pub(super) fn fresh(&mut self, prefix: &str) -> String {
        self.fresh += 1;
        format!("{prefix}{}", self.fresh)
    }

    pub(super) fn bind(&mut self, v: IndexVar, name: String, extent: u32) -> (Option<String>, Option<(i64, i64)>) {
        (
            self.names.insert(v, name),
            self.ranges.insert(Sym::Index(v), (0, i64::from(extent) - 1)),
        )
    }

    pub(super) fn unbind(&mut self, v: IndexVar, old: (Option<String>, Option<(i64, i64)>)) {
        match old.0 {
            Some(name) => self.names.insert(v, name),
            None => self.names.remove(&v),
        };
        match old.1 {
            Some(range) => self.ranges.insert(Sym::Index(v), range),
            None => self.ranges.remove(&Sym::Index(v)),
        };
    }

    fn cached(&self, term: &Term) -> Option<String> {
        self.cache
            .iter()
            .find(|(t, _)| t.alpha_eq(term))
            .map(|(_, n)| n.clone())
    }

    /// `a` as an `int` expression, checked to stay in the 32-bit range.
    fn affine(&self, a: &Affine) -> Result<Expr, LowerError> {
        let range = |s: Sym| match s {
            Sym::Param(p) => {
                let r = self.region.param_range(p);
                Some((r.start, r.end - 1))
            }
            Sym::Index(_) => self.ranges.get(&s).copied(),
        };
        let (lo, hi) = a.bounds(&range).ok_or(LowerError::IndexRange)?;
        if lo < i64::from(i32::MIN) || hi > i64::from(i32::MAX) {
            return Err(LowerError::IndexRange);
        }
        let mut expr: Option<Expr> = None;
        for &(sym, coefficient) in a.terms() {
            let v = match sym {
                Sym::Index(v) => var(self.names.get(&v).ok_or(LowerError::IndexRange)?),
                Sym::Param(p) => var(&format!("q{}", p.0)),
            };
            let term = if coefficient == 1 {
                v
            } else {
                bin(BinOp::Mul, int(coefficient)?, v)
            };
            expr = Some(match expr {
                None => term,
                Some(e) => bin(BinOp::Add, e, term),
            });
        }
        Ok(match (expr, a.constant_term()) {
            (None, c) => int(c)?,
            (Some(e), 0) => e,
            (Some(e), c) => bin(BinOp::Add, e, int(c)?),
        })
    }

    pub(super) fn term(&mut self, t: &Term, out: &mut Vec<Stmt>) -> Result<Expr, LowerError> {
        Ok(match t {
            Term::Lit(v) => float(*v),
            Term::Scalar(s) => var(&scalar_name(*s)),
            Term::IndexValue(a) => cast(self.affine(a)?, "float"),
            Term::Read { value, index } => {
                let storage = self
                    .region
                    .value(*value)
                    .and_then(|v| v.storage.as_ref())
                    .expect("after substitution every read is of an input");
                let load = Expr::Index {
                    base: Box::new(var(&parcel_name(storage.parcel))),
                    index: Box::new(cast(self.affine(&storage.element(index))?, "uint")),
                };
                let depth = self.depth;
                let Some((_, loads)) = self.loads.as_mut().filter(|(d, _)| *d == depth) else {
                    return Ok(load);
                };
                if let Some((_, name)) = loads.iter().find(|(l, _)| *l == load) {
                    return Ok(var(name));
                }
                let name = self.fresh("x");
                out.push(let_float(&name, load.clone()));
                self.loads.as_mut().expect("checked above").1.push((load, name.clone()));
                var(&name)
            }
            Term::Unary { op, arg } => {
                let x = self.term(arg, out)?;
                match op {
                    UnaryOp::Neg => Expr::Unary {
                        op: IrUnaryOp::Neg,
                        expr: Box::new(x),
                    },
                    UnaryOp::Recip => bin(BinOp::Div, float(1.0), x),
                    UnaryOp::Abs => call(BuiltinFn::Abs, vec![x]),
                    UnaryOp::Exp => call(BuiltinFn::Exp, vec![x]),
                    UnaryOp::Log => call(BuiltinFn::Log, vec![x]),
                    UnaryOp::Sqrt => call(BuiltinFn::Sqrt, vec![x]),
                    UnaryOp::Sin => call(BuiltinFn::Sin, vec![x]),
                    UnaryOp::Cos => call(BuiltinFn::Cos, vec![x]),
                }
            }
            Term::Binary { op, lhs, rhs } => {
                let a = self.term(lhs, out)?;
                let b = self.term(rhs, out)?;
                match op {
                    BinaryOp::Add => bin(BinOp::Add, a, b),
                    BinaryOp::Sub => bin(BinOp::Sub, a, b),
                    BinaryOp::Mul => bin(BinOp::Mul, a, b),
                    BinaryOp::Div => bin(BinOp::Div, a, b),
                    BinaryOp::Min => call(BuiltinFn::Min, vec![a, b]),
                    BinaryOp::Max => call(BuiltinFn::Max, vec![a, b]),
                }
            }
            Term::Select {
                lhs,
                cmp,
                rhs,
                then,
                otherwise,
            } => {
                let name = self.fresh("v");
                out.push(Stmt::Let {
                    name: name.clone(),
                    mutable: true,
                    ty: Some("float".into()),
                    init: float(0.0),
                });
                let op = match cmp {
                    CmpOp::Lt => BinOp::Lt,
                    CmpOp::Le => BinOp::Le,
                    CmpOp::Eq => BinOp::Eq,
                    CmpOp::Ne => BinOp::Ne,
                };
                let cond = bin(op, self.affine(lhs)?, self.affine(rhs)?);
                self.depth += 1;
                let mut branches = Vec::new();
                for branch in [then, otherwise] {
                    let mut stmts = Vec::new();
                    let x = self.term(branch, &mut stmts)?;
                    stmts.push(assign(&name, x));
                    branches.push(stmts);
                }
                self.depth -= 1;
                let otherwise = branches.pop();
                out.push(Stmt::If {
                    cond,
                    then_body: branches.pop().expect("two branches"),
                    else_body: otherwise,
                });
                var(&name)
            }
            Term::Reduce {
                op,
                index,
                extent,
                order,
                body,
            } => {
                if let Some(name) = self.cached(t).filter(|_| self.depth == 0) {
                    return Ok(var(&name));
                }
                if *order != ReduceOrder::Sequential {
                    return Err(LowerError::Order {
                        output: self.output.clone(),
                    });
                }
                let acc = self.fresh("a");
                out.push(let_float(&acc, float(op.identity())));
                let k = self.fresh("k");
                out.push(let_int(&k, int(0)?));
                let old = self.bind(*index, k.clone(), *extent);
                self.depth += 1;
                let mut stmts = Vec::new();
                let x = self.term(body, &mut stmts);
                self.depth -= 1;
                self.unbind(*index, old);
                stmts.push(assign(&acc, combine(*op, var(&acc), x?)));
                stmts.push(assign(&k, bin(BinOp::Add, var(&k), int(1)?)));
                out.push(Stmt::While {
                    cond: bin(BinOp::Lt, var(&k), int(i64::from(*extent))?),
                    body: stmts,
                });
                if self.depth == 0 {
                    self.cache.push((t.clone(), acc.clone()));
                }
                var(&acc)
            }
        })
    }

    /// Emits sibling lane-ordered reductions `group`, which share an extent and an
    /// accumulator count, as one strided loop over each element's lane group. Returns
    /// each one's partial for [`Self::exchange`].
    fn lanes(&mut self, group: &[&Term], group_lanes: u32, out: &mut Vec<Stmt>) -> Result<Vec<Expr>, LowerError> {
        let Term::Reduce {
            extent,
            order: ReduceOrder::Lanes { lanes, accumulators },
            ..
        } = group[0]
        else {
            unreachable!("only lane-ordered reductions are spread over lanes");
        };
        if *lanes != group_lanes {
            return Err(LowerError::Order {
                output: self.output.clone(),
            });
        }
        let folded = folded(group);
        let (l, count) = (i64::from(*lanes), *accumulators as usize);
        let accs: Vec<Vec<String>> = folded
            .iter()
            .map(|&(op, ..)| {
                (0..count)
                    .map(|_| {
                        let acc = self.fresh("a");
                        out.push(let_float(&acc, float(op.identity())));
                        acc
                    })
                    .collect()
            })
            .collect();
        let nth = |a: usize| accs.iter().map(|accs| accs[a].as_str()).collect::<Vec<_>>();
        let j = self.fresh("j");
        let n = int(i64::from(*extent))?;
        let offset = |a: i64| -> Result<Expr, LowerError> {
            Ok(match a * l {
                0 => var(&j),
                d => bin(BinOp::Add, var(&j), int(d)?),
            })
        };
        let mut strided = vec![let_int(&j, var("lane"))];
        let mut step = Vec::new();
        for a in 0..count {
            self.step(&folded, &nth(a), *extent, offset(a as i64)?, &mut step)?;
        }
        step.push(assign(&j, bin(BinOp::Add, var(&j), int(l * count as i64)?)));
        strided.push(Stmt::While {
            cond: bin(BinOp::Lt, offset(count as i64 - 1)?, n.clone()),
            body: step,
        });
        for a in 0..count - 1 {
            let mut tail = Vec::new();
            self.step(&folded, &nth(a), *extent, offset(a as i64)?, &mut tail)?;
            strided.push(Stmt::If {
                cond: bin(BinOp::Lt, offset(a as i64)?, n.clone()),
                then_body: tail,
                else_body: None,
            });
        }
        out.push(Stmt::If {
            cond: var("valid"),
            then_body: strided,
            else_body: None,
        });
        Ok(folded
            .iter()
            .zip(&accs)
            .map(|(&(op, ..), accs)| accs[1..].iter().fold(var(&accs[0]), |p, acc| combine(op, p, var(acc))))
            .collect())
    }

    /// Combines each reduction's lane partials over its group by the tree of
    /// [`ReduceOrder::Lanes`], all of them level by level, and caches each result on
    /// the group's first lane. Every thread must reach it. Returns the workgroup memory
    /// it uses.
    fn exchange(
        &mut self,
        reduced: &[(&Term, Expr)],
        lanes: u32,
        elements: u32,
        exchange: Exchange,
        decls: &mut Vec<Stmt>,
        out: &mut Vec<Stmt>,
    ) -> Result<u32, LowerError> {
        if reduced.is_empty() {
            return Ok(0);
        }
        let op = |t: &Term| match t {
            Term::Reduce { op, .. } => *op,
            _ => unreachable!("only reductions are exchanged"),
        };
        let s = self.fresh("s");
        let halve = assign(&s, bin(BinOp::Div, var(&s), int(2)?));
        let more = bin(BinOp::Gt, var(&s), int(0)?);
        let levels = |body: Vec<Stmt>| Stmt::While { cond: more, body };
        match exchange {
            Exchange::Workgroup => {
                let slots: Vec<String> = reduced.iter().map(|_| self.fresh("r")).collect();
                let slot = |scratch: &str, at: Expr| Expr::Index {
                    base: Box::new(var(scratch)),
                    index: Box::new(at),
                };
                for (scratch, (_, partial)) in slots.iter().zip(reduced) {
                    decls.push(Stmt::WorkgroupArray {
                        name: scratch.clone(),
                        elem: "float".into(),
                        len: lanes * elements,
                    });
                    out.push(Stmt::Assign {
                        target: slot(scratch, var("local")),
                        value: partial.clone(),
                    });
                }
                out.push(barrier());
                out.push(let_int(&s, int(i64::from(lanes / 2))?));
                let combined = slots
                    .iter()
                    .zip(reduced)
                    .map(|(scratch, (t, _))| Stmt::Assign {
                        target: slot(scratch, var("local")),
                        value: combine(
                            op(t),
                            slot(scratch, var("local")),
                            slot(scratch, bin(BinOp::Add, var("local"), var(&s))),
                        ),
                    })
                    .collect();
                out.push(levels(vec![
                    Stmt::If {
                        cond: bin(BinOp::Lt, var("lane"), var(&s)),
                        then_body: combined,
                        else_body: None,
                    },
                    barrier(),
                    halve,
                ]));
                for (scratch, (t, _)) in slots.iter().zip(reduced) {
                    let result = self.fresh("v");
                    out.push(let_float(
                        &result,
                        slot(scratch, bin(BinOp::Sub, var("local"), var("lane"))),
                    ));
                    self.cache.push(((*t).clone(), result));
                }
                Ok(reduced.len() as u32 * lanes * elements * 4)
            }
            Exchange::Subgroup => {
                // Lanes can arrive from divergent loops, and a subgroup read needs all
                // of them.
                out.push(barrier());
                let values: Vec<String> = reduced
                    .iter()
                    .map(|(_, partial)| {
                        let v = self.fresh("v");
                        out.push(let_float(&v, partial.clone()));
                        v
                    })
                    .collect();
                out.push(let_int(&s, int(i64::from(lanes / 2))?));
                let partner = bin(
                    BinOp::BitXor,
                    call(BuiltinFn::SubgroupLane, Vec::new()),
                    cast(var(&s), "uint"),
                );
                // Lanes at or past `s` combine too; only lanes below it are read later.
                let mut body: Vec<Stmt> = values
                    .iter()
                    .zip(reduced)
                    .map(|(v, (t, _))| {
                        let other = call(BuiltinFn::SubgroupRead, vec![var(v), partner.clone()]);
                        assign(v, combine(op(t), var(v), other))
                    })
                    .collect();
                body.push(halve);
                out.push(levels(body));
                for (v, (t, _)) in values.into_iter().zip(reduced) {
                    self.cache.push(((*t).clone(), v));
                }
                Ok(0)
            }
        }
    }

    /// Emits sibling sequential reductions `group`, which share an extent, as one loop
    /// and caches each result.
    fn sequential(&mut self, group: &[&Term], out: &mut Vec<Stmt>) -> Result<(), LowerError> {
        let Term::Reduce { extent, .. } = group[0] else {
            unreachable!("only reductions are merged");
        };
        let folded = folded(group);
        let accs: Vec<String> = folded
            .iter()
            .map(|&(op, ..)| {
                let acc = self.fresh("a");
                out.push(let_float(&acc, float(op.identity())));
                acc
            })
            .collect();
        let j = self.fresh("j");
        out.push(let_int(&j, int(0)?));
        let mut step = Vec::new();
        let names: Vec<&str> = accs.iter().map(String::as_str).collect();
        self.step(&folded, &names, *extent, var(&j), &mut step)?;
        step.push(assign(&j, bin(BinOp::Add, var(&j), int(1)?)));
        out.push(Stmt::While {
            cond: bin(BinOp::Lt, var(&j), int(i64::from(*extent))?),
            body: step,
        });
        for (t, acc) in group.iter().zip(accs) {
            self.cache.push(((*t).clone(), acc));
        }
        Ok(())
    }

    /// `accs[r] = accs[r] ⊕ term` for each folded reduction `r`, with its index at
    /// `at`. A load the terms share is made once.
    fn step(
        &mut self,
        folded: &[Folded],
        accs: &[&str],
        extent: u32,
        at: Expr,
        out: &mut Vec<Stmt>,
    ) -> Result<(), LowerError> {
        let k = self.fresh("k");
        out.push(let_int(&k, at));
        self.depth += 1;
        let outer = self.loads.replace((self.depth, Vec::new()));
        let mut result = Ok(());
        for (&(op, index, body), acc) in folded.iter().zip(accs) {
            let old = self.bind(index, k.clone(), extent);
            let x = self.term(body, out);
            self.unbind(index, old);
            match x {
                Ok(x) => out.push(assign(acc, combine(op, var(acc), x))),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        self.loads = outer;
        self.depth -= 1;
        result
    }
}

fn folded<'t>(group: &[&'t Term]) -> Vec<Folded<'t>> {
    group
        .iter()
        .map(|t| match t {
            Term::Reduce { op, index, body, .. } => (*op, *index, &**body),
            _ => unreachable!("only reductions are folded"),
        })
        .collect()
}

/// Inclusive ranges of the index variables in scope.
type Ranges = HashMap<Sym, (i64, i64)>;

impl Term {
    /// Calls `f` for every read with the ranges of the indices bound around it.
    fn walk_reads(&self, ranges: &mut Ranges, f: &mut dyn FnMut(ValueId, &[Affine], &Ranges)) {
        match self {
            Term::Read { value, index } => f(*value, index, ranges),
            Term::Reduce {
                index, extent, body, ..
            } => {
                let old = ranges.insert(Sym::Index(*index), (0, i64::from(*extent) - 1));
                body.walk_reads(ranges, f);
                match old {
                    Some(r) => ranges.insert(Sym::Index(*index), r),
                    None => ranges.remove(&Sym::Index(*index)),
                };
            }
            _ => self.for_each_child(&mut |c| c.walk_reads(ranges, f)),
        }
    }
}

pub(super) fn var(name: &str) -> Expr {
    Expr::Var(name.to_string())
}

pub(super) fn bin(op: BinOp, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

pub(super) fn call(func: BuiltinFn, args: Vec<Expr>) -> Expr {
    Expr::Call { func, args }
}

pub(super) fn cast(expr: Expr, ty: &str) -> Expr {
    Expr::Cast {
        expr: Box::new(expr),
        ty: ty.to_string(),
    }
}

pub(super) fn field(builtin: BuiltinFn) -> Expr {
    Expr::Field {
        base: Box::new(call(builtin, Vec::new())),
        field: "x".into(),
    }
}

pub(super) fn int(value: i64) -> Result<Expr, LowerError> {
    i32::try_from(value)
        .map(Expr::LitI32)
        .map_err(|_| LowerError::IndexRange)
}

/// An f32 literal; non-finite values, which have no literal, as quotients.
pub(super) fn float(value: f32) -> Expr {
    if value.is_finite() {
        return Expr::LitF32(value);
    }
    let numerator = if value.is_nan() {
        0.0
    } else if value > 0.0 {
        1.0
    } else {
        -1.0
    };
    bin(BinOp::Div, Expr::LitF32(numerator), Expr::LitF32(0.0))
}

fn combine(op: ReduceOp, acc: Expr, x: Expr) -> Expr {
    match op {
        ReduceOp::Sum => bin(BinOp::Add, acc, x),
        ReduceOp::Max => call(BuiltinFn::Max, vec![acc, x]),
        ReduceOp::Min => call(BuiltinFn::Min, vec![acc, x]),
    }
}

pub(super) fn barrier() -> Stmt {
    Stmt::Expr(call(BuiltinFn::WorkgroupBarrier, Vec::new()))
}

pub(super) fn assign(name: &str, value: Expr) -> Stmt {
    Stmt::Assign {
        target: var(name),
        value,
    }
}

pub(super) fn let_int(name: &str, init: Expr) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        mutable: true,
        ty: Some("int".into()),
        init,
    }
}

pub(super) fn let_float(name: &str, init: Expr) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        mutable: true,
        ty: Some("float".into()),
        init,
    }
}
