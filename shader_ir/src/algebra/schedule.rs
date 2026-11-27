//! Schedules: how a region's outputs map onto a compute grid, and lowering to a kernel.
//!
//! Lowering first substitutes every temporary and every output into its readers, so
//! each output is one term over inputs. Outputs are then grouped into parts. The
//! outputs of one part have one rank once unit axes are dropped, and the part's domain
//! is their largest extent on each axis; element `t` of it is one thread, or one lane
//! group, which computes every output of the part that spans `t` and then stores them
//! all. Parts are sibling domains: each takes its own range of workgroups in the one
//! dispatch, so outputs of different ranks, or contractions over different extents,
//! run side by side. Inputs are read before any store, which keeps the region's entry
//! semantics for storage that an output overwrites in place.
//!
//! A subterm of a reduction's body that depends on no index but its element's, and
//! that itself reduces, is invariant in the loop around it. Lowering hoists it: each
//! thread computes it once, before any loop, and every use reads the register. A
//! normalization inside a matrix product's factor then costs one pass over its input
//! per thread rather than one per term of the product.

use super::affine::{Affine, IndexParam, IndexVar, Sym};
use super::cost::{self, Estimate};
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
/// Longest vector a lane-ordered product may materialize for the product that
/// consumes it. Longer chains stay separate dispatches.
/// Scratch bound for a chained vector. Profitability ends well below this; the
/// cost model is what declines a chain.
const CHAIN_VECTOR_MAX: u32 = 256;

/// Where the runtime value of an index parameter comes from: one integer element of
/// a parcel, read once per thread.
///
/// `element` is an affine of the composed region's index variables. Lowering rewrites
/// each variable to the coordinate of the output axis that binds it. A partial
/// workgroup clamps those coordinates into the part's extents, so the read stays
/// inside the vector the caller sized. A variable no live output binds declines the
/// kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSource {
    pub parcel: ParcelId,
    pub element: Affine,
    /// [`ElementType::I32`] or [`ElementType::U32`].
    pub element_type: ElementType,
    /// When the loaded value is outside the parameter's declared range, stores that
    /// depend on it write zero instead of the value computed from the clamped index.
    pub zero_if_outside: bool,
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
    /// Domains the kernel dispatches side by side; see [`lower_on`].
    pub parts: usize,
    pub estimate: Estimate,
}

/// Why a region cannot run as one kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    Invalid(RegionError),
    NoOutputs,
    /// The domains do not fit one grid.
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
    /// A read of a gathered operand addresses another element of an index parameter's
    /// source than the one its thread loads.
    GatheredRead {
        param: String,
    },
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::Invalid(e) => e.fmt(f),
            LowerError::NoOutputs => f.write_str("the region has no outputs"),
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
            LowerError::GatheredRead { param } => {
                write!(
                    f,
                    "a read needs another element of index parameter `{param}` than its thread loads"
                )
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
///
/// An output joins the first part of its rank on which every output spanning only a
/// prefix of the part's domain reuses the lane reductions of the outputs spanning all
/// of it, since lane groups reduce over the whole domain; otherwise it starts a part.
/// Under [`Schedule::Lanes`], a part whose outputs have no lane-ordered reduction runs
/// one element per thread.
pub fn lower_on(region: &Region, sources: &[IndexSource], target: &Target) -> Result<Lowered, LowerError> {
    let prepared = Prepared::rounded(region)?;
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
    let prepared = Prepared::rounded(region)?;
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

fn is_lane_ordered(r: &&Term) -> bool {
    matches!(
        r,
        Term::Reduce {
            order: ReduceOrder::Lanes { .. },
            ..
        }
    )
}

/// Outputs of squeezed `shapes` grouped into parts, in order; see [`lower_on`].
fn partition(shapes: &[Vec<u32>], bodies: &[(ValueId, Term)], coords: &[IndexVar]) -> Vec<Part> {
    let admits = |members: &[usize], extents: &[u32]| {
        let full: Vec<&Term> = members
            .iter()
            .filter(|&&m| shapes[m] == extents)
            .flat_map(|&m| top_reductions(&bodies[m].1))
            .collect();
        members.iter().all(|&m| {
            shapes[m] == extents
                || top_reductions(&bodies[m].1)
                    .into_iter()
                    .filter(is_lane_ordered)
                    .all(|r| full.iter().any(|f| f.alpha_eq(r)))
        })
    };
    let mut parts: Vec<Part> = Vec::new();
    for (n, shape) in shapes.iter().enumerate() {
        let widened = |p: &Part| -> Vec<u32> { p.extents.iter().zip(shape).map(|(&a, &b)| a.max(b)).collect() };
        let joined = parts.iter().position(|p| {
            p.extents.len() == shape.len() && admits(&[p.outputs.as_slice(), &[n]].concat(), &widened(p))
        });
        match joined {
            Some(k) => {
                parts[k].extents = widened(&parts[k]);
                parts[k].outputs.push(n);
            }
            None => parts.push(Part {
                extents: shape.clone(),
                coords: coords[..shape.len()].to_vec(),
                outputs: vec![n],
            }),
        }
    }
    parts
}

/// Subterms of `term` inside a reduction's body that depend on no index but `coords`,
/// reduce, and reduce only sequentially: the loops around them do not change them.
/// Inner ones come first, so each is computed after those it reads.
fn hoistable(term: &Term, coords: &[IndexVar], inside: bool, found: &mut Vec<Term>) {
    let mut sequential = true;
    let mut reduces = false;
    term.for_each_reduction(&mut |order| {
        reduces = true;
        sequential &= order == ReduceOrder::Sequential;
    });
    let candidate = inside && reduces && sequential && term.free_indices().iter().all(|v| coords.contains(v));
    match term {
        Term::Reduce { body, .. } => hoistable(body, coords, true, found),
        _ => term.for_each_child(&mut |c| hoistable(c, coords, inside && !candidate, found)),
    }
    if candidate && !found.iter().any(|f| f.alpha_eq(term)) {
        found.push(term.clone());
    }
}

/// Outputs lowered over one domain, in one range of workgroups.
pub(super) struct Part {
    /// Extents of the domain.
    pub(super) extents: Vec<u32>,
    pub(super) coords: Vec<IndexVar>,
    /// Positions in [`Prepared::bodies`] of the outputs it computes.
    outputs: Vec<usize>,
}

impl Part {
    fn elements(&self) -> u64 {
        self.extents.iter().map(|&e| u64::from(e)).product()
    }
}

/// A region whose outputs are single terms over inputs, grouped into [`Part`]s.
pub(super) struct Prepared {
    region: Region,
    pub(super) parts: Vec<Part>,
    /// Each output and its body on its part's coordinates.
    bodies: Vec<(ValueId, Term)>,
    /// Each output's index on its part's coordinates.
    at: Vec<Vec<Affine>>,
    /// Per output, the axes of its part it spans only a prefix of, and that prefix.
    limits: Vec<Vec<(usize, u32)>>,
    /// The part of each output.
    part_of: Vec<usize>,
}

impl Prepared {
    /// [`Self::new`] after [`Region::round_results`], so the kernel rounds every
    /// operation's result as the separate dispatches would.
    fn rounded(region: &Region) -> Result<Self, LowerError> {
        let mut region = region.clone();
        region.round_results();
        Self::new(&region)
    }

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
        if outputs.is_empty() {
            return Err(LowerError::NoOutputs);
        }
        let squeeze = |shape: &[u32]| shape.iter().copied().filter(|&e| e != 1).collect::<Vec<_>>();
        let shapes: Vec<Vec<u32>> = outputs
            .iter()
            .map(|&o| squeeze(&region.value(o).expect("live").shape))
            .collect();
        let rank = shapes.iter().map(Vec::len).max().unwrap_or(0);
        // Parts of one rank share coordinates; each binds them in its own scope.
        let coords: Vec<IndexVar> = (0..rank).map(|a| region.index(&format!("c{a}"))).collect();
        let mut bodies = Vec::new();
        let mut at = Vec::new();
        for &o in &outputs {
            let value = region.value(o).expect("live");
            let def = value.definition.as_ref().expect("an output has a definition");
            let mut next = coords.iter();
            let index: Vec<Affine> = value
                .shape
                .iter()
                .map(|&e| match e {
                    1 => Affine::constant(0),
                    _ => Affine::from(*next.next().expect("no wider than the widest output")),
                })
                .collect();
            let map: HashMap<IndexVar, Affine> = def.domain.iter().copied().zip(index.iter().cloned()).collect();
            bodies.push((o, def.body.instantiate(&|v| map.get(&v).cloned(), &HashMap::new())));
            at.push(index);
        }
        let parts = partition(&shapes, &bodies, &coords);
        let mut part_of = vec![0; outputs.len()];
        for (k, part) in parts.iter().enumerate() {
            for &n in &part.outputs {
                part_of[n] = k;
            }
        }
        let limits = shapes
            .iter()
            .zip(&part_of)
            .map(|(shape, &k)| {
                shape
                    .iter()
                    .zip(&parts[k].extents)
                    .enumerate()
                    .filter(|(_, (s, e))| s < e)
                    .map(|(a, (&s, _))| (a, s))
                    .collect()
            })
            .collect();
        let prepared = Self {
            region,
            parts,
            bodies,
            at,
            limits,
            part_of,
        };
        prepared.check()?;
        Ok(prepared)
    }

    /// The only part, when the region has one domain.
    pub(super) fn single(&self) -> Option<&Part> {
        match self.parts.as_slice() {
            [part] => Some(part),
            _ => None,
        }
    }

    /// Whether an output of part `k` has a lane-ordered reduction outside a select.
    fn has_lanes(&self, k: usize) -> bool {
        self.parts[k]
            .outputs
            .iter()
            .any(|&n| top_reductions(&self.bodies[n].1).iter().any(is_lane_ordered))
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
        for (k, part) in self.parts.iter().enumerate() {
            let mut ranges: HashMap<Sym, (i64, i64)> = part
                .coords
                .iter()
                .zip(&part.extents)
                .map(|(&c, &e)| (Sym::Index(c), (0, i64::from(e) - 1)))
                .collect();
            for &n in &part.outputs {
                let mut failure = None;
                self.bodies[n].1.walk_reads(&mut ranges, &mut |value, index, ranges| {
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
                    for (m, ((o, _), at)) in self.bodies.iter().zip(&self.at).enumerate() {
                        let written = self.storage(*o);
                        // Only a thread of the same part writes the element it reads.
                        let own = self.part_of[m] == k && written.element(at) == element;
                        if written.parcel != storage.parcel || own {
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
        }
        Ok(())
    }

    /// Distinct bytes the kernel moves: every input something reads and every output,
    /// each at most once per element its storage spans.
    pub(super) fn footprint(&self) -> u64 {
        let mut bytes = 0;
        for (id, value) in self.region.values() {
            if value.role == Role::Temporary || (value.role == Role::Input && self.region.readers(id) == 0) {
                continue;
            }
            let storage = value.storage.as_ref().expect("validated");
            let elements: u64 = value.shape.iter().map(|&e| u64::from(e)).product();
            let spanned = storage
                .interval(&value.shape, &|p| self.param_range(p))
                .map_or(elements, |(lo, hi)| (hi - lo + 1).max(0) as u64);
            bytes += 4 * elements.min(spanned);
        }
        bytes
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

    /// An emitter in the first part; see [`Self::enter`].
    pub(super) fn emitter(&self) -> Emit<'_> {
        let mut emit = Emit {
            region: &self.region,
            names: HashMap::new(),
            ranges: HashMap::new(),
            fresh: 0,
            cache: Vec::new(),
            hoisted: Vec::new(),
            loads: None,
            depth: 0,
            output: String::new(),
            lane_guard: true,
            chains: Vec::new(),
        };
        self.enter(&mut emit, 0);
        emit
    }

    /// Binds each coordinate of part `k` to its local `c{n}`, and forgets what `emit`
    /// computed in the part before, whose locals are out of scope.
    fn enter(&self, emit: &mut Emit, k: usize) {
        let part = &self.parts[k];
        for (&c, &e) in part.coords.iter().zip(&part.extents) {
            emit.bind(c, coord_name(c), e);
        }
        emit.cache.clear();
        emit.hoisted.clear();
        emit.chains.clear();
        emit.output = self.name(self.bodies[part.outputs[0]].0).to_string();
    }

    /// Computes each term [`hoistable`] finds in part `k` into a local, once per
    /// thread, where every later use reads it. Returns the terms.
    fn hoist(&self, k: usize, emit: &mut Emit, out: &mut Vec<Stmt>) -> Result<Vec<Term>, LowerError> {
        let part = &self.parts[k];
        let mut found = Vec::new();
        for &n in &part.outputs {
            hoistable(&self.bodies[n].1, &part.coords, false, &mut found);
        }
        for term in &found {
            let name = emit.fresh("h");
            if term.free_indices().is_empty() {
                let x = emit.term(term, out)?;
                out.push(let_float(&name, x));
            } else {
                // It reads at the element's coordinates, which only valid threads have.
                out.push(let_float(&name, float(0.0)));
                let mut guarded = Vec::new();
                emit.depth += 1;
                let x = emit.term(term, &mut guarded);
                emit.depth -= 1;
                guarded.push(assign(&name, x?));
                out.push(Stmt::If {
                    cond: var("valid"),
                    then_body: guarded,
                    else_body: None,
                });
            }
            emit.hoisted.push((term.clone(), name));
        }
        Ok(found)
    }

    /// Dependent steps and loads of one thread, and operations of all of them, in part
    /// `k`: its hoisted terms, the lane reductions `lanes` spreads over each lane group
    /// of an element, and its outputs. A value computed once is counted once, and
    /// `computed` values, which another stage of the kernel provides, not at all.
    pub(super) fn estimate(&self, k: usize, lanes: Option<u32>, hoisted: &[Term], computed: &[&Term]) -> Estimate {
        let part = &self.parts[k];
        let mut known: Vec<&Term> = computed.to_vec();
        let mut chained = Vec::new();
        for &n in &part.outputs {
            collect_chains(&self.bodies[n].1, None, &mut chained);
        }
        // The chained product is computed once per workgroup, not inside every row.
        known.extend(chained.iter().map(|slot| &slot.term));
        let (mut serial, mut loads, mut per_thread) = (0, 0, 0);
        for term in hoisted {
            serial += cost::serial(term, &known);
            loads += cost::loads(term, &known);
            per_thread += cost::ops(term, &known);
            known.push(term);
        }
        if let Some(lanes) = lanes {
            let pending = self.top_reductions(k, true, is_lane_ordered);
            let groups = siblings(pending.clone());
            for group in &groups {
                let Term::Reduce { extent, .. } = group[0] else {
                    unreachable!("only reductions are grouped");
                };
                let steps = u64::from(extent.div_ceil(lanes));
                let bodies = group.iter().map(|r| match r {
                    Term::Reduce { body, .. } => &**body,
                    _ => unreachable!("only reductions are grouped"),
                });
                serial += steps * (1 + bodies.clone().map(|b| cost::serial(b, &known)).max().unwrap_or(0));
                loads += steps * bodies.clone().map(|b| cost::loads(b, &known)).max().unwrap_or(0);
                per_thread += steps * bodies.map(|b| 1 + cost::ops(b, &known)).sum::<u64>();
            }
            if !groups.is_empty() {
                serial += u64::from(lanes.max(1).ilog2());
            }
            known.extend(pending);
        }
        let per_group = match lanes {
            Some(lanes) => (LANE_WORKGROUP_THREADS / lanes).max(1),
            None => WORKGROUP_THREADS,
        };
        let mut barriers = 0u64;
        if let Some(lanes) = lanes {
            // `per_group` row-groups compute that many chain elements at once.
            let wave = u64::from(per_group.max(1));
            let tree = u64::from(lanes.max(2).ilog2());
            for slot in &chained {
                let chunks = u64::from(slot.extent).div_ceil(wave).max(1);
                let per_thread = cost::serial(&slot.term, &[]).div_ceil(u64::from(lanes));
                serial += chunks * per_thread;
                loads += chunks * cost::loads(&slot.term, &[]).div_ceil(u64::from(lanes));
                // Scratch write, the lane tree, and the store, once per chunk.
                barriers += chunks * (2 + tree);
            }
        }
        let threads = part.elements() * u64::from(lanes.unwrap_or(1));
        let mut work = threads * per_thread;
        let groups = part.elements().div_ceil(u64::from(per_group.max(1))).max(1);
        for slot in &chained {
            // Every workgroup recomputes the vector.
            work += groups * u64::from(slot.extent) * cost::ops(&slot.term, &[]);
        }
        for &n in &part.outputs {
            let body = &self.bodies[n].1;
            serial += cost::serial(body, &known) + 1;
            loads += cost::loads(body, &known);
            work += part.elements() * (cost::ops(body, &known) + 1);
            known.extend(top_reductions(body));
        }
        Estimate {
            serial,
            loads,
            work,
            barriers,
            ..Estimate::default()
        }
    }

    /// `q{p}`: the value of each index parameter, read from its source and clamped
    /// into the range the region declared. `in{p}` is whether the raw value was
    /// already inside that range. Every source's element must be a constant; a
    /// coordinate-dependent element uses [`Self::load_index_params`].
    pub(super) fn index_params(&self, sources: &[IndexSource], body: &mut Vec<Stmt>) -> Result<(), LowerError> {
        for (p, source) in sources.iter().enumerate() {
            self.load_one(p, source, &HashMap::new(), &HashMap::new(), body)?;
        }
        Ok(())
    }

    /// Rewrites each source element from the region's index variables to the
    /// coordinates [`Self::new`] assigned to the output axes that bind them.
    fn resolve_index_sources(&self, sources: &[IndexSource]) -> Result<Vec<IndexSource>, LowerError> {
        let map = self.bound_coordinates()?;
        self.check_gathered_reads(sources, &map)?;
        Ok(sources
            .iter()
            .map(|source| IndexSource {
                element: source.element.substitute(&|v| map.get(&v).cloned()),
                ..source.clone()
            })
            .collect())
    }

    /// Each output's domain variables, as the coordinates [`Self::new`] assigned to them.
    fn bound_coordinates(&self) -> Result<HashMap<IndexVar, Affine>, LowerError> {
        let mut map: HashMap<IndexVar, Affine> = HashMap::new();
        for (n, &(id, _)) in self.bodies.iter().enumerate() {
            let value = self.region.value(id).expect("an output is live");
            let domain = &value.definition.as_ref().expect("an output is defined").domain;
            for (&var, affine) in domain.iter().zip(&self.at[n]) {
                if let Some(prev) = map.get(&var) {
                    if prev != affine {
                        return Err(LowerError::IndexRange);
                    }
                } else {
                    map.insert(var, affine.clone());
                }
            }
        }
        Ok(map)
    }

    /// Declines a read the thread's index parameters do not address.
    ///
    /// A parameter whose source element depends on an index variable gathers along the
    /// one stride-zero axis of each storage that mentions it, and each thread loads it
    /// once, at the coordinate that variable is bound to. Substituting a gathered value
    /// into a reader can leave a read on that axis elsewhere, such as at a constant
    /// slot; that read needs a different element of the source than the thread holds.
    fn check_gathered_reads(&self, sources: &[IndexSource], map: &HashMap<IndexVar, Affine>) -> Result<(), LowerError> {
        for (p, source) in sources.iter().enumerate() {
            let vars: Vec<IndexVar> = source.element.indices().collect();
            if vars.is_empty() {
                continue;
            }
            let param = IndexParam(p as u32);
            let declined = || LowerError::GatheredRead {
                param: self.region.param_name(param).to_string(),
            };
            let bound = match vars.as_slice() {
                [v] => map.get(v).ok_or_else(declined)?,
                _ => return Err(declined()),
            };
            let mut addressed = true;
            for (_, body) in &self.bodies {
                body.for_each_read(&mut |value, index| {
                    let storage = self.storage(value);
                    if !storage.offset.mentions(Sym::Param(param)) {
                        return;
                    }
                    let mut gathered = (0..storage.strides.len()).filter(|&a| storage.strides[a] == 0);
                    addressed &= match (gathered.next(), gathered.next()) {
                        (Some(axis), None) => index.get(axis) == Some(bound),
                        _ => false,
                    };
                });
            }
            if !addressed {
                return Err(declined());
            }
        }
        Ok(())
    }

    /// Whether output `n` addresses parameter `p`: a read whose storage offset
    /// mentions it, or the output's own storage.
    fn output_mentions_param(&self, n: usize, p: IndexParam) -> bool {
        let sym = Sym::Param(p);
        let stored = self
            .region
            .value(self.bodies[n].0)
            .and_then(|v| v.storage.as_ref())
            .is_some_and(|s| s.offset.mentions(sym));
        if stored {
            return true;
        }
        let mut used = false;
        self.bodies[n].1.for_each_read(&mut |value, _| {
            let mentions = self
                .region
                .value(value)
                .and_then(|v| v.storage.as_ref())
                .is_some_and(|s| s.offset.mentions(sym));
            used |= mentions;
        });
        used
    }

    /// Inclusive upper bound of coordinate `c` on output `n`, from that output's own
    /// extent rather than a widened part.
    fn coord_limit(&self, n: usize, c: IndexVar) -> Option<i64> {
        let id = self.bodies[n].0;
        let shape = &self.region.value(id)?.shape;
        self.at[n].iter().zip(shape).find_map(|(affine, &extent)| {
            let one_coord =
                affine.terms().len() == 1 && affine.coefficient(Sym::Index(c)) == 1 && affine.constant_term() == 0;
            one_coord.then_some(i64::from(extent) - 1)
        })
    }

    /// Loads the index parameters part `k` reads, after its coordinates are bound.
    /// `loaded[p]` is whether `q{p}` and `in{p}` now exist.
    ///
    /// A coordinate is clamped to the tightest extent of an output in this part that
    /// reads the parameter. A widened part must not address past the vector that
    /// output's batch sized.
    fn load_index_params(
        &self,
        k: usize,
        sources: &[IndexSource],
        body: &mut Vec<Stmt>,
    ) -> Result<Vec<bool>, LowerError> {
        let part = &self.parts[k];
        let mut names = HashMap::new();
        for &c in &part.coords {
            names.insert(c, coord_name(c));
        }
        let mut loaded = Vec::with_capacity(sources.len());
        for (p, source) in sources.iter().enumerate() {
            let param = IndexParam(p as u32);
            let readers: Vec<usize> = part
                .outputs
                .iter()
                .copied()
                .filter(|&n| self.output_mentions_param(n, param))
                .collect();
            loaded.push(!readers.is_empty());
            if readers.is_empty() {
                continue;
            }
            let mut ranges = HashMap::new();
            for (&c, &e) in part.coords.iter().zip(&part.extents) {
                let mut hi = i64::from(e) - 1;
                for &n in &readers {
                    if let Some(limit) = self.coord_limit(n, c) {
                        hi = hi.min(limit);
                    }
                }
                ranges.insert(c, (0i64, hi));
            }
            self.load_one(p, source, &names, &ranges, body)?;
        }
        Ok(loaded)
    }

    fn load_one(
        &self,
        p: usize,
        source: &IndexSource,
        names: &HashMap<IndexVar, String>,
        ranges: &HashMap<IndexVar, (i64, i64)>,
        body: &mut Vec<Stmt>,
    ) -> Result<(), LowerError> {
        let index = element_address(&source.element, names, ranges, body)?;
        let at = Expr::Index {
            base: Box::new(var(&parcel_name(source.parcel))),
            index: Box::new(index),
        };
        let q = format!("q{p}");
        let inn = format!("in{p}");
        body.push(let_int(&q, cast(at, "int")));
        let range = self.region.param_range(IndexParam(p as u32));
        let lo = int(range.start)?;
        let hi = int(range.end.saturating_sub(1))?;
        body.push(Stmt::Let {
            name: inn.clone(),
            mutable: false,
            ty: Some("bool".into()),
            init: bin(
                BinOp::And,
                bin(BinOp::Ge, var(&q), lo.clone()),
                bin(BinOp::Le, var(&q), hi),
            ),
        });
        body.push(Stmt::If {
            cond: Expr::Unary {
                op: IrUnaryOp::Not,
                expr: Box::new(var(&inn)),
            },
            then_body: vec![assign(&q, lo)],
            else_body: None,
        });
        Ok(())
    }

    fn lower(&self, schedule: Schedule, sources: &[IndexSource]) -> Result<Lowered, LowerError> {
        let Resources {
            params,
            parcels,
            scalars,
        } = self.resources(sources)?;
        let sources = self.resolve_index_sources(sources)?;
        let threads = match schedule {
            Schedule::Threads { workgroup } => workgroup,
            Schedule::Lanes { lanes, elements, .. } => lanes * elements,
            Schedule::Matrix { .. } => unreachable!("matrix schedules lower from a graph"),
        };
        let single = self.parts.len() == 1;
        let mut emit = self.emitter();
        let mut decls = Vec::new();
        let mut body = Vec::new();
        let builtins = match (single, schedule) {
            (true, Schedule::Threads { .. }) => BuiltinMask {
                global_id: true,
                ..BuiltinMask::NONE
            },
            _ => BuiltinMask {
                local_id: true,
                workgroup_id: true,
                ..BuiltinMask::NONE
            },
        };
        if !single {
            body.push(let_int("local", cast(field(BuiltinFn::LocalId), "int")));
            body.push(let_int("group", cast(field(BuiltinFn::WorkgroupId), "int")));
        }
        let mut workgroup_bytes = 0u32;
        let mut groups = 0u64;
        let mut estimate = Estimate {
            bytes: self.footprint(),
            ..Estimate::default()
        };
        for (k, part) in self.parts.iter().enumerate() {
            if k > 0 {
                self.enter(&mut emit, k);
            }
            let lanes = match schedule {
                Schedule::Lanes {
                    lanes,
                    elements,
                    exchange,
                } if single || self.has_lanes(k) => Some((lanes, elements, exchange)),
                _ => None,
            };
            let elements = part.elements();
            let per_group = lanes.map_or(threads, |(_, elements, _)| elements);
            // A chained product is recomputed in every workgroup, so the consumer
            // keeps its normal grid.
            let part_groups = elements.div_ceil(u64::from(per_group.max(1)));
            if elements > i32::MAX as u64 {
                return Err(LowerError::Grid { elements });
            }
            let mut work = Vec::new();
            // Element `t` of the part: its workgroup within the part's range, and its
            // thread or lane group within the workgroup.
            let (group, local) = match single {
                true => (cast(field(BuiltinFn::WorkgroupId), "int"), None),
                false => (bin(BinOp::Sub, var("group"), int(groups as i64)?), Some(var("local"))),
            };
            match lanes {
                None if single => work.push(let_int("t", cast(field(BuiltinFn::GlobalId), "int"))),
                None => work.push(let_int(
                    "t",
                    bin(
                        BinOp::Add,
                        bin(BinOp::Mul, group, int(i64::from(threads))?),
                        var("local"),
                    ),
                )),
                Some((lanes, row_elems, _)) => {
                    if local.is_none() {
                        work.push(let_int("local", cast(field(BuiltinFn::LocalId), "int")));
                    }
                    work.push(let_int("lane", bin(BinOp::Rem, var("local"), int(lanes as i64)?)));
                    work.push(let_int(
                        "t",
                        bin(
                            BinOp::Add,
                            bin(BinOp::Mul, group, int(row_elems as i64)?),
                            bin(BinOp::Div, var("local"), int(lanes as i64)?),
                        ),
                    ));
                }
            }
            work.push(Stmt::Let {
                name: "valid".into(),
                mutable: false,
                ty: Some("bool".into()),
                init: bin(BinOp::Lt, var("t"), int(elements as i64)?),
            });
            let mut rest = var("t");
            for (a, (&c, &e)) in part.coords.iter().zip(&part.extents).enumerate().rev() {
                let name = coord_name(c);
                if a == 0 {
                    work.push(let_int(&name, rest.clone()));
                } else {
                    work.push(let_int(&name, bin(BinOp::Rem, rest.clone(), int(i64::from(e))?)));
                    work.push(let_int(&format!("rest{a}"), bin(BinOp::Div, rest, int(i64::from(e))?)));
                    rest = var(&format!("rest{a}"));
                }
            }
            let loaded = self.load_index_params(k, &sources, &mut work)?;
            let hoisted = self.hoist(k, &mut emit, &mut work)?;

            if let Some((lanes, row_elems, exchange)) = lanes {
                if !self.chain_slots(k).is_empty() {
                    let mut chains = self.chain_slots(k);
                    for slot in &mut chains {
                        slot.slot = emit.fresh("y");
                    }
                    let mut chain_decls = Vec::new();
                    workgroup_bytes +=
                        emit.precompute_chains(&chains, lanes, row_elems, &mut chain_decls, &mut work)?;
                    decls.extend(chain_decls);
                    emit.chains = chains;
                }
                let pending = self.top_reductions(k, true, is_lane_ordered);
                let mut reduced = Vec::new();
                let mut reductions = Vec::new();
                for group in siblings(pending) {
                    let partials = emit.lanes(&group, lanes, &mut reductions)?;
                    reduced.extend(group.into_iter().zip(partials));
                }
                let mut part_decls = Vec::new();
                workgroup_bytes +=
                    emit.exchange(&reduced, lanes, row_elems, exchange, &mut part_decls, &mut reductions)?;
                let decl_dest = if !single { &mut decls } else { &mut work };
                decl_dest.extend(part_decls);
                work.extend(reductions);
            }
            let values = self.outputs(k, &mut emit, &sources, &loaded)?;
            let cond = match lanes {
                Some(_) => bin(BinOp::And, var("valid"), bin(BinOp::Eq, var("lane"), int(0)?)),
                None => var("valid"),
            };
            work.push(Stmt::If {
                cond,
                then_body: values,
                else_body: None,
            });

            let part = self.estimate(k, lanes.map(|(lanes, ..)| lanes), &hoisted, &[]);
            estimate.serial = estimate.serial.max(part.serial);
            estimate.loads = estimate.loads.max(part.loads);
            estimate.work += part.work;
            match single {
                true => body.extend(work),
                false => {
                    let first = int(groups as i64)?;
                    let end = int((groups + part_groups) as i64)?;
                    body.push(Stmt::If {
                        cond: bin(
                            BinOp::And,
                            bin(BinOp::Ge, var("group"), first),
                            bin(BinOp::Lt, var("group"), end),
                        ),
                        then_body: work,
                        else_body: None,
                    });
                }
            }
            groups += part_groups;
        }
        if groups > 65_535 {
            return Err(LowerError::Grid {
                elements: self.parts.iter().map(Part::elements).sum(),
            });
        }
        decls.extend(body);

        Ok(Lowered {
            kernel: ShaderKernel {
                name: "tensor_region".into(),
                workgroup_size: [threads, 1, 1],
                params,
                builtins,
                body: decls,
                source_map: SourceMap::default(),
                type_decls: Vec::new(),
            },
            schedule,
            groups: [groups as u32, 1, 1],
            parcels,
            scalars,
            workgroup_bytes,
            parts: self.parts.len(),
            estimate,
        })
    }

    /// Lane products nested in this part's outputs whose vector fits in one workgroup.
    fn chain_slots(&self, k: usize) -> Vec<ChainSlot> {
        let mut found = Vec::new();
        for &n in &self.parts[k].outputs {
            collect_chains(&self.bodies[n].1, None, &mut found);
        }
        found
    }

    /// The distinct reductions `keep` selects outside any select and any other
    /// reduction of part `k`'s outputs, in evaluation order; with `whole`, of only the
    /// outputs that span the part's whole domain.
    pub(super) fn top_reductions(&self, k: usize, whole: bool, keep: impl Fn(&&Term) -> bool) -> Vec<&Term> {
        let mut found: Vec<&Term> = Vec::new();
        for &n in &self.parts[k].outputs {
            if whole && !self.limits[n].is_empty() {
                continue;
            }
            for r in top_reductions(&self.bodies[n].1).into_iter().filter(&keep) {
                if !found.iter().any(|f| f.alpha_eq(r)) {
                    found.push(r);
                }
            }
        }
        found
    }

    /// Computes every output of part `k` at element `c{n}` into a local and stores them
    /// all, the stores after every computation. Sequential sibling reductions of the
    /// outputs that span the whole domain share loops.
    pub(super) fn outputs(
        &self,
        k: usize,
        emit: &mut Emit,
        sources: &[IndexSource],
        loaded: &[bool],
    ) -> Result<Vec<Stmt>, LowerError> {
        let mut values = Vec::new();
        let sequential = self.top_reductions(k, true, |r| {
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
        let coords = &self.parts[k].coords;
        for &n in &self.parts[k].outputs {
            let ((o, term), at, limits) = (&self.bodies[n], &self.at[n], &self.limits[n]);
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
                .map(|&(a, e)| Ok(bin(BinOp::Lt, var(&coord_name(coords[a])), int(i64::from(e))?)))
                .reduce(|a, b| Ok(bin(BinOp::And, a?, b?)))
                .transpose()?
            else {
                let x = emit.term(term, &mut values)?;
                values.push(let_float(&name, x));
                stores.extend(zero_outside(sources, loaded, &name));
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
            let mut guarded_store = zero_outside(sources, loaded, &name);
            guarded_store.push(store);
            stores.push(Stmt::If {
                cond: within,
                then_body: guarded_store,
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

/// Nested lane-ordered products whose only free index is the enclosing reduction's.
/// `CHAIN_VECTOR_MAX` only bounds scratch; the cost model decides whether the chain
/// is worth fusing. Inner products come first.
fn collect_chains(term: &Term, enclosing: Option<(IndexVar, u32)>, out: &mut Vec<ChainSlot>) {
    match term {
        Term::Reduce {
            index,
            extent,
            order,
            body,
            ..
        } => {
            collect_chains(body, Some((*index, *extent)), out);
            if let (Some((enc_index, enc_extent)), ReduceOrder::Lanes { .. }) = (enclosing, *order) {
                let free = term.free_indices();
                let fits = enc_extent > 0 && enc_extent <= CHAIN_VECTOR_MAX;
                if fits
                    && free.len() == 1
                    && free.contains(&enc_index)
                    && !out.iter().any(|slot| slot.term.alpha_eq(term))
                {
                    out.push(ChainSlot {
                        term: term.clone(),
                        slot: String::new(),
                        index: enc_index,
                        extent: enc_extent,
                    });
                }
            }
        }
        Term::Select { .. } => {}
        other => other.for_each_child(&mut |child| collect_chains(child, enclosing, out)),
    }
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
    /// Terms hoisted out of every loop: any use, at any depth, reads the local.
    hoisted: Vec<(Term, String)>,
    /// Loads made at the depth of `.0`, by their expression: the step of a sibling
    /// loop reads each element its reductions share once.
    loads: Option<(u32, Vec<(Expr, String)>)>,
    /// Nesting inside selects and reductions; only depth 0 is cached.
    pub(super) depth: u32,
    /// The output being lowered, for errors.
    output: String,
    /// When false, a lane reduction runs on every thread. The consumer's `valid`
    /// flag does not apply to a vector computed once per workgroup.
    lane_guard: bool,
    /// Nested lane reductions materialized in workgroup memory for this part.
    chains: Vec<ChainSlot>,
}

/// A lane-ordered product computed once per workgroup into `slot`, then read at `index`.
struct ChainSlot {
    term: Term,
    slot: String,
    index: IndexVar,
    extent: u32,
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

    fn forget(&mut self, term: &Term) {
        self.cache.retain(|(t, _)| !t.alpha_eq(term));
    }

    fn chain_slot(&self, term: &Term) -> Option<(String, IndexVar)> {
        self.chains
            .iter()
            .find(|slot| slot.term.alpha_eq(term))
            .map(|slot| (slot.slot.clone(), slot.index))
    }

    /// Computes each nested lane product in `slots` into workgroup memory, once per
    /// workgroup, in the same 32-lane association the product names. Every thread
    /// participates. Returns the bytes of workgroup memory.
    fn precompute_chains(
        &mut self,
        slots: &[ChainSlot],
        lanes: u32,
        elements: u32,
        decls: &mut Vec<Stmt>,
        out: &mut Vec<Stmt>,
    ) -> Result<u32, LowerError> {
        if slots.is_empty() {
            return Ok(0);
        }
        let scratch = self.fresh("r");
        decls.push(Stmt::WorkgroupArray {
            name: scratch.clone(),
            elem: "float".into(),
            len: lanes * elements,
        });
        let mut bytes = lanes * elements * 4;
        let slot_at = |name: &str, at: Expr| Expr::Index {
            base: Box::new(var(name)),
            index: Box::new(at),
        };
        for slot in slots {
            let Term::Reduce { op, .. } = &slot.term else {
                continue;
            };
            let op = *op;
            decls.push(Stmt::WorkgroupArray {
                name: slot.slot.clone(),
                elem: "float".into(),
                len: slot.extent,
            });
            bytes += slot.extent * 4;
            // Row group `g` computes element `base + g`, so the four groups in a
            // 128-thread workgroup do not repeat the same product.
            let base = self.fresh("i");
            let elem = self.fresh("e");
            out.push(let_int(&base, int(0)?));
            out.push(let_int(
                &elem,
                bin(
                    BinOp::Add,
                    var(&base),
                    bin(BinOp::Div, var("local"), int(i64::from(lanes))?),
                ),
            ));
            let old = self.bind(slot.index, elem.clone(), slot.extent);
            self.lane_guard = false;
            let mut body = Vec::new();
            body.push(assign(
                &elem,
                bin(
                    BinOp::Add,
                    var(&base),
                    bin(BinOp::Div, var("local"), int(i64::from(lanes))?),
                ),
            ));
            let active = bin(BinOp::Lt, var(&elem), int(i64::from(slot.extent))?);
            // An inactive group still runs the reduction, on element 0, so the
            // barriers stay uniform and the load stays in range.
            body.push(Stmt::If {
                cond: Expr::Unary {
                    op: IrUnaryOp::Not,
                    expr: Box::new(active.clone()),
                },
                then_body: vec![assign(&elem, int(0)?)],
                else_body: None,
            });
            let group = [&slot.term];
            let partials = self.lanes(&group, lanes, &mut body)?;
            self.lane_guard = true;
            let partial = partials.into_iter().next().expect("one reduction");
            body.push(Stmt::Assign {
                target: slot_at(&scratch, var("local")),
                value: partial,
            });
            body.push(barrier());
            let s = self.fresh("s");
            body.push(let_int(&s, int(i64::from(lanes / 2))?));
            let combined = Stmt::Assign {
                target: slot_at(&scratch, var("local")),
                value: combine(
                    op,
                    slot_at(&scratch, var("local")),
                    slot_at(&scratch, bin(BinOp::Add, var("local"), var(&s))),
                ),
            };
            body.push(Stmt::While {
                cond: bin(BinOp::Gt, var(&s), int(0)?),
                body: vec![
                    Stmt::If {
                        cond: bin(BinOp::Lt, var("lane"), var(&s)),
                        then_body: vec![combined],
                        else_body: None,
                    },
                    barrier(),
                    assign(&s, bin(BinOp::Div, var(&s), int(2)?)),
                ],
            });
            let result = self.fresh("v");
            body.push(let_float(
                &result,
                slot_at(&scratch, bin(BinOp::Sub, var("local"), var("lane"))),
            ));
            body.push(Stmt::If {
                cond: bin(BinOp::And, active, bin(BinOp::Eq, var("lane"), int(0)?)),
                then_body: vec![Stmt::Assign {
                    target: slot_at(&slot.slot, cast(var(&elem), "uint")),
                    value: var(&result),
                }],
                else_body: None,
            });
            body.push(barrier());
            body.push(assign(&base, bin(BinOp::Add, var(&base), int(i64::from(elements))?)));
            self.unbind(slot.index, old);
            self.forget(&slot.term);
            out.push(Stmt::While {
                cond: bin(BinOp::Lt, var(&base), int(i64::from(slot.extent))?),
                body,
            });
        }
        Ok(bytes)
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
        if let Some((_, name)) = self.hoisted.iter().find(|(h, _)| h.alpha_eq(t)) {
            return Ok(var(name));
        }
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
            Term::Unary {
                op: UnaryOp::Round,
                arg,
            } => match &**arg {
                Term::Binary {
                    op: op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div),
                    lhs,
                    rhs,
                } => {
                    let a = self.term(lhs, out)?;
                    let b = self.term(rhs, out)?;
                    let exact = match op {
                        BinaryOp::Add => BuiltinFn::ExactAdd,
                        BinaryOp::Sub => BuiltinFn::ExactSub,
                        BinaryOp::Mul => BuiltinFn::ExactMul,
                        _ => BuiltinFn::ExactDiv,
                    };
                    call(exact, vec![a, b])
                }
                Term::Unary {
                    op: UnaryOp::Recip,
                    arg,
                } => {
                    let x = self.term(arg, out)?;
                    call(BuiltinFn::ExactDiv, vec![float(1.0), x])
                }
                other => self.term(other, out)?,
            },
            Term::Unary { op, arg } => {
                let x = self.term(arg, out)?;
                match op {
                    UnaryOp::Round => x,
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
                    UnaryOp::Tanh => call(BuiltinFn::Tanh, vec![x]),
                    UnaryOp::Sigmoid => call(BuiltinFn::Sigmoid, vec![x]),
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
                if let Some((slot, axis)) = self.chain_slot(t) {
                    let name = self.names.get(&axis).cloned().ok_or_else(|| LowerError::Order {
                        output: self.output.clone(),
                    })?;
                    return Ok(Expr::Index {
                        base: Box::new(var(&slot)),
                        index: Box::new(cast(var(&name), "uint")),
                    });
                }
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
        if self.lane_guard {
            out.push(Stmt::If {
                cond: var("valid"),
                then_body: strided,
                else_body: None,
            });
        } else {
            out.extend(strided);
        }
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

/// The parent-buffer element `element` names. A coordinate is clamped into the
/// range `ranges` gives it, so a thread past the domain still reads an element
/// the caller allocated.
fn element_address(
    element: &Affine,
    names: &HashMap<IndexVar, String>,
    ranges: &HashMap<IndexVar, (i64, i64)>,
    body: &mut Vec<Stmt>,
) -> Result<Expr, LowerError> {
    if let Some(c) = element.as_constant() {
        let c = u32::try_from(c).map_err(|_| LowerError::IndexRange)?;
        return Ok(Expr::LitU32(c));
    }
    let mut lo = element.constant_term();
    let mut hi = element.constant_term();
    let mut var_part: Option<Expr> = None;
    for &(sym, coefficient) in element.terms() {
        let Sym::Index(v) = sym else {
            return Err(LowerError::IndexRange);
        };
        if coefficient <= 0 {
            return Err(LowerError::IndexRange);
        }
        let (start, end) = ranges.get(&v).copied().ok_or(LowerError::IndexRange)?;
        if start != 0 {
            return Err(LowerError::IndexRange);
        }
        lo += coefficient.saturating_mul(start);
        hi += coefficient.saturating_mul(end);
        let name = names.get(&v).ok_or(LowerError::IndexRange)?;
        let clamped = format!("{name}_ix{coefficient}");
        body.push(let_int(&clamped, call(BuiltinFn::Min, vec![var(name), int(end)?])));
        let term = if coefficient == 1 {
            var(&clamped)
        } else {
            bin(BinOp::Mul, var(&clamped), int(coefficient)?)
        };
        var_part = Some(match var_part {
            None => term,
            Some(e) => bin(BinOp::Add, e, term),
        });
    }
    if lo < 0 || hi > i64::from(u32::MAX) {
        return Err(LowerError::IndexRange);
    }
    let c = element.constant_term();
    let c = u32::try_from(c).map_err(|_| LowerError::IndexRange)?;
    let Some(var_part) = var_part else {
        return Ok(Expr::LitU32(c));
    };
    let sum = cast(var_part, "uint");
    Ok(match c {
        0 => sum,
        _ => bin(BinOp::Add, sum, Expr::LitU32(c)),
    })
}

/// Stores that depend on an out-of-range index parameter write zero.
/// `loaded[p]` is whether this part read parameter `p`.
fn zero_outside(sources: &[IndexSource], loaded: &[bool], name: &str) -> Vec<Stmt> {
    sources
        .iter()
        .enumerate()
        .filter(|(p, source)| source.zero_if_outside && loaded.get(*p).copied().unwrap_or(false))
        .map(|(p, _)| Stmt::If {
            cond: Expr::Unary {
                op: IrUnaryOp::Not,
                expr: Box::new(var(&format!("in{p}"))),
            },
            then_body: vec![assign(name, float(0.0))],
            else_body: None,
        })
        .collect()
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
