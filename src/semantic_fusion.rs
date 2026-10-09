//! Semantic fusion: recorded operations described as tensor algebra, and runs of them
//! synthesized as one kernel.
//!
//! An operation whose meaning Goldy knows records a [`SemanticSite`] beside its node: a
//! named [`Op`] over the parcels it binds, in which tensor views are storage maps. A
//! matrix product is a [`Contraction`] whose axes carry their roles, so its structure
//! survives composition. Automatic fusion (`fusion_plan.rs`) composes adjacent sites in
//! a [`Graph`], which forwards what one site stores to the later sites that read it, and
//! lowers the composition to one kernel with [`algebra::lower_graph`], choosing among
//! the schedules the device's [`Target`] offers. Each lifted reduction names the
//! association its recorded kernel uses and lowering reproduces it, so the synthesized
//! kernel stores what the recorded sequence stores, bit for bit, unless the scheme
//! admits [`ContractionPrecision::F16Factors`] and a contraction runs on matrix units.
//!
//! Sites name whole buffers. A view's offset and strides are relative to its buffer, and
//! the kernel binds the buffer's own descriptor, as the recorded kernels do. Nothing
//! lifted binds a scheme temporary, so every value a site stores stays an output.

use crate::backend::BufferHandle;
use crate::fusion_plan::{KernelSite, SiteArg};
use crate::kernel::LIMITS;
use crate::ops::matmul::{MatMulFallback, MatMulOperand};
use crate::ops::matmul_kernel::GEMV_ORDER;
use crate::ops::MatMulDesc;
use crate::parcel::Parcel;
use crate::task_graph::{DispatchDim, NodeKind, ResourceId, TaskNode};
use crate::types::ResourceAccess;
use goldy_shader_ir::algebra::{
    self, BinaryOp, Contraction, ContractionPrecision, Graph, IndexSource, Lowered, Map, Op, OpKind, Operand, Params,
    ParcelId, ReduceOrder, Storage, Target, Term, UnaryOp,
};
use goldy_shader_ir::{
    BinOp, BuiltinFn, ElementType, Expr, FusionRejection, KernelDef, KernelId, ParamCategory, ScalarType, Stmt,
    UnaryOp as IrUnaryOp,
};
use std::collections::HashMap;

/// A buffer a semantic site reads or writes, with its whole-buffer descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SiteParcel {
    pub(crate) buffer: BufferHandle,
    pub(crate) srv: Option<u32>,
    pub(crate) uav: Option<u32>,
}

impl SiteParcel {
    /// `parcel`, if it is a whole buffer.
    pub(crate) fn of(parcel: &Parcel) -> Option<Self> {
        let ResourceId::Buffer(buffer) = parcel.resource_id() else {
            return None;
        };
        Some(Self {
            buffer,
            srv: parcel.resource_index(ResourceAccess::Read),
            uav: parcel.resource_index(ResourceAccess::Write),
        })
    }
}

/// What a recorded node computes, as a named tensor operation.
#[derive(Debug, Clone)]
pub(crate) struct SemanticSite {
    /// The `ParcelId(i)` of its operands is `parcels[i]`.
    pub(crate) op: Op,
    pub(crate) parcels: Vec<SiteParcel>,
    /// Where each index parameter's value is read, by parameter.
    pub(crate) sources: Vec<IndexSource>,
    /// The node user slot each scalar parameter binds, by parameter.
    pub(crate) scalars: Vec<usize>,
    /// Whether `op` sums in the association the recorded node does. A library product
    /// sums in its own, so its site joins a region only under a
    /// [`ContractionPrecision`] that lets contractions reassociate.
    pub(crate) exact: bool,
}

/// A [`SemanticSite`] under construction.
#[derive(Default)]
pub(crate) struct SiteBuilder {
    params: Params,
    parcels: Vec<SiteParcel>,
    sources: Vec<IndexSource>,
    scalars: Vec<usize>,
}

impl SiteBuilder {
    pub(crate) fn parcel(&mut self, parcel: SiteParcel) -> ParcelId {
        let at = match self.parcels.iter().position(|p| p.buffer == parcel.buffer) {
            Some(at) => at,
            None => {
                self.parcels.push(parcel);
                self.parcels.len() - 1
            }
        };
        ParcelId(at as u32)
    }

    /// A scalar parameter bound by user slot `slot`.
    pub(crate) fn scalar(&mut self, name: &str, slot: usize) -> algebra::ScalarParam {
        self.scalars.push(slot);
        self.params.scalar(name)
    }

    /// An index parameter in `range`, read from `source`.
    #[cfg(feature = "tensor")]
    pub(crate) fn index_param(
        &mut self,
        name: &str,
        range: std::ops::Range<i64>,
        source: IndexSource,
    ) -> algebra::IndexParam {
        self.sources.push(source);
        self.params.index_param(name, range)
    }

    /// The site computing `kind`, if it is well formed.
    pub(crate) fn finish(self, kind: OpKind) -> Option<SemanticSite> {
        let op = Op::new(self.params, kind);
        op.expand().ok()?;
        Some(SemanticSite {
            op,
            parcels: self.parcels,
            sources: self.sources,
            scalars: self.scalars,
            exact: true,
        })
    }
}

/// `C = op(A) @ op(B)` on the backend library, as a sequential contraction: a site that
/// states what the product is but not how the library sums it.
pub(crate) fn library_matmul(
    desc: &MatMulDesc,
    fallback: MatMulFallback,
    operands: [(&MatMulOperand, SiteParcel); 3],
) -> Option<SemanticSite> {
    use crate::ops::matmul::{packed_leading_dim, OperandKind};
    let kinds = [OperandKind::A, OperandKind::B, OperandKind::C];
    let packed = operands
        .iter()
        .zip(kinds)
        .all(|((o, _), kind)| o.leading_dim == packed_leading_dim(desc, kind));
    if fallback != MatMulFallback::Gemm || !packed {
        return None;
    }
    let site = matmul(desc, fallback, operands)?;
    Some(SemanticSite { exact: false, ..site })
}

/// `C = op(A) @ op(B)` as the stdlib kernel `fallback` computes it.
pub(crate) fn matmul(
    desc: &MatMulDesc,
    fallback: MatMulFallback,
    [(a, pa), (b, pb), (c, pc)]: [(&MatMulOperand, SiteParcel); 3],
) -> Option<SemanticSite> {
    if !desc.requires_stdlib_identity_epilogue() {
        return None;
    }
    let (m, n, k) = (desc.m, desc.n, desc.k);
    let mut s = SiteBuilder::default();
    let (pa, pb, pc) = (s.parcel(pa), s.parcel(pb), s.parcel(pc));
    let operand = |name: &str, shape: &[u32], parcel, o: &MatMulOperand, strides: &[i64]| {
        let offset = i64::try_from(o.offset_elements).ok()?;
        Some(Operand::new(name, shape, Storage::strided(parcel, offset, strides)))
    };
    let contraction = match fallback {
        // y[i] = sum{s}(A[i, s] * x[s]), labels i and s.
        MatMulFallback::Gemv => {
            let ld = |o: &MatMulOperand| i64::from(o.leading_dim);
            Contraction {
                extents: vec![m, k],
                lhs: operand("A", &[m, k], pa, a, &[ld(a), 1])?,
                rhs: operand("x", &[k], pb, b, &[ld(b)])?,
                out: operand("y", &[m], pc, c, &[ld(c)])?,
                labels: [vec![0, 1], vec![1], vec![0]],
                order: GEMV_ORDER,
            }
        }
        // C[i, j] = sum{s}(A[i, s] * B[s, j]), labels i, j and s.
        MatMulFallback::Gemm => {
            let (m64, n64, k64) = (i64::from(m), i64::from(n), i64::from(k));
            let a_strides = if desc.transpose_a { [1, m64] } else { [k64, 1] };
            let b_strides = if desc.transpose_b { [1, k64] } else { [n64, 1] };
            Contraction {
                extents: vec![m, n, k],
                lhs: operand("A", &[m, k], pa, a, &a_strides)?,
                rhs: operand("B", &[k, n], pb, b, &b_strides)?,
                out: operand("C", &[m, n], pc, c, &[n64, 1])?,
                labels: [vec![0, 2], vec![2, 1], vec![0, 1]],
                order: ReduceOrder::Sequential,
            }
        }
    };
    s.finish(OpKind::Contraction(contraction))
}

/// A generated kernel whose body is a guarded pointwise store at the thread's own index,
/// `let i = global_id().x; if i < n { out[i] = f(x[i], ..); }`, dispatched in 1D.
///
/// `f` is f32 arithmetic over whole-buffer elements at `i`, f32 scalars and literals.
/// The extent is `n`, capped by the grid; an integer `n` is the value its user slot
/// holds now, which the region then depends on. A read after a store in the body sees
/// the stored value.
pub(crate) fn lift_kernel(site: &KernelSite, node: &TaskNode) -> Option<SemanticSite> {
    let NodeKind::Dispatch {
        resource_slots,
        user_slots,
        dispatch: DispatchDim::Direct { x, y: 1, z: 1 },
        ..
    } = &node.kind
    else {
        return None;
    };
    let def = site.kernel.definition.as_ref()?;
    let [wx, 1, 1] = site.kernel.workgroup_size else {
        return None;
    };
    let threads = x.checked_mul(wx)?;

    let mut s = SiteBuilder::default();
    let mut formals = HashMap::new();
    for (param, arg) in site.kernel.params.iter().zip(&site.args) {
        let formal = match (*arg, param.scalar) {
            (
                SiteArg::Resource {
                    resource: Some(ResourceId::Buffer(buffer)),
                    slot,
                    descriptor,
                },
                None,
            ) if !param.is_tensor
                && param.slang_type == ElementType::F32.slang_name()
                && matches!(
                    param.category,
                    ParamCategory::BufferRead | ParamCategory::BufferReadWrite | ParamCategory::BufferWrite
                ) =>
            {
                let bindless = *resource_slots.get(slot)?;
                let read = descriptor == ResourceAccess::Read;
                let parcel = s.parcel(SiteParcel {
                    buffer,
                    srv: read.then_some(bindless),
                    uav: (!read).then_some(bindless),
                });
                Formal::Buffer(parcel)
            }
            (SiteArg::Scalar { slot }, Some(ScalarType::F32)) => Formal::F32(slot),
            (SiteArg::Scalar { slot }, Some(ScalarType::U32 | ScalarType::I32)) => Formal::Int(*user_slots.get(slot)?),
            _ => return None,
        };
        formals.insert(param.name.as_str(), formal);
    }

    let (own, bound, body) = guarded_body(&def.body)?;
    let extent = match bound {
        Expr::Var(name) => match formals.get(name.as_str())? {
            Formal::Int(n) => *n,
            _ => return None,
        },
        Expr::LitU32(n) => *n,
        Expr::LitI32(n) => u32::try_from(*n).ok()?,
        _ => return None,
    }
    .min(threads);
    if extent == 0 {
        return None;
    }

    let mut lift = Lift {
        site: s,
        formals,
        own,
        locals: HashMap::new(),
        inputs: Vec::new(),
        stores: Vec::new(),
        scalars: HashMap::new(),
    };
    for stmt in body {
        lift.stmt(stmt)?;
    }
    let Lift {
        site: s,
        inputs,
        stores,
        ..
    } = lift;
    if stores.is_empty() {
        return None;
    }
    let operand = |prefix: &str, parcel: ParcelId| {
        Operand::new(
            &format!("{prefix}{}", parcel.0),
            &[extent],
            Storage::strided(parcel, 0, &[1]),
        )
    };
    s.finish(OpKind::Map(Map {
        shape: vec![extent],
        inputs: inputs.into_iter().map(|p| operand("x", p)).collect(),
        outputs: stores.into_iter().map(|(p, term)| (operand("p", p), term)).collect(),
    }))
}

/// `(i, n, body)` of `let i = global_id().x;` followed by `if i < n { body }` or by
/// `if i >= n { return; } body`.
fn guarded_body(stmts: &[Stmt]) -> Option<(&str, &Expr, &[Stmt])> {
    let stmts = match stmts {
        [rest @ .., Stmt::Return { value: None }] => rest,
        _ => stmts,
    };
    let [Stmt::Let {
        name,
        mutable: false,
        init,
        ..
    }, Stmt::If {
        cond: Expr::Binary { op, left, right },
        then_body,
        else_body: None,
    }, rest @ ..] = stmts
    else {
        return None;
    };
    if !is_global_x(init) || !matches!(&**left, Expr::Var(v) if v == name) {
        return None;
    }
    match (op, then_body.as_slice(), rest) {
        (BinOp::Lt, body, []) => Some((name, right, body)),
        (BinOp::Ge, [Stmt::Return { value: None }], body) => Some((name, right, body)),
        _ => None,
    }
}

fn is_global_x(expr: &Expr) -> bool {
    match expr {
        Expr::Field { base, field } => {
            field == "x"
                && matches!(
                    &**base,
                    Expr::Call {
                        func: BuiltinFn::GlobalId,
                        ..
                    }
                )
        }
        Expr::Cast { expr, ty } => ty == "uint" && is_global_x(expr),
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum Formal {
    Buffer(ParcelId),
    /// An f32 scalar in this user slot.
    F32(usize),
    /// An integer scalar and the value its slot holds.
    Int(u32),
}

struct Lift<'a> {
    site: SiteBuilder,
    formals: HashMap<&'a str, Formal>,
    own: &'a str,
    locals: HashMap<&'a str, Term>,
    /// The parcels read before any store to them, by argument position.
    inputs: Vec<ParcelId>,
    /// Each stored parcel's latest value over the arguments, in first-store order.
    stores: Vec<(ParcelId, Term)>,
    scalars: HashMap<usize, algebra::ScalarParam>,
}

impl<'a> Lift<'a> {
    fn stmt(&mut self, stmt: &'a Stmt) -> Option<()> {
        match stmt {
            Stmt::Let { name, init, .. } if name != self.own => {
                let term = self.expr(init)?;
                self.locals.insert(name, term);
            }
            Stmt::Assign {
                target: Expr::Var(name),
                value,
            } if self.locals.contains_key(name.as_str()) => {
                let term = self.expr(value)?;
                self.locals.insert(name, term);
            }
            Stmt::Assign { target, value } => {
                let parcel = self.element(target)?;
                let term = self.expr(value)?;
                match self.stores.iter_mut().find(|(p, _)| *p == parcel) {
                    Some(store) => store.1 = term,
                    None => self.stores.push((parcel, term)),
                }
            }
            _ => return None,
        }
        Some(())
    }

    /// The parcel `expr` indexes at the thread's own index.
    fn element(&self, expr: &Expr) -> Option<ParcelId> {
        let Expr::Index { base, index } = expr else {
            return None;
        };
        let (Expr::Var(base), Expr::Var(index)) = (&**base, &**index) else {
            return None;
        };
        if index != self.own || self.locals.contains_key(base.as_str()) {
            return None;
        }
        match self.formals.get(base.as_str())? {
            Formal::Buffer(parcel) => Some(*parcel),
            _ => None,
        }
    }

    fn expr(&mut self, expr: &Expr) -> Option<Term> {
        Some(match expr {
            Expr::LitF32(v) => Term::lit(*v),
            Expr::Var(name) => match self.locals.get(name.as_str()) {
                Some(term) => term.clone(),
                None => match *self.formals.get(name.as_str())? {
                    Formal::F32(slot) => {
                        let site = &mut self.site;
                        let param = *self.scalars.entry(slot).or_insert_with(|| site.scalar(name, slot));
                        Term::scalar(param)
                    }
                    _ => return None,
                },
            },
            Expr::Index { .. } => {
                let parcel = self.element(expr)?;
                if let Some((_, stored)) = self.stores.iter().find(|(p, _)| *p == parcel) {
                    return Some(stored.clone());
                }
                let arg = match self.inputs.iter().position(|p| *p == parcel) {
                    Some(arg) => arg,
                    None => {
                        self.inputs.push(parcel);
                        self.inputs.len() - 1
                    }
                };
                Term::arg(arg)
            }
            Expr::Binary { op, left, right } => {
                let op = match op {
                    BinOp::Add => BinaryOp::Add,
                    BinOp::Sub => BinaryOp::Sub,
                    BinOp::Mul => BinaryOp::Mul,
                    BinOp::Div => BinaryOp::Div,
                    _ => return None,
                };
                Term::binary(op, self.expr(left)?, self.expr(right)?)
            }
            Expr::Unary {
                op: IrUnaryOp::Neg,
                expr,
            } => Term::unary(UnaryOp::Neg, self.expr(expr)?),
            Expr::Call { func, args } => match (func, args.as_slice()) {
                (BuiltinFn::Min, [a, b]) => Term::binary(BinaryOp::Min, self.expr(a)?, self.expr(b)?),
                (BuiltinFn::Max, [a, b]) => Term::binary(BinaryOp::Max, self.expr(a)?, self.expr(b)?),
                (BuiltinFn::ExactMul, [a, b]) => Term::unary(
                    UnaryOp::Round,
                    Term::binary(BinaryOp::Mul, self.expr(a)?, self.expr(b)?),
                ),
                (BuiltinFn::ExactDiv, [a, b]) => Term::unary(
                    UnaryOp::Round,
                    Term::binary(BinaryOp::Div, self.expr(a)?, self.expr(b)?),
                ),
                (f, [a]) => {
                    let op = match f {
                        BuiltinFn::Abs => UnaryOp::Abs,
                        BuiltinFn::Sqrt => UnaryOp::Sqrt,
                        BuiltinFn::Exp => UnaryOp::Exp,
                        BuiltinFn::Log => UnaryOp::Log,
                        BuiltinFn::Sin => UnaryOp::Sin,
                        BuiltinFn::Cos => UnaryOp::Cos,
                        _ => return None,
                    };
                    Term::unary(op, self.expr(a)?)
                }
                _ => return None,
            },
            _ => return None,
        })
    }
}

/// A run of semantic sites composed and lowered to one kernel.
pub(crate) struct SemanticProgram {
    pub(crate) def: KernelDef,
    pub(crate) lowered: Lowered,
    /// `ParcelId(i)` of the composed region is `parcels[i]`.
    pub(crate) parcels: Vec<SiteParcel>,
    /// Per scalar parameter of the composed region: the constituent and user slot it binds.
    pub(crate) scalars: Vec<(usize, usize)>,
    /// The constituents' operations, composed; operands are named `p{parcel}`.
    pub(crate) graph: Graph,
}

impl SemanticProgram {
    pub(crate) fn id(&self) -> KernelId {
        self.def.id()
    }

    /// Inputs that read a value an earlier constituent defines rather than storage.
    pub(crate) fn forwarded(&self) -> usize {
        self.graph.edges().len()
    }

    /// The composition's contractions with their prologues and epilogues, one per line.
    pub(crate) fn structure(&self) -> String {
        self.graph.structure().to_string()
    }

    /// Each synthesized scalar parameter's origin, `constituent:slot`.
    pub(crate) fn scalar_origins(&self) -> Vec<String> {
        self.lowered
            .scalars
            .iter()
            .map(|s| {
                let (k, slot) = self.scalars[s.index()];
                format!("{k}:slot{slot}")
            })
            .collect()
    }
}

/// What a device with `caps` offers semantic fusion.
pub(crate) fn target(caps: &crate::runtime::RuntimeCapabilities) -> Target {
    Target {
        subgroup: caps.subgroup_width,
        matrix: caps.matrix_multiply,
    }
}

/// Compose `sites`, in execution order, and lower the composition to one kernel for
/// `target`, rounding contractions no more than `precision` admits.
pub(crate) fn synthesize(
    sites: &[&SemanticSite],
    target: &Target,
    precision: ContractionPrecision,
) -> Result<SemanticProgram, FusionRejection> {
    let mut parcels: Vec<SiteParcel> = Vec::new();
    let mut graph = Graph::default();
    let mut sources = Vec::new();
    let mut scalars = Vec::new();
    for (k, site) in sites.iter().enumerate() {
        let map: Vec<ParcelId> = site
            .parcels
            .iter()
            .map(|p| match parcels.iter_mut().position(|q| q.buffer == p.buffer) {
                Some(at) => {
                    let q = &mut parcels[at];
                    q.srv = q.srv.or(p.srv);
                    q.uav = q.uav.or(p.uav);
                    ParcelId(at as u32)
                }
                None => {
                    parcels.push(*p);
                    ParcelId(parcels.len() as u32 - 1)
                }
            })
            .collect();
        let mut op = site.op.clone();
        op.map_parcels(|p| map[p.0 as usize]);
        op.rename_operands(|o| format!("p{}", o.storage.parcel.0));
        sources.extend(site.sources.iter().map(|s| IndexSource {
            parcel: map[s.parcel.0 as usize],
            ..*s
        }));
        scalars.extend(site.scalars.iter().map(|&slot| (k, slot)));
        graph.push(op).map_err(|e| FusionRejection::Semantic {
            stage: k,
            reason: e.to_string(),
        })?;
    }
    if graph.ops().is_empty() {
        return Err(FusionRejection::Empty);
    }
    let lowered = algebra::lower_graph(&graph, &sources, target, precision).map_err(|e| FusionRejection::Semantic {
        stage: sites.len() - 1,
        reason: e.to_string(),
    })?;
    if lowered.parcels.len() > LIMITS.resources {
        return Err(FusionRejection::ResourceLimit {
            count: lowered.parcels.len(),
            max: LIMITS.resources,
        });
    }
    if lowered.scalars.len() > LIMITS.scalars {
        return Err(FusionRejection::ScalarLimit {
            count: lowered.scalars.len(),
            max: LIMITS.scalars,
        });
    }
    if lowered.workgroup_bytes > LIMITS.workgroup_bytes {
        return Err(FusionRejection::WorkgroupMemory {
            bytes: lowered.workgroup_bytes,
            max: LIMITS.workgroup_bytes,
        });
    }
    let def = goldy_shader_ir::emit_canonical_compute_source(&lowered.kernel);
    Ok(SemanticProgram {
        def,
        lowered,
        parcels,
        scalars,
        graph,
    })
}
