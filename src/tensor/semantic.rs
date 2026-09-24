//! Index-notation meaning of the tensor recorder's kernels (see `semantic_fusion.rs`).
//!
//! Each function states what one recorded dispatch of `kernels.rs` stores, reading its
//! operands through their views. It returns `None` for the cases it does not describe;
//! those dispatches stay opaque to semantic fusion.

use super::dtype::TensorDType;
use super::kernels::{
    OP_ABS, OP_ADD, OP_ADD_SCALAR, OP_DIV, OP_DIV_SCALAR, OP_EXP, OP_LOG, OP_MAX, OP_MAX_SCALAR, OP_MEAN, OP_MIN,
    OP_MIN_SCALAR, OP_MUL, OP_MUL_SCALAR, OP_NEG, OP_RECIP, OP_RMAX, OP_RMIN, OP_SQRT, OP_SUB, OP_SUB_SCALAR, OP_SUM,
};
use super::view::TensorView;
use crate::semantic_fusion::{SemanticSite, SiteBuilder, SiteParcel};
use goldy_shader_ir::algebra::{Affine, BinaryOp, IndexSource, IndexVar, ReduceOp, Storage, Term, UnaryOp};
use goldy_shader_ir::ElementType;

/// The accumulator `tensor_reduce_f32` starts a max reduction from; a min starts from its negation.
const REDUCE_MAX_START: f32 = -3.402823e38;

fn storage(s: &mut SiteBuilder, view: TensorView<'_>) -> Option<Storage> {
    let parcel = s.parcel(SiteParcel::of(view.buffer().whole())?);
    let offset = i64::try_from(view.storage_offset()).ok()?;
    Some(Storage::strided(parcel, offset, view.layout().strides()))
}

fn dims(view: TensorView<'_>) -> Option<Vec<u32>> {
    let dims = view.shape().dims().to_vec();
    (view.dtype() == TensorDType::F32 && !dims.contains(&0)).then_some(dims)
}

fn indices(s: &mut SiteBuilder, rank: usize) -> Vec<IndexVar> {
    (0..rank).map(|a| s.region.index(&format!("i{a}"))).collect()
}

/// `out[i] = body(inputs[i]..)` over `out`'s shape; `inputs` already have that shape.
///
/// With `scalar`, the body also gets the f32 in user slot 0.
fn elementwise(
    inputs: &[TensorView<'_>],
    out: TensorView<'_>,
    scalar: bool,
    body: impl FnOnce(Vec<Term>, Option<Term>) -> Option<Term>,
) -> Option<SemanticSite> {
    let shape = dims(out)?;
    let mut s = SiteBuilder::default();
    let at = indices(&mut s, shape.len());
    let mut reads = Vec::with_capacity(inputs.len());
    for (n, &view) in inputs.iter().enumerate() {
        if dims(view)? != shape {
            return None;
        }
        let storage = storage(&mut s, view)?;
        let value = s.region.input(&format!("x{n}"), &shape, storage);
        reads.push(Term::read(value, at.iter().copied()));
    }
    let scalar = scalar.then(|| Term::scalar(s.scalar("scalar", 0)));
    let body = body(reads, scalar)?;
    let storage = storage(&mut s, out)?;
    s.region.output("out", &shape, &at, body, storage);
    s.finish()
}

/// `tensor_unary_f32` with `op` from `src` into `dst`.
pub(crate) fn unary(op: u32, src: TensorView<'_>, dst: TensorView<'_>) -> Option<SemanticSite> {
    let op = match op {
        OP_NEG => UnaryOp::Neg,
        OP_ABS => UnaryOp::Abs,
        OP_EXP => UnaryOp::Exp,
        OP_LOG => UnaryOp::Log,
        OP_SQRT => UnaryOp::Sqrt,
        OP_RECIP => UnaryOp::Recip,
        _ => return None,
    };
    elementwise(&[src], dst, false, |x, _| Some(Term::unary(op, x.into_iter().next()?)))
}

/// `tensor_unary_f32` filling `out` with its scalar.
pub(crate) fn fill(out: TensorView<'_>) -> Option<SemanticSite> {
    elementwise(&[], out, true, |_, scalar| scalar)
}

/// An f32 `tensor_copy_u32` from `src` into `dst`.
pub(crate) fn copy(src: TensorView<'_>, dst: TensorView<'_>) -> Option<SemanticSite> {
    elementwise(&[src], dst, false, |x, _| x.into_iter().next())
}

/// `tensor_binary_f32` with `op`. Scalar ops read only `a`.
pub(crate) fn binary(op: u32, a: TensorView<'_>, b: TensorView<'_>, out: TensorView<'_>) -> Option<SemanticSite> {
    let (op, scalar) = match op {
        OP_ADD => (BinaryOp::Add, false),
        OP_SUB => (BinaryOp::Sub, false),
        OP_MUL => (BinaryOp::Mul, false),
        OP_DIV => (BinaryOp::Div, false),
        OP_MIN => (BinaryOp::Min, false),
        OP_MAX => (BinaryOp::Max, false),
        OP_ADD_SCALAR => (BinaryOp::Add, true),
        OP_SUB_SCALAR => (BinaryOp::Sub, true),
        OP_MUL_SCALAR => (BinaryOp::Mul, true),
        OP_DIV_SCALAR => (BinaryOp::Div, true),
        OP_MIN_SCALAR => (BinaryOp::Min, true),
        OP_MAX_SCALAR => (BinaryOp::Max, true),
        _ => return None,
    };
    let inputs: &[TensorView<'_>] = if scalar { &[a] } else { &[a, b] };
    elementwise(inputs, out, scalar, |x, s| {
        let mut x = x.into_iter();
        let lhs = x.next()?;
        let rhs = if scalar { s? } else { x.next()? };
        Some(Term::binary(op, lhs, rhs))
    })
}

/// `tensor_reduce_f32` with `op` over `axis` of `src` into `out`, which keeps that axis
/// with extent one.
pub(crate) fn reduce(op: u32, src: TensorView<'_>, axis: usize, out: TensorView<'_>) -> Option<SemanticSite> {
    let shape = dims(src)?;
    let len = *shape.get(axis)?;
    let mut kept = shape.clone();
    kept[axis] = 1;
    if dims(out)? != kept {
        return None;
    }
    let mut s = SiteBuilder::default();
    let at = indices(&mut s, shape.len());
    let k = s.region.index("k");
    let storage_src = storage(&mut s, src)?;
    let x = s.region.input("x", &shape, storage_src);
    let term = || {
        let index = at
            .iter()
            .enumerate()
            .map(|(a, &i)| Affine::from(if a == axis { k } else { i }));
        Term::read(x, index)
    };
    // The kernel folds from its own starting accumulator, which bounds the reduction.
    let body = match op {
        OP_SUM => Term::sum(k, len, term()),
        OP_MEAN => Term::binary(BinaryOp::Div, Term::sum(k, len, term()), Term::lit(len as f32)),
        OP_RMAX => Term::lit(REDUCE_MAX_START).max(Term::reduce(ReduceOp::Max, k, len, term())),
        OP_RMIN => Term::lit(-REDUCE_MAX_START).min(Term::reduce(ReduceOp::Min, k, len, term())),
        _ => return None,
    };
    let storage_out = storage(&mut s, out)?;
    s.region.output("out", &kept, &at, body, storage_out);
    s.finish()
}

/// A unique-write `tensor_scatter_f32` of one slice along `axis`: `src` has extent one
/// there, and every element reads the same `index` element. The slice then lands at
/// the position that element holds, an index parameter over `dst`'s extent.
pub(crate) fn scatter_slice(
    src: TensorView<'_>,
    index: TensorView<'_>,
    dst: TensorView<'_>,
    axis: usize,
) -> Option<SemanticSite> {
    let shape = dims(src)?;
    let target = dims(dst)?;
    let index_strides = index.layout().strides().to_vec();
    let fits = shape.len() == target.len()
        && index_strides.len() == shape.len()
        && shape.get(axis) == Some(&1)
        && shape.iter().zip(&target).all(|(s, t)| s <= t)
        && shape
            .iter()
            .zip(&index_strides)
            .all(|(&e, &stride)| e == 1 || stride == 0);
    if !fits || !matches!(index.dtype(), TensorDType::I32 | TensorDType::U32) {
        return None;
    }
    let mut s = SiteBuilder::default();
    let source = IndexSource {
        parcel: s.parcel(SiteParcel::of(index.buffer().whole())?),
        element: u32::try_from(index.storage_offset()).ok()?,
        // The kernel binds every index parcel as i32.
        element_type: ElementType::I32,
    };
    let position = s.index_param("position", 0..i64::from(target[axis]), source);
    let at = indices(&mut s, shape.len());
    let storage_src = storage(&mut s, src)?;
    let x = s.region.input("x", &shape, storage_src);
    let mut to = storage(&mut s, dst)?;
    to.offset = to.offset + position * to.strides[axis];
    s.region
        .output("out", &shape, &at, Term::read(x, at.iter().copied()), to);
    s.finish()
}
