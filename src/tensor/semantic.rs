//! Tensor-algebra meaning of the tensor recorder's kernels (see `semantic_fusion.rs`).
//!
//! Each function states what one recorded dispatch of `kernels.rs` stores, as a typed
//! operation over its operands' views. It returns `None` for the cases it does not
//! describe; those dispatches stay opaque to semantic fusion.

use super::dtype::TensorDType;
use super::kernels::{
    OP_ABS, OP_ADD, OP_ADD_SCALAR, OP_DIV, OP_DIV_SCALAR, OP_EXP, OP_LOG, OP_MAX, OP_MAX_SCALAR, OP_MEAN, OP_MIN,
    OP_MIN_SCALAR, OP_MUL, OP_MUL_SCALAR, OP_NEG, OP_RECIP, OP_RMAX, OP_RMIN, OP_SIGMOID, OP_SQRT, OP_SUB,
    OP_SUB_SCALAR, OP_SUM, OP_TANH,
};
use super::view::TensorView;
use crate::semantic_fusion::{SemanticSite, SiteBuilder};
use goldy_shader_ir::algebra::{
    Affine, BinaryOp, CmpOp, IndexSource, Map, OpKind, Operand, ReduceOp, ReduceOrder, Reduction, Storage, Sym, Term,
    UnaryOp,
};
use goldy_shader_ir::ElementType;

/// The accumulator `tensor_reduce_f32` starts a max reduction from; a min starts from its negation.
const REDUCE_MAX_START: f32 = -3.402823e38;

fn storage(s: &mut SiteBuilder, view: TensorView<'_>) -> Option<Storage> {
    let parcel = s.parcel(view.site_parcel()?);
    let offset = i64::try_from(view.storage_offset()).ok()?;
    Some(Storage::strided(parcel, offset, view.layout().strides()))
}

fn dims(view: TensorView<'_>) -> Option<Vec<u32>> {
    let dims = view.shape().dims().to_vec();
    (view.dtype() == TensorDType::F32 && !dims.contains(&0)).then_some(dims)
}

fn operand(s: &mut SiteBuilder, name: &str, view: TensorView<'_>) -> Option<Operand> {
    let shape = dims(view)?;
    Some(Operand::new(name, &shape, storage(s, view)?))
}

fn index_source(s: &mut SiteBuilder, view: TensorView<'_>, add: i64, modulus: Option<u32>) -> Option<IndexSource> {
    (view.dtype() == TensorDType::U32 && view.numel_u32() == 1).then_some(IndexSource {
        parcel: s.parcel(view.site_parcel()?),
        element: Affine::constant(i64::try_from(view.storage_offset()).ok()?),
        element_type: ElementType::U32,
        add,
        modulus,
        zero_if_outside: false,
    })
}

pub(crate) fn cyclic_store(
    src: TensorView<'_>,
    state: TensorView<'_>,
    position: TensorView<'_>,
) -> Option<SemanticSite> {
    let state_shape = state.shape();
    let (&width, rows) = state_shape.dims().split_last()?;
    if src.shape().dims() != rows || width == 0 {
        return None;
    }
    let mut s = SiteBuilder::default();
    let source = index_source(&mut s, position, 0, Some(width))?;
    let slot = s.index_param("slot", 0..i64::from(width), source);
    let input = operand(&mut s, "x", src)?;
    let mut target = storage(&mut s, state)?;
    let along = target.strides.pop()?;
    target.offset = target.offset + Affine::sym(Sym::Param(slot)) * along;
    s.finish(OpKind::Map(Map {
        shape: rows.to_vec(),
        inputs: vec![input],
        outputs: vec![(Operand::new("state", rows, target), Term::arg(0))],
    }))
}

/// `tensor_store_at_f32`: `state[rows.., position] = src` for a position inside the
/// history. Past it the slot parameter clamps into the history, and the store writes
/// back what the clamped slot holds, which leaves the history as the kernel does.
pub(crate) fn bounded_store(
    src: TensorView<'_>,
    state: TensorView<'_>,
    position: TensorView<'_>,
) -> Option<SemanticSite> {
    let state_shape = state.shape();
    let (&width, rows) = state_shape.dims().split_last()?;
    if src.shape().dims() != rows || width == 0 {
        return None;
    }
    let mut s = SiteBuilder::default();
    let raw_source = index_source(&mut s, position, 0, None)?;
    let raw = s.index_param("position", 0..i64::from(i32::MAX) + 1, raw_source);
    let slot_source = index_source(&mut s, position, 0, None)?;
    let slot = s.index_param("slot", 0..i64::from(width), slot_source);
    let input = operand(&mut s, "x", src)?;
    let mut target = storage(&mut s, state)?;
    let along = target.strides.pop()?;
    target.offset = target.offset + Affine::sym(Sym::Param(slot)) * along;
    let inside = Term::select(
        Affine::sym(Sym::Param(raw)),
        CmpOp::Lt,
        i64::from(width),
        Term::arg(0),
        Term::arg(1),
    );
    s.finish(OpKind::Map(Map {
        shape: rows.to_vec(),
        inputs: vec![input, Operand::new("held", rows, target.clone())],
        outputs: vec![(Operand::new("state", rows, target), inside)],
    }))
}

pub(crate) fn cyclic_dot(
    state: TensorView<'_>,
    weight: TensorView<'_>,
    position: TensorView<'_>,
    out: TensorView<'_>,
) -> Option<SemanticSite> {
    let state_shape = state.shape();
    let (&width, rows) = state_shape.dims().split_last()?;
    if width == 0 || weight.shape().dims() != state_shape.dims() || out.shape().dims() != rows {
        return None;
    }
    let mut s = SiteBuilder::default();
    let raw_source = index_source(&mut s, position, 0, None)?;
    let raw = s.index_param("position", 0..i64::from(i32::MAX), raw_source);
    let mut state_storage = storage(&mut s, state)?;
    let mut weight_storage = storage(&mut s, weight)?;
    let state_along = state_storage.strides.pop()?;
    let weight_along = weight_storage.strides.pop()?;
    let mut inputs = Vec::with_capacity(width as usize * 2);
    let mut acc = Term::lit(0.0);
    for lag in 0..width {
        let slot_source = index_source(&mut s, position, -i64::from(lag), Some(width))?;
        let slot = s.index_param(&format!("slot_{lag}"), 0..i64::from(width), slot_source);
        let tap = width - 1 - lag;
        let state_at = state_storage.offset.clone() + Affine::sym(Sym::Param(slot)) * state_along;
        let weight_at = weight_storage.offset.clone() + i64::from(tap) * weight_along;
        let weight_arg = inputs.len();
        inputs.push(Operand::new(
            &format!("weight_{tap}"),
            rows,
            Storage::strided(weight_storage.parcel, weight_at, &weight_storage.strides),
        ));
        let state_arg = inputs.len();
        inputs.push(Operand::new(
            &format!("state_{lag}"),
            rows,
            Storage::strided(state_storage.parcel, state_at, &state_storage.strides),
        ));
        let value = Term::select(
            i64::from(lag),
            CmpOp::Le,
            Affine::sym(Sym::Param(raw)),
            Term::arg(state_arg),
            Term::lit(0.0),
        );
        acc = acc + Term::arg(weight_arg) * value;
    }
    let output = operand(&mut s, "out", out)?;
    s.finish(OpKind::Map(Map {
        shape: rows.to_vec(),
        inputs,
        outputs: vec![(output, acc)],
    }))
}

/// `out = body(inputs..)` element-wise over `out`'s shape; `inputs` already have that
/// shape, and argument `n` of the body is `inputs[n]`.
///
/// With `scalar`, the body also gets the f32 in user slot 0.
fn elementwise(
    inputs: &[TensorView<'_>],
    out: TensorView<'_>,
    scalar: bool,
    body: impl FnOnce(Option<Term>) -> Option<Term>,
) -> Option<SemanticSite> {
    let shape = dims(out)?;
    let mut s = SiteBuilder::default();
    let inputs = inputs
        .iter()
        .enumerate()
        .map(|(n, &view)| operand(&mut s, &format!("x{n}"), view))
        .collect::<Option<Vec<_>>>()?;
    let scalar = scalar.then(|| Term::scalar(s.scalar("scalar", 0)));
    let body = body(scalar)?;
    let out = operand(&mut s, "out", out)?;
    s.finish(OpKind::Map(Map {
        shape,
        inputs,
        outputs: vec![(out, body)],
    }))
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
        OP_TANH => UnaryOp::Tanh,
        OP_SIGMOID => UnaryOp::Sigmoid,
        _ => return None,
    };
    elementwise(&[src], dst, false, |_| Some(Term::unary(op, Term::arg(0))))
}

/// `tensor_unary_f32` filling `out` with its scalar.
pub(crate) fn fill(out: TensorView<'_>) -> Option<SemanticSite> {
    elementwise(&[], out, true, |scalar| scalar)
}

/// An f32 `tensor_copy_u32` from `src` into `dst`.
pub(crate) fn copy(src: TensorView<'_>, dst: TensorView<'_>) -> Option<SemanticSite> {
    elementwise(&[src], dst, false, |_| Some(Term::arg(0)))
}

/// A `tensor_cast_{i32,u32}_f32` of one integer: the f32 nearest it, as an index value.
///
/// An index parameter holds an `int`, so a u32 source is described for values below
/// 2^31 only, the domain the cyclic sites assume of a position; an i32 source is
/// described over its whole range.
pub(crate) fn integer_value(src: TensorView<'_>, dst: TensorView<'_>) -> Option<SemanticSite> {
    let (element_type, range) = match src.dtype() {
        TensorDType::U32 => (ElementType::U32, 0..i64::from(i32::MAX) + 1),
        TensorDType::I32 => (ElementType::I32, i64::from(i32::MIN)..i64::from(i32::MAX) + 1),
        _ => return None,
    };
    if src.numel_u32() != 1 {
        return None;
    }
    let shape = dims(dst)?;
    let mut s = SiteBuilder::default();
    let source = IndexSource {
        parcel: s.parcel(src.site_parcel()?),
        element: Affine::constant(i64::try_from(src.storage_offset()).ok()?),
        element_type,
        add: 0,
        modulus: None,
        zero_if_outside: false,
    };
    let value = s.index_param("value", range, source);
    let out = operand(&mut s, "out", dst)?;
    s.finish(OpKind::Map(Map {
        shape,
        inputs: Vec::new(),
        outputs: vec![(out, Term::index_value(Affine::sym(Sym::Param(value))))],
    }))
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
    elementwise(inputs, out, scalar, |s| {
        let rhs = if scalar { s? } else { Term::arg(1) };
        Some(Term::binary(op, Term::arg(0), rhs))
    })
}

/// `tensor_reduce_f32` with `op` over `axis` of `src` into `out`, which keeps that axis
/// with extent one.
pub(crate) fn reduce(op: u32, src: TensorView<'_>, axis: usize, out: TensorView<'_>) -> Option<SemanticSite> {
    let len = *dims(src)?.get(axis)?;
    // The kernel folds from its own starting accumulator, which bounds the reduction.
    // It divides a mean by the length it reads at run time, which the device cannot
    // turn into a multiply by the reciprocal; rounding keeps the constant a divisor.
    let (op, finish) = match op {
        OP_SUM => (ReduceOp::Sum, Term::arg(0)),
        OP_MEAN => (
            ReduceOp::Sum,
            Term::unary(UnaryOp::Round, Term::arg(0) / Term::lit(len as f32)),
        ),
        OP_RMAX => (ReduceOp::Max, Term::lit(REDUCE_MAX_START).max(Term::arg(0))),
        OP_RMIN => (ReduceOp::Min, Term::lit(-REDUCE_MAX_START).min(Term::arg(0))),
        _ => return None,
    };
    let mut s = SiteBuilder::default();
    let input = operand(&mut s, "x", src)?;
    let out = operand(&mut s, "out", out)?;
    s.finish(OpKind::Reduction(Reduction {
        op,
        order: ReduceOrder::Sequential,
        axis,
        input,
        out,
        finish,
    }))
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
        parcel: s.parcel(index.site_parcel()?),
        element: Affine::constant(i64::try_from(index.storage_offset()).ok()?),
        // The kernel binds every index parcel as i32.
        element_type: ElementType::I32,
        add: 0,
        modulus: None,
        zero_if_outside: false,
    };
    let position = s.index_param("position", 0..i64::from(target[axis]), source);
    let input = operand(&mut s, "x", src)?;
    let mut to = storage(&mut s, dst)?;
    to.offset = to.offset + position * to.strides[axis];
    s.finish(OpKind::Map(Map {
        shape: shape.clone(),
        inputs: vec![input],
        outputs: vec![(Operand::new("out", &shape, to), Term::arg(0))],
    }))
}
