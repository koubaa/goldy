//! Tensor recording facade: shape inference, allocation, Scheme nodes.

use super::dtype::TensorDType;
use super::kernels::{
    BatchedMatMulKernel, BinaryF32Kernel, CastF32I32Kernel, CastF32U32Kernel, CastI32F32Kernel, CastU32F32Kernel,
    CopyU32Kernel, GatherF32Kernel, ReduceF32Kernel, ScatterF32Kernel, TensorOpMeta, UnaryF32Kernel, OP_ABS, OP_ADD,
    OP_ADD_SCALAR, OP_COPY, OP_DIV, OP_DIV_SCALAR, OP_EXP, OP_FILL, OP_GATHER, OP_LOG, OP_MAX, OP_MAX_SCALAR, OP_MEAN,
    OP_MIN, OP_MIN_SCALAR, OP_MUL, OP_MUL_SCALAR, OP_NEG, OP_RECIP, OP_RMAX, OP_RMIN, OP_SCATTER_ADD, OP_SCATTER_MAX,
    OP_SCATTER_MIN, OP_SCATTER_SET, OP_SQRT, OP_SUB, OP_SUB_SCALAR, OP_SUM,
};
use super::layout::GoldyTensorLayout;
use super::shape::TensorShape;
use super::view::{broadcast_shapes, Tensor, TensorView};
use super::MAX_TENSOR_RANK;
use crate::error::GoldyError;
use crate::parcel::Buffer;
use crate::runtime::Runtime;
use crate::scheme::Scheme;
use crate::types::BufferKind;

/// Host scalar for fill / scalar-broadcast binary ops.
#[derive(Debug, Clone, Copy)]
pub enum TensorScalar {
    F32(f32),
    U32(u32),
    I32(i32),
}

impl TensorScalar {
    pub fn dtype(self) -> TensorDType {
        match self {
            Self::F32(_) => TensorDType::F32,
            Self::U32(_) => TensorDType::U32,
            Self::I32(_) => TensorDType::I32,
        }
    }

    pub fn bits(self) -> u32 {
        match self {
            Self::F32(v) => v.to_bits(),
            Self::U32(v) => v,
            Self::I32(v) => v as u32,
        }
    }

    pub fn as_f32(self) -> Result<f32, GoldyError> {
        match self {
            Self::F32(v) => Ok(v),
            _ => Err(GoldyError::Validation("tensor scalar: expected F32".into())),
        }
    }
}

/// Collision policy for [`TensorRecorder::scatter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterMode {
    /// Each destination index is written at most once. Collisions are a data race.
    UniqueWrite,
    /// Add colliding values.
    Add,
    /// Keep the minimum colliding value.
    Min,
    /// Keep the maximum colliding value.
    Max,
}

pub(crate) struct PreparedOps {
    pub unary: UnaryF32Kernel,
    pub binary: BinaryF32Kernel,
    pub copy: CopyU32Kernel,
    pub reduce: ReduceF32Kernel,
    pub gather: GatherF32Kernel,
    pub scatter: ScatterF32Kernel,
    pub batched: BatchedMatMulKernel,
    pub cast_f32_i32: CastF32I32Kernel,
    pub cast_f32_u32: CastF32U32Kernel,
    pub cast_i32_f32: CastI32F32Kernel,
    pub cast_u32_f32: CastU32F32Kernel,
}

impl PreparedOps {
    fn prepare(runtime: &Runtime) -> Result<Self, GoldyError> {
        Ok(Self {
            unary: UnaryF32Kernel::prepare(runtime).map_err(GoldyError::from)?,
            binary: BinaryF32Kernel::prepare(runtime).map_err(GoldyError::from)?,
            copy: CopyU32Kernel::prepare(runtime).map_err(GoldyError::from)?,
            reduce: ReduceF32Kernel::prepare(runtime).map_err(GoldyError::from)?,
            gather: GatherF32Kernel::prepare(runtime).map_err(GoldyError::from)?,
            scatter: ScatterF32Kernel::prepare(runtime).map_err(GoldyError::from)?,
            batched: BatchedMatMulKernel::prepare(runtime).map_err(GoldyError::from)?,
            cast_f32_i32: CastF32I32Kernel::prepare(runtime).map_err(GoldyError::from)?,
            cast_f32_u32: CastF32U32Kernel::prepare(runtime).map_err(GoldyError::from)?,
            cast_i32_f32: CastI32F32Kernel::prepare(runtime).map_err(GoldyError::from)?,
            cast_u32_f32: CastU32F32Kernel::prepare(runtime).map_err(GoldyError::from)?,
        })
    }
}

/// Long-lived tensor session: prepared pipelines plus layout-parcel keepalive.
///
/// Keep this alive for as long as schemes that recorded through it still exist.
pub struct TensorContext {
    pub(crate) runtime: Runtime,
    pub(crate) ops: PreparedOps,
    keepalive: Vec<Buffer>,
    labels: Vec<String>,
}

impl TensorContext {
    pub fn new(runtime: &Runtime) -> Result<Self, GoldyError> {
        Ok(Self {
            runtime: runtime.clone(),
            ops: PreparedOps::prepare(runtime)?,
            keepalive: Vec::new(),
            labels: Vec::new(),
        })
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Borrow `scheme` and record tensor ops into it.
    pub fn recorder<'a>(&'a mut self, scheme: &'a mut Scheme) -> TensorRecorder<'a> {
        TensorRecorder { ctx: self, scheme }
    }

    pub(crate) fn intern_label(&mut self, label: impl Into<String>) -> &'static str {
        self.labels.push(label.into());
        let s = self.labels.last().unwrap().as_str();
        // SAFETY: labels live as long as TensorContext, which outlives recorded schemes.
        unsafe { std::mem::transmute::<&str, &'static str>(s) }
    }

    pub(crate) fn push_meta(&mut self, meta: TensorOpMeta) -> Result<usize, GoldyError> {
        let buf = self
            .runtime
            .acquire_buffer_with_data(&[meta], BufferKind::Scattered)
            .map_err(GoldyError::from)?;
        self.keepalive.push(buf);
        Ok(self.keepalive.len() - 1)
    }

    pub(crate) fn meta(&self, index: usize) -> &Buffer {
        &self.keepalive[index]
    }
}

/// Facade that borrows a [`TensorContext`] and a mutable [`Scheme`].
pub struct TensorRecorder<'a> {
    pub(crate) ctx: &'a mut TensorContext,
    pub(crate) scheme: &'a mut Scheme,
}

impl<'a> TensorRecorder<'a> {
    pub fn scheme(&mut self) -> &mut Scheme {
        self.scheme
    }

    pub fn zeros(&mut self, shape: TensorShape, dtype: TensorDType) -> Result<Tensor, GoldyError> {
        Tensor::zeros(&self.ctx.runtime, shape, dtype)
    }

    pub fn fill(&mut self, label: &str, out: TensorView<'_>, value: TensorScalar) -> Result<(), GoldyError> {
        out.layout().require_writeable("fill")?;
        if value.dtype() != out.dtype() {
            return Err(GoldyError::Validation("tensor fill: scalar dtype mismatch".into()));
        }
        let meta = encode_meta(OP_FILL, 0, value.bits(), 0, None, None, Some(out))?;
        let label = self.ctx.intern_label(label);
        let idx = self.ctx.push_meta(meta)?;
        let n = out.numel_u32().max(1);
        if out.dtype() == TensorDType::F32 {
            self.ctx
                .ops
                .unary
                .record(
                    self.scheme,
                    label,
                    out.buffer(),
                    out.buffer(),
                    self.ctx.meta(idx),
                    value.as_f32()?,
                )
                .over_1d(n);
        } else {
            self.ctx
                .ops
                .copy
                .record(
                    self.scheme,
                    label,
                    out.buffer(),
                    out.buffer(),
                    self.ctx.meta(idx),
                    value.bits(),
                )
                .over_1d(n);
        }
        Ok(())
    }

    pub fn copy(&mut self, label: &str, src: TensorView<'_>, dst: TensorView<'_>) -> Result<(), GoldyError> {
        dst.layout().require_writeable("copy")?;
        if src.dtype() != dst.dtype() {
            return Err(GoldyError::Validation("tensor copy: dtype mismatch".into()));
        }
        let src = src.broadcast_to(dst.shape())?;
        let meta = encode_meta(OP_COPY, 0, 0, 0, Some(src), None, Some(dst))?;
        let label = self.ctx.intern_label(label);
        let idx = self.ctx.push_meta(meta)?;
        self.ctx
            .ops
            .copy
            .record(self.scheme, label, src.buffer(), dst.buffer(), self.ctx.meta(idx), 0)
            .over_1d(dst.numel_u32().max(1));
        Ok(())
    }

    pub fn contiguous(&mut self, label: &str, src: TensorView<'_>) -> Result<Tensor, GoldyError> {
        let out = Tensor::zeros(&self.ctx.runtime, src.shape(), src.dtype())?;
        self.copy(label, src, out.view())?;
        Ok(out)
    }

    pub fn cast(&mut self, label: &str, src: TensorView<'_>, dtype: TensorDType) -> Result<Tensor, GoldyError> {
        let out = Tensor::zeros(&self.ctx.runtime, src.shape(), dtype)?;
        self.cast_into(label, src, out.view())?;
        Ok(out)
    }

    pub fn cast_into(&mut self, label: &str, src: TensorView<'_>, dst: TensorView<'_>) -> Result<(), GoldyError> {
        dst.layout().require_writeable("cast")?;
        if src.numel()? != dst.numel()? {
            return Err(GoldyError::Validation("tensor cast: numel mismatch".into()));
        }
        if src.dtype() == dst.dtype() {
            return self.copy(label, src, dst);
        }
        let meta = encode_meta(OP_COPY, 0, 0, 0, Some(src), None, Some(dst))?;
        let label = self.ctx.intern_label(label);
        let idx = self.ctx.push_meta(meta)?;
        let n = dst.numel_u32().max(1);
        match (src.dtype(), dst.dtype()) {
            (TensorDType::F32, TensorDType::I32) => {
                self.ctx
                    .ops
                    .cast_f32_i32
                    .record(self.scheme, label, src.buffer(), dst.buffer(), self.ctx.meta(idx))
                    .over_1d(n);
            }
            (TensorDType::F32, TensorDType::U32) => {
                self.ctx
                    .ops
                    .cast_f32_u32
                    .record(self.scheme, label, src.buffer(), dst.buffer(), self.ctx.meta(idx))
                    .over_1d(n);
            }
            (TensorDType::I32, TensorDType::F32) => {
                self.ctx
                    .ops
                    .cast_i32_f32
                    .record(self.scheme, label, src.buffer(), dst.buffer(), self.ctx.meta(idx))
                    .over_1d(n);
            }
            (TensorDType::U32, TensorDType::F32) => {
                self.ctx
                    .ops
                    .cast_u32_f32
                    .record(self.scheme, label, src.buffer(), dst.buffer(), self.ctx.meta(idx))
                    .over_1d(n);
            }
            (TensorDType::I32, TensorDType::U32) | (TensorDType::U32, TensorDType::I32) => {
                self.ctx
                    .ops
                    .copy
                    .record(self.scheme, label, src.buffer(), dst.buffer(), self.ctx.meta(idx), 0)
                    .over_1d(n);
            }
            _ => {
                return Err(GoldyError::Validation(format!(
                    "tensor cast: {} -> {} is not supported",
                    src.dtype().name(),
                    dst.dtype().name()
                )))
            }
        }
        Ok(())
    }

    pub fn neg(&mut self, label: &str, src: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.unary(label, src, OP_NEG)
    }
    pub fn abs(&mut self, label: &str, src: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.unary(label, src, OP_ABS)
    }
    pub fn exp(&mut self, label: &str, src: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.unary(label, src, OP_EXP)
    }
    pub fn log(&mut self, label: &str, src: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.unary(label, src, OP_LOG)
    }
    pub fn sqrt(&mut self, label: &str, src: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.unary(label, src, OP_SQRT)
    }
    pub fn reciprocal(&mut self, label: &str, src: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.unary(label, src, OP_RECIP)
    }

    pub fn add(&mut self, label: &str, a: TensorView<'_>, b: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.binary(label, a, b, OP_ADD)
    }
    pub fn sub(&mut self, label: &str, a: TensorView<'_>, b: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.binary(label, a, b, OP_SUB)
    }
    pub fn mul(&mut self, label: &str, a: TensorView<'_>, b: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.binary(label, a, b, OP_MUL)
    }
    pub fn div(&mut self, label: &str, a: TensorView<'_>, b: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.binary(label, a, b, OP_DIV)
    }
    pub fn min(&mut self, label: &str, a: TensorView<'_>, b: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.binary(label, a, b, OP_MIN)
    }
    pub fn max(&mut self, label: &str, a: TensorView<'_>, b: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.binary(label, a, b, OP_MAX)
    }

    pub fn add_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_into(label, a, b, out, OP_ADD)
    }

    pub fn add_scalar(&mut self, label: &str, a: TensorView<'_>, scalar: f32) -> Result<Tensor, GoldyError> {
        self.binary_scalar(label, a, scalar, OP_ADD_SCALAR)
    }
    pub fn sub_scalar(&mut self, label: &str, a: TensorView<'_>, scalar: f32) -> Result<Tensor, GoldyError> {
        self.binary_scalar(label, a, scalar, OP_SUB_SCALAR)
    }
    pub fn mul_scalar(&mut self, label: &str, a: TensorView<'_>, scalar: f32) -> Result<Tensor, GoldyError> {
        self.binary_scalar(label, a, scalar, OP_MUL_SCALAR)
    }
    pub fn div_scalar(&mut self, label: &str, a: TensorView<'_>, scalar: f32) -> Result<Tensor, GoldyError> {
        self.binary_scalar(label, a, scalar, OP_DIV_SCALAR)
    }
    pub fn min_scalar(&mut self, label: &str, a: TensorView<'_>, scalar: f32) -> Result<Tensor, GoldyError> {
        self.binary_scalar(label, a, scalar, OP_MIN_SCALAR)
    }
    pub fn max_scalar(&mut self, label: &str, a: TensorView<'_>, scalar: f32) -> Result<Tensor, GoldyError> {
        self.binary_scalar(label, a, scalar, OP_MAX_SCALAR)
    }

    pub fn sum(&mut self, label: &str, src: TensorView<'_>, axis: usize) -> Result<Tensor, GoldyError> {
        self.reduce(label, src, axis, OP_SUM, false)
    }
    pub fn max_reduce(&mut self, label: &str, src: TensorView<'_>, axis: usize) -> Result<Tensor, GoldyError> {
        self.reduce(label, src, axis, OP_RMAX, false)
    }
    pub fn min_reduce(&mut self, label: &str, src: TensorView<'_>, axis: usize) -> Result<Tensor, GoldyError> {
        self.reduce(label, src, axis, OP_RMIN, false)
    }
    pub fn mean(&mut self, label: &str, src: TensorView<'_>, axis: usize) -> Result<Tensor, GoldyError> {
        self.reduce(label, src, axis, OP_MEAN, false)
    }

    /// Softmax along `axis` as max-subtract-exp-sum-div. Not a fused NN operator.
    pub fn softmax(&mut self, label: &str, src: TensorView<'_>, axis: usize) -> Result<Tensor, GoldyError> {
        src.dtype().require_f32("softmax")?;
        let max = self.reduce(&format!("{label}_max"), src, axis, OP_RMAX, true)?;
        let shifted = self.sub(&format!("{label}_sub"), src, max.view())?;
        let exp = self.exp(&format!("{label}_exp"), shifted.view())?;
        let sum = self.reduce(&format!("{label}_sum"), exp.view(), axis, OP_SUM, true)?;
        self.div(&format!("{label}_div"), exp.view(), sum.view())
    }

    pub fn gather(
        &mut self,
        label: &str,
        src: TensorView<'_>,
        index: TensorView<'_>,
        axis: usize,
    ) -> Result<Tensor, GoldyError> {
        src.dtype().require_f32("gather")?;
        if index.dtype() != TensorDType::I32 && index.dtype() != TensorDType::U32 {
            return Err(GoldyError::Validation(
                "tensor gather: index dtype must be I32 or U32".into(),
            ));
        }
        src.shape().dim(axis)?;
        let out = Tensor::zeros(&self.ctx.runtime, index.shape(), src.dtype())?;
        let meta = encode_meta(OP_GATHER, axis as u32, 0, 0, Some(src), Some(index), Some(out.view()))?;
        let label = self.ctx.intern_label(label);
        let idx = self.ctx.push_meta(meta)?;
        self.ctx
            .ops
            .gather
            .record(
                self.scheme,
                label,
                src.buffer(),
                index.buffer(),
                out.buffer(),
                self.ctx.meta(idx),
            )
            .over_1d(out.view().numel_u32().max(1));
        Ok(out)
    }

    pub fn scatter(
        &mut self,
        label: &str,
        src: TensorView<'_>,
        index: TensorView<'_>,
        dst: TensorView<'_>,
        axis: usize,
        mode: ScatterMode,
    ) -> Result<(), GoldyError> {
        src.dtype().require_f32("scatter")?;
        dst.layout().require_writeable("scatter")?;
        dst.shape().dim(axis)?;
        let op = match mode {
            ScatterMode::UniqueWrite => OP_SCATTER_SET,
            ScatterMode::Add => OP_SCATTER_ADD,
            ScatterMode::Min => OP_SCATTER_MIN,
            ScatterMode::Max => OP_SCATTER_MAX,
        };
        let meta = encode_meta(op, axis as u32, 0, 0, Some(src), Some(index), Some(dst))?;
        let label = self.ctx.intern_label(label);
        let idx = self.ctx.push_meta(meta)?;
        self.ctx
            .ops
            .scatter
            .record(
                self.scheme,
                label,
                src.buffer(),
                index.buffer(),
                dst.buffer(),
                self.ctx.meta(idx),
            )
            .groups([1, 1, 1]);
        Ok(())
    }

    fn unary(&mut self, label: &str, src: TensorView<'_>, op: u32) -> Result<Tensor, GoldyError> {
        src.dtype().require_f32("unary")?;
        let out = Tensor::zeros(&self.ctx.runtime, src.shape(), TensorDType::F32)?;
        self.unary_into(label, src, out.view(), op)?;
        Ok(out)
    }

    fn unary_into(&mut self, label: &str, src: TensorView<'_>, dst: TensorView<'_>, op: u32) -> Result<(), GoldyError> {
        dst.layout().require_writeable("unary")?;
        src.dtype().require_f32("unary")?;
        dst.dtype().require_f32("unary")?;
        let meta = encode_meta(op, 0, 0, 0, Some(src), None, Some(dst))?;
        let label = self.ctx.intern_label(label);
        let idx = self.ctx.push_meta(meta)?;
        self.ctx
            .ops
            .unary
            .record(self.scheme, label, src.buffer(), dst.buffer(), self.ctx.meta(idx), 0.0)
            .over_1d(dst.numel_u32().max(1));
        Ok(())
    }

    fn binary(&mut self, label: &str, a: TensorView<'_>, b: TensorView<'_>, op: u32) -> Result<Tensor, GoldyError> {
        a.dtype().require_f32("binary")?;
        b.dtype().require_f32("binary")?;
        let shape = broadcast_shapes(a.shape(), b.shape())?;
        let out = Tensor::zeros(&self.ctx.runtime, shape, TensorDType::F32)?;
        self.binary_into(label, a, b, out.view(), op)?;
        Ok(out)
    }

    fn binary_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
        op: u32,
    ) -> Result<(), GoldyError> {
        out.layout().require_writeable("binary")?;
        a.dtype().require_f32("binary")?;
        b.dtype().require_f32("binary")?;
        out.dtype().require_f32("binary")?;
        let a = a.broadcast_to(out.shape())?;
        let b = b.broadcast_to(out.shape())?;
        let meta = encode_meta(op, 0, 0, 0, Some(a), Some(b), Some(out))?;
        let label = self.ctx.intern_label(label);
        let idx = self.ctx.push_meta(meta)?;
        self.ctx
            .ops
            .binary
            .record(
                self.scheme,
                label,
                a.buffer(),
                b.buffer(),
                out.buffer(),
                self.ctx.meta(idx),
                0.0,
            )
            .over_1d(out.numel_u32().max(1));
        Ok(())
    }

    fn binary_scalar(&mut self, label: &str, a: TensorView<'_>, scalar: f32, op: u32) -> Result<Tensor, GoldyError> {
        a.dtype().require_f32("binary_scalar")?;
        let out = Tensor::zeros(&self.ctx.runtime, a.shape(), TensorDType::F32)?;
        let meta = encode_meta(op, 0, scalar.to_bits(), 0, Some(a), None, Some(out.view()))?;
        let label = self.ctx.intern_label(label);
        let idx = self.ctx.push_meta(meta)?;
        self.ctx
            .ops
            .binary
            .record(
                self.scheme,
                label,
                a.buffer(),
                a.buffer(),
                out.buffer(),
                self.ctx.meta(idx),
                scalar,
            )
            .over_1d(out.view().numel_u32().max(1));
        Ok(out)
    }

    fn reduce(
        &mut self,
        label: &str,
        src: TensorView<'_>,
        axis: usize,
        op: u32,
        keepdim: bool,
    ) -> Result<Tensor, GoldyError> {
        src.dtype().require_f32("reduce")?;
        let reduce_len = src.shape().dim(axis)?;
        let src_shape = src.shape();
        let mut keep_dims = src_shape.dims().to_vec();
        keep_dims[axis] = 1;
        let keep_shape = TensorShape::from_dims(&keep_dims)?;
        let out_shape = if keepdim {
            keep_shape
        } else {
            src.shape().squeeze_axis(axis)?
        };
        let out = Tensor::zeros(&self.ctx.runtime, out_shape, TensorDType::F32)?;
        let kernel_layout = super::layout::TensorLayout::packed(out.dtype(), keep_shape, 0)?;
        let kernel_view = TensorView::new(out.buffer(), kernel_layout)?;
        let meta = encode_meta(op, axis as u32, 0, reduce_len, Some(src), None, Some(kernel_view))?;
        let label = self.ctx.intern_label(label);
        let idx = self.ctx.push_meta(meta)?;
        self.ctx
            .ops
            .reduce
            .record(self.scheme, label, src.buffer(), out.buffer(), self.ctx.meta(idx))
            .over_1d(kernel_view.numel_u32().max(1));
        Ok(out)
    }

    pub(crate) fn intern_label(&mut self, label: impl Into<String>) -> &'static str {
        self.ctx.intern_label(label)
    }
}

pub(crate) fn encode_meta(
    op: u32,
    axis: u32,
    scalar_bits: u32,
    reduce_len: u32,
    a: Option<TensorView<'_>>,
    b: Option<TensorView<'_>>,
    o: Option<TensorView<'_>>,
) -> Result<TensorOpMeta, GoldyError> {
    let a = a
        .map(|v| v.layout().gpu_coords())
        .transpose()?
        .unwrap_or(empty_coords());
    let b = b
        .map(|v| v.layout().gpu_coords())
        .transpose()?
        .unwrap_or(empty_coords());
    let o = o
        .map(|v| v.layout().gpu_coords())
        .transpose()?
        .unwrap_or(empty_coords());
    Ok(pack_meta(op, axis, scalar_bits, reduce_len, a, b, o))
}

fn empty_coords() -> GoldyTensorLayout {
    GoldyTensorLayout {
        offset: 0,
        rank: 0,
        numel: 0,
        shape: [1; MAX_TENSOR_RANK],
        stride: [0; MAX_TENSOR_RANK],
        pad: 0,
    }
}

fn pack_meta(
    op: u32,
    axis: u32,
    scalar_bits: u32,
    reduce_len: u32,
    a: GoldyTensorLayout,
    b: GoldyTensorLayout,
    o: GoldyTensorLayout,
) -> TensorOpMeta {
    TensorOpMeta {
        op,
        axis,
        scalar_bits,
        reduce_len,
        a_off: a.offset,
        a_numel: a.numel,
        a_s0: a.stride[0],
        a_s1: a.stride[1],
        a_s2: a.stride[2],
        a_s3: a.stride[3],
        a_d0: a.shape[0],
        a_d1: a.shape[1],
        a_d2: a.shape[2],
        a_d3: a.shape[3],
        b_off: b.offset,
        b_numel: b.numel,
        b_s0: b.stride[0],
        b_s1: b.stride[1],
        b_s2: b.stride[2],
        b_s3: b.stride[3],
        b_d0: b.shape[0],
        b_d1: b.shape[1],
        b_d2: b.shape[2],
        b_d3: b.shape[3],
        o_off: o.offset,
        o_numel: o.numel,
        o_s0: o.stride[0],
        o_s1: o.stride[1],
        o_s2: o.stride[2],
        o_s3: o.stride[3],
        o_d0: o.shape[0],
        o_d1: o.shape[1],
        o_d2: o.shape[2],
        o_d3: o.shape[3],
    }
}
