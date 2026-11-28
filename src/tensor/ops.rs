//! Tensor recording facade: shape inference, allocation, Scheme nodes.

use super::dtype::TensorDType;
use super::kernels::{
    BatchedMatMulKernel, BinaryF32Kernel, CastF32I32Kernel, CastF32U32Kernel, CastI32F32Kernel, CastU32F32Kernel,
    CopyU32Kernel, CyclicDotF32Kernel, CyclicStoreF32Kernel, GatherF32Kernel, ReduceF32Kernel, ScatterF32Kernel,
    UnaryF32Kernel, OP_ABS, OP_ADD, OP_ADD_SCALAR, OP_COPY, OP_DIV, OP_DIV_SCALAR, OP_EXP, OP_FILL, OP_LOG, OP_MAX,
    OP_MAX_SCALAR, OP_MEAN, OP_MIN, OP_MIN_SCALAR, OP_MUL, OP_MUL_SCALAR, OP_NEG, OP_RECIP, OP_RMAX, OP_RMIN,
    OP_SCATTER_ADD, OP_SCATTER_MAX, OP_SCATTER_MIN, OP_SCATTER_SET, OP_SIGMOID, OP_SQRT, OP_SUB, OP_SUB_SCALAR, OP_SUM,
    OP_TANH,
};
use super::semantic;
use super::shape::TensorShape;
use super::view::{broadcast_shapes, Tensor, TensorView};
use crate::error::GoldyError;
use crate::runtime::Runtime;
use crate::scheme::Scheme;

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
    pub cyclic_store: CyclicStoreF32Kernel,
    pub cyclic_dot: CyclicDotF32Kernel,
    pub batched: BatchedMatMulKernel,
    pub cast_f32_i32: CastF32I32Kernel,
    pub cast_f32_u32: CastF32U32Kernel,
    pub cast_i32_f32: CastI32F32Kernel,
    pub cast_u32_f32: CastU32F32Kernel,
}

impl PreparedOps {
    fn prepare(runtime: &Runtime, policy: crate::ArithmeticPolicy) -> Result<Self, GoldyError> {
        Ok(Self {
            unary: UnaryF32Kernel::prepare_for_policy(runtime, policy)?,
            binary: BinaryF32Kernel::prepare_for_policy(runtime, policy)?,
            copy: CopyU32Kernel::prepare_for_policy(runtime, policy)?,
            reduce: ReduceF32Kernel::prepare_for_policy(runtime, policy)?,
            gather: GatherF32Kernel::prepare_for_policy(runtime, policy)?,
            scatter: ScatterF32Kernel::prepare_for_policy(runtime, policy)?,
            cyclic_store: CyclicStoreF32Kernel::prepare_for_policy(runtime, policy)?,
            cyclic_dot: CyclicDotF32Kernel::prepare_for_policy(runtime, policy)?,
            batched: BatchedMatMulKernel::prepare_for_policy(runtime, policy)?,
            cast_f32_i32: CastF32I32Kernel::prepare_for_policy(runtime, policy)?,
            cast_f32_u32: CastF32U32Kernel::prepare_for_policy(runtime, policy)?,
            cast_i32_f32: CastI32F32Kernel::prepare_for_policy(runtime, policy)?,
            cast_u32_f32: CastU32F32Kernel::prepare_for_policy(runtime, policy)?,
        })
    }
}

/// Prepared portable tensor kernels for one [`Runtime`].
///
/// Compile once, then [`Self::recorder`] onto any scheme on that runtime. Layout and
/// op-meta parcels intern onto the scheme at record time.
pub struct TensorKernels {
    pub(crate) runtime: Runtime,
    pub(crate) ops: PreparedOps,
    /// Length-1 stand-ins so a matmul kernel can always bind an id or a global scale.
    pub(crate) unit_u32: Tensor,
    pub(crate) unit_f32: Tensor,
}

impl TensorKernels {
    pub fn new(runtime: &Runtime) -> Result<Self, GoldyError> {
        Self::new_for_policy(runtime, crate::ArithmeticPolicy::MatchRecorded)
    }

    /// Prepare only `policy`; the alternate Metal variant is compiled on first use.
    pub fn new_for_policy(runtime: &Runtime, policy: crate::ArithmeticPolicy) -> Result<Self, GoldyError> {
        Ok(Self {
            runtime: runtime.clone(),
            ops: PreparedOps::prepare(runtime, policy)?,
            unit_u32: Tensor::from_u32(runtime, TensorShape::vector(1), &[0])?,
            unit_f32: Tensor::from_f32(runtime, TensorShape::vector(1), &[1.0])?,
        })
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Borrow `scheme` and record tensor ops into it.
    pub fn recorder<'a>(&'a self, scheme: &'a mut Scheme) -> TensorRecorder<'a> {
        TensorRecorder { kernels: self, scheme }
    }
}

/// Facade that borrows [`TensorKernels`] and a mutable [`Scheme`].
pub struct TensorRecorder<'a> {
    pub(crate) kernels: &'a TensorKernels,
    pub(crate) scheme: &'a mut Scheme,
}

impl<'a> TensorRecorder<'a> {
    pub fn scheme(&mut self) -> &mut Scheme {
        self.scheme
    }

    /// States `node`, a dispatch already recorded on this scheme, as the tensor
    /// operations `describe` records, so automatic fusion can compose it with its
    /// neighbours. The operations are not recorded on the scheme: a fused region either
    /// runs all of them in place of the node, or the node runs as recorded.
    ///
    /// The operations must store what the dispatch stores. With `exact` they compute it
    /// bit for bit: the dispatch rounds wherever an operation stores and sums in the
    /// order the operations do. Otherwise the description joins a region only when the
    /// scheme's [`crate::ContractionPrecision`] lets contractions reassociate.
    ///
    /// The scheme binds every buffer the operations bind, as it binds a node's parcels,
    /// whether or not a region runs them. A buffer only the operations use carries values
    /// between them: the caller owns it, typically a retained buffer held beside the
    /// scheme and written with `_into` forms, and must keep it alive while the scheme
    /// submits. Dropping it first makes [`Scheme::submit`] return
    /// [`GoldyError::StaleResource`]. Scalars are fixed at the values recorded.
    pub fn describe(
        &mut self,
        node: crate::NodeId,
        exact: bool,
        describe: impl FnOnce(&mut TensorRecorder<'_>) -> Result<(), GoldyError>,
    ) -> Result<(), GoldyError> {
        let mut sketch = self.scheme.sketch();
        describe(&mut self.kernels.recorder(&mut sketch))?;
        self.scheme.describe_node(node, &sketch, exact)
    }

    pub fn zeros(&mut self, shape: TensorShape, dtype: TensorDType) -> Result<Tensor, GoldyError> {
        Tensor::zeros(&self.kernels.runtime, shape, dtype)
    }

    pub fn fill(&mut self, label: &str, out: TensorView<'_>, value: TensorScalar) -> Result<(), GoldyError> {
        out.layout().require_writeable("fill")?;
        if value.dtype() != out.dtype() {
            return Err(GoldyError::Validation("tensor fill: scalar dtype mismatch".into()));
        }
        let n = out.numel_u32().max(1);
        // The source operand is never read by a fill.
        if out.dtype() == TensorDType::F32 {
            let node = self
                .kernels
                .ops
                .unary
                .record(self.scheme, label, out, out, value.as_f32()?, OP_FILL)?
                .over_1d(n)
                .node();
            self.scheme.record_semantic_site(node, semantic::fill(out));
        } else {
            let out = out.reinterpret(TensorDType::U32)?;
            self.kernels
                .ops
                .copy
                .record(self.scheme, label, out, out, OP_FILL, value.bits())?
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
        let node = self
            .kernels
            .ops
            .copy
            .record(
                self.scheme,
                label,
                src.reinterpret(TensorDType::U32)?,
                dst.reinterpret(TensorDType::U32)?,
                OP_COPY,
                0,
            )?
            .over_1d(dst.numel_u32().max(1))
            .node();
        self.scheme.record_semantic_site(node, semantic::copy(src, dst));
        Ok(())
    }

    pub fn contiguous(&mut self, label: &str, src: TensorView<'_>) -> Result<Tensor, GoldyError> {
        let out = Tensor::zeros(&self.kernels.runtime, src.shape(), src.dtype())?;
        self.copy(label, src, out.view())?;
        Ok(out)
    }

    pub fn cast(&mut self, label: &str, src: TensorView<'_>, dtype: TensorDType) -> Result<Tensor, GoldyError> {
        let out = Tensor::zeros(&self.kernels.runtime, src.shape(), dtype)?;
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
        // Elements pair up in logical order.
        let n = dst.numel_u32().max(1);
        let ops = &self.kernels.ops;
        match (src.dtype(), dst.dtype()) {
            (TensorDType::F32, TensorDType::I32) => {
                ops.cast_f32_i32.record(self.scheme, label, src, dst)?.over_1d(n);
            }
            (TensorDType::F32, TensorDType::U32) => {
                ops.cast_f32_u32.record(self.scheme, label, src, dst)?.over_1d(n);
            }
            (TensorDType::I32, TensorDType::F32) => {
                ops.cast_i32_f32.record(self.scheme, label, src, dst)?.over_1d(n);
            }
            (TensorDType::U32, TensorDType::F32) => {
                ops.cast_u32_f32.record(self.scheme, label, src, dst)?.over_1d(n);
            }
            (TensorDType::I32, TensorDType::U32) | (TensorDType::U32, TensorDType::I32) => {
                ops.copy
                    .record(
                        self.scheme,
                        label,
                        src.reinterpret(TensorDType::U32)?,
                        dst.reinterpret(TensorDType::U32)?,
                        OP_COPY,
                        0,
                    )?
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
    /// Numerically stable hyperbolic tangent. `|x| >= 10` saturates to ±1.
    pub fn tanh(&mut self, label: &str, src: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.unary(label, src, OP_TANH)
    }
    /// Numerically stable logistic sigmoid. Large `|x|` saturates to 0 or 1.
    pub fn sigmoid(&mut self, label: &str, src: TensorView<'_>) -> Result<Tensor, GoldyError> {
        self.unary(label, src, OP_SIGMOID)
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

    // Into an existing view, which may be one of the inputs: each element is read
    // before it is written by the same invocation.
    pub fn neg_into(&mut self, label: &str, src: TensorView<'_>, out: TensorView<'_>) -> Result<(), GoldyError> {
        self.unary_into(label, src, out, OP_NEG)
    }
    pub fn abs_into(&mut self, label: &str, src: TensorView<'_>, out: TensorView<'_>) -> Result<(), GoldyError> {
        self.unary_into(label, src, out, OP_ABS)
    }
    pub fn exp_into(&mut self, label: &str, src: TensorView<'_>, out: TensorView<'_>) -> Result<(), GoldyError> {
        self.unary_into(label, src, out, OP_EXP)
    }
    pub fn log_into(&mut self, label: &str, src: TensorView<'_>, out: TensorView<'_>) -> Result<(), GoldyError> {
        self.unary_into(label, src, out, OP_LOG)
    }
    pub fn sqrt_into(&mut self, label: &str, src: TensorView<'_>, out: TensorView<'_>) -> Result<(), GoldyError> {
        self.unary_into(label, src, out, OP_SQRT)
    }
    pub fn reciprocal_into(&mut self, label: &str, src: TensorView<'_>, out: TensorView<'_>) -> Result<(), GoldyError> {
        self.unary_into(label, src, out, OP_RECIP)
    }
    pub fn tanh_into(&mut self, label: &str, src: TensorView<'_>, out: TensorView<'_>) -> Result<(), GoldyError> {
        self.unary_into(label, src, out, OP_TANH)
    }
    pub fn sigmoid_into(&mut self, label: &str, src: TensorView<'_>, out: TensorView<'_>) -> Result<(), GoldyError> {
        self.unary_into(label, src, out, OP_SIGMOID)
    }
    pub fn sub_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_into(label, a, b, out, OP_SUB)
    }
    pub fn mul_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_into(label, a, b, out, OP_MUL)
    }
    pub fn div_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_into(label, a, b, out, OP_DIV)
    }
    pub fn min_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_into(label, a, b, out, OP_MIN)
    }
    pub fn max_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_into(label, a, b, out, OP_MAX)
    }
    pub fn add_scalar_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        scalar: f32,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_scalar_into(label, a, scalar, out, OP_ADD_SCALAR)
    }
    pub fn sub_scalar_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        scalar: f32,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_scalar_into(label, a, scalar, out, OP_SUB_SCALAR)
    }
    pub fn mul_scalar_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        scalar: f32,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_scalar_into(label, a, scalar, out, OP_MUL_SCALAR)
    }
    pub fn div_scalar_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        scalar: f32,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_scalar_into(label, a, scalar, out, OP_DIV_SCALAR)
    }
    pub fn min_scalar_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        scalar: f32,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_scalar_into(label, a, scalar, out, OP_MIN_SCALAR)
    }
    pub fn max_scalar_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        scalar: f32,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.binary_scalar_into(label, a, scalar, out, OP_MAX_SCALAR)
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
    /// [`Self::sum`] into `out`, with or without the reduced axis.
    pub fn sum_into(
        &mut self,
        label: &str,
        src: TensorView<'_>,
        axis: usize,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let kept = Self::kept(src, axis, out)?;
        self.reduce_into(label, src, axis, kept, OP_SUM)
    }
    /// [`Self::mean`] into `out`, with or without the reduced axis.
    pub fn mean_into(
        &mut self,
        label: &str,
        src: TensorView<'_>,
        axis: usize,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let kept = Self::kept(src, axis, out)?;
        self.reduce_into(label, src, axis, kept, OP_MEAN)
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
        let out = Tensor::zeros(&self.kernels.runtime, index.shape(), src.dtype())?;
        self.kernels
            .ops
            .gather
            .record(
                self.scheme,
                label,
                src,
                index.reinterpret(TensorDType::I32)?,
                out.view(),
                axis as u32,
            )?
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
        let node = self
            .kernels
            .ops
            .scatter
            .record(
                self.scheme,
                label,
                src,
                index.reinterpret(TensorDType::I32)?,
                dst,
                op,
                axis as u32,
            )?
            .groups([1, 1, 1])
            .node();
        if mode == ScatterMode::UniqueWrite {
            self.scheme
                .record_semantic_site(node, semantic::scatter_slice(src, index, dst, axis));
        }
        Ok(())
    }

    /// Store `src` in a row-major cyclic history and compute one causal row dot:
    /// [`Self::cyclic_store_at`] then [`Self::cyclic_window_dot_at`], labelled
    /// `{label}_store` and `{label}_dot`.
    ///
    /// `state` and `weight` are `[rows.., width]` and `src`/`out` are `[rows..]`, so a row
    /// may carry several axes; `position` is one `u32`.
    /// The newest value uses the last weight column and positions before zero
    /// contribute zero. Recording emits ordinary generated kernels so automatic
    /// fusion may compose the store, dot, and neighboring tensor algebra.
    pub fn cyclic_dot_at(
        &mut self,
        label: &str,
        src: TensorView<'_>,
        weight: TensorView<'_>,
        state: TensorView<'_>,
        position: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        Self::cyclic_rows("cyclic_dot_at", weight, state, position, out)?;
        self.cyclic_store_at(&format!("{label}_store"), src, state, position)?;
        self.cyclic_window_dot_at(&format!("{label}_dot"), weight, state, position, out)
    }

    /// Store `src` (`[rows..]`) at slot `position % width` of the last axis of `state`
    /// (`[rows.., width]`). `position` is one `u32`.
    pub fn cyclic_store_at(
        &mut self,
        label: &str,
        src: TensorView<'_>,
        state: TensorView<'_>,
        position: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let rows = Self::cyclic_rows("cyclic_store_at", state, state, position, src)?;
        state.layout().require_writeable(&format!("{label} state"))?;
        let node = self
            .kernels
            .ops
            .cyclic_store
            .record(self.scheme, label, src, state, position)?
            .over_1d(rows)
            .node();
        self.scheme
            .record_semantic_site(node, semantic::cyclic_store(src, state, position));
        Ok(())
    }

    /// The causal row dot of [`Self::cyclic_dot_at`] over a history that already holds
    /// the value at `position`.
    pub fn cyclic_window_dot_at(
        &mut self,
        label: &str,
        weight: TensorView<'_>,
        state: TensorView<'_>,
        position: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let rows = Self::cyclic_rows("cyclic_window_dot_at", weight, state, position, out)?;
        out.layout().require_writeable(&format!("{label} out"))?;
        let node = self
            .kernels
            .ops
            .cyclic_dot
            .record(self.scheme, label, state, weight, position, out)?
            .over_1d(rows)
            .node();
        self.scheme
            .record_semantic_site(node, semantic::cyclic_dot(state, weight, position, out));
        Ok(())
    }

    /// Checks a cyclic history `state` (`[rows.., width]`), a `peer` of its shape, and a
    /// `row` view of `[rows..]`; returns the row count.
    fn cyclic_rows(
        op: &str,
        peer: TensorView<'_>,
        state: TensorView<'_>,
        position: TensorView<'_>,
        row: TensorView<'_>,
    ) -> Result<u32, GoldyError> {
        for (name, view) in [("peer", peer), ("state", state), ("row", row)] {
            view.dtype().require_f32(&format!("{op} {name}"))?;
        }
        if position.dtype() != TensorDType::U32 || position.numel_u32() != 1 {
            return Err(GoldyError::Validation(format!("{op}: position must be one U32")));
        }
        let state_shape = state.shape();
        let state_dims = state_shape.dims();
        let Some((&width, rows)) = state_dims.split_last() else {
            return Err(GoldyError::Validation(format!("{op}: state needs a width axis")));
        };
        if rows.is_empty() || width == 0 || peer.shape().dims() != state_dims || row.shape().dims() != rows {
            return Err(GoldyError::Validation(format!(
                "{op}: rows must be [rows..] and the history [rows.., width]"
            )));
        }
        Ok(row.numel_u32().max(1))
    }

    fn unary(&mut self, label: &str, src: TensorView<'_>, op: u32) -> Result<Tensor, GoldyError> {
        src.dtype().require_f32("unary")?;
        let out = Tensor::zeros(&self.kernels.runtime, src.shape(), TensorDType::F32)?;
        self.unary_into(label, src, out.view(), op)?;
        Ok(out)
    }

    fn unary_into(&mut self, label: &str, src: TensorView<'_>, dst: TensorView<'_>, op: u32) -> Result<(), GoldyError> {
        dst.layout().require_writeable("unary")?;
        src.dtype().require_f32("unary")?;
        dst.dtype().require_f32("unary")?;
        let node = self
            .kernels
            .ops
            .unary
            .record(self.scheme, label, src, dst, 0.0, op)?
            .over_1d(dst.numel_u32().max(1))
            .node();
        self.scheme.record_semantic_site(node, semantic::unary(op, src, dst));
        Ok(())
    }

    fn binary(&mut self, label: &str, a: TensorView<'_>, b: TensorView<'_>, op: u32) -> Result<Tensor, GoldyError> {
        a.dtype().require_f32("binary")?;
        b.dtype().require_f32("binary")?;
        let shape = broadcast_shapes(a.shape(), b.shape())?;
        let out = Tensor::zeros(&self.kernels.runtime, shape, TensorDType::F32)?;
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
        let node = self
            .kernels
            .ops
            .binary
            .record(self.scheme, label, a, b, out, 0.0, op)?
            .over_1d(out.numel_u32().max(1))
            .node();
        self.scheme.record_semantic_site(node, semantic::binary(op, a, b, out));
        Ok(())
    }

    fn binary_scalar(&mut self, label: &str, a: TensorView<'_>, scalar: f32, op: u32) -> Result<Tensor, GoldyError> {
        a.dtype().require_f32("binary_scalar")?;
        let out = Tensor::zeros(&self.kernels.runtime, a.shape(), TensorDType::F32)?;
        self.binary_scalar_into(label, a, scalar, out.view(), op)?;
        Ok(out)
    }

    fn binary_scalar_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        scalar: f32,
        out: TensorView<'_>,
        op: u32,
    ) -> Result<(), GoldyError> {
        out.layout().require_writeable("binary_scalar")?;
        a.dtype().require_f32("binary_scalar")?;
        out.dtype().require_f32("binary_scalar")?;
        let identity = (op == super::kernels::OP_ADD_SCALAR && scalar == 0.0)
            || (op == super::kernels::OP_SUB_SCALAR && scalar == 0.0)
            || (op == super::kernels::OP_MUL_SCALAR && scalar == 1.0)
            || (op == super::kernels::OP_DIV_SCALAR && scalar == 1.0);
        if identity {
            return self.copy(label, a, out);
        }
        let a = a.broadcast_to(out.shape())?;
        // `b` is never read by a scalar op.
        let node = self
            .kernels
            .ops
            .binary
            .record(self.scheme, label, a, a, out, scalar, op)?
            .over_1d(out.numel_u32().max(1))
            .node();
        self.scheme.record_semantic_site(node, semantic::binary(op, a, a, out));
        Ok(())
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
        let src_shape = src.shape();
        src_shape.dim(axis)?;
        let mut keep_dims = src_shape.dims().to_vec();
        keep_dims[axis] = 1;
        let keep_shape = TensorShape::from_dims(&keep_dims)?;
        let out_shape = if keepdim {
            keep_shape
        } else {
            src.shape().squeeze_axis(axis)?
        };
        let out = Tensor::zeros(&self.kernels.runtime, out_shape, TensorDType::F32)?;
        let kernel_layout = super::layout::TensorLayout::packed(out.dtype(), keep_shape, 0)?;
        self.reduce_into(label, src, axis, TensorView::new(out.buffer(), kernel_layout)?, op)?;
        Ok(out)
    }

    /// Reduces `src` along `axis` into `out`, whose shape keeps `axis` with extent one.
    fn reduce_into(
        &mut self,
        label: &str,
        src: TensorView<'_>,
        axis: usize,
        kernel_view: TensorView<'_>,
        op: u32,
    ) -> Result<(), GoldyError> {
        kernel_view.layout().require_writeable("reduce")?;
        src.dtype().require_f32("reduce")?;
        kernel_view.dtype().require_f32("reduce")?;
        let reduce_len = src.shape().dim(axis)?;
        let inner = src.shape().dims()[axis + 1..].iter().product::<u32>();
        let node = self
            .kernels
            .ops
            .reduce
            .record(self.scheme, label, src, kernel_view, op, reduce_len, inner)?
            .over_1d(kernel_view.numel_u32().max(1))
            .node();
        self.scheme
            .record_semantic_site(node, semantic::reduce(op, src, axis, kernel_view));
        Ok(())
    }

    /// `out`, reshaped to `src`'s shape with `axis` kept at extent one.
    fn kept<'v>(src: TensorView<'_>, axis: usize, out: TensorView<'v>) -> Result<TensorView<'v>, GoldyError> {
        let mut dims = src.shape().dims().to_vec();
        *dims
            .get_mut(axis)
            .ok_or_else(|| GoldyError::Validation(format!("reduce: axis {axis} out of range")))? = 1;
        out.reshape(&dims)
    }
}
