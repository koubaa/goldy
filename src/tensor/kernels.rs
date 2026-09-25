//! Portable `#[goldy::compute]` kernels for the dense tensor op set.
//!
//! Operands are tensor views, so element offsets travel as launch words and shapes bake
//! through the specializer. The op code, axis, reduction length and scalar operand are
//! scalars fixed when the op is recorded, so they bake at the node's first submit too.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

pub const OP_COPY: u32 = 0;
pub const OP_FILL: u32 = 1;
pub const OP_CAST: u32 = 2;
pub const OP_NEG: u32 = 3;
pub const OP_ABS: u32 = 4;
pub const OP_EXP: u32 = 5;
pub const OP_LOG: u32 = 6;
pub const OP_SQRT: u32 = 7;
pub const OP_RECIP: u32 = 8;
pub const OP_ADD: u32 = 9;
pub const OP_SUB: u32 = 10;
pub const OP_MUL: u32 = 11;
pub const OP_DIV: u32 = 12;
pub const OP_MIN: u32 = 13;
pub const OP_MAX: u32 = 14;
pub const OP_ADD_SCALAR: u32 = 15;
pub const OP_SUB_SCALAR: u32 = 16;
pub const OP_MUL_SCALAR: u32 = 17;
pub const OP_DIV_SCALAR: u32 = 18;
pub const OP_MIN_SCALAR: u32 = 19;
pub const OP_MAX_SCALAR: u32 = 20;
pub const OP_SUM: u32 = 21;
pub const OP_RMAX: u32 = 22;
pub const OP_RMIN: u32 = 23;
pub const OP_MEAN: u32 = 24;
pub const OP_GATHER: u32 = 25;
pub const OP_SCATTER_SET: u32 = 26;
pub const OP_SCATTER_ADD: u32 = 27;
pub const OP_SCATTER_MIN: u32 = 28;
pub const OP_SCATTER_MAX: u32 = 29;
pub const OP_CAST_F32_I32: u32 = 30;
pub const OP_CAST_F32_U32: u32 = 31;
pub const OP_CAST_I32_F32: u32 = 32;
pub const OP_CAST_U32_F32: u32 = 33;
pub const OP_CAST_I32_U32: u32 = 34;
pub const OP_CAST_U32_I32: u32 = 35;

/// `dst = op(src)`, or `scalar` for [`OP_FILL`]. `src` has `dst`'s shape.
///
/// `scalar` is user slot 0, where semantic sites read it (`semantic.rs`).
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_unary_f32(src: goldy::gpu::Tensor<f32>, dst: goldy::gpu::TensorWrite<f32>, scalar: f32, op: u32) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        let mut v = scalar;
        if op != 1 {
            v = src[i];
            if op == 3 {
                v = -v;
            } else if op == 4 {
                v = goldy::gpu::abs(v);
            } else if op == 5 {
                v = goldy::gpu::exp(v);
            } else if op == 6 {
                v = goldy::gpu::log(v);
            } else if op == 7 {
                v = goldy::gpu::sqrt(v);
            } else if op == 8 {
                v = 1.0 / v;
            }
        }
        dst[i] = v;
    }
}

/// `dst = a op b`, or `a op scalar` for the `_SCALAR` ops. `a` and `b` have `dst`'s shape.
///
/// `scalar` is user slot 0, where semantic sites read it (`semantic.rs`).
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_binary_f32(
    a: goldy::gpu::Tensor<f32>,
    b: goldy::gpu::Tensor<f32>,
    dst: goldy::gpu::TensorWrite<f32>,
    scalar: f32,
    op: u32,
) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        let av = a[i];
        let mut bv = scalar;
        if op < 15 {
            bv = b[i];
        }
        let mut v = 0.0;
        if op == 9 || op == 15 {
            v = av + bv;
        } else if op == 10 || op == 16 {
            v = av - bv;
        } else if op == 11 || op == 17 {
            v = av * bv;
        } else if op == 12 || op == 18 {
            v = av / bv;
        } else if op == 13 || op == 19 {
            v = goldy::gpu::min(av, bv);
        } else if op == 14 || op == 20 {
            v = goldy::gpu::max(av, bv);
        }
        dst[i] = v;
    }
}

/// 32-bit copy, or `bits` for [`OP_FILL`]. Views of other 32-bit dtypes are reinterpreted.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_copy_u32(src: goldy::gpu::Tensor<u32>, dst: goldy::gpu::TensorWrite<u32>, op: u32, bits: u32) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        if op == 1 {
            dst[i] = bits;
        } else {
            dst[i] = src[i];
        }
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_cast_f32_i32(src: goldy::gpu::Tensor<f32>, dst: goldy::gpu::TensorWrite<i32>) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        dst[i] = src[i] as i32;
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_cast_f32_u32(src: goldy::gpu::Tensor<f32>, dst: goldy::gpu::TensorWrite<u32>) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        dst[i] = src[i] as u32;
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_cast_i32_f32(src: goldy::gpu::Tensor<i32>, dst: goldy::gpu::TensorWrite<f32>) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        dst[i] = src[i] as f32;
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_cast_u32_f32(src: goldy::gpu::Tensor<u32>, dst: goldy::gpu::TensorWrite<f32>) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        dst[i] = src[i] as f32;
    }
}

/// Reduce `src` along one axis into `dst`, which has `src`'s shape with that axis at one.
///
/// `inner` is the element count of the axes after the reduced one, so element `i` of `dst`
/// reduces `src[(i / inner) * reduce_len * inner + k * inner + i % inner]` for `k` in order.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_reduce_f32(
    src: goldy::gpu::Tensor<f32>,
    dst: goldy::gpu::TensorWrite<f32>,
    op: u32,
    reduce_len: u32,
    inner: u32,
) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        let outer = i / inner;
        let base = outer * reduce_len * inner + (i - outer * inner);
        let mut acc = 0.0;
        if op == 22 {
            acc = -3.402823e38;
        } else if op == 23 {
            acc = 3.402823e38;
        }
        let mut k = 0u32;
        while k < reduce_len {
            let v = src[base + k * inner];
            if op == 21 || op == 24 {
                acc = acc + v;
            } else if op == 22 {
                acc = goldy::gpu::max(acc, v);
            } else if op == 23 {
                acc = goldy::gpu::min(acc, v);
            }
            k = k + 1;
        }
        if op == 24 {
            acc = acc / (reduce_len as f32);
        }
        dst[i] = acc;
    }
}

/// `dst[c] = src[c with c[axis] = index[c]]`. `index` has `dst`'s shape.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_gather_f32(
    src: goldy::gpu::Tensor<f32>,
    index: goldy::gpu::Tensor<i32>,
    dst: goldy::gpu::TensorWrite<f32>,
    axis: u32,
) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        let mut rest = i;
        let mut c3 = rest % dst.dim(3);
        rest = rest / dst.dim(3);
        let mut c2 = rest % dst.dim(2);
        rest = rest / dst.dim(2);
        let mut c1 = rest % dst.dim(1);
        let mut c0 = rest / dst.dim(1);
        let g = index[i] as u32;
        if axis == 0 {
            c0 = g;
        } else if axis == 1 {
            c1 = g;
        } else if axis == 2 {
            c2 = g;
        } else {
            c3 = g;
        }
        dst[i] = src[((c0 * src.dim(1) + c1) * src.dim(2) + c2) * src.dim(3) + c3];
    }
}

/// Collision-safe scatter: one thread walks every index so add/min/max are defined.
///
/// `dst[c with c[axis] = index[c]] op= src[c]`. `index` has `src`'s shape.
#[goldy::compute(workgroup_size = [1, 1, 1])]
fn tensor_scatter_f32(
    src: goldy::gpu::Tensor<f32>,
    index: goldy::gpu::Tensor<i32>,
    dst: goldy::gpu::TensorMut<f32>,
    op: u32,
    axis: u32,
) {
    let lid = goldy::gpu::global_id().x;
    if lid != 0 {
        return;
    }
    let mut i = 0u32;
    while i < src.len() {
        let mut rest = i;
        let mut c3 = rest % src.dim(3);
        rest = rest / src.dim(3);
        let mut c2 = rest % src.dim(2);
        rest = rest / src.dim(2);
        let mut c1 = rest % src.dim(1);
        let mut c0 = rest / src.dim(1);
        let g = index[i] as u32;
        if axis == 0 {
            c0 = g;
        } else if axis == 1 {
            c1 = g;
        } else if axis == 2 {
            c2 = g;
        } else {
            c3 = g;
        }
        let di = ((c0 * dst.dim(1) + c1) * dst.dim(2) + c2) * dst.dim(3) + c3;
        let sv = src[i];
        if op == 26 {
            dst[di] = sv;
        } else if op == 27 {
            dst[di] = dst[di] + sv;
        } else if op == 28 {
            dst[di] = goldy::gpu::min(dst[di], sv);
        } else if op == 29 {
            dst[di] = goldy::gpu::max(dst[di], sv);
        }
        i = i + 1;
    }
}

/// Rank-3 GEMM over any strides: `C[b,i,j] = A[b,i,p] @ B[b,p,j]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_batched_matmul_f32(
    #[tensor(shape = [batches, rows, depth])] a: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [batches, depth, cols])] b: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [batches, rows, cols])] dst: goldy::gpu::TensorWrite<f32>,
) {
    let idx = goldy::gpu::global_id().x;
    if idx < dst.len() {
        let n = dst.dim(2);
        let m = dst.dim(1);
        let kdim = a.dim(2);
        let i = idx / n;
        let j = idx - i * n;
        let batch = i / m;
        let row = i - batch * m;
        let mut acc = 0.0;
        let mut p = 0u32;
        while p < kdim {
            let av = a[(batch * m + row) * kdim + p];
            let bv = b[(batch * kdim + p) * n + j];
            acc = acc + av * bv;
            p = p + 1;
        }
        dst[idx] = acc;
    }
}

pub use tensor_batched_matmul_f32::Kernel as BatchedMatMulKernel;
pub use tensor_binary_f32::Kernel as BinaryF32Kernel;
pub use tensor_cast_f32_i32::Kernel as CastF32I32Kernel;
pub use tensor_cast_f32_u32::Kernel as CastF32U32Kernel;
pub use tensor_cast_i32_f32::Kernel as CastI32F32Kernel;
pub use tensor_cast_u32_f32::Kernel as CastU32F32Kernel;
pub use tensor_copy_u32::Kernel as CopyU32Kernel;
pub use tensor_gather_f32::Kernel as GatherF32Kernel;
pub use tensor_reduce_f32::Kernel as ReduceF32Kernel;
pub use tensor_scatter_f32::Kernel as ScatterF32Kernel;
pub use tensor_unary_f32::Kernel as UnaryF32Kernel;
