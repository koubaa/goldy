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
pub const OP_TANH: u32 = 36;
pub const OP_SIGMOID: u32 = 37;

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
            } else if op == 36 {
                v = goldy::gpu::tanh(v);
            } else if op == 37 {
                v = goldy::gpu::sigmoid(v);
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

/// Copy slab `ids[slot]` out of a packed `[n_slabs, stride]` tensor.
///
/// An id past `n_slabs` writes zeros and does not read `src`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_gather_slab_f32(
    src: goldy::gpu::Tensor<f32>,
    ids: goldy::gpu::Tensor<u32>,
    dst: goldy::gpu::TensorWrite<f32>,
    slot: u32,
    stride: u32,
    n_slabs: u32,
) {
    let i = goldy::gpu::global_id().x;
    if i >= dst.len() {
        return;
    }
    let id = ids[slot];
    if id >= n_slabs || i >= stride {
        dst[i] = 0.0;
        return;
    }
    dst[i] = src[id * stride + i];
}

pub use tensor_batched_matmul_f32::Kernel as BatchedMatMulKernel;
pub use tensor_binary_f32::Kernel as BinaryF32Kernel;
pub use tensor_cast_f32_i32::Kernel as CastF32I32Kernel;
pub use tensor_cast_f32_u32::Kernel as CastF32U32Kernel;
pub use tensor_cast_i32_f32::Kernel as CastI32F32Kernel;
pub use tensor_cast_u32_f32::Kernel as CastU32F32Kernel;
pub use tensor_copy_u32::Kernel as CopyU32Kernel;
pub use tensor_gather_f32::Kernel as GatherF32Kernel;
pub use tensor_gather_slab_f32::Kernel as GatherSlabF32Kernel;
pub use tensor_matvec_bf16::Kernel as MatVecBf16Kernel;
pub use tensor_matvec_nvfp4::Kernel as MatVecNvfp4Kernel;
pub use tensor_reduce_f32::Kernel as ReduceF32Kernel;
pub use tensor_scatter_f32::Kernel as ScatterF32Kernel;
pub use tensor_unary_f32::Kernel as UnaryF32Kernel;

/// `out[row] = sum_k bf16_to_f32(weight[row, k]) * x[k]`.
///
/// `bits` is the weight's bytes viewed as `u32` words, little-endian. Element
/// `e` is the low or high 16 bits of word `e / 2`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_matvec_bf16(
    bits: goldy::gpu::Tensor<u32>,
    x: goldy::gpu::Tensor<f32>,
    out: goldy::gpu::TensorWrite<f32>,
    k: u32,
) {
    let row = goldy::gpu::global_id().x;
    if row >= out.len() {
        return;
    }
    let mut acc = 0.0;
    let mut col = 0u32;
    while col < k {
        let elem = row * k + col;
        let word = bits[elem >> 1];
        let mut half = word & 65535u32;
        if (elem & 1u32) != 0u32 {
            half = word >> 16;
        }
        let w = goldy::gpu::as_f32(half << 16);
        acc = acc + w * x[col];
        col = col + 1;
    }
    out[row] = acc;
}

/// NVFP4 matvec. `codes` holds two E2M1 nibbles per byte, even column in the
/// low nibble. `scales` is one E4M3 byte per 16 columns. `factors[id]` is
/// `amax / outer_scale_denominator` for that expert. An id past `experts`
/// writes zeros.
///
/// Layout matches transformers `Inkling2LoopDequantizeExperts`: codes
/// `[experts, rows, k/2]`, scales `[experts, rows, k/16]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_matvec_nvfp4(
    codes: goldy::gpu::Tensor<u32>,
    scales: goldy::gpu::Tensor<u32>,
    factors: goldy::gpu::Tensor<f32>,
    ids: goldy::gpu::Tensor<u32>,
    x: goldy::gpu::Tensor<f32>,
    out: goldy::gpu::TensorWrite<f32>,
    rows: u32,
    k: u32,
    slot: u32,
) {
    let row = goldy::gpu::global_id().x;
    if row >= rows {
        return;
    }
    let id = ids[slot];
    let experts = factors.len();
    if id >= experts {
        out[row] = 0.0;
        return;
    }
    let factor = factors[id];
    let code_bytes = k >> 1;
    let scale_bytes = k >> 4;
    let mut acc = 0.0;
    let mut col = 0u32;
    while col < k {
        let byte_index = (id * rows + row) * code_bytes + (col >> 1);
        let word = codes[byte_index >> 2];
        let shift = (byte_index & 3u32) * 8u32;
        let byte = (word >> shift) & 255u32;
        let mut nib = byte & 15u32;
        if (col & 1u32) != 0u32 {
            nib = byte >> 4;
        }
        let mut mag = 0.0;
        if nib == 1u32 || nib == 9u32 {
            mag = 0.5;
        } else if nib == 2u32 || nib == 10u32 {
            mag = 1.0;
        } else if nib == 3u32 || nib == 11u32 {
            mag = 1.5;
        } else if nib == 4u32 || nib == 12u32 {
            mag = 2.0;
        } else if nib == 5u32 || nib == 13u32 {
            mag = 3.0;
        } else if nib == 6u32 || nib == 14u32 {
            mag = 4.0;
        } else if nib == 7u32 || nib == 15u32 {
            mag = 6.0;
        }
        if nib >= 8u32 {
            mag = -mag;
        }
        let scale_index = (id * rows + row) * scale_bytes + (col >> 4);
        let sword = scales[scale_index >> 2];
        let sshift = (scale_index & 3u32) * 8u32;
        let sb = (sword >> sshift) & 255u32;
        let sign = sb >> 7;
        let exp = (sb >> 3) & 15u32;
        let mant = sb & 7u32;
        let mut bits = 0u32;
        if exp == 0u32 {
            if mant == 1u32 {
                bits = 989855744u32;
            } else if mant == 2u32 {
                bits = 998244352u32;
            } else if mant == 3u32 {
                bits = 1002438656u32;
            } else if mant == 4u32 {
                bits = 1006632960u32;
            } else if mant == 5u32 {
                bits = 1008730112u32;
            } else if mant == 6u32 {
                bits = 1010827264u32;
            } else if mant == 7u32 {
                bits = 1012924416u32;
            }
        } else if exp == 15u32 && mant == 7u32 {
            bits = 2143289344u32;
        } else {
            bits = ((exp + 120u32) << 23) | (mant << 20);
        }
        let mut scale = goldy::gpu::as_f32(bits);
        if sign != 0u32 {
            scale = -scale;
        }
        acc = acc + mag * scale * factor * x[col];
        col = col + 1;
    }
    out[row] = acc;
}
