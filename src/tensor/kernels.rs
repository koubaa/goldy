//! Portable `#[goldy::compute]` kernels for the dense tensor op set.

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

#[goldy::gpu]
pub struct TensorOpMeta {
    pub op: u32,
    pub axis: u32,
    pub scalar_bits: u32,
    pub reduce_len: u32,
    pub a_off: u32,
    pub a_numel: u32,
    pub a_s0: u32,
    pub a_s1: u32,
    pub a_s2: u32,
    pub a_s3: u32,
    pub a_d0: u32,
    pub a_d1: u32,
    pub a_d2: u32,
    pub a_d3: u32,
    pub b_off: u32,
    pub b_numel: u32,
    pub b_s0: u32,
    pub b_s1: u32,
    pub b_s2: u32,
    pub b_s3: u32,
    pub b_d0: u32,
    pub b_d1: u32,
    pub b_d2: u32,
    pub b_d3: u32,
    pub o_off: u32,
    pub o_numel: u32,
    pub o_s0: u32,
    pub o_s1: u32,
    pub o_s2: u32,
    pub o_s3: u32,
    pub o_d0: u32,
    pub o_d1: u32,
    pub o_d2: u32,
    pub o_d3: u32,
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_unary_f32(src: &[f32], dst: goldy::gpu::Scattered<f32>, meta: &[TensorOpMeta], scalar: f32) {
    let i = goldy::gpu::global_id().x;
    let m: TensorOpMeta = meta[0];
    if i < m.o_numel {
        let mut rest = i;
        let i3 = rest % m.o_d3;
        rest = rest / m.o_d3;
        let i2 = rest % m.o_d2;
        rest = rest / m.o_d2;
        let i1 = rest % m.o_d1;
        rest = rest / m.o_d1;
        let i0 = rest % m.o_d0;
        let di = m.o_off + i0 * m.o_s0 + i1 * m.o_s1 + i2 * m.o_s2 + i3 * m.o_s3;
        let mut v = scalar;
        if m.op != 1 {
            let si = m.a_off + i0 * m.a_s0 + i1 * m.a_s1 + i2 * m.a_s2 + i3 * m.a_s3;
            v = src[si];
            if m.op == 3 {
                v = -v;
            } else if m.op == 4 {
                v = goldy::gpu::abs(v);
            } else if m.op == 5 {
                v = goldy::gpu::exp(v);
            } else if m.op == 6 {
                v = goldy::gpu::log(v);
            } else if m.op == 7 {
                v = goldy::gpu::sqrt(v);
            } else if m.op == 8 {
                v = 1.0 / v;
            }
        }
        dst[di] = v;
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_binary_f32(a: &[f32], b: &[f32], dst: goldy::gpu::Scattered<f32>, meta: &[TensorOpMeta], scalar: f32) {
    let i = goldy::gpu::global_id().x;
    let m: TensorOpMeta = meta[0];
    if i < m.o_numel {
        let mut rest = i;
        let i3 = rest % m.o_d3;
        rest = rest / m.o_d3;
        let i2 = rest % m.o_d2;
        rest = rest / m.o_d2;
        let i1 = rest % m.o_d1;
        rest = rest / m.o_d1;
        let i0 = rest % m.o_d0;
        let ai = m.a_off + i0 * m.a_s0 + i1 * m.a_s1 + i2 * m.a_s2 + i3 * m.a_s3;
        let bi = m.b_off + i0 * m.b_s0 + i1 * m.b_s1 + i2 * m.b_s2 + i3 * m.b_s3;
        let di = m.o_off + i0 * m.o_s0 + i1 * m.o_s1 + i2 * m.o_s2 + i3 * m.o_s3;
        let av = a[ai];
        let mut bv = scalar;
        if m.op < 15 {
            bv = b[bi];
        }
        let mut v = 0.0;
        if m.op == 9 || m.op == 15 {
            v = av + bv;
        } else if m.op == 10 || m.op == 16 {
            v = av - bv;
        } else if m.op == 11 || m.op == 17 {
            v = av * bv;
        } else if m.op == 12 || m.op == 18 {
            v = av / bv;
        } else if m.op == 13 || m.op == 19 {
            v = goldy::gpu::min(av, bv);
        } else if m.op == 14 || m.op == 20 {
            v = goldy::gpu::max(av, bv);
        }
        dst[di] = v;
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_copy_u32(src: &[u32], dst: goldy::gpu::Scattered<u32>, meta: &[TensorOpMeta], bits: u32) {
    let i = goldy::gpu::global_id().x;
    let m: TensorOpMeta = meta[0];
    if i < m.o_numel {
        let mut rest = i;
        let i3 = rest % m.o_d3;
        rest = rest / m.o_d3;
        let i2 = rest % m.o_d2;
        rest = rest / m.o_d2;
        let i1 = rest % m.o_d1;
        rest = rest / m.o_d1;
        let i0 = rest % m.o_d0;
        let si = m.a_off + i0 * m.a_s0 + i1 * m.a_s1 + i2 * m.a_s2 + i3 * m.a_s3;
        let di = m.o_off + i0 * m.o_s0 + i1 * m.o_s1 + i2 * m.o_s2 + i3 * m.o_s3;
        if m.op == 1 {
            dst[di] = bits;
        } else {
            dst[di] = src[si];
        }
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_cast_f32_i32(src: &[f32], dst: goldy::gpu::Scattered<i32>, meta: &[TensorOpMeta]) {
    let i = goldy::gpu::global_id().x;
    let m: TensorOpMeta = meta[0];
    if i < m.o_numel {
        let mut rest = i;
        let i3 = rest % m.o_d3;
        rest = rest / m.o_d3;
        let i2 = rest % m.o_d2;
        rest = rest / m.o_d2;
        let i1 = rest % m.o_d1;
        rest = rest / m.o_d1;
        let i0 = rest % m.o_d0;
        let si = m.a_off + i0 * m.a_s0 + i1 * m.a_s1 + i2 * m.a_s2 + i3 * m.a_s3;
        let di = m.o_off + i0 * m.o_s0 + i1 * m.o_s1 + i2 * m.o_s2 + i3 * m.o_s3;
        dst[di] = src[si] as i32;
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_cast_f32_u32(src: &[f32], dst: goldy::gpu::Scattered<u32>, meta: &[TensorOpMeta]) {
    let i = goldy::gpu::global_id().x;
    let m: TensorOpMeta = meta[0];
    if i < m.o_numel {
        let mut rest = i;
        let i3 = rest % m.o_d3;
        rest = rest / m.o_d3;
        let i2 = rest % m.o_d2;
        rest = rest / m.o_d2;
        let i1 = rest % m.o_d1;
        rest = rest / m.o_d1;
        let i0 = rest % m.o_d0;
        let si = m.a_off + i0 * m.a_s0 + i1 * m.a_s1 + i2 * m.a_s2 + i3 * m.a_s3;
        let di = m.o_off + i0 * m.o_s0 + i1 * m.o_s1 + i2 * m.o_s2 + i3 * m.o_s3;
        dst[di] = src[si] as u32;
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_cast_i32_f32(src: &[i32], dst: goldy::gpu::Scattered<f32>, meta: &[TensorOpMeta]) {
    let i = goldy::gpu::global_id().x;
    let m: TensorOpMeta = meta[0];
    if i < m.o_numel {
        let mut rest = i;
        let i3 = rest % m.o_d3;
        rest = rest / m.o_d3;
        let i2 = rest % m.o_d2;
        rest = rest / m.o_d2;
        let i1 = rest % m.o_d1;
        rest = rest / m.o_d1;
        let i0 = rest % m.o_d0;
        let si = m.a_off + i0 * m.a_s0 + i1 * m.a_s1 + i2 * m.a_s2 + i3 * m.a_s3;
        let di = m.o_off + i0 * m.o_s0 + i1 * m.o_s1 + i2 * m.o_s2 + i3 * m.o_s3;
        dst[di] = src[si] as f32;
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_cast_u32_f32(src: &[u32], dst: goldy::gpu::Scattered<f32>, meta: &[TensorOpMeta]) {
    let i = goldy::gpu::global_id().x;
    let m: TensorOpMeta = meta[0];
    if i < m.o_numel {
        let mut rest = i;
        let i3 = rest % m.o_d3;
        rest = rest / m.o_d3;
        let i2 = rest % m.o_d2;
        rest = rest / m.o_d2;
        let i1 = rest % m.o_d1;
        rest = rest / m.o_d1;
        let i0 = rest % m.o_d0;
        let si = m.a_off + i0 * m.a_s0 + i1 * m.a_s1 + i2 * m.a_s2 + i3 * m.a_s3;
        let di = m.o_off + i0 * m.o_s0 + i1 * m.o_s1 + i2 * m.o_s2 + i3 * m.o_s3;
        dst[di] = src[si] as f32;
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_reduce_f32(src: &[f32], dst: goldy::gpu::Scattered<f32>, meta: &[TensorOpMeta]) {
    let i = goldy::gpu::global_id().x;
    let m: TensorOpMeta = meta[0];
    if i < m.o_numel {
        let mut rest = i;
        let i3 = rest % m.o_d3;
        rest = rest / m.o_d3;
        let i2 = rest % m.o_d2;
        rest = rest / m.o_d2;
        let i1 = rest % m.o_d1;
        rest = rest / m.o_d1;
        let i0 = rest % m.o_d0;
        let mut acc = 0.0;
        if m.op == 22 {
            acc = -3.402823e38;
        } else if m.op == 23 {
            acc = 3.402823e38;
        }
        let mut k = 0u32;
        while k < m.reduce_len {
            let mut s0 = i0;
            let mut s1 = i1;
            let mut s2 = i2;
            let mut s3 = i3;
            if m.axis == 0 {
                s0 = k;
            } else if m.axis == 1 {
                s1 = k;
            } else if m.axis == 2 {
                s2 = k;
            } else {
                s3 = k;
            }
            let si = m.a_off + s0 * m.a_s0 + s1 * m.a_s1 + s2 * m.a_s2 + s3 * m.a_s3;
            let v = src[si];
            if m.op == 21 || m.op == 24 {
                acc = acc + v;
            } else if m.op == 22 {
                acc = goldy::gpu::max(acc, v);
            } else if m.op == 23 {
                acc = goldy::gpu::min(acc, v);
            }
            k = k + 1;
        }
        if m.op == 24 {
            acc = acc / (m.reduce_len as f32);
        }
        let di = m.o_off + i0 * m.o_s0 + i1 * m.o_s1 + i2 * m.o_s2 + i3 * m.o_s3;
        dst[di] = acc;
    }
}

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_gather_f32(src: &[f32], index: &[i32], dst: goldy::gpu::Scattered<f32>, meta: &[TensorOpMeta]) {
    let i = goldy::gpu::global_id().x;
    let m: TensorOpMeta = meta[0];
    if i < m.o_numel {
        let mut rest = i;
        let i3 = rest % m.o_d3;
        rest = rest / m.o_d3;
        let i2 = rest % m.o_d2;
        rest = rest / m.o_d2;
        let i1 = rest % m.o_d1;
        rest = rest / m.o_d1;
        let i0 = rest % m.o_d0;
        let ii = m.b_off + i0 * m.b_s0 + i1 * m.b_s1 + i2 * m.b_s2 + i3 * m.b_s3;
        let g = index[ii] as u32;
        let mut s0 = i0;
        let mut s1 = i1;
        let mut s2 = i2;
        let mut s3 = i3;
        if m.axis == 0 {
            s0 = g;
        } else if m.axis == 1 {
            s1 = g;
        } else if m.axis == 2 {
            s2 = g;
        } else {
            s3 = g;
        }
        let si = m.a_off + s0 * m.a_s0 + s1 * m.a_s1 + s2 * m.a_s2 + s3 * m.a_s3;
        let di = m.o_off + i0 * m.o_s0 + i1 * m.o_s1 + i2 * m.o_s2 + i3 * m.o_s3;
        dst[di] = src[si];
    }
}

/// Collision-safe scatter: one thread walks every index so add/min/max are defined.
#[goldy::compute(workgroup_size = [1, 1, 1])]
fn tensor_scatter_f32(src: &[f32], index: &[i32], dst: goldy::gpu::Scattered<f32>, meta: &[TensorOpMeta]) {
    let lid = goldy::gpu::global_id().x;
    if lid != 0 {
        return;
    }
    let m: TensorOpMeta = meta[0];
    let mut i = 0u32;
    while i < m.a_numel {
        let mut rest = i;
        let i3 = rest % m.a_d3;
        rest = rest / m.a_d3;
        let i2 = rest % m.a_d2;
        rest = rest / m.a_d2;
        let i1 = rest % m.a_d1;
        rest = rest / m.a_d1;
        let i0 = rest % m.a_d0;
        let si = m.a_off + i0 * m.a_s0 + i1 * m.a_s1 + i2 * m.a_s2 + i3 * m.a_s3;
        let ii = m.b_off + i0 * m.b_s0 + i1 * m.b_s1 + i2 * m.b_s2 + i3 * m.b_s3;
        let g = index[ii] as u32;
        let mut d0 = i0;
        let mut d1 = i1;
        let mut d2 = i2;
        let mut d3 = i3;
        if m.axis == 0 {
            d0 = g;
        } else if m.axis == 1 {
            d1 = g;
        } else if m.axis == 2 {
            d2 = g;
        } else {
            d3 = g;
        }
        let di = m.o_off + d0 * m.o_s0 + d1 * m.o_s1 + d2 * m.o_s2 + d3 * m.o_s3;
        let sv = src[si];
        if m.op == 26 {
            dst[di] = sv;
        } else if m.op == 27 {
            dst[di] = dst[di] + sv;
        } else if m.op == 28 {
            dst[di] = goldy::gpu::min(dst[di], sv);
        } else if m.op == 29 {
            dst[di] = goldy::gpu::max(dst[di], sv);
        }
        i = i + 1;
    }
}

/// Packed rank-3 GEMM: `C[b,i,j] = A[b,i,p] @ B[b,p,j]`.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn tensor_batched_matmul_f32(a: &[f32], b: &[f32], dst: goldy::gpu::Scattered<f32>, meta: &[TensorOpMeta]) {
    let idx = goldy::gpu::global_id().x;
    let m: TensorOpMeta = meta[0];
    if idx < m.o_numel {
        let n = m.o_d3;
        let kdim = m.reduce_len;
        let i = idx / n;
        let j = idx - i * n;
        let batch = i / m.o_d2;
        let row = i - batch * m.o_d2;
        let mut acc = 0.0;
        let mut p = 0u32;
        while p < kdim {
            let av = a[m.a_off + batch * m.a_s0 + row * m.a_s1 + p * m.a_s2];
            let bv = b[m.b_off + batch * m.b_s0 + p * m.b_s1 + j * m.b_s2];
            acc = acc + av * bv;
            p = p + 1;
        }
        dst[m.o_off + batch * m.o_s0 + row * m.o_s1 + j * m.o_s2] = acc;
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
