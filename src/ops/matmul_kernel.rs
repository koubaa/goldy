//! Portable FP32 GEMM/GEMV fallback (`C = op(A) @ op(B)`, alpha = 1, beta = 0).

#![allow(clippy::too_many_arguments)]

/// Packed flags: bit 0 = transpose A, bit 1 = transpose B.
pub const FLAG_TRANSPOSE_A: u32 = 1;
pub const FLAG_TRANSPOSE_B: u32 = 2;

/// Row-major `C[m, n] = op(A)[m, k] @ op(B)[k, n]` with optional static element offsets.
///
/// Leading dimensions are packed: `lda = k` (or `m` if A is transposed), `ldb = n`
/// (or `k` if B is transposed), `ldc = n`. Custom leading dims are native-only.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn matmul_f32(
    a: &[f32],
    b: &[f32],
    c: goldy::gpu::Scattered<f32>,
    m: u32,
    n: u32,
    k: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    flags: u32,
) {
    let idx = goldy::gpu::global_id().x;
    let total = m * n;
    if idx < total {
        let i = idx / n;
        let j = idx - i * n;
        let mut acc = 0.0;
        for p in 0..k {
            let mut av = 0.0;
            if (flags & 1) == 1 {
                av = a[a_off + p * m + i];
            } else {
                av = a[a_off + i * k + p];
            }
            let mut bv = 0.0;
            if (flags & 2) == 2 {
                bv = b[b_off + j * k + p];
            } else {
                bv = b[b_off + p * n + j];
            }
            acc = acc + av * bv;
        }
        c[c_off + i * n + j] = acc;
    }
}

/// Rows reduced by one [`gemv_f32`] workgroup: 128 threads, 32 lanes per row.
pub const GEMV_ROWS_PER_GROUP: u32 = 4;

/// The association in which [`gemv_f32`] sums a row: 32 lanes with two strided
/// accumulators each, then a tree over the lanes.
pub(crate) const GEMV_ORDER: goldy_shader_ir::algebra::ReduceOrder = goldy_shader_ir::algebra::ReduceOrder::Lanes {
    lanes: 32,
    accumulators: 2,
};

/// Row-major `y[i] = sum_j A[i, j] * x[j]` with explicit leading dimension and strides.
///
/// Each row is reduced by 32 lanes reading consecutive columns, so loads coalesce and
/// there is no split-K pass. Four rows share a workgroup.
#[goldy::compute(workgroup_size = [128, 1, 1])]
fn gemv_f32(
    a: &[f32],
    x: &[f32],
    y: goldy::gpu::Scattered<f32>,
    m: u32,
    k: u32,
    lda: u32,
    x_stride: u32,
    y_stride: u32,
    a_off: u32,
    x_off: u32,
    y_off: u32,
) {
    let mut partial = goldy::gpu::workgroup_array::<f32, 128>();
    let local = goldy::gpu::local_id().x;
    let lane = local % 32;
    let row = goldy::gpu::workgroup_id().x * 4 + local / 32;
    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    if row < m {
        let a_row = a_off + row * lda;
        let mut j = lane;
        while j + 32 < k {
            acc0 = acc0 + a[a_row + j] * x[x_off + j * x_stride];
            acc1 = acc1 + a[a_row + j + 32] * x[x_off + (j + 32) * x_stride];
            j = j + 64;
        }
        if j < k {
            acc0 = acc0 + a[a_row + j] * x[x_off + j * x_stride];
        }
    }
    partial[local] = acc0 + acc1;
    goldy::gpu::workgroup_barrier();
    let mut s = 16;
    while s > 0 {
        if lane < s {
            partial[local] = partial[local] + partial[local + s];
        }
        goldy::gpu::workgroup_barrier();
        s = s / 2;
    }
    if lane == 0 {
        if row < m {
            y[y_off + row * y_stride] = partial[local];
        }
    }
}

pub use gemv_f32::Kernel as GemvF32Kernel;
pub use matmul_f32::Kernel as MatMulF32Kernel;
