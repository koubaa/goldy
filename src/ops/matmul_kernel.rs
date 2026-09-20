//! Portable FP32 GEMM/GEMV fallback (`C = op(A) @ op(B)`, alpha = 1, beta = 0).

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

pub use matmul_f32::Kernel as MatMulF32Kernel;
