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

/// `gemv_f32` with `a_off = ids[slot] * stride` read on the device.
///
/// `experts == 0` is not used; an id past `experts` writes zeros and does not
/// read `a`. Every thread takes the same barriers as [`gemv_f32`].
#[goldy::compute(workgroup_size = [128, 1, 1])]
fn gemv_f32_indexed(
    a: &[f32],
    x: &[f32],
    y: goldy::gpu::Scattered<f32>,
    ids: &[u32],
    m: u32,
    k: u32,
    lda: u32,
    stride: u32,
    experts: u32,
    slot: u32,
) {
    let mut partial = goldy::gpu::workgroup_array::<f32, 128>();
    let local = goldy::gpu::local_id().x;
    let lane = local % 32;
    let row = goldy::gpu::workgroup_id().x * 4 + local / 32;
    let id = ids[slot];
    let mut ok = 1u32;
    if id >= experts {
        ok = 0u32;
    }
    let a_off = id * stride;
    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    if ok != 0u32 && row < m {
        let a_row = a_off + row * lda;
        let mut j = lane;
        while j + 32 < k {
            acc0 = acc0 + a[a_row + j] * x[j];
            acc1 = acc1 + a[a_row + j + 32] * x[j + 32];
            j = j + 64;
        }
        if j < k {
            acc0 = acc0 + a[a_row + j] * x[j];
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
    if lane == 0 && row < m {
        let mut value = partial[local];
        if ok == 0u32 {
            value = 0.0;
        }
        y[row] = value;
    }
}

/// Row-major BF16 matrix times an f32 vector. Same 32-lane reduction as [`gemv_f32`].
#[goldy::compute(workgroup_size = [128, 1, 1])]
fn gemv_bf16(
    bits: &[u32],
    x: &[f32],
    y: goldy::gpu::Scattered<f32>,
    ids: &[u32],
    m: u32,
    k: u32,
    a_off: u32,
    stride: u32,
    experts: u32,
    slot: u32,
) {
    let mut partial = goldy::gpu::workgroup_array::<f32, 128>();
    let local = goldy::gpu::local_id().x;
    let lane = local % 32;
    let row = goldy::gpu::workgroup_id().x * 4 + local / 32;
    let mut ok = 1u32;
    let mut base = a_off;
    if experts != 0u32 {
        let id = ids[slot];
        if id >= experts {
            ok = 0u32;
        }
        base = id * stride;
    }
    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    if ok != 0u32 && row < m {
        let a_row = base + row * k;
        let mut j = lane;
        while j + 32 < k {
            let e0 = a_row + j;
            let w0 = bits[e0 >> 1];
            let mut h0 = w0 & 65535u32;
            if (e0 & 1u32) != 0u32 {
                h0 = w0 >> 16;
            }
            let e1 = a_row + j + 32;
            let w1 = bits[e1 >> 1];
            let mut h1 = w1 & 65535u32;
            if (e1 & 1u32) != 0u32 {
                h1 = w1 >> 16;
            }
            acc0 = acc0 + goldy::gpu::as_f32(h0 << 16) * x[j];
            acc1 = acc1 + goldy::gpu::as_f32(h1 << 16) * x[j + 32];
            j = j + 64;
        }
        if j < k {
            let e0 = a_row + j;
            let w0 = bits[e0 >> 1];
            let mut h0 = w0 & 65535u32;
            if (e0 & 1u32) != 0u32 {
                h0 = w0 >> 16;
            }
            acc0 = acc0 + goldy::gpu::as_f32(h0 << 16) * x[j];
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
    if lane == 0 && row < m {
        let mut value = partial[local];
        if ok == 0u32 {
            value = 0.0;
        }
        y[row] = value;
    }
}

/// One thread per output. BF16 `A` is promoted inside the product.
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn gemm_bf16(
    bits: &[u32],
    b: &[f32],
    c: goldy::gpu::Scattered<f32>,
    ids: &[u32],
    m: u32,
    n: u32,
    k: u32,
    a_off: u32,
    stride: u32,
    experts: u32,
    slot: u32,
) {
    let idx = goldy::gpu::global_id().x;
    let total = m * n;
    if idx >= total {
        return;
    }
    let mut ok = 1u32;
    let mut base = a_off;
    if experts != 0u32 {
        let id = ids[slot];
        if id >= experts {
            ok = 0u32;
        }
        base = id * stride;
    }
    let i = idx / n;
    let j = idx - i * n;
    let mut acc = 0.0;
    if ok != 0u32 {
        let mut p = 0u32;
        while p < k {
            let elem = base + i * k + p;
            let word = bits[elem >> 1];
            let mut half = word & 65535u32;
            if (elem & 1u32) != 0u32 {
                half = word >> 16;
            }
            acc = acc + goldy::gpu::as_f32(half << 16) * b[p * n + j];
            p = p + 1;
        }
    }
    c[i * n + j] = acc;
}

/// Block-scaled GEMV. `elem_kind` 0 is E2M1 (even column in the low nibble),
/// 1 is one E4M3 byte per column. `scale_kind` 0 is E4M3, 1 is E8M0.
/// `experts == 0` reads slab 0 at the start of `codes`.
#[goldy::compute(workgroup_size = [128, 1, 1])]
fn gemv_block(
    codes: &[u32],
    x: &[f32],
    y: goldy::gpu::Scattered<f32>,
    scales: &[u32],
    global: &[f32],
    ids: &[u32],
    m: u32,
    k: u32,
    kind: u32,
    block_cols: u32,
    experts: u32,
    slot: u32,
    rows: u32,
    global_per_slab: u32,
) {
    let mut partial = goldy::gpu::workgroup_array::<f32, 128>();
    let mut e2m1 = goldy::gpu::workgroup_array::<f32, 16>();
    let mut scale_s = goldy::gpu::workgroup_array::<f32, 8>();
    let local = goldy::gpu::local_id().x;
    let lane = local % 32;
    let row_in_group = local / 32;
    let row = goldy::gpu::workgroup_id().x * 4 + row_in_group;
    if local < 16u32 {
        let mut mag = 0.0;
        if local == 1u32 || local == 9u32 {
            mag = 0.5;
        } else if local == 2u32 || local == 10u32 {
            mag = 1.0;
        } else if local == 3u32 || local == 11u32 {
            mag = 1.5;
        } else if local == 4u32 || local == 12u32 {
            mag = 2.0;
        } else if local == 5u32 || local == 13u32 {
            mag = 3.0;
        } else if local == 6u32 || local == 14u32 {
            mag = 4.0;
        } else if local == 7u32 || local == 15u32 {
            mag = 6.0;
        }
        if local >= 8u32 {
            mag = -mag;
        }
        e2m1[local] = mag;
    }
    goldy::gpu::workgroup_barrier();
    let mut ok = 1u32;
    let mut slab = 0u32;
    if experts != 0u32 {
        let id = ids[slot];
        if id >= experts {
            ok = 0u32;
        } else {
            slab = id;
        }
    }
    let mut factor = global[0];
    if global_per_slab != 0u32 && ok != 0u32 {
        factor = global[slab];
    }
    let mut acc = 0.0;
    let mut base = 0u32;
    while base < k {
        if lane == 0u32 || lane == 16u32 {
            let mut col = base;
            if lane == 16u32 {
                col = base + 16u32;
            }
            let mut decoded = 0.0;
            if ok != 0u32 && row < m && col < k {
                let scale_cols = k / block_cols;
                let mut srow = row;
                let mut scale_rows = rows;
                if block_cols == 32u32 {
                    srow = row / 32u32;
                    scale_rows = rows / 32u32;
                }
                let scol = col / block_cols;
                let scale_index = (slab * scale_rows + srow) * scale_cols + scol;
                let sword = scales[scale_index >> 2];
                let sshift = (scale_index & 3u32) * 8u32;
                let sb = (sword >> sshift) & 255u32;
                let scale_kind = kind >> 1;
                if scale_kind == 1u32 {
                    decoded = goldy::gpu::as_f32(sb << 23);
                } else {
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
                    decoded = goldy::gpu::as_f32(bits);
                    if sign != 0u32 {
                        decoded = -decoded;
                    }
                }
            }
            if lane == 0u32 {
                scale_s[row_in_group * 2] = decoded;
            }
            if lane == 16u32 {
                scale_s[row_in_group * 2 + 1] = decoded;
            }
        }
        goldy::gpu::workgroup_barrier();
        let col = base + lane;
        if ok != 0u32 && row < m && col < k {
            let mut which = 0u32;
            if lane >= 16u32 {
                which = 1u32;
            }
            let scale = scale_s[row_in_group * 2 + which] * factor;
            let mut value = 0.0;
            if (kind & 1u32) == 0u32 {
                let byte_index = (slab * rows + row) * (k >> 1) + (col >> 1);
                let word = codes[byte_index >> 2];
                let shift = (byte_index & 3u32) * 8u32;
                let byte = (word >> shift) & 255u32;
                let mut nib = byte & 15u32;
                if (col & 1u32) != 0u32 {
                    nib = byte >> 4;
                }
                value = e2m1[nib];
            } else {
                let byte_index = (slab * rows + row) * k + col;
                let word = codes[byte_index >> 2];
                let shift = (byte_index & 3u32) * 8u32;
                let sb = (word >> shift) & 255u32;
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
                value = goldy::gpu::as_f32(bits);
                if sign != 0u32 {
                    value = -value;
                }
            }
            acc = acc + value * scale * x[col];
        }
        goldy::gpu::workgroup_barrier();
        base = base + 32u32;
    }
    partial[local] = acc;
    goldy::gpu::workgroup_barrier();
    let mut s = 16;
    while s > 0 {
        if lane < s {
            partial[local] = partial[local] + partial[local + s];
        }
        goldy::gpu::workgroup_barrier();
        s = s / 2;
    }
    if lane == 0u32 && row < m {
        let mut value = partial[local];
        if ok == 0u32 {
            value = 0.0;
        }
        y[row] = value;
    }
}

pub use gemm_bf16::Kernel as GemmBf16Kernel;
pub use gemv_bf16::Kernel as GemvBf16Kernel;
pub use gemv_block::Kernel as GemvBlockKernel;
pub use gemv_f32::Kernel as GemvF32Kernel;
pub use gemv_f32_indexed::Kernel as GemvF32IndexedKernel;
pub use matmul_f32::Kernel as MatMulF32Kernel;
