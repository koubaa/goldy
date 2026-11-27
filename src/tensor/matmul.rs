//! Tensor matmul: checked views lowered to the semantic [`crate::ops::MatMulDesc`] node.

use super::dtype::TensorDType;
use super::ops::TensorRecorder;
use super::shape::TensorShape;
use super::view::{Tensor, TensorView};
use crate::error::GoldyError;
use crate::ops::{MatMulBlock, MatMulDType, MatMulDesc, MatMulIndex, MatMulView, PackedElem, PackedScale};

/// Block-scaled matrix. Codes and scales stay packed. `global`, when set, is an
/// f32 vector: length 1 broadcasts, and a longer vector is indexed by slab id.
pub struct BlockScaled<'a> {
    pub codes: TensorView<'a>,
    pub scales: TensorView<'a>,
    pub elems: PackedElem,
    pub scale: PackedScale,
    pub block: [u32; 2],
    pub global: Option<TensorView<'a>>,
}

impl<'a> TensorRecorder<'a> {
    /// `out = a @ b` with broadcasting on batch dims. Rank-1/2 packed cases use
    /// [`Scheme::matmul`](crate::Scheme::matmul); remaining layouts use a portable kernel.
    pub fn matmul(&mut self, label: &str, a: TensorView<'_>, b: TensorView<'_>) -> Result<Tensor, GoldyError> {
        a.dtype().require_f32("matmul")?;
        b.dtype().require_f32("matmul")?;
        let out_shape = matmul_out_shape(a.shape(), b.shape())?;
        let out = Tensor::zeros(&self.kernels.runtime, out_shape, TensorDType::F32)?;
        self.matmul_into(label, a, b, out.view())?;
        Ok(out)
    }

    pub fn matmul_into(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        a.dtype().require_f32("matmul")?;
        b.dtype().require_f32("matmul")?;
        out.dtype().require_f32("matmul")?;
        out.layout().require_writeable("matmul")?;
        let want = matmul_out_shape(a.shape(), b.shape())?;
        if out.shape() != want {
            return Err(GoldyError::Validation(format!(
                "tensor matmul: output shape {:?} != {:?}",
                out.shape().dims(),
                want.dims()
            )));
        }

        let ra = a.shape().rank();
        let rb = b.shape().rank();
        if ra <= 2 && rb <= 2 {
            return self.matmul_semantic(label, a, b, out);
        }
        if ra == 3 && rb == 3 {
            return self.matmul_batched(label, a, b, out);
        }
        Err(GoldyError::Validation(format!(
            "tensor matmul: ranks {} and {} are not supported (use rank 1–3)",
            ra, rb
        )))
    }

    /// `out = weight[ids[slot]] @ x`.
    ///
    /// `weight` is a packed `[slabs, rows, k]` tensor of F32 or BF16. `indices` is
    /// a rank-1 `u32` buffer. The GEMV reads the slab in place (`index * rows * k`);
    /// it does not copy a tile. An index `>= slabs` yields a zero row (masked gather).
    /// `x` is `[k]` and `out` is `[rows]`. View offsets are part of the address.
    pub fn matmul_indexed_into(
        &mut self,
        label: &str,
        weight: TensorView<'_>,
        x: TensorView<'_>,
        ids: TensorView<'_>,
        slot: u32,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        x.dtype().require_f32("matmul_indexed")?;
        out.dtype().require_f32("matmul_indexed")?;
        if weight.shape().rank() != 3 || !weight.is_contiguous() {
            return Err(GoldyError::Validation(
                "matmul_indexed: weight must be a packed [slabs, rows, k] tensor".into(),
            ));
        }
        if ids.dtype() != TensorDType::U32 || ids.shape().rank() != 1 {
            return Err(GoldyError::Validation(
                "matmul_indexed: indices must be a rank-1 U32 tensor".into(),
            ));
        }
        self.matmul_indexed_batch_into(label, weight, x, ids, slot, 1, 0, out)
    }

    /// `count` ids starting at `slot`. `out` is `[count * rows]`.
    ///
    /// `x_stride == 0` uses one `x` of length `k` for every id. Otherwise `x`
    /// holds `count` vectors and id `i` reads at `i * x_stride`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_indexed_batch_into(
        &mut self,
        label: &str,
        weight: TensorView<'_>,
        x: TensorView<'_>,
        ids: TensorView<'_>,
        slot: u32,
        count: u32,
        x_stride: u32,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        x.dtype().require_f32("matmul_indexed")?;
        out.dtype().require_f32("matmul_indexed")?;
        if count == 0 {
            return Err(GoldyError::Validation("matmul_indexed: count is 0".into()));
        }
        if weight.shape().rank() != 3 || !weight.is_contiguous() {
            return Err(GoldyError::Validation(
                "matmul_indexed: weight must be a packed [slabs, rows, k] tensor".into(),
            ));
        }
        if ids.dtype() != TensorDType::U32 || ids.shape().rank() != 1 {
            return Err(GoldyError::Validation(
                "matmul_indexed: indices must be a rank-1 U32 tensor".into(),
            ));
        }
        let index_len = ids.shape().dim(0)?;
        let end = slot
            .checked_add(count)
            .ok_or_else(|| GoldyError::Validation("matmul_indexed: first index plus batch overflows u32".into()))?;
        if slot >= index_len || end > index_len {
            return Err(GoldyError::Validation(
                "matmul_indexed: first index and batch run past the index vector".into(),
            ));
        }
        let slabs = weight.shape().dim(0)?;
        let rows = weight.shape().dim(1)?;
        let k = weight.shape().dim(2)?;
        let x_need = if x_stride == 0 {
            u64::from(k)
        } else {
            u64::from(count - 1) * u64::from(x_stride) + u64::from(k)
        };
        let x_ok = x.shape().rank() == 1 && u64::from(x.shape().dim(0)?) >= x_need;
        let out_len = rows.saturating_mul(count);
        if !x_ok || out.shape().dims() != [out_len] {
            return Err(GoldyError::Validation(
                "matmul_indexed: x matches the stride and out is [count * rows]".into(),
            ));
        }
        let elems = (slabs as u64).saturating_mul(rows as u64).saturating_mul(k as u64);
        if elems > u64::from(u32::MAX) {
            return Err(GoldyError::Validation(
                "matmul_indexed: weight does not fit a u32 element index".into(),
            ));
        }
        let dtype = match weight.dtype() {
            TensorDType::F32 => MatMulDType::F32,
            TensorDType::BF16 if count == 1 && x_stride == 0 => MatMulDType::BF16,
            TensorDType::BF16 => {
                return Err(GoldyError::Validation(
                    "matmul_indexed: a run of slab indices is an f32 weight".into(),
                ))
            }
            _ => {
                return Err(GoldyError::Validation(
                    "matmul_indexed: weight must be F32 or BF16".into(),
                ))
            }
        };
        let mut desc = MatMulDesc::gemv(rows, k);
        desc.dtype = dtype;
        desc.index = Some(MatMulIndex {
            first: slot,
            slabs,
            slab_stride: rows.saturating_mul(k),
            global_per_slab: false,
            batch: count,
            x_stride,
        });
        let a_buf = if dtype == MatMulDType::BF16 {
            weight.packed_u32_words()?.buffer()
        } else {
            weight.buffer()
        };
        let a_view = MatMulView::strided(weight.storage_offset(), k);
        self.scheme
            .matmul(label, desc)
            .a(a_buf, a_view)
            .b(x.buffer(), vector_matmul_view(x)?)
            .out(out.buffer(), vector_matmul_view(out)?)
            .extra(ids.buffer(), vector_matmul_view(ids)?)
            .record();
        Ok(())
    }

    /// `out = cap * tanh(x / cap)`. `cap == 1` records a single `tanh`.
    ///
    /// Otherwise this records divide, tanh, and multiply. The returned tensors
    /// are those intermediates; keep them alive for every submit.
    pub fn tanh_scaled_into(
        &mut self,
        label: &str,
        x: TensorView<'_>,
        cap: f32,
        out: TensorView<'_>,
    ) -> Result<Vec<Tensor>, GoldyError> {
        if cap == 1.0 {
            self.tanh_into(label, x, out)?;
            return Ok(Vec::new());
        }
        let scaled = self.div_scalar(&format!("{label}_div"), x, cap)?;
        let curved = self.tanh(&format!("{label}_tanh"), scaled.view())?;
        self.mul_scalar_into(label, curved.view(), cap, out)?;
        Ok(vec![scaled, curved])
    }

    /// `out = weight @ x` for a packed BF16 matrix. Recorded as a [`MatMulDesc`]
    /// with [`MatMulDType::BF16`], so a backend can select a native GEMM later.
    /// The stdlib fallback promotes each element inside the dot.
    pub fn matmul_bf16_into(
        &mut self,
        label: &str,
        weight: TensorView<'_>,
        x: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        if weight.dtype() != TensorDType::BF16 || weight.shape().rank() != 2 {
            return Err(GoldyError::Validation(
                "matmul_bf16: weight must be a rank-2 BF16 tensor".into(),
            ));
        }
        x.dtype().require_f32("matmul_bf16")?;
        out.dtype().require_f32("matmul_bf16")?;
        let rows = weight.shape().dim(0)?;
        let k = weight.shape().dim(1)?;
        if !weight.is_contiguous() {
            return Err(GoldyError::Validation("matmul_bf16: weight must be contiguous".into()));
        }
        if x.shape().rank() > 2 || out.shape().dims() != matmul_out_shape(weight.shape(), x.shape())?.dims() {
            return Err(GoldyError::Validation(
                "matmul_bf16: x is [k] or [k, n] and out matches weight @ x".into(),
            ));
        }
        let n = if x.shape().rank() == 1 { 1 } else { x.shape().dim(1)? };
        let mut desc = MatMulDesc::gemm(rows, n, k);
        desc.dtype = MatMulDType::BF16;
        let bits = weight.packed_u32_words()?;
        self.scheme
            .matmul(label, desc)
            .a(bits.buffer(), MatMulView::packed())
            .b(x.buffer(), MatMulView::packed())
            .out(out.buffer(), MatMulView::packed())
            .extra(self.kernels.unit_u32.buffer(), MatMulView::packed())
            .record();
        Ok(())
    }

    /// `out = dequant(weight) @ x` with `x` a vector. Recorded as a block-scaled
    /// [`MatMulDesc`]. `ids` selects a slab when `codes` is rank 3.
    pub fn matmul_block_scaled_into(
        &mut self,
        label: &str,
        weight: BlockScaled<'_>,
        x: TensorView<'_>,
        ids: Option<(TensorView<'_>, u32)>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        x.dtype().require_f32("matmul_block")?;
        out.dtype().require_f32("matmul_block")?;
        let codes = weight.codes;
        let scales = weight.scales;
        if !codes.is_contiguous() || !scales.is_contiguous() {
            return Err(GoldyError::Validation(
                "matmul_block: codes and scales must be contiguous".into(),
            ));
        }
        let rank = codes.shape().rank();
        if rank != 2 && rank != 3 {
            return Err(GoldyError::Validation(
                "matmul_block: codes are [rows, packed] or [slabs, rows, packed]".into(),
            ));
        }
        let (slabs, rows, packed) = if rank == 2 {
            (1, codes.shape().dim(0)?, codes.shape().dim(1)?)
        } else {
            (codes.shape().dim(0)?, codes.shape().dim(1)?, codes.shape().dim(2)?)
        };
        let per = match weight.elems {
            PackedElem::E2M1 => 2,
            PackedElem::E4M3 => 1,
        };
        let k = packed.saturating_mul(per);
        if x.shape().dims() != [k] || out.shape().dims() != [rows] {
            return Err(GoldyError::Validation(
                "matmul_block: x is [k] and out is [rows]".into(),
            ));
        }
        let scale_cols = k / weight.block[1];
        let scale_rows = rows / weight.block[0];
        let expect_scale = if rank == 2 {
            vec![scale_rows, scale_cols]
        } else {
            vec![slabs, scale_rows, scale_cols]
        };
        if scales.dtype()
            != match weight.scale {
                PackedScale::E4M3 => TensorDType::F8E4M3,
                PackedScale::E8M0 => TensorDType::U8,
            }
            || scales.shape().dims() != expect_scale
        {
            return Err(GoldyError::Validation(
                "matmul_block: scales do not match the block grid".into(),
            ));
        }
        let code_dtype = match weight.elems {
            PackedElem::E2M1 | PackedElem::E4M3 => TensorDType::U8,
        };
        if codes.dtype() != code_dtype && !(weight.elems == PackedElem::E4M3 && codes.dtype() == TensorDType::F8E4M3) {
            return Err(GoldyError::Validation(
                "matmul_block: codes must be U8, or F8E4M3 when the elements are E4M3".into(),
            ));
        }
        let mut index = None;
        let ids_buf = if let Some((ids, slot)) = ids {
            if rank != 3 {
                return Err(GoldyError::Validation(
                    "matmul_block: an index selects a rank-3 code tensor".into(),
                ));
            }
            if ids.dtype() != TensorDType::U32 || slot >= ids.shape().dim(0)? {
                return Err(GoldyError::Validation(
                    "matmul_block: indices are a rank-1 U32 tensor and the first index is in range".into(),
                ));
            }
            index = Some(MatMulIndex {
                first: slot,
                slabs,
                slab_stride: rows,
                global_per_slab: weight.global.is_some_and(|g| g.shape().dim(0).unwrap_or(1) != 1),
                batch: 1,
                x_stride: 0,
            });
            ids.buffer()
        } else {
            self.kernels.unit_u32.buffer()
        };
        let global_buf = if let Some(global) = weight.global {
            global.dtype().require_f32("matmul_block")?;
            global.buffer()
        } else {
            self.kernels.unit_f32.buffer()
        };
        let mut desc = MatMulDesc::gemv(rows, k);
        desc.dtype = MatMulDType::BlockScaled;
        desc.block = Some(MatMulBlock {
            elems: weight.elems,
            scale: weight.scale,
            block_rows: weight.block[0],
            block_cols: weight.block[1],
        });
        desc.index = index;
        let code_bits = codes.packed_u32_words()?;
        let scale_bits = scales.packed_u32_words()?;
        self.scheme
            .matmul(label, desc)
            .a(code_bits.buffer(), MatMulView::packed())
            .b(x.buffer(), MatMulView::packed())
            .out(out.buffer(), MatMulView::packed())
            .extra(scale_bits.buffer(), MatMulView::packed())
            .extra(global_buf, MatMulView::packed())
            .extra(ids_buf, MatMulView::packed())
            .record();
        Ok(())
    }

    fn matmul_semantic(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let g = gemm_views(a, b)?;
        let desc = {
            let mut d = MatMulDesc::gemm(g.m, g.n, g.k);
            d.transpose_a = g.transpose_a;
            d.transpose_b = g.transpose_b;
            d
        };
        let c_view = packed_or_strided(out, false)?;
        self.scheme
            .matmul(label, desc)
            .a(a.buffer(), g.a)
            .b(b.buffer(), g.b)
            .out(out.buffer(), c_view)
            .record();
        Ok(())
    }

    fn matmul_batched(
        &mut self,
        label: &str,
        a: TensorView<'_>,
        b: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let ashape = a.shape();
        let bshape = b.shape();
        let oshape = out.shape();
        let ad = ashape.dims();
        let bd = bshape.dims();
        if ad[0] != bd[0] || ad[0] != oshape.dims()[0] {
            return Err(GoldyError::Validation(
                "tensor matmul: batch dimensions must match".into(),
            ));
        }
        if ad[2] != bd[1] {
            return Err(GoldyError::Validation(
                "tensor matmul: inner dimensions must match".into(),
            ));
        }
        let batch = ad[0];
        // Prefer looping semantic GEMM for packed slices so cuBLAS/MPS still run.
        if a.is_contiguous() && b.is_contiguous() && out.is_contiguous() {
            for bi in 0..batch {
                let av = a.narrow(0, bi, 1)?.reshape(&[ad[1], ad[2]])?;
                let bv = b.narrow(0, bi, 1)?.reshape(&[bd[1], bd[2]])?;
                let ov = out.narrow(0, bi, 1)?.reshape(&[ad[1], bd[2]])?;
                self.matmul_semantic(&format!("{label}_b{bi}"), av, bv, ov)?;
            }
            return Ok(());
        }
        self.kernels
            .ops
            .batched
            .record(self.scheme, label, a, b, out)?
            .over_1d(out.numel_u32().max(1));
        Ok(())
    }
}

fn matmul_out_shape(a: TensorShape, b: TensorShape) -> Result<TensorShape, GoldyError> {
    match (a.rank(), b.rank()) {
        (1, 1) => {
            if a.dims()[0] != b.dims()[0] {
                return Err(GoldyError::Validation("tensor matmul: 1D inner dims mismatch".into()));
            }
            TensorShape::from_dims(&[])
        }
        (2, 1) => {
            if a.dims()[1] != b.dims()[0] {
                return Err(GoldyError::Validation("tensor matmul: GEMV inner dims mismatch".into()));
            }
            Ok(TensorShape::vector(a.dims()[0]))
        }
        (1, 2) => {
            if a.dims()[0] != b.dims()[0] {
                return Err(GoldyError::Validation(
                    "tensor matmul: vec-mat inner dims mismatch".into(),
                ));
            }
            Ok(TensorShape::vector(b.dims()[1]))
        }
        (2, 2) => {
            if a.dims()[1] != b.dims()[0] {
                return Err(GoldyError::Validation("tensor matmul: GEMM inner dims mismatch".into()));
            }
            Ok(TensorShape::matrix(a.dims()[0], b.dims()[1]))
        }
        (3, 3) => {
            if a.dims()[0] != b.dims()[0] || a.dims()[2] != b.dims()[1] {
                return Err(GoldyError::Validation(
                    "tensor matmul: batched GEMM dims mismatch".into(),
                ));
            }
            TensorShape::from_dims(&[a.dims()[0], a.dims()[1], b.dims()[2]])
        }
        (ra, rb) => Err(GoldyError::Validation(format!(
            "tensor matmul: ranks {ra} and {rb} are not supported"
        ))),
    }
}

struct GemmViews {
    m: u32,
    n: u32,
    k: u32,
    a: MatMulView,
    b: MatMulView,
    transpose_a: bool,
    transpose_b: bool,
}

fn gemm_views(a: TensorView<'_>, b: TensorView<'_>) -> Result<GemmViews, GoldyError> {
    let (m, k, a_view, ta) = matrix_operand(a, true)?;
    let (k2, n, b_view, tb) = matrix_operand(b, false)?;
    if k != k2 {
        return Err(GoldyError::Validation(
            "tensor matmul: inner dimensions must match".into(),
        ));
    }
    Ok(GemmViews {
        m,
        n,
        k,
        a: a_view,
        b: b_view,
        transpose_a: ta,
        transpose_b: tb,
    })
}

fn matrix_operand(v: TensorView<'_>, is_a: bool) -> Result<(u32, u32, MatMulView, bool), GoldyError> {
    match v.shape().rank() {
        0 => Err(GoldyError::Validation("tensor matmul: rank-0 operand".into())),
        1 => {
            let vshape = v.shape();
            let n = vshape.dims()[0];
            if is_a {
                Ok((1, n, packed_or_strided(v, true)?, false))
            } else {
                Ok((n, 1, packed_or_strided(v, false)?, false))
            }
        }
        2 => {
            let vshape = v.shape();
            let rows = vshape.dims()[0];
            let cols = vshape.dims()[1];
            let layout = v.layout();
            let s0 = layout.strides()[0];
            let s1 = layout.strides()[1];
            if s1 == 1 {
                let ld = u32::try_from(s0.max(1)).unwrap_or(1);
                Ok((rows, cols, MatMulView::strided(v.storage_offset(), ld), false))
            } else if s0 == 1 {
                let ld = u32::try_from(s1.max(1)).unwrap_or(1);
                Ok((rows, cols, MatMulView::strided(v.storage_offset(), ld), true))
            } else {
                Err(GoldyError::Validation(
                    "tensor matmul: operand is not row-major or column-major; call contiguous() first".into(),
                ))
            }
        }
        _ => Err(GoldyError::Validation("tensor matmul: expected rank 1 or 2".into())),
    }
}

fn vector_matmul_view(v: TensorView<'_>) -> Result<MatMulView, GoldyError> {
    let stride = v.layout().strides().first().copied().unwrap_or(1);
    if stride == 1 {
        return Ok(MatMulView::offset(v.storage_offset()));
    }
    let ld = u32::try_from(stride.max(1))
        .map_err(|_| GoldyError::Validation("matmul_indexed: view stride does not fit u32".into()))?;
    Ok(MatMulView::strided(v.storage_offset(), ld))
}

fn packed_or_strided(v: TensorView<'_>, _as_row: bool) -> Result<MatMulView, GoldyError> {
    if v.shape().rank() == 1 {
        return Ok(MatMulView::offset(v.storage_offset()));
    }
    if v.is_contiguous() {
        return Ok(MatMulView::offset(v.storage_offset()));
    }
    let layout = v.layout();
    let s0 = layout.strides().first().copied().unwrap_or(1);
    let ld = u32::try_from(s0.max(1)).unwrap_or(1);
    Ok(MatMulView::strided(v.storage_offset(), ld))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(dims: &[u32]) -> TensorShape {
        TensorShape::from_dims(dims).unwrap()
    }

    #[test]
    fn gemv_out_shape() {
        let out = matmul_out_shape(shape(&[3, 4]), shape(&[4])).unwrap();
        assert_eq!(out.dims(), &[3]);
    }

    #[test]
    fn gemm_out_shape() {
        let out = matmul_out_shape(shape(&[2, 3]), shape(&[3, 5])).unwrap();
        assert_eq!(out.dims(), &[2, 5]);
    }

    #[test]
    fn batched_out_shape() {
        let out = matmul_out_shape(shape(&[4, 2, 3]), shape(&[4, 3, 1])).unwrap();
        assert_eq!(out.dims(), &[4, 2, 1]);
    }

    #[test]
    fn inner_dim_mismatch() {
        assert!(matmul_out_shape(shape(&[2, 3]), shape(&[4, 5])).is_err());
    }
}
