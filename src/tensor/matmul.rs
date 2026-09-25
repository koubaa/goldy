//! Tensor matmul: checked views lowered to the semantic [`crate::ops::MatMulDesc`] node.

use super::dtype::TensorDType;
use super::ops::TensorRecorder;
use super::shape::TensorShape;
use super::view::{Tensor, TensorView};
use crate::error::GoldyError;
use crate::ops::{MatMulDesc, MatMulView};

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
