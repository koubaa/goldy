//! cuBLAS realization of [`crate::backend::GpuCommand::MatMul`].

use super::pending_submit::{bake_device_ptr, CudaOp};
use super::CudaBackend;
use crate::ops::matmul::MatMulOperand;
use crate::ops::MatMulDesc;
use anyhow::{Context, Result};
use cudarc::cublas::{result as cublas, sys as cublas_sys};
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::{Arc, Mutex};

pub(super) struct CublasHandle {
    handle: cublas_sys::cublasHandle_t,
}

// SAFETY: cuBLAS handles are documented as usable from one stream at a time; Goldy
// serializes submits per context onto the submission worker.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let handle = cublas::create_handle().context("CUDA: cublasCreate failed")?;
        unsafe { cublas::set_stream(handle, stream.cu_stream() as _) }.context("CUDA: cublasSetStream failed")?;
        Ok(Self { handle })
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = cublas::destroy_handle(self.handle);
        }
    }
}

pub(super) fn materialize(
    backend: &mut CudaBackend,
    ctx: crate::backend::ContextHandle,
    stream: &Arc<CudaStream>,
    label: Option<crate::SchemeLabel>,
    desc: MatMulDesc,
    a: MatMulOperand,
    b: MatMulOperand,
    c: MatMulOperand,
) -> Result<CudaOp> {
    let handle = if let Some(existing) = backend.cublas.get(&ctx) {
        Arc::clone(existing)
    } else {
        let created = Arc::new(CublasHandle::new(stream)?);
        backend.cublas.insert(ctx, Arc::clone(&created));
        created
    };
    let (a_mem, a_ptr, _) = operand_ptr(backend, stream, &a, "A", false)?;
    let (b_mem, b_ptr, _) = operand_ptr(backend, stream, &b, "B", false)?;
    let (c_mem, c_ptr, _) = operand_ptr(backend, stream, &c, "C", true)?;
    Ok(CudaOp::MatMul {
        label,
        desc,
        a: BlasOperand {
            memory: a_mem,
            device_ptr: a_ptr,
            leading_dim: a.leading_dim,
        },
        b: BlasOperand {
            memory: b_mem,
            device_ptr: b_ptr,
            leading_dim: b.leading_dim,
        },
        c: BlasOperand {
            memory: c_mem,
            device_ptr: c_ptr,
            leading_dim: c.leading_dim,
        },
        handle,
    })
}

fn operand_ptr(
    backend: &mut CudaBackend,
    stream: &Arc<CudaStream>,
    operand: &MatMulOperand,
    name: &str,
    writes: bool,
) -> Result<(Arc<Mutex<CudaSlice<u8>>>, u64, u64)> {
    let buffer = backend
        .buffers
        .get_mut(&operand.buffer)
        .with_context(|| format!("CUDA: MatMul invalid {name} buffer"))?;
    if writes {
        buffer.bump_content_epoch();
    }
    let memory = Arc::clone(buffer.memory_arc()?);
    let byte_off = operand
        .offset_elements
        .checked_mul(4)
        .context("CUDA: MatMul element offset overflow")?;
    let abs = buffer.offset + byte_off;
    let ptr = bake_device_ptr(stream, &memory, abs);
    Ok((memory, ptr, abs))
}

#[derive(Clone)]
pub(super) struct BlasOperand {
    pub memory: Arc<Mutex<CudaSlice<u8>>>,
    pub device_ptr: u64,
    pub leading_dim: u32,
}

pub(super) fn execute(op: &CudaOp) -> Result<()> {
    let CudaOp::MatMul {
        desc,
        a,
        b,
        c,
        handle,
        label,
        ..
    } = op
    else {
        anyhow::bail!("CUDA: expected MatMul op");
    };
    let name = label.as_deref().unwrap_or("matmul");
    if desc.dtype != crate::ops::MatMulDType::F32 {
        anyhow::bail!("CUDA: MatMul `{name}` only supports F32");
    }
    run_f32(handle.handle, desc, a, b, c).with_context(|| format!("CUDA: cuBLAS MatMul `{name}`"))
}

fn run_f32(
    handle: cublas_sys::cublasHandle_t,
    desc: &MatMulDesc,
    a: &BlasOperand,
    b: &BlasOperand,
    c: &BlasOperand,
) -> Result<()> {
    let m = i32::try_from(desc.m).context("m")?;
    let n = i32::try_from(desc.n).context("n")?;
    let k = i32::try_from(desc.k).context("k")?;
    let lda = i32::try_from(a.leading_dim).context("lda")?;
    let ldb = i32::try_from(b.leading_dim).context("ldb")?;
    let ldc = i32::try_from(c.leading_dim).context("ldc")?;
    let alpha = desc.alpha;
    let beta = desc.beta;
    let ap = a.device_ptr as *const f32;
    let bp = b.device_ptr as *const f32;
    let cp = c.device_ptr as *mut f32;
    let n_op = cublas_sys::cublasOperation_t::CUBLAS_OP_N;
    let t_op = cublas_sys::cublasOperation_t::CUBLAS_OP_T;

    // Row-major C = op(A) @ op(B) via column-major BLAS on the transposes:
    // Cᵀ = op(B)ᵀ @ op(A)ᵀ.
    if n == 1 && !desc.transpose_b {
        let trans = if desc.transpose_a { n_op } else { t_op };
        let (cm, cn) = if desc.transpose_a { (m, k) } else { (k, m) };
        unsafe { cublas::sgemv(handle, trans, cm, cn, &alpha, ap, lda, bp, ldb, &beta, cp, ldc) }
            .context("cublasSgemv")?;
        return Ok(());
    }

    let trans_a = if desc.transpose_a { t_op } else { n_op };
    let trans_b = if desc.transpose_b { t_op } else { n_op };
    unsafe {
        cublas::sgemm(
            handle, trans_b, trans_a, n, m, k, &alpha, bp, ldb, ap, lda, &beta, cp, ldc,
        )
    }
    .context("cublasSgemm")?;
    Ok(())
}
