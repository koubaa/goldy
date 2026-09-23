//! Page-locked host staging for uploads and readback.
//!
//! CUDA graphs capture the source pointer of host→device memcpy nodes. Pageable
//! `Vec<u8>` staging forces the driver to bounce through an internal pin and is
//! illegal to capture. CPU-written upload staging is the CUDA analogue of a D3D12
//! UPLOAD heap: write-combined, GPU copies from a stable address on replay.
//! CPU-read readback staging is cacheable page-locked memory: the GPU DMA-writes
//! it, then the host copies out after the producing stream settles.

use anyhow::{Context as _, Result};
use cudarc::driver::{sys, CudaContext};
use std::sync::Arc;

pub(crate) struct CudaPinnedHost {
    ctx: Arc<CudaContext>,
    ptr: *mut u8,
    len: usize,
    flags: u32,
}

// SAFETY: the allocation is process-local CUDA host memory. CPU_WRITABLE staging is
// mutated under the backend lock and read by the submission worker (HtoD). Readback
// staging is written by the worker (DtoH) and read on the API thread after that
// stream has settled.
unsafe impl Send for CudaPinnedHost {}
unsafe impl Sync for CudaPinnedHost {}

impl CudaPinnedHost {
    /// Write-combined pinned host memory for CPU→GPU uploads.
    pub(super) fn alloc(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        Self::alloc_with_flags(ctx, len, sys::CU_MEMHOSTALLOC_WRITECOMBINED as u32)
    }

    /// Cacheable pinned host memory for GPU→CPU readback.
    pub(super) fn alloc_readback(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        Self::alloc_with_flags(ctx, len, 0)
    }

    fn alloc_with_flags(ctx: &Arc<CudaContext>, len: usize, flags: u32) -> Result<Self> {
        let len = len.max(1);
        let _gate = super::capture_gate::lock_capture_alloc_gate();
        ctx.bind_to_thread()
            .context("CUDA: bind context for pinned host alloc")?;
        // SAFETY: `cuMemHostAlloc` returns unset host memory of `len` bytes.
        let ptr = unsafe { cudarc::driver::result::malloc_host(len, flags) }.context("CUDA: cuMemHostAlloc failed")?;
        let ptr = ptr as *mut u8;
        // SAFETY: freshly allocated `len` bytes.
        unsafe { std::ptr::write_bytes(ptr, 0, len) };
        Ok(Self {
            ctx: Arc::clone(ctx),
            ptr,
            len,
            flags,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is a live `cuMemHostAlloc` range of `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: exclusive `&mut self`; `ptr` is a live host allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Grow or shrink in place by allocating a replacement range with the same flags.
    pub(super) fn resize(&mut self, new_len: usize, preserve: bool) -> Result<()> {
        let mut next = Self::alloc_with_flags(&self.ctx, new_len, self.flags)?;
        if preserve {
            let n = self.len.min(next.len);
            next.as_mut_slice()[..n].copy_from_slice(&self.as_slice()[..n]);
        }
        *self = next;
        Ok(())
    }
}

impl Drop for CudaPinnedHost {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let _gate = super::capture_gate::lock_capture_alloc_gate();
        let _ = self.ctx.bind_to_thread();
        // SAFETY: `ptr` came from `malloc_host` and is not used after this.
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr as *mut std::ffi::c_void) };
        self.ptr = std::ptr::null_mut();
    }
}
