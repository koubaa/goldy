//! Page-locked host staging for [`crate::types::BufferFlags::CPU_WRITABLE`].
//!
//! CUDA graphs capture the source pointer of host→device memcpy nodes. Pageable
//! `Vec<u8>` staging forces the driver to bounce through an internal pin and is
//! illegal to capture. This allocation is the CUDA analogue of a D3D12 UPLOAD heap:
//! CPU writes into a stable address, GPU copies from that address on replay.

use anyhow::{Context as _, Result};
use cudarc::driver::{sys, CudaContext};
use std::sync::Arc;

pub(super) struct CudaPinnedHost {
    ctx: Arc<CudaContext>,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the allocation is process-local CUDA host memory; Goldy only mutates it
// under the backend lock (CPU writes) or reads it from the submission worker (HtoD).
unsafe impl Send for CudaPinnedHost {}
unsafe impl Sync for CudaPinnedHost {}

impl CudaPinnedHost {
    pub(super) fn alloc(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        let len = len.max(1);
        let _gate = super::capture_gate::lock_capture_alloc_gate();
        ctx.bind_to_thread()
            .context("CUDA: bind context for pinned host alloc")?;
        // SAFETY: `cuMemHostAlloc` returns unset host memory of `len` bytes.
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(len, sys::CU_MEMHOSTALLOC_WRITECOMBINED as u32)
        }
        .context("CUDA: cuMemHostAlloc failed")?;
        let ptr = ptr as *mut u8;
        // SAFETY: freshly allocated `len` bytes.
        unsafe { std::ptr::write_bytes(ptr, 0, len) };
        Ok(Self {
            ctx: Arc::clone(ctx),
            ptr,
            len,
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

    /// Grow or shrink in place by allocating a replacement range.
    pub(super) fn resize(&mut self, new_len: usize, preserve: bool) -> Result<()> {
        let mut next = Self::alloc(&self.ctx, new_len)?;
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
