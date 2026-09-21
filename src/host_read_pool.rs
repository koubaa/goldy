//! Context-shared staging pool for host-claim readback copies.
//!
//! Used when a parcel is not host-coherent: `take()` copies into a pooled READBACK
//! buffer, waits, memcpy's into a `Vec`, then returns the handle immediately.

use crate::backend::BufferHandle;
use crate::context::Context;
use crate::error::GoldyError;
use crate::texture::TextureCopyFootprint;
use crate::timeline::TimelineValue;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct StampedStaging {
    handle: BufferHandle,
    ready_after: TimelineValue,
    capacity: u64,
    texture: bool,
}

/// Shared pool of host-read staging buffers for one [`Context`].
pub(crate) struct HostReadStagingPool {
    entries: Mutex<Vec<StampedStaging>>,
    alloc_count: AtomicUsize,
}

impl HostReadStagingPool {
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            alloc_count: AtomicUsize::new(0),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<StampedStaging>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn take_or_alloc_buffer(&self, ctx: &Context, need: u64) -> Result<BufferHandle, GoldyError> {
        self.take_or_alloc(ctx, need, false, |backend, device| {
            backend.alloc_readback_buffer(device, need)
        })
    }

    pub(crate) fn take_or_alloc_texture(
        &self,
        ctx: &Context,
        layout: TextureCopyFootprint,
    ) -> Result<BufferHandle, GoldyError> {
        self.take_or_alloc(ctx, layout.staging_bytes, true, |backend, device| {
            backend.alloc_texture_readback_staging(device, layout)
        })
    }

    fn take_or_alloc(
        &self,
        ctx: &Context,
        need: u64,
        texture: bool,
        alloc: impl FnOnce(
            &mut dyn crate::backend::GpuBackend,
            crate::backend::DeviceHandle,
        ) -> anyhow::Result<BufferHandle>,
    ) -> Result<BufferHandle, GoldyError> {
        let progress = ctx.gpu_progress();
        {
            let mut entries = self.lock();
            if let Some(i) = entries
                .iter()
                .position(|e| e.texture == texture && e.capacity >= need && e.ready_after <= progress)
            {
                return Ok(entries.swap_remove(i).handle);
            }
        }
        let device = ctx.runtime().inner.handle;
        let handle = {
            let mut backend = ctx.runtime().inner.backend.lock().unwrap();
            alloc(&mut **backend, device).map_err(|e| ctx.classify(e))?
        };
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        Ok(handle)
    }

    pub(crate) fn return_handle(&self, handle: BufferHandle, ready_after: TimelineValue, capacity: u64, texture: bool) {
        self.lock().push(StampedStaging {
            handle,
            ready_after,
            capacity,
            texture,
        });
    }

    pub(crate) fn alloc_count(&self) -> usize {
        self.alloc_count.load(Ordering::Relaxed)
    }

    pub(crate) fn drain_backend(&self, device: &crate::runtime::Runtime, _ctx: crate::backend::ContextHandle) {
        let mut entries = self.lock();
        if entries.is_empty() {
            return;
        }
        let mut backend = device.inner.backend.lock().unwrap();
        for entry in entries.drain(..) {
            backend.free_readback_buffer(entry.handle);
        }
    }
}
