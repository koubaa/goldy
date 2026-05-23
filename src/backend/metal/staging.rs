//! Metal staging belt and texture staging pool for buffer and texture uploads.
//!
//! `StagingBelt` wraps `StagingBeltCore<MetalBeltChunk>` using `StorageModeShared`
//! `MTLBuffer` chunks that are permanently mapped and bump-allocated.  Timeline-gated
//! reclaim (via `MTLSharedEvent.signaled_value()`) avoids per-frame allocations.
//!
//! `TextureStagingPool` provides a free-list of permanently-mapped staging buffers for
//! texture uploads, eliminating a fresh `MTLBuffer` allocation per `WriteTexture` call.
//!
//! Lifecycle: `acquire`/`write` → CPU fills → GPU blit copy → `release`/`finish`
//! (tagged with timeline signal value) → `reclaim` (when `signaled_value >= token`).

use super::super::shared::{BeltChunk as BeltChunkTrait, StagingBeltCore};
use ::metal as mtl;
use anyhow::Result;

pub(super) use super::super::shared::DEFAULT_STAGING_CHUNK_SIZE;

// ── StagingBelt ──────────────────────────────────────────────────────────────

struct MetalBeltChunk {
    buffer: mtl::Buffer,
    capacity: u64,
    offset: u64,
    /// Cached pointer from `buffer.contents()`. Valid for the lifetime of `buffer`.
    mapped: *mut u8,
}

// SAFETY: `mapped` is owned by this chunk and only accessed by the CPU staging path,
// which runs single-threaded under the Metal backend mutex.
unsafe impl Send for MetalBeltChunk {}
unsafe impl Sync for MetalBeltChunk {}

impl BeltChunkTrait for MetalBeltChunk {
    fn capacity(&self) -> u64 {
        self.capacity
    }
    fn offset(&self) -> u64 {
        self.offset
    }
    fn offset_mut(&mut self) -> &mut u64 {
        &mut self.offset
    }
    fn mapped_ptr(&self) -> *mut u8 {
        self.mapped
    }
}

fn allocate_chunk(device: &mtl::DeviceRef, size: u64) -> Result<MetalBeltChunk> {
    let buffer = device.new_buffer(size, mtl::MTLResourceOptions::StorageModeShared);
    let mapped = buffer.contents() as *mut u8;
    anyhow::ensure!(
        !mapped.is_null(),
        "StagingBelt: new_buffer returned null contents (size={})",
        size
    );
    Ok(MetalBeltChunk {
        buffer,
        capacity: size,
        offset: 0,
        mapped,
    })
}

/// Timeline-gated staging belt for `WriteBuffer` uploads.
///
/// Bump-allocates into permanently-mapped `StorageModeShared` chunks; once
/// committed, chunks are held in-flight until `reclaim(signaled_value)` confirms
/// the GPU copy has completed, then recycled for the next frame.
pub(super) struct StagingBelt {
    core: StagingBeltCore<MetalBeltChunk>,
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            core: StagingBeltCore::new(chunk_size),
        }
    }

    /// Return completed chunks to the free list.
    ///
    /// `completed` is the current value from `MTLSharedEvent.signaled_value()`.
    /// Chunks whose token ≤ `completed` are safe to recycle (the GPU blit has finished).
    pub fn reclaim(&mut self, completed: u64) {
        let mut i = 0;
        while i < self.core.in_flight.len() {
            if self.core.in_flight[i].0 <= completed {
                let (_, mut chunks) = self.core.in_flight.remove(i);
                for ch in &mut chunks {
                    ch.reset();
                }
                self.core.free.extend(chunks);
            } else {
                i += 1;
            }
        }
    }

    /// Bump-allocate `data.len()` bytes, `memcpy` `data` in, and return
    /// `(MTLBuffer, offset)` for use in a `copy_from_buffer` blit call.
    pub fn write(&mut self, device: &mtl::DeviceRef, data: &[u8]) -> Result<(mtl::Buffer, u64)> {
        let (idx, start) = self.core.write(data, |size| allocate_chunk(device, size))?;
        Ok((self.core.active[idx].buffer.clone(), start))
    }

    /// Tag all active chunks with `token` and move them to in-flight.
    /// Call immediately after the command buffer is committed.
    pub fn finish(&mut self, token: u64) {
        self.core.finish(token);
    }

    /// Drop oversized free chunks (MTLBuffer is ARC-managed; just drop).
    #[allow(dead_code)]
    pub fn trim(&mut self) {
        self.core.trim_free(|_ch| {});
    }

    pub fn destroy_all(&mut self) {
        self.core.destroy_all(|_ch| {});
    }
}

// ── TextureStagingPool ────────────────────────────────────────────────────────

/// A permanently-mapped staging buffer for a single texture upload.
///
/// Unlike belt chunks (bump-allocated sub-regions), each entry corresponds to
/// one full texture region and is returned to the pool as a whole unit.
pub(super) struct TextureStagingEntry {
    pub buffer: mtl::Buffer,
    pub capacity: u64,
    /// Cached pointer from `buffer.contents()`.
    mapped: *mut u8,
}

// SAFETY: see MetalBeltChunk.
unsafe impl Send for TextureStagingEntry {}
unsafe impl Sync for TextureStagingEntry {}

impl TextureStagingEntry {
    /// CPU-writable pointer into the staging buffer.
    pub fn mapped_ptr(&self) -> *mut u8 {
        self.mapped
    }
}

fn allocate_texture_staging_entry(
    device: &mtl::DeviceRef,
    size: u64,
) -> Result<TextureStagingEntry> {
    let buffer = device.new_buffer(size, mtl::MTLResourceOptions::StorageModeShared);
    let mapped = buffer.contents() as *mut u8;
    anyhow::ensure!(
        !mapped.is_null(),
        "TextureStagingPool: new_buffer returned null contents (size={})",
        size
    );
    Ok(TextureStagingEntry {
        buffer,
        capacity: size,
        mapped,
    })
}

/// Timeline-gated free-list pool for texture-upload staging buffers.
///
/// Lifecycle per entry: `acquire` → CPU `memcpy` → GPU `copy_from_buffer_to_texture`
/// → `release(timeline_value)` → `reclaim(completed)` → back to free list.
pub(super) struct TextureStagingPool {
    free: Vec<TextureStagingEntry>,
    in_flight: Vec<(u64, Vec<TextureStagingEntry>)>,
}

impl TextureStagingPool {
    pub fn new() -> Self {
        Self {
            free: Vec::new(),
            in_flight: Vec::new(),
        }
    }

    /// Acquire an entry with at least `size` bytes of capacity.
    ///
    /// Returns a recycled free entry on a pool hit; allocates a new permanently-mapped
    /// buffer on a miss.
    pub fn acquire(&mut self, device: &mtl::DeviceRef, size: u64) -> Result<TextureStagingEntry> {
        let _tz = crate::tracy_zone!("mtl.texture_staging.acquire");
        if let Some(pos) = self.free.iter().rposition(|e| e.capacity >= size) {
            return Ok(self.free.swap_remove(pos));
        }
        allocate_texture_staging_entry(device, size)
    }

    /// Tag `entries` with `timeline_value` and move them to in-flight.
    pub fn release(&mut self, timeline_value: u64, entries: Vec<TextureStagingEntry>) {
        if !entries.is_empty() {
            let _tz = crate::tracy_zone!("mtl.texture_staging.release");
            self.in_flight.push((timeline_value, entries));
        }
    }

    /// Move entries whose timeline has completed back to the free list.
    pub fn reclaim(&mut self, completed: u64) {
        let _tz = crate::tracy_zone!("mtl.texture_staging.reclaim");
        let mut i = 0;
        while i < self.in_flight.len() {
            if self.in_flight[i].0 <= completed {
                let (_, entries) = self.in_flight.swap_remove(i);
                self.free.extend(entries);
            } else {
                i += 1;
            }
        }
    }

    /// Destroy all free and in-flight entries.
    ///
    /// Safe to call at any time on device destroy — MTLBuffer is ARC-managed.
    pub fn destroy_all(&mut self) {
        self.free.clear();
        for (_, entries) in self.in_flight.drain(..) {
            drop(entries);
        }
    }
}
