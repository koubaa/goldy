//! Per-submission staging belt for batched uploads on the compute queue, and a
//! pooled staging allocator for texture uploads.
//!
//! `StagingBelt` pools HOST_VISIBLE chunks so consecutive `WriteBuffer` submissions
//! do not reuse the same staging memory before GPU copies finish.
//!
//! `TextureStagingPool` pools individual HOST_VISIBLE buffers for texture uploads,
//! eliminating per-frame `vkAllocateMemory` / `vkFreeMemory` calls.

use super::super::shared::{BeltChunk as BeltChunkTrait, StagingBeltCore};
use super::super::DeviceHandle;
use super::types::LogicalDevice;
use super::utils::find_memory_type;
use anyhow::{Context, Result};
use ash::{vk, Instance};
use std::collections::HashMap;

/// Default belt chunk size — many small uploads per submit without fragmentation.
pub(super) const DEFAULT_STAGING_CHUNK_SIZE: u64 = super::super::shared::DEFAULT_STAGING_CHUNK_SIZE;

struct VkBeltChunk {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: u64,
    offset: u64,
    mapped: usize,
}

impl BeltChunkTrait for VkBeltChunk {
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
        self.mapped as *mut u8
    }
}

impl VkBeltChunk {
    unsafe fn destroy(self, device: &LogicalDevice) {
        let _ = device.unmap_memory2(self.memory);
        device.device.destroy_buffer(self.buffer, None);
        device.device.free_memory(self.memory, None);
    }
}

pub(super) struct StagingBelt {
    core: StagingBeltCore<VkBeltChunk>,
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            core: StagingBeltCore::new(chunk_size),
        }
    }

    /// Return completed chunks to the free list. Call at the start of each compute submit,
    /// **before** [`super::compute`] reaps signaled fences so `VkFence` handles are valid.
    ///
    /// `completed_timeline` is the current device timeline counter (from
    /// `vkGetSemaphoreCounterValue`). Chunks tagged with timeline-semaphore values
    /// (i.e. tokens ≤ `completed_timeline`) are safe to recycle because the GPU has
    /// executed past them. Chunks tagged with fence-pool tokens are recycled only
    /// once the corresponding `VkFence` signals (if any remain in the pool).
    pub fn reclaim(
        &mut self,
        compute_fence_pool: &HashMap<u64, (DeviceHandle, vk::Fence, Option<vk::CommandBuffer>)>,
        devices: &HashMap<DeviceHandle, LogicalDevice>,
        completed_timeline: u64,
    ) -> Result<()> {
        let mut i = 0;
        while i < self.core.in_flight.len() {
            let (token, _) = &self.core.in_flight[i];
            let done = if let Some((device_handle, fence, _)) = compute_fence_pool.get(token) {
                let logical_device = devices
                    .get(device_handle)
                    .context("StagingBelt::reclaim: device missing")?;
                match unsafe { logical_device.device.get_fence_status(*fence) } {
                    Ok(true) => true,
                    Ok(false) => false,
                    Err(vk::Result::NOT_READY) => false,
                    Err(e) => {
                        tracing::warn!(?e, token, "get_fence_status during belt reclaim");
                        false
                    }
                }
            } else {
                // Timeline-semaphore path: the token IS the timeline signal value.
                // Recycle only once the GPU timeline has advanced past it.
                *token <= completed_timeline
            };
            if done {
                let (_, mut chunks) = self.core.in_flight.remove(i);
                for ch in &mut chunks {
                    ch.reset();
                }
                self.core.free.extend(chunks);
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Allocate `data.len()` bytes, copy `data`, return source buffer + offset for `vkCmdCopyBuffer`.
    pub fn write(
        &mut self,
        instance: &Instance,
        logical_device: &LogicalDevice,
        data: &[u8],
    ) -> Result<(vk::Buffer, u64)> {
        let (idx, start) = self.core.write(data, |size| {
            allocate_chunk(
                instance,
                logical_device,
                logical_device.physical_device,
                size,
            )
        })?;
        Ok((self.core.active[idx].buffer, start))
    }

    pub fn finish(&mut self, fence_token: u64) {
        self.core.finish(fence_token);
    }

    /// Drop all free chunks whose capacity exceeds `chunk_size`.
    ///
    /// Safe to call at any time: `free_chunks` only holds chunks whose GPU fence has
    /// already signaled, so no GPU wait is needed.
    pub unsafe fn trim(&mut self, device: &LogicalDevice) {
        self.core.trim_free(|ch| ch.destroy(device));
    }

    pub unsafe fn destroy_all(&mut self, device: &LogicalDevice) {
        self.core.destroy_all(|ch| ch.destroy(device));
    }
}

fn allocate_chunk(
    instance: &Instance,
    device: &LogicalDevice,
    physical_device: vk::PhysicalDevice,
    size: u64,
) -> Result<VkBeltChunk> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.device.create_buffer(&info, None) }
        .context("StagingBelt: create_buffer failed")?;

    let reqs = unsafe { device.device.get_buffer_memory_requirements(buffer) };
    let mem_type = find_memory_type(
        instance,
        physical_device,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .context("StagingBelt: no HOST_VISIBLE|HOST_COHERENT memory type")?;

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);

    let memory = unsafe { device.device.allocate_memory(&alloc, None) }
        .context("StagingBelt: allocate_memory failed")?;

    unsafe { device.device.bind_buffer_memory(buffer, memory, 0) }
        .context("StagingBelt: bind_buffer_memory failed")?;

    let mapped = unsafe {
        device
            .map_memory2(memory, 0, size)
            .context("StagingBelt: map_memory2 failed")?
    } as usize;

    Ok(VkBeltChunk {
        buffer,
        memory,
        capacity: size,
        offset: 0,
        mapped,
    })
}

// ── TextureStagingPool ────────────────────────────────────────────────────────

/// A permanently-mapped, pre-allocated staging buffer for a single texture upload.
///
/// Unlike `StagingBelt` chunks (bump-allocated), each entry corresponds to one
/// texture region and is returned to the pool as a whole unit.
pub(super) struct TextureStagingEntry {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// Allocated byte capacity of this entry.
    pub capacity: u64,
    /// Permanently-mapped CPU pointer into `memory`, stored as `usize` for `Send`.
    mapped: usize,
}

impl TextureStagingEntry {
    /// Returns the permanently-mapped write pointer for this entry.
    pub fn mapped_ptr(&self) -> *mut u8 {
        self.mapped as *mut u8
    }

    /// Destroy this entry's Vulkan resources. Caller must ensure the GPU is idle
    /// with respect to any in-flight copy commands that referenced this buffer.
    pub(super) unsafe fn destroy(self, logical_device: &LogicalDevice) {
        let _ = logical_device.unmap_memory2(self.memory);
        logical_device.device.destroy_buffer(self.buffer, None);
        logical_device.device.free_memory(self.memory, None);
    }
}

// SAFETY: the raw pointer in `mapped` is owned exclusively by this entry and
// only accessed by the CPU owner between acquire and release.
unsafe impl Send for TextureStagingEntry {}
unsafe impl Sync for TextureStagingEntry {}

/// Timeline-gated free-list pool for texture-upload staging buffers.
///
/// Eliminates per-frame `vkAllocateMemory` / `vkFreeMemory` calls by recycling
/// entries whose GPU copy timeline has advanced past their release point.
///
/// Lifecycle per entry: `acquire` → CPU fills → GPU copy → `release` (tagged
/// with timeline signal value) → `reclaim` (when GPU completed) → back to free list.
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

    /// Acquire a staging entry with at least `size` bytes of capacity.
    ///
    /// Returns a recycled free entry on a pool hit. On a miss, allocates a new
    /// permanently-mapped entry. The entry's mapped memory is ready for `memcpy`.
    pub fn acquire(
        &mut self,
        instance: &Instance,
        logical_device: &LogicalDevice,
        size: u64,
    ) -> Result<TextureStagingEntry> {
        if let Some(pos) = self.free.iter().rposition(|e| e.capacity >= size) {
            let _tz = crate::tracy_zone!("vk.texture_staging.acquire.hit");
            return Ok(self.free.swap_remove(pos));
        }
        let _tz = crate::tracy_zone!("vk.texture_staging.acquire.miss");
        allocate_texture_staging_entry(instance, logical_device, size)
    }

    /// Try to acquire from the free list only (no allocation).
    ///
    /// Returns `Some(entry)` on a hit, `None` on a miss. Used in tests to
    /// verify pooling without requiring a real Vulkan device.
    #[cfg(test)]
    pub(super) fn acquire_from_free_only(&mut self, size: u64) -> Option<TextureStagingEntry> {
        let pos = self.free.iter().rposition(|e| e.capacity >= size)?;
        Some(self.free.swap_remove(pos))
    }

    /// Tag `entries` with `timeline_value` and move them to in-flight.
    ///
    /// Entries become available for reuse once `reclaim(completed)` is called
    /// with `completed >= timeline_value`.
    pub fn release(&mut self, timeline_value: u64, entries: Vec<TextureStagingEntry>) {
        if !entries.is_empty() {
            let _tz = crate::tracy_zone!("vk.texture_staging.release");
            self.in_flight.push((timeline_value, entries));
        }
    }

    /// Move entries whose timeline has completed back to the free list.
    pub fn reclaim(&mut self, completed_timeline: u64) {
        let _tz = crate::tracy_zone!("vk.texture_staging.reclaim");
        let mut i = 0;
        while i < self.in_flight.len() {
            if self.in_flight[i].0 <= completed_timeline {
                let (_, entries) = self.in_flight.swap_remove(i);
                self.free.extend(entries);
            } else {
                i += 1;
            }
        }
    }

    /// Destroy all free and in-flight entries unconditionally.
    ///
    /// # Safety
    /// Must only be called when the device is idle — all GPU copy commands
    /// referencing in-flight entries must have completed.
    pub unsafe fn destroy_all(&mut self, logical_device: &LogicalDevice) {
        for entry in self.free.drain(..) {
            entry.destroy(logical_device);
        }
        for (_, entries) in self.in_flight.drain(..) {
            for entry in entries {
                entry.destroy(logical_device);
            }
        }
    }
}

fn allocate_texture_staging_entry(
    instance: &Instance,
    logical_device: &LogicalDevice,
    size: u64,
) -> Result<TextureStagingEntry> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { logical_device.device.create_buffer(&info, None) }
        .context("TextureStagingPool: create_buffer failed")?;

    let reqs = unsafe { logical_device.device.get_buffer_memory_requirements(buffer) };
    let mem_type = find_memory_type(
        instance,
        logical_device.physical_device,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .context("TextureStagingPool: no HOST_VISIBLE|HOST_COHERENT memory type")?;

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);

    let memory = unsafe { logical_device.device.allocate_memory(&alloc, None) }
        .context("TextureStagingPool: allocate_memory failed")?;

    unsafe { logical_device.device.bind_buffer_memory(buffer, memory, 0) }
        .context("TextureStagingPool: bind_buffer_memory failed")?;

    let mapped = unsafe {
        logical_device
            .map_memory2(memory, 0, size)
            .context("TextureStagingPool: map_memory2 failed")?
    } as usize;

    Ok(TextureStagingEntry {
        buffer,
        memory,
        capacity: size,
        mapped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a dummy entry with null Vulkan handles for pool logic tests.
    /// These entries must never be `destroy()`ed.
    fn dummy_entry(capacity: u64) -> TextureStagingEntry {
        TextureStagingEntry {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            capacity,
            mapped: 0,
        }
    }

    #[test]
    fn pool_reclaim_moves_entries_to_free_list() {
        let mut pool = TextureStagingPool::new();
        pool.release(10, vec![dummy_entry(256), dummy_entry(512)]);
        pool.release(20, vec![dummy_entry(1024)]);

        // Nothing reclaimed yet.
        assert_eq!(pool.free.len(), 0);
        assert_eq!(pool.in_flight.len(), 2);

        // Reclaim up to timeline 10 — first batch freed, second stays in-flight.
        pool.reclaim(10);
        assert_eq!(
            pool.free.len(),
            2,
            "two entries from timeline 10 should be free"
        );
        assert_eq!(pool.in_flight.len(), 1, "timeline 20 batch still in-flight");

        // Reclaim up to timeline 20 — second batch freed.
        pool.reclaim(20);
        assert_eq!(pool.free.len(), 3, "all three entries should now be free");
        assert_eq!(pool.in_flight.len(), 0);
    }

    #[test]
    fn pool_acquire_reuses_free_entries() {
        let mut pool = TextureStagingPool::new();
        pool.free.push(dummy_entry(512));
        pool.free.push(dummy_entry(256));

        // Requesting <= 256 bytes hits the 256-entry.
        // (acquire searches back-to-front for `capacity >= size`)
        let entry_256 = pool.acquire_from_free_only(256);
        assert!(entry_256.is_some());
        assert_eq!(entry_256.unwrap().capacity, 256);
        assert_eq!(pool.free.len(), 1);

        // Requesting 512 bytes hits the remaining 512-entry.
        let entry_512 = pool.acquire_from_free_only(512);
        assert!(entry_512.is_some());
        assert_eq!(entry_512.unwrap().capacity, 512);
        assert_eq!(pool.free.len(), 0);

        // Pool is empty; no more free entries.
        let miss = pool.acquire_from_free_only(1);
        assert!(miss.is_none());
    }

    #[test]
    fn pool_acquire_requires_sufficient_capacity() {
        let mut pool = TextureStagingPool::new();
        pool.free.push(dummy_entry(128));

        // Requesting more than the free entry's capacity → miss.
        let miss = pool.acquire_from_free_only(256);
        assert!(
            miss.is_none(),
            "entry with capacity 128 should not satisfy 256"
        );
        assert_eq!(
            pool.free.len(),
            1,
            "entry should remain in free list on miss"
        );
    }

    #[test]
    fn pool_reclaim_partial_timeline() {
        let mut pool = TextureStagingPool::new();
        pool.release(5, vec![dummy_entry(64)]);
        pool.release(15, vec![dummy_entry(64)]);
        pool.release(25, vec![dummy_entry(64)]);

        pool.reclaim(14);
        assert_eq!(pool.free.len(), 1, "only timeline 5 should be reclaimed");
        assert_eq!(pool.in_flight.len(), 2);
    }
}
