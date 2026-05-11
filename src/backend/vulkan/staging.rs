//! Per-submission staging belt for batched uploads on the compute queue.
//!
//! Pools HOST_VISIBLE chunks so consecutive `ComputeCommand::WriteBuffer` submissions
//! do not reuse the same staging memory before GPU copies finish.

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
