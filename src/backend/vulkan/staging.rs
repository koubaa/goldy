//! Per-submission staging belt for batched uploads on the compute queue.
//!
//! Pools HOST_VISIBLE chunks so consecutive `ComputeCommand::WriteBuffer` submissions
//! do not reuse the same staging memory before GPU copies finish.

use super::super::DeviceHandle;
use super::types::LogicalDevice;
use super::utils::find_memory_type;
use anyhow::{Context, Result};
use ash::{vk, Instance};
use std::collections::HashMap;

/// Default belt chunk size — many small uploads per submit without fragmentation.
pub(super) const DEFAULT_STAGING_CHUNK_SIZE: u64 = 256 * 1024;

const COPY_ALIGN: u64 = 256;

struct BeltChunk {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: u64,
    /// Next write offset (may be unaligned until `align_up` at append time).
    offset: u64,
    mapped: usize,
}

impl BeltChunk {
    fn reset(&mut self) {
        self.offset = 0;
    }

    unsafe fn destroy(&mut self, device: &LogicalDevice) {
        let _ = device.unmap_memory2(self.memory);
        device.device.destroy_buffer(self.buffer, None);
        device.device.free_memory(self.memory, None);
    }
}

pub(super) struct StagingBelt {
    free_chunks: Vec<BeltChunk>,
    active_chunks: Vec<BeltChunk>,
    /// Chunks pinned until the corresponding compute fence completes.
    in_flight: Vec<(u64, Vec<BeltChunk>)>,
    chunk_size: u64,
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            free_chunks: Vec::new(),
            active_chunks: Vec::new(),
            in_flight: Vec::new(),
            chunk_size,
        }
    }

    /// Return completed chunks to the free list. Call at the start of each compute submit,
    /// **before** [`super::compute::reap_signaled_fences`] so `VkFence` handles are valid.
    ///
    /// `completed_timeline` is the current device timeline counter (from
    /// `vkGetSemaphoreCounterValue`).  Chunks tagged with timeline-semaphore values
    /// (i.e. tokens ≤ `completed_timeline`) are safe to recycle because the GPU has
    /// executed past them.  Chunks tagged with fence-pool tokens are recycled only
    /// once the corresponding `VkFence` signals (if any remain in the pool).
    pub fn reclaim(
        &mut self,
        compute_fence_pool: &HashMap<u64, (DeviceHandle, vk::Fence, Option<vk::CommandBuffer>)>,
        devices: &HashMap<DeviceHandle, LogicalDevice>,
        completed_timeline: u64,
    ) -> Result<()> {
        let mut i = 0;
        while i < self.in_flight.len() {
            let (token, _) = &self.in_flight[i];
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
                let (_, mut chunks) = self.in_flight.remove(i);
                for ch in &mut chunks {
                    ch.reset();
                }
                self.free_chunks.extend(chunks);
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
        if data.is_empty() {
            anyhow::bail!("StagingBelt::write: empty data");
        }
        let len = data.len() as u64;

        if let Some(ch) = self.active_chunks.last_mut() {
            let start = align_up(ch.offset, COPY_ALIGN);
            if start + len <= ch.capacity {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        (ch.mapped as *mut u8).add(start as usize),
                        data.len(),
                    );
                }
                ch.offset = start + len;
                return Ok((ch.buffer, start));
            }
        }

        let alloc_size = self.chunk_size.max(align_up(len, COPY_ALIGN));

        // Linear scan from the back (most-recently-freed first) to avoid the
        // push-then-immediately-pop infinite loop the old pattern had when every
        // free chunk was smaller than `len`.
        let mut chunk = if let Some(pos) = self.free_chunks.iter().rposition(|c| c.capacity >= len)
        {
            let mut c = self.free_chunks.swap_remove(pos);
            c.reset();
            c
        } else {
            allocate_chunk(
                instance,
                logical_device,
                logical_device.physical_device,
                alloc_size,
            )?
        };

        debug_assert!(chunk.offset == 0);
        let start = 0u64;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (chunk.mapped as *mut u8).add(start as usize),
                data.len(),
            );
        }
        chunk.offset = start + len;
        let buf = chunk.buffer;
        self.active_chunks.push(chunk);
        Ok((buf, start))
    }

    pub fn finish(&mut self, fence_token: u64) {
        if self.active_chunks.is_empty() {
            return;
        }
        self.in_flight
            .push((fence_token, std::mem::take(&mut self.active_chunks)));
    }

    /// Drop all free chunks whose capacity exceeds `chunk_size`.
    ///
    /// Safe to call at any time: `free_chunks` only holds chunks whose GPU fence has
    /// already signaled, so no GPU wait is needed.
    pub unsafe fn trim(&mut self, device: &LogicalDevice) {
        let chunk_size = self.chunk_size;
        let mut i = 0;
        while i < self.free_chunks.len() {
            if self.free_chunks[i].capacity > chunk_size {
                self.free_chunks.swap_remove(i).destroy(device);
            } else {
                i += 1;
            }
        }
    }

    pub unsafe fn destroy_all(&mut self, device: &LogicalDevice) {
        for ch in self.free_chunks.drain(..) {
            let mut c = ch;
            c.destroy(device);
        }
        for ch in self.active_chunks.drain(..) {
            let mut c = ch;
            c.destroy(device);
        }
        for (_, mut vec) in self.in_flight.drain(..) {
            for ch in vec.drain(..) {
                let mut c = ch;
                c.destroy(device);
            }
        }
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

fn allocate_chunk(
    instance: &Instance,
    device: &LogicalDevice,
    physical_device: vk::PhysicalDevice,
    size: u64,
) -> Result<BeltChunk> {
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

    Ok(BeltChunk {
        buffer,
        memory,
        capacity: size,
        offset: 0,
        mapped,
    })
}
