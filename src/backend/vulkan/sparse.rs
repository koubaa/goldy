//! Sparse page pool and [`vkQueueBindSparse`] helpers.
//!
//! [`vkQueueBindSparse`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueBindSparse.html

use super::utils::find_memory_type;
use anyhow::{Context, Result};
use ash::vk;

/// Pages per DEVICE_LOCAL chunk (16 MiB at 64 KiB blocks).
const PAGES_PER_CHUNK: u32 = 256;

#[derive(Debug)]
struct MemoryChunk {
    memory: vk::DeviceMemory,
    page_size: vk::DeviceSize,
    num_pages: u32,
    free_indices: Vec<u32>,
}

/// Sub-allocates aligned pages from large [`vk::DeviceMemory`] chunks for sparse binding.
pub(crate) struct SparsePagePool {
    page_size: vk::DeviceSize,
    memory_type_index: u32,
    chunks: Vec<MemoryChunk>,
}

impl SparsePagePool {
    pub fn new(page_size: vk::DeviceSize, memory_type_index: u32) -> Self {
        Self {
            page_size,
            memory_type_index,
            chunks: Vec::new(),
        }
    }

    #[allow(dead_code)] // used by tests / future introspection
    pub fn page_size(&self) -> vk::DeviceSize {
        self.page_size
    }

    fn push_chunk_and_take_first_page(
        &mut self,
        device: &ash::Device,
    ) -> Result<(vk::DeviceMemory, vk::DeviceSize)> {
        let chunk_bytes = self.page_size * vk::DeviceSize::from(PAGES_PER_CHUNK);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(chunk_bytes)
            .memory_type_index(self.memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .with_context(|| format!("SparsePagePool: allocate {chunk_bytes} bytes"))?;

        // Page 0 of this chunk is returned; 1..PAGES_PER_CHUNK go to the free list.
        let mut free_indices: Vec<u32> = (1..PAGES_PER_CHUNK).collect();

        // Pack dense free list (optional micro-optimization)
        free_indices.reverse();

        self.chunks.push(MemoryChunk {
            memory,
            page_size: self.page_size,
            num_pages: PAGES_PER_CHUNK,
            free_indices,
        });

        Ok((memory, 0))
    }

    pub fn alloc_page(
        &mut self,
        device: &ash::Device,
    ) -> Result<(vk::DeviceMemory, vk::DeviceSize)> {
        for ch in &mut self.chunks {
            if let Some(idx) = ch.free_indices.pop() {
                let offset = vk::DeviceSize::from(idx) * ch.page_size;
                return Ok((ch.memory, offset));
            }
        }
        self.push_chunk_and_take_first_page(device)
    }

    pub fn free_page(&mut self, memory: vk::DeviceMemory, offset: vk::DeviceSize) {
        for ch in &mut self.chunks {
            if ch.memory == memory {
                debug_assert!(ch.page_size > 0 && offset.is_multiple_of(ch.page_size));
                let idx = u32::try_from(offset / ch.page_size).unwrap_or(0);
                debug_assert!(idx < ch.num_pages);
                ch.free_indices.push(idx);
                return;
            }
        }
        tracing::warn!(
            ?memory,
            offset,
            "SparsePagePool::free_page: unknown chunk (leak?)"
        );
    }
}

/// Query sparse **buffer** binding alignment (typically 64 KiB).
pub(crate) fn query_sparse_buffer_block_size(device: &ash::Device) -> Result<vk::DeviceSize> {
    let buf_info = vk::BufferCreateInfo::default()
        .size(256 * 1024)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);

    unsafe {
        let probe = device
            .create_buffer(&buf_info, None)
            .context("sparse probe create_buffer")?;
        let req = device.get_buffer_memory_requirements(probe);
        device.destroy_buffer(probe, None);
        let align = req.alignment.max(64 * 1024);
        Ok(align)
    }
}

/// `memory_type_index` + `memory_type_bits` for sparse buffers (DEVICE_LOCAL).
pub(crate) fn sparse_storage_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
) -> Result<(u32, u32)> {
    let buf_info = vk::BufferCreateInfo::default()
        .size(256 * 1024)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);

    unsafe {
        let probe = device
            .create_buffer(&buf_info, None)
            .context("sparse mem type probe create_buffer")?;
        let req = device.get_buffer_memory_requirements(probe);
        device.destroy_buffer(probe, None);
        let bits = req.memory_type_bits;
        let index = find_memory_type(
            instance,
            physical_device,
            bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .context("no DEVICE_LOCAL memory type for sparse buffer")?;
        Ok((index, bits))
    }
}

pub(crate) fn num_sparse_pages(allocation_size: u64, block_size: u64) -> u32 {
    if allocation_size == 0 || block_size == 0 {
        return 0;
    }
    u32::try_from(allocation_size.div_ceil(block_size)).unwrap_or(u32::MAX)
}

pub(crate) fn pages_needed_for_bytes(size: u64, block_size: u64) -> u32 {
    if size == 0 {
        return 0;
    }
    u32::try_from(size.div_ceil(block_size)).unwrap_or(u32::MAX)
}

/// Blocking sparse bind (correctness-first).
pub(crate) fn queue_bind_sparse_sync(
    device: &ash::Device,
    bind_queue: vk::Queue,
    buffer: vk::Buffer,
    binds: &[vk::SparseMemoryBind],
) -> Result<()> {
    if binds.is_empty() {
        return Ok(());
    }
    let buffer_bind = vk::SparseBufferMemoryBindInfo::default()
        .buffer(buffer)
        .binds(binds);

    let bind_info = vk::BindSparseInfo::default().buffer_binds(std::slice::from_ref(&buffer_bind));

    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe {
        device
            .create_fence(&fence_info, None)
            .context("sparse fence")?
    };

    unsafe {
        device
            .queue_bind_sparse(bind_queue, std::slice::from_ref(&bind_info), fence)
            .context("queue_bind_sparse")?;
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .context("wait_for_fences after sparse")?;
        device.destroy_fence(fence, None);
    }
    Ok(())
}

pub(crate) fn collect_sparse_binds_for_teardown(
    block_size: u64,
    page_map: &[Option<(vk::DeviceMemory, vk::DeviceSize)>],
) -> Vec<(u64, vk::DeviceMemory, vk::DeviceSize)> {
    let mut out = Vec::new();
    for (i, slot) in page_map.iter().enumerate() {
        if let Some((mem, off)) = *slot {
            let resource_offset = (i as u64).saturating_mul(block_size);
            out.push((resource_offset, mem, off));
        }
    }
    out
}
