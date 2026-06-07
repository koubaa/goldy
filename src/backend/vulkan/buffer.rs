//! Buffer management logic.

use super::sparse;
use super::types::{self, BufferState};
use super::utils::find_memory_type;
use super::{BufferHandle, DeviceHandle};
use crate::backend::BufferKind;
use crate::types::BufferFlags;
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::Ordering;

/// Submit a one-shot vkCmdCopyBuffer between two buffers and wait for completion.
fn submit_copy(
    device: &types::LogicalDevice,
    src: vk::Buffer,
    dst: vk::Buffer,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
) -> Result<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe { device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate transfer command buffer")?;
    let cmd = cmd_buffers[0];

    let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    let region = vk::BufferCopy {
        src_offset,
        dst_offset,
        size,
    };

    unsafe {
        device.device.begin_command_buffer(cmd, &begin_info)?;
        device
            .device
            .cmd_copy_buffer(cmd, src, dst, std::slice::from_ref(&region));

        let mem_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            );
        let dep_info = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
        device.device.cmd_pipeline_barrier2(cmd, &dep_info);

        device.device.end_command_buffer(cmd)?;

        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        device
            .device
            .queue_submit(device.queue, &[submit_info], vk::Fence::null())?;
        device.device.queue_wait_idle(device.queue)?;
        device.device.free_command_buffers(device.command_pool, &cmd_buffers);
    }

    Ok(())
}

/// Create a buffer with the given size and access pattern.
#[allow(clippy::too_many_arguments)]
pub(super) fn create(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    next_buffer_handle: &mut BufferHandle,
    instance: &ash::Instance,
    device_handle: DeviceHandle,
    logical_size: u64,
    allocation_size: u64,
    access: BufferKind,
    element_stride: Option<u32>,
    flags: BufferFlags,
) -> Result<BufferHandle> {
    assert!(logical_size <= allocation_size);
    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

    // Map access pattern to Vulkan buffer usage flags
    // All buffers get TRANSFER_SRC | TRANSFER_DST for flexibility
    let mut vk_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let is_storage = match access {
        BufferKind::Scattered => {
            vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER;
            // Indirect dispatch reads 3× u32 (12 bytes) from a storage buffer
            if logical_size >= 12 {
                vk_usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
            }
            true
        }
        BufferKind::Broadcast => {
            vk_usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
            false
        }
    };

    let cpu_readable = flags.contains(BufferFlags::CPU_READABLE);
    if cpu_readable && !is_storage {
        anyhow::bail!("BufferFlags::CPU_READABLE is only valid for BufferKind::Scattered (storage) buffers");
    }

    let bindless_descriptor_set = logical_device.bindless_descriptor_set;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(allocation_size)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer =
        unsafe { logical_device.device.create_buffer(&buffer_info, None) }.context("Failed to create buffer")?;

    let mem_requirements = unsafe { logical_device.device.get_buffer_memory_requirements(buffer) };

    // Storage buffers → DEVICE_LOCAL for GPU compute performance, unless CPU_READABLE
    // (host-visible storage for persistent map + stable UAV bindless use).
    // Uniform buffers → HOST_VISIBLE|HOST_COHERENT for frequent CPU writes.
    let desired_flags = if is_storage {
        if cpu_readable {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        }
    } else {
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    };

    let memory_type = find_memory_type(
        instance,
        logical_device.physical_device,
        mem_requirements.memory_type_bits,
        desired_flags,
    )
    .context("Failed to find suitable memory type")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    let memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
        .inspect_err(|_e| {
            crate::signal::push_sync_signal(crate::signal::Signal::Oversubscribed {
                reason: crate::signal::OversubscribedReason::BufferHeap,
                size_hint: mem_requirements.size,
            });
        })
        .context("Failed to allocate buffer memory")?;

    unsafe { logical_device.device.bind_buffer_memory(buffer, memory, 0) }.context("Failed to bind buffer memory")?;

    // Staging buffer is created lazily on first write() to avoid doubling memory
    // for buffers that are only GPU-written (intermediate compute buffers, pool backing).
    // Clear uses vkCmdFillBuffer (GPU-side) so staging isn't needed for that.
    let (staging_buffer, staging_memory) = (None, None);

    let host_mapped: Option<usize> = if cpu_readable && is_storage {
        let device = devices
            .get(&device_handle)
            .context("Buffer's device is invalid for host map")?;
        let ptr = unsafe { device.map_memory2(memory, 0, allocation_size) }
            .context("Failed to map CPU_READABLE buffer memory")?;
        let p = ptr as *mut u8;
        if p.is_null() {
            anyhow::bail!("map_memory2 returned null for CPU_READABLE buffer");
        }
        Some(p as usize)
    } else {
        None
    };

    let handle = *next_buffer_handle;
    *next_buffer_handle += 1;

    // Register buffer in bindless descriptor set (UNIFORM or STORAGE)
    let bindless_index = {
        let logical_device = devices.get(&device_handle).unwrap();
        let index = logical_device
            .ledger
            .lock()
            .unwrap()
            .resource_registry
            .register_buffer(handle, is_storage);

        // Update the global descriptor set with this buffer
        if let Some(descriptor_set) = bindless_descriptor_set {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(0)
                .range(logical_size.max(1));

            let binding = if is_storage {
                types::bindless_bindings::STORAGE_BUFFERS
            } else {
                types::bindless_bindings::UNIFORM_BUFFERS
            };

            let descriptor_type = if is_storage {
                vk::DescriptorType::STORAGE_BUFFER
            } else {
                vk::DescriptorType::UNIFORM_BUFFER
            };

            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding)
                .dst_array_element(index)
                .descriptor_type(descriptor_type)
                .buffer_info(std::slice::from_ref(&buffer_info));

            unsafe {
                logical_device
                    .device
                    .update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }

            tracing::trace!(
                "Registered buffer {} at bindless index {} (storage={})",
                handle,
                index,
                is_storage
            );
        }

        Some(index)
    };

    buffers.insert(
        handle,
        BufferState {
            device_handle,
            buffer,
            memory,
            size: logical_size,
            allocation_size,
            bindless_index,
            is_storage,
            element_stride,
            staging_buffer,
            staging_memory,
            is_view: false,
            host_mapped,
            flags,
            transient_heap_suballoc: false,
            view_byte_offset: None,
            is_sparse: false,
            sparse_block_size: 0,
            sparse_pages: Vec::new(),
        },
    );

    Ok(handle)
}

fn align_sparse_capacity(cap: u64, block: u64) -> u64 {
    match NonZeroU64::new(block) {
        None => cap,
        Some(block) => {
            let block = block.get();
            cap.div_ceil(block).saturating_mul(block)
        }
    }
}

/// Sparse **device-local** storage buffer: virtual size `capacity`, initially backed pages for `logical_size`.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_sparse_with_capacity(
    _instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    next_buffer_handle: &mut BufferHandle,
    device_handle: DeviceHandle,
    logical_size: u64,
    capacity: u64,
    element_stride: Option<u32>,
    flags: BufferFlags,
) -> Result<BufferHandle> {
    let ld = devices.get(&device_handle).context("Invalid device handle")?;
    let block = ld.sparse_buffer_block_size;
    anyhow::ensure!(block > 0, "sparse block size not initialized");

    let cap = capacity.max(logical_size);
    let allocation_size = align_sparse_capacity(cap, block);
    let num_pages = sparse::num_sparse_pages(allocation_size, block) as usize;
    anyhow::ensure!(num_pages > 0, "sparse buffer capacity too small");

    let mut vk_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    vk_usage |=
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER;
    if logical_size >= 12 {
        vk_usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }

    let buffer_info = vk::BufferCreateInfo::default()
        .size(allocation_size)
        .usage(vk_usage)
        .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { ld.device.create_buffer(&buffer_info, None) }.context("Failed to create sparse buffer")?;

    let initial_pages = sparse::pages_needed_for_bytes(logical_size, block) as usize;
    let mut sparse_pages: Vec<Option<(vk::DeviceMemory, vk::DeviceSize)>> = vec![None; num_pages];
    let mut binds: Vec<vk::SparseMemoryBind> = Vec::with_capacity(initial_pages);

    let bind_queue = ld.sparse_binding_queue;
    let dev = &ld.device;

    let mut pool_guard = ld.sparse_page_pool.lock().unwrap();
    let pool = pool_guard.as_mut().context("internal: sparse page pool missing")?;

    for (i, sparse_page) in sparse_pages.iter_mut().enumerate().take(initial_pages) {
        let (mem, mem_off) = pool.alloc_page(dev)?;
        let resource_offset = (i as u64).saturating_mul(block);
        *sparse_page = Some((mem, mem_off));
        binds.push(
            vk::SparseMemoryBind::default()
                .resource_offset(resource_offset)
                .size(block)
                .memory(mem)
                .memory_offset(mem_off)
                .flags(vk::SparseMemoryBindFlags::empty()),
        );
    }

    sparse::queue_bind_sparse_sync(dev, bind_queue, buffer, &binds)?;
    drop(pool_guard);

    let bindless_descriptor_set = ld.bindless_descriptor_set;
    let handle = *next_buffer_handle;
    *next_buffer_handle += 1;

    let bindless_index = {
        let index = ld
            .ledger
            .lock()
            .unwrap()
            .resource_registry
            .register_buffer(handle, true);
        if let Some(descriptor_set) = bindless_descriptor_set {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(0)
                .range(logical_size.max(1));

            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(types::bindless_bindings::STORAGE_BUFFERS)
                .dst_array_element(index)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));
            unsafe {
                ld.device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }
        }
        Some(index)
    };

    buffers.insert(
        handle,
        BufferState {
            device_handle,
            buffer,
            memory: vk::DeviceMemory::null(),
            size: logical_size,
            allocation_size,
            bindless_index,
            is_storage: true,
            element_stride,
            staging_buffer: None,
            staging_memory: None,
            is_view: false,
            host_mapped: None,
            flags,
            transient_heap_suballoc: false,
            view_byte_offset: None,
            is_sparse: true,
            sparse_block_size: block,
            sparse_pages,
        },
    );

    Ok(handle)
}

/// Hint unused sparse pages at and above `offset` (bytes).
pub(super) fn hint_unused_above(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
    offset: u64,
) {
    let Some(buf) = buffers.get(&buffer_handle) else {
        return;
    };
    if !buf.is_sparse {
        return;
    }
    let block = buf.sparse_block_size;
    if block == 0 {
        return;
    }
    let device_handle = buf.device_handle;
    let vkbuf = buf.buffer;
    let alloc = buf.allocation_size;
    let first_page = ((offset.saturating_add(block.saturating_sub(1))) / block) as usize;
    let total_pages = sparse::num_sparse_pages(alloc, block) as usize;
    if first_page >= total_pages {
        return;
    }

    {
        let Some(ld) = devices.get(&device_handle) else {
            return;
        };
        let bind_queue = ld.sparse_binding_queue;
        let dev: &ash::Device = &ld.device;
        let sparse_pages = &mut buffers.get_mut(&buffer_handle).expect("buffer missing").sparse_pages;

        let mut binds = Vec::new();
        let mut to_free: Vec<(vk::DeviceMemory, vk::DeviceSize)> = Vec::new();
        for i in first_page..total_pages {
            if let Some((mem, mem_off)) = sparse_pages.get_mut(i).and_then(|s| s.take()) {
                let resource_offset = (i as u64).saturating_mul(block);
                binds.push(
                    vk::SparseMemoryBind::default()
                        .resource_offset(resource_offset)
                        .size(block)
                        .memory(vk::DeviceMemory::null())
                        .memory_offset(0)
                        .flags(vk::SparseMemoryBindFlags::empty()),
                );
                to_free.push((mem, mem_off));
            }
        }
        if binds.is_empty() {
            return;
        }
        if let Err(e) = sparse::queue_bind_sparse_sync(dev, bind_queue, vkbuf, &binds) {
            tracing::warn!(?e, "hint_unused_above sparse unbind failed");
            return;
        }
        let mut pool_guard = ld.sparse_page_pool.lock().unwrap();
        if let Some(pool) = pool_guard.as_mut() {
            for (mem, off) in to_free {
                pool.free_page(mem, off);
            }
        }
    }
}

/// Byte size of the underlying VkBuffer allocation.
pub(super) fn capacity(buffers: &HashMap<BufferHandle, BufferState>, buffer_handle: BufferHandle) -> u64 {
    buffers.get(&buffer_handle).map(|b| b.allocation_size).unwrap_or(0)
}

pub(super) fn set_logical_size(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    new_logical_size: u64,
) -> Result<()> {
    let is_sparse = buffers.get(&buffer_handle).map(|b| b.is_sparse).unwrap_or(false);
    if is_sparse {
        return set_logical_size_sparse(devices, buffers, device_handle, buffer_handle, new_logical_size);
    }

    let (bindless_index, is_storage, vkbuf, old_logical) = {
        let buf = buffers.get(&buffer_handle).context("Invalid buffer handle")?;
        if buf.is_view {
            anyhow::bail!("cannot set logical size on buffer views");
        }
        if buf.transient_heap_suballoc {
            anyhow::bail!("cannot change logical size on transient heap sub-allocations");
        }
        if buf.device_handle != device_handle {
            anyhow::bail!("buffer belongs to a different device");
        }
        if new_logical_size > buf.allocation_size {
            anyhow::bail!("logical size exceeds allocation");
        }
        if new_logical_size == 0 {
            anyhow::bail!("buffer size must be non-zero");
        }
        (buf.bindless_index, buf.is_storage, buf.buffer, buf.size)
    };

    buffers.get_mut(&buffer_handle).unwrap().size = new_logical_size;

    if old_logical == new_logical_size {
        return Ok(());
    }

    // When growing, the existing staging buffer was sized for the old (smaller) logical
    // size. Invalidate it so ensure_staging recreates it at the new size.
    if new_logical_size > old_logical {
        let (old_stg_buf, old_stg_mem) = {
            let buf = buffers.get_mut(&buffer_handle).unwrap();
            (buf.staging_buffer.take(), buf.staging_memory.take())
        };
        if let (Some(stg_buf), Some(stg_mem)) = (old_stg_buf, old_stg_mem) {
            let ld = devices.get(&device_handle).unwrap();
            let barrier = ld.timeline_next.load(Ordering::Relaxed).saturating_sub(1);
            ld.deletion_queue.lock().unwrap().queue(
                barrier,
                types::PendingDeletion::ReplacedBufferGpu {
                    buffer: stg_buf,
                    memory: stg_mem,
                    staging_buffer: None,
                    staging_memory: None,
                },
            );
        }
    }

    let bindless_descriptor_set = devices
        .get(&device_handle)
        .context("Invalid device handle")?
        .bindless_descriptor_set;

    if let (Some(descriptor_set), Some(bindless_index)) = (bindless_descriptor_set, bindless_index) {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(vkbuf)
            .offset(0)
            .range(new_logical_size.max(1));

        let binding = if is_storage {
            types::bindless_bindings::STORAGE_BUFFERS
        } else {
            types::bindless_bindings::UNIFORM_BUFFERS
        };

        let descriptor_type = if is_storage {
            vk::DescriptorType::STORAGE_BUFFER
        } else {
            vk::DescriptorType::UNIFORM_BUFFER
        };

        let logical_device = devices.get(&device_handle).unwrap();
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(binding)
            .dst_array_element(bindless_index)
            .descriptor_type(descriptor_type)
            .buffer_info(std::slice::from_ref(&buffer_info));
        unsafe {
            logical_device
                .device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    }

    Ok(())
}

fn set_logical_size_sparse(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    new_logical_size: u64,
) -> Result<()> {
    let (bindless_index, is_storage, vkbuf, old_logical, block) = {
        let buf = buffers.get(&buffer_handle).context("Invalid buffer handle")?;
        if buf.is_view {
            anyhow::bail!("cannot set logical size on buffer views");
        }
        if buf.transient_heap_suballoc {
            anyhow::bail!("cannot change logical size on transient heap sub-allocations");
        }
        if buf.device_handle != device_handle {
            anyhow::bail!("buffer belongs to a different device");
        }
        if new_logical_size > buf.allocation_size {
            anyhow::bail!("logical size exceeds allocation");
        }
        if new_logical_size == 0 {
            anyhow::bail!("buffer size must be non-zero");
        }
        (
            buf.bindless_index,
            buf.is_storage,
            buf.buffer,
            buf.size,
            buf.sparse_block_size,
        )
    };

    if old_logical == new_logical_size {
        return Ok(());
    }

    let old_pages = sparse::pages_needed_for_bytes(old_logical, block) as usize;
    let new_pages = sparse::pages_needed_for_bytes(new_logical_size, block) as usize;

    {
        let ld = devices.get(&device_handle).context("Invalid device handle")?;
        let bind_queue = ld.sparse_binding_queue;
        let dev: &ash::Device = &ld.device;

        let sparse_pages = &mut buffers
            .get_mut(&buffer_handle)
            .context("buffer disappeared")?
            .sparse_pages;

        let mut pool_guard = ld.sparse_page_pool.lock().unwrap();
        let pool = pool_guard.as_mut().context("internal: sparse pool missing")?;

        if new_pages > old_pages {
            let mut binds = Vec::with_capacity(new_pages - old_pages);
            for i in old_pages..new_pages {
                let (mem, mem_off) = pool.alloc_page(dev)?;
                let resource_offset = (i as u64).saturating_mul(block);
                if i >= sparse_pages.len() {
                    anyhow::bail!("internal: sparse page index out of range");
                }
                sparse_pages[i] = Some((mem, mem_off));
                binds.push(
                    vk::SparseMemoryBind::default()
                        .resource_offset(resource_offset)
                        .size(block)
                        .memory(mem)
                        .memory_offset(mem_off)
                        .flags(vk::SparseMemoryBindFlags::empty()),
                );
            }
            sparse::queue_bind_sparse_sync(dev, bind_queue, vkbuf, &binds)?;
        } else if new_pages < old_pages {
            let mut binds = Vec::new();
            let mut to_free: Vec<(vk::DeviceMemory, vk::DeviceSize)> = Vec::new();
            for i in new_pages..old_pages {
                if let Some((mem, mem_off)) = sparse_pages.get_mut(i).and_then(|s| s.take()) {
                    let resource_offset = (i as u64).saturating_mul(block);
                    binds.push(
                        vk::SparseMemoryBind::default()
                            .resource_offset(resource_offset)
                            .size(block)
                            .memory(vk::DeviceMemory::null())
                            .memory_offset(0)
                            .flags(vk::SparseMemoryBindFlags::empty()),
                    );
                    to_free.push((mem, mem_off));
                }
            }
            if !binds.is_empty() {
                sparse::queue_bind_sparse_sync(dev, bind_queue, vkbuf, &binds)?;
            }
            for (mem, off) in to_free {
                pool.free_page(mem, off);
            }
        }
    }

    buffers.get_mut(&buffer_handle).unwrap().size = new_logical_size;

    // When growing, the existing staging buffer was sized for the old (smaller) logical
    // size. Invalidate it so ensure_staging recreates it at the new size.
    if new_logical_size > old_logical {
        let (old_stg_buf, old_stg_mem) = {
            let buf = buffers.get_mut(&buffer_handle).unwrap();
            (buf.staging_buffer.take(), buf.staging_memory.take())
        };
        if let (Some(stg_buf), Some(stg_mem)) = (old_stg_buf, old_stg_mem) {
            let ld = devices.get(&device_handle).unwrap();
            let barrier = ld.timeline_next.load(Ordering::Relaxed).saturating_sub(1);
            ld.deletion_queue.lock().unwrap().queue(
                barrier,
                types::PendingDeletion::ReplacedBufferGpu {
                    buffer: stg_buf,
                    memory: stg_mem,
                    staging_buffer: None,
                    staging_memory: None,
                },
            );
        }
    }

    let bindless_descriptor_set = devices
        .get(&device_handle)
        .context("Invalid device handle")?
        .bindless_descriptor_set;

    if let (Some(descriptor_set), Some(bindless_index)) = (bindless_descriptor_set, bindless_index) {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(vkbuf)
            .offset(0)
            .range(new_logical_size.max(1));

        let binding = if is_storage {
            types::bindless_bindings::STORAGE_BUFFERS
        } else {
            types::bindless_bindings::UNIFORM_BUFFERS
        };

        let descriptor_type = if is_storage {
            vk::DescriptorType::STORAGE_BUFFER
        } else {
            vk::DescriptorType::UNIFORM_BUFFER
        };

        let logical_device = devices.get(&device_handle).unwrap();
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(binding)
            .dst_array_element(bindless_index)
            .descriptor_type(descriptor_type)
            .buffer_info(std::slice::from_ref(&buffer_info));
        unsafe {
            logical_device
                .device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    }

    Ok(())
}

/// Allocate VkBuffer + memory (no bindless registration). Matches [`create`] rules.
fn allocate_vk_buffer_memory(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    device_handle: DeviceHandle,
    size: u64,
    is_storage: bool,
    flags: BufferFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, Option<usize>)> {
    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

    let mut vk_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    if is_storage {
        vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER;
        if size >= 12 {
            vk_usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }
    } else {
        vk_usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }

    let cpu_readable = flags.contains(BufferFlags::CPU_READABLE);
    if cpu_readable && !is_storage {
        anyhow::bail!("BufferFlags::CPU_READABLE is only valid for BufferKind::Scattered (storage) buffers");
    }

    let desired_flags = if is_storage {
        if cpu_readable {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        }
    } else {
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    };

    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { logical_device.device.create_buffer(&buffer_info, None) }
        .context("Failed to create buffer (resize)")?;

    let mem_requirements = unsafe { logical_device.device.get_buffer_memory_requirements(buffer) };

    let memory_type = find_memory_type(
        instance,
        logical_device.physical_device,
        mem_requirements.memory_type_bits,
        desired_flags,
    )
    .context("Failed to find suitable memory type (resize)")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    let memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
        .context("Failed to allocate buffer memory (resize)")?;

    unsafe { logical_device.device.bind_buffer_memory(buffer, memory, 0) }
        .context("Failed to bind buffer memory (resize)")?;

    let host_mapped: Option<usize> = if cpu_readable && is_storage {
        let device = devices
            .get(&device_handle)
            .context("Buffer's device is invalid for host map")?;
        let ptr = unsafe { device.map_memory2(memory, 0, size) }
            .context("Failed to map CPU_READABLE buffer memory (resize)")?;
        let p = ptr as *mut u8;
        if p.is_null() {
            anyhow::bail!("map_memory2 returned null for CPU_READABLE buffer (resize)");
        }
        Some(p as usize)
    } else {
        None
    };

    Ok((buffer, memory, host_mapped))
}

/// Copy preserved prefix, then zero `[zero_from, new_size)` using a small host-visible source.
fn submit_resize_transfer(
    instance: &ash::Instance,
    device: &types::LogicalDevice,
    old_buf: vk::Buffer,
    new_buf: vk::Buffer,
    copy_len: u64,
    new_size: u64,
    zero_from: u64,
) -> Result<()> {
    const CHUNK: u64 = 4096;
    let tail = new_size.saturating_sub(zero_from);
    let need_zero = tail > 0;

    let (zero_staging, zero_mem) = if need_zero {
        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let staging_info = vk::BufferCreateInfo::default()
            .size(CHUNK)
            .usage(staging_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let zb = unsafe { device.device.create_buffer(&staging_info, None) }.context("resize: zero staging buffer")?;
        let req = unsafe { device.device.get_buffer_memory_requirements(zb) };
        let mt = find_memory_type(
            instance,
            device.physical_device,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .context("resize: zero staging memory type")?;
        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mt);
        let zm = unsafe { device.device.allocate_memory(&ai, None) }.context("resize: zero staging alloc")?;
        unsafe { device.device.bind_buffer_memory(zb, zm, 0) }.context("resize: zero bind")?;
        unsafe {
            let ptr = device.map_memory2(zm, 0, CHUNK).context("resize: map zero staging")? as *mut u8;
            std::ptr::write_bytes(ptr, 0, CHUNK as usize);
            device.unmap_memory2(zm)?;
        }
        (Some(zb), Some(zm))
    } else {
        (None, None)
    };

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe { device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate transfer command buffer (resize)")?;
    let cmd = cmd_buffers[0];

    let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device.device.begin_command_buffer(cmd, &begin_info)?;
        if copy_len > 0 {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: copy_len,
            };
            device
                .device
                .cmd_copy_buffer(cmd, old_buf, new_buf, std::slice::from_ref(&region));
        }
        if let Some(zb) = zero_staging {
            let mut pos = zero_from;
            while pos < new_size {
                let n = (new_size - pos).min(CHUNK);
                let region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: pos,
                    size: n,
                };
                device
                    .device
                    .cmd_copy_buffer(cmd, zb, new_buf, std::slice::from_ref(&region));
                pos += n;
            }
        }

        let mem_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            );
        let dep_info = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
        device.device.cmd_pipeline_barrier2(cmd, &dep_info);

        device.device.end_command_buffer(cmd)?;

        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        device
            .device
            .queue_submit(device.queue, &[submit_info], vk::Fence::null())?;
        device.device.queue_wait_idle(device.queue)?;
        device.device.free_command_buffers(device.command_pool, &cmd_buffers);
    }

    if let (Some(zb), Some(zm)) = (zero_staging, zero_mem) {
        unsafe {
            device.device.destroy_buffer(zb, None);
            device.device.free_memory(zm, None);
        }
    }

    Ok(())
}

/// Resize a root buffer in place. [`BufferHandle`] and bindless slot stay stable.
pub(super) fn resize(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    new_size: u64,
    preserve_contents: bool,
) -> Result<()> {
    let old_state = buffers.get(&buffer_handle).context("Invalid buffer handle")?.clone();

    if old_state.is_view {
        anyhow::bail!("cannot resize buffer views");
    }
    if old_state.transient_heap_suballoc {
        anyhow::bail!("cannot resize buffers sub-allocated from transient heaps");
    }
    if old_state.device_handle != device_handle {
        anyhow::bail!("buffer belongs to a different device");
    }
    if new_size == old_state.size {
        return Ok(());
    }
    if new_size == 0 {
        anyhow::bail!("buffer size must be non-zero");
    }

    let bindless_descriptor_set = devices
        .get(&device_handle)
        .context("Invalid device handle")?
        .bindless_descriptor_set;
    let bindless_index = old_state
        .bindless_index
        .context("buffer resize requires bindless descriptor support")?;

    if old_state.is_sparse && new_size <= old_state.allocation_size {
        anyhow::bail!(
            "sparse buffer: growth within virtual capacity must use set_buffer_logical_size, not resize_buffer"
        );
    }

    let is_storage = old_state.is_storage;

    let (new_buffer, new_memory, new_host_mapped) =
        allocate_vk_buffer_memory(instance, devices, device_handle, new_size, is_storage, old_state.flags)?;

    let copy_len = if preserve_contents {
        old_state.size.min(new_size)
    } else {
        0
    };
    let zero_from = if preserve_contents { copy_len } else { new_size };

    let logical_ref = devices.get(&device_handle).context("Invalid device handle")?;
    submit_resize_transfer(
        instance,
        logical_ref,
        old_state.buffer,
        new_buffer,
        copy_len,
        new_size,
        zero_from,
    )?;

    if let Some(descriptor_set) = bindless_descriptor_set {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(new_buffer)
            .offset(0)
            .range(new_size);

        let binding = if is_storage {
            types::bindless_bindings::STORAGE_BUFFERS
        } else {
            types::bindless_bindings::UNIFORM_BUFFERS
        };

        let descriptor_type = if is_storage {
            vk::DescriptorType::STORAGE_BUFFER
        } else {
            vk::DescriptorType::UNIFORM_BUFFER
        };

        let logical_device = devices.get(&device_handle).unwrap();
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(binding)
            .dst_array_element(bindless_index)
            .descriptor_type(descriptor_type)
            .buffer_info(std::slice::from_ref(&buffer_info));
        unsafe {
            logical_device
                .device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    }

    if old_state.host_mapped.is_some() && !old_state.is_sparse {
        let dev = devices.get(&device_handle).unwrap();
        if let Err(e) = unsafe { dev.unmap_memory2(old_state.memory) } {
            tracing::warn!(?e, "unmap_memory2 failed for old buffer during resize");
        }
    }

    let barrier = devices
        .get(&device_handle)
        .unwrap()
        .timeline_next
        .load(Ordering::Relaxed)
        .saturating_sub(1);
    let pending = if old_state.is_sparse {
        let binds = sparse::collect_sparse_binds_for_teardown(old_state.sparse_block_size, &old_state.sparse_pages);
        types::PendingDeletion::ReplacedSparseBufferGpu {
            buffer: old_state.buffer,
            allocation_size: old_state.allocation_size,
            block_size: old_state.sparse_block_size,
            binds,
            staging_buffer: old_state.staging_buffer,
            staging_memory: old_state.staging_memory,
        }
    } else {
        types::PendingDeletion::ReplacedBufferGpu {
            buffer: old_state.buffer,
            memory: old_state.memory,
            staging_buffer: old_state.staging_buffer,
            staging_memory: old_state.staging_memory,
        }
    };
    devices
        .get(&device_handle)
        .unwrap()
        .deletion_queue
        .lock()
        .unwrap()
        .queue(barrier, pending);

    let old_parent_vk = old_state.buffer;

    *buffers.get_mut(&buffer_handle).unwrap() = BufferState {
        device_handle,
        buffer: new_buffer,
        memory: new_memory,
        size: new_size,
        allocation_size: new_size,
        bindless_index: old_state.bindless_index,
        is_storage: old_state.is_storage,
        element_stride: old_state.element_stride,
        staging_buffer: None,
        staging_memory: None,
        is_view: false,
        host_mapped: new_host_mapped,
        flags: old_state.flags,
        transient_heap_suballoc: false,
        view_byte_offset: None,
        is_sparse: false,
        sparse_block_size: 0,
        sparse_pages: Vec::new(),
    };

    let view_handles: Vec<BufferHandle> = buffers
        .iter()
        .filter(|(h, st)| **h != buffer_handle && st.is_view && st.buffer == old_parent_vk)
        .map(|(h, _)| *h)
        .collect();

    if let Some(descriptor_set) = bindless_descriptor_set {
        for vh in view_handles {
            let (off, view_size, idx) = {
                let st = buffers.get(&vh).context("view missing")?;
                (
                    st.view_byte_offset.context("internal: view byte offset")?,
                    st.size,
                    st.bindless_index.context("internal: view bindless")?,
                )
            };
            let range = view_size.max(1);
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(new_buffer)
                .offset(off)
                .range(range);
            let logical_device = devices.get(&device_handle).unwrap();
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(types::bindless_bindings::STORAGE_BUFFERS)
                .dst_array_element(idx)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));
            unsafe {
                logical_device
                    .device
                    .update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }
            buffers.get_mut(&vh).unwrap().buffer = new_buffer;
        }
    } else {
        for vh in view_handles {
            buffers.get_mut(&vh).unwrap().buffer = new_buffer;
        }
    }

    Ok(())
}

/// Destroy a buffer, queueing both the Vk resources and the bindless descriptor
/// index for deferred deletion after in-flight GPU work completes.
/// For views, only the descriptor index is deferred — the underlying VkBuffer/memory
/// belongs to the parent and is not freed.
pub(super) fn destroy(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
) {
    if let Some(buffer) = buffers.remove(&buffer_handle) {
        if let Some(device) = devices.get(&buffer.device_handle) {
            let barrier = device.timeline_next.load(Ordering::Relaxed).saturating_sub(1);

            if buffer.is_view {
                device
                    .deletion_queue
                    .lock()
                    .unwrap()
                    .queue(barrier, types::PendingDeletion::BufferView { buffer_handle });
                return;
            }

            if buffer.transient_heap_suballoc {
                device.ledger.lock().unwrap().reclaim_buffer_slots(buffer_handle);
                unsafe {
                    device.device.destroy_buffer(buffer.buffer, None);
                }
                return;
            }
            if buffer.host_mapped.is_some() && !buffer.is_sparse {
                if let Err(e) = unsafe { device.unmap_memory2(buffer.memory) } {
                    tracing::warn!(?e, "unmap_memory2 failed for CPU_READABLE buffer on destroy");
                }
            }
            let sparse_teardown = if buffer.is_sparse {
                Some(types::SparseBufferTeardown {
                    allocation_size: buffer.allocation_size,
                    block_size: buffer.sparse_block_size,
                    binds: sparse::collect_sparse_binds_for_teardown(buffer.sparse_block_size, &buffer.sparse_pages),
                })
            } else {
                None
            };
            device.deletion_queue.lock().unwrap().queue(
                barrier,
                types::PendingDeletion::Buffer {
                    buffer_handle,
                    buffer: buffer.buffer,
                    memory: if buffer.is_sparse {
                        vk::DeviceMemory::null()
                    } else {
                        buffer.memory
                    },
                    staging_buffer: buffer.staging_buffer,
                    staging_memory: buffer.staging_memory,
                    sparse_teardown,
                },
            );
        }
    }
}

/// Create a view into a sub-region of an existing buffer.
///
/// The view gets its own bindless descriptor at `[offset, offset+size)` of the parent.
/// It shares the parent's VkBuffer and staging resources.
pub(super) fn create_view(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    next_buffer_handle: &mut BufferHandle,
    parent_handle: BufferHandle,
    offset: u64,
    size: u64,
    element_stride: Option<u32>,
) -> Result<BufferHandle> {
    let parent = buffers.get(&parent_handle).context("Invalid parent buffer handle")?;

    if offset + size > parent.size {
        anyhow::bail!(
            "View [{}, {}) exceeds parent buffer size {}",
            offset,
            offset + size,
            parent.size
        );
    }

    if !parent.is_storage {
        anyhow::bail!("Buffer views are only supported for storage (Scattered) buffers");
    }

    let device_handle = parent.device_handle;
    let vk_buffer = parent.buffer;
    let is_storage = parent.is_storage;
    let parent_flags = parent.flags;

    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

    let bindless_descriptor_set = logical_device.bindless_descriptor_set;

    let bindless_index = if size == 0 {
        // A zero-byte view has no addressable data; VkDescriptorBufferInfo.range must be > 0
        // (VUID-VkDescriptorBufferInfo-range-00341), so skip registration entirely.
        // bindless_handle() returns None, which is correct — zero-size views cannot
        // be bound to shaders.
        None
    } else {
        let handle_for_registry = *next_buffer_handle;
        let index = logical_device
            .ledger
            .lock()
            .unwrap()
            .resource_registry
            .register_buffer(handle_for_registry, is_storage);

        if let Some(descriptor_set) = bindless_descriptor_set {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(vk_buffer)
                .offset(offset)
                .range(size);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(types::bindless_bindings::STORAGE_BUFFERS)
                .dst_array_element(index)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));

            unsafe {
                logical_device
                    .device
                    .update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }

            tracing::trace!(
                "Registered buffer view {} at bindless index {} (parent={}, offset={}, size={})",
                handle_for_registry,
                index,
                parent_handle,
                offset,
                size
            );
        }

        Some(index)
    };

    let handle = *next_buffer_handle;
    *next_buffer_handle += 1;

    buffers.insert(
        handle,
        BufferState {
            device_handle,
            buffer: vk_buffer,
            memory: vk::DeviceMemory::null(),
            size,
            allocation_size: parent.allocation_size,
            bindless_index,
            is_storage,
            element_stride,
            staging_buffer: None,
            staging_memory: None,
            is_view: true,
            host_mapped: None,
            flags: parent_flags,
            transient_heap_suballoc: false,
            view_byte_offset: Some(offset),
            is_sparse: false,
            sparse_block_size: 0,
            sparse_pages: Vec::new(),
        },
    );

    Ok(handle)
}

/// Lazily create the HOST_VISIBLE staging buffer for a DEVICE_LOCAL storage buffer.
pub(super) fn ensure_staging(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
) -> Result<()> {
    let buffer = buffers.get(&buffer_handle).context("Invalid buffer handle")?;
    if !buffer.is_storage || buffer.staging_buffer.is_some() || buffer.flags.contains(BufferFlags::CPU_READABLE) {
        return Ok(());
    }
    let size = buffer.size;
    let device = devices
        .get(&buffer.device_handle)
        .context("Buffer's device is invalid")?;

    let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let staging_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(staging_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let stg_buf =
        unsafe { device.device.create_buffer(&staging_info, None) }.context("Failed to create staging buffer")?;

    let stg_reqs = unsafe { device.device.get_buffer_memory_requirements(stg_buf) };

    let stg_mem_type = find_memory_type(
        instance,
        device.physical_device,
        stg_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .context("Failed to find HOST_VISIBLE memory type for staging buffer")?;

    let stg_alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(stg_reqs.size)
        .memory_type_index(stg_mem_type);

    let stg_mem = unsafe { device.device.allocate_memory(&stg_alloc, None) }
        .context("Failed to allocate staging buffer memory")?;

    unsafe { device.device.bind_buffer_memory(stg_buf, stg_mem, 0) }.context("Failed to bind staging buffer memory")?;

    let buf = buffers.get_mut(&buffer_handle).unwrap();
    buf.staging_buffer = Some(stg_buf);
    buf.staging_memory = Some(stg_mem);
    Ok(())
}

/// Write data to a buffer at the specified offset.
///
/// For DEVICE_LOCAL storage buffers, lazily creates a staging buffer then
/// copies via GPU. For HOST_VISIBLE uniform buffers, maps directly.
pub(super) fn write(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
    offset: u64,
    data: &[u8],
) -> Result<()> {
    // vkMapMemory2 and vkCmdCopyBuffer both require size > 0.
    if data.is_empty() {
        return Ok(());
    }

    {
        let buffer = buffers.get(&buffer_handle).context("Invalid buffer handle")?;
        if offset + data.len() as u64 > buffer.size {
            anyhow::bail!("Write would exceed buffer bounds");
        }
    }

    {
        let buffer = buffers.get(&buffer_handle).context("Invalid buffer handle")?;
        if let Some(base) = buffer.host_mapped {
            let p = base as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), p.add(offset as usize), data.len());
            }
            return Ok(());
        }
    }

    // Lazily create staging buffer for storage buffers
    ensure_staging(instance, devices, buffers, buffer_handle)?;

    let buffer = buffers.get(&buffer_handle).context("Invalid buffer handle")?;

    let device = devices
        .get(&buffer.device_handle)
        .context("Buffer's device is invalid")?;

    if let (Some(stg_buf), Some(stg_mem)) = (buffer.staging_buffer, buffer.staging_memory) {
        // DEVICE_LOCAL path: write to staging, then GPU copy
        unsafe {
            let ptr = device
                .map_memory2(stg_mem, offset, data.len() as u64)
                .context("Failed to map staging buffer")?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            device
                .unmap_memory2(stg_mem)
                .context("Failed to unmap staging buffer")?;
        }

        submit_copy(device, stg_buf, buffer.buffer, offset, offset, data.len() as u64)?;
    } else {
        // HOST_VISIBLE path: direct map
        unsafe {
            let ptr = device
                .map_memory2(buffer.memory, offset, data.len() as u64)
                .context("Failed to map buffer memory")?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            device
                .unmap_memory2(buffer.memory)
                .context("Failed to unmap buffer memory")?;
        }
    }

    Ok(())
}

/// Get the size of a buffer in bytes.
pub(super) fn size(buffers: &HashMap<BufferHandle, BufferState>, buffer_handle: BufferHandle) -> u64 {
    buffers.get(&buffer_handle).map(|b| b.size).unwrap_or(0)
}

/// Get the bindless descriptor index for a buffer, if any.
pub(super) fn bindless_index(buffers: &HashMap<BufferHandle, BufferState>, buffer_handle: BufferHandle) -> Option<u32> {
    buffers.get(&buffer_handle).and_then(|b| b.bindless_index)
}

/// Read buffer contents to CPU. Copies from offset 0 for length output.len().
///
/// For DEVICE_LOCAL storage buffers, lazily creates a staging buffer, then issues
/// a GPU copy and maps. For HOST_VISIBLE uniform buffers, maps directly.
pub(super) fn read_to_cpu(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    output: &mut [u8],
) -> Result<()> {
    let _tz = crate::tracy_zone!("vk.buffer.read_to_cpu");
    {
        let _validate = crate::tracy_zone!("vk.buffer.read_to_cpu.validate");
        let buffer = buffers.get(&buffer_handle).context("Invalid buffer handle")?;
        if buffer.device_handle != device_handle {
            anyhow::bail!("Buffer belongs to different device");
        }
        let len = output.len() as u64;
        if len > buffer.size {
            anyhow::bail!("Read would exceed buffer bounds");
        }
        if let Some(base) = buffer.host_mapped {
            let _copy = crate::tracy_zone!("vk.buffer.read_to_cpu.host_mapped_copy");
            let p = base as *const u8;
            unsafe {
                std::ptr::copy_nonoverlapping(p, output.as_mut_ptr(), output.len());
            }
            return Ok(());
        }
    }

    // Lazily create staging buffer for storage buffers
    {
        let _staging = crate::tracy_zone!("vk.buffer.read_to_cpu.ensure_staging");
        ensure_staging(instance, devices, buffers, buffer_handle)?;
    }

    let buffer = {
        let _lookup = crate::tracy_zone!("vk.buffer.read_to_cpu.lookup_after_staging");
        buffers.get(&buffer_handle).context("Invalid buffer handle")?
    };

    let device = devices.get(&device_handle).context("Invalid device handle")?;

    if buffer.device_handle != device_handle {
        anyhow::bail!("Buffer belongs to different device");
    }

    let len = output.len() as u64;
    if len > buffer.size {
        anyhow::bail!("Read would exceed buffer bounds");
    }

    if let (Some(stg_buf), Some(stg_mem)) = (buffer.staging_buffer, buffer.staging_memory) {
        // DEVICE_LOCAL path: GPU copy to staging, then map staging
        {
            let _copy = crate::tracy_zone!("vk.buffer.read_to_cpu.submit_copy");
            submit_copy(device, buffer.buffer, stg_buf, 0, 0, len)?;
        }

        unsafe {
            let _map = crate::tracy_zone!("vk.buffer.read_to_cpu.staging_map_copy_unmap");
            let ptr = device
                .map_memory2(stg_mem, 0, len)
                .context("Failed to map staging buffer for readback")?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), output.len());
            device
                .unmap_memory2(stg_mem)
                .context("Failed to unmap staging buffer")?;
        }
    } else {
        // HOST_VISIBLE path: direct map
        unsafe {
            let _map = crate::tracy_zone!("vk.buffer.read_to_cpu.direct_map_copy_unmap");
            let ptr = device
                .map_memory2(buffer.memory, 0, len)
                .context("Failed to map buffer memory")?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), output.len());
            device
                .unmap_memory2(buffer.memory)
                .context("Failed to unmap buffer memory")?;
        }
    }

    Ok(())
}

/// Fill buffer region with zeros. If size is 0, clears from offset to end of buffer.
pub(super) fn clear(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &HashMap<BufferHandle, BufferState>,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    offset: u64,
    size: u64,
) -> Result<()> {
    let buffer = buffers.get(&buffer_handle).context("Invalid buffer handle")?;

    let device = devices.get(&device_handle).context("Invalid device handle")?;

    if buffer.device_handle != device_handle {
        anyhow::bail!("Buffer belongs to different device");
    }

    let clear_size = super::super::shared::resolve_clear_size(buffer.size, offset, size);

    if offset + clear_size > buffer.size {
        anyhow::bail!("Clear would exceed buffer bounds");
    }

    if clear_size == 0 {
        return Ok(());
    }

    // Allocate command buffer
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers =
        unsafe { device.device.allocate_command_buffers(&alloc_info) }.context("Failed to allocate command buffer")?;
    let cmd = cmd_buffers[0];

    // Record fill command
    let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device.device.begin_command_buffer(cmd, &begin_info)?;
        device.device.cmd_fill_buffer(cmd, buffer.buffer, offset, clear_size, 0);

        let mem_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
        let dep_info = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
        device.device.cmd_pipeline_barrier2(cmd, &dep_info);

        device.device.end_command_buffer(cmd)?;
    }

    // Submit and wait
    let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
    unsafe {
        device
            .device
            .queue_submit(device.queue, &[submit_info], vk::Fence::null())?;
        device.device.queue_wait_idle(device.queue)?;
        device.device.free_command_buffers(device.command_pool, &cmd_buffers);
    }

    Ok(())
}
