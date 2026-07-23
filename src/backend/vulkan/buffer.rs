//! Buffer management logic.

use super::sparse;
use super::types::{self, BufferState, SharedBufferTable};
use super::utils::{find_memory_type, with_buffer_sharing};
use super::{BufferHandle, DeviceHandle};
use crate::backend::BufferKind;
use crate::types::BufferFlags;
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::num::NonZeroU64;

/// Submit a one-shot vkCmdCopyBuffer between two buffers and wait for completion.
fn submit_copy(
    device: &types::LogicalDevice,
    src: vk::Buffer,
    dst: vk::Buffer,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
) -> Result<()> {
    let cmd = device.acquire_device_cmd_buffer()?;
    let cmd_buffers = [cmd];

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
        device.synchronized_queue_submit(&[submit_info], vk::Fence::null())?;
        device.synchronized_queue_wait_idle()?;
        device.recycle_device_cmd_buffer(cmd);
    }

    Ok(())
}

/// Create a buffer with the given size and access pattern.
#[allow(clippy::too_many_arguments)]
pub(super) fn create(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
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
    let cpu_writable = flags.contains(BufferFlags::CPU_WRITABLE);
    if cpu_readable && !is_storage {
        anyhow::bail!("BufferFlags::CPU_READABLE is only valid for BufferKind::Scattered (storage) buffers");
    }
    if cpu_writable && !is_storage {
        anyhow::bail!("BufferFlags::CPU_WRITABLE is only valid for BufferKind::Scattered (storage) buffers");
    }
    if cpu_writable && (cpu_readable || flags.contains(BufferFlags::GPU_ONLY)) {
        anyhow::bail!("BufferFlags::CPU_WRITABLE cannot be combined with CPU_READABLE or GPU_ONLY");
    }

    let bindless_descriptor_set = logical_device.bindless_descriptor_set;

    let qf = logical_device.concurrent_queue_families();
    let buffer_info = with_buffer_sharing(
        vk::BufferCreateInfo::default().size(allocation_size).usage(vk_usage),
        qf.as_ref(),
    );

    let buffer =
        unsafe { logical_device.device.create_buffer(&buffer_info, None) }.context("Failed to create buffer")?;

    let mem_requirements = unsafe { logical_device.device.get_buffer_memory_requirements(buffer) };

    // Storage buffers → DEVICE_LOCAL for GPU compute performance, unless CPU_READABLE/CPU_WRITABLE
    // (host-visible storage for persistent map + stable UAV bindless use).
    // Uniform buffers → HOST_VISIBLE|HOST_COHERENT for frequent CPU writes.
    let desired_flags = if is_storage {
        if cpu_readable || cpu_writable {
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

    let host_mapped: Option<usize> = if (cpu_readable || cpu_writable) && is_storage {
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

    let handle = buffers.write().unwrap().alloc_handle();

    // Register buffer in bindless descriptor set (UNIFORM or STORAGE)
    let bindless_index = {
        let logical_device = devices.get(&device_handle).unwrap();
        let index = logical_device
            .descriptors
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

    buffers.write().unwrap().entries.insert(
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
            is_withdraw_staging: false,
            texture_copy_footprint: None,
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
    buffers: &SharedBufferTable,
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

    let qf = ld.concurrent_queue_families();
    let buffer_info = with_buffer_sharing(
        vk::BufferCreateInfo::default()
            .size(allocation_size)
            .usage(vk_usage)
            .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY),
        qf.as_ref(),
    );

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

    sparse::queue_bind_sparse_sync(dev, &ld.queue_lock, bind_queue, buffer, &binds)?;
    drop(pool_guard);

    let bindless_descriptor_set = ld.bindless_descriptor_set;
    let handle = buffers.write().unwrap().alloc_handle();

    let bindless_index = {
        let index = ld
            .descriptors
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

    buffers.write().unwrap().entries.insert(
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
            is_withdraw_staging: false,
            texture_copy_footprint: None,
        },
    );

    Ok(handle)
}

/// Hint unused sparse pages at and above `offset` (bytes).
pub(super) fn hint_unused_above(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
    buffer_handle: BufferHandle,
    offset: u64,
) {
    let (device_handle, vkbuf, _alloc, block, first_page, total_pages) = {
        let buffers_read = buffers.read().unwrap();
        let Some(buf) = buffers_read.entries.get(&buffer_handle) else {
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
        (device_handle, vkbuf, alloc, block, first_page, total_pages)
    };

    {
        let Some(ld) = devices.get(&device_handle) else {
            return;
        };
        let bind_queue = ld.sparse_binding_queue;
        let dev: &ash::Device = &ld.device;
        let mut buffers_write = buffers.write().unwrap();
        let sparse_pages = &mut buffers_write
            .entries
            .get_mut(&buffer_handle)
            .expect("buffer missing")
            .sparse_pages;

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
        if let Err(e) = sparse::queue_bind_sparse_sync(dev, &ld.queue_lock, bind_queue, vkbuf, &binds) {
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
pub(super) fn capacity(buffers: &SharedBufferTable, buffer_handle: BufferHandle) -> u64 {
    buffers
        .read()
        .unwrap()
        .entries
        .get(&buffer_handle)
        .map(|b| b.allocation_size)
        .unwrap_or(0)
}

pub(super) fn set_logical_size(
    state: &super::types::VulkanState,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    new_logical_size: u64,
) -> Result<()> {
    let is_sparse = {
        let buffers_guard = buffers.read().unwrap();
        buffers_guard
            .entries
            .get(&buffer_handle)
            .map(|b| b.is_sparse)
            .unwrap_or(false)
    };
    if is_sparse {
        return set_logical_size_sparse(state, devices, buffers, device_handle, buffer_handle, new_logical_size);
    }

    let (bindless_index, is_storage, vkbuf, old_logical) = {
        let buffers_guard = buffers.read().unwrap();
        let buf = buffers_guard
            .entries
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;
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

    buffers.write().unwrap().entries.get_mut(&buffer_handle).unwrap().size = new_logical_size;

    if old_logical == new_logical_size {
        return Ok(());
    }

    // When growing, the existing staging buffer was sized for the old (smaller) logical
    // size. Invalidate it so ensure_staging recreates it at the new size.
    if new_logical_size > old_logical {
        let (old_stg_buf, old_stg_mem) = {
            let mut buffers_write = buffers.write().unwrap();
            let buf = buffers_write.entries.get_mut(&buffer_handle).unwrap();
            (buf.staging_buffer.take(), buf.staging_memory.take())
        };
        if let (Some(stg_buf), Some(stg_mem)) = (old_stg_buf, old_stg_mem) {
            let ld = devices.get(&device_handle).unwrap();
            let requirements = super::context::reclamation_requirements(
                state,
                device_handle,
                super::context::destroy_attribution_context(state, device_handle),
            );
            ld.deletion_queue.lock().unwrap().queue(
                requirements,
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
    state: &super::types::VulkanState,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    new_logical_size: u64,
) -> Result<()> {
    let (bindless_index, is_storage, vkbuf, old_logical, block) = {
        let buffers_guard = buffers.read().unwrap();
        let buf = buffers_guard
            .entries
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;
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

        let mut buffers_write = buffers.write().unwrap();
        let sparse_pages = &mut buffers_write
            .entries
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
            sparse::queue_bind_sparse_sync(dev, &ld.queue_lock, bind_queue, vkbuf, &binds)?;
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
                sparse::queue_bind_sparse_sync(dev, &ld.queue_lock, bind_queue, vkbuf, &binds)?;
            }
            for (mem, off) in to_free {
                pool.free_page(mem, off);
            }
        }
    }

    buffers.write().unwrap().entries.get_mut(&buffer_handle).unwrap().size = new_logical_size;

    // When growing, the existing staging buffer was sized for the old (smaller) logical
    // size. Invalidate it so ensure_staging recreates it at the new size.
    if new_logical_size > old_logical {
        let (old_stg_buf, old_stg_mem) = {
            let mut buffers_write = buffers.write().unwrap();
            let buf = buffers_write.entries.get_mut(&buffer_handle).unwrap();
            (buf.staging_buffer.take(), buf.staging_memory.take())
        };
        if let (Some(stg_buf), Some(stg_mem)) = (old_stg_buf, old_stg_mem) {
            let ld = devices.get(&device_handle).unwrap();
            let requirements = super::context::reclamation_requirements(
                state,
                device_handle,
                super::context::destroy_attribution_context(state, device_handle),
            );
            ld.deletion_queue.lock().unwrap().queue(
                requirements,
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
    let cpu_writable = flags.contains(BufferFlags::CPU_WRITABLE);
    if cpu_readable && !is_storage {
        anyhow::bail!("BufferFlags::CPU_READABLE is only valid for BufferKind::Scattered (storage) buffers");
    }
    if cpu_writable && !is_storage {
        anyhow::bail!("BufferFlags::CPU_WRITABLE is only valid for BufferKind::Scattered (storage) buffers");
    }
    if cpu_writable && (cpu_readable || flags.contains(BufferFlags::GPU_ONLY)) {
        anyhow::bail!("BufferFlags::CPU_WRITABLE cannot be combined with CPU_READABLE or GPU_ONLY");
    }

    let desired_flags = if is_storage {
        if cpu_readable || cpu_writable {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        }
    } else {
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    };

    let qf = logical_device.concurrent_queue_families();
    let buffer_info = with_buffer_sharing(vk::BufferCreateInfo::default().size(size).usage(vk_usage), qf.as_ref());

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

    let host_mapped: Option<usize> = if (cpu_readable || cpu_writable) && is_storage {
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
        let qf = device.concurrent_queue_families();
        let staging_info = with_buffer_sharing(
            vk::BufferCreateInfo::default().size(CHUNK).usage(staging_usage),
            qf.as_ref(),
        );
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

    let cmd = device.acquire_device_cmd_buffer()?;
    let cmd_buffers = [cmd];

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
        device.synchronized_queue_submit(&[submit_info], vk::Fence::null())?;
        device.synchronized_queue_wait_idle()?;
        device.recycle_device_cmd_buffer(cmd);
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
#[allow(clippy::too_many_arguments)]
pub(super) fn resize(
    state: &super::types::VulkanState,
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    new_size: u64,
    preserve_contents: bool,
) -> Result<()> {
    let old_state = {
        let buffers_guard = buffers.read().unwrap();
        buffers_guard
            .entries
            .get(&buffer_handle)
            .context("Invalid buffer handle")?
            .clone()
    };

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

    let ctx_h = super::context::destroy_attribution_context(state, device_handle);
    let base = super::context::reclamation_requirements(state, device_handle, ctx_h);
    let requirements = {
        let ld = devices.get(&device_handle).context("resize: queue deletion")?;
        let registry = ld.descriptors.lock().unwrap();
        registry.bindless_retirement_requirements_for_buffer(buffer_handle, base)
    };
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
        .queue(requirements, pending);

    let old_parent_vk = old_state.buffer;

    *buffers.write().unwrap().entries.get_mut(&buffer_handle).unwrap() = BufferState {
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
        is_withdraw_staging: false,
        texture_copy_footprint: None,
    };

    let view_handles: Vec<BufferHandle> = buffers
        .read()
        .unwrap()
        .entries
        .iter()
        .filter(|(h, st)| **h != buffer_handle && st.is_view && st.buffer == old_parent_vk)
        .map(|(h, _)| *h)
        .collect();

    if let Some(descriptor_set) = bindless_descriptor_set {
        for vh in view_handles {
            let (off, view_size, idx) = {
                let buffers_guard = buffers.read().unwrap();
                let st = buffers_guard.entries.get(&vh).context("view missing")?;
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
            buffers.write().unwrap().entries.get_mut(&vh).unwrap().buffer = new_buffer;
        }
    } else {
        for vh in view_handles {
            buffers.write().unwrap().entries.get_mut(&vh).unwrap().buffer = new_buffer;
        }
    }

    Ok(())
}

/// Destroy a buffer, queueing both the Vk resources and the bindless descriptor
/// index for deferred deletion after in-flight GPU work completes.
/// For views, only the descriptor index is deferred — the underlying VkBuffer/memory
/// belongs to the parent and is not freed.
pub(super) fn destroy(state: &super::types::VulkanState, buffer_handle: BufferHandle) {
    let buffer = {
        let mut buffers = state.buffers.write().unwrap();
        buffers.entries.remove(&buffer_handle)
    };
    let Some(buffer) = buffer else {
        return;
    };
    if let Some(device) = state.devices.get(&buffer.device_handle) {
        let slots = device.descriptors.lock().unwrap().buffer_slot_keys(buffer_handle);
        super::compute::evict_retained_graphs_using_slots(state, buffer.device_handle, &slots);

        let ctx_h = super::context::destroy_attribution_context(state, buffer.device_handle);
        let base = super::context::reclamation_requirements(state, buffer.device_handle, ctx_h);
        let requirements = {
            let registry = device.descriptors.lock().unwrap();
            registry.bindless_retirement_requirements_for_buffer(buffer_handle, base)
        };

        if buffer.is_view {
            let deletion = types::PendingDeletion::BufferView { buffer_handle };
            queue_pending_deletion(device, requirements, deletion);
            return;
        }

        if buffer.transient_heap_suballoc {
            device.descriptors.lock().unwrap().reclaim_buffer_slots(buffer_handle);
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
        let deletion = types::PendingDeletion::Buffer {
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
        };
        queue_pending_deletion(device, requirements, deletion);
    }
}

/// Queue bindless-tracked buffer/view teardown on the device-level requirement-gated queue.
fn queue_pending_deletion(
    device: &types::SharedLogicalDevice,
    requirements: Vec<(super::ContextHandle, u64)>,
    deletion: types::PendingDeletion,
) {
    device.deletion_queue.lock().unwrap().queue(requirements, deletion);
}

/// Create a view into a sub-region of an existing buffer.
///
/// The view gets its own bindless descriptor at `[offset, offset+size)` of the parent.
/// It shares the parent's VkBuffer and staging resources.
pub(super) fn create_view(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
    parent_handle: BufferHandle,
    offset: u64,
    size: u64,
    element_stride: Option<u32>,
) -> Result<BufferHandle> {
    let (device_handle, vk_buffer, is_storage, parent_flags, parent_allocation_size) = {
        let buffers_guard = buffers.read().unwrap();
        let parent = buffers_guard
            .entries
            .get(&parent_handle)
            .context("Invalid parent buffer handle")?;

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

        (
            parent.device_handle,
            parent.buffer,
            parent.is_storage,
            parent.flags,
            parent.allocation_size,
        )
    };

    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

    let bindless_descriptor_set = logical_device.bindless_descriptor_set;

    let handle = buffers.write().unwrap().alloc_handle();

    let bindless_index = if size == 0 {
        // A zero-byte view has no addressable data; VkDescriptorBufferInfo.range must be > 0
        // (VUID-VkDescriptorBufferInfo-range-00341), so skip registration entirely.
        // bindless_handle() returns None, which is correct — zero-size views cannot
        // be bound to shaders.
        None
    } else {
        let handle_for_registry = handle;
        let index = logical_device
            .descriptors
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

    buffers.write().unwrap().entries.insert(
        handle,
        BufferState {
            device_handle,
            buffer: vk_buffer,
            memory: vk::DeviceMemory::null(),
            size,
            allocation_size: parent_allocation_size,
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
            is_withdraw_staging: false,
            texture_copy_footprint: None,
        },
    );

    Ok(handle)
}

/// Lazily create the HOST_VISIBLE staging buffer for a DEVICE_LOCAL storage buffer.
pub(super) fn ensure_staging(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
    buffer_handle: BufferHandle,
) -> Result<()> {
    let (size, device_handle, needs_staging) = {
        let buffers_guard = buffers.read().unwrap();
        let buffer = buffers_guard
            .entries
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;
        let needs =
            buffer.is_storage && buffer.staging_buffer.is_none() && !buffer.flags.contains(BufferFlags::CPU_READABLE);
        (buffer.size, buffer.device_handle, needs)
    };
    if !needs_staging {
        return Ok(());
    }

    let device = devices.get(&device_handle).context("Buffer's device is invalid")?;

    let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let qf = device.concurrent_queue_families();
    let staging_info = with_buffer_sharing(
        vk::BufferCreateInfo::default().size(size).usage(staging_usage),
        qf.as_ref(),
    );

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

    let mut buffers_write = buffers.write().unwrap();
    let buf = buffers_write.entries.get_mut(&buffer_handle).unwrap();
    buf.staging_buffer = Some(stg_buf);
    buf.staging_memory = Some(stg_mem);
    Ok(())
}

/// View tightly packed bytes in a CPU-writable storage buffer's host mapping.
pub(super) fn cpu_writable_flat_slice(
    buffers: &SharedBufferTable,
    buffer_handle: BufferHandle,
    offset: u64,
    len: usize,
) -> Result<&[u8]> {
    let (host_mapped, size, flags) = {
        let buffers_guard = buffers.read().unwrap();
        let buffer = buffers_guard
            .entries
            .get(&buffer_handle)
            .context("cpu_writable_flat_slice: invalid buffer handle")?;
        if !buffer.flags.contains(BufferFlags::CPU_WRITABLE) {
            anyhow::bail!("cpu_writable_flat_slice: buffer is not CPU_WRITABLE");
        }
        if offset + len as u64 > buffer.size {
            anyhow::bail!("cpu_writable_flat_slice: slice exceeds buffer bounds");
        }
        (
            buffer
                .host_mapped
                .context("cpu_writable_flat_slice: missing host mapping")?,
            buffer.size,
            buffer.flags,
        )
    };
    let _ = (size, flags);
    Ok(unsafe { std::slice::from_raw_parts((host_mapped as *const u8).add(offset as usize), len) })
}

/// Write data to a buffer at the specified offset.
///
/// For DEVICE_LOCAL storage buffers, lazily creates a staging buffer then
/// copies via GPU. For HOST_VISIBLE uniform buffers, maps directly.
pub(super) fn write(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
    buffer_handle: BufferHandle,
    offset: u64,
    data: &[u8],
) -> Result<()> {
    // vkMapMemory2 and vkCmdCopyBuffer both require size > 0.
    if data.is_empty() {
        return Ok(());
    }

    {
        let buffers_guard = buffers.read().unwrap();
        let buffer = buffers_guard
            .entries
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;
        if offset + data.len() as u64 > buffer.size {
            anyhow::bail!("Write would exceed buffer bounds");
        }
    }

    {
        let buffers_guard = buffers.read().unwrap();
        let buffer = buffers_guard
            .entries
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;
        if let Some(base) = buffer.host_mapped {
            let p = base as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), p.add(offset as usize), data.len());
            }
            return Ok(());
        }
    }
    ensure_staging(instance, devices, buffers, buffer_handle)?;

    let buffers_guard = buffers.read().unwrap();
    let buffer = buffers_guard
        .entries
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

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
pub(super) fn size(buffers: &SharedBufferTable, buffer_handle: BufferHandle) -> u64 {
    buffers
        .read()
        .unwrap()
        .entries
        .get(&buffer_handle)
        .map(|b| b.size)
        .unwrap_or(0)
}

/// Get the bindless descriptor index for a buffer, if any.
pub(super) fn bindless_index(buffers: &SharedBufferTable, buffer_handle: BufferHandle) -> Option<u32> {
    buffers
        .read()
        .unwrap()
        .entries
        .get(&buffer_handle)
        .and_then(|b| b.bindless_index)
}

/// Fill buffer region with zeros. If size is 0, clears from offset to end of buffer.
pub(super) fn clear(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    offset: u64,
    size: u64,
) -> Result<()> {
    let buffers_guard = buffers.read().unwrap();
    let buffer = buffers_guard
        .entries
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

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
    let cmd = device.acquire_device_cmd_buffer()?;
    let cmd_buffers = [cmd];

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
    device.synchronized_queue_submit(&[submit_info], vk::Fence::null())?;
    device.synchronized_queue_wait_idle()?;
    device.recycle_device_cmd_buffer(cmd);

    Ok(())
}

/// Allocate a persistently mapped host-visible staging buffer for withdraw staging.
pub(super) fn alloc_readback_buffer(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
    device_handle: DeviceHandle,
    size: u64,
) -> Result<BufferHandle> {
    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;
    let qf = logical_device.concurrent_queue_families();
    let buffer_info = with_buffer_sharing(
        vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST),
        qf.as_ref(),
    );
    let buffer = unsafe { logical_device.device.create_buffer(&buffer_info, None) }
        .context("Failed to create readback buffer")?;
    let mem_requirements = unsafe { logical_device.device.get_buffer_memory_requirements(buffer) };
    let desired_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let memory_type = find_memory_type(
        instance,
        logical_device.physical_device,
        mem_requirements.memory_type_bits,
        desired_flags,
    )
    .context("Failed to find readback memory type")?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);
    let memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
        .context("Failed to allocate readback buffer memory")?;
    unsafe { logical_device.device.bind_buffer_memory(buffer, memory, 0) }
        .context("Failed to bind readback buffer memory")?;
    let ptr =
        unsafe { logical_device.map_memory2(memory, 0, size) }.context("Failed to map withdraw staging buffer")?;
    let host_mapped = Some(ptr as usize);

    let handle = buffers.write().unwrap().alloc_handle();
    buffers.write().unwrap().entries.insert(
        handle,
        BufferState {
            device_handle,
            buffer,
            memory,
            size,
            allocation_size: size,
            bindless_index: None,
            is_storage: false,
            element_stride: None,
            staging_buffer: None,
            staging_memory: None,
            is_view: false,
            host_mapped,
            flags: BufferFlags::empty(),
            transient_heap_suballoc: false,
            view_byte_offset: None,
            is_sparse: false,
            sparse_block_size: 0,
            sparse_pages: Vec::new(),
            is_withdraw_staging: true,
            texture_copy_footprint: None,
        },
    );
    Ok(handle)
}

pub(super) fn query_texture_copy_footprint(
    width: u32,
    height: u32,
    format: crate::types::TextureFormat,
) -> crate::backend::TextureCopyFootprint {
    let row_pitch = width.saturating_mul(format.bytes_per_pixel());
    let logical_bytes = row_pitch as u64 * height as u64;
    crate::backend::TextureCopyFootprint {
        width,
        height,
        format,
        logical_bytes,
        staging_bytes: logical_bytes,
        row_pitch,
        footprint_offset: 0,
    }
}

pub(super) fn alloc_texture_readback_staging(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    buffers: &SharedBufferTable,
    device_handle: DeviceHandle,
    layout: crate::backend::TextureCopyFootprint,
) -> Result<BufferHandle> {
    let handle = alloc_readback_buffer(instance, devices, buffers, device_handle, layout.staging_bytes)?;
    if let Some(buf) = buffers.write().unwrap().entries.get_mut(&handle) {
        buf.texture_copy_footprint = Some(layout);
    }
    Ok(handle)
}

pub(super) fn read_texture_readback_staging(
    buffers: &SharedBufferTable,
    buffer_handle: BufferHandle,
    layout: crate::backend::TextureCopyFootprint,
    output: &mut [u8],
) -> Result<()> {
    if output.len() as u64 != layout.logical_bytes {
        anyhow::bail!("read_texture_readback_staging size mismatch");
    }
    let buffers_guard = buffers.read().unwrap();
    let buffer = buffers_guard
        .entries
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;
    if !buffer.is_withdraw_staging {
        anyhow::bail!("read_texture_readback_staging requires a withdraw staging buffer");
    }
    let base = buffer
        .host_mapped
        .context("texture withdraw staging buffer not mapped")?;
    let row_bytes = layout.tight_row_bytes() as usize;
    let pitch = layout.row_pitch as usize;
    let p = base as *const u8;
    for row in 0..layout.height as usize {
        let src_offset = layout.footprint_offset as usize + row * pitch;
        let dst_offset = row * row_bytes;
        unsafe {
            std::ptr::copy_nonoverlapping(p.add(src_offset), output.as_mut_ptr().add(dst_offset), row_bytes);
        }
    }
    Ok(())
}

/// Read bytes from a withdraw staging staging buffer.
pub(super) fn read_readback_buffer(
    buffers: &SharedBufferTable,
    buffer_handle: BufferHandle,
    output: &mut [u8],
) -> Result<()> {
    let buffers_guard = buffers.read().unwrap();
    let buffer = buffers_guard
        .entries
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;
    if !buffer.is_withdraw_staging {
        anyhow::bail!("read_readback_buffer requires a withdraw staging buffer");
    }
    let base = buffer.host_mapped.context("withdraw staging buffer not mapped")?;
    if output.len() as u64 > buffer.size {
        anyhow::bail!("read_readback_buffer would exceed buffer bounds");
    }
    unsafe {
        std::ptr::copy_nonoverlapping(base as *const u8, output.as_mut_ptr(), output.len());
    }
    Ok(())
}
