//! Buffer management logic.

use super::types::{self, BufferState};
use super::utils::find_memory_type;
use super::{BufferHandle, DeviceHandle};
use crate::backend::DataAccess;
use crate::types::BufferFlags;
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

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

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

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
            .dst_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
            )
            .dst_access_mask(
                vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            );
        let dep_info =
            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
        device.device.cmd_pipeline_barrier2(cmd, &dep_info);

        device.device.end_command_buffer(cmd)?;

        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        device
            .device
            .queue_submit(device.queue, &[submit_info], vk::Fence::null())?;
        device.device.queue_wait_idle(device.queue)?;
        device
            .device
            .free_command_buffers(device.command_pool, &cmd_buffers);
    }

    Ok(())
}

/// Create a buffer with the given size and access pattern.
#[allow(clippy::too_many_arguments)]
pub(super) fn create(
    devices: &mut HashMap<DeviceHandle, types::LogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    next_buffer_handle: &mut BufferHandle,
    instance: &ash::Instance,
    device_handle: DeviceHandle,
    size: u64,
    access: DataAccess,
    element_stride: Option<u32>,
    flags: BufferFlags,
) -> Result<BufferHandle> {
    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Map access pattern to Vulkan buffer usage flags
    // All buffers get TRANSFER_SRC | TRANSFER_DST for flexibility
    let mut vk_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let is_storage = match access {
        DataAccess::Scattered => {
            vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER;
            // Indirect dispatch reads 3× u32 (12 bytes) from a storage buffer
            if size >= 12 {
                vk_usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
            }
            true
        }
        DataAccess::Broadcast => {
            vk_usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
            false
        }
    };

    let cpu_readable = flags.contains(BufferFlags::CPU_READABLE);
    if cpu_readable && !is_storage {
        anyhow::bail!(
            "BufferFlags::CPU_READABLE is only valid for DataAccess::Scattered (storage) buffers"
        );
    }

    let bindless_descriptor_set = logical_device.bindless_descriptor_set;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { logical_device.device.create_buffer(&buffer_info, None) }
        .context("Failed to create buffer")?;

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
        .context("Failed to allocate buffer memory")?;

    unsafe { logical_device.device.bind_buffer_memory(buffer, memory, 0) }
        .context("Failed to bind buffer memory")?;

    // Staging buffer is created lazily on first write() to avoid doubling memory
    // for buffers that are only GPU-written (intermediate compute buffers, pool backing).
    // Clear uses vkCmdFillBuffer (GPU-side) so staging isn't needed for that.
    let (staging_buffer, staging_memory) = (None, None);

    let host_mapped: Option<usize> = if cpu_readable && is_storage {
        let device = devices
            .get(&device_handle)
            .context("Buffer's device is invalid for host map")?;
        let ptr = unsafe { device.map_memory2(memory, 0, size) }
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
        let logical_device = devices.get_mut(&device_handle).unwrap();
        let index = logical_device
            .resource_registry
            .register_buffer(handle, is_storage);

        // Update the global descriptor set with this buffer
        if let Some(descriptor_set) = bindless_descriptor_set {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(0)
                .range(size);

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
            size,
            bindless_index,
            is_storage,
            element_stride,
            staging_buffer,
            staging_memory,
            is_view: false,
            host_mapped,
            flags,
        },
    );

    Ok(handle)
}

/// Destroy a buffer, unregistering it from bindless and queueing for deferred deletion.
/// For views, only the descriptor is unregistered — the underlying VkBuffer/memory belongs
/// to the parent and is not freed.
pub(super) fn destroy(
    devices: &mut HashMap<DeviceHandle, types::LogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
) {
    if let Some(buffer) = buffers.remove(&buffer_handle) {
        if let Some(device) = devices.get_mut(&buffer.device_handle) {
            // Unregister from bindless registry
            device.resource_registry.unregister_buffer(buffer_handle);

            if !buffer.is_view {
                if buffer.host_mapped.is_some() {
                    if let Err(e) = unsafe { device.unmap_memory2(buffer.memory) } {
                        tracing::warn!(
                            ?e,
                            "unmap_memory2 failed for CPU_READABLE buffer on destroy"
                        );
                    }
                }
                // Queue for deferred deletion - the buffer may still be in use by in-flight commands
                device.deletion_queue.queue(types::PendingDeletion::Buffer {
                    buffer: buffer.buffer,
                    memory: buffer.memory,
                    staging_buffer: buffer.staging_buffer,
                    staging_memory: buffer.staging_memory,
                });
            }
        }
    }
}

/// Create a view into a sub-region of an existing buffer.
///
/// The view gets its own bindless descriptor at `[offset, offset+size)` of the parent.
/// It shares the parent's VkBuffer and staging resources.
pub(super) fn create_view(
    devices: &mut HashMap<DeviceHandle, types::LogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    next_buffer_handle: &mut BufferHandle,
    parent_handle: BufferHandle,
    offset: u64,
    size: u64,
    element_stride: Option<u32>,
) -> Result<BufferHandle> {
    let parent = buffers
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

    let device_handle = parent.device_handle;
    let vk_buffer = parent.buffer;
    let is_storage = parent.is_storage;
    let parent_flags = parent.flags;

    let logical_device = devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

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
            bindless_index,
            is_storage,
            element_stride,
            staging_buffer: None,
            staging_memory: None,
            is_view: true,
            host_mapped: None,
            flags: parent_flags,
        },
    );

    Ok(handle)
}

/// Lazily create the HOST_VISIBLE staging buffer for a DEVICE_LOCAL storage buffer.
pub(super) fn ensure_staging(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
) -> Result<()> {
    let buffer = buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;
    if !buffer.is_storage
        || buffer.staging_buffer.is_some()
        || buffer.flags.contains(BufferFlags::CPU_READABLE)
    {
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

    let stg_buf = unsafe { device.device.create_buffer(&staging_info, None) }
        .context("Failed to create staging buffer")?;

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

    unsafe { device.device.bind_buffer_memory(stg_buf, stg_mem, 0) }
        .context("Failed to bind staging buffer memory")?;

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
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
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
        let buffer = buffers
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;
        if offset + data.len() as u64 > buffer.size {
            anyhow::bail!("Write would exceed buffer bounds");
        }
    }

    {
        let buffer = buffers
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

    // Lazily create staging buffer for storage buffers
    ensure_staging(instance, devices, buffers, buffer_handle)?;

    let buffer = buffers
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

        submit_copy(
            device,
            stg_buf,
            buffer.buffer,
            offset,
            offset,
            data.len() as u64,
        )?;
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
pub(super) fn size(
    buffers: &HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
) -> u64 {
    buffers.get(&buffer_handle).map(|b| b.size).unwrap_or(0)
}

/// Get the bindless descriptor index for a buffer, if any.
pub(super) fn bindless_index(
    buffers: &HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
) -> Option<u32> {
    buffers.get(&buffer_handle).and_then(|b| b.bindless_index)
}

/// Read buffer contents to CPU. Copies from offset 0 for length output.len().
///
/// For DEVICE_LOCAL storage buffers, lazily creates a staging buffer, then issues
/// a GPU copy and maps. For HOST_VISIBLE uniform buffers, maps directly.
pub(super) fn read_to_cpu(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    output: &mut [u8],
) -> Result<()> {
    {
        let buffer = buffers
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;
        if buffer.device_handle != device_handle {
            anyhow::bail!("Buffer belongs to different device");
        }
        let len = output.len() as u64;
        if len > buffer.size {
            anyhow::bail!("Read would exceed buffer bounds");
        }
        if let Some(base) = buffer.host_mapped {
            let p = base as *const u8;
            unsafe {
                std::ptr::copy_nonoverlapping(p, output.as_mut_ptr(), output.len());
            }
            return Ok(());
        }
    }

    // Lazily create staging buffer for storage buffers
    ensure_staging(instance, devices, buffers, buffer_handle)?;

    let buffer = buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    let device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    if buffer.device_handle != device_handle {
        anyhow::bail!("Buffer belongs to different device");
    }

    let len = output.len() as u64;
    if len > buffer.size {
        anyhow::bail!("Read would exceed buffer bounds");
    }

    if let (Some(stg_buf), Some(stg_mem)) = (buffer.staging_buffer, buffer.staging_memory) {
        // DEVICE_LOCAL path: GPU copy to staging, then map staging
        submit_copy(device, buffer.buffer, stg_buf, 0, 0, len)?;

        unsafe {
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
    devices: &mut HashMap<DeviceHandle, types::LogicalDevice>,
    buffers: &HashMap<BufferHandle, BufferState>,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    offset: u64,
    size: u64,
) -> Result<()> {
    let buffer = buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    let device = devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    if buffer.device_handle != device_handle {
        anyhow::bail!("Buffer belongs to different device");
    }

    let clear_size = if size == 0 {
        buffer.size.saturating_sub(offset)
    } else {
        size
    };

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

    let cmd_buffers = unsafe { device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer")?;
    let cmd = cmd_buffers[0];

    // Record fill command
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device.device.begin_command_buffer(cmd, &begin_info)?;
        device
            .device
            .cmd_fill_buffer(cmd, buffer.buffer, offset, clear_size, 0);

        let mem_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
        let dep_info =
            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
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
        device
            .device
            .free_command_buffers(device.command_pool, &cmd_buffers);
    }

    Ok(())
}
