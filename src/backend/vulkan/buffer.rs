//! Buffer management logic.

use super::types::{self, BufferState};
use super::utils::find_memory_type;
use super::{BufferHandle, DeviceHandle};
use crate::backend::DataAccess;
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
    _element_stride: Option<u32>,
) -> Result<BufferHandle> {
    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Map access pattern to Vulkan buffer usage flags
    // All buffers get TRANSFER_SRC | TRANSFER_DST for flexibility
    let mut vk_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let is_storage = match access {
        DataAccess::Scattered => {
            vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
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

    // All buffers are registered for bindless access
    let should_register_bindless = true;
    let bindless_enabled = logical_device.bindless_enabled;
    let bindless_descriptor_set = logical_device.bindless_descriptor_set;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { logical_device.device.create_buffer(&buffer_info, None) }
        .context("Failed to create buffer")?;

    let mem_requirements = unsafe { logical_device.device.get_buffer_memory_requirements(buffer) };

    // Storage buffers → DEVICE_LOCAL for GPU compute performance.
    // Uniform buffers → HOST_VISIBLE|HOST_COHERENT for frequent CPU writes.
    let desired_flags = if is_storage {
        vk::MemoryPropertyFlags::DEVICE_LOCAL
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

    // For storage buffers, create a HOST_VISIBLE staging buffer for CPU upload/readback
    let (staging_buffer, staging_memory) = if is_storage {
        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let staging_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(staging_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let stg_buf = unsafe { logical_device.device.create_buffer(&staging_info, None) }
            .context("Failed to create staging buffer")?;

        let stg_reqs = unsafe {
            logical_device
                .device
                .get_buffer_memory_requirements(stg_buf)
        };

        let stg_mem_type = find_memory_type(
            instance,
            logical_device.physical_device,
            stg_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .context("Failed to find HOST_VISIBLE memory type for staging buffer")?;

        let stg_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(stg_reqs.size)
            .memory_type_index(stg_mem_type);

        let stg_mem = unsafe { logical_device.device.allocate_memory(&stg_alloc, None) }
            .context("Failed to allocate staging buffer memory")?;

        unsafe {
            logical_device
                .device
                .bind_buffer_memory(stg_buf, stg_mem, 0)
        }
        .context("Failed to bind staging buffer memory")?;

        (Some(stg_buf), Some(stg_mem))
    } else {
        (None, None)
    };

    let handle = *next_buffer_handle;
    *next_buffer_handle += 1;

    // Register buffer in bindless descriptor set if enabled AND buffer is UNIFORM or STORAGE
    // (VERTEX/INDEX buffers should not be in the uniform/storage descriptor arrays)
    let bindless_index = if bindless_enabled && should_register_bindless {
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
    } else {
        None
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
            staging_buffer,
            staging_memory,
            is_view: false,
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
    _element_stride: Option<u32>,
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

    let logical_device = devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    let bindless_enabled = logical_device.bindless_enabled;
    let bindless_descriptor_set = logical_device.bindless_descriptor_set;

    let bindless_index = if bindless_enabled {
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
    } else {
        None
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
            staging_buffer: None,
            staging_memory: None,
            is_view: true,
        },
    );

    Ok(handle)
}

/// Write data to a buffer at the specified offset.
///
/// For DEVICE_LOCAL storage buffers, writes go through the staging buffer then
/// a GPU copy. For HOST_VISIBLE uniform buffers, maps directly.
pub(super) fn write(
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    buffers: &HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
    offset: u64,
    data: &[u8],
) -> Result<()> {
    let buffer = buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    let device = devices
        .get(&buffer.device_handle)
        .context("Buffer's device is invalid")?;

    if offset + data.len() as u64 > buffer.size {
        anyhow::bail!("Write would exceed buffer bounds");
    }

    if let (Some(stg_buf), Some(stg_mem)) = (buffer.staging_buffer, buffer.staging_memory) {
        // DEVICE_LOCAL path: write to staging, then GPU copy
        unsafe {
            let ptr = device
                .device
                .map_memory(
                    stg_mem,
                    offset,
                    data.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .context("Failed to map staging buffer")?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            device.device.unmap_memory(stg_mem);
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
                .device
                .map_memory(
                    buffer.memory,
                    offset,
                    data.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .context("Failed to map buffer memory")?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            device.device.unmap_memory(buffer.memory);
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
/// For DEVICE_LOCAL storage buffers, issues a GPU copy to the staging buffer
/// then maps the staging buffer. For HOST_VISIBLE uniform buffers, maps directly.
pub(super) fn read_to_cpu(
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    buffers: &HashMap<BufferHandle, BufferState>,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    output: &mut [u8],
) -> Result<()> {
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
                .device
                .map_memory(stg_mem, 0, len, vk::MemoryMapFlags::empty())
                .context("Failed to map staging buffer for readback")?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), output.len());
            device.device.unmap_memory(stg_mem);
        }
    } else {
        // HOST_VISIBLE path: direct map
        unsafe {
            let ptr = device
                .device
                .map_memory(buffer.memory, 0, len, vk::MemoryMapFlags::empty())
                .context("Failed to map buffer memory")?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), output.len());
            device.device.unmap_memory(buffer.memory);
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
