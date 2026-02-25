//! Buffer management logic.

use super::types::{self, BufferState};
use super::utils::find_memory_type;
use super::{BufferHandle, DeviceHandle};
use crate::backend::DataAccess;
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

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
            // Scattered: any thread any address - use storage buffer
            vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
            true
        }
        DataAccess::Broadcast => {
            // Broadcast: all threads same address - use uniform buffer
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

    let memory_type = find_memory_type(
        instance,
        logical_device.physical_device,
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .context("Failed to find suitable memory type")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    let memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
        .context("Failed to allocate buffer memory")?;

    unsafe { logical_device.device.bind_buffer_memory(buffer, memory, 0) }
        .context("Failed to bind buffer memory")?;

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
        },
    );

    Ok(handle)
}

/// Destroy a buffer, unregistering it from bindless and queueing for deferred deletion.
pub(super) fn destroy(
    devices: &mut HashMap<DeviceHandle, types::LogicalDevice>,
    buffers: &mut HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
) {
    if let Some(buffer) = buffers.remove(&buffer_handle) {
        if let Some(device) = devices.get_mut(&buffer.device_handle) {
            // Unregister from bindless registry
            device.resource_registry.unregister_buffer(buffer_handle);

            // Queue for deferred deletion - the buffer may still be in use by in-flight commands
            device.deletion_queue.queue(types::PendingDeletion::Buffer {
                buffer: buffer.buffer,
                memory: buffer.memory,
            });
        }
    }
}

/// Write data to a buffer at the specified offset.
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
