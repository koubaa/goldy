//! Texture management logic.

use super::types::{self, TextureState};
use super::utils::format_to_vk;
use super::{DeviceHandle, TextureHandle};
use crate::types::{SpatialAccess, TextureFlags, TextureFormat};
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

/// Create a texture with the given dimensions, format, access pattern, and flags.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_find)]
pub(super) fn create(
    instance: &ash::Instance,
    devices: &mut HashMap<DeviceHandle, types::LogicalDevice>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    next_texture_handle: &mut TextureHandle,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: SpatialAccess,
    flags: TextureFlags,
) -> Result<TextureHandle> {
    // Get physical device for memory type lookup
    let physical_device = {
        let logical_device = devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        logical_device.physical_device
    };

    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let find_mem_type = |type_filter: u32, properties: vk::MemoryPropertyFlags| -> Option<u32> {
        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (mem_props.memory_types[i as usize].property_flags & properties) == properties
            {
                return Some(i);
            }
        }
        None
    };

    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Map access pattern and flags to Vulkan image usage
    let mut vk_usage = vk::ImageUsageFlags::TRANSFER_DST;

    // Interpolated access -> sampled image, Direct access -> storage image
    match access {
        SpatialAccess::Interpolated => {
            vk_usage |= vk::ImageUsageFlags::SAMPLED;
        }
        SpatialAccess::Direct => {
            vk_usage |= vk::ImageUsageFlags::STORAGE;
        }
    }

    // Apply additional flags
    if flags.contains(TextureFlags::RENDER_TARGET) {
        vk_usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if flags.contains(TextureFlags::COPY_SRC) {
        vk_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
    }

    // Create texture image
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format_to_vk(format))
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { logical_device.device.create_image(&image_info, None) }
        .context("Failed to create texture image")?;

    let mem_reqs = unsafe { logical_device.device.get_image_memory_requirements(image) };
    let memory_type = find_mem_type(
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .context("Failed to find memory type for texture")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(memory_type);

    let memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
        .context("Failed to allocate texture memory")?;

    unsafe { logical_device.device.bind_image_memory(image, memory, 0) }
        .context("Failed to bind texture memory")?;

    // Create image view
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format_to_vk(format))
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let view = unsafe { logical_device.device.create_image_view(&view_info, None) }
        .context("Failed to create texture view")?;

    let bindless_enabled = logical_device.bindless_enabled;
    let bindless_descriptor_set = logical_device.bindless_descriptor_set;

    let handle = *next_texture_handle;
    *next_texture_handle += 1;

    // Register texture in bindless descriptor set if enabled
    let bindless_index = if bindless_enabled {
        let logical_device = devices.get_mut(&device_handle).unwrap();
        let index = logical_device.resource_registry.register_texture(handle);

        // Update the global descriptor set with this texture
        if let Some(descriptor_set) = bindless_descriptor_set {
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(types::bindless_bindings::SAMPLED_IMAGES)
                .dst_array_element(index)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&image_info));

            unsafe {
                logical_device
                    .device
                    .update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }

            tracing::trace!(
                "Registered texture {} at bindless index {}",
                handle,
                index
            );
        }

        Some(index)
    } else {
        None
    };

    textures.insert(
        handle,
        TextureState {
            device_handle,
            width,
            height,
            format,
            image,
            memory,
            view,
            staging_buffer: None,
            staging_memory: None,
            bindless_index,
        },
    );

    tracing::debug!("Created texture {}x{} (handle={})", width, height, handle);
    Ok(handle)
}

/// Write data to a texture, uploading via a staging buffer.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_find)]
pub(super) fn write(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    textures: &HashMap<TextureHandle, TextureState>,
    texture_handle: TextureHandle,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<()> {
    let texture = textures
        .get(&texture_handle)
        .context("Invalid texture handle")?;

    let device_handle = texture.device_handle;
    let image = texture.image;
    let tex_width = texture.width;
    let tex_height = texture.height;

    // Validate dimensions
    if width != tex_width || height != tex_height {
        anyhow::bail!(
            "Texture dimensions mismatch: expected {}x{}, got {}x{}",
            tex_width,
            tex_height,
            width,
            height
        );
    }

    // Get physical device for memory type lookup
    let physical_device = {
        let logical_device = devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        logical_device.physical_device
    };

    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let find_mem_type = |type_filter: u32, properties: vk::MemoryPropertyFlags| -> Option<u32> {
        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (mem_props.memory_types[i as usize].property_flags & properties) == properties
            {
                return Some(i);
            }
        }
        None
    };

    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Create staging buffer
    let buffer_size = data.len() as u64;
    let staging_buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = unsafe { logical_device.device.create_buffer(&staging_buffer_info, None) }
        .context("Failed to create staging buffer")?;

    let staging_mem_reqs = unsafe {
        logical_device
            .device
            .get_buffer_memory_requirements(staging_buffer)
    };
    let staging_memory_type = find_mem_type(
        staging_mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .context("Failed to find memory type for staging buffer")?;

    let staging_alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(staging_mem_reqs.size)
        .memory_type_index(staging_memory_type);

    let staging_memory = unsafe {
        logical_device
            .device
            .allocate_memory(&staging_alloc_info, None)
    }
    .context("Failed to allocate staging memory")?;

    unsafe {
        logical_device
            .device
            .bind_buffer_memory(staging_buffer, staging_memory, 0)
    }
    .context("Failed to bind staging memory")?;

    // Copy data to staging buffer
    unsafe {
        let ptr = logical_device
            .device
            .map_memory(
                staging_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
            .context("Failed to map staging memory")?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        logical_device.device.unmap_memory(staging_memory);
    }

    // Allocate command buffer
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(logical_device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer")?;
    let cmd_buffer = cmd_buffers[0];

    // Record commands
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        logical_device
            .device
            .begin_command_buffer(cmd_buffer, &begin_info)
            .context("Failed to begin command buffer")?;

        // Transition image to transfer dst
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dep_info = vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
        logical_device
            .device
            .cmd_pipeline_barrier2(cmd_buffer, &dep_info);

        // Copy buffer to image
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        logical_device.device.cmd_copy_buffer_to_image(
            cmd_buffer,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        // Transition image to shader read
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dep_info = vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
        logical_device
            .device
            .cmd_pipeline_barrier2(cmd_buffer, &dep_info);

        logical_device
            .device
            .end_command_buffer(cmd_buffer)
            .context("Failed to end command buffer")?;

        // Submit and wait
        let cmd_buffers = [cmd_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);

        logical_device
            .device
            .queue_submit(logical_device.queue, &[submit_info], vk::Fence::null())
            .context("Failed to submit command buffer")?;
        logical_device
            .device
            .queue_wait_idle(logical_device.queue)
            .context("Failed to wait for queue")?;

        // Cleanup
        logical_device
            .device
            .free_command_buffers(logical_device.command_pool, &[cmd_buffer]);
        logical_device.device.destroy_buffer(staging_buffer, None);
        logical_device.device.free_memory(staging_memory, None);
    }

    tracing::debug!(
        "Wrote {}x{} texture data ({} bytes)",
        width,
        height,
        data.len()
    );
    Ok(())
}

/// Destroy a texture, unregistering it from bindless and cleaning up GPU resources.
pub(super) fn destroy(
    devices: &mut HashMap<DeviceHandle, types::LogicalDevice>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    texture_handle: TextureHandle,
) {
    if let Some(texture) = textures.remove(&texture_handle) {
        if let Some(logical_device) = devices.get_mut(&texture.device_handle) {
            // Unregister from bindless registry
            logical_device
                .resource_registry
                .unregister_texture(texture_handle);

            unsafe {
                logical_device.device.device_wait_idle().ok();
                logical_device.device.destroy_image_view(texture.view, None);
                logical_device.device.destroy_image(texture.image, None);
                logical_device.device.free_memory(texture.memory, None);
                if let Some(staging_buffer) = texture.staging_buffer {
                    logical_device
                        .device
                        .destroy_buffer(staging_buffer, None);
                }
                if let Some(staging_memory) = texture.staging_memory {
                    logical_device.device.free_memory(staging_memory, None);
                }
            }
        }
    }
}

/// Get the bindless descriptor index for a texture, if any.
pub(super) fn bindless_index(
    textures: &HashMap<TextureHandle, TextureState>,
    texture_handle: TextureHandle,
) -> Option<u32> {
    textures
        .get(&texture_handle)
        .and_then(|t| t.bindless_index)
}
