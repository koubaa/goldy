//! Texture management logic.

use super::types::{self, TextureState};
use super::utils::format_to_vk;
use super::{DeviceHandle, TextureHandle};
use crate::types::{TextureFlags, TextureFormat, TextureKind};
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

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
    access: TextureKind,
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
        TextureKind::Interpolated => {
            vk_usage |= vk::ImageUsageFlags::SAMPLED;
        }
        TextureKind::Direct => {
            vk_usage |= vk::ImageUsageFlags::STORAGE;
        }
        TextureKind::DirectInterpolated => {
            vk_usage |= vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
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
        .inspect_err(|_e| {
            crate::signal::push_sync_signal(crate::signal::Signal::Oversubscribed {
                reason: crate::signal::OversubscribedReason::TextureHeap,
                size_hint: mem_reqs.size,
            });
        })
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

    let bindless_descriptor_set = logical_device.bindless_descriptor_set;

    let handle = *next_texture_handle;
    *next_texture_handle += 1;

    let is_storage_image = matches!(
        access,
        TextureKind::Direct | TextureKind::DirectInterpolated
    );
    let is_dual_access = matches!(access, TextureKind::DirectInterpolated);

    let bindless_index = {
        let logical_device = devices.get_mut(&device_handle).unwrap();
        let index = logical_device
            .ledger
            .lock()
            .unwrap()
            .resource_registry
            .register_texture(handle, is_storage_image);

        // Update the global descriptor set with this texture
        if let Some(descriptor_set) = bindless_descriptor_set {
            let (binding, descriptor_type, image_layout) = if is_storage_image {
                (
                    types::bindless_bindings::STORAGE_IMAGES,
                    vk::DescriptorType::STORAGE_IMAGE,
                    vk::ImageLayout::GENERAL,
                )
            } else {
                (
                    types::bindless_bindings::SAMPLED_IMAGES,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
            };

            let image_info = vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(image_layout);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding)
                .dst_array_element(index)
                .descriptor_type(descriptor_type)
                .image_info(std::slice::from_ref(&image_info));

            unsafe {
                logical_device
                    .device
                    .update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }

            tracing::trace!(
                "Registered texture {} at bindless index {} ({})",
                handle,
                index,
                if is_storage_image {
                    "storage_image"
                } else {
                    "sampled_image"
                }
            );
        }

        Some(index)
    };

    // For DirectInterpolated, also register a sampled-texture (SRV) slot.
    let sampled_bindless_index = if is_dual_access {
        let logical_device = devices.get_mut(&device_handle).unwrap();
        // Register in the sampled pool (is_storage_image = false).
        let index = logical_device
            .ledger
            .lock()
            .unwrap()
            .resource_registry
            .register_texture(handle, false);

        if let Some(descriptor_set) = bindless_descriptor_set {
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL);

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
        }
        Some(index)
    } else {
        None
    };

    let initial_layout = if is_storage_image {
        let logical_device = devices.get(&device_handle).unwrap();
        transition_image_layout(
            logical_device,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        )?;
        vk::ImageLayout::GENERAL
    } else {
        vk::ImageLayout::UNDEFINED
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
            sampled_bindless_index,
            current_layout: initial_layout,
            transient_heap_suballoc: false,
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
    textures: &mut HashMap<TextureHandle, TextureState>,
    texture_handle: TextureHandle,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<()> {
    let texture = textures
        .get(&texture_handle)
        .context("Invalid texture handle")?;

    let device_handle = texture.device_handle;
    let old_layout = texture.current_layout;
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

    let staging_buffer = unsafe {
        logical_device
            .device
            .create_buffer(&staging_buffer_info, None)
    }
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
            .map_memory2(staging_memory, 0, buffer_size)
            .context("Failed to map staging memory")?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        logical_device
            .unmap_memory2(staging_memory)
            .context("Failed to unmap staging memory")?;
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
        let (src_stage, src_access) = match old_layout {
            vk::ImageLayout::UNDEFINED => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            _ => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
        };
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dep_info =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
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

        let dep_info =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
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

    if let Some(tex) = textures.get_mut(&texture_handle) {
        tex.current_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    }

    tracing::debug!(
        "Wrote {}x{} texture data ({} bytes)",
        width,
        height,
        data.len()
    );
    Ok(())
}

/// Write data to a subregion of a texture.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_region(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    texture_handle: TextureHandle,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<()> {
    let texture = textures
        .get(&texture_handle)
        .context("Invalid texture handle")?;

    let device_handle = texture.device_handle;
    let image = texture.image;
    let tex_width = texture.width;
    let tex_height = texture.height;
    let old_layout = texture.current_layout;

    if x + width > tex_width || y + height > tex_height {
        anyhow::bail!(
            "Region out of bounds: {}x{} at ({},{}) exceeds {}x{} texture",
            width,
            height,
            x,
            y,
            tex_width,
            tex_height
        );
    }

    let bytes_per_pixel = texture.format.bytes_per_pixel();
    let expected_size = (width * height * bytes_per_pixel) as usize;
    if data.len() != expected_size {
        anyhow::bail!(
            "Data size mismatch: expected {} bytes, got {}",
            expected_size,
            data.len()
        );
    }

    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let physical_device = logical_device.physical_device;
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let find_mem_type = |type_filter: u32, properties: vk::MemoryPropertyFlags| -> Option<u32> {
        (0..mem_props.memory_type_count).find(|&i| {
            (type_filter & (1 << i)) != 0
                && (mem_props.memory_types[i as usize].property_flags & properties) == properties
        })
    };

    let buffer_size = data.len() as u64;
    let staging_buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = unsafe {
        logical_device
            .device
            .create_buffer(&staging_buffer_info, None)
    }
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

    unsafe {
        let ptr = logical_device
            .map_memory2(staging_memory, 0, buffer_size)
            .context("Failed to map staging memory")?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        logical_device
            .unmap_memory2(staging_memory)
            .context("Failed to unmap staging memory")?;
    }

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(logical_device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer")?;
    let cmd_buffer = cmd_buffers[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        logical_device
            .device
            .begin_command_buffer(cmd_buffer, &begin_info)
            .context("Failed to begin command buffer")?;

        let (src_stage, src_access) = match old_layout {
            vk::ImageLayout::UNDEFINED => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            _ => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
        };
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let dep_info =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
        logical_device
            .device
            .cmd_pipeline_barrier2(cmd_buffer, &dep_info);

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
            .image_offset(vk::Offset3D {
                x: x as i32,
                y: y as i32,
                z: 0,
            })
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

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            )
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
        let dep_info =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
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

    if let Some(tex) = textures.get_mut(&texture_handle) {
        tex.current_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    }

    tracing::debug!(
        "Wrote {}x{} region at ({},{}) to texture ({} bytes)",
        width,
        height,
        x,
        y,
        data.len()
    );
    Ok(())
}

/// Read texture contents to CPU memory.
/// The texture must have been created with TextureFlags::COPY_SRC.
#[allow(clippy::too_many_arguments)]
pub(super) fn read_to_cpu(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    texture_handle: TextureHandle,
    output: &mut [u8],
) -> Result<()> {
    let (device_handle, width, height, format, image, old_layout, existing_sb, existing_sm) = {
        let texture = textures
            .get(&texture_handle)
            .context("Invalid texture handle")?;
        (
            texture.device_handle,
            texture.width,
            texture.height,
            texture.format,
            texture.image,
            texture.current_layout,
            texture.staging_buffer,
            texture.staging_memory,
        )
    };

    let expected_size = (width * height * format.bytes_per_pixel()) as usize;
    if output.len() < expected_size {
        anyhow::bail!(
            "Output buffer too small: {} < {}",
            output.len(),
            expected_size
        );
    }

    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let physical_device = logical_device.physical_device;
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let find_mem_type = |type_filter: u32, properties: vk::MemoryPropertyFlags| -> Option<u32> {
        (0..mem_props.memory_type_count).find(|&i| {
            (type_filter & (1 << i)) != 0
                && (mem_props.memory_types[i as usize].property_flags & properties) == properties
        })
    };

    // Lazy-create staging buffer
    let (staging_buffer, staging_memory) = match (existing_sb, existing_sm) {
        (Some(buf), Some(mem)) => (buf, mem),
        (None, _) | (_, None) => {
            let buffer_size = expected_size as u64;
            let staging_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let sb = unsafe { logical_device.device.create_buffer(&staging_info, None) }
                .context("Failed to create staging buffer")?;

            let staging_reqs = unsafe { logical_device.device.get_buffer_memory_requirements(sb) };
            let staging_memory_type = find_mem_type(
                staging_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .context("Failed to find memory type for staging buffer")?;

            let staging_alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(staging_reqs.size)
                .memory_type_index(staging_memory_type);

            let sm = unsafe { logical_device.device.allocate_memory(&staging_alloc, None) }
                .context("Failed to allocate staging buffer memory")?;

            unsafe { logical_device.device.bind_buffer_memory(sb, sm, 0) }
                .context("Failed to bind staging buffer memory")?;

            let tex = textures.get_mut(&texture_handle).unwrap();
            tex.staging_buffer = Some(sb);
            tex.staging_memory = Some(sm);

            (sb, sm)
        }
    };

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(logical_device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer")?;
    let cmd = cmd_buffers[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        logical_device
            .device
            .begin_command_buffer(cmd, &begin_info)
            .context("Failed to begin command buffer")?;

        // Transition image to transfer src
        let (src_stage, src_access) = match old_layout {
            vk::ImageLayout::UNDEFINED => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::GENERAL => (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
            ),
            _ => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
        };
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dep_info =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);

        // Copy image to staging buffer
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

        logical_device.device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging_buffer,
            std::slice::from_ref(&region),
        );

        logical_device
            .device
            .end_command_buffer(cmd)
            .context("Failed to end command buffer")?;
    }

    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    unsafe {
        logical_device.device.queue_submit(
            logical_device.queue,
            std::slice::from_ref(&submit_info),
            vk::Fence::null(),
        )
    }
    .context("Failed to submit command buffer")?;

    unsafe { logical_device.device.queue_wait_idle(logical_device.queue) }
        .context("Failed to wait for queue")?;

    unsafe {
        logical_device
            .device
            .free_command_buffers(logical_device.command_pool, &[cmd]);
    }

    // Read from staging buffer
    unsafe {
        let ptr = logical_device
            .map_memory2(staging_memory, 0, expected_size as u64)
            .context("Failed to map staging buffer")?;

        std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), expected_size);

        logical_device
            .unmap_memory2(staging_memory)
            .context("Failed to unmap staging buffer")?;
    }

    if let Some(tex) = textures.get_mut(&texture_handle) {
        tex.current_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    }

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
            if texture.transient_heap_suballoc {
                logical_device
                    .ledger
                    .lock()
                    .unwrap()
                    .reclaim_texture_slots(texture_handle);
                unsafe {
                    logical_device.device.destroy_image_view(texture.view, None);
                    logical_device.device.destroy_image(texture.image, None);
                }
                return;
            }

            let barrier = logical_device
                .timeline_next
                .load(Ordering::Relaxed)
                .saturating_sub(1);
            logical_device.deletion_queue.queue(
                barrier,
                types::PendingDeletion::Texture {
                    texture_handle,
                    image: texture.image,
                    view: texture.view,
                    memory: texture.memory,
                    staging_buffer: texture.staging_buffer,
                    staging_memory: texture.staging_memory,
                },
            );
        }
    }
}

/// Transition an image between layouts using a one-shot command buffer.
pub(super) fn transition_image_layout(
    logical_device: &types::LogicalDevice,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(logical_device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer for layout transition")?;
    let cmd = cmd_buffers[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        logical_device
            .device
            .begin_command_buffer(cmd, &begin_info)
            .context("Failed to begin command buffer")?;

        let (src_stage, src_access) = match old_layout {
            vk::ImageLayout::UNDEFINED => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
            vk::ImageLayout::GENERAL => (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            _ => (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
            ),
        };

        let (dst_stage, dst_access) = match new_layout {
            vk::ImageLayout::GENERAL => (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            _ => (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
            ),
        };

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dep_info =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);

        logical_device
            .device
            .end_command_buffer(cmd)
            .context("Failed to end command buffer")?;

        let cmd_buffers_arr = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers_arr);
        let sub = logical_device.device.queue_submit(
            logical_device.queue,
            &[submit_info],
            vk::Fence::null(),
        );
        sub.context("Failed to submit layout transition")?;
        let wait = logical_device.device.queue_wait_idle(logical_device.queue);
        wait.context("Failed to wait for layout transition")?;

        logical_device
            .device
            .free_command_buffers(logical_device.command_pool, &cmd_buffers_arr);
    }

    Ok(())
}

/// Host-visible staging for a batched texture upload (see compute submit path).
///
/// The `entry` is a pooled, permanently-mapped staging buffer. When the GPU
/// copy completes, it is returned to the [`super::staging::TextureStagingPool`]
/// for reuse rather than freed.
pub(super) struct ComputeTextureScratch {
    pub entry: super::staging::TextureStagingEntry,
    pub texture_handle: TextureHandle,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Acquire a staging entry from `pool` and CPU-fill it for
/// [`GpuCommand::WriteTexture`](crate::backend::GpuCommand::WriteTexture)
/// / [`GpuCommand::WriteTextureRegion`](crate::backend::GpuCommand::WriteTextureRegion).
///
/// The acquired entry is permanently mapped; data is copied in directly.
/// After the GPU copy, the entry should be returned to the pool via
/// [`TextureStagingPool::release`](super::staging::TextureStagingPool::release).
#[allow(clippy::too_many_arguments)]
pub(super) fn allocate_compute_texture_staging(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    textures: &HashMap<TextureHandle, TextureState>,
    pool: &mut super::staging::TextureStagingPool,
    texture_handle: TextureHandle,
    data: &[u8],
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<ComputeTextureScratch> {
    let texture = textures
        .get(&texture_handle)
        .context("allocate_compute_texture_staging: invalid texture")?;

    if x + width > texture.width || y + height > texture.height {
        anyhow::bail!(
            "Region out of bounds: {}x{} at ({},{}) exceeds {}x{} texture",
            width,
            height,
            x,
            y,
            texture.width,
            texture.height
        );
    }
    let expected_size = (width * height * texture.format.bytes_per_pixel()) as usize;
    if data.len() != expected_size {
        anyhow::bail!(
            "Data size mismatch: expected {} bytes, got {}",
            expected_size,
            data.len()
        );
    }

    let device_handle = texture.device_handle;
    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let buffer_size = data.len() as u64;
    let entry = pool
        .acquire(instance, logical_device, buffer_size)
        .context("compute texture staging: pool acquire")?;

    // Copy data into the permanently-mapped entry.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), entry.mapped_ptr(), data.len());
    }

    Ok(ComputeTextureScratch {
        entry,
        texture_handle,
        x,
        y,
        width,
        height,
    })
}

/// Record buffer→image copy + layout transitions into an open command buffer.
pub(super) fn record_compute_texture_upload(
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    cmd: vk::CommandBuffer,
    scratch: &ComputeTextureScratch,
) -> Result<()> {
    let (device_handle, width, height, format, image, old_layout) = {
        let texture = textures
            .get(&scratch.texture_handle)
            .context("record_compute_texture_upload: invalid texture")?;
        (
            texture.device_handle,
            texture.width,
            texture.height,
            texture.format,
            texture.image,
            texture.current_layout,
        )
    };

    if scratch.x + scratch.width > width || scratch.y + scratch.height > height {
        anyhow::bail!("record_compute_texture_upload: scratch region out of bounds");
    }

    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let (src_stage, src_access) = match old_layout {
        vk::ImageLayout::UNDEFINED => (
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        ),
        vk::ImageLayout::GENERAL => (
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
        ),
        _ => (
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
        ),
    };

    let barrier_to_dst = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .old_layout(old_layout)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let dep_info =
        vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier_to_dst));
    unsafe { logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info) };

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
        .image_offset(vk::Offset3D {
            x: scratch.x as i32,
            y: scratch.y as i32,
            z: 0,
        })
        .image_extent(vk::Extent3D {
            width: scratch.width,
            height: scratch.height,
            depth: 1,
        });

    unsafe {
        logical_device.device.cmd_copy_buffer_to_image(
            cmd,
            scratch.entry.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&region),
        );
    }

    let barrier_to_shader = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
        )
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

    let dep_info2 = vk::DependencyInfo::default()
        .image_memory_barriers(std::slice::from_ref(&barrier_to_shader));
    unsafe { logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info2) };

    if let Some(tex) = textures.get_mut(&scratch.texture_handle) {
        tex.current_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    }

    let _ = format;
    Ok(())
}

/// Get the bindless descriptor index for a texture, if any.
pub(super) fn bindless_index(
    textures: &HashMap<TextureHandle, TextureState>,
    texture_handle: TextureHandle,
) -> Option<u32> {
    textures.get(&texture_handle).and_then(|t| t.bindless_index)
}

/// For `TextureKind::DirectInterpolated` textures, return the sampled-texture (SRV) slot.
pub(super) fn bindless_sampled_index(
    textures: &HashMap<TextureHandle, TextureState>,
    texture_handle: TextureHandle,
) -> Option<u32> {
    textures
        .get(&texture_handle)
        .and_then(|t| t.sampled_bindless_index)
}
