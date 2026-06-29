//! Render target management logic.
//!
//! Handles creation, destruction, rendering, and readback of off-screen render targets.

use super::types::{LogicalDevice, RenderTargetState, SharedRenderTargetTable};
use super::utils::{depth_aspect_mask, depth_format_to_vk, format_to_vk};
use super::{DeviceHandle, PipelineHandle, RenderTargetHandle};
use crate::backend::RenderCommand;
use crate::types::{Color, TextureFormat};
use anyhow::{Context, Result};
use ash::{vk, Instance};
use std::collections::HashMap;

/// Helper to find a suitable memory type.
#[allow(clippy::manual_find)]
fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && (mem_props.memory_types[i as usize].property_flags & properties) == properties
        {
            return Some(i);
        }
    }
    None
}

/// Create a render target without depth buffer.
#[allow(clippy::too_many_arguments)]
pub(super) fn create(
    instance: &Instance,
    devices: &HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    render_targets: &SharedRenderTargetTable,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Result<RenderTargetHandle> {
    // Get physical device for memory type lookup
    let physical_device = {
        let logical_device = devices.get(&device_handle).context("Invalid device handle")?;
        logical_device.physical_device
    };

    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

    // Create render target image (GPU only - no staging yet)
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
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { logical_device.device.create_image(&image_info, None) }
        .context("Failed to create render target image")?;

    let mem_reqs = unsafe { logical_device.device.get_image_memory_requirements(image) };
    let memory_type = find_memory_type(
        &mem_props,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .context("Failed to find memory type for render target")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(memory_type);

    let image_memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
        .context("Failed to allocate render target memory")?;

    unsafe { logical_device.device.bind_image_memory(image, image_memory, 0) }
        .context("Failed to bind render target memory")?;

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

    let image_view = unsafe { logical_device.device.create_image_view(&view_info, None) }
        .context("Failed to create render target view")?;

    // Allocate command buffer
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(logical_device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer")?;

    let handle = render_targets.write().unwrap().alloc_handle();

    render_targets.write().unwrap().entries.insert(
        handle,
        RenderTargetState {
            device_handle,
            width,
            height,
            format,
            image,
            image_memory,
            image_view,
            depth_format: None,
            depth_image: None,
            depth_memory: None,
            depth_view: None,
            staging_buffer: None,
            staging_memory: None,
            command_buffer: command_buffers[0],
            has_rendered: std::sync::atomic::AtomicBool::new(false),
        },
    );

    tracing::debug!("Created render target {}x{} (handle={})", width, height, handle);
    Ok(handle)
}

/// Create a render target with optional depth buffer.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_with_depth(
    instance: &Instance,
    devices: &HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    render_targets: &SharedRenderTargetTable,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    color_format: TextureFormat,
    depth_format: Option<crate::types::DepthFormat>,
) -> Result<RenderTargetHandle> {
    // Get physical device for memory type lookup
    let physical_device = {
        let logical_device = devices.get(&device_handle).context("Invalid device handle")?;
        logical_device.physical_device
    };

    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

    // Create color render target image
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format_to_vk(color_format))
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { logical_device.device.create_image(&image_info, None) }
        .context("Failed to create render target image")?;

    let mem_reqs = unsafe { logical_device.device.get_image_memory_requirements(image) };
    let memory_type = find_memory_type(
        &mem_props,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .context("Failed to find memory type for render target")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(memory_type);

    let image_memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
        .context("Failed to allocate render target memory")?;

    unsafe { logical_device.device.bind_image_memory(image, image_memory, 0) }
        .context("Failed to bind render target memory")?;

    // Create color image view
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format_to_vk(color_format))
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let image_view = unsafe { logical_device.device.create_image_view(&view_info, None) }
        .context("Failed to create render target view")?;

    // Create depth buffer if requested
    let (depth_image, depth_memory, depth_view) = if let Some(df) = depth_format {
        let vk_depth_format = depth_format_to_vk(df);

        let depth_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_depth_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let d_image = unsafe { logical_device.device.create_image(&depth_info, None) }
            .context("Failed to create depth buffer image")?;

        let d_mem_reqs = unsafe { logical_device.device.get_image_memory_requirements(d_image) };
        let d_memory_type = find_memory_type(
            &mem_props,
            d_mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .context("Failed to find memory type for depth buffer")?;

        let d_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(d_mem_reqs.size)
            .memory_type_index(d_memory_type);

        let d_memory = unsafe { logical_device.device.allocate_memory(&d_alloc_info, None) }
            .context("Failed to allocate depth buffer memory")?;

        unsafe { logical_device.device.bind_image_memory(d_image, d_memory, 0) }
            .context("Failed to bind depth buffer memory")?;

        let d_view_info = vk::ImageViewCreateInfo::default()
            .image(d_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_depth_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: depth_aspect_mask(df),
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let d_view = unsafe { logical_device.device.create_image_view(&d_view_info, None) }
            .context("Failed to create depth buffer view")?;

        (Some(d_image), Some(d_memory), Some(d_view))
    } else {
        (None, None, None)
    };

    // Allocate command buffer
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(logical_device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer")?;

    let handle = render_targets.write().unwrap().alloc_handle();

    render_targets.write().unwrap().entries.insert(
        handle,
        RenderTargetState {
            device_handle,
            width,
            height,
            format: color_format,
            image,
            image_memory,
            image_view,
            depth_format,
            depth_image,
            depth_memory,
            depth_view,
            staging_buffer: None,
            staging_memory: None,
            command_buffer: command_buffers[0],
            has_rendered: std::sync::atomic::AtomicBool::new(false),
        },
    );

    tracing::debug!(
        "Created render target {}x{} with depth={:?} (handle={})",
        width,
        height,
        depth_format.is_some(),
        handle
    );
    Ok(handle)
}

/// Destroy a render target and free GPU resources.
pub(super) fn destroy(
    devices: &HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    render_targets: &SharedRenderTargetTable,
    target: RenderTargetHandle,
) {
    if let Some(state) = render_targets.write().unwrap().entries.remove(&target) {
        if let Some(logical_device) = devices.get(&state.device_handle) {
            unsafe {
                let _ = logical_device.synchronized_device_wait_idle();
                logical_device.device.destroy_image_view(state.image_view, None);
                logical_device.device.destroy_image(state.image, None);
                logical_device.device.free_memory(state.image_memory, None);
                if let Some(depth_view) = state.depth_view {
                    logical_device.device.destroy_image_view(depth_view, None);
                }
                if let Some(depth_image) = state.depth_image {
                    logical_device.device.destroy_image(depth_image, None);
                }
                if let Some(depth_memory) = state.depth_memory {
                    logical_device.device.free_memory(depth_memory, None);
                }
                if let Some(staging_buffer) = state.staging_buffer {
                    logical_device.device.destroy_buffer(staging_buffer, None);
                }
                if let Some(staging_memory) = state.staging_memory {
                    logical_device.device.free_memory(staging_memory, None);
                }
            }
        }
    }
}

/// Render commands to a render target.
#[allow(clippy::too_many_arguments)]
/// Record an offscreen render pass into an existing command buffer without submitting.
///
/// Records layout transitions (UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL),
/// dynamic rendering, draw commands, and the post-render barrier
/// (COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL) into `cmd`.
/// Does NOT begin/end the command buffer and does NOT submit.
pub(super) fn record_render_pass_to_buffer<F>(
    devices: &HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    render_targets: &SharedRenderTargetTable,
    device_handle: DeviceHandle,
    target: RenderTargetHandle,
    commands: &[RenderCommand],
    cmd: vk::CommandBuffer,
    record_commands_fn: F,
) -> Result<()>
where
    F: FnOnce(vk::CommandBuffer, &[RenderCommand], &LogicalDevice, &mut Option<PipelineHandle>) -> Result<()>,
{
    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

    let render_targets_guard = render_targets.read().unwrap();
    let render_target = render_targets_guard
        .entries
        .get(&target)
        .context("Invalid render target handle")?;

    if render_target.device_handle != device_handle {
        anyhow::bail!("Render target belongs to a different device");
    }

    let width = render_target.width;
    let height = render_target.height;
    let image = render_target.image;
    let image_view = render_target.image_view;
    let depth_view = render_target.depth_view;
    let depth_format = render_target.depth_format;

    let clear_color = commands
        .iter()
        .find_map(|c| match c {
            RenderCommand::Clear(color) => Some(*color),
            _ => None,
        })
        .unwrap_or(Color::BLACK);

    let clear_depth = commands
        .iter()
        .find_map(|c| match c {
            RenderCommand::ClearDepth(depth) => Some(*depth),
            _ => None,
        })
        .unwrap_or(1.0);

    // Transition image to color attachment
    let color_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let mut barriers = vec![color_barrier];

    if let (Some(depth_img), Some(df)) = (render_target.depth_image, depth_format) {
        let depth_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image(depth_img)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: depth_aspect_mask(df),
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        barriers.push(depth_barrier);
    }

    let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    unsafe { logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info) };

    // Begin dynamic rendering
    let color_attachment = vk::RenderingAttachmentInfo::default()
        .image_view(image_view)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [clear_color.r, clear_color.g, clear_color.b, clear_color.a],
            },
        });

    let depth_attachment = depth_view.map(|dv| {
        vk::RenderingAttachmentInfo::default()
            .image_view(dv)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: clear_depth,
                    stencil: 0,
                },
            })
    });

    let mut rendering_info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        })
        .layer_count(1)
        .color_attachments(std::slice::from_ref(&color_attachment));

    if let Some(ref depth_att) = depth_attachment {
        rendering_info = rendering_info.depth_attachment(depth_att);
    }

    unsafe { logical_device.device.cmd_begin_rendering(cmd, &rendering_info) };

    let viewport = vk::Viewport {
        x: 0.0,
        y: height as f32,
        width: width as f32,
        height: -(height as f32),
        min_depth: 0.0,
        max_depth: 1.0,
    };
    unsafe {
        logical_device
            .device
            .cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport))
    };

    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D { width, height },
    };
    unsafe {
        logical_device
            .device
            .cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor))
    };

    let mut current_pipeline: Option<PipelineHandle> = None;
    record_commands_fn(cmd, commands, logical_device, &mut current_pipeline)?;

    unsafe { logical_device.device.cmd_end_rendering(cmd) };

    // Transition image to transfer src (ready for potential readback)
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let dep_info = vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
    unsafe { logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info) };

    Ok(())
}

pub(super) struct RenderToResources<'a> {
    pub(super) contexts: &'a super::types::SharedContextMap,
    pub(super) devices: &'a HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    pub(super) frame_tables: &'a super::types::SharedFrameTableMap,
    pub(super) buffers: &'a super::types::SharedBufferTable,
    pub(super) pipelines: &'a super::types::SharedPipelineTable,
}

pub(super) fn render_to<F>(
    resources: RenderToResources<'_>,
    render_targets: &SharedRenderTargetTable,
    device_handle: DeviceHandle,
    target: RenderTargetHandle,
    commands: &[RenderCommand],
    record_commands_fn: F,
) -> Result<()>
where
    F: FnOnce(vk::CommandBuffer, &[RenderCommand], &LogicalDevice, &mut Option<PipelineHandle>) -> Result<()>,
{
    let logical_device = resources.devices.get(&device_handle).context("Invalid device handle")?;

    let (staging_data, lowered, has_bindings) =
        super::frame_table::prepare_render_commands(resources.buffers, resources.pipelines, commands)?;

    let cmd = render_targets
        .read()
        .unwrap()
        .entries
        .get(&target)
        .context("Invalid render target handle")?
        .command_buffer;

    let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
        .context("Failed to begin command buffer")?;

    if has_bindings {
        super::frame_table::record_prologue_for_tables(
            resources.contexts,
            resources.frame_tables,
            resources.buffers,
            device_handle,
            logical_device,
            cmd,
            &staging_data,
        )?;
    }

    record_render_pass_to_buffer(
        resources.devices,
        render_targets,
        device_handle,
        target,
        &lowered,
        cmd,
        record_commands_fn,
    )?;

    let logical_device = resources.devices.get(&device_handle).context("Invalid device handle")?;

    unsafe { logical_device.device.end_command_buffer(cmd) }.context("Failed to end command buffer")?;

    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    logical_device
        .synchronized_queue_submit(std::slice::from_ref(&submit_info), vk::Fence::null())
        .context("Failed to submit command buffer")?;

    logical_device.synchronized_queue_wait_idle().context("Failed to wait for queue")?;

    if let Some(rt) = render_targets.read().unwrap().entries.get(&target) {
        rt.has_rendered.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    Ok(())
}

/// Read render target contents to CPU memory.
#[allow(clippy::too_many_arguments)]
pub(super) fn read_to_cpu(
    instance: &Instance,
    devices: &HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    render_targets: &SharedRenderTargetTable,
    target: RenderTargetHandle,
    output: &mut [u8],
) -> Result<()> {
    // Get render target info and device
    let (device_handle, width, height, format, image, physical_device) = {
        let render_targets_guard = render_targets.read().unwrap();
        let render_target = render_targets_guard
            .entries
            .get(&target)
            .context("Invalid render target handle")?;

        if !render_target.has_rendered.load(std::sync::atomic::Ordering::Relaxed) {
            anyhow::bail!("Cannot read from render target that hasn't been rendered to");
        }

        let logical_device = devices
            .get(&render_target.device_handle)
            .context("Invalid device handle")?;

        (
            render_target.device_handle,
            render_target.width,
            render_target.height,
            render_target.format,
            render_target.image,
            logical_device.physical_device,
        )
    };

    let expected_size = (width * height * format.bytes_per_pixel()) as usize;
    if output.len() < expected_size {
        anyhow::bail!("Output buffer too small: {} < {}", output.len(), expected_size);
    }

    // Ensure staging buffer exists (lazy creation)
    let needs_staging = {
        let render_targets_guard = render_targets.read().unwrap();
        let render_target = render_targets_guard.entries.get(&target).unwrap();
        render_target.staging_buffer.is_none()
    };

    if needs_staging {
        // Create staging buffer
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let logical_device = devices.get(&device_handle).unwrap();
        let buffer_size = expected_size as u64;

        let staging_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe { logical_device.device.create_buffer(&staging_info, None) }
            .context("Failed to create staging buffer")?;

        let staging_reqs = unsafe { logical_device.device.get_buffer_memory_requirements(staging_buffer) };
        let staging_memory_type = find_memory_type(
            &mem_props,
            staging_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .context("Failed to find memory type for staging buffer")?;

        let staging_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_reqs.size)
            .memory_type_index(staging_memory_type);

        let staging_memory = unsafe { logical_device.device.allocate_memory(&staging_alloc, None) }
            .context("Failed to allocate staging buffer memory")?;

        unsafe {
            logical_device
                .device
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
        }
        .context("Failed to bind staging buffer memory")?;

        let mut render_targets_write = render_targets.write().unwrap();
        let render_target = render_targets_write.entries.get_mut(&target).unwrap();
        render_target.staging_buffer = Some(staging_buffer);
        render_target.staging_memory = Some(staging_memory);

        tracing::debug!("Created staging buffer for render target {}", target);
    }

    // Now copy and read
    let render_targets_guard = render_targets.read().unwrap();
    let render_target = render_targets_guard.entries.get(&target).unwrap();
    let staging_buffer = render_target.staging_buffer.unwrap();
    let staging_memory = render_target.staging_memory.unwrap();
    let cmd = render_target.command_buffer;

    let logical_device = devices.get(&device_handle).unwrap();

    // Graph/render submits may still be in flight on the async worker; wait for the
    // render-pass layout transition (COLOR_ATTACHMENT → TRANSFER_SRC) before copy.
    logical_device.submission_worker.flush()?;
    logical_device.synchronized_queue_wait_idle()?;

    // Reset and record copy command
    unsafe {
        logical_device
            .device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
    }
    .context("Failed to reset command buffer")?;

    let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
        .context("Failed to begin command buffer")?;

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

    unsafe {
        logical_device.device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging_buffer,
            std::slice::from_ref(&region),
        );
    }

    unsafe { logical_device.device.end_command_buffer(cmd) }.context("Failed to end command buffer")?;

    // Submit
    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    logical_device
        .synchronized_queue_submit(std::slice::from_ref(&submit_info), vk::Fence::null())
        .context("Failed to submit command buffer")?;

    logical_device.synchronized_queue_wait_idle().context("Failed to wait for queue")?;

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

    Ok(())
}
