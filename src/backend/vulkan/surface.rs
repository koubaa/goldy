//! Surface and swapchain management for window presentation.

use super::types::{self, FrameSync, LogicalDevice, SurfaceState, MAX_FRAMES_IN_FLIGHT};
use super::utils;
use super::{DeviceHandle, PipelineHandle, SurfaceHandle, SwapchainImageHandle};
use crate::backend::RenderCommand;
use crate::types::{Color, TextureFormat};
use anyhow::{Context, Result};
use ash::{khr, vk, Entry, Instance};
use std::collections::HashMap;

#[cfg(target_os = "windows")]
use raw_window_handle::RawWindowHandle;

#[cfg(target_os = "linux")]
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

/// Create platform-specific Vulkan surface.
pub(super) fn create_platform_surface(
    entry: &Entry,
    instance: &Instance,
    window: &dyn raw_window_handle::HasWindowHandle,
    _display: &dyn raw_window_handle::HasDisplayHandle,
) -> Result<vk::SurfaceKHR> {
    #[cfg(target_os = "windows")]
    let window_handle = window
        .window_handle()
        .map_err(|e| anyhow::anyhow!("Failed to get window handle: {:?}", e))?;

    #[cfg(target_os = "linux")]
    let window_handle = window
        .window_handle()
        .map_err(|e| anyhow::anyhow!("Failed to get window handle: {:?}", e))?;

    // Silence unused warning on platforms where surface creation isn't supported
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    let _ = window;

    #[cfg(target_os = "windows")]
    {
        match window_handle.as_raw() {
            RawWindowHandle::Win32(h) => {
                let create_info = vk::Win32SurfaceCreateInfoKHR::default()
                    .hwnd(h.hwnd.get() as isize)
                    .hinstance(h.hinstance.map(|i| i.get() as isize).unwrap_or(0));

                let win32_surface = khr::win32_surface::Instance::new(entry, instance);
                unsafe { win32_surface.create_win32_surface(&create_info, None) }
                    .context("Failed to create Win32 surface")
            }
            _ => anyhow::bail!("Expected Win32 window handle on Windows"),
        }
    }

    #[cfg(target_os = "linux")]
    {
        let display_handle = _display
            .display_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get display handle: {:?}", e))?;

        match (window_handle.as_raw(), display_handle.as_raw()) {
            (RawWindowHandle::Wayland(w), RawDisplayHandle::Wayland(d)) => {
                let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                    .display(d.display.as_ptr())
                    .surface(w.surface.as_ptr());

                let wayland_surface = khr::wayland_surface::Instance::new(entry, instance);
                unsafe { wayland_surface.create_wayland_surface(&create_info, None) }
                    .context("Failed to create Wayland surface")
            }
            _ => anyhow::bail!(
                "Expected Wayland window/display handles on Linux (X11 not supported)"
            ),
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        anyhow::bail!(
            "Surface creation not supported on this platform - use Metal backend on macOS"
        )
    }
}

/// Create a new surface for window presentation.
#[allow(clippy::too_many_arguments)]
pub(super) fn create(
    entry: &Entry,
    instance: &Instance,
    devices: &HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    next_surface_handle: &mut SurfaceHandle,
    device_handle: DeviceHandle,
    window: &dyn raw_window_handle::HasWindowHandle,
    display: &dyn raw_window_handle::HasDisplayHandle,
) -> Result<SurfaceHandle> {
    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    let physical_device = logical_device.physical_device;

    // Create platform-specific surface
    let surface = create_platform_surface(entry, instance, window, display)?;

    // Get surface capabilities
    let surface_loader = khr::surface::Instance::new(entry, instance);
    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }
    .context("Failed to get surface capabilities")?;

    // Choose surface format (prefer BGRA8 for better compatibility)
    let formats = unsafe {
        surface_loader.get_physical_device_surface_formats(physical_device, surface)
    }
    .context("Failed to get surface formats")?;

    let format = formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB || f.format == vk::Format::B8G8R8A8_UNORM
        })
        .or_else(|| formats.first())
        .context("No suitable surface format")?;

    // Choose present mode (FIFO = vsync)
    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
    }
    .context("Failed to get present modes")?;

    let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX // Triple buffering if available
    } else {
        vk::PresentModeKHR::FIFO // Vsync (always available)
    };

    // Determine extent
    let extent = if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: capabilities
                .min_image_extent
                .width
                .max(800)
                .min(capabilities.max_image_extent.width),
            height: capabilities
                .min_image_extent
                .height
                .max(600)
                .min(capabilities.max_image_extent.height),
        }
    };

    // Create swapchain
    let image_count = (capabilities.min_image_count + 1).min(if capabilities.max_image_count > 0 {
        capabilities.max_image_count
    } else {
        u32::MAX
    });

    let swapchain_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);

    let swapchain_loader = khr::swapchain::Device::new(instance, &logical_device.device);
    let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None) }
        .context("Failed to create swapchain")?;

    // Get swapchain images
    let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
        .context("Failed to get swapchain images")?;

    // Create image views
    let swapchain_image_views: Vec<vk::ImageView> = swapchain_images
        .iter()
        .map(|&image| {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe { logical_device.device.create_image_view(&view_info, None) }
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("Failed to create swapchain image views")?;

    // Create per-frame synchronization resources
    let mut frame_sync = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        // Create semaphores
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let image_available_semaphore =
            unsafe { logical_device.device.create_semaphore(&semaphore_info, None) }
                .context("Failed to create image available semaphore")?;
        let render_finished_semaphore =
            unsafe { logical_device.device.create_semaphore(&semaphore_info, None) }
                .context("Failed to create render finished semaphore")?;

        // Create fence (signaled so first wait succeeds)
        let fence_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight_fence = unsafe { logical_device.device.create_fence(&fence_info, None) }
            .context("Failed to create in-flight fence")?;

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(logical_device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers =
            unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
                .context("Failed to allocate command buffer")?;

        frame_sync.push(FrameSync {
            command_buffer: command_buffers[0],
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        });
    }

    let handle = *next_surface_handle;
    *next_surface_handle += 1;

    surfaces.insert(
        handle,
        SurfaceState {
            device_handle,
            surface,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            width: extent.width,
            height: extent.height,
            format: format.format,
            current_frame: 0,
            current_image_index: None,
            frame_sync,
        },
    );

    tracing::info!(
        "Created surface {}x{} with {} images",
        extent.width,
        extent.height,
        image_count
    );
    Ok(handle)
}

/// Destroy a surface and all associated resources.
pub(super) fn destroy(
    entry: &Entry,
    instance: &Instance,
    devices: &HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    surface_handle: SurfaceHandle,
) {
    if let Some(surface_state) = surfaces.remove(&surface_handle) {
        if let Some(logical_device) = devices.get(&surface_state.device_handle) {
            unsafe {
                let _ = logical_device.device.device_wait_idle();

                // Destroy per-frame sync resources
                for frame in surface_state.frame_sync {
                    logical_device
                        .device
                        .destroy_semaphore(frame.image_available_semaphore, None);
                    logical_device
                        .device
                        .destroy_semaphore(frame.render_finished_semaphore, None);
                    logical_device
                        .device
                        .destroy_fence(frame.in_flight_fence, None);
                }

                for view in surface_state.swapchain_image_views {
                    logical_device.device.destroy_image_view(view, None);
                }

                let swapchain_loader =
                    khr::swapchain::Device::new(instance, &logical_device.device);
                swapchain_loader.destroy_swapchain(surface_state.swapchain, None);

                let surface_loader = khr::surface::Instance::new(entry, instance);
                surface_loader.destroy_surface(surface_state.surface, None);
            }
        }
    }
}

/// Acquire the next swapchain image for rendering.
pub(super) fn acquire(
    instance: &Instance,
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    surface_handle: SurfaceHandle,
) -> Result<SwapchainImageHandle> {
    // Get surface state and current frame index
    let (device_handle, _current_frame, swapchain, in_flight_fence, image_available_semaphore) = {
        let surface_state = surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        let frame = &surface_state.frame_sync[surface_state.current_frame];
        (
            surface_state.device_handle,
            surface_state.current_frame,
            surface_state.swapchain,
            frame.in_flight_fence,
            frame.image_available_semaphore,
        )
    };

    let logical_device = devices
        .get(&device_handle)
        .context("Surface's device is invalid")?;

    // Wait for the previous frame using this slot to finish
    unsafe {
        logical_device
            .device
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)
    }
    .context("Failed to wait for frame fence")?;

    // Process deferred deletions - resources from frames that have now completed
    // Since we just waited for the fence, frame (current_deletion_frame - MAX_FRAMES_IN_FLIGHT) has completed
    {
        let logical_device = devices
            .get_mut(&device_handle)
            .context("Surface's device is invalid")?;
        let current_frame = logical_device.deletion_queue.current_frame;
        if current_frame >= types::MAX_FRAMES_IN_FLIGHT as u64 {
            let completed_frame = current_frame - types::MAX_FRAMES_IN_FLIGHT as u64;
            logical_device
                .deletion_queue
                .process_deletions(&logical_device.device, completed_frame);
        }
    }

    let logical_device = devices
        .get(&device_handle)
        .context("Surface's device is invalid")?;

    // Reset fence for this frame
    unsafe { logical_device.device.reset_fences(&[in_flight_fence]) }
        .context("Failed to reset frame fence")?;

    // Acquire next swapchain image
    let swapchain_loader = khr::swapchain::Device::new(instance, &logical_device.device);

    let acquire_result = unsafe {
        swapchain_loader.acquire_next_image(
            swapchain,
            u64::MAX,
            image_available_semaphore,
            vk::Fence::null(),
        )
    };

    match acquire_result {
        Ok((image_index, suboptimal)) => {
            if suboptimal {
                tracing::debug!("Swapchain suboptimal - consider resizing");
            }
            // Update surface state
            let surface_state = surfaces.get_mut(&surface_handle).unwrap();
            surface_state.current_image_index = Some(image_index);
            Ok(image_index as SwapchainImageHandle)
        }
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            // Swapchain is out of date - caller should resize and retry
            tracing::info!("Swapchain out of date - resize required");
            anyhow::bail!("Surface out of date - call resize() and retry")
        }
        Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
            tracing::error!("Surface lost");
            anyhow::bail!("Surface lost - recreate surface")
        }
        Err(e) => {
            anyhow::bail!("Failed to acquire swapchain image: {:?}", e)
        }
    }
}

/// Render commands to the surface's current swapchain image.
#[allow(clippy::too_many_arguments)]
pub(super) fn render<F>(
    devices: &HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &HashMap<SurfaceHandle, SurfaceState>,
    surface_handle: SurfaceHandle,
    _image: SwapchainImageHandle,
    commands: &[RenderCommand],
    record_commands_fn: F,
) -> Result<()>
where
    F: FnOnce(
        vk::CommandBuffer,
        &[RenderCommand],
        &LogicalDevice,
        &mut Option<PipelineHandle>,
    ),
{
    let surface_state = surfaces
        .get(&surface_handle)
        .context("Invalid surface handle")?;

    let image_index = surface_state
        .current_image_index
        .context("No image acquired - call surface_acquire first")?;

    let logical_device = devices
        .get(&surface_state.device_handle)
        .context("Surface's device is invalid")?;

    let current_frame = surface_state.current_frame;
    let frame = &surface_state.frame_sync[current_frame];
    let cmd = frame.command_buffer;
    let width = surface_state.width;
    let height = surface_state.height;
    let image = surface_state.swapchain_images[image_index as usize];
    let image_view = surface_state.swapchain_image_views[image_index as usize];

    // Find clear color
    let clear_color = commands
        .iter()
        .find_map(|c| match c {
            RenderCommand::Clear(color) => Some(*color),
            _ => None,
        })
        .unwrap_or(Color::BLACK);

    // Begin command buffer
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
        .context("Failed to begin command buffer")?;

    // Transition image to color attachment
    let barrier = vk::ImageMemoryBarrier2::default()
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

    let dep_info = vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

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

    let rendering_info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        })
        .layer_count(1)
        .color_attachments(std::slice::from_ref(&color_attachment));

    unsafe {
        logical_device
            .device
            .cmd_begin_rendering(cmd, &rendering_info)
    };

    // Set viewport and scissor
    // Use negative height to flip Y axis - makes Vulkan coordinate system match DX12
    // This requires VK_KHR_maintenance1 (core in Vulkan 1.1+)
    let viewport = vk::Viewport {
        x: 0.0,
        y: height as f32,         // Start from bottom
        width: width as f32,
        height: -(height as f32), // Negative height flips Y
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

    // Track current pipeline for bind group binding
    let mut current_pipeline: Option<PipelineHandle> = None;

    // Execute render commands using provided callback
    record_commands_fn(cmd, commands, logical_device, &mut current_pipeline);

    // End dynamic rendering
    unsafe { logical_device.device.cmd_end_rendering(cmd) };

    // Transition image for presentation
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
        .dst_access_mask(vk::AccessFlags2::NONE)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
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

    // End command buffer
    unsafe { logical_device.device.end_command_buffer(cmd) }
        .context("Failed to end command buffer")?;

    // Get per-frame sync primitives
    let frame = &surface_state.frame_sync[current_frame];
    let wait_semaphores = [frame.image_available_semaphore];
    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let signal_semaphores = [frame.render_finished_semaphore];

    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(std::slice::from_ref(&cmd))
        .signal_semaphores(&signal_semaphores);

    // Submit with fence for frame tracking
    unsafe {
        logical_device.device.queue_submit(
            logical_device.queue,
            std::slice::from_ref(&submit_info),
            frame.in_flight_fence,
        )
    }
    .context("Failed to submit command buffer")?;

    Ok(())
}

/// Present the rendered image to the screen.
pub(super) fn present(
    instance: &Instance,
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    surface_handle: SurfaceHandle,
    _image: SwapchainImageHandle,
) -> Result<()> {
    let surface_state = surfaces
        .get_mut(&surface_handle)
        .context("Invalid surface handle")?;

    let image_index = surface_state
        .current_image_index
        .context("No image to present - call surface_render first")?;

    let current_frame = surface_state.current_frame;
    let frame = &surface_state.frame_sync[current_frame];

    let logical_device = devices
        .get(&surface_state.device_handle)
        .context("Surface's device is invalid")?;

    let swapchain_loader = khr::swapchain::Device::new(instance, &logical_device.device);

    let swapchains = [surface_state.swapchain];
    let image_indices = [image_index];
    let wait_semaphores = [frame.render_finished_semaphore];

    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&wait_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    let result = unsafe { swapchain_loader.queue_present(logical_device.queue, &present_info) };

    // Clear the current image and advance frame counter
    let device_handle = surface_state.device_handle;
    let surface_state = surfaces.get_mut(&surface_handle).unwrap();
    surface_state.current_image_index = None;
    surface_state.current_frame = (surface_state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

    // Advance the deletion queue's frame counter
    if let Some(device) = devices.get_mut(&device_handle) {
        device.deletion_queue.advance_frame();
    }

    // Handle suboptimal or out of date
    match result {
        Ok(_) => Ok(()),
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
            // TODO: Signal that resize is needed
            Ok(())
        }
        Err(e) => Err(anyhow::anyhow!("Failed to present: {:?}", e)),
    }
}

/// Resize the surface's swapchain.
#[allow(clippy::too_many_arguments)]
pub(super) fn resize(
    entry: &Entry,
    instance: &Instance,
    devices: &HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    surface_handle: SurfaceHandle,
    width: u32,
    height: u32,
) -> Result<()> {
    // Get surface info we need
    let (device_handle, surface, old_swapchain, format) = {
        let surface_state = surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        (
            surface_state.device_handle,
            surface_state.surface,
            surface_state.swapchain,
            surface_state.format,
        )
    };

    let logical_device = devices
        .get(&device_handle)
        .context("Surface's device is invalid")?;
    let physical_device = logical_device.physical_device;

    // Wait for all in-flight frames to complete before resizing
    unsafe { logical_device.device.device_wait_idle() }?;

    // Destroy old image views
    if let Some(surface_state) = surfaces.get(&surface_handle) {
        for view in &surface_state.swapchain_image_views {
            unsafe { logical_device.device.destroy_image_view(*view, None) };
        }
    }

    // Get new capabilities
    let surface_loader = khr::surface::Instance::new(entry, instance);
    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }
    .context("Failed to get surface capabilities")?;

    let extent = vk::Extent2D {
        width: width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    };

    let image_count = (capabilities.min_image_count + 1).min(if capabilities.max_image_count > 0 {
        capabilities.max_image_count
    } else {
        u32::MAX
    });

    // Create new swapchain (reusing old one for efficiency)
    let swapchain_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(old_swapchain);

    let swapchain_loader = khr::swapchain::Device::new(instance, &logical_device.device);
    let new_swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None) }
        .context("Failed to recreate swapchain")?;

    // Destroy old swapchain
    unsafe { swapchain_loader.destroy_swapchain(old_swapchain, None) };

    // Get new images and create views
    let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(new_swapchain) }
        .context("Failed to get swapchain images")?;

    let swapchain_image_views: Vec<vk::ImageView> = swapchain_images
        .iter()
        .map(|&image| {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe { logical_device.device.create_image_view(&view_info, None) }
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("Failed to create swapchain image views")?;

    // Update surface state - reset frame counter since we waited for idle
    if let Some(surface_state) = surfaces.get_mut(&surface_handle) {
        surface_state.swapchain = new_swapchain;
        surface_state.swapchain_images = swapchain_images;
        surface_state.swapchain_image_views = swapchain_image_views;
        surface_state.width = extent.width;
        surface_state.height = extent.height;
        surface_state.current_frame = 0;
        surface_state.current_image_index = None;
    }

    tracing::debug!("Resized surface to {}x{}", extent.width, extent.height);
    Ok(())
}

/// Get the current size of the surface.
pub(super) fn size(
    surfaces: &HashMap<SurfaceHandle, SurfaceState>,
    surface_handle: SurfaceHandle,
) -> (u32, u32) {
    surfaces
        .get(&surface_handle)
        .map(|s| (s.width, s.height))
        .unwrap_or((0, 0))
}

/// Get the format of the surface.
pub(super) fn format(
    surfaces: &HashMap<SurfaceHandle, SurfaceState>,
    surface_handle: SurfaceHandle,
) -> TextureFormat {
    surfaces
        .get(&surface_handle)
        .and_then(|s| utils::vk_to_format(s.format))
        .unwrap_or(TextureFormat::Bgra8UnormSrgb) // Safe fallback
}
