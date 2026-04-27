//! Surface and swapchain management for window presentation.

use super::types::{
    self, FrameSync, LogicalDevice, SurfaceState, TextureState, MAX_FRAMES_IN_FLIGHT,
};
use super::utils::{depth_aspect_mask, depth_format_to_vk, find_memory_type};
use super::{DeviceHandle, PipelineHandle, SurfaceHandle, SwapchainImageHandle, TextureHandle};
use crate::backend::RenderCommand;
use crate::types::{Color, DepthFormat, TextureFormat};
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
        let _ = (entry, instance);
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
    depth_format: Option<DepthFormat>,
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
    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }
            .context("Failed to get surface formats")?;

    let format = formats
        .iter()
        .find(|f| f.format == vk::Format::B8G8R8A8_SRGB || f.format == vk::Format::B8G8R8A8_UNORM)
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

    // Create swapchain: need at least `MAX_FRAMES_IN_FLIGHT` images so
    // `acquire_next_image` does not stall on availability independently of
    // the in-flight fence.
    let image_count = (capabilities.min_image_count + 1)
        .max(MAX_FRAMES_IN_FLIGHT as u32)
        .min(if capabilities.max_image_count > 0 {
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
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
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
        let image_available_semaphore = unsafe {
            logical_device
                .device
                .create_semaphore(&semaphore_info, None)
        }
        .context("Failed to create image available semaphore")?;
        let image_ready_semaphore = unsafe {
            logical_device
                .device
                .create_semaphore(&semaphore_info, None)
        }
        .context("Failed to create image ready semaphore")?;
        let render_finished_semaphore = unsafe {
            logical_device
                .device
                .create_semaphore(&semaphore_info, None)
        }
        .context("Failed to create render finished semaphore")?;

        // Create fence (signaled so first wait succeeds)
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight_fence = unsafe { logical_device.device.create_fence(&fence_info, None) }
            .context("Failed to create in-flight fence")?;

        // Allocate two primary command buffers per frame: one for the acquire-time
        // layout-prep submit and one for the render-path submit (graphics commands or
        // the compute-only present barrier).
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(logical_device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(2);
        let command_buffers =
            unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
                .context("Failed to allocate command buffer")?;

        frame_sync.push(FrameSync {
            command_buffer: command_buffers[0],
            prep_command_buffer: command_buffers[1],
            image_available_semaphore,
            image_ready_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            render_pass_submitted: false,
        });
    }

    // Create depth buffer if requested
    let (depth_image, depth_memory, depth_view) = if let Some(df) = depth_format {
        let vk_depth_format = depth_format_to_vk(df);
        let depth_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_depth_format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
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
            .context("Failed to create surface depth image")?;

        let d_mem_reqs = unsafe { logical_device.device.get_image_memory_requirements(d_image) };
        let d_memory_type = find_memory_type(
            instance,
            physical_device,
            d_mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .context("Failed to find memory type for surface depth buffer")?;

        let d_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(d_mem_reqs.size)
            .memory_type_index(d_memory_type);

        let d_memory = unsafe { logical_device.device.allocate_memory(&d_alloc_info, None) }
            .context("Failed to allocate surface depth memory")?;

        unsafe {
            logical_device
                .device
                .bind_image_memory(d_image, d_memory, 0)
        }
        .context("Failed to bind surface depth memory")?;

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
            .context("Failed to create surface depth view")?;

        (Some(d_image), Some(d_memory), Some(d_view))
    } else {
        (None, None, None)
    };

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
            present_mode,
            current_frame: 0,
            current_image_index: None,
            frame_sync,
            depth_format,
            depth_image,
            depth_memory,
            depth_view,
            current_texture_handle: None,
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
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    surface_handle: SurfaceHandle,
) {
    // Clean up any outstanding surface texture
    if let Some(tex_handle) = surfaces
        .get(&surface_handle)
        .and_then(|s| s.current_texture_handle)
    {
        unregister_surface_texture(devices, textures, tex_handle);
    }

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
                        .destroy_semaphore(frame.image_ready_semaphore, None);
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

                if let Some(depth_view) = surface_state.depth_view {
                    logical_device.device.destroy_image_view(depth_view, None);
                }
                if let Some(depth_image) = surface_state.depth_image {
                    logical_device.device.destroy_image(depth_image, None);
                }
                if let Some(depth_memory) = surface_state.depth_memory {
                    logical_device.device.free_memory(depth_memory, None);
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
///
/// After acquiring, the swapchain image is registered in the bindless descriptor
/// set as a storage image. The resulting `TextureHandle` is available via
/// `frame_texture()` until `present()` is called.
pub(super) fn acquire(
    instance: &Instance,
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    next_texture_handle: &mut TextureHandle,
    surface_handle: SurfaceHandle,
) -> Result<SwapchainImageHandle> {
    // Clean up any previously acquired surface texture that wasn't presented
    if let Some(surface_state) = surfaces.get(&surface_handle) {
        if let Some(tex_handle) = surface_state.current_texture_handle {
            tracing::warn!("Previous surface texture was not presented; cleaning up");
            unregister_surface_texture(devices, textures, tex_handle);
        }
    }

    // Get surface state and current frame index
    let (device_handle, current_frame, swapchain, in_flight_fence, image_available_semaphore) = {
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
    let wait_result = unsafe {
        logical_device
            .device
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)
    };
    if let Err(e) = &wait_result {
        tracing::warn!(
            surface_handle,
            %device_handle,
            current_frame,
            pending_deferred = logical_device.deletion_queue.pending.len(),
            result = ?e,
            "surface acquire: wait_for_fences (frame fence) failed"
        );
    }
    wait_result.context("Failed to wait for frame fence")?;

    {
        let surface_state = surfaces.get_mut(&surface_handle).unwrap();
        let cf = surface_state.current_frame;
        surface_state.frame_sync[cf].render_pass_submitted = false;
    }

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
    let reset_result = unsafe { logical_device.device.reset_fences(&[in_flight_fence]) };
    if let Err(e) = &reset_result {
        tracing::warn!(
            surface_handle,
            %device_handle,
            current_frame,
            result = ?e,
            "surface acquire: reset_fences (frame fence) failed"
        );
    }
    reset_result.context("Failed to reset frame fence")?;

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
            // Submit a "prep" barrier on the acquired image: `UNDEFINED → GENERAL`,
            // waiting on `image_available_semaphore` and signaling `image_ready_semaphore`.
            //
            // After `vkAcquireNextImageKHR`, the swapchain image must not be accessed
            // until `image_available_semaphore` has been waited on, and its layout is
            // either `UNDEFINED` (first use) or `PRESENT_SRC_KHR` (recycled). Compute
            // shaders writing via `RWTexture2D` require `GENERAL`. This prep submit
            // consumes the acquire semaphore, performs the layout transition once, and
            // signals `image_ready_semaphore` which both the graphics render path and
            // the compute-only present path wait on — so downstream compute submits
            // can safely write the image in any order on the same queue.
            let (prep_cmd, image_ready_semaphore, image) = {
                let surface_state = surfaces.get_mut(&surface_handle).unwrap();
                surface_state.current_image_index = Some(image_index);
                let frame = &surface_state.frame_sync[surface_state.current_frame];
                (
                    frame.prep_command_buffer,
                    frame.image_ready_semaphore,
                    surface_state.swapchain_images[image_index as usize],
                )
            };

            unsafe {
                logical_device
                    .device
                    .reset_command_buffer(prep_cmd, vk::CommandBufferResetFlags::empty())
            }
            .context("Failed to reset acquire prep command buffer")?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                logical_device
                    .device
                    .begin_command_buffer(prep_cmd, &begin_info)
            }
            .context("Failed to begin acquire prep command buffer")?;

            let prep_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags2::TRANSFER_WRITE,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let dep_info = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&prep_barrier));
            unsafe {
                logical_device
                    .device
                    .cmd_pipeline_barrier2(prep_cmd, &dep_info)
            };

            unsafe { logical_device.device.end_command_buffer(prep_cmd) }
                .context("Failed to end acquire prep command buffer")?;

            let wait_semaphores = [image_available_semaphore];
            let wait_stages = [vk::PipelineStageFlags::TOP_OF_PIPE];
            let signal_semaphores = [image_ready_semaphore];
            let prep_submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(std::slice::from_ref(&prep_cmd))
                .signal_semaphores(&signal_semaphores);
            let prep_submit_result = unsafe {
                logical_device.device.queue_submit(
                    logical_device.queue,
                    std::slice::from_ref(&prep_submit),
                    vk::Fence::null(),
                )
            };
            if let Err(e) = &prep_submit_result {
                tracing::warn!(
                    surface_handle,
                    %device_handle,
                    current_frame,
                    image_index,
                    result = ?e,
                    "surface acquire: prep barrier queue_submit failed"
                );
            }
            prep_submit_result.context("Failed to submit acquire prep barrier")?;

            // Register the swapchain image as a storage texture for compute access.
            // The image view is created for GENERAL layout so compute shaders can write
            // via RWTexture2D in bindless.
            let (width, height, vk_format) = {
                let surface_state = surfaces.get(&surface_handle).unwrap();
                (
                    surface_state.width,
                    surface_state.height,
                    surface_state.format,
                )
            };
            let goldy_format =
                super::utils::vk_to_format(vk_format).unwrap_or(TextureFormat::Bgra8UnormSrgb);

            let tex_handle = register_surface_texture(
                devices,
                textures,
                next_texture_handle,
                device_handle,
                image,
                vk_format,
                goldy_format,
                width,
                height,
            )?;

            let surface_state = surfaces.get_mut(&surface_handle).unwrap();
            surface_state.current_texture_handle = Some(tex_handle);

            Ok(image_index as SwapchainImageHandle)
        }
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            tracing::info!("Swapchain out of date - resize required");
            anyhow::bail!("Surface out of date - call resize() and retry")
        }
        Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
            tracing::error!("Surface lost");
            anyhow::bail!("Surface lost - recreate surface")
        }
        Err(e) => {
            tracing::warn!(
                surface_handle,
                %device_handle,
                current_frame,
                result = ?e,
                "acquire_next_image failed"
            );
            anyhow::bail!("Failed to acquire swapchain image: {:?}", e)
        }
    }
}

/// Get the texture handle for the currently acquired surface frame.
pub(super) fn frame_texture(
    surfaces: &HashMap<SurfaceHandle, SurfaceState>,
    surface_handle: SurfaceHandle,
) -> Option<TextureHandle> {
    surfaces
        .get(&surface_handle)
        .and_then(|s| s.current_texture_handle)
}

/// Render commands to the surface's current swapchain image.
#[allow(clippy::too_many_arguments)]
pub(super) fn render<F>(
    devices: &HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
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
    ) -> Result<()>,
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

    // Find clear color and clear depth
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
            RenderCommand::ClearDepth(d) => Some(*d),
            _ => None,
        })
        .unwrap_or(1.0);

    // Begin command buffer
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
        .context("Failed to begin command buffer")?;

    // Transition image to color attachment. The image is in `GENERAL` layout at this
    // point (acquire submits a prep barrier `UNDEFINED → GENERAL`), but we pass
    // `UNDEFINED` as old_layout to let the driver discard any prior contents — the
    // render path is expected to fully overwrite the frame (clear + draw), so we
    // don't need to preserve compute writes here.
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

    // Prepare image barriers - color always, depth if present
    let mut barriers = vec![color_barrier];

    // Add depth barrier if depth buffer exists
    if let (Some(depth_img), Some(df)) = (surface_state.depth_image, surface_state.depth_format) {
        let depth_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
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

    // Create depth attachment if present
    let depth_attachment = surface_state.depth_view.map(|dv| {
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

    // Add depth attachment if present
    if let Some(ref depth_att) = depth_attachment {
        rendering_info = rendering_info.depth_attachment(depth_att);
    }

    unsafe {
        logical_device
            .device
            .cmd_begin_rendering(cmd, &rendering_info)
    };

    // Negative viewport height flips Y to match DX12 (core since Vulkan 1.1)
    let viewport = vk::Viewport {
        x: 0.0,
        y: height as f32, // Start from bottom
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
    record_commands_fn(cmd, commands, logical_device, &mut current_pipeline)?;

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

    let dep_info =
        vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

    unsafe { logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info) };

    // End command buffer
    unsafe { logical_device.device.end_command_buffer(cmd) }
        .context("Failed to end command buffer")?;

    // Get per-frame sync primitives. We wait on `image_ready_semaphore` (signaled by
    // the acquire-time prep submit) rather than `image_available_semaphore` — the
    // prep submit already consumed the latter to transition the image into `GENERAL`.
    let frame = &surface_state.frame_sync[current_frame];
    let wait_semaphores = [frame.image_ready_semaphore];
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

    if let Some(surface_state) = surfaces.get_mut(&surface_handle) {
        surface_state.frame_sync[current_frame].render_pass_submitted = true;
    }

    Ok(())
}

/// Present the rendered image to the screen.
///
/// Unregisters the transient surface texture from the bindless descriptor set,
/// then queues the swapchain image for presentation.
pub(super) fn present(
    instance: &Instance,
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    surface_handle: SurfaceHandle,
    _image: SwapchainImageHandle,
) -> Result<()> {
    // Unregister the transient surface texture before presenting
    if let Some(tex_handle) = surfaces
        .get(&surface_handle)
        .and_then(|s| s.current_texture_handle)
    {
        unregister_surface_texture(devices, textures, tex_handle);
    }

    let surface_state = surfaces
        .get_mut(&surface_handle)
        .context("Invalid surface handle")?;

    surface_state.current_texture_handle = None;

    let image_index = surface_state
        .current_image_index
        .context("No image to present - call surface_render first")?;

    let current_frame = surface_state.current_frame;
    let image = surface_state.swapchain_images[image_index as usize];
    let render_pass_submitted = surface_state.frame_sync[current_frame].render_pass_submitted;
    let device_handle = surface_state.device_handle;

    let logical_device = devices
        .get(&device_handle)
        .context("Surface's device is invalid")?;

    // Graphics path: `surface_render` waits `image_available`, writes the swapchain image,
    // signals `render_finished`, and sets `render_pass_submitted`.
    // Compute-only path: host waits on the compute fence, but nothing signals
    // `render_finished` or `in_flight_fence` — `queue_present` would block forever.
    // Submit a layout barrier (GENERAL → PRESENT_SRC) with the same semaphore/fence pattern.
    if !render_pass_submitted {
        let frame = &surface_state.frame_sync[current_frame];
        let cmd = frame.command_buffer;

        unsafe {
            logical_device
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
        }
        .context("Failed to reset compute-present barrier command buffer")?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
            .context("Failed to begin compute-present barrier command buffer")?;

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
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
        unsafe { logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info) };

        unsafe { logical_device.device.end_command_buffer(cmd) }
            .context("Failed to end compute-present barrier command buffer")?;

        // Wait on `image_ready_semaphore` (signaled by the acquire-time prep submit).
        // The prep barrier already consumed `image_available_semaphore` and transitioned
        // the image from `UNDEFINED` into `GENERAL`, so compute shaders have already
        // written it at the correct layout by the time we reach this post-barrier.
        let frame = &surface_state.frame_sync[current_frame];
        let wait_semaphores = [frame.image_ready_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
        let signal_semaphores = [frame.render_finished_semaphore];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd))
            .signal_semaphores(&signal_semaphores);

        let submit_result = unsafe {
            logical_device.device.queue_submit(
                logical_device.queue,
                std::slice::from_ref(&submit_info),
                frame.in_flight_fence,
            )
        };
        if let Err(e) = &submit_result {
            tracing::warn!(
                surface_handle,
                %device_handle,
                current_frame,
                image_index,
                result = ?e,
                "compute-present layout barrier queue_submit failed"
            );
        }
        submit_result.context("Failed to submit compute-present layout barrier")?;
    }

    let frame = &surface_state.frame_sync[current_frame];

    let swapchain_loader = khr::swapchain::Device::new(instance, &logical_device.device);

    let swapchains = [surface_state.swapchain];
    let image_indices = [image_index];
    let wait_semaphores = [frame.render_finished_semaphore];

    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&wait_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    let result = unsafe { swapchain_loader.queue_present(logical_device.queue, &present_info) };
    if let Err(e) = &result {
        tracing::warn!(
            surface_handle,
            %device_handle,
            current_frame,
            image_index,
            result = ?e,
            "queue_present failed"
        );
    }

    // Clear the current image and advance frame counter
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
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => Ok(()),
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
    let (device_handle, surface, old_swapchain, format, depth_fmt, stored_present_mode) = {
        let surface_state = surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        (
            surface_state.device_handle,
            surface_state.surface,
            surface_state.swapchain,
            surface_state.format,
            surface_state.depth_format,
            surface_state.present_mode,
        )
    };

    let logical_device = devices
        .get(&device_handle)
        .context("Surface's device is invalid")?;
    let physical_device = logical_device.physical_device;

    // Wait for all in-flight frames to complete before resizing
    unsafe { logical_device.device.device_wait_idle() }?;

    // Destroy old depth buffer (must be before swapchain recreation)
    if let Some(surface_state) = surfaces.get(&surface_handle) {
        if let Some(depth_view) = surface_state.depth_view {
            unsafe { logical_device.device.destroy_image_view(depth_view, None) };
        }
        if let Some(depth_image) = surface_state.depth_image {
            unsafe { logical_device.device.destroy_image(depth_image, None) };
        }
        if let Some(depth_memory) = surface_state.depth_memory {
            unsafe { logical_device.device.free_memory(depth_memory, None) };
        }
    }

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

    let image_count = (capabilities.min_image_count + 1)
        .max(MAX_FRAMES_IN_FLIGHT as u32)
        .min(if capabilities.max_image_count > 0 {
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
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(stored_present_mode)
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

    // Recreate depth buffer if the surface had one
    let (new_depth_image, new_depth_memory, new_depth_view) = if let Some(df) = depth_fmt {
        let vk_depth_format = depth_format_to_vk(df);
        let depth_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_depth_format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
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
            .context("Failed to create surface depth image on resize")?;

        let d_mem_reqs = unsafe { logical_device.device.get_image_memory_requirements(d_image) };
        let d_memory_type = find_memory_type(
            instance,
            physical_device,
            d_mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .context("Failed to find memory type for surface depth on resize")?;

        let d_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(d_mem_reqs.size)
            .memory_type_index(d_memory_type);

        let d_memory = unsafe { logical_device.device.allocate_memory(&d_alloc_info, None) }
            .context("Failed to allocate surface depth memory on resize")?;

        unsafe {
            logical_device
                .device
                .bind_image_memory(d_image, d_memory, 0)
        }
        .context("Failed to bind surface depth memory on resize")?;

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
            .context("Failed to create surface depth view on resize")?;

        (Some(d_image), Some(d_memory), Some(d_view))
    } else {
        (None, None, None)
    };

    // Update surface state - reset frame counter since we waited for idle
    if let Some(surface_state) = surfaces.get_mut(&surface_handle) {
        surface_state.swapchain = new_swapchain;
        surface_state.swapchain_images = swapchain_images;
        surface_state.swapchain_image_views = swapchain_image_views;
        surface_state.width = extent.width;
        surface_state.height = extent.height;
        surface_state.current_frame = 0;
        surface_state.current_image_index = None;
        surface_state.current_texture_handle = None;
        surface_state.depth_image = new_depth_image;
        surface_state.depth_memory = new_depth_memory;
        surface_state.depth_view = new_depth_view;
        surface_state.present_mode = stored_present_mode;
    }

    tracing::debug!("Resized surface to {}x{}", extent.width, extent.height);

    Ok(())
}

/// Set swapchain present mode (vsync). Recreates the swapchain when the mode changes.
pub(super) fn set_present_mode(
    entry: &Entry,
    instance: &Instance,
    devices: &HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    surface_handle: SurfaceHandle,
    mode: crate::types::PresentMode,
) -> Result<()> {
    let (w, h, current_vk) = {
        let s = surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        (s.width, s.height, s.present_mode)
    };

    let (physical_device, vk_surface) = {
        let surface_state = surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        let pd = devices
            .get(&surface_state.device_handle)
            .context("Surface's device is invalid")?
            .physical_device;
        (pd, surface_state.surface)
    };

    let surface_loader = khr::surface::Instance::new(entry, instance);
    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, vk_surface)
    }
    .context("Failed to get present modes")?;

    let vk_mode = pick_vk_present_mode(mode, &present_modes)?;
    if vk_mode == current_vk {
        return Ok(());
    }

    {
        let surface_state = surfaces
            .get_mut(&surface_handle)
            .context("Invalid surface handle")?;
        surface_state.present_mode = vk_mode;
    }

    resize(entry, instance, devices, surfaces, surface_handle, w, h)
}

fn pick_vk_present_mode(
    requested: crate::types::PresentMode,
    present_modes: &[vk::PresentModeKHR],
) -> Result<vk::PresentModeKHR> {
    use crate::types::PresentMode;
    let vk_target = match requested {
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        PresentMode::Auto => {
            if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else {
                vk::PresentModeKHR::FIFO
            }
        }
    };
    if !present_modes.contains(&vk_target) {
        anyhow::bail!(
            "Requested present mode {:?} is not supported by this surface",
            requested
        );
    }
    Ok(vk_target)
}

/// Active swapchain present mode as a Goldy enum.
pub(super) fn get_present_mode(
    surfaces: &HashMap<SurfaceHandle, SurfaceState>,
    surface_handle: SurfaceHandle,
) -> crate::types::PresentMode {
    surfaces
        .get(&surface_handle)
        .map(|s| vk_to_goldy_present_mode(s.present_mode))
        .unwrap_or_default()
}

fn vk_to_goldy_present_mode(mode: vk::PresentModeKHR) -> crate::types::PresentMode {
    use crate::types::PresentMode;
    match mode {
        vk::PresentModeKHR::FIFO | vk::PresentModeKHR::FIFO_RELAXED => PresentMode::Fifo,
        vk::PresentModeKHR::MAILBOX => PresentMode::Mailbox,
        vk::PresentModeKHR::IMMEDIATE => PresentMode::Immediate,
        _ => PresentMode::Fifo,
    }
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
        .and_then(|s| super::utils::vk_to_format(s.format))
        .unwrap_or(TextureFormat::Bgra8UnormSrgb) // Safe fallback
}

// ---------------------------------------------------------------------------
// Transient surface texture registration
// ---------------------------------------------------------------------------
// These functions register/unregister swapchain images in the global bindless
// descriptor set so that compute shaders can write to them via RWTexture2D.
// The image view and descriptor are created on acquire and torn down on present.
// The underlying VkImage is owned by the swapchain — we must NOT destroy it.

/// Register a swapchain image as a transient storage texture.
///
/// Creates a VkImageView with GENERAL layout intent and writes a storage-image
/// descriptor into the bindless set. Returns a TextureHandle that the caller
/// stores in `SurfaceState::current_texture_handle`.
#[allow(clippy::too_many_arguments)]
fn register_surface_texture(
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    next_texture_handle: &mut TextureHandle,
    device_handle: DeviceHandle,
    image: vk::Image,
    vk_format: vk::Format,
    goldy_format: TextureFormat,
    width: u32,
    height: u32,
) -> Result<TextureHandle> {
    let handle = *next_texture_handle;
    *next_texture_handle += 1;

    let logical_device = devices
        .get_mut(&device_handle)
        .context("Device no longer valid")?;

    // Create an image view for compute storage access
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk_format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let view = unsafe { logical_device.device.create_image_view(&view_info, None) }
        .context("Failed to create surface texture image view")?;

    // Register as a storage image in the bindless descriptor set
    let is_storage_image = true;
    let bindless_index = logical_device
        .resource_registry
        .register_texture(handle, is_storage_image);

    // Write the storage-image descriptor
    if let Some(descriptor_set) = logical_device.bindless_descriptor_set {
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(types::bindless_bindings::STORAGE_IMAGES)
            .dst_array_element(bindless_index)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&image_info));

        unsafe {
            logical_device
                .device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }

        tracing::trace!(
            "Registered surface texture {} at storage image bindless index {}",
            handle,
            bindless_index,
        );
    }

    textures.insert(
        handle,
        TextureState {
            device_handle,
            width,
            height,
            format: goldy_format,
            image,
            // Swapchain images don't have separately allocated memory — null sentinel
            memory: vk::DeviceMemory::null(),
            view,
            staging_buffer: None,
            staging_memory: None,
            bindless_index: Some(bindless_index),
            current_layout: vk::ImageLayout::GENERAL,
        },
    );

    tracing::debug!(
        "Registered surface texture {} ({}x{}, bindless={})",
        handle,
        width,
        height,
        bindless_index,
    );

    Ok(handle)
}

/// Unregister a transient surface texture.
///
/// Removes the TextureState, destroys the image view, and unregisters from
/// the bindless resource registry. Does NOT destroy the VkImage (owned by the
/// swapchain).
fn unregister_surface_texture(
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    tex_handle: TextureHandle,
) {
    if let Some(tex_state) = textures.remove(&tex_handle) {
        if let Some(device) = devices.get_mut(&tex_state.device_handle) {
            device.resource_registry.unregister_texture(tex_handle);

            // Destroy the image view we created in register_surface_texture.
            // The VkImage itself belongs to the swapchain — do NOT destroy it.
            unsafe {
                device.device.destroy_image_view(tex_state.view, None);
            }
        }
        tracing::debug!("Unregistered surface texture {}", tex_handle);
    }
}
