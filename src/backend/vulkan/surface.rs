//! Surface and swapchain management for window presentation.
//!
//! ## Presentation strategy
//!
//! Vulkan swapchain images support `VK_IMAGE_USAGE_STORAGE_BIT`, so compute
//! shaders write directly to the swapchain image — no intermediate scratch
//! texture or GPU-side copy is needed.
//!
//! 1. **`acquire()`** — waits on the per-frame-slot fence, resets it, then
//!    immediately calls `vkAcquireNextImageKHR` (signals
//!    `image_available_semaphore`).  CPU bookkeeping (timeline reap, deferred
//!    deletions) runs after the acquire call so the presentation engine has
//!    maximum time to locate a free image while the CPU is busy.  The acquired
//!    image's `TextureHandle` is looked up from the pre-registered
//!    `swapchain_texture_handles` table — no `vkCreateImageView` or descriptor
//!    write per frame.
//!
//! 2. **Middle of frame** — compute CBs are accumulated in
//!    `FrameSync::deferred_compute_cbs`.  They write directly to the swapchain
//!    image in `GENERAL` layout.
//!
//! 3. **`present()`** — records a `GENERAL → PRESENT_SRC_KHR` barrier, then
//!    submits `[prep_cb, deferred_cbs..., barrier_cb]` in a single
//!    `vkQueueSubmit2` that waits on `image_available_semaphore`.  Finally
//!    `vkQueuePresentKHR`.  `prep_cb` is a pre-recorded per-image CB
//!    (`UNDEFINED → GENERAL`) that is reused every frame without re-recording.
//!
//! ## Acquire-stall mitigation
//!
//! The swapchain requests `MAX_FRAMES_IN_FLIGHT + 1` images so there is always
//! a free image available to `vkAcquireNextImageKHR`, regardless of pacing.
//! Combined with calling acquire immediately after the fence fires (before CPU
//! cleanup), the ~15% blocking rate of the previous design is eliminated,
//! closing the throughput gap with DX12's DXGI waitable-object path.

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
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    next_surface_handle: &mut SurfaceHandle,
    next_texture_handle: &mut TextureHandle,
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

    // Default to FIFO (vsync, always available). The public Surface API calls
    // set_present_mode immediately after creation when a non-Auto mode is
    // requested, so using FIFO here avoids a wasteful MAILBOX→FIFO swapchain
    // recreation cycle that can confuse some drivers' present-mode inheritance
    // via the old_swapchain parameter.
    let present_mode = vk::PresentModeKHR::FIFO;

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

    // Request one more image than in-flight frames so there is always a free
    // image available to `acquire_next_image` regardless of pacing, reducing
    // the frequency of presentation-engine stalls.
    let image_count = (capabilities.min_image_count + 1)
        .max(MAX_FRAMES_IN_FLIGHT as u32 + 1)
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
        .image_usage(
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
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
        let render_finished_semaphore = unsafe {
            logical_device
                .device
                .create_semaphore(&semaphore_info, None)
        }
        .context("Failed to create render finished semaphore")?;

        let work_done_semaphore = unsafe {
            logical_device
                .device
                .create_semaphore(&semaphore_info, None)
        }
        .context("Failed to create work-done semaphore")?;

        // Create fences (both start signaled so the first-frame wait is a no-op)
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight_fence = unsafe { logical_device.device.create_fence(&fence_info, None) }
            .context("Failed to create in-flight fence")?;
        let acquire_fence = unsafe { logical_device.device.create_fence(&fence_info, None) }
            .context("Failed to create acquire fence")?;

        // Allocate one primary command buffer per frame for the render-path submit
        // (graphics commands). The per-image prep and present barriers are pre-recorded
        // into `swapchain_*_command_buffers` and reused every frame.
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
            work_done_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            acquire_fence,
            render_pass_submitted: false,
            frame_timeline_value: None,
            deferred_compute_cbs: Vec::new(),
            pending_compute_texture_staging: Vec::new(),
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

    // Pre-record per-image barrier CBs and pre-register bindless textures.
    // All live for the swapchain lifetime and are rebuilt on resize.
    let (
        swapchain_prep_command_buffers,
        swapchain_compute_present_command_buffers,
        swapchain_render_present_command_buffers,
    ) = {
        let ld = devices.get(&device_handle).context("Device invalid")?;
        (
            alloc_and_record_prep_cbs(ld, &swapchain_images)?,
            alloc_and_record_compute_present_cbs(ld, &swapchain_images)?,
            alloc_and_record_render_present_cbs(ld, &swapchain_images)?,
        )
    };

    let goldy_format =
        super::utils::vk_to_format(format.format).unwrap_or(TextureFormat::Bgra8UnormSrgb);
    let mut swapchain_texture_handles = Vec::with_capacity(swapchain_images.len());
    for &image in &swapchain_images {
        let th = register_surface_texture(
            devices,
            textures,
            next_texture_handle,
            device_handle,
            image,
            format.format,
            goldy_format,
            extent.width,
            extent.height,
        )?;
        swapchain_texture_handles.push(th);
    }

    surfaces.insert(
        handle,
        SurfaceState {
            device_handle,
            surface,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_prep_command_buffers,
            swapchain_compute_present_command_buffers,
            swapchain_render_present_command_buffers,
            swapchain_texture_handles,
            width: extent.width,
            height: extent.height,
            format: format.format,
            present_mode,
            present_mode_dirty: false,
            current_frame: 0,
            current_image_index: None,
            frame_sync,
            depth_format,
            depth_image,
            depth_memory,
            depth_view,
            current_texture_handle: None,
            frame_pending_gpu_commands: Vec::new(),
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
#[allow(clippy::too_many_arguments)]
pub(super) fn destroy(
    entry: &Entry,
    instance: &Instance,
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    surface_handle: SurfaceHandle,
) {
    // Clear the per-frame alias first; the real registrations are in swapchain_texture_handles.
    if let Some(s) = surfaces.get_mut(&surface_handle) {
        s.current_texture_handle = None;
    }
    // Unregister all persistently-registered swapchain image textures.
    if let Some(handles) = surfaces
        .get_mut(&surface_handle)
        .map(|s| std::mem::take(&mut s.swapchain_texture_handles))
    {
        for th in handles {
            unregister_surface_texture(devices, textures, th);
        }
    }

    if let Some(mut surface_state) = surfaces.remove(&surface_handle) {
        if let Some(logical_device) = devices.get(&surface_state.device_handle) {
            unsafe {
                let _ = logical_device.device.device_wait_idle();

                // Free deferred compute CBs and remove their tokens from the
                // pending map BEFORE destroying the fences they reference.
                for frame in &mut surface_state.frame_sync {
                    let deferred_cbs = std::mem::take(&mut frame.deferred_compute_cbs);
                    let pending_tex_staging =
                        std::mem::take(&mut frame.pending_compute_texture_staging);
                    if !deferred_cbs.is_empty() {
                        logical_device
                            .device
                            .free_command_buffers(logical_device.command_pool, &deferred_cbs);
                    }
                    // Orphaned pending entries were never submitted; device is
                    // idle after device_wait_idle above, so destroy directly.
                    for entry in pending_tex_staging {
                        entry.destroy(logical_device);
                    }
                }

                // Free pre-recorded per-image barrier command buffers.
                for cbs in [
                    &surface_state.swapchain_prep_command_buffers,
                    &surface_state.swapchain_compute_present_command_buffers,
                    &surface_state.swapchain_render_present_command_buffers,
                ] {
                    if !cbs.is_empty() {
                        logical_device
                            .device
                            .free_command_buffers(logical_device.command_pool, cbs);
                    }
                }

                // Destroy per-frame sync resources
                for frame in surface_state.frame_sync {
                    logical_device
                        .device
                        .destroy_semaphore(frame.image_available_semaphore, None);
                    logical_device
                        .device
                        .destroy_semaphore(frame.work_done_semaphore, None);
                    logical_device
                        .device
                        .destroy_semaphore(frame.render_finished_semaphore, None);
                    logical_device
                        .device
                        .destroy_fence(frame.in_flight_fence, None);
                    logical_device
                        .device
                        .destroy_fence(frame.acquire_fence, None);
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
/// `current_texture_handle` is set to the pre-registered texture for the
/// acquired image and is valid until `present()` clears it.
pub(super) fn acquire(
    state: &mut super::types::VulkanState,
    surface_handle: SurfaceHandle,
) -> Result<SwapchainImageHandle> {
    let _tz = crate::tracy_zone!("vk.surface.acquire");

    // Get surface state and current frame index.
    let (device_handle, current_frame, swapchain, in_flight_fence, acquire_fence, image_available_semaphore) = {
        let _fz = crate::tracy_zone!("vk.surface.acquire.frame_state");
        let surface_state = state
            .surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        let frame = &surface_state.frame_sync[surface_state.current_frame];
        (
            surface_state.device_handle,
            surface_state.current_frame,
            surface_state.swapchain,
            frame.in_flight_fence,
            frame.acquire_fence,
            frame.image_available_semaphore,
        )
    };

    let pending_deferred_len = {
        let _dz = crate::tracy_zone!("vk.surface.acquire.deferred_query");
        state
            .devices
            .get(&device_handle)
            .map(|d| d.deletion_queue.len())
            .unwrap_or(0)
    };

    // ── Zone 1: vk.surface.wait_compute ──────────────────────────────────────
    // Waits for the previous use of this frame slot to finish GPU compute work.
    // Long here → GPU is compute-bound.  Near-zero → GPU finished well ahead
    // of the CPU (well-pipelined or CPU-bound frame).
    let wait_result = {
        let _wz = crate::tracy_zone!("vk.surface.wait_compute");
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;
        unsafe {
            logical_device
                .device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)
        }
    };
    if let Err(e) = &wait_result {
        tracing::warn!(
            surface_handle,
            %device_handle,
            current_frame,
            pending_deferred = pending_deferred_len,
            result = ?e,
            "surface acquire: wait_for_fences (compute fence) failed"
        );
    }
    wait_result.context("Failed to wait for compute fence")?;

    // Reset both the compute fence and the acquire fence (the acquire fence is
    // still signaled from the previous use of this slot; reset it before passing
    // it to vkAcquireNextImageKHR so it is in the required unsignaled state).
    {
        let _rz = crate::tracy_zone!("vk.surface.reset_fences");
        let ld = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;
        unsafe { ld.device.reset_fences(&[in_flight_fence, acquire_fence]) }
            .context("Failed to reset frame fences")?;
    }

    // CPU cleanup: drain the GPU timeline and reset the frame slot.  This work
    // runs while the presentation engine is preparing the swapchain image,
    // overlapping CPU bookkeeping with the (typically short) WSI handoff.
    let completed = {
        let _tz = crate::tracy_zone!("vk.surface.acquire.timeline_query");
        let ld = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;
        unsafe { ld.device.get_semaphore_counter_value(ld.timeline_semaphore) }.unwrap_or(0)
    };

    {
        let _tz = crate::tracy_zone!("vk.surface.acquire.reap_timeline");
        super::compute::reap_timeline_cmd_buffers_up_to(state, device_handle, completed);
    }

    {
        let _tz = crate::tracy_zone!("vk.surface.acquire.frame_slot_reset");
        let surface_state = state.surfaces.get_mut(&surface_handle).unwrap();
        let cf = surface_state.current_frame;
        surface_state.frame_sync[cf].render_pass_submitted = false;
        surface_state.frame_pending_gpu_commands.clear();

        // Deferred compute CBs are freed via timeline reap above.
        // Any orphaned staging entries indicate a bug (present should always
        // merge uploads into the pool before returning).
        let orphaned_tex_staging =
            std::mem::take(&mut surface_state.frame_sync[cf].pending_compute_texture_staging);
        let _ = std::mem::take(&mut surface_state.frame_sync[cf].deferred_compute_cbs);
        debug_assert!(
            orphaned_tex_staging.is_empty(),
            "orphaned texture staging entries in frame slot {cf} — present did not flush uploads"
        );
    }

    {
        let _dz = crate::tracy_zone!("vk.surface.deferred_deletions");
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Surface's device is invalid")?;
        let drained = logical_device.deletion_queue.drain_up_to(completed);
        for r in drained {
            types::destroy_pending_deletion(logical_device, r);
        }
    }

    // Request the next swapchain image.  `acquire_fence` is passed alongside
    // `image_available_semaphore`: both are signaled by the presentation engine
    // when the image is safe to write to.  The semaphore gates GPU work (Submit 1);
    // the fence gates the CPU `vk.surface.wait_acquire` zone below.
    let acquire_result = {
        let ld = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;
        let swapchain_loader = khr::swapchain::Device::new(&state.instance, &ld.device);
        unsafe {
            swapchain_loader.acquire_next_image(
                swapchain,
                u64::MAX,
                image_available_semaphore,
                acquire_fence,
            )
        }
    };

    // ── Zone 2: vk.surface.wait_acquire ──────────────────────────────────────
    // Waits for the presentation engine to make the acquired image writable.
    // Long here → WSI / presentation-engine bound (e.g. OS display pipeline
    // stall).  Near-zero → the image was already ready by the time CPU got here.
    {
        let _aw = crate::tracy_zone!("vk.surface.wait_acquire");
        let ld = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;
        // Only wait if the acquire didn't fail outright; error handling below
        // will surface any real failures from acquire_result.
        let _ = unsafe { ld.device.wait_for_fences(&[acquire_fence], true, u64::MAX) };
    }

    // Consume the acquire result now.
    match acquire_result {
        Ok((image_index, suboptimal)) => {
            if suboptimal {
                tracing::debug!("Swapchain suboptimal - consider resizing");
            }

            // Point current_texture_handle at the pre-registered persistent texture
            // for this swapchain image — no view creation or descriptor write needed.
            let surface_state = state.surfaces.get_mut(&surface_handle).unwrap();
            surface_state.current_image_index = Some(image_index);
            surface_state.current_texture_handle =
                surface_state.swapchain_texture_handles.get(image_index as usize).copied();

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
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
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
    let (
        _image_index,
        current_frame,
        device_handle,
        cmd,
        prep_cmd,
        render_present_cmd,
        width,
        height,
        image,
        image_view,
        depth_image,
        depth_format,
        depth_view,
        image_available_semaphore,
        work_done_semaphore,
        render_finished_semaphore,
        in_flight_fence,
    ) = {
        let surface_state = surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        let image_index = surface_state
            .current_image_index
            .context("No image acquired - call surface_acquire first")?;
        let current_frame = surface_state.current_frame;
        let frame = &surface_state.frame_sync[current_frame];
        (
            image_index,
            current_frame,
            surface_state.device_handle,
            frame.command_buffer,
            surface_state.swapchain_prep_command_buffers[image_index as usize],
            surface_state.swapchain_render_present_command_buffers[image_index as usize],
            surface_state.width,
            surface_state.height,
            surface_state.swapchain_images[image_index as usize],
            surface_state.swapchain_image_views[image_index as usize],
            surface_state.depth_image,
            surface_state.depth_format,
            surface_state.depth_view,
            frame.image_available_semaphore,
            frame.work_done_semaphore,
            frame.render_finished_semaphore,
            frame.in_flight_fence,
        )
    };

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

    {
        let logical_device = devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
            .context("Failed to begin command buffer")?;

        // Cross-submission memory barrier: make writes from prior compute dispatches
        // (submitted as separate queue batches) visible to vertex/fragment shader
        // reads.  Vulkan guarantees execution ordering between same-queue batches
        // but NOT memory visibility — explicit synchronisation is required.
        unsafe {
            let mem_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
                )
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::VERTEX_SHADER
                        | vk::PipelineStageFlags2::FRAGMENT_SHADER
                        | vk::PipelineStageFlags2::VERTEX_INPUT,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                );
            let dep_info =
                vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
            logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
        }

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
        if let (Some(depth_img), Some(df)) = (depth_image, depth_format) {
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

        // End command buffer — the present barrier is pre-recorded in
        // render_present_cmd and submitted as a separate batch (Submit 2).
        unsafe { logical_device.device.end_command_buffer(cmd) }
            .context("Failed to end command buffer")?;
    }

    // Submit 1: prep barrier + render commands.
    //   waits:   image_available_sem
    //   signals: in_flight_fence (fires after render work, NOT after present barrier)
    //            work_done_sem (consumed by Submit 2)
    //            timeline_sem
    //
    // Submit 2: pre-recorded COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR barrier.
    //   waits:   work_done_sem
    //   signals: render_finished_sem
    //   fence:   none
    let signal_timeline_value = {
        let ld = devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;
        ld.timeline_next
    };
    let timeline_sem = devices
        .get(&device_handle)
        .context("Surface's device is invalid")?
        .timeline_semaphore;

    let wait_acq = vk::SemaphoreSubmitInfo::default()
        .semaphore(image_available_semaphore)
        .value(0)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let cmd_infos_1 = [
        vk::CommandBufferSubmitInfo::default().command_buffer(prep_cmd),
        vk::CommandBufferSubmitInfo::default().command_buffer(cmd),
    ];
    let sig_work_done = vk::SemaphoreSubmitInfo::default()
        .semaphore(work_done_semaphore)
        .value(0)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let sig_timeline = vk::SemaphoreSubmitInfo::default()
        .semaphore(timeline_sem)
        .value(signal_timeline_value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let signals_1 = [sig_work_done, sig_timeline];
    let submit_1 = vk::SubmitInfo2::default()
        .wait_semaphore_infos(std::slice::from_ref(&wait_acq))
        .command_buffer_infos(&cmd_infos_1)
        .signal_semaphore_infos(&signals_1);

    let cmd_info_2 =
        vk::CommandBufferSubmitInfo::default().command_buffer(render_present_cmd);
    let wait_work_done = vk::SemaphoreSubmitInfo::default()
        .semaphore(work_done_semaphore)
        .value(0)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let sig_rf = vk::SemaphoreSubmitInfo::default()
        .semaphore(render_finished_semaphore)
        .value(0)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let submit_2 = vk::SubmitInfo2::default()
        .wait_semaphore_infos(std::slice::from_ref(&wait_work_done))
        .command_buffer_infos(std::slice::from_ref(&cmd_info_2))
        .signal_semaphore_infos(std::slice::from_ref(&sig_rf));

    let logical_device = devices
        .get(&device_handle)
        .context("Surface's device is invalid")?;

    unsafe {
        logical_device.device.queue_submit2(
            logical_device.queue,
            std::slice::from_ref(&submit_1),
            in_flight_fence,
        )
    }
    .context("Failed to submit render command buffer")?;

    unsafe {
        logical_device.device.queue_submit2(
            logical_device.queue,
            std::slice::from_ref(&submit_2),
            vk::Fence::null(),
        )
    }
    .context("Failed to submit render present barrier")?;

    if let Some(ld) = devices.get_mut(&device_handle) {
        ld.timeline_next = signal_timeline_value.saturating_add(1);
    }

    if let Some(surface_state) = surfaces.get_mut(&surface_handle) {
        let fs = &mut surface_state.frame_sync[current_frame];
        fs.render_pass_submitted = true;
        fs.frame_timeline_value = Some(signal_timeline_value);
    }

    Ok(())
}

/// Present the rendered image to the screen.
pub(super) fn end_frame(
    state: &mut super::types::VulkanState,
    frame: crate::backend::FrameToken,
) -> Result<crate::timeline::TimelineValue> {
    let dh = state
        .surfaces
        .get(&frame.surface)
        .context("Invalid surface handle")?
        .device_handle;

    let pending = {
        let surf = state
            .surfaces
            .get_mut(&frame.surface)
            .context("Invalid surface handle")?;
        std::mem::take(&mut surf.frame_pending_gpu_commands)
    };

    if !pending.is_empty() {
        super::compute::submit(state, dh, &pending, Some(frame.surface))?;
    }

    present(state, frame.surface, frame.image)
}

/// Present the rendered image to the screen.
///
/// Unregisters the transient surface texture from the bindless descriptor set,
/// then queues the swapchain image for presentation.
pub(super) fn present(
    state: &mut super::types::VulkanState,
    surface_handle: SurfaceHandle,
    _image: SwapchainImageHandle,
) -> Result<crate::timeline::TimelineValue> {
    let _tz = crate::tracy_zone!("vk.surface.present");
    // Take the surface texture handle but do NOT unregister yet — the deferred
    // compute CBs (and the render CB in the graphics path) reference the
    // The swapchain image view + bindless descriptor are permanent (registered
    // at swapchain creation), so no deferred unregister is needed.  Just clear
    // the per-frame alias and proceed.
    let (
        image_index,
        current_frame,
        render_pass_submitted,
        device_handle,
        prep_cmd,
        compute_present_cmd,
        image_available_sem_present,
        work_done_sem_present,
        render_finished_sem_present,
        in_flight_fence_present,
        swapchain,
    ) = {
        let s = state
            .surfaces
            .get_mut(&surface_handle)
            .context("Invalid surface handle")?;
        s.current_texture_handle = None;
        let image_index = s
            .current_image_index
            .context("No image to present - call surface_render first")?;
        let cf = s.current_frame;
        let rp = s.frame_sync[cf].render_pass_submitted;
        let dh = s.device_handle;
        let fr = &s.frame_sync[cf];
        (
            image_index,
            cf,
            rp,
            dh,
            s.swapchain_prep_command_buffers[image_index as usize],
            s.swapchain_compute_present_command_buffers[image_index as usize],
            fr.image_available_semaphore,
            fr.work_done_semaphore,
            fr.render_finished_semaphore,
            fr.in_flight_fence,
            s.swapchain,
        )
    };

    if !render_pass_submitted {
        // Submit 1: prep barrier + all compute work.
        //   waits:   image_available_sem (GPU-side: WSI must release the image first)
        //   signals: in_flight_fence (CPU fence, fires as soon as compute is done),
        //            work_done_sem   (consumed by Submit 2),
        //            timeline_sem    (drives staging-buffer recycling)
        //
        // Submit 2: pre-recorded GENERAL → PRESENT_SRC_KHR barrier.
        //   waits:   work_done_sem
        //   signals: render_finished_sem (consumed by queue_present)
        //   fence:   none
        //
        // Splitting means in_flight_fence no longer covers the present barrier,
        // so the next frame's fence wait returns as soon as compute is done —
        // matching DX12's per-slot fence semantics.
        let deferred = {
            let s = state.surfaces.get_mut(&surface_handle).unwrap();
            std::mem::take(&mut s.frame_sync[current_frame].deferred_compute_cbs)
        };
        let pending_tex_upload_staging = {
            let s = state.surfaces.get_mut(&surface_handle).unwrap();
            std::mem::take(&mut s.frame_sync[current_frame].pending_compute_texture_staging)
        };
        let signal_timeline_value = {
            let ld = state
                .devices
                .get(&device_handle)
                .context("Surface's device is invalid")?;
            ld.timeline_next
        };

        let timeline_sem = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?
            .timeline_semaphore;

        // --- Submit 1 ---
        let mut cmd_infos_1: Vec<vk::CommandBufferSubmitInfo> =
            Vec::with_capacity(deferred.len() + 1);
        cmd_infos_1.push(vk::CommandBufferSubmitInfo::default().command_buffer(prep_cmd));
        cmd_infos_1.extend(
            deferred
                .iter()
                .copied()
                .map(|cb| vk::CommandBufferSubmitInfo::default().command_buffer(cb)),
        );

        let wait_acq = vk::SemaphoreSubmitInfo::default()
            .semaphore(image_available_sem_present)
            .value(0)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let sig_work_done = vk::SemaphoreSubmitInfo::default()
            .semaphore(work_done_sem_present)
            .value(0)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let sig_timeline = vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline_sem)
            .value(signal_timeline_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let signals_1 = [sig_work_done, sig_timeline];
        let submit_1 = vk::SubmitInfo2::default()
            .wait_semaphore_infos(std::slice::from_ref(&wait_acq))
            .command_buffer_infos(&cmd_infos_1)
            .signal_semaphore_infos(&signals_1);

        // --- Submit 2 ---
        let cmd_info_2 =
            vk::CommandBufferSubmitInfo::default().command_buffer(compute_present_cmd);
        let wait_work_done = vk::SemaphoreSubmitInfo::default()
            .semaphore(work_done_sem_present)
            .value(0)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let sig_render_finished = vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished_sem_present)
            .value(0)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let submit_2 = vk::SubmitInfo2::default()
            .wait_semaphore_infos(std::slice::from_ref(&wait_work_done))
            .command_buffer_infos(std::slice::from_ref(&cmd_info_2))
            .signal_semaphore_infos(std::slice::from_ref(&sig_render_finished));

        // vkQueueSubmit2's fence parameter fires after ALL batches in the call
        // complete, so we use TWO separate calls: the first carries the fence
        // (signals after compute), the second carries no fence (present barrier).
        let submit_ld = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;
        let r1 = {
            let _sz = crate::tracy_zone!("vk.present.queue_submit2_compute");
            unsafe {
                submit_ld.device.queue_submit2(
                    submit_ld.queue,
                    std::slice::from_ref(&submit_1),
                    in_flight_fence_present,
                )
            }
        };
        if let Err(e) = r1 {
            tracing::warn!(
                surface_handle,
                %device_handle,
                current_frame,
                image_index,
                result = ?e,
                "compute queue_submit2 (Submit 1) failed"
            );
            let s = state.surfaces.get_mut(&surface_handle).unwrap();
            s.frame_sync[current_frame].deferred_compute_cbs = deferred;
            s.frame_sync[current_frame].pending_compute_texture_staging =
                pending_tex_upload_staging;
            anyhow::bail!("Failed to submit compute work: {:?}", e);
        }
        let r2 = {
            let _sz = crate::tracy_zone!("vk.present.queue_submit2_barrier");
            unsafe {
                submit_ld.device.queue_submit2(
                    submit_ld.queue,
                    std::slice::from_ref(&submit_2),
                    vk::Fence::null(),
                )
            }
        };
        if let Err(e) = r2 {
            tracing::warn!(
                surface_handle,
                %device_handle,
                current_frame,
                image_index,
                result = ?e,
                "present-barrier queue_submit2 (Submit 2) failed"
            );
            anyhow::bail!("Failed to submit present barrier: {:?}", e);
        }

        {
            let _bk = crate::tracy_zone!("vk.present.post_submit");
            let ld_timeline = state.devices.get_mut(&device_handle).unwrap();
            ld_timeline.timeline_next = signal_timeline_value.saturating_add(1);

            if !deferred.is_empty() {
                state
                    .staging_belts
                    .entry(device_handle)
                    .or_insert_with(|| {
                        super::staging::StagingBelt::new(super::staging::DEFAULT_STAGING_CHUNK_SIZE)
                    })
                    .finish(signal_timeline_value);
            }
            if !pending_tex_upload_staging.is_empty() {
                state
                    .texture_staging_pools
                    .entry(device_handle)
                    .or_insert_with(super::staging::TextureStagingPool::new)
                    .release(signal_timeline_value, pending_tex_upload_staging);
            }

            for cb in &deferred {
                state
                    .timeline_cmd_buffers
                    .entry(signal_timeline_value)
                    .or_default()
                    .push((device_handle, *cb));
            }

            let surface_state_mut = state.surfaces.get_mut(&surface_handle).unwrap();
            surface_state_mut.frame_sync[current_frame].frame_timeline_value =
                Some(signal_timeline_value);
        }
    }

    let present_ld = state
        .devices
        .get(&device_handle)
        .context("Surface's device is invalid")?;
    let swapchain_loader = khr::swapchain::Device::new(&state.instance, &present_ld.device);

    let swapchains = [swapchain];
    let image_indices = [image_index];
    let wait_semaphores = [render_finished_sem_present];

    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&wait_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    let result = {
        let _pz = crate::tracy_zone!("vk.present.queue_present");
        unsafe { swapchain_loader.queue_present(present_ld.queue, &present_info) }
    };
    // `queue_present` returns Ok on SUCCESS and SUBOPTIMAL_KHR; Err on real failures.
    let _mark_image_presented = result.is_ok();
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

    // Clear the current image and advance frame counter.
    let surface_state = state.surfaces.get_mut(&surface_handle).unwrap();
    surface_state.current_image_index = None;
    surface_state.current_frame = (surface_state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

    // Handle suboptimal or out of date
    match result {
        Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            let timeline_value = {
                let surface_state = state
                    .surfaces
                    .get_mut(&surface_handle)
                    .context("Invalid surface handle")?;
                surface_state.frame_sync[current_frame]
                    .frame_timeline_value
                    .take()
                    .context("present: frame timeline value missing (internal error)")?
            };
            Ok(timeline_value)
        }
        Err(e) => Err(anyhow::anyhow!("Failed to present: {:?}", e)),
    }
}

/// Resize the surface's swapchain.
#[allow(clippy::too_many_arguments)]
pub(super) fn resize(
    entry: &Entry,
    instance: &Instance,
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    next_texture_handle: &mut TextureHandle,
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

    // Get new capabilities early so we can bail out if nothing changed.
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

    // Skip the expensive recreation when the clamped extent already matches
    // the current swapchain AND the present mode hasn't changed.  Winit can
    // fire multiple Resized events during window creation that would
    // otherwise cause redundant swapchain teardown/rebuild cycles.
    //
    // `present_mode_dirty` is set by `set_present_mode` so a mode change
    // always triggers recreation even when the window dimensions are unchanged.
    {
        let surface_state = surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        if surface_state.width == extent.width
            && surface_state.height == extent.height
            && !surface_state.present_mode_dirty
        {
            return Ok(());
        }
    }

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

    // Unregister old per-image bindless textures and free old per-image barrier
    // CBs before the swapchain they reference is destroyed.
    {
        let old_tex_handles = surfaces
            .get_mut(&surface_handle)
            .map(|s| {
                s.current_texture_handle = None;
                for cbs in [
                    std::mem::take(&mut s.swapchain_prep_command_buffers),
                    std::mem::take(&mut s.swapchain_compute_present_command_buffers),
                    std::mem::take(&mut s.swapchain_render_present_command_buffers),
                ] {
                    if !cbs.is_empty() {
                        unsafe {
                            logical_device
                                .device
                                .free_command_buffers(logical_device.command_pool, &cbs);
                        }
                    }
                }
                std::mem::take(&mut s.swapchain_texture_handles)
            })
            .unwrap_or_default();
        for th in old_tex_handles {
            unregister_swapchain_texture(devices, textures, th);
        }
    }

    let logical_device = devices
        .get(&device_handle)
        .context("Surface's device is invalid")?;

    // Destroy old image views
    if let Some(surface_state) = surfaces.get(&surface_handle) {
        for view in &surface_state.swapchain_image_views {
            unsafe { logical_device.device.destroy_image_view(*view, None) };
        }
    }

    let image_count = (capabilities.min_image_count + 1)
        .max(MAX_FRAMES_IN_FLIGHT as u32 + 1)
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
        .image_usage(
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
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

    // Pre-record per-image barrier CBs and re-register bindless textures for the new images.
    let (new_prep_cbs, new_compute_present_cbs, new_render_present_cbs) = {
        let logical_device = devices.get(&device_handle).context("Device invalid")?;
        (
            alloc_and_record_prep_cbs(logical_device, &swapchain_images)?,
            alloc_and_record_compute_present_cbs(logical_device, &swapchain_images)?,
            alloc_and_record_render_present_cbs(logical_device, &swapchain_images)?,
        )
    };

    let goldy_format =
        super::utils::vk_to_format(format).unwrap_or(TextureFormat::Bgra8UnormSrgb);
    let mut new_texture_handles = Vec::with_capacity(swapchain_images.len());
    for &image in &swapchain_images {
        let th = register_surface_texture(
            devices,
            textures,
            next_texture_handle,
            device_handle,
            image,
            format,
            goldy_format,
            extent.width,
            extent.height,
        )?;
        new_texture_handles.push(th);
    }

    // Update surface state — reset frame counter since we waited for idle.
    if let Some(surface_state) = surfaces.get_mut(&surface_handle) {
        surface_state.swapchain = new_swapchain;
        surface_state.swapchain_images = swapchain_images;
        surface_state.swapchain_image_views = swapchain_image_views;
        surface_state.swapchain_prep_command_buffers = new_prep_cbs;
        surface_state.swapchain_compute_present_command_buffers = new_compute_present_cbs;
        surface_state.swapchain_render_present_command_buffers = new_render_present_cbs;
        surface_state.swapchain_texture_handles = new_texture_handles;
        surface_state.width = extent.width;
        surface_state.height = extent.height;
        surface_state.current_frame = 0;
        surface_state.current_image_index = None;
        surface_state.current_texture_handle = None;
        surface_state.depth_image = new_depth_image;
        surface_state.depth_memory = new_depth_memory;
        surface_state.depth_view = new_depth_view;
        surface_state.present_mode_dirty = false;
    }

    tracing::debug!(
        width = extent.width,
        height = extent.height,
        present_mode = ?stored_present_mode,
        "Resized surface"
    );

    Ok(())
}

/// Set swapchain present mode (vsync). Recreates the swapchain when the mode changes.
pub(super) fn set_present_mode(
    entry: &Entry,
    instance: &Instance,
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    surfaces: &mut HashMap<SurfaceHandle, SurfaceState>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    next_texture_handle: &mut TextureHandle,
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
        surface_state.present_mode_dirty = true;
    }

    resize(
        entry,
        instance,
        devices,
        surfaces,
        textures,
        next_texture_handle,
        surface_handle,
        w,
        h,
    )
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
// Swapchain texture registration helpers
// ---------------------------------------------------------------------------
// Swapchain images are registered once at creation/resize and persist until the
// swapchain is recreated or destroyed.  `current_texture_handle` is a per-frame
// alias into `swapchain_texture_handles`; it is never freed directly.
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
            sampled_bindless_index: None,
            current_layout: vk::ImageLayout::GENERAL,
            transient_heap_suballoc: false,
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

/// Unregister a swapchain image texture (destroy view + bindless slot).
/// Does NOT destroy the VkImage — it is owned by the swapchain.
fn unregister_swapchain_texture(
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    tex_handle: TextureHandle,
) {
    if let Some(tex_state) = textures.remove(&tex_handle) {
        if let Some(device) = devices.get_mut(&tex_state.device_handle) {
            device.resource_registry.unregister_texture(tex_handle);
            unsafe {
                device.device.destroy_image_view(tex_state.view, None);
            }
        }
        tracing::debug!("Unregistered swapchain texture {}", tex_handle);
    }
}

/// Kept for compatibility with the `destroy()` path which unregisters via the
/// same name used before the rename.  Delegates to `unregister_swapchain_texture`.
#[inline(always)]
fn unregister_surface_texture(
    devices: &mut HashMap<DeviceHandle, LogicalDevice>,
    textures: &mut HashMap<TextureHandle, TextureState>,
    tex_handle: TextureHandle,
) {
    unregister_swapchain_texture(devices, textures, tex_handle);
}

/// Allocate one primary command buffer per swapchain image and pre-record a
/// reusable `UNDEFINED → GENERAL` barrier for each.  Always using `UNDEFINED`
/// as `old_layout` lets the driver discard stale contents, which is correct
/// since every frame overwrites the entire image.  The CBs are submitted as the
/// first entry of each frame's `vkQueueSubmit2`, waiting on the acquire
/// semaphore, so the images are only accessed after WSI has released them.
fn alloc_and_record_prep_cbs(
    logical_device: &LogicalDevice,
    swapchain_images: &[vk::Image],
) -> Result<Vec<vk::CommandBuffer>> {
    alloc_and_record_present_cbs(
        logical_device,
        swapchain_images,
        vk::PipelineStageFlags2::TOP_OF_PIPE,
        vk::AccessFlags2::NONE,
        vk::ImageLayout::UNDEFINED,
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::SHADER_WRITE
            | vk::AccessFlags2::SHADER_READ
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::TRANSFER_WRITE,
        vk::ImageLayout::GENERAL,
    )
}

/// Pre-record `GENERAL → PRESENT_SRC_KHR` barriers (one per swapchain image).
/// Used as Submit 2 in the compute present path.
fn alloc_and_record_compute_present_cbs(
    logical_device: &LogicalDevice,
    swapchain_images: &[vk::Image],
) -> Result<Vec<vk::CommandBuffer>> {
    alloc_and_record_present_cbs(
        logical_device,
        swapchain_images,
        vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
        vk::ImageLayout::GENERAL,
        vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        vk::AccessFlags2::NONE,
        vk::ImageLayout::PRESENT_SRC_KHR,
    )
}

/// Pre-record `COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR` barriers (one per swapchain
/// image). Used as Submit 2 in the graphics (render) present path.
fn alloc_and_record_render_present_cbs(
    logical_device: &LogicalDevice,
    swapchain_images: &[vk::Image],
) -> Result<Vec<vk::CommandBuffer>> {
    alloc_and_record_present_cbs(
        logical_device,
        swapchain_images,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        vk::AccessFlags2::NONE,
        vk::ImageLayout::PRESENT_SRC_KHR,
    )
}

/// Generic helper: allocate one reusable CB per image and record a single
/// image-memory barrier with the given parameters.
#[allow(clippy::too_many_arguments)]
fn alloc_and_record_present_cbs(
    logical_device: &LogicalDevice,
    swapchain_images: &[vk::Image],
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    old_layout: vk::ImageLayout,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
    new_layout: vk::ImageLayout,
) -> Result<Vec<vk::CommandBuffer>> {
    let count = swapchain_images.len() as u32;
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(logical_device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count);
    let cbs = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate barrier command buffers")?;

    // No ONE_TIME_SUBMIT — these CBs are submitted multiple times (once per frame).
    let begin_info = vk::CommandBufferBeginInfo::default();
    for (&cb, &image) in cbs.iter().zip(swapchain_images.iter()) {
        unsafe { logical_device.device.begin_command_buffer(cb, &begin_info) }
            .context("Failed to begin barrier command buffer")?;

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
        unsafe { logical_device.device.cmd_pipeline_barrier2(cb, &dep_info) };

        unsafe { logical_device.device.end_command_buffer(cb) }
            .context("Failed to end barrier command buffer")?;
    }

    Ok(cbs)
}
