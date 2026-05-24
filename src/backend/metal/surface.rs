//! Surface (window presentation) management logic.
//!
//! The acquire/render/present cycle is decoupled:
//! - `acquire()` calls `nextDrawable` early and returns a persistent compute scratch texture
//! - compute work writes to the scratch texture through normal bindless storage-image slots
//! - `present()` blits the scratch texture into the retained drawable and schedules presentation
//!
//! ## Deprecation note
//! This file uses `cocoa::base::id`, `NSRect`, and related types from the
//! `cocoa`/`objc` 0.2.x ecosystem, which are deprecated in favour of the
//! `objc2` crate. Migration is deferred until the `metal` and `cocoa` crates
//! offer stable `objc2`-compatible bindings for `CAMetalLayer`.
#![allow(deprecated)]
//! - `render()` uses the already-acquired drawable (does NOT call `nextDrawable` again)
//! - `present()` performs the final scratch-to-drawable blit and presents the drawable

use super::super::{
    DeviceHandle, FrameToken, RenderCommand, SurfaceHandle, SwapchainImageHandle, TextureHandle,
};
use super::compute;
use super::render_commands::{create_render_pass, record};
use super::types::{MetalState, SurfaceState, MAX_FRAMES_IN_FLIGHT};
use super::utils::depth_format_to_mtl;
use crate::types::{DepthFormat, PresentMode, SpatialAccess, TextureFlags, TextureFormat};
use anyhow::{Context, Result};
use cocoa::base::{id, nil, NO, YES};
use core_graphics_types::geometry::CGSize;
use foreign_types::ForeignTypeRef;
use metal as mtl;
use mtl::{MTLPixelFormat, MTLStorageMode, MTLTextureUsage, TextureDescriptor};
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};

/// Create a surface for window presentation.
/// When `depth_format` is `Some`, a depth buffer is created for 3D rendering.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    window: &dyn HasWindowHandle,
    _display: &dyn HasDisplayHandle,
    depth_format: Option<DepthFormat>,
) -> Result<SurfaceHandle> {
    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    let window_handle = window
        .window_handle()
        .map_err(|e| anyhow::anyhow!("Failed to get window handle: {:?}", e))?;

    let ns_view = match window_handle.as_raw() {
        RawWindowHandle::AppKit(handle) => handle.ns_view.as_ptr() as id,
        _ => anyhow::bail!("Expected AppKit window handle on macOS"),
    };

    let (layer, width, height) = unsafe {
        let layer: id = msg_send![class!(CAMetalLayer), layer];
        let () = msg_send![layer, setDevice: logical_device.device.as_ptr()];
        // Keep the drawable format in lockstep with the compute scratch texture
        // so the final blit is a byte-preserving texture copy.
        let () = msg_send![layer, setPixelFormat: MTLPixelFormat::RGBA8Unorm];
        // Blitting into the drawable requires non-framebuffer-only textures.
        let () = msg_send![layer, setFramebufferOnly: NO];

        let () = msg_send![ns_view, setWantsLayer: YES];
        let () = msg_send![ns_view, setLayer: layer];

        let frame: cocoa::foundation::NSRect = msg_send![ns_view, frame];
        let size = CGSize::new(frame.size.width, frame.size.height);
        let () = msg_send![layer, setDrawableSize: size];

        let w = (frame.size.width as u32).max(1);
        let h = (frame.size.height as u32).max(1);

        (layer, w, h)
    };

    let depth_texture = depth_format.map(|df| {
        let depth_desc = TextureDescriptor::new();
        depth_desc.set_width(width as u64);
        depth_desc.set_height(height as u64);
        depth_desc.set_pixel_format(depth_format_to_mtl(df));
        depth_desc.set_usage(MTLTextureUsage::RenderTarget);
        depth_desc.set_storage_mode(MTLStorageMode::Private);
        logical_device.device.new_texture(&depth_desc)
    });

    let handle = state.next_surface_handle;
    state.next_surface_handle += 1;

    state.surfaces.insert(
        handle,
        SurfaceState {
            device_handle,
            width,
            height,
            format: TextureFormat::Rgba8Unorm,
            depth_format,
            depth_texture,
            current_frame: 0,
            layer: layer as *mut std::ffi::c_void,
            current_drawable: None,
            current_texture_handle: None,
            compute_scratch_textures: [None; MAX_FRAMES_IN_FLIGHT],
            present_mode: PresentMode::Auto,
            frame_pending_gpu_commands: Vec::new(),
        },
    );
    tracing::info!("Created Metal surface {}", handle);
    Ok(handle)
}

/// Destroy a surface.
pub(super) fn destroy(state: &mut MetalState, surface: SurfaceHandle) {
    if let Some(surface_state) = state.surfaces.remove(&surface) {
        if let Some(drawable_ptr) = surface_state.current_drawable {
            unsafe {
                let (): () = msg_send![drawable_ptr as id, release];
            }
        }
        for tex_handle in surface_state.compute_scratch_textures.into_iter().flatten() {
            super::texture::destroy(state, tex_handle);
        }
    }
}

/// Acquire the next swapchain image.
///
/// Calls `nextDrawable` on the CAMetalLayer and returns a heap-backed scratch
/// texture for compute work. The drawable is retained until `present()`, where
/// the scratch texture is blitted into it.
pub(super) fn acquire(
    state: &mut MetalState,
    surface: SurfaceHandle,
) -> Result<SwapchainImageHandle> {
    let (device_handle, frame_idx, width, height, format) = {
        let surface_state = state
            .surfaces
            .get_mut(&surface)
            .context("Invalid surface handle")?;

        if let Some(drawable_ptr) = surface_state.current_drawable.take() {
            tracing::warn!("Previous drawable was not presented; releasing it before acquire");
            unsafe {
                let (): () = msg_send![drawable_ptr as id, release];
            }
        }
        surface_state.current_texture_handle = None;

        let layer = surface_state.layer as id;
        let size: CGSize = unsafe { msg_send![layer, drawableSize] };
        surface_state.width = (size.width as u32).max(1);
        surface_state.height = (size.height as u32).max(1);
        surface_state.current_frame = (surface_state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        let drawable: id = unsafe { msg_send![layer, nextDrawable] };
        if drawable == nil {
            anyhow::bail!("Failed to get next drawable from CAMetalLayer");
        }
        unsafe {
            let () = msg_send![drawable, retain];
        }
        surface_state.current_drawable = Some(drawable as *mut std::ffi::c_void);

        (
            surface_state.device_handle,
            surface_state.current_frame,
            surface_state.width,
            surface_state.height,
            surface_state.format,
        )
    };

    let tex_handle = ensure_compute_scratch_texture(
        state,
        surface,
        frame_idx,
        width,
        height,
        format,
        device_handle,
    )?;

    let surface_state = state
        .surfaces
        .get_mut(&surface)
        .expect("surface must be registered before acquiring a frame");
    surface_state.current_texture_handle = Some(tex_handle);

    if let Some(ld) = state.devices.get_mut(&device_handle) {
        ld.process_deletion_queue_up_to_signaled();
    }

    Ok(frame_idx as u64)
}

/// Get the texture handle for the currently acquired surface frame.
pub(super) fn frame_texture(state: &MetalState, surface: SurfaceHandle) -> Option<TextureHandle> {
    state
        .surfaces
        .get(&surface)
        .and_then(|s| s.current_texture_handle)
}

/// Render commands to the swapchain using the already-acquired drawable.
pub(super) fn render(
    state: &mut MetalState,
    surface: SurfaceHandle,
    _image: SwapchainImageHandle,
    commands: &[RenderCommand],
) -> Result<()> {
    let surface_state = state
        .surfaces
        .get(&surface)
        .context("Invalid surface handle")?;

    let drawable_ptr = surface_state
        .current_drawable
        .context("No drawable acquired — call surface_acquire first")?;
    let drawable = drawable_ptr as id;

    let logical_device = state
        .devices
        .get(&surface_state.device_handle)
        .context("Device no longer valid")?;

    let texture_ptr: *mut Object = unsafe { msg_send![drawable, texture] };
    let texture: &mtl::TextureRef = unsafe { &*(texture_ptr as *const mtl::TextureRef) };

    let mut clear_color = None;
    let mut clear_depth = None;
    for cmd in commands {
        match cmd {
            RenderCommand::Clear(color) => clear_color = Some(*color),
            RenderCommand::ClearDepth(depth) => clear_depth = Some(*depth),
            _ => {}
        }
    }
    let render_pass = create_render_pass(
        texture,
        surface_state.depth_texture.as_deref(),
        clear_color,
        clear_depth,
    );

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let encoder = command_buffer.new_render_command_encoder(render_pass);

    let render_stages = mtl::MTLRenderStages::Vertex | mtl::MTLRenderStages::Fragment;
    logical_device
        .heap_allocator
        .use_heaps_for_render(encoder, render_stages);
    logical_device
        .texture_heap
        .use_heaps_for_render(encoder, render_stages);
    for buf_state in state.buffers.values() {
        if buf_state.device_handle == surface_state.device_handle {
            encoder.use_resource_at(
                &buf_state.buffer,
                mtl::MTLResourceUsage::Read | mtl::MTLResourceUsage::Write,
                render_stages,
            );
        }
    }

    encoder.set_vertex_buffer(0, Some(&logical_device.argument_buffer), 0);
    encoder.set_fragment_buffer(0, Some(&logical_device.argument_buffer), 0);
    tracing::trace!("Bound global argument buffer at slot 0");

    encoder.set_viewport(mtl::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: surface_state.width as f64,
        height: surface_state.height as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    encoder.set_scissor_rect(mtl::MTLScissorRect {
        x: 0,
        y: 0,
        width: surface_state.width as u64,
        height: surface_state.height as u64,
    });

    record(encoder, commands, &state.pipelines, &state.buffers)?;

    encoder.end_encoding();
    command_buffer.commit();

    Ok(())
}

/// Present the acquired drawable and unregister its temporary texture.
///
/// This is the sole place where `presentDrawable:` is called. A lightweight
/// command buffer is created to schedule the presentation. If `render()` was
/// called first, Metal's serial queue ordering guarantees the render commands
/// complete before the present is scheduled by the GPU.
pub(super) fn present(
    state: &mut MetalState,
    surface: SurfaceHandle,
    _image: SwapchainImageHandle,
    copy_scratch_to_drawable: bool,
) -> Result<crate::timeline::TimelineValue> {
    let surface_state = state
        .surfaces
        .get(&surface)
        .context("Invalid surface handle")?;

    let device_handle = surface_state.device_handle;
    let scratch_handle = surface_state.current_texture_handle;

    let drawable_ptr = match surface_state.current_drawable {
        Some(d) => d,
        None => {
            let ld = state
                .devices
                .get_mut(&device_handle)
                .context("Device no longer valid")?;
            ld.process_deletion_queue_up_to_signaled();
            return Ok(ld.timeline_event.as_ref().signaled_value());
        }
    };
    let scratch_handle = if copy_scratch_to_drawable {
        Some(scratch_handle.context("No acquired compute scratch texture")?)
    } else {
        None
    };

    let signal_value = {
        let ld = state
            .devices
            .get_mut(&device_handle)
            .context("Device no longer valid")?;
        let v = ld.timeline_next;
        ld.timeline_next += 1;
        ld.timeline_scheduled_max = ld.timeline_scheduled_max.max(v);
        v
    };

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Device no longer valid")?;

    let command_buffer = logical_device.command_queue.new_command_buffer();

    let drawable = drawable_ptr as id;
    let drawable_texture = if copy_scratch_to_drawable {
        let texture_ptr: *mut Object = unsafe { msg_send![drawable, texture] };
        Some(unsafe { &*(texture_ptr as *const mtl::TextureRef) })
    } else {
        None
    };

    if let (Some(scratch_handle), Some(drawable_texture)) = (scratch_handle, drawable_texture) {
        let scratch_state = state
            .textures
            .get(&scratch_handle)
            .context("Compute scratch texture not found")?;
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_texture(
            &scratch_state.texture,
            0,
            0,
            mtl::MTLOrigin { x: 0, y: 0, z: 0 },
            mtl::MTLSize {
                width: scratch_state.width as u64,
                height: scratch_state.height as u64,
                depth: 1,
            },
            drawable_texture,
            0,
            0,
            mtl::MTLOrigin { x: 0, y: 0, z: 0 },
        );
        blit.end_encoding();
    }

    command_buffer.encode_signal_event(logical_device.timeline_event.as_ref(), signal_value);

    let waiter = logical_device.timeline_waiter.clone();
    let handler = block::ConcreteBlock::new(move |_cb: &mtl::CommandBufferRef| {
        waiter.signal(signal_value);
    })
    .copy();
    command_buffer.add_completed_handler(&handler);

    let drawable_ref: &mtl::DrawableRef = unsafe { &*(drawable as *const mtl::DrawableRef) };
    command_buffer.present_drawable(drawable_ref);
    command_buffer.commit();

    // Release the retained drawable
    unsafe {
        let (): () = msg_send![drawable, release];
    }

    // Clear the drawable state
    let surface_state = state
        .surfaces
        .get_mut(&surface)
        .expect("surface must be registered before presenting a frame");
    surface_state.current_drawable = None;
    surface_state.current_texture_handle = None;

    if let Some(ld) = state.devices.get_mut(&device_handle) {
        ld.process_deletion_queue_up_to_signaled();
    }

    Ok(signal_value)
}
pub(super) fn end_frame(
    state: &mut MetalState,
    frame: FrameToken,
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

    let copy_scratch_to_drawable = !pending.is_empty();
    if copy_scratch_to_drawable {
        compute::submit(state, dh, &pending)?;
    }

    present(state, frame.surface, frame.image, copy_scratch_to_drawable)
}

/// Set the present mode on the CAMetalLayer.
pub(super) fn set_present_mode(
    state: &mut MetalState,
    surface: SurfaceHandle,
    mode: PresentMode,
) -> Result<()> {
    let surface_state = state
        .surfaces
        .get_mut(&surface)
        .context("Invalid surface handle")?;

    let layer = surface_state.layer as id;
    let sync_enabled = match mode {
        PresentMode::Immediate => false,
        PresentMode::Fifo | PresentMode::Mailbox | PresentMode::Auto => true,
    };
    unsafe {
        let () = msg_send![layer, setDisplaySyncEnabled: sync_enabled];
    }
    surface_state.present_mode = mode;
    tracing::debug!(
        "Set surface {} present mode to {:?} (displaySyncEnabled={})",
        surface,
        mode,
        sync_enabled
    );
    Ok(())
}

/// Get the current present mode.
pub(super) fn present_mode(state: &MetalState, surface: SurfaceHandle) -> PresentMode {
    state
        .surfaces
        .get(&surface)
        .map(|s| s.present_mode)
        .unwrap_or(PresentMode::Auto)
}

/// Resize the surface.
pub(super) fn resize(
    state: &mut MetalState,
    surface: SurfaceHandle,
    width: u32,
    height: u32,
) -> Result<()> {
    let (device_handle, depth_format, layer, old_scratch) = {
        let surface_state = state
            .surfaces
            .get_mut(&surface)
            .context("Invalid surface handle")?;

        surface_state.width = width;
        surface_state.height = height;
        surface_state.current_texture_handle = None;
        let old_scratch = std::mem::replace(
            &mut surface_state.compute_scratch_textures,
            [None; MAX_FRAMES_IN_FLIGHT],
        );
        (
            surface_state.device_handle,
            surface_state.depth_format,
            surface_state.layer as id,
            old_scratch,
        )
    };

    for tex_handle in old_scratch.into_iter().flatten() {
        super::texture::destroy(state, tex_handle);
    }

    // Recreate depth texture if present
    if let Some(df) = depth_format {
        let depth_texture = {
            let logical_device = state
                .devices
                .get(&device_handle)
                .context("Device no longer valid")?;

            let w = width.max(1);
            let h = height.max(1);
            let depth_desc = TextureDescriptor::new();
            depth_desc.set_width(w as u64);
            depth_desc.set_height(h as u64);
            depth_desc.set_pixel_format(depth_format_to_mtl(df));
            depth_desc.set_usage(MTLTextureUsage::RenderTarget);
            depth_desc.set_storage_mode(MTLStorageMode::Private);
            logical_device.device.new_texture(&depth_desc)
        };
        let surface_state = state
            .surfaces
            .get_mut(&surface)
            .context("Invalid surface handle")?;
        surface_state.depth_texture = Some(depth_texture);
    }

    let size = CGSize::new(width as f64, height as f64);
    unsafe {
        let () = msg_send![layer, setDrawableSize: size];
    }

    tracing::debug!("Resized surface {} to {}x{}", surface, width, height);
    Ok(())
}

/// Get surface dimensions.
pub(super) fn size(state: &MetalState, surface: SurfaceHandle) -> (u32, u32) {
    state
        .surfaces
        .get(&surface)
        .map(|s| (s.width, s.height))
        .unwrap_or((0, 0))
}

/// Get surface format.
pub(super) fn format(state: &MetalState, surface: SurfaceHandle) -> TextureFormat {
    state
        .surfaces
        .get(&surface)
        .map(|s| s.format)
        .unwrap_or(TextureFormat::Rgba8Unorm)
}

fn ensure_compute_scratch_texture(
    state: &mut MetalState,
    surface: SurfaceHandle,
    idx: usize,
    width: u32,
    height: u32,
    format: TextureFormat,
    device_handle: DeviceHandle,
) -> Result<TextureHandle> {
    let stale = {
        let surface_state = state
            .surfaces
            .get(&surface)
            .context("Invalid surface handle")?;
        let slot = surface_state.compute_scratch_textures[idx];
        if let Some(handle) = slot {
            if let Some(tex) = state.textures.get(&handle) {
                if tex.width == width && tex.height == height && tex.format == format {
                    return Ok(handle);
                }
            }
            Some(handle)
        } else {
            None
        }
    };

    if let Some(old) = stale {
        super::texture::destroy(state, old);
        let surface_state = state
            .surfaces
            .get_mut(&surface)
            .context("Invalid surface handle")?;
        surface_state.compute_scratch_textures[idx] = None;
    }

    let handle = super::texture::create(
        state,
        device_handle,
        width,
        height,
        format,
        SpatialAccess::Direct,
        TextureFlags::COPY_SRC,
    );
    let handle = handle.context("failed to create Metal surface compute scratch texture")?;
    let surface_state = state
        .surfaces
        .get_mut(&surface)
        .context("Invalid surface handle")?;
    surface_state.compute_scratch_textures[idx] = Some(handle);

    tracing::debug!(
        "Prepared Metal surface scratch texture {} ({}x{}, frame_slot={})",
        handle,
        width,
        height,
        idx,
    );

    Ok(handle)
}
