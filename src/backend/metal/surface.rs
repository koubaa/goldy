//! Surface (window presentation) management logic.
//!
//! The acquire/render/present cycle is decoupled:
//! - `acquire()` calls `nextDrawable` and registers the drawable's texture for bindless access
//! - `frame_texture()` returns the registered texture handle
//! - `render()` uses the already-acquired drawable (does NOT call `nextDrawable` again)
//! - `present()` presents the drawable and unregisters the temporary texture

use super::super::{
    DeviceHandle, RenderCommand, SurfaceHandle, SwapchainImageHandle, TextureHandle,
};
use super::render_commands::{create_render_pass, record};
use super::types::{
    MetalState, ResourceRegistry, SurfaceState, TextureState, ARGUMENT_BUFFER_SIZE,
    MAX_FRAMES_IN_FLIGHT,
};
use super::utils::depth_format_to_mtl;
use crate::types::{DepthFormat, PresentMode, TextureFormat};
use ::metal as mtl;
use anyhow::{Context, Result};
use cocoa::base::{id, nil, NO, YES};
use core_graphics_types::geometry::CGSize;
use foreign_types::{ForeignType, ForeignTypeRef};
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
        .get(&device_handle)
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
        let () = msg_send![layer, setPixelFormat: MTLPixelFormat::BGRA8Unorm];
        // Don't set framebufferOnly so the texture can be used with compute
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
            format: TextureFormat::Bgra8Unorm,
            depth_format,
            depth_texture,
            current_frame: 0,
            layer: layer as *mut std::ffi::c_void,
            current_drawable: None,
            current_texture_handle: None,
            present_mode: PresentMode::Auto,
        },
    );
    tracing::info!("Created Metal surface {}", handle);
    Ok(handle)
}

/// Destroy a surface.
pub(super) fn destroy(state: &mut MetalState, surface: SurfaceHandle) {
    // Clean up any acquired drawable texture before removing the surface
    if let Some(surface_state) = state.surfaces.get(&surface) {
        if let Some(tex_handle) = surface_state.current_texture_handle {
            unregister_surface_texture(state, tex_handle);
        }
    }
    state.surfaces.remove(&surface);
}

/// Acquire the next swapchain image.
///
/// Calls `nextDrawable` on the CAMetalLayer and registers the drawable's
/// texture in the bindless descriptor set. The texture handle is available
/// via `frame_texture()` until `present()` is called.
pub(super) fn acquire(
    state: &mut MetalState,
    surface: SurfaceHandle,
) -> Result<SwapchainImageHandle> {
    // Clean up any previously acquired drawable that wasn't presented
    if let Some(surface_state) = state.surfaces.get(&surface) {
        if let Some(tex_handle) = surface_state.current_texture_handle {
            tracing::warn!("Previous drawable was not presented; cleaning up");
            unregister_surface_texture(state, tex_handle);
        }
    }

    let surface_state = state
        .surfaces
        .get_mut(&surface)
        .context("Invalid surface handle")?;

    let layer = surface_state.layer as id;

    let size: CGSize = unsafe { msg_send![layer, drawableSize] };
    surface_state.width = size.width as u32;
    surface_state.height = size.height as u32;

    surface_state.current_frame = (surface_state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

    // Get the next drawable from the layer
    let drawable: id = unsafe { msg_send![layer, nextDrawable] };
    if drawable == nil {
        anyhow::bail!("Failed to get next drawable from CAMetalLayer");
    }
    unsafe {
        let () = msg_send![drawable, retain];
    }
    surface_state.current_drawable = Some(drawable as *mut std::ffi::c_void);

    // Get the drawable's texture and register it for bindless access
    let texture_ptr: *mut Object = unsafe { msg_send![drawable, texture] };
    let texture: &mtl::TextureRef = unsafe { &*(texture_ptr as *const mtl::TextureRef) };

    let device_handle = surface_state.device_handle;
    let width = surface_state.width;
    let height = surface_state.height;
    let format = surface_state.format;

    let tex_handle =
        register_surface_texture(state, device_handle, texture, width, height, format)?;

    let surface_state = state.surfaces.get_mut(&surface).unwrap();
    surface_state.current_texture_handle = Some(tex_handle);

    Ok(surface_state.current_frame as u64)
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

    record(encoder, commands, &state.pipelines, &state.buffers);

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
) -> Result<()> {
    let surface_state = state
        .surfaces
        .get(&surface)
        .context("Invalid surface handle")?;

    let drawable_ptr = match surface_state.current_drawable {
        Some(d) => d,
        None => {
            return Ok(());
        }
    };
    let tex_handle = surface_state.current_texture_handle;
    let device_handle = surface_state.device_handle;

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Device no longer valid")?;

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let drawable = drawable_ptr as id;
    let _: () = unsafe { msg_send![command_buffer.as_ptr(), presentDrawable: drawable] };
    command_buffer.commit();

    // Release the retained drawable
    unsafe {
        let () = msg_send![drawable, release];
    }

    // Unregister the temporary surface texture
    if let Some(th) = tex_handle {
        unregister_surface_texture(state, th);
    }

    // Clear the drawable state
    let surface_state = state.surfaces.get_mut(&surface).unwrap();
    surface_state.current_drawable = None;
    surface_state.current_texture_handle = None;

    Ok(())
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
    let surface_state = state
        .surfaces
        .get_mut(&surface)
        .context("Invalid surface handle")?;

    surface_state.width = width;
    surface_state.height = height;

    // Recreate depth texture if present
    if let Some(df) = surface_state.depth_format {
        let logical_device = state
            .devices
            .get(&surface_state.device_handle)
            .context("Device no longer valid")?;

        let w = width.max(1);
        let h = height.max(1);
        let depth_desc = TextureDescriptor::new();
        depth_desc.set_width(w as u64);
        depth_desc.set_height(h as u64);
        depth_desc.set_pixel_format(depth_format_to_mtl(df));
        depth_desc.set_usage(MTLTextureUsage::RenderTarget);
        depth_desc.set_storage_mode(MTLStorageMode::Private);
        surface_state.depth_texture = Some(logical_device.device.new_texture(&depth_desc));
    }

    let layer = surface_state.layer as id;
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
        .unwrap_or(TextureFormat::Bgra8Unorm)
}

// ---------------------------------------------------------------------------
// Internal helpers for transient surface texture management
// ---------------------------------------------------------------------------

/// Register a drawable's MTLTexture in the texture system for bindless access.
///
/// The texture is stored as a `TextureState` with a unique handle so it can be
/// referenced via `texture_bindless_index` just like any other texture. It is
/// registered as a storage image (Direct spatial access) since compute shaders
/// need write access.
fn register_surface_texture(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    texture: &mtl::TextureRef,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Result<TextureHandle> {
    let handle = state.next_texture_handle;
    state.next_texture_handle += 1;

    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Device no longer valid")?;

    // Register as a storage image so compute shaders can write to it
    let local_idx = logical_device
        .resource_registry
        .register_storage_image(handle);
    let global_idx = ResourceRegistry::storage_image_global_index(local_idx);

    // Encode the texture into the argument buffer
    let encoded_length = logical_device.texture_encoder.encoded_length();
    let offset = (global_idx as u64) * encoded_length;
    if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
        logical_device
            .texture_encoder
            .set_argument_buffer(&logical_device.argument_buffer, offset);
        logical_device.texture_encoder.set_texture(0, texture);
        tracing::trace!(
            "Encoded surface texture {} at arg buffer offset {} (global slot {})",
            handle,
            offset,
            global_idx
        );
    }

    // Retain the MTLTexture so it stays valid while we hold a reference
    let texture_obj: &mtl::Texture =
        unsafe { &*(texture as *const mtl::TextureRef as *const mtl::Texture) };

    state.textures.insert(
        handle,
        TextureState {
            device_handle,
            width,
            height,
            format,
            texture: texture_obj.clone(),
            arg_buffer_index: local_idx,
        },
    );

    tracing::debug!(
        "Registered surface texture {} ({}x{}, bindless local={} global={})",
        handle,
        width,
        height,
        local_idx,
        global_idx,
    );

    Ok(handle)
}

/// Unregister a transient surface texture from the texture system.
fn unregister_surface_texture(state: &mut MetalState, tex_handle: TextureHandle) {
    if let Some(tex_state) = state.textures.remove(&tex_handle) {
        if let Some(device) = state.devices.get_mut(&tex_state.device_handle) {
            device.resource_registry.unregister_texture(tex_handle);
        }
        tracing::debug!("Unregistered surface texture {}", tex_handle);
    }
}
