//! Surface (window presentation) management logic.

use super::super::{DeviceHandle, RenderCommand, SurfaceHandle, SwapchainImageHandle};
use super::render_commands::{create_render_pass, record};
use super::types::{MetalState, SurfaceState, MAX_FRAMES_IN_FLIGHT};
use super::utils::depth_format_to_mtl;
use crate::types::{DepthFormat, TextureFormat};
use ::metal as mtl;
use anyhow::{Context, Result};
use cocoa::base::{id, nil, YES};
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
        let () = msg_send![layer, setFramebufferOnly: YES];

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
        },
    );

    tracing::info!("Created Metal surface {}", handle);
    Ok(handle)
}

/// Destroy a surface.
pub(super) fn destroy(state: &mut MetalState, surface: SurfaceHandle) {
    state.surfaces.remove(&surface);
}

/// Acquire the next swapchain image.
pub(super) fn acquire(
    state: &mut MetalState,
    surface: SurfaceHandle,
) -> Result<SwapchainImageHandle> {
    let surface_state = state
        .surfaces
        .get_mut(&surface)
        .context("Invalid surface handle")?;

    let layer = surface_state.layer as id;

    let size: CGSize = unsafe { msg_send![layer, drawableSize] };
    surface_state.width = size.width as u32;
    surface_state.height = size.height as u32;

    surface_state.current_frame = (surface_state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    Ok(surface_state.current_frame as u64)
}

/// Render commands to the swapchain and present.
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

    let logical_device = state
        .devices
        .get(&surface_state.device_handle)
        .context("Device no longer valid")?;

    let layer = surface_state.layer as id;

    let drawable: id = unsafe { msg_send![layer, nextDrawable] };
    if drawable == nil {
        anyhow::bail!("Failed to get next drawable");
    }

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
    if logical_device.heap_buffer_count > 0 {
        encoder.use_heap_at(&logical_device.buffer_heap, render_stages);
    }
    if logical_device.heap_texture_count > 0 {
        encoder.use_heap_at(&logical_device.texture_heap, render_stages);
    }
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

    let _: () = unsafe { msg_send![command_buffer.as_ptr(), presentDrawable: drawable] };
    command_buffer.commit();

    Ok(())
}

/// Present is a no-op; presentation happens in render.
pub(super) fn present(
    _state: &mut MetalState,
    _surface: SurfaceHandle,
    _image: SwapchainImageHandle,
) -> Result<()> {
    Ok(())
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
