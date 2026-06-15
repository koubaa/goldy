//! Surface (window presentation) management logic.
//!
//! The acquire/render/present cycle is decoupled:
//! - `acquire()` ensures the per-slot scratch texture and returns its bindless handle
//! - `frame_texture()` returns the scratch texture handle for the current slot
//! - `present()` acquires a drawable, blits scratch → drawable, then presents
//!
//! ## Deprecation note
//! This file uses `cocoa::base::id`, `NSRect`, and related types from the
//! `cocoa`/`objc` 0.2.x ecosystem, which are deprecated in favour of the
//! `objc2` crate. Migration is deferred until the `metal` and `cocoa` crates
//! offer stable `objc2`-compatible bindings for `CAMetalLayer`.
#![allow(deprecated)]
//! - `render()` targets the scratch texture for the current in-flight slot
//! - `present()` calls `nextDrawable`, blits scratch → drawable, then presents
//!
//! # Argument-buffer race avoidance (triple-buffered bindless slots)
//!
//! The renderer uses a pipelined frame loop:
//!
//! ```text
//! Frame N:   acquire() → render/copy to scratch → present() { nextDrawable; blit; present }
//! Frame N+1: acquire() → …
//! ```
//!
//! The fix: reserve `MAX_FRAMES_IN_FLIGHT` (3) storage-image slots per surface
//! and rotate through them. Frame N writes to slot `N % 3` while the GPU reads
//! slot `(N-1) % 3` — the slots never alias across concurrent frames.

use super::super::{
    ContextHandle, DeviceHandle, FrameToken, RenderCommand, SurfaceHandle, SwapchainImageHandle, TextureHandle,
};
use super::compute;
use super::render_commands::{create_render_pass, record};
use super::types::{MetalState, SurfaceState, MAX_FRAMES_IN_FLIGHT};
use super::utils::depth_format_to_mtl;
use crate::types::{DepthFormat, PresentMode, TextureFormat};
use ::metal as mtl;
use anyhow::{Context, Result};
use cocoa::base::{id, nil, NO, YES};
use core_graphics_types::geometry::CGSize;
use foreign_types::ForeignTypeRef;
use mtl::{MTLPixelFormat, MTLStorageMode, MTLTextureUsage, TextureDescriptor};
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};
use std::sync::atomic::Ordering;

/// Create a surface for window presentation.
/// When `depth_format` is `Some`, a depth buffer is created for 3D rendering.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    window: &dyn HasWindowHandle,
    _display: &dyn HasDisplayHandle,
    depth_format: Option<DepthFormat>,
) -> Result<SurfaceHandle> {
    let logical_device = state.devices.get_mut(&device_handle).context("Invalid device handle")?;

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
        // Use RGBA8Unorm instead of BGRA8Unorm: on Apple Silicon, BGRA8Unorm
        // does not support storage-class access (compute shader UAV writes),
        // so writing to a BGRA swapchain drawable from a compute pass causes a
        // GPU address fault (kIOGPUCommandBufferCallbackErrorPageFault).
        // RGBA8Unorm supports both display and compute storage access.
        let () = msg_send![layer, setPixelFormat: MTLPixelFormat::RGBA8Unorm];
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

    // Reserve MAX_FRAMES_IN_FLIGHT storage-image bindless slots for this
    // surface. Each frame uses a different slot so the CPU never re-encodes a
    // slot that the GPU is concurrently reading (see module-level doc comment).
    let bindless_storage_slots: [u32; MAX_FRAMES_IN_FLIGHT] = std::array::from_fn(|_| {
        logical_device
            .ledger
            .lock()
            .unwrap()
            .resource_registry
            .reserve_storage_image_slot()
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
            scratch_texture_handles: [None; MAX_FRAMES_IN_FLIGHT],
            bindless_storage_slots,
            present_mode: PresentMode::Auto,
            frame_pending_gpu_commands: Vec::new(),
            pending_acquire_count: 0,
            last_acquired_image_index: None,
        },
    );
    tracing::info!(
        "Created Metal surface {} (bindless storage slots={:?})",
        handle,
        bindless_storage_slots
    );
    Ok(handle)
}

/// Destroy a surface.
pub(super) fn destroy(state: &mut MetalState, surface: SurfaceHandle) {
    let (device_handle, slots, scratch_handles, drawable) = match state.surfaces.get(&surface) {
        Some(s) => (
            Some(s.device_handle),
            Some(s.bindless_storage_slots),
            s.scratch_texture_handles.iter().filter_map(|h| *h).collect::<Vec<_>>(),
            s.current_drawable,
        ),
        None => (None, None, Vec::new(), None),
    };

    for tex_handle in scratch_handles {
        super::texture::destroy(state, tex_handle);
    }
    if let Some(drawable) = drawable {
        unsafe {
            let (): () = msg_send![drawable as id, release];
        }
    }

    // Release all persistent bindless storage-image slots back to the device
    // registry's free list so another surface can claim them.
    let gpu_idle = super::gpu_is_idle(state);
    if let (Some(dev), Some(slot_arr)) = (device_handle, slots) {
        if let Some(logical_device) = state.devices.get(&dev) {
            let barrier = logical_device
                .timeline_scheduled_max
                .load(std::sync::atomic::Ordering::Relaxed);
            let slot_barrier = if gpu_idle { None } else { Some(barrier) };
            for &local in &slot_arr {
                logical_device
                    .ledger
                    .lock()
                    .unwrap()
                    .resource_registry
                    .release_storage_image_slot(local, slot_barrier);
            }
        }
    }

    state.surfaces.remove(&surface);
}

/// Acquire the next in-flight frame slot.
///
/// Ensures the per-slot scratch texture exists and registers it in the bindless
/// descriptor set. The drawable is acquired at `present()` time; until then the
/// scheme writes to the stable scratch handle for this slot.
pub(super) fn acquire(
    state: &mut MetalState,
    surface: SurfaceHandle,
    ctx: super::ContextHandle,
) -> Result<SwapchainImageHandle> {
    let _tz = crate::tracy_zone!("mtl.surface.acquire");

    let (device_handle, width, height, format, bindless_slot, frame_slot) = {
        let surface_state = state.surfaces.get_mut(&surface).context("Invalid surface handle")?;

        let layer = surface_state.layer as id;
        let size: CGSize = unsafe { msg_send![layer, drawableSize] };
        surface_state.width = (size.width as u32).max(1);
        surface_state.height = (size.height as u32).max(1);

        surface_state.current_frame = (surface_state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        let frame_slot = surface_state.current_frame;

        (
            surface_state.device_handle,
            surface_state.width,
            surface_state.height,
            surface_state.format,
            surface_state.bindless_storage_slots[frame_slot],
            frame_slot,
        )
    };

    let tex_handle = ensure_scratch_texture_slot(
        state,
        surface,
        device_handle,
        frame_slot,
        width,
        height,
        format,
        bindless_slot,
    )?;

    let image_index = frame_slot as u32;
    {
        let surface_state = state
            .surfaces
            .get_mut(&surface)
            .expect("surface must be registered before acquiring a frame");
        surface_state.current_texture_handle = Some(tex_handle);
        surface_state.last_acquired_image_index = Some(image_index);
        surface_state.pending_acquire_count = surface_state.pending_acquire_count.saturating_add(1);
    }

    if let Some(sc_arc) = state.contexts.get(&ctx) {
        sc_arc
            .lock()
            .unwrap()
            .signal_queue
            .push(crate::signal::Signal::SwapchainAcquired { image_index });
    }

    // Drain per-context deletion queue on the context's own clock (hot path),
    // then the device-level queue as the async GC safety net (see issue #190).
    if let Some(sc_arc) = state.contexts.get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        let ctx_signaled = sc.timeline_event.as_ref().signaled_value();
        sc.deletion_queue.process_up_to(ctx_signaled);
    }
    {
        let retired = super::context::device_retired(state, device_handle);
        if let Some(ld) = state.devices.get(&device_handle) {
            ld.process_deletion_queue_up_to(retired);
        }
    }

    Ok(image_index as u64)
}

/// Get the texture handle for the currently acquired surface frame.
pub(super) fn frame_texture(state: &MetalState, surface: SurfaceHandle) -> Option<TextureHandle> {
    state.surfaces.get(&surface).and_then(|s| s.current_texture_handle)
}

/// Render commands to the swapchain scratch texture for the current in-flight slot.
pub(super) fn render(
    state: &mut MetalState,
    surface: SurfaceHandle,
    _image: SwapchainImageHandle,
    commands: &[RenderCommand],
) -> Result<()> {
    let (device_handle, width, height, depth_texture, scratch_handle) = {
        let surface_state = state.surfaces.get(&surface).context("Invalid surface handle")?;
        let frame_slot = surface_state.current_frame;
        let scratch = surface_state.scratch_texture_handles[frame_slot]
            .or(surface_state.current_texture_handle)
            .context("No scratch texture acquired — call surface_acquire first")?;
        (
            surface_state.device_handle,
            surface_state.width,
            surface_state.height,
            surface_state.depth_texture.clone(),
            scratch,
        )
    };

    let scratch_tex = state
        .textures
        .get(&scratch_handle)
        .context("Surface scratch texture not found")?
        .texture
        .clone();

    let logical_device = state.devices.get(&device_handle).context("Device no longer valid")?;

    let (staging_data, lowered_commands, has_bindings) =
        super::frame_table::prepare_render_commands(&state.buffers, &state.pipelines, commands)?;

    let completed = super::context::device_retired(state, device_handle);
    let prologue_row = if has_bindings {
        Some(super::frame_table::run_prologue_for_device(
            state,
            device_handle,
            logical_device,
            &staging_data,
            completed,
        )?)
    } else {
        None
    };

    let mut clear_color = None;
    let mut clear_depth = None;
    for cmd in commands {
        match cmd {
            RenderCommand::Clear(color) => clear_color = Some(*color),
            RenderCommand::ClearDepth(depth) => clear_depth = Some(*depth),
            _ => {}
        }
    }
    let render_pass = create_render_pass(scratch_tex.as_ref(), depth_texture.as_deref(), clear_color, clear_depth);

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let encoder = command_buffer.new_render_command_encoder(render_pass);

    let render_stages = mtl::MTLRenderStages::Vertex | mtl::MTLRenderStages::Fragment;
    logical_device
        .heap_allocator
        .lock()
        .unwrap()
        .use_heaps_for_render(encoder, render_stages);
    logical_device
        .texture_heap
        .lock()
        .unwrap()
        .use_heaps_for_render(encoder, render_stages);
    for buf_state in state.buffers.values() {
        if buf_state.device_handle == device_handle {
            encoder.use_resource_at(
                &buf_state.buffer,
                mtl::MTLResourceUsage::Read | mtl::MTLResourceUsage::Write,
                render_stages,
            );
        }
    }
    {
        let ft = logical_device.frame_table.lock().unwrap();
        encoder.use_resource_at(ft.table_buffer(), mtl::MTLResourceUsage::Read, render_stages);
    }

    encoder.set_vertex_buffer(0, Some(&logical_device.argument_buffer), 0);
    encoder.set_fragment_buffer(0, Some(&logical_device.argument_buffer), 0);
    tracing::trace!("Bound global argument buffer at slot 0");

    encoder.set_viewport(mtl::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: width as f64,
        height: height as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    encoder.set_scissor_rect(mtl::MTLScissorRect {
        x: 0,
        y: 0,
        width: width as u64,
        height: height as u64,
    });

    record(
        encoder,
        &lowered_commands,
        &state.pipelines,
        &state.buffers,
        prologue_row,
    )?;

    encoder.end_encoding();
    command_buffer.commit();

    // When the frame table was used we must wait for the GPU to finish reading
    // the ring row before we can allow it to be overwritten.  Surface renders
    // don't carry a context-level timeline signal, so we block here.  For frames
    // without any frame-table bindings (prologue_row == None) we remain async.
    if let Some(row) = prologue_row {
        command_buffer.wait_until_completed();
        if let Some(ld) = state.devices.get(&device_handle) {
            super::frame_table::record_submission_for_device(ld, row, completed);
        }
    }

    Ok(())
}

/// Present the current frame: acquire a drawable, blit scratch → drawable, present.
///
/// The scratch texture handle is cleared but the per-slot scratch texture persists
/// across frames for retained scheme resubmission.
pub(super) fn present(
    state: &mut MetalState,
    surface: SurfaceHandle,
    _image: SwapchainImageHandle,
    ctx: ContextHandle,
) -> Result<crate::timeline::TimelineValue> {
    let (device_handle, width, height, scratch_handle, layer, return_image) = {
        let surface_state = state.surfaces.get(&surface).context("Invalid surface handle")?;
        let scratch = surface_state
            .current_texture_handle
            .context("No frame acquired — call surface_acquire first")?;
        (
            surface_state.device_handle,
            surface_state.width,
            surface_state.height,
            scratch,
            surface_state.layer as id,
            surface_state.last_acquired_image_index,
        )
    };

    let scratch_mtl = state
        .textures
        .get(&scratch_handle)
        .context("Surface scratch texture not found")?
        .texture
        .clone();

    let signal_value = {
        let ld = state.devices.get(&device_handle).context("Device no longer valid")?;
        let v = ld.timeline_next.fetch_add(1, Ordering::Relaxed);
        ld.timeline_scheduled_max.fetch_max(v, Ordering::Relaxed);
        v
    };

    let logical_device = state.devices.get(&device_handle).context("Device no longer valid")?;

    let drawable: id = {
        let _dz = crate::tracy_zone!("mtl.surface.nextDrawable");
        unsafe { msg_send![layer, nextDrawable] }
    };
    if drawable == nil {
        anyhow::bail!("Failed to get next drawable from CAMetalLayer");
    }
    unsafe {
        let () = msg_send![drawable, retain];
    }

    let texture_ptr: *mut Object = unsafe { msg_send![drawable, texture] };
    let drawable_tex: &mtl::TextureRef = unsafe { &*(texture_ptr as *const mtl::TextureRef) };

    let owned_command_buffer = logical_device.command_queue.new_command_buffer().to_owned();
    let command_buffer = owned_command_buffer.as_ref();

    let blit = command_buffer.new_blit_command_encoder();
    let w = width.max(1) as u64;
    let h = height.max(1) as u64;
    blit.copy_from_texture(
        scratch_mtl.as_ref(),
        0,
        0,
        mtl::MTLOrigin { x: 0, y: 0, z: 0 },
        mtl::MTLSize {
            width: w,
            height: h,
            depth: 1,
        },
        drawable_tex,
        0,
        0,
        mtl::MTLOrigin { x: 0, y: 0, z: 0 },
    );
    blit.end_encoding();

    let (timeline_event, waiter, signal_queue_present, return_pending) = {
        let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
        let sc = sc_arc.lock().unwrap();
        (
            sc.timeline_event.clone(),
            sc.timeline_waiter.clone(),
            std::sync::Arc::clone(&sc.signal_queue),
            sc.pending_swapchain_returns.clone(),
        )
    };
    command_buffer.encode_signal_event(timeline_event.as_ref(), signal_value);

    let handler = block::ConcreteBlock::new(move |_cb: &mtl::CommandBufferRef| {
        waiter.signal(signal_value);
        if let Some(idx) = return_image {
            signal_queue_present.push(crate::signal::Signal::SwapchainReturned { image_index: idx });
            if let Ok(mut pending) = return_pending.lock() {
                pending.push((surface, idx));
            }
        }
    })
    .copy();
    command_buffer.add_completed_handler(&handler);

    let drawable_ref: &mtl::DrawableRef = unsafe { &*(drawable as *const mtl::DrawableRef) };
    command_buffer.present_drawable(drawable_ref);
    command_buffer.commit();

    unsafe {
        let (): () = msg_send![drawable, release];
    }

    let surface_state = state
        .surfaces
        .get_mut(&surface)
        .expect("surface must be registered before presenting a frame");
    surface_state.current_drawable = None;
    surface_state.current_texture_handle = None;
    surface_state.last_acquired_image_index = None;

    if let Some(sc_arc) = state.contexts.get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        sc.in_flight_command_buffers
            .push_back((signal_value, owned_command_buffer));
        sc.last_submitted_seq = signal_value;
    }
    if let Some(sc_arc) = state.contexts.get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        let ctx_signaled = sc.timeline_event.as_ref().signaled_value();
        sc.deletion_queue.process_up_to(ctx_signaled);
    }
    {
        let retired = super::context::device_retired(state, device_handle);
        if let Some(ld) = state.devices.get(&device_handle) {
            ld.process_deletion_queue_up_to(retired);
        }
    }

    Ok(signal_value)
}
pub(super) fn submit_frame(state: &mut MetalState, frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
    let pending = {
        let surf = state
            .surfaces
            .get_mut(&frame.surface)
            .context("Invalid surface handle")?;
        std::mem::take(&mut surf.frame_pending_gpu_commands)
    };

    if !pending.is_empty() {
        return compute::submit(state, frame.context, &pending);
    }

    let sc_arc = state.contexts.get(&frame.context).context("Invalid context handle")?;
    let sc = sc_arc.lock().unwrap();
    Ok(sc
        .timeline_event
        .as_ref()
        .signaled_value()
        .max(sc.last_committed_timeline.unwrap_or(0)))
}

pub(super) fn present_frame(
    state: &mut MetalState,
    frame: FrameToken,
    _submit_tv: crate::timeline::TimelineValue,
) -> Result<crate::timeline::TimelineValue> {
    present(state, frame.surface, frame.image, frame.context)
}

/// Set the present mode on the CAMetalLayer.
pub(super) fn set_present_mode(state: &mut MetalState, surface: SurfaceHandle, mode: PresentMode) -> Result<()> {
    let surface_state = state.surfaces.get_mut(&surface).context("Invalid surface handle")?;

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
pub(super) fn resize(state: &mut MetalState, surface: SurfaceHandle, width: u32, height: u32) -> Result<()> {
    let (device_handle, scratch_handles) = {
        let surface_state = state.surfaces.get_mut(&surface).context("Invalid surface handle")?;

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

        let size = CGSize::new(width as f64, height as f64);
        unsafe {
            let () = msg_send![surface_state.layer as id, setDrawableSize: size];
        }

        let scratch_handles: Vec<TextureHandle> = surface_state
            .scratch_texture_handles
            .iter()
            .filter_map(|h| *h)
            .collect();
        surface_state.scratch_texture_handles = [None; MAX_FRAMES_IN_FLIGHT];
        surface_state.current_texture_handle = None;
        surface_state.pending_acquire_count = 0;

        (surface_state.device_handle, scratch_handles)
    };

    for tex_handle in scratch_handles {
        super::texture::destroy(state, tex_handle);
    }

    for sc_arc in state.contexts.values() {
        let sc = sc_arc.lock().unwrap();
        if sc.device == device_handle {
            sc.pending_swapchain_returns.lock().unwrap().clear();
        }
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

// ---------------------------------------------------------------------------
// Internal helpers for per-slot scratch texture management
// ---------------------------------------------------------------------------

/// Ensure the per-in-flight-slot scratch texture exists at the current surface size.
///
/// Returns a stable `TextureHandle` for `frame_slot` so retained scheme partitions
/// can bake the destination into cached command streams (Vulkan/DX12 parity).
fn ensure_scratch_texture_slot(
    state: &mut MetalState,
    surface: SurfaceHandle,
    device_handle: DeviceHandle,
    frame_slot: usize,
    width: u32,
    height: u32,
    format: TextureFormat,
    bindless_slot: u32,
) -> Result<TextureHandle> {
    if let Some(handle) = state
        .surfaces
        .get(&surface)
        .and_then(|s| s.scratch_texture_handles[frame_slot])
    {
        if let Some(ts) = state.textures.get(&handle) {
            if ts.width == width && ts.height == height {
                return Ok(handle);
            }
        }
    }

    if let Some(old) = state
        .surfaces
        .get_mut(&surface)
        .and_then(|s| s.scratch_texture_handles[frame_slot].take())
    {
        super::texture::destroy(state, old);
    }

    let handle =
        super::texture::create_scratch_for_surface_slot(state, device_handle, width, height, format, bindless_slot)?;

    state.surfaces.get_mut(&surface).unwrap().scratch_texture_handles[frame_slot] = Some(handle);
    Ok(handle)
}
