//! Surface (window presentation) management logic.
//!
//! The acquire/render/present cycle is decoupled:
//! - `acquire()` calls `nextDrawable` and registers the drawable's texture for bindless access
//! - `frame_texture()` returns the registered texture handle
//!
//! ## Deprecation note
//! This file uses `cocoa::base::id`, `NSRect`, and related types from the
//! `cocoa`/`objc` 0.2.x ecosystem, which are deprecated in favour of the
//! `objc2` crate. Migration is deferred until the `metal` and `cocoa` crates
//! offer stable `objc2`-compatible bindings for `CAMetalLayer`.
#![allow(deprecated)]
//! - `render()` uses the already-acquired drawable (does NOT call `nextDrawable` again)
//! - `present()` presents the drawable and unregisters the temporary texture
//!
//! # Argument-buffer race avoidance (triple-buffered bindless slots)
//!
//! The renderer uses a pipelined frame loop:
//!
//! ```text
//! Frame N:   acquire() → render_to_texture() { wait(N-1); submit(N) } → present()
//! Frame N+1: acquire() → render_to_texture() { wait(N);   submit(N+1) } → present()
//! ```
//!
//! `acquire()` re-encodes the drawable's `MTLTexture` GPU resource ID into the
//! global argument buffer. Because `acquire()` runs BEFORE `wait(prev_frame)`,
//! the CPU can overwrite a bindless slot that the GPU is still reading from the
//! previous frame's fine rasterization dispatch. On Apple Silicon this manifests
//! as `kIOGPUCommandBufferCallbackErrorPageFault` on the last compute encoder.
//!
//! The fix: reserve `MAX_FRAMES_IN_FLIGHT` (3) storage-image slots per surface
//! and rotate through them. Frame N writes to slot `N % 3` while the GPU reads
//! slot `(N-1) % 3` — the slots never alias across concurrent frames.

use super::super::{
    DeviceHandle, FrameToken, RenderCommand, SurfaceHandle, SwapchainImageHandle, TextureHandle,
};
use super::compute;
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
        // Use RGBA8Unorm instead of BGRA8Unorm: on Apple Silicon, BGRA8Unorm
        // does not support storage-class access (compute shader UAV writes),
        // so writing to a BGRA swapchain drawable from a compute pass causes a
        // GPU address fault (kIOGPUCommandBufferCallbackErrorPageFault).
        // RGBA8Unorm supports both display and compute storage access.
        let () = msg_send![layer, setPixelFormat: MTLPixelFormat::RGBA8Unorm];
        // Don't set framebufferOnly so the texture can be used with compute
        let () = msg_send![layer, setFramebufferOnly: NO];

        // Pin triple-buffered pool size to match `MAX_FRAMES_IN_FLIGHT`.
        //
        // Per Apple, `maximumDrawableCount` must be `2` or `3`; the documented
        // default on current platforms is already `3`.  We set it explicitly so
        // the layer semantics stay aligned with our bindless slot rotation even
        // if OS defaults drift, and so the requirement is visible in code review.
        let () = msg_send![layer, setMaximumDrawableCount: MAX_FRAMES_IN_FLIGHT as u64];

        let enforced: usize = msg_send![layer, maximumDrawableCount];
        tracing::debug!(
            enforced,
            requested = MAX_FRAMES_IN_FLIGHT,
            "CAMetalLayer maximumDrawableCount (read-back after set)"
        );
        debug_assert_eq!(enforced, MAX_FRAMES_IN_FLIGHT);

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
            bindless_storage_slots,
            present_mode: PresentMode::Auto,
            frame_pending_gpu_commands: Vec::new(),
            pending_render_cb: None,
            drawable_texture_cache: [None; MAX_FRAMES_IN_FLIGHT],
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
    let (device_handle, slots, cached_handles) = match state.surfaces.get(&surface) {
        Some(s) => {
            let handles: [Option<TextureHandle>; MAX_FRAMES_IN_FLIGHT] =
                s.drawable_texture_cache.map(|entry| entry.map(|(_, h)| h));
            (
                Some(s.device_handle),
                Some(s.bindless_storage_slots),
                handles,
            )
        }
        None => (None, None, [None; MAX_FRAMES_IN_FLIGHT]),
    };

    // Release all drawable texture cache entries (kept alive across frames).
    for handle in cached_handles.into_iter().flatten() {
        unregister_surface_texture(state, handle);
    }

    // Release all persistent bindless storage-image slots back to the device
    // registry's free list so another surface can claim them.
    let gpu_idle = super::gpu_is_idle(state);
    if let (Some(dev), Some(slot_arr)) = (device_handle, slots) {
        if let Some(logical_device) = state.devices.get_mut(&dev) {
            let barrier = logical_device.timeline_scheduled_max;
            let slot_barrier = if gpu_idle { None } else { Some(barrier) };
            for &local in &slot_arr {
                logical_device
                    .resource_registry
                    .release_storage_image_slot(local, slot_barrier);
            }
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
    // Texture handles are cached for the surface's lifetime; a dropped frame
    // doesn't need to unregister anything — just warn.
    if let Some(ss) = state.surfaces.get(&surface) {
        if ss.current_texture_handle.is_some() {
            tracing::warn!("Previous drawable was not presented; frame dropped");
        }
    }

    // --- Collect everything from surface state so we can drop the borrow. ---
    let (layer, device_handle, format) = {
        let ss = state
            .surfaces
            .get_mut(&surface)
            .context("Invalid surface handle")?;
        let layer = ss.layer as id;

        let size: CGSize = unsafe { msg_send![layer, drawableSize] };
        ss.width = size.width as u32;
        ss.height = size.height as u32;
        ss.current_frame = (ss.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        (layer, ss.device_handle, ss.format)
    };

    // --- nextDrawable (the blocking call) happens here, outside the borrow. ---
    let drawable: id = unsafe { msg_send![layer, nextDrawable] };
    if drawable == nil {
        anyhow::bail!("Failed to get next drawable from CAMetalLayer");
    }
    unsafe {
        let () = msg_send![drawable, retain];
    }

    let texture_ptr: *mut Object = unsafe { msg_send![drawable, texture] };
    let texture_key = texture_ptr as usize;

    // Re-borrow to read width/height (updated above) and the cache entry.
    let (width, height, current_frame, bindless_slot, cached) = {
        let ss = state
            .surfaces
            .get(&surface)
            .expect("surface still valid after nextDrawable");
        (
            ss.width,
            ss.height,
            ss.current_frame,
            ss.bindless_storage_slots[ss.current_frame],
            ss.drawable_texture_cache[ss.current_frame],
        )
    };

    // Store the drawable pointer (needed for present).
    state.surfaces.get_mut(&surface).unwrap().current_drawable =
        Some(drawable as *mut std::ffi::c_void);

    // --- Drawable texture cache ---
    //
    // CAMetalLayer rotates the same MTLTexture objects through the drawable
    // pool.  Once a frame slot has been seen, the texture pointer never
    // changes, so we can skip the ObjC retain, argument-buffer encoding, and
    // HashMap insert on every subsequent frame for that slot.
    let tex_handle = match cached {
        Some((cached_ptr, cached_handle)) if cached_ptr == texture_key => {
            // Fast path: same texture as last time — reuse the registration.
            tracing::trace!("acquire: drawable texture cache hit for frame slot {current_frame}");
            cached_handle
        }
        Some((_, stale_handle)) => {
            // Texture pointer changed (shouldn't happen in practice, but handle it).
            tracing::debug!(
                "acquire: drawable texture changed for slot {current_frame}, re-registering"
            );
            unregister_surface_texture(state, stale_handle);
            let texture: &mtl::TextureRef = unsafe { &*(texture_ptr as *const mtl::TextureRef) };
            let h = register_surface_texture(
                state,
                device_handle,
                texture,
                width,
                height,
                format,
                bindless_slot,
            )?;
            state
                .surfaces
                .get_mut(&surface)
                .unwrap()
                .drawable_texture_cache[current_frame] = Some((texture_key, h));
            h
        }
        None => {
            // Cold path: first time seeing this frame slot — register once.
            let texture: &mtl::TextureRef = unsafe { &*(texture_ptr as *const mtl::TextureRef) };
            let h = register_surface_texture(
                state,
                device_handle,
                texture,
                width,
                height,
                format,
                bindless_slot,
            )?;
            state
                .surfaces
                .get_mut(&surface)
                .unwrap()
                .drawable_texture_cache[current_frame] = Some((texture_key, h));
            h
        }
    };

    let surface_state = state
        .surfaces
        .get_mut(&surface)
        .expect("surface must be registered before acquiring a frame");
    surface_state.current_texture_handle = Some(tex_handle);

    if let Some(ld) = state.devices.get_mut(&device_handle) {
        ld.process_deletion_queue_up_to_signaled();
    }

    Ok(current_frame as u64)
}

/// Get the texture handle for the currently acquired surface frame.
pub(super) fn frame_texture(state: &MetalState, surface: SurfaceHandle) -> Option<TextureHandle> {
    state
        .surfaces
        .get(&surface)
        .and_then(|s| s.current_texture_handle)
}

/// Render commands to the swapchain using the already-acquired drawable.
///
/// The command buffer is **not committed here** — it is stored in
/// `SurfaceState::pending_render_cb` and committed inside `present()` after
/// appending `present_drawable` and the timeline signal. This merges render
/// and present into a single command buffer, halving per-frame CB submissions.
pub(super) fn render(
    state: &mut MetalState,
    surface: SurfaceHandle,
    _image: SwapchainImageHandle,
    commands: &[RenderCommand],
) -> Result<()> {
    // --- Extract everything we need from surface/device state first so
    //     we can release the immutable borrows before the mutable store. ---
    let (drawable_ptr, device_handle, surface_width, surface_height) = {
        let ss = state
            .surfaces
            .get(&surface)
            .context("Invalid surface handle")?;
        (
            ss.current_drawable
                .context("No drawable acquired — call surface_acquire first")?,
            ss.device_handle,
            ss.width,
            ss.height,
        )
    };

    let drawable = drawable_ptr as id;
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

    // Build render pass descriptor (borrows depth_texture transiently).
    let render_pass = {
        let ss = state
            .surfaces
            .get(&surface)
            .context("Invalid surface handle")?;
        create_render_pass(
            texture,
            ss.depth_texture.as_deref(),
            clear_color,
            clear_depth,
        )
    };

    // Create an owned command buffer so we can store it without keeping a
    // borrow on `logical_device` alive past the encoding phase.
    let owned_cb = {
        let ld = state
            .devices
            .get(&device_handle)
            .context("Device no longer valid")?;
        ld.command_queue.new_command_buffer().to_owned()
    };

    let cb = owned_cb.as_ref();
    let encoder = cb.new_render_command_encoder(render_pass);

    // Declare heap/resource residency and bind the global argument buffer.
    {
        let ld = state
            .devices
            .get(&device_handle)
            .context("Device no longer valid")?;
        let render_stages = mtl::MTLRenderStages::Vertex | mtl::MTLRenderStages::Fragment;
        ld.heap_allocator
            .use_heaps_for_render(encoder, render_stages);
        ld.texture_heap.use_heaps_for_render(encoder, render_stages);
        super::transient::use_transient_heaps_for_render(ld, encoder, render_stages);
        for buf_state in state.buffers.values() {
            if buf_state.device_handle == device_handle {
                encoder.use_resource_at(
                    &buf_state.buffer,
                    mtl::MTLResourceUsage::Read | mtl::MTLResourceUsage::Write,
                    render_stages,
                );
            }
        }
        encoder.set_vertex_buffer(0, Some(&ld.argument_buffer), 0);
        encoder.set_fragment_buffer(0, Some(&ld.argument_buffer), 0);
    }
    tracing::trace!("Bound global argument buffer at slot 0");

    encoder.set_viewport(mtl::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: surface_width as f64,
        height: surface_height as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    encoder.set_scissor_rect(mtl::MTLScissorRect {
        x: 0,
        y: 0,
        width: surface_width as u64,
        height: surface_height as u64,
    });

    record(encoder, commands, &state.pipelines, &state.buffers)?;
    encoder.end_encoding();

    // Store the uncommitted CB — present() will append present_drawable,
    // signal the timeline, and commit it.
    state
        .surfaces
        .get_mut(&surface)
        .expect("surface must exist after render encoding")
        .pending_render_cb = Some(owned_cb);

    Ok(())
}

/// Present the acquired drawable.
///
/// Takes `pending_render_cb` stored by `render()` (or allocates a new empty
/// CB if `render()` was skipped), appends `present_drawable` and the timeline
/// signal, and commits — exactly **one** command buffer per frame.
pub(super) fn present(
    state: &mut MetalState,
    surface: SurfaceHandle,
    _image: SwapchainImageHandle,
) -> Result<crate::timeline::TimelineValue> {
    // Extract what we need before taking the mutable borrow.
    let (device_handle, drawable_ptr) = {
        let ss = state
            .surfaces
            .get(&surface)
            .context("Invalid surface handle")?;
        (ss.device_handle, ss.current_drawable)
    };

    let Some(drawable_ptr) = drawable_ptr else {
        let ld = state
            .devices
            .get_mut(&device_handle)
            .context("Device no longer valid")?;
        ld.process_deletion_queue_up_to_signaled();
        return Ok(ld.timeline_event.as_ref().signaled_value());
    };

    // Assign the next timeline value.
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

    // Take the pending render CB (encoded but not committed) or allocate a
    // fresh one if render() was not called this frame.
    let owned_cb = state
        .surfaces
        .get_mut(&surface)
        .expect("surface must exist at present time")
        .pending_render_cb
        .take()
        .map(Ok)
        .unwrap_or_else(|| {
            state
                .devices
                .get(&device_handle)
                .context("Device no longer valid")
                .map(|ld| ld.command_queue.new_command_buffer().to_owned())
        })?;

    let cb = owned_cb.as_ref();

    // Append present + timeline signal to the same CB as the render pass.
    let drawable = drawable_ptr as id;
    let drawable_ref: &mtl::DrawableRef = unsafe { &*(drawable as *const mtl::DrawableRef) };
    cb.present_drawable(drawable_ref);

    {
        let ld = state
            .devices
            .get(&device_handle)
            .context("Device no longer valid")?;
        cb.encode_signal_event(ld.timeline_event.as_ref(), signal_value);
        let waiter = ld.timeline_waiter.clone();
        let handler = block::ConcreteBlock::new(move |_cb: &mtl::CommandBufferRef| {
            waiter.signal(signal_value);
        })
        .copy();
        cb.add_completed_handler(&handler);
    }

    cb.commit();

    // Release the retained drawable.
    unsafe {
        let (): () = msg_send![drawable, release];
    }

    // Texture handle is now owned by the cache — do not unregister here.
    // Just clear the per-frame "in use" marker.
    let ss = state
        .surfaces
        .get_mut(&surface)
        .expect("surface must be registered before presenting a frame");
    ss.current_drawable = None;
    ss.current_texture_handle = None;

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

    if !pending.is_empty() {
        compute::submit(state, dh, &pending)?;
    }

    present(state, frame.surface, frame.image)
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
        .unwrap_or(TextureFormat::Rgba8Unorm)
}

// ---------------------------------------------------------------------------
// Internal helpers for transient surface texture management
// ---------------------------------------------------------------------------

/// Register the current drawable's MTLTexture at the surface's pre-reserved
/// bindless storage-image slot, so it's visible to compute shaders via
/// `goldy_direct_spatial<T>(n)` and to the public API via
/// [`crate::surface::Frame::texture`].
///
/// The slot (`bindless_slot`) is one of MAX_FRAMES_IN_FLIGHT rotating slots
/// allocated in `create()` and released in `destroy()`. Only the *texture
/// object* at offset `storage_image_global_index(bindless_slot) * encoded_length`
/// is rewritten per frame. This is the "transient allocation path that doesn't
/// leak indices" anticipated by `abstract-gpu-surface.md` (risk #2).
///
/// The drawable `MTLTexture` is owned via [`ForeignType::from_ptr`] after an
/// explicit `retain` (never cast `&TextureRef` → `&Texture` and `clone()` —
/// that was UB / PAC faults).
fn register_surface_texture(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    texture: &mtl::TextureRef,
    width: u32,
    height: u32,
    format: TextureFormat,
    bindless_slot: u32,
) -> Result<TextureHandle> {
    let handle = state.next_texture_handle;
    state.next_texture_handle += 1;

    let raw = texture.as_ptr();
    unsafe {
        let () = msg_send![raw as id, retain];
    }
    let texture_owned = unsafe { mtl::Texture::from_ptr(raw) };

    let global_idx = {
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Device no longer valid")?;

        logical_device
            .resource_registry
            .bind_storage_image_slot(handle, bindless_slot);
        let global = ResourceRegistry::storage_image_global_index(bindless_slot);

        // Surface drawable is used as a storage image (RWTexture2D write target),
        // so encode with the ReadWrite storage image encoder.
        let encoded_length = logical_device.storage_image_encoder.encoded_length();
        let offset = (global as u64) * encoded_length;
        if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
            logical_device
                .storage_image_encoder
                .set_argument_buffer(&logical_device.argument_buffer, offset);
            logical_device
                .storage_image_encoder
                .set_texture(0, texture_owned.as_ref());
        } else {
            tracing::error!(
                "register_surface_texture: argument buffer overflow — \
                 offset {offset} + encoded_length {encoded_length} exceeds \
                 ARGUMENT_BUFFER_SIZE {ARGUMENT_BUFFER_SIZE}; \
                 drawable will not be encoded and compute shaders may see a stale binding"
            );
        }

        global
    };

    state.textures.insert(
        handle,
        TextureState {
            device_handle,
            width,
            height,
            format,
            texture: texture_owned,
            arg_buffer_index: bindless_slot,
            is_storage_image: true,
            slot_owned_externally: true,
        },
    );

    tracing::debug!(
        "Encoded surface drawable into texture {} ({}x{}, bindless storage local={}, global={})",
        handle,
        width,
        height,
        bindless_slot,
        global_idx,
    );

    Ok(handle)
}

/// Unregister the per-frame TextureHandle for the drawable, releasing the
/// owned `MTLTexture` (via `Drop` → `release`) and clearing the handle→slot
/// map entry. The surface's bindless slot itself is NOT released here — it
/// stays reserved across frames until the surface is destroyed.
fn unregister_surface_texture(state: &mut MetalState, tex_handle: TextureHandle) {
    if let Some(tex_state) = state.textures.remove(&tex_handle) {
        if let Some(device) = state.devices.get_mut(&tex_state.device_handle) {
            device.resource_registry.unregister_texture(tex_handle);
        }
        tracing::debug!("Unregistered surface drawable texture {}", tex_handle);
    }
}
