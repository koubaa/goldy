//! Surface (window presentation) management logic.
//!
//! The acquire/render/present cycle:
//! - `acquire()` calls `nextDrawable` and registers the drawable's texture in a
//!   rotating bindless storage-image slot
//! - `frame_texture()` returns the registered texture handle for the current frame
//! - `render()` targets the already-acquired drawable
//! - `present()` presents the drawable and unregisters its temporary texture handle
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

use super::super::{DeviceHandle, FrameToken, RenderCommand, SurfaceHandle, SwapchainImageHandle, TextureHandle};
use super::compute;
use super::render_commands::{create_render_pass, record};
use super::types::{
    MetalState, ResourceRegistry, SurfaceState, TextureState, ARGUMENT_BUFFER_SIZE, MAX_FRAMES_IN_FLIGHT,
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
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::types::{SharedLogicalDevice, SharedMetalSubmissionContext, TimelineWaiter};
use crate::backend::{PresentFinishState, PresentGpuWork, ScheduledPresentBlockingWait, ScheduledPresentWaitOutcome};
use crate::timeline::TimelineValue;

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
            .descriptors
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
            drawable_slots: std::array::from_fn(|_| None),
            drawable_texture_handles: std::array::from_fn(|_| None),
            current_texture_handle: None,
            bindless_storage_slots,
            present_mode: PresentMode::Auto,
            frame_pending_gpu_commands: Vec::new(),
            pending_acquire_count: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            last_acquired_image_index: None,
            pending_acquire: None,
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
    let (device_handle, slots, drawable_texture_handles, drawable_slots, current_texture_handle) =
        match state.surfaces.get(&surface) {
            Some(s) => (
                Some(s.device_handle),
                Some(s.bindless_storage_slots),
                s.drawable_texture_handles,
                s.drawable_slots,
                s.current_texture_handle,
            ),
            None => (
                None,
                None,
                [None; MAX_FRAMES_IN_FLIGHT],
                [None; MAX_FRAMES_IN_FLIGHT],
                None,
            ),
        };

    for slot in 0..MAX_FRAMES_IN_FLIGHT {
        if let Some(tex_handle) = drawable_texture_handles[slot] {
            unregister_surface_texture(state, tex_handle);
        }
        if let Some(d) = drawable_slots[slot] {
            unsafe {
                let (): () = msg_send![d as id, release];
            }
        }
    }
    if let Some(tex_handle) = current_texture_handle {
        unregister_surface_texture(state, tex_handle);
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
                    .descriptors
                    .lock()
                    .unwrap()
                    .resource_registry
                    .release_storage_image_slot(local, slot_barrier);
            }
        }
    }

    state.surfaces.remove(&surface);
}

/// State prepared under the global backend lock before a blocking `nextDrawable`.
pub(super) struct MetalAcquirePending {
    pub surface: SurfaceHandle,
    pub ctx: super::ContextHandle,
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: crate::types::TextureFormat,
    pub frame_slot: usize,
    pub bindless_slot: u32,
}

struct MetalSurfaceAcquireWork {
    layer: Option<usize>,
}

impl Drop for MetalSurfaceAcquireWork {
    fn drop(&mut self) {
        if let Some(layer_ptr) = self.layer.take() {
            unsafe {
                let (): () = msg_send![layer_ptr as id, release];
            }
        }
    }
}

impl crate::backend::SurfaceAcquireWork for MetalSurfaceAcquireWork {
    fn run(mut self: Box<Self>) -> Result<crate::backend::SurfaceAcquireDrawable> {
        let layer = self.layer.take().expect("acquire work run twice") as id;
        let _dz = crate::tracy_zone!("mtl.surface.nextDrawable");
        let drawable: id = unsafe { msg_send![layer, nextDrawable] };
        unsafe {
            let (): () = msg_send![layer, release];
        }
        if super::api_log::enabled() {
            super::api_log::log_next_drawable(drawable != nil);
        }
        Ok(crate::backend::SurfaceAcquireDrawable {
            ptr: drawable as usize,
            ok: drawable != nil,
        })
    }
}

pub(super) fn take_surface_acquire_work(
    state: &mut MetalState,
    surface: SurfaceHandle,
    ctx: super::ContextHandle,
) -> Result<Box<dyn crate::backend::SurfaceAcquireWork>> {
    let (pending, layer_ptr) = prepare_acquire(state, surface, ctx)?;
    state.surfaces.get_mut(&surface).unwrap().pending_acquire = Some(pending);
    Ok(Box::new(MetalSurfaceAcquireWork { layer: Some(layer_ptr) }))
}

pub(super) fn finish_surface_acquire_from_drawable(
    state: &mut MetalState,
    surface: SurfaceHandle,
    ctx: super::ContextHandle,
    drawable: crate::backend::SurfaceAcquireDrawable,
) -> Result<(SwapchainImageHandle, u32)> {
    let pending = state
        .surfaces
        .get_mut(&surface)
        .and_then(|ss| ss.pending_acquire.take())
        .context("finish_surface_acquire: missing pending acquire state")?;
    if pending.surface != surface {
        anyhow::bail!("finish_surface_acquire: pending acquire surface mismatch");
    }
    if pending.ctx != ctx {
        anyhow::bail!("finish_surface_acquire: pending acquire context mismatch");
    }
    finish_acquire(state, pending, drawable)
}

pub(super) fn prepare_acquire(
    state: &mut MetalState,
    surface: SurfaceHandle,
    ctx: super::ContextHandle,
) -> Result<(MetalAcquirePending, usize)> {
    let _tz = crate::tracy_zone!("mtl.surface.prepare_acquire");

    if let Some(ss) = state.surfaces.get_mut(&surface) {
        if ss.pending_acquire.is_some() {
            tracing::warn!("clearing incomplete surface acquire for surface {surface}");
            ss.pending_acquire = None;
        }
    }

    let (device_handle, width, height, format, frame_slot, bindless_slot) = {
        let ss = state.surfaces.get_mut(&surface).context("Invalid surface handle")?;
        let layer = ss.layer as id;
        let size: CGSize = unsafe { msg_send![layer, drawableSize] };
        ss.width = (size.width as u32).max(1);
        ss.height = (size.height as u32).max(1);
        ss.current_frame = (ss.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        let frame_slot = ss.current_frame;
        (
            ss.device_handle,
            ss.width,
            ss.height,
            ss.format,
            frame_slot,
            ss.bindless_storage_slots[frame_slot],
        )
    };

    if let Some(tex_handle) = state
        .surfaces
        .get(&surface)
        .and_then(|s| s.drawable_texture_handles[frame_slot])
    {
        tracing::warn!("Previous drawable in slot {frame_slot} was not presented; cleaning up");
        unregister_surface_texture(state, tex_handle);
    }
    if let Some(ss) = state.surfaces.get_mut(&surface) {
        if let Some(d) = ss.drawable_slots[frame_slot].take() {
            unsafe {
                let (): () = msg_send![d as id, release];
            }
        }
        ss.drawable_texture_handles[frame_slot] = None;
    }

    let layer = state.surfaces.get(&surface).unwrap().layer as id;
    unsafe {
        let (): () = msg_send![layer, retain];
    }

    Ok((
        MetalAcquirePending {
            surface,
            ctx,
            device_handle,
            width,
            height,
            format,
            frame_slot,
            bindless_slot,
        },
        layer as usize,
    ))
}

pub(super) fn finish_acquire(
    state: &mut MetalState,
    pending: MetalAcquirePending,
    drawable: crate::backend::SurfaceAcquireDrawable,
) -> Result<(SwapchainImageHandle, u32)> {
    let _tz = crate::tracy_zone!("mtl.surface.finish_acquire");
    let MetalAcquirePending {
        surface,
        ctx,
        device_handle,
        width,
        height,
        format,
        frame_slot,
        bindless_slot,
    } = pending;

    let drawable_id = drawable.ptr as id;
    if !drawable.ok || drawable_id == nil {
        anyhow::bail!("Failed to get next drawable from CAMetalLayer");
    }
    unsafe {
        let () = msg_send![drawable_id, retain];
    }
    {
        let ss = state.surfaces.get_mut(&surface).unwrap();
        ss.drawable_slots[frame_slot] = Some(drawable_id as *mut std::ffi::c_void);
    }

    let texture_ptr: *mut Object = unsafe { msg_send![drawable_id, texture] };
    let texture: &mtl::TextureRef = unsafe { &*(texture_ptr as *const mtl::TextureRef) };

    let tex_handle = register_surface_texture(state, device_handle, texture, width, height, format, bindless_slot)?;

    let image_index = {
        let ss = state.surfaces.get_mut(&surface).expect("surface registered above");
        let image_index = frame_slot as u32;
        ss.drawable_texture_handles[frame_slot] = Some(tex_handle);
        ss.current_texture_handle = Some(tex_handle);
        ss.last_acquired_image_index = Some(image_index);
        ss.pending_acquire_count
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        image_index
    };

    if let Some(sc_arc) = state.contexts.get(&ctx) {
        sc_arc
            .lock()
            .unwrap()
            .signal_queue
            .push(crate::signal::Signal::SwapchainAcquired { image_index });
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

    Ok((image_index as SwapchainImageHandle, frame_slot as u32))
}

/// Abandon an acquired frame without presenting.
pub(super) fn cancel_frame(state: &mut MetalState, frame: crate::backend::FrameToken) -> Result<()> {
    let _tz = crate::tracy_zone!("mtl.surface.cancel_frame");
    let surface = frame.surface;
    let present_slot = frame.present_slot as usize;

    let (drawable, tex_handle) = {
        let ss = state.surfaces.get_mut(&surface).context("Invalid surface handle")?;

        let drawable = ss.drawable_slots[present_slot].take();
        let tex_handle = ss.drawable_texture_handles[present_slot].take();
        if ss.current_texture_handle == tex_handle {
            ss.current_texture_handle = None;
        }
        ss.last_acquired_image_index = None;
        ss.frame_pending_gpu_commands.clear();

        let prev = ss
            .pending_acquire_count
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        if prev == 0 {
            tracing::warn!(
                surface,
                present_slot,
                "cancel_frame: pending_acquire_count was already 0"
            );
        }
        (drawable, tex_handle)
    };

    if let Some(d) = drawable {
        unsafe {
            let (): () = msg_send![d as id, release];
        }
    }
    if let Some(tex_handle) = tex_handle {
        unregister_surface_texture(state, tex_handle);
    }
    Ok(())
}

/// Acquire the next swapchain image (legacy single-lock entry).
///
/// Prefer [`take_surface_acquire_work`] + [`finish_acquire`] so `nextDrawable` runs
/// outside the global backend mutex.
pub(super) fn acquire(
    state: &mut MetalState,
    surface: SurfaceHandle,
    ctx: super::ContextHandle,
) -> Result<(SwapchainImageHandle, u32)> {
    let _tz = crate::tracy_zone!("mtl.surface.acquire");
    let (pending, layer_ptr) = prepare_acquire(state, surface, ctx)?;
    let layer = layer_ptr as id;
    let drawable: id = unsafe { msg_send![layer, nextDrawable] };
    unsafe {
        let (): () = msg_send![layer, release];
    }
    finish_acquire(
        state,
        pending,
        crate::backend::SurfaceAcquireDrawable {
            ptr: drawable as usize,
            ok: drawable != nil,
        },
    )
}

/// Get the texture handle for the currently acquired surface frame.
pub(super) fn frame_texture(state: &MetalState, surface: SurfaceHandle) -> Option<TextureHandle> {
    state.surfaces.get(&surface).and_then(|s| s.current_texture_handle)
}

/// Render commands to the swapchain using the already-acquired drawable.
pub(super) fn render(
    state: &mut MetalState,
    surface: SurfaceHandle,
    _image: SwapchainImageHandle,
    present_slot: u32,
    commands: &[RenderCommand],
) -> Result<()> {
    let surface_state = state.surfaces.get(&surface).context("Invalid surface handle")?;
    let present_slot = present_slot as usize;

    let drawable_ptr =
        surface_state.drawable_slots[present_slot].context("No drawable acquired — call surface_acquire first")?;
    let drawable = drawable_ptr as id;

    let device_handle = surface_state.device_handle;
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

/// Clone present resources under the global backend lock for lock-free GPU enqueue.
pub(super) fn prepare_present_work(
    state: &mut MetalState,
    frame: crate::backend::FrameToken,
    _submit_tv: TimelineValue,
) -> Result<Box<dyn PresentGpuWork>> {
    let surface = frame.surface;
    let present_slot = frame.present_slot as usize;
    let ctx = frame.context;

    let surface_state = state.surfaces.get_mut(&surface).context("Invalid surface handle")?;
    let device_handle = surface_state.device_handle;
    let return_image = surface_state.last_acquired_image_index;
    let pending_acquire_count = std::sync::Arc::clone(&surface_state.pending_acquire_count);
    let drawable_ptr = surface_state.drawable_slots[present_slot].take();
    let tex_handle = surface_state.drawable_texture_handles[present_slot].take();
    if surface_state.current_texture_handle == tex_handle {
        surface_state.current_texture_handle = None;
    }
    surface_state.last_acquired_image_index = None;

    let Some(drawable_ptr) = drawable_ptr else {
        let present_timeline = if let Some(sc_arc) = state.contexts.get(&ctx) {
            let mut sc = sc_arc.lock().unwrap();
            let ctx_signaled = sc.timeline_event.as_ref().signaled_value();
            sc.deletion_queue.process_up_to(ctx_signaled);
            ctx_signaled
        } else {
            super::context::device_retired(state, device_handle)
        };
        let retired = super::context::device_retired(state, device_handle);
        if let Some(ld) = state.devices.get(&device_handle) {
            ld.process_deletion_queue_up_to(retired);
        }
        return Ok(Box::new(MetalSkipPresentGpuWork {
            frame,
            present_timeline,
        }));
    };

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Device no longer valid")?
        .clone();
    let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?.clone();
    let (timeline_event, waiter, signal_queue_present, return_pending) = {
        let sc = sc_arc.lock().unwrap();
        (
            sc.timeline_event.clone(),
            sc.timeline_waiter.clone(),
            Arc::clone(&sc.signal_queue),
            sc.pending_swapchain_returns.clone(),
        )
    };

    Ok(Box::new(MetalPresentGpuWork {
        frame,
        drawable_ptr: drawable_ptr as usize,
        tex_handle,
        logical_device,
        sc_arc,
        timeline_event,
        waiter,
        signal_queue_present,
        return_pending,
        return_image,
        pending_acquire_count,
    }))
}

/// Bookkeeping after [`MetalPresentGpuWork::run`] — brief global backend lock only.
pub(super) fn finish_present(
    state: &mut MetalState,
    finish: PresentFinishState,
    _submit_tv: TimelineValue,
) -> Result<TimelineValue> {
    let surface = finish.frame.surface;
    let present_slot = finish.frame.present_slot as usize;
    let ctx = finish.frame.context;
    let device_handle = state
        .surfaces
        .get(&surface)
        .map(|s| s.device_handle)
        .context("Invalid surface handle")?;

    if finish.present_ok {
        if let Some(th) = finish.scratch_texture {
            unregister_surface_texture(state, th);
        }
        if let Some(surface_state) = state.surfaces.get_mut(&surface) {
            surface_state.drawable_slots[present_slot] = None;
            surface_state.drawable_texture_handles[present_slot] = None;
        }
    }

    if let Some(signal_timeline) = finish.signal_timeline {
        if let Some(sc_arc) = state.contexts.get(&ctx) {
            let mut sc = sc_arc.lock().unwrap();
            sc.last_submitted_seq = signal_timeline;
            super::drain_completed_cbs(&mut sc);
        }
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

    Ok(finish.present_timeline)
}

pub(super) fn schedule_present_on_submission_worker(
    state: &mut MetalState,
    frame: crate::backend::FrameToken,
    _submit_tv: TimelineValue,
) -> Result<TimelineValue> {
    let surface = frame.surface;
    let present_slot = frame.present_slot as usize;
    let ctx = frame.context;

    let (
        device_handle,
        return_image,
        pending_acquire_count,
        drawable_ptr,
        tex_handle,
    ) = {
        let surface_state = state.surfaces.get_mut(&surface).context("Invalid surface handle")?;
        let device_handle = surface_state.device_handle;
        let return_image = surface_state.last_acquired_image_index;
        let pending_acquire_count = std::sync::Arc::clone(&surface_state.pending_acquire_count);
        let drawable_ptr = surface_state.drawable_slots[present_slot].take();
        let tex_handle = surface_state.drawable_texture_handles[present_slot].take();
        if surface_state.current_texture_handle == tex_handle {
            surface_state.current_texture_handle = None;
        }
        surface_state.last_acquired_image_index = None;
        (
            device_handle,
            return_image,
            pending_acquire_count,
            drawable_ptr,
            tex_handle,
        )
    };

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Device no longer valid")?
        .clone();
    let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?.clone();
    let (timeline_event, waiter, signal_queue_present, return_pending) = {
        let sc = sc_arc.lock().unwrap();
        (
            sc.timeline_event.clone(),
            sc.timeline_waiter.clone(),
            std::sync::Arc::clone(&sc.signal_queue),
            sc.pending_swapchain_returns.clone(),
        )
    };

    let signal_value = super::pending_submit::preallocate_device_timeline(&logical_device);
    let present_ok = drawable_ptr.is_some();
    let finish = PresentFinishState {
        frame,
        return_fence: 0,
        scratch_texture: tex_handle,
        scratch_layout_updated: false,
        present_timeline: signal_value,
        copy_timeline: None,
        frame_compute_timeline: None,
        signal_timeline: Some(signal_value),
        render_pass_submitted: false,
        present_ok,
    };

    let pending_finishes = std::sync::Arc::clone(&state.pending_present_finishes);

    if let Some(drawable_ptr) = drawable_ptr {
        let command_buffer = logical_device.command_queue.new_command_buffer().to_owned();
        super::pending_submit::enqueue_metal_scheduled_present(
            &logical_device,
            command_buffer,
            signal_value,
            timeline_event,
            waiter,
            sc_arc,
            drawable_ptr as usize,
            return_image,
            signal_queue_present,
            return_pending,
            pending_acquire_count,
            finish,
            pending_finishes,
        )?;
    } else {
        super::pending_submit::enqueue_metal_skip_present(
            &logical_device,
            signal_value,
            waiter,
            finish,
            pending_finishes,
        )?;
    }

    Ok(signal_value)
}

fn synthesize_scheduled_present_finish(
    frame: crate::backend::FrameToken,
    present_tv: TimelineValue,
) -> PresentFinishState {
    PresentFinishState {
        frame,
        return_fence: 0,
        scratch_texture: None,
        scratch_layout_updated: false,
        present_timeline: present_tv,
        copy_timeline: None,
        frame_compute_timeline: None,
        signal_timeline: Some(present_tv),
        render_pass_submitted: false,
        present_ok: true,
    }
}

pub(super) fn take_scheduled_present_blocking_wait(
    state: &MetalState,
    frame: crate::backend::FrameToken,
    present_tv: TimelineValue,
) -> Result<Option<Box<dyn ScheduledPresentBlockingWait>>> {
    let device_handle = state
        .surfaces
        .get(&frame.surface)
        .context("Invalid surface handle")?
        .device_handle;
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Surface's device is invalid")?
        .clone();
    let pending_finishes = std::sync::Arc::clone(&state.pending_present_finishes);
    Ok(Some(Box::new(MetalScheduledPresentWait {
        worker: std::sync::Arc::clone(&logical_device.submission_worker),
        pending_finishes,
        frame,
        present_tv,
    })))
}

struct MetalScheduledPresentWait {
    worker: std::sync::Arc<crate::backend::submission_worker::SubmissionWorker>,
    pending_finishes: std::sync::Arc<std::sync::Mutex<Vec<PresentFinishState>>>,
    frame: crate::backend::FrameToken,
    present_tv: TimelineValue,
}

impl ScheduledPresentBlockingWait for MetalScheduledPresentWait {
    fn run(self: Box<Self>) -> Result<ScheduledPresentWaitOutcome> {
        let _tz = crate::tracy_zone!("goldy.mtl.scheduled_present.wait");
        self.worker.wait_submitted(self.present_tv)?;
        self.worker.check_error()?;

        let finish = {
            let mut pending = self.pending_finishes.lock().unwrap();
            pending
                .iter()
                .position(|f| {
                    f.frame.surface == self.frame.surface
                        && f.frame.image == self.frame.image
                        && f.frame.present_slot == self.frame.present_slot
                })
                .map(|idx| pending.remove(idx))
        };

        Ok(ScheduledPresentWaitOutcome {
            frame: self.frame,
            present_tv: self.present_tv,
            finish,
            return_fence: 0,
        })
    }
}

pub(super) fn apply_scheduled_present_bookkeeping(
    state: &mut MetalState,
    outcome: ScheduledPresentWaitOutcome,
) -> Result<()> {
    let finish = match outcome.finish {
        Some(finish) => finish,
        None => {
            tracing::warn!(
                target: "goldy::metal",
                surface = outcome.frame.surface,
                image = outcome.frame.image,
                present_slot = outcome.frame.present_slot,
                present_tv = outcome.present_tv,
                "scheduled present finish missing from pending queue; synthesizing surface bookkeeping"
            );
            synthesize_scheduled_present_finish(outcome.frame, outcome.present_tv)
        }
    };
    finish_present(state, finish, outcome.present_tv)?;
    Ok(())
}

struct MetalSkipPresentGpuWork {
    frame: crate::backend::FrameToken,
    present_timeline: TimelineValue,
}

impl PresentGpuWork for MetalSkipPresentGpuWork {
    fn run(self: Box<Self>) -> Result<PresentFinishState> {
        Ok(PresentFinishState {
            frame: self.frame,
            return_fence: 0,
            scratch_texture: None,
            scratch_layout_updated: false,
            present_timeline: self.present_timeline,
            copy_timeline: None,
            frame_compute_timeline: None,
            signal_timeline: None,
            render_pass_submitted: false,
            present_ok: false,
        })
    }
}

struct MetalPresentGpuWork {
    frame: crate::backend::FrameToken,
    /// Drawable retained until GPU present; stored as usize for `Send`.
    drawable_ptr: usize,
    tex_handle: Option<TextureHandle>,
    logical_device: SharedLogicalDevice,
    sc_arc: SharedMetalSubmissionContext,
    timeline_event: mtl::SharedEvent,
    waiter: TimelineWaiter,
    signal_queue_present: Arc<crate::signal::SignalQueue>,
    return_pending: Arc<std::sync::Mutex<Vec<super::types::PendingSwapchainReturn>>>,
    return_image: Option<u32>,
    pending_acquire_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl PresentGpuWork for MetalPresentGpuWork {
    fn run(self: Box<Self>) -> Result<PresentFinishState> {
        let _tz = crate::tracy_zone!("mtl.present.gpu");
        let owned_command_buffer = self.logical_device.command_queue.new_command_buffer().to_owned();
        let signal_value = super::pending_submit::preallocate_device_timeline(&self.logical_device);
        super::pending_submit::enqueue_metal_present(
            &self.logical_device,
            owned_command_buffer,
            signal_value,
            self.timeline_event,
            self.waiter,
            self.sc_arc,
            self.drawable_ptr,
            self.return_image,
            self.signal_queue_present,
            self.return_pending,
            self.pending_acquire_count,
        )?;
        self.logical_device.submission_worker.wait_submitted(signal_value)?;
        self.logical_device.submission_worker.check_error()?;

        Ok(PresentFinishState {
            frame: self.frame,
            return_fence: 0,
            scratch_texture: self.tex_handle,
            scratch_layout_updated: false,
            present_timeline: signal_value,
            copy_timeline: None,
            frame_compute_timeline: None,
            signal_timeline: Some(signal_value),
            render_pass_submitted: false,
            present_ok: true,
        })
    }
}

/// Present the acquired drawable (legacy single-lock entry — prefer split path).
pub(super) fn present(state: &mut MetalState, frame: crate::backend::FrameToken) -> Result<TimelineValue> {
    let work = prepare_present_work(state, frame, 0)?;
    let finish = work.run()?;
    finish_present(state, finish, 0)
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
        return compute::submit(state, frame.context, &pending, None);
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
    present(state, frame)
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

    let layer = surface_state.layer as id;
    let size = CGSize::new(width as f64, height as f64);
    unsafe {
        let () = msg_send![layer, setDrawableSize: size];
    }

    surface_state
        .pending_acquire_count
        .store(0, std::sync::atomic::Ordering::Release);
    if let Some(device_handle) = state.surfaces.get(&surface).map(|s| s.device_handle) {
        for sc_arc in state.contexts.values() {
            let sc = sc_arc.lock().unwrap();
            if sc.device == device_handle {
                sc.pending_swapchain_returns.lock().unwrap().clear();
            }
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
        let logical_device = state.devices.get(&device_handle).context("Device no longer valid")?;

        logical_device
            .descriptors
            .lock()
            .unwrap()
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
            sampled_arg_buffer_index: None,
            is_storage_image: true,
            slot_owned_externally: true,
            is_heap_allocated: false,
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
        if let Some(device) = state.devices.get(&tex_state.device_handle) {
            device
                .descriptors
                .lock()
                .unwrap()
                .resource_registry
                .unregister_texture(tex_handle);
        }
        tracing::debug!("Unregistered surface drawable texture {}", tex_handle);
    }
}

/// Headless unit-test surface with no CAMetalLayer (skip-present / bookkeeping only).
#[cfg(all(test, target_os = "macos", feature = "metal"))]
pub(super) fn register_stub_surface_for_test(
    state: &mut MetalState,
    device_handle: DeviceHandle,
) -> Result<SurfaceHandle> {
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
    let bindless_storage_slots: [u32; MAX_FRAMES_IN_FLIGHT] = std::array::from_fn(|_| {
        logical_device
            .descriptors
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
            width: 64,
            height: 64,
            format: TextureFormat::Rgba8Unorm,
            depth_format: None,
            depth_texture: None,
            current_frame: 0,
            layer: std::ptr::null_mut(),
            drawable_slots: std::array::from_fn(|_| None),
            drawable_texture_handles: std::array::from_fn(|_| None),
            current_texture_handle: None,
            bindless_storage_slots,
            present_mode: PresentMode::Auto,
            frame_pending_gpu_commands: Vec::new(),
            pending_acquire_count: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            last_acquired_image_index: None,
            pending_acquire: None,
        },
    );
    Ok(handle)
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
fn metal_fifo_test_lock() -> std::fs::File {
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;
    let path = std::env::temp_dir().join("goldy-metal-fifo-test.lock");
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(&path)
        .expect("open metal fifo test lock");
    let fd = file.as_raw_fd();
    loop {
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if rc == 0 {
            break;
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EINTR) {
            panic!("flock({}): {err}", path.display());
        }
    }
    file
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod fifo_present_tests {
    use super::*;
    use crate::backend::metal::MetalBackend;
    use crate::backend::{FrameToken, GpuBackend, GpuBackendPresentSplit};

    #[test]
    fn fifo_present_enabled_by_default() {
        let backend = MetalBackend::new().expect("Metal backend");
        assert!(
            backend.schedules_present_on_submit_worker(),
            "Metal must schedule present on the submission worker by default"
        );
    }

    #[test]
    fn no_drawable_skip_path_unblocks_wait() {
        let _gpu_lock = metal_fifo_test_lock();
        let mut backend = MetalBackend::new().expect("Metal backend");
        let device = backend.create_device(0).expect("device");
        let ctx = backend.create_context(device).expect("context");
        let surface = backend.test_register_stub_surface(device).expect("stub surface");
        let token = FrameToken {
            surface,
            image: 0,
            context: ctx,
            frame_slot: 0,
            present_slot: 0,
        };

        let present_tv = backend
            .schedule_present_on_submission_worker(token, 0)
            .expect("schedule skip present");
        let wait = backend
            .take_scheduled_present_blocking_wait(token, present_tv)
            .expect("take wait")
            .expect("FIFO wait object");
        let outcome = wait.run().expect("wait run");
        assert_eq!(outcome.present_tv, present_tv);
        assert!(
            outcome.finish.as_ref().is_some_and(|f| !f.present_ok),
            "skip path must record present_ok=false"
        );

        backend
            .apply_scheduled_present_bookkeeping(outcome)
            .expect("bookkeeping");
    }
}
