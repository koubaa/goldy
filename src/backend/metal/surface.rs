//! Surface (window presentation) management logic.
//!
//! The acquire/render/present cycle:
//! - `acquire()` calls `nextDrawable` inside an autorelease pool, retains exactly
//!   one strong drawable ref and one strong texture ref, then registers the
//!   texture in a rotating bindless storage-image slot
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

use super::super::{DeviceHandle, FrameToken, SurfaceHandle, SwapchainImageHandle, TextureHandle};
use super::compute;
use super::types::{
    MetalState, ResourceRegistry, SurfaceState, TextureState, ARGUMENT_BUFFER_SIZE, MAX_FRAMES_IN_FLIGHT,
};
use super::utils::depth_format_to_mtl;
use crate::types::{DepthFormat, PresentMode, TextureFormat};
use super::objc_id::{id, nil, NO, YES};
use ::metal as mtl;
use anyhow::{Context, Result};
use core_graphics_types::geometry::CGSize;
use foreign_types::{ForeignType, ForeignTypeRef};
use mtl::{MTLPixelFormat, MTLStorageMode, MTLTextureUsage, TextureDescriptor};
use objc::rc::autoreleasepool;
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};
use std::sync::Arc;

use super::types::{SharedLogicalDevice, SharedMetalSubmissionContext, TimelineWaiter};
use crate::backend::{PresentFinishState, PresentGpuWork};
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

    let view = match window_handle.as_raw() {
        #[cfg(target_os = "macos")]
        RawWindowHandle::AppKit(handle) => handle.ns_view.as_ptr() as id,
        #[cfg(target_os = "ios")]
        RawWindowHandle::UiKit(handle) => handle.ui_view.as_ptr() as id,
        other => anyhow::bail!("expected AppKit/UiKit window handle, got {other:?}"),
    };

    let (layer, width, height) = unsafe {
        let mut layer: id = msg_send![class!(CAMetalLayer), layer];
        let () = msg_send![layer, setDevice: logical_device.device.as_ptr()];
        // Use RGBA8Unorm instead of BGRA8Unorm: on Apple Silicon, BGRA8Unorm
        // does not support storage-class access (compute shader UAV writes),
        // so writing to a BGRA swapchain drawable from a compute pass causes a
        // GPU address fault (kIOGPUCommandBufferCallbackErrorPageFault).
        // RGBA8Unorm supports both display and compute storage access.
        let () = msg_send![layer, setPixelFormat: MTLPixelFormat::RGBA8Unorm];
        // Don't set framebufferOnly so the texture can be used with compute
        let () = msg_send![layer, setFramebufferOnly: NO];

        #[cfg(target_os = "macos")]
        {
            let () = msg_send![view, setWantsLayer: YES];
            let () = msg_send![view, setLayer: layer];
        }
        #[cfg(target_os = "ios")]
        {
            let scale: f64 = msg_send![view, contentScaleFactor];
            let () = msg_send![layer, setContentsScale: scale];
            let existing: id = msg_send![view, layer];
            let is_metal: objc::runtime::BOOL = msg_send![existing, isKindOfClass: class!(CAMetalLayer)];
            if is_metal != NO {
                layer = existing;
                let () = msg_send![layer, setDevice: logical_device.device.as_ptr()];
                let () = msg_send![layer, setPixelFormat: MTLPixelFormat::RGBA8Unorm];
                let () = msg_send![layer, setFramebufferOnly: NO];
                let () = msg_send![layer, setContentsScale: scale];
            } else {
                let () = msg_send![existing, addSublayer: layer];
            }
        }

        let (fw, fh) = {
            #[cfg(target_os = "macos")]
            {
                let r: cocoa::foundation::NSRect = msg_send![view, frame];
                (r.size.width, r.size.height)
            }
            #[cfg(target_os = "ios")]
            {
                let r: core_graphics_types::geometry::CGRect = msg_send![view, bounds];
                let scale: f64 = msg_send![view, contentScaleFactor];
                (r.size.width * scale, r.size.height * scale)
            }
        };
        let size = CGSize::new(fw, fh);
        let () = msg_send![layer, setDrawableSize: size];

        let w = (fw as u32).max(1);
        let h = (fh as u32).max(1);

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
    if let (Some(dev), Some(slot_arr)) = (device_handle, slots) {
        if let Some(logical_device) = state.devices.get(&dev) {
            let mut registry = logical_device.descriptors.lock().unwrap();
            for &local in &slot_arr {
                registry.release_storage_image_slot(local);
            }
        }
    }

    state.surfaces.remove(&surface);
}

/// Acquire the next swapchain image.
///
/// Calls `nextDrawable` on the CAMetalLayer inside an autorelease pool and
/// registers the drawable's texture in the bindless descriptor set. The API
/// returns autoreleased objects; Goldy retains exactly one strong drawable ref
/// and one strong texture ref (via [`register_surface_texture`]) before the pool
/// drains. The texture handle is available via `frame_texture()` until
/// `present()` is called.
pub(super) fn acquire(
    state: &mut MetalState,
    surface: SurfaceHandle,
    ctx: super::ContextHandle,
) -> Result<(SwapchainImageHandle, u32)> {
    let _tz = crate::tracy_zone!("mtl.surface.acquire");

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

    // Clean up any previously acquired drawable in this slot that wasn't presented.
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
    let (drawable_ptr, tex_handle) = autoreleasepool(|| -> Result<(*mut std::ffi::c_void, TextureHandle)> {
        let _dz = crate::tracy_zone!("mtl.surface.nextDrawable");
        let drawable: id = unsafe { msg_send![layer, nextDrawable] };

        if super::api_log::enabled() {
            super::api_log::log_next_drawable(drawable != nil);
        }

        if drawable == nil {
            anyhow::bail!("Failed to get next drawable from CAMetalLayer");
        }
        unsafe {
            let () = msg_send![drawable, retain];
        }

        let texture_ptr: *mut Object = unsafe { msg_send![drawable, texture] };
        let texture: &mtl::TextureRef = unsafe { &*(texture_ptr as *const mtl::TextureRef) };

        let tex_handle = register_surface_texture(state, device_handle, texture, width, height, format, bindless_slot)?;
        Ok((drawable as *mut std::ffi::c_void, tex_handle))
    })?;

    {
        let ss = state.surfaces.get_mut(&surface).unwrap();
        ss.drawable_slots[frame_slot] = Some(drawable_ptr);
    }

    let image_index = {
        let ss = state.surfaces.get_mut(&surface).expect("surface registered above");
        let image_index = frame_slot as u32;
        ss.drawable_texture_handles[frame_slot] = Some(tex_handle);
        ss.current_texture_handle = Some(tex_handle);
        ss.last_acquired_image_index = Some(image_index);
        ss.pending_acquire_count = ss.pending_acquire_count.saturating_add(1);
        image_index
    };

    if let Some(sc_arc) = state.contexts.get(&ctx) {
        sc_arc
            .lock()
            .unwrap()
            .signal_queue
            .push(crate::signal::Signal::SwapchainAcquired { image_index });
    }

    // Drain per-context deletion queue on the context's own clock (hot path),
    // then the device-level queue as the async GC safety net (see issue #190).
    if let Some(ld) = state.devices.get(&device_handle) {
        if let Some(sc_arc) = state.contexts.get(&ctx) {
            let mut sc = sc_arc.lock().unwrap();
            let ctx_signaled = sc.timeline_event.as_ref().signaled_value();
            super::drain_context_deletion_queue_up_to(ld, &mut sc.deletion_queue, ctx_signaled);
        }
        let retired = super::context::device_retired(state, device_handle);
        ld.process_deletion_queue_up_to(
            retired,
            Some(&super::context::snapshot_context_completed_values(state, device_handle)),
        );
    }

    Ok((image_index as SwapchainImageHandle, frame_slot as u32))
}

/// Get the texture handle for the currently acquired surface frame.
pub(super) fn frame_texture(state: &MetalState, surface: SurfaceHandle) -> Option<TextureHandle> {
    state.surfaces.get(&surface).and_then(|s| s.current_texture_handle)
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
    let drawable_ptr = surface_state.drawable_slots[present_slot].take();
    let tex_handle = surface_state.drawable_texture_handles[present_slot].take();
    if surface_state.current_texture_handle == tex_handle {
        surface_state.current_texture_handle = None;
    }
    surface_state.last_acquired_image_index = None;

    let Some(drawable_ptr) = drawable_ptr else {
        let present_timeline = if let Some(ld) = state.devices.get(&device_handle) {
            if let Some(sc_arc) = state.contexts.get(&ctx) {
                let mut sc = sc_arc.lock().unwrap();
                let ctx_signaled = sc.timeline_event.as_ref().signaled_value();
                super::drain_context_deletion_queue_up_to(ld, &mut sc.deletion_queue, ctx_signaled);
                ctx_signaled
            } else {
                super::context::device_retired(state, device_handle)
            }
        } else {
            super::context::device_retired(state, device_handle)
        };
        let retired = super::context::device_retired(state, device_handle);
        if let Some(ld) = state.devices.get(&device_handle) {
            ld.process_deletion_queue_up_to(
                retired,
                Some(&super::context::snapshot_context_completed_values(state, device_handle)),
            );
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
        surface,
        drawable_ptr: drawable_ptr as usize,
        tex_handle,
        logical_device,
        sc_arc,
        timeline_event,
        waiter,
        signal_queue_present,
        return_pending,
        return_image,
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

    if let Some(ld) = state.devices.get(&device_handle) {
        if let Some(sc_arc) = state.contexts.get(&ctx) {
            let mut sc = sc_arc.lock().unwrap();
            let ctx_signaled = sc.timeline_event.as_ref().signaled_value();
            super::drain_context_deletion_queue_up_to(ld, &mut sc.deletion_queue, ctx_signaled);
        }
        let retired = super::context::device_retired(state, device_handle);
        ld.process_deletion_queue_up_to(
            retired,
            Some(&super::context::snapshot_context_completed_values(state, device_handle)),
        );
    }

    Ok(finish.present_timeline)
}

pub(super) fn submit_frame(state: &mut MetalState, frame: &FrameToken) -> Result<TimelineValue> {
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
    surface: SurfaceHandle,
    /// Drawable retained until GPU present; stored as usize for `Send`.
    drawable_ptr: usize,
    tex_handle: Option<TextureHandle>,
    logical_device: SharedLogicalDevice,
    sc_arc: SharedMetalSubmissionContext,
    timeline_event: mtl::SharedEvent,
    waiter: TimelineWaiter,
    signal_queue_present: Arc<crate::signal::SignalQueue>,
    return_pending: Arc<std::sync::Mutex<Vec<(SurfaceHandle, u32)>>>,
    return_image: Option<u32>,
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
            self.surface,
            self.drawable_ptr,
            self.return_image,
            self.signal_queue_present,
            self.return_pending,
        )?;

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

    surface_state.pending_acquire_count = 0;
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
