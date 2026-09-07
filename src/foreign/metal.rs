//! Metal as a foreign graphics object: no Goldy device, verbs under one lock.
//!
//! Creates its own `MTLDevice` / `MTLCommandQueue`. Offscreen surfaces hold a
//! private texture plus a shared staging buffer. [`ForeignSurface::blit`] copies
//! host pixels through `MTLBlitCommandEncoder` and a copy-back into the shared
//! buffer so [`ForeignSurface::snapshot`] can assert GPU contents.
//!
//! Windowed surfaces attach a `CAMetalLayer` to the view. `blit` copies into
//! `nextDrawable` and presents. Scheme submissions stay on the Goldy CPU
//! (or other compute) device.

#![allow(deprecated)]

use crate::backend::metal::format_to_mtl;
use crate::pixel::{PixelSink, PixmapLayout};
use crate::types::{PresentMode, TextureFormat};
use crate::GoldyError;
use core_graphics_types::geometry::CGSize;
use foreign_types::ForeignType;
use metal as mtl;
use mtl::{
    Buffer, CommandBuffer, CommandQueue, Device, MTLBlitOption, MTLOrigin, MTLResourceOptions, MTLSize, MTLStorageMode,
    MTLTextureUsage, Texture, TextureDescriptor,
};
use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(target_os = "macos")]
use cocoa::base::{id, nil, NO, YES};

#[cfg(target_os = "ios")]
type id = *mut Object;

#[cfg(target_os = "ios")]
const nil: id = std::ptr::null_mut();

#[cfg(target_os = "ios")]
const YES: objc::runtime::BOOL = true;

#[cfg(target_os = "ios")]
const NO: objc::runtime::BOOL = false;

/// Process-wide Metal adapter. Lazily created on [`try_adapter`].
pub struct ForeignMetal {
    state: Mutex<AdapterState>,
}

struct AdapterState {
    device: Device,
    queue: CommandQueue,
    next_id: u32,
    surfaces: HashMap<u32, SurfaceSlot>,
}

struct SurfaceSlot {
    width: u32,
    height: u32,
    format: TextureFormat,
    generation: u64,
    /// Private offscreen texture. `None` for windowed surfaces.
    texture: Option<Texture>,
    /// Retained `CAMetalLayer*`. `None` for offscreen surfaces.
    layer: Option<usize>,
    staging: Buffer,
    staging_size: usize,
    last_cb: Option<CommandBuffer>,
    dropped: bool,
}

struct SurfaceHandle {
    adapter: Arc<ForeignMetal>,
    id: u32,
}

impl Drop for SurfaceHandle {
    fn drop(&mut self) {
        self.adapter.release(self.id);
    }
}

/// Metal texture or `CAMetalLayer` owned by the foreign singleton.
#[derive(Clone)]
pub struct ForeignSurface {
    inner: Arc<SurfaceHandle>,
}

static ADAPTER: OnceLock<Result<Arc<ForeignMetal>, String>> = OnceLock::new();

/// Return the process-wide adapter, creating it on first success.
///
/// Returns `None` when no Metal device is present. Failures are cached.
pub fn try_adapter() -> Option<Arc<ForeignMetal>> {
    match ADAPTER.get_or_init(init_adapter) {
        Ok(a) => Some(Arc::clone(a)),
        Err(e) => {
            tracing::debug!("foreign Metal adapter unavailable: {e}");
            None
        }
    }
}

fn init_adapter() -> Result<Arc<ForeignMetal>, String> {
    init_adapter_inner().map_err(|e| e.detail())
}

fn init_adapter_inner() -> Result<Arc<ForeignMetal>, GoldyError> {
    let device = Device::system_default()
        .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("foreign Metal: no system default device")))?;
    let name = device.name().to_string();
    let queue = device.new_command_queue();
    tracing::info!(adapter = %name, "foreign Metal adapter");
    Ok(Arc::new(ForeignMetal {
        state: Mutex::new(AdapterState {
            device,
            queue,
            next_id: 1,
            surfaces: HashMap::new(),
        }),
    }))
}

fn alloc_staging(device: &Device, layout: PixmapLayout) -> (Buffer, usize) {
    let staging_size = layout.staging_bytes().max(1) as usize;
    let staging = device.new_buffer(staging_size as u64, MTLResourceOptions::StorageModeShared);
    (staging, staging_size)
}

fn attach_layer(
    device: &Device,
    window: &dyn HasWindowHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    present_mode: PresentMode,
) -> Result<usize, GoldyError> {
    let window_handle = window
        .window_handle()
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("foreign Metal: window handle: {e:?}")))?;
    let view = match window_handle.as_raw() {
        #[cfg(target_os = "macos")]
        RawWindowHandle::AppKit(handle) => handle.ns_view.as_ptr() as id,
        #[cfg(target_os = "ios")]
        RawWindowHandle::UiKit(handle) => handle.ui_view.as_ptr() as id,
        other => {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign Metal windowed expected AppKit/UiKit, got {other:?}"
            )));
        }
    };
    if view == nil {
        return Err(GoldyError::Backend(anyhow::anyhow!("foreign Metal: nil NSView")));
    }

    unsafe {
        let layer: id = msg_send![class!(CAMetalLayer), layer];
        if layer == nil {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign Metal: CAMetalLayer alloc failed"
            )));
        }
        let () = msg_send![layer, setDevice: device.as_ptr()];
        let () = msg_send![layer, setPixelFormat: format_to_mtl(format)];
        let () = msg_send![layer, setFramebufferOnly: NO];
        let sync = match present_mode {
            PresentMode::Immediate => NO,
            PresentMode::Fifo | PresentMode::Mailbox | PresentMode::Auto => YES,
        };
        let () = msg_send![layer, setDisplaySyncEnabled: sync];

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
                let () = msg_send![existing, setDevice: device.as_ptr()];
                let () = msg_send![existing, setPixelFormat: format_to_mtl(format)];
                let () = msg_send![existing, setFramebufferOnly: NO];
                let () = msg_send![existing, setContentsScale: scale];
                let () = msg_send![existing, setDisplaySyncEnabled: sync];
                let size = CGSize::new(width.max(1) as f64, height.max(1) as f64);
                let () = msg_send![existing, setDrawableSize: size];
                let () = msg_send![existing, retain];
                return Ok(existing as usize);
            }
            let bounds: core_graphics_types::geometry::CGRect = msg_send![view, bounds];
            let () = msg_send![layer, setFrame: bounds];
            let mask: usize = 18;
            let () = msg_send![layer, setAutoresizingMask: mask];
            let () = msg_send![existing, addSublayer: layer];
        }

        let size = CGSize::new(width.max(1) as f64, height.max(1) as f64);
        let () = msg_send![layer, setDrawableSize: size];
        let () = msg_send![layer, retain];
        Ok(layer as usize)
    }
}

impl ForeignMetal {
    /// Offscreen `width × height` texture. No window, no swapchain.
    pub fn offscreen(
        self: &Arc<Self>,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<ForeignSurface, GoldyError> {
        let layout = PixmapLayout::tight(width, height, format);
        layout.validate()?;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.reap();
        let slot = SurfaceSlot::offscreen(&state.device, layout)?;
        Ok(state.insert(Arc::clone(self), slot))
    }

    /// Windowed `CAMetalLayer` attached to `window`. `blit` presents.
    pub fn windowed(
        self: &Arc<Self>,
        window: &dyn HasWindowHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        present_mode: PresentMode,
    ) -> Result<ForeignSurface, GoldyError> {
        let layout = PixmapLayout::tight(width.max(1), height.max(1), format);
        layout.validate()?;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.reap();
        let layer = attach_layer(&state.device, window, layout.width, layout.height, format, present_mode)?;
        let slot = SurfaceSlot::windowed(&state.device, layout, layer);
        Ok(state.insert(Arc::clone(self), slot))
    }
}

impl AdapterState {
    fn insert(&mut self, adapter: Arc<ForeignMetal>, slot: SurfaceSlot) -> ForeignSurface {
        let id = self.next_id;
        self.next_id += 1;
        self.surfaces.insert(id, slot);
        ForeignSurface {
            inner: Arc::new(SurfaceHandle { adapter, id }),
        }
    }

    fn reap(&mut self) {
        let ids: Vec<u32> = self
            .surfaces
            .iter()
            .filter(|(_, s)| s.dropped)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            if let Some(slot) = self.surfaces.remove(&id) {
                slot.destroy();
            }
        }
    }
}

impl SurfaceSlot {
    fn offscreen(device: &Device, layout: PixmapLayout) -> Result<Self, GoldyError> {
        let descriptor = TextureDescriptor::new();
        descriptor.set_width(layout.width as u64);
        descriptor.set_height(layout.height as u64);
        descriptor.set_pixel_format(format_to_mtl(layout.format));
        descriptor.set_storage_mode(MTLStorageMode::Private);
        descriptor.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let texture = device.new_texture(&descriptor);
        let (staging, staging_size) = alloc_staging(device, layout);
        Ok(Self {
            width: layout.width,
            height: layout.height,
            format: layout.format,
            generation: 1,
            texture: Some(texture),
            layer: None,
            staging,
            staging_size,
            last_cb: None,
            dropped: false,
        })
    }

    fn windowed(device: &Device, layout: PixmapLayout, layer: usize) -> Self {
        let (staging, staging_size) = alloc_staging(device, layout);
        Self {
            width: layout.width,
            height: layout.height,
            format: layout.format,
            generation: 1,
            texture: None,
            layer: Some(layer),
            staging,
            staging_size,
            last_cb: None,
            dropped: false,
        }
    }

    fn destroy(self) {
        if let Some(cb) = self.last_cb {
            cb.wait_until_completed();
        }
        if let Some(layer) = self.layer {
            unsafe {
                let layer = layer as id;
                let (): () = msg_send![layer, release];
            }
        }
    }

    fn wait(&mut self) {
        if let Some(cb) = self.last_cb.take() {
            cb.wait_until_completed();
        }
    }
}

fn copy_size(layout: PixmapLayout) -> MTLSize {
    MTLSize {
        width: layout.width as u64,
        height: layout.height as u64,
        depth: 1,
    }
}

impl ForeignMetal {
    fn blit(&self, id: u32, pixels: &[u8], layout: PixmapLayout) -> Result<(), GoldyError> {
        layout.validate()?;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.reap();
        let AdapterState { queue, surfaces, .. } = &mut *state;
        let slot = surfaces
            .get_mut(&id)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("foreign Metal surface {id} is gone")))?;
        if slot.dropped {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign Metal surface {id} has been dropped"
            )));
        }
        if layout.width != slot.width || layout.height != slot.height || layout.format != slot.format {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign Metal blit layout {}x{} {:?} does not match surface {}x{} {:?}",
                layout.width,
                layout.height,
                layout.format,
                slot.width,
                slot.height,
                slot.format
            )));
        }
        if pixels.len() < layout.staging_bytes() as usize {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign Metal blit: {} source bytes, need {}",
                pixels.len(),
                layout.staging_bytes()
            )));
        }
        slot.wait();
        let n = layout.staging_bytes() as usize;
        // SAFETY: `contents()` is the shared staging mapping of `staging_size` bytes;
        // exclusive under `AdapterState`'s mutex, and the previous command buffer has completed.
        unsafe {
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), slot.staging.contents() as *mut u8, n);
        }
        let bytes_per_row = layout.row_pitch_bytes();
        let origin = MTLOrigin { x: 0, y: 0, z: 0 };
        let size = copy_size(layout);

        if let Some(layer) = slot.layer {
            let staging = slot.staging.clone();
            let cb = autoreleasepool(|| -> Result<CommandBuffer, GoldyError> {
                let layer = layer as id;
                let drawable: id = unsafe { msg_send![layer, nextDrawable] };
                if drawable == nil {
                    return Err(GoldyError::Backend(anyhow::anyhow!(
                        "foreign Metal: CAMetalLayer nextDrawable returned nil"
                    )));
                }
                let texture_ptr: *mut Object = unsafe { msg_send![drawable, texture] };
                let texture: &mtl::TextureRef = unsafe { &*(texture_ptr as *const mtl::TextureRef) };
                let cb = queue.new_command_buffer().to_owned();
                let blit = cb.new_blit_command_encoder();
                blit.copy_from_buffer_to_texture(
                    &staging,
                    0,
                    bytes_per_row,
                    0,
                    size,
                    texture,
                    0,
                    0,
                    origin,
                    MTLBlitOption::empty(),
                );
                blit.end_encoding();
                let drawable_ref: &mtl::DrawableRef = unsafe { &*(drawable as *const mtl::DrawableRef) };
                cb.present_drawable(drawable_ref);
                cb.commit();
                Ok(cb)
            })?;
            slot.last_cb = Some(cb);
            return Ok(());
        }

        let texture = slot.texture.as_ref().ok_or_else(|| {
            GoldyError::Backend(anyhow::anyhow!("foreign Metal surface {id} has no offscreen texture"))
        })?;
        let cb = queue.new_command_buffer().to_owned();
        let blit = cb.new_blit_command_encoder();
        blit.copy_from_buffer_to_texture(
            &slot.staging,
            0,
            bytes_per_row,
            0,
            size,
            texture,
            0,
            0,
            origin,
            MTLBlitOption::empty(),
        );
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            origin,
            size,
            &slot.staging,
            0,
            bytes_per_row,
            layout.staging_bytes(),
            MTLBlitOption::empty(),
        );
        blit.end_encoding();
        cb.commit();
        slot.last_cb = Some(cb);
        Ok(())
    }

    fn snapshot(&self, id: u32, layout: PixmapLayout) -> Result<Vec<u8>, GoldyError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let slot = state
            .surfaces
            .get_mut(&id)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("foreign Metal surface {id} is gone")))?;
        slot.wait();
        let staging = unsafe { std::slice::from_raw_parts(slot.staging.contents() as *const u8, slot.staging_size) };
        let mut tight = vec![0u8; layout.logical_bytes() as usize];
        layout.unpack_into(&staging[..layout.staging_bytes() as usize], &mut tight)?;
        Ok(tight)
    }

    fn generation(&self, id: u32) -> u64 {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.surfaces.get(&id).map(|s| s.generation).unwrap_or(0)
    }

    fn size(&self, id: u32) -> (u32, u32) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.surfaces.get(&id).map(|s| (s.width, s.height)).unwrap_or((0, 0))
    }

    fn resize(&self, id: u32, width: u32, height: u32) -> Result<(), GoldyError> {
        let width = width.max(1);
        let height = height.max(1);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let AdapterState { device, surfaces, .. } = &mut *state;
        let slot = surfaces
            .get_mut(&id)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("foreign Metal surface {id} is gone")))?;
        if slot.width == width && slot.height == height {
            return Ok(());
        }
        slot.wait();
        let layout = PixmapLayout::tight(width, height, slot.format);
        layout.validate()?;
        if let Some(layer) = slot.layer {
            unsafe {
                let layer = layer as id;
                let size = CGSize::new(width as f64, height as f64);
                let () = msg_send![layer, setDrawableSize: size];
            }
        } else {
            let descriptor = TextureDescriptor::new();
            descriptor.set_width(width as u64);
            descriptor.set_height(height as u64);
            descriptor.set_pixel_format(format_to_mtl(slot.format));
            descriptor.set_storage_mode(MTLStorageMode::Private);
            descriptor.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
            slot.texture = Some(device.new_texture(&descriptor));
        }
        let (staging, staging_size) = alloc_staging(device, layout);
        slot.staging = staging;
        slot.staging_size = staging_size;
        slot.width = width;
        slot.height = height;
        slot.generation = slot.generation.saturating_add(1);
        Ok(())
    }

    fn set_present_mode(&self, id: u32, mode: PresentMode) -> Result<(), GoldyError> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let slot = state
            .surfaces
            .get(&id)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("foreign Metal surface {id} is gone")))?;
        let Some(layer) = slot.layer else {
            return Ok(());
        };
        let sync = match mode {
            PresentMode::Immediate => NO,
            PresentMode::Fifo | PresentMode::Mailbox | PresentMode::Auto => YES,
        };
        unsafe {
            let layer = layer as id;
            let () = msg_send![layer, setDisplaySyncEnabled: sync];
        }
        Ok(())
    }

    fn release(&self, id: u32) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = state.surfaces.get_mut(&id) {
            slot.dropped = true;
        }
        state.reap();
    }
}

impl PixelSink for ForeignSurface {
    fn blit(&self, pixels: &[u8], layout: PixmapLayout) -> Result<(), GoldyError> {
        self.inner.adapter.blit(self.inner.id, pixels, layout)
    }

    fn generation(&self) -> u64 {
        self.inner.adapter.generation(self.inner.id)
    }

    fn size(&self) -> (u32, u32) {
        self.inner.adapter.size(self.inner.id)
    }
}

impl super::WindowSink for ForeignSurface {
    fn resize(&self, width: u32, height: u32) -> Result<(), GoldyError> {
        self.inner.adapter.resize(self.inner.id, width, height)
    }

    fn set_present_mode(&self, mode: PresentMode) -> Result<(), GoldyError> {
        self.inner.adapter.set_present_mode(self.inner.id, mode)
    }
}

impl ForeignSurface {
    /// Tightly packed pixels after the last blit (GPU copy-back, or staging for windowed).
    pub fn snapshot(&self, layout: PixmapLayout) -> Result<Vec<u8>, GoldyError> {
        self.inner.adapter.snapshot(self.inner.id, layout)
    }
}
