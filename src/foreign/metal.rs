//! Metal as a foreign graphics object: no Goldy device, verbs under one lock.
//!
//! Creates its own `MTLDevice` / `MTLCommandQueue`. Offscreen surfaces hold a
//! private texture plus a shared staging buffer. [`ForeignSurface::blit`] copies
//! host pixels through `MTLBlitCommandEncoder` and a copy-back into the shared
//! buffer so [`ForeignSurface::snapshot`] can assert GPU contents.
//!
//! Windowed `CAMetalLayer` present is a later verb on this same singleton.

use crate::backend::metal::format_to_mtl;
use crate::pixel::{PixelSink, PixmapLayout};
use crate::types::TextureFormat;
use crate::GoldyError;
use metal as mtl;
use mtl::{
    Buffer, CommandBuffer, CommandQueue, Device, MTLBlitOption, MTLOrigin, MTLResourceOptions, MTLSize, MTLStorageMode,
    MTLTextureUsage, Texture, TextureDescriptor,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

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
    texture: Texture,
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

/// Offscreen Metal texture owned by the foreign singleton.
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
        let slot = SurfaceSlot::create(&state.device, layout)?;
        let id = state.next_id;
        state.next_id += 1;
        state.surfaces.insert(id, slot);
        Ok(ForeignSurface {
            inner: Arc::new(SurfaceHandle {
                adapter: Arc::clone(self),
                id,
            }),
        })
    }
}

impl AdapterState {
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
    fn create(device: &Device, layout: PixmapLayout) -> Result<Self, GoldyError> {
        let descriptor = TextureDescriptor::new();
        descriptor.set_width(layout.width as u64);
        descriptor.set_height(layout.height as u64);
        descriptor.set_pixel_format(format_to_mtl(layout.format));
        descriptor.set_storage_mode(MTLStorageMode::Private);
        descriptor.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let texture = device.new_texture(&descriptor);
        let staging_size = layout.staging_bytes().max(1) as usize;
        let staging = device.new_buffer(staging_size as u64, MTLResourceOptions::StorageModeShared);
        Ok(Self {
            width: layout.width,
            height: layout.height,
            format: layout.format,
            generation: 1,
            texture,
            staging,
            staging_size,
            last_cb: None,
            dropped: false,
        })
    }

    fn destroy(self) {
        if let Some(cb) = self.last_cb {
            cb.wait_until_completed();
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
        let cb = queue.new_command_buffer().to_owned();
        let blit = cb.new_blit_command_encoder();
        let bytes_per_row = layout.row_pitch_bytes();
        blit.copy_from_buffer_to_texture(
            &slot.staging,
            0,
            bytes_per_row,
            0,
            copy_size(layout),
            &slot.texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLBlitOption::empty(),
        );
        blit.copy_from_texture_to_buffer(
            &slot.texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            copy_size(layout),
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

impl ForeignSurface {
    /// Tightly packed pixels after the last blit (GPU copy-back).
    pub fn snapshot(&self, layout: PixmapLayout) -> Result<Vec<u8>, GoldyError> {
        self.inner.adapter.snapshot(self.inner.id, layout)
    }
}
