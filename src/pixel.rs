//! Pixel exchange: a buffer pixmap handed to a foreign graphics object.
//!
//! The CPU device has no textures. Fine raster (and any other host pixmap)
//! writes a tightly packed or pitched buffer parcel. [`PixelExchange`] records a
//! withdraw of that parcel and, on [`PixelClaim::consume`], copies the bytes
//! into a [`PixelSink`].
//!
//! A sink is *not* a Goldy backend. [`HostPixelSink`] is a `Vec<u8>`.
//! [`crate::foreign`] adapters own a process-wide graphics singleton and expose
//! only verbs (`blit`, `resize`, drop) under that lock.

use crate::context::Context;
use crate::error::GoldyError;
use crate::exchange::{MemoryExchange, WithdrawClaim};
use crate::parcel::Parcel;
use crate::scheme::{Scheme, Submission};
use crate::types::TextureFormat;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// CPU pixmap layout for a buffer-shaped fine output.
///
/// `row_pitch == 0` means tightly packed (`width * bytes_per_pixel`). Otherwise
/// `row_pitch` is bytes from the start of one row to the next and must be at
/// least [`Self::tight_row_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixmapLayout {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub row_pitch: u64,
}

impl PixmapLayout {
    /// Tightly packed pixmap of `width × height` in `format`.
    pub fn tight(width: u32, height: u32, format: TextureFormat) -> Self {
        Self {
            width,
            height,
            format,
            row_pitch: 0,
        }
    }

    pub fn bytes_per_pixel(&self) -> u32 {
        self.format.bytes_per_pixel()
    }

    pub fn tight_row_bytes(&self) -> u64 {
        self.width as u64 * u64::from(self.bytes_per_pixel())
    }

    pub fn row_pitch_bytes(&self) -> u64 {
        if self.row_pitch == 0 {
            self.tight_row_bytes()
        } else {
            self.row_pitch
        }
    }

    /// Bytes the source parcel must hold (`height * row_pitch`).
    pub fn staging_bytes(&self) -> u64 {
        self.height as u64 * self.row_pitch_bytes()
    }

    /// Tightly packed pixel bytes (`height * tight_row_bytes`).
    pub fn logical_bytes(&self) -> u64 {
        self.height as u64 * self.tight_row_bytes()
    }

    pub fn validate(&self) -> Result<(), GoldyError> {
        if self.width == 0 || self.height == 0 {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PixmapLayout requires non-zero width and height"
            )));
        }
        if self.row_pitch != 0 && self.row_pitch < self.tight_row_bytes() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PixmapLayout row_pitch {} is smaller than tight row {}",
                self.row_pitch,
                self.tight_row_bytes()
            )));
        }
        Ok(())
    }

    /// Unpack pitched `src` into a tightly packed `dst` of [`Self::logical_bytes`].
    pub fn unpack_into(&self, src: &[u8], dst: &mut [u8]) -> Result<(), GoldyError> {
        self.validate()?;
        let staging = usize::try_from(self.staging_bytes())
            .map_err(|_| GoldyError::Backend(anyhow::anyhow!("PixmapLayout staging size exceeds address space")))?;
        let logical = usize::try_from(self.logical_bytes())
            .map_err(|_| GoldyError::Backend(anyhow::anyhow!("PixmapLayout logical size exceeds address space")))?;
        if src.len() < staging {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "pixmap source is {} bytes, layout staging is {staging}",
                src.len()
            )));
        }
        if dst.len() < logical {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "pixmap destination is {} bytes, layout logical is {logical}",
                dst.len()
            )));
        }
        let row = self.tight_row_bytes() as usize;
        let pitch = self.row_pitch_bytes() as usize;
        for y in 0..self.height as usize {
            let s = y * pitch;
            let d = y * row;
            dst[d..d + row].copy_from_slice(&src[s..s + row]);
        }
        Ok(())
    }
}

/// Foreign destination for a pixmap. Implementations must be safe to call from
/// any thread; they serialise internally (typically one mutex per adapter).
pub trait PixelSink: Send + Sync {
    fn blit(&self, pixels: &[u8], layout: PixmapLayout) -> Result<(), GoldyError>;
    fn generation(&self) -> u64;
    fn size(&self) -> (u32, u32);
}

/// Host-visible pixmap: a `Vec<u8>` behind a mutex.
///
/// Headless CPU debug and tests. Resize is a verb that bumps [`Self::generation`].
pub struct HostPixelSink {
    inner: Mutex<HostInner>,
    generation: AtomicU64,
}

struct HostInner {
    width: u32,
    height: u32,
    format: TextureFormat,
    pixels: Vec<u8>,
}

impl HostPixelSink {
    pub fn new(width: u32, height: u32, format: TextureFormat) -> Result<Self, GoldyError> {
        let layout = PixmapLayout::tight(width, height, format);
        layout.validate()?;
        let logical = usize::try_from(layout.logical_bytes())
            .map_err(|_| GoldyError::Backend(anyhow::anyhow!("HostPixelSink size exceeds address space")))?;
        Ok(Self {
            inner: Mutex::new(HostInner {
                width,
                height,
                format,
                pixels: vec![0u8; logical],
            }),
            generation: AtomicU64::new(1),
        })
    }

    pub fn format(&self) -> TextureFormat {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).format
    }

    /// Tightly packed snapshot of the last successful blit (or zeros).
    pub fn pixels(&self) -> Vec<u8> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).pixels.clone()
    }

    /// Recreate the pixmap. In-flight [`PixelTransaction`] claims become stale.
    pub fn resize(&self, width: u32, height: u32) -> Result<(), GoldyError> {
        let format = self.format();
        let layout = PixmapLayout::tight(width, height, format);
        layout.validate()?;
        let logical = usize::try_from(layout.logical_bytes())
            .map_err(|_| GoldyError::Backend(anyhow::anyhow!("HostPixelSink size exceeds address space")))?;
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.width = width;
        inner.height = height;
        inner.pixels = vec![0u8; logical];
        self.generation.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

impl PixelSink for HostPixelSink {
    fn blit(&self, pixels: &[u8], layout: PixmapLayout) -> Result<(), GoldyError> {
        layout.validate()?;
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if layout.width != inner.width || layout.height != inner.height || layout.format != inner.format {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PixelSink blit layout {}x{} {:?} does not match sink {}x{} {:?}",
                layout.width,
                layout.height,
                layout.format,
                inner.width,
                inner.height,
                inner.format
            )));
        }
        inner.pixels.resize(layout.logical_bytes() as usize, 0);
        layout.unpack_into(pixels, &mut inner.pixels)
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn size(&self) -> (u32, u32) {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        (inner.width, inner.height)
    }
}

/// Buffer pixmap → foreign sink. Scheme submissions stay on the Goldy device;
/// [`PixelClaim::consume`] is the only graphics verb.
pub struct PixelExchange {
    ctx: Context,
    sink: Arc<dyn PixelSink>,
}

impl PixelExchange {
    pub fn new(ctx: &Context, sink: Arc<dyn PixelSink>) -> Self {
        Self { ctx: ctx.clone(), sink }
    }

    /// Bind a buffer parcel as the pixmap source.
    ///
    /// Records a withdraw. The parcel must be a buffer whose byte size equals
    /// `layout.staging_bytes()`. Texture parcels are rejected: the CPU device
    /// has no textures, and this exchange exists so fine can write a buffer.
    pub fn bind_source(
        &self,
        scheme: &mut Scheme,
        parcel: &Parcel,
        layout: PixmapLayout,
    ) -> Result<PixelTransaction, GoldyError> {
        let _ = &self.ctx;
        layout.validate()?;
        if parcel.texture_handle().is_some() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PixelExchange::bind_source requires a buffer parcel (textures are a graphics-device concern)"
            )));
        }
        if parcel.buffer_handle().is_none() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PixelExchange::bind_source requires a buffer parcel"
            )));
        }
        if parcel.byte_size() != layout.staging_bytes() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PixelExchange source parcel is {} bytes, layout staging is {}",
                parcel.byte_size(),
                layout.staging_bytes()
            )));
        }
        let withdraw = MemoryExchange::new(&self.ctx).bind_withdraw(scheme, parcel)?;
        Ok(PixelTransaction {
            withdraw,
            sink: Arc::clone(&self.sink),
            layout,
            generation: self.sink.generation(),
        })
    }
}

impl std::fmt::Debug for PixelExchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelExchange")
            .field("generation", &self.sink.generation())
            .field("size", &self.sink.size())
            .finish_non_exhaustive()
    }
}

/// Stable pixmap-to-sink relationship recorded in one [`Scheme`].
#[derive(Clone)]
pub struct PixelTransaction {
    withdraw: crate::exchange::WithdrawTransaction,
    sink: Arc<dyn PixelSink>,
    layout: PixmapLayout,
    generation: u64,
}

impl PixelTransaction {
    pub fn layout(&self) -> PixmapLayout {
        self.layout
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Take this submission's withdraw and pair it with the sink.
    ///
    /// Fails when the sink was resized (generation mismatch) after bind.
    pub fn claim(&self, submission: &mut Submission) -> Result<PixelClaim, GoldyError> {
        let current = self.sink.generation();
        if current != self.generation {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PixelTransaction generation {} is stale for sink generation {current}",
                self.generation
            )));
        }
        let withdraw = self.withdraw.claim(submission)?;
        Ok(PixelClaim {
            withdraw: Some(withdraw),
            sink: Arc::clone(&self.sink),
            layout: self.layout,
            generation: self.generation,
        })
    }
}

impl std::fmt::Debug for PixelTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelTransaction")
            .field("layout", &self.layout)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

/// Linear claim: wait for the CPU/GPU withdraw, then blit into the sink.
pub struct PixelClaim {
    withdraw: Option<WithdrawClaim>,
    sink: Arc<dyn PixelSink>,
    layout: PixmapLayout,
    generation: u64,
}

impl PixelClaim {
    /// Wait, read pixmap bytes, blit into the sink. Terminal even on error.
    pub fn consume(mut self) -> Result<(), GoldyError> {
        let withdraw = self
            .withdraw
            .take()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("pixel claim already settled")))?;
        if self.sink.generation() != self.generation {
            let _ = withdraw.discard();
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PixelClaim generation {} is stale for sink generation {}",
                self.generation,
                self.sink.generation()
            )));
        }
        let bytes = withdraw.consume()?;
        self.sink.blit(&bytes, self.layout)
    }

    /// Recycle withdraw staging without blitting.
    pub fn discard(mut self) -> Result<(), GoldyError> {
        if let Some(withdraw) = self.withdraw.take() {
            withdraw.discard()?;
        }
        Ok(())
    }
}

impl Drop for PixelClaim {
    fn drop(&mut self) {
        if let Some(withdraw) = self.withdraw.take() {
            let _ = withdraw.discard();
        }
    }
}

impl std::fmt::Debug for PixelClaim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelClaim")
            .field("settled", &self.withdraw.is_none())
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::device::Device;
    use crate::retained_pool::RetainedPool;
    use crate::types::{TextureFlags, TextureKind};
    use crate::BufferKind;
    use std::sync::Arc;

    fn mock_device() -> Arc<Device> {
        Arc::new(Device::from_backend(Box::new(MockBackend::new())).expect("mock device"))
    }

    #[test]
    fn host_sink_round_trip() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let pixels: Vec<u32> = (0..4).map(|i| 0xFF000000 | i).collect();
        let buf = pool.acquire_buffer_with_data(&pixels, BufferKind::Scattered).unwrap();
        let layout = PixmapLayout::tight(2, 2, TextureFormat::Rgba8Unorm);
        let sink = Arc::new(HostPixelSink::new(2, 2, TextureFormat::Rgba8Unorm).unwrap());
        let exchange = PixelExchange::new(&ctx, sink.clone());
        let mut scheme = Scheme::new(&ctx);
        let tx = exchange.bind_source(&mut scheme, buf.whole(), layout).unwrap();
        let mut submission = scheme.submit().unwrap();
        tx.claim(&mut submission).unwrap().consume().unwrap();
        let out = sink.pixels();
        assert_eq!(out.len(), 16);
        let words: Vec<u32> = bytemuck::cast_slice(&out).to_vec();
        assert_eq!(words, pixels);
    }

    #[test]
    fn bind_source_rejects_texture() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let tex = pool
            .acquire_texture(
                2,
                2,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::COPY_SRC,
                None,
            )
            .unwrap();
        let layout = PixmapLayout::tight(2, 2, TextureFormat::Rgba8Unorm);
        let sink = Arc::new(HostPixelSink::new(2, 2, TextureFormat::Rgba8Unorm).unwrap());
        let exchange = PixelExchange::new(&ctx, sink);
        let mut scheme = Scheme::new(&ctx);
        let err = exchange
            .bind_source(&mut scheme, tex.whole(), layout)
            .expect_err("texture parcel");
        assert!(err.detail().contains("buffer parcel"), "{err:?}");
    }

    #[test]
    fn bind_source_rejects_size_mismatch() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let pixels = vec![0u32; 4];
        let buf = pool.acquire_buffer_with_data(&pixels, BufferKind::Scattered).unwrap();
        let layout = PixmapLayout::tight(4, 2, TextureFormat::Rgba8Unorm);
        let sink = Arc::new(HostPixelSink::new(4, 2, TextureFormat::Rgba8Unorm).unwrap());
        let exchange = PixelExchange::new(&ctx, sink);
        let mut scheme = Scheme::new(&ctx);
        let err = exchange
            .bind_source(&mut scheme, buf.whole(), layout)
            .expect_err("size mismatch");
        assert!(err.detail().contains("bytes"), "{err:?}");
    }

    #[test]
    fn resize_invalidates_claim() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let pixels = vec![0u32; 4];
        let buf = pool.acquire_buffer_with_data(&pixels, BufferKind::Scattered).unwrap();
        let layout = PixmapLayout::tight(2, 2, TextureFormat::Rgba8Unorm);
        let sink = Arc::new(HostPixelSink::new(2, 2, TextureFormat::Rgba8Unorm).unwrap());
        let exchange = PixelExchange::new(&ctx, sink.clone());
        let mut scheme = Scheme::new(&ctx);
        let tx = exchange.bind_source(&mut scheme, buf.whole(), layout).unwrap();
        sink.resize(4, 4).unwrap();
        let mut submission = scheme.submit().unwrap();
        let err = tx.claim(&mut submission).expect_err("stale generation");
        assert!(err.detail().contains("stale"), "{err:?}");
    }

    #[test]
    fn discard_does_not_blit() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let pixels: Vec<u32> = vec![0x11223344; 4];
        let buf = pool.acquire_buffer_with_data(&pixels, BufferKind::Scattered).unwrap();
        let layout = PixmapLayout::tight(2, 2, TextureFormat::Rgba8Unorm);
        let sink = Arc::new(HostPixelSink::new(2, 2, TextureFormat::Rgba8Unorm).unwrap());
        let exchange = PixelExchange::new(&ctx, sink.clone());
        let mut scheme = Scheme::new(&ctx);
        let tx = exchange.bind_source(&mut scheme, buf.whole(), layout).unwrap();
        let mut submission = scheme.submit().unwrap();
        tx.claim(&mut submission).unwrap().discard().unwrap();
        assert!(sink.pixels().iter().all(|&b| b == 0));
    }

    #[test]
    fn pitched_layout_unpacks() {
        let layout = PixmapLayout {
            width: 2,
            height: 2,
            format: TextureFormat::Rgba8Unorm,
            row_pitch: 16,
        };
        assert_eq!(layout.tight_row_bytes(), 8);
        assert_eq!(layout.staging_bytes(), 32);
        let mut src = vec![0u8; 32];
        src[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        src[16..24].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
        let mut dst = vec![0u8; 16];
        layout.unpack_into(&src, &mut dst).unwrap();
        assert_eq!(&dst[..], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    }
}
