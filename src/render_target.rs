//! RenderTarget - GPU buffer that stays on GPU with optional CPU readback.
//!
//! Render to a [`RenderTarget`] via [`crate::Scheme::render_pass`] and
//! [`crate::Scheme::submit`], or blit to the swapchain with
//! [`crate::Scheme::copy_to_present`].

use crate::backend::{GpuBackend, RenderTargetHandle};
use crate::device::Device;
use crate::parcel::ParcelStamp;
use crate::types::*;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A GPU render target that stays on the GPU until explicitly read.
///
/// `RenderTarget` represents a GPU texture that can be rendered to.
/// Unlike the legacy `FrameOutput`, it does not automatically copy
/// pixels to CPU memory after rendering. This enables efficient
/// multi-consumer scenarios:
///
/// - **Streaming**: GPU encode directly from texture
/// - **Windowing**: Compositor samples texture directly
/// - **CPU processing**: Explicit `read_to_cpu()` when needed
///
/// Optionally includes a depth buffer for 3D rendering with depth testing.
pub struct RenderTarget {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    handle: RenderTargetHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    depth_format: Option<DepthFormat>,
    /// Scheme submit / present-easement stamp (WAR against copy-to-present readers).
    stamp: Arc<ParcelStamp>,
}

impl RenderTarget {
    /// Create a new render target without a depth buffer.
    pub fn new(device: &Device, width: u32, height: u32, format: TextureFormat) -> Result<Self> {
        Self::new_with_depth(device, width, height, format, None)
    }

    /// Create a new render target with an optional depth buffer.
    pub fn new_with_depth(
        device: &Device,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<Self> {
        tracing::debug!(
            width, height,
            color_format = ?color_format,
            ?depth_format,
            "Creating render target"
        );
        let handle = {
            let mut backend = device.inner.backend.lock().unwrap();
            backend.create_render_target_with_depth(device.inner.handle, width, height, color_format, depth_format)?
        };

        Ok(Self {
            _device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            width,
            height,
            format: color_format,
            depth_format,
            stamp: Arc::new(ParcelStamp::new(Arc::downgrade(&device.inner))),
        })
    }

    /// Opaque backend handle for use with mixed compute+render task graphs.
    #[inline]
    pub(crate) fn backend_handle(&self) -> RenderTargetHandle {
        self.handle
    }

    /// Stamp cell for present-easement / cross-submit WAR tracking.
    #[inline]
    pub(crate) fn stamp_handle(&self) -> Arc<ParcelStamp> {
        Arc::clone(&self.stamp)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub fn depth_format(&self) -> Option<DepthFormat> {
        self.depth_format
    }

    pub fn has_depth(&self) -> bool {
        self.depth_format.is_some()
    }

    pub fn buffer_size(&self) -> usize {
        (self.width * self.height * self.format.bytes_per_pixel()) as usize
    }

    pub fn read_to_cpu(&self) -> Result<Vec<u8>> {
        let mut output = vec![0u8; self.buffer_size()];
        self.read_to_buffer(&mut output)?;
        Ok(output)
    }

    pub fn read_to_buffer(&self, output: &mut [u8]) -> Result<()> {
        if output.len() != self.buffer_size() {
            anyhow::bail!(
                "read_to_buffer: expected {} bytes, got {}",
                self.buffer_size(),
                output.len()
            );
        }
        let mut backend = self.backend.lock().unwrap();
        backend.read_target_to_cpu(self.handle, output)
    }
}

impl Drop for RenderTarget {
    fn drop(&mut self) {
        self.stamp.mark_dead();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;

    fn create_test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn test_render_target_creation() {
        let device = create_test_device();
        let target = RenderTarget::new(&device, 800, 600, TextureFormat::Rgba8Unorm).unwrap();

        assert_eq!(target.width(), 800);
        assert_eq!(target.height(), 600);
        assert_eq!(target.format(), TextureFormat::Rgba8Unorm);
        assert_eq!(target.buffer_size(), 800 * 600 * 4);
    }

    #[test]
    fn test_render_target_with_depth() {
        let device = create_test_device();
        let target = RenderTarget::new_with_depth(
            &device,
            800,
            600,
            TextureFormat::Rgba8Unorm,
            Some(DepthFormat::Depth24Plus),
        )
        .unwrap();

        assert_eq!(target.width(), 800);
        assert_eq!(target.height(), 600);
        assert_eq!(target.format(), TextureFormat::Rgba8Unorm);
        assert_eq!(target.depth_format(), Some(DepthFormat::Depth24Plus));
        assert!(target.has_depth());
    }

    #[test]
    fn test_render_target_without_depth() {
        let device = create_test_device();
        let target = RenderTarget::new(&device, 100, 100, TextureFormat::Rgba8Unorm).unwrap();
        assert_eq!(target.depth_format(), None);
        assert!(!target.has_depth());
    }
}
