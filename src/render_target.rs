//! Render to a render target via [`crate::Scheme::render_pass`] and
//! [`crate::Scheme::submit`], or blit to the swapchain with
//! [`crate::Scheme::copy_to_present`].

use crate::backend::RenderTargetHandle;
use crate::device::Device;
use crate::parcel::ParcelStamp;
use crate::types::*;
use anyhow::Result;
use std::sync::Arc;

/// GPU render target backing a [`crate::Lease<crate::LeaseRenderTarget>`].
pub(crate) struct RenderTarget {
    _device: Device,
    handle: RenderTargetHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    /// Scheme submit / present-easement stamp (WAR against copy-to-present readers).
    stamp: Arc<ParcelStamp>,
}

impl RenderTarget {
    /// Create a new render target with an optional depth buffer.
    pub(crate) fn new_with_depth(
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
            handle,
            width,
            height,
            format: color_format,
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

    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    pub(crate) fn height(&self) -> u32 {
        self.height
    }

    pub(crate) fn format(&self) -> TextureFormat {
        self.format
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
        let target =
            RenderTarget::new_with_depth(&device, 800, 600, TextureFormat::Rgba8Unorm, None).unwrap();

        assert_eq!(target.width(), 800);
        assert_eq!(target.height(), 600);
        assert_eq!(target.format(), TextureFormat::Rgba8Unorm);
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
    }
}
