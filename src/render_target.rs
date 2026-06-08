//! RenderTarget - GPU buffer that stays on GPU with optional CPU readback.
//!
//! Render to a [`RenderTarget`] via [`crate::TaskGraph::render_pass`] and
//! [`crate::RenderPassBuilder::finish_recorded`], then submit with
//! [`crate::Context::submit`] or blit to the swapchain with
//! [`crate::TaskGraph::copy_render_target_to_swapchain`].

use crate::backend::{GpuBackend, RenderTargetHandle};
use crate::device::Device;
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
        })
    }

    /// Opaque backend handle for use with mixed compute+render task graphs.
    #[inline]
    pub(crate) fn backend_handle(&self) -> RenderTargetHandle {
        self.handle
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::task_graph::TaskGraph;

    fn create_test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn graph_clear(device: &Device, target: &RenderTarget, color: Color) {
        let ctx = device.create_context().unwrap();
        let mut graph = TaskGraph::new();
        let mut pass = graph.render_pass("clear", target);
        pass.clear(color);
        pass.finish_recorded();
        graph.dispatch(&ctx).unwrap();
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
    fn test_render_without_readback() {
        let device = create_test_device();
        let target = RenderTarget::new(&device, 100, 100, TextureFormat::Rgba8Unorm).unwrap();
        graph_clear(&device, &target, Color::RED);
    }

    #[test]
    fn test_explicit_readback() {
        let device = create_test_device();
        let target = RenderTarget::new(&device, 2, 2, TextureFormat::Rgba8Unorm).unwrap();
        graph_clear(&device, &target, Color::RED);

        let pixels = target.read_to_cpu().unwrap();
        assert_eq!(pixels.len(), 2 * 2 * 4);
        assert_eq!(pixels[0], 255);
        assert_eq!(pixels[1], 0);
        assert_eq!(pixels[2], 0);
        assert_eq!(pixels[3], 255);
    }

    #[test]
    fn test_read_to_buffer() {
        let device = create_test_device();
        let target = RenderTarget::new(&device, 2, 2, TextureFormat::Rgba8Unorm).unwrap();
        graph_clear(&device, &target, Color::GREEN);

        let mut buffer = vec![0u8; target.buffer_size()];
        target.read_to_buffer(&mut buffer).unwrap();
        assert_eq!(buffer[0], 0);
        assert_eq!(buffer[1], 255);
        assert_eq!(buffer[2], 0);
        assert_eq!(buffer[3], 255);
    }

    #[test]
    fn test_multiple_renders() {
        let device = create_test_device();
        let target = RenderTarget::new(&device, 10, 10, TextureFormat::Rgba8Unorm).unwrap();

        for color in [Color::RED, Color::GREEN, Color::BLUE] {
            graph_clear(&device, &target, color);
        }

        let pixels = target.read_to_cpu().unwrap();
        assert_eq!(pixels[0], 0);
        assert_eq!(pixels[1], 0);
        assert_eq!(pixels[2], 255);
        assert_eq!(pixels[3], 255);
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

    #[test]
    fn test_render_with_depth_clear() {
        let device = create_test_device();
        let target = RenderTarget::new_with_depth(
            &device,
            100,
            100,
            TextureFormat::Rgba8Unorm,
            Some(DepthFormat::Depth32Float),
        )
        .unwrap();

        let ctx = device.create_context().unwrap();
        let mut graph = TaskGraph::new();
        let mut pass = graph.render_pass("depth_clear", &target);
        pass.clear(Color::CORNFLOWER_BLUE);
        pass.clear_depth(1.0);
        pass.finish_recorded();
        graph.dispatch(&ctx).unwrap();
    }
}
