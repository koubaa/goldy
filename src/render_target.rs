//! RenderTarget - GPU buffer that stays on GPU with optional CPU readback.
//!
//! This module provides the primary rendering abstraction for Goldy.
//! Unlike the legacy `FrameOutput` which always copies to CPU, `RenderTarget`
//! keeps the rendered image on the GPU until explicitly requested.
//!
//! # Example
//!
//! ```rust,no_run
//! use goldy::{Device, RenderTarget, CommandEncoder, TextureFormat};
//!
//! fn render_frame(device: &Device) -> anyhow::Result<()> {
//!     // Create a render target (GPU-only by default)
//!     let target = RenderTarget::new(device, 800, 600, TextureFormat::Rgba8Unorm)?;
//!     
//!     // Build render commands
//!     let mut encoder = CommandEncoder::new();
//!     {
//!         let mut pass = encoder.begin_render_pass();
//!         pass.clear(goldy::Color::CORNFLOWER_BLUE);
//!         // ... more rendering ...
//!     }
//!     
//!     // Render to target (stays on GPU)
//!     target.render(encoder)?;
//!     
//!     // Only read to CPU when actually needed
//!     let pixels = target.read_to_cpu()?;
//!     
//!     Ok(())
//! }
//! ```

use crate::backend::{GpuBackend, RenderTargetHandle};
use crate::device::Device;
use crate::encoder::CommandEncoder;
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
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    device_handle: u64,
    handle: RenderTargetHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    depth_format: Option<DepthFormat>,
}

impl RenderTarget {
    /// Create a new render target without a depth buffer.
    ///
    /// This allocates GPU resources for the render target but does not
    /// allocate any CPU-side staging buffers until `read_to_cpu()` is called.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device to create the render target on
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels  
    /// * `format` - Pixel format for the render target
    ///
    /// # Errors
    ///
    /// Returns an error if GPU resource allocation fails.
    pub fn new(device: &Device, width: u32, height: u32, format: TextureFormat) -> Result<Self> {
        Self::new_with_depth(device, width, height, format, None)
    }

    /// Create a new render target with an optional depth buffer.
    ///
    /// This allocates GPU resources for the color buffer and optionally a depth buffer.
    /// Use this for 3D rendering that requires depth testing.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device to create the render target on
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels  
    /// * `color_format` - Pixel format for the color buffer
    /// * `depth_format` - Optional depth buffer format (None = no depth buffer)
    ///
    /// # Errors
    ///
    /// Returns an error if GPU resource allocation fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use goldy::{Device, RenderTarget, TextureFormat, DepthFormat};
    ///
    /// fn create_3d_target(device: &Device) -> anyhow::Result<RenderTarget> {
    ///     RenderTarget::new_with_depth(
    ///         device, 1920, 1080,
    ///         TextureFormat::Rgba8Unorm,
    ///         Some(DepthFormat::Depth24Plus),
    ///     )
    /// }
    /// ```
    pub fn new_with_depth(
        device: &Device,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<Self> {
        let handle = {
            let mut backend = device.backend.lock().unwrap();
            backend.create_render_target_with_depth(
                device.handle,
                width,
                height,
                color_format,
                depth_format,
            )?
        };

        Ok(Self {
            backend: Arc::clone(&device.backend),
            device_handle: device.handle,
            handle,
            width,
            height,
            format: color_format,
            depth_format,
        })
    }

    /// Get the width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the color texture format.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Get the depth buffer format, if any.
    pub fn depth_format(&self) -> Option<DepthFormat> {
        self.depth_format
    }

    /// Returns true if this render target has a depth buffer.
    pub fn has_depth(&self) -> bool {
        self.depth_format.is_some()
    }

    /// Get the size of the pixel data in bytes.
    pub fn buffer_size(&self) -> usize {
        (self.width * self.height * self.format.bytes_per_pixel()) as usize
    }

    /// Render commands to this target.
    ///
    /// This executes the render commands and stores the result in the GPU texture.
    /// The data stays on the GPU - no CPU copy occurs.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder containing render commands
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, encoder: CommandEncoder) -> Result<()> {
        let commands = encoder.finish();
        let mut backend = self.backend.lock().unwrap();
        backend.render_to_target(self.device_handle, self.handle, &commands)
    }

    /// Read the rendered pixels to a CPU buffer.
    ///
    /// This performs a GPU-to-CPU copy, which may stall the pipeline.
    /// Only call this when you actually need the pixel data on the CPU.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No render has been performed yet
    /// - The GPU-to-CPU copy fails
    pub fn read_to_cpu(&self) -> Result<Vec<u8>> {
        let mut output = vec![0u8; self.buffer_size()];
        self.read_to_buffer(&mut output)?;
        Ok(output)
    }

    /// Read the rendered pixels into an existing buffer.
    ///
    /// This is more efficient than `read_to_cpu()` when you want to
    /// reuse an existing buffer to avoid allocation.
    ///
    /// # Arguments
    ///
    /// * `output` - Buffer to write pixel data into. Must be at least
    ///   `buffer_size()` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The buffer is too small
    /// - No render has been performed yet
    /// - The GPU-to-CPU copy fails
    pub fn read_to_buffer(&self, output: &mut [u8]) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.read_target_to_cpu(self.handle, output)
    }
}

impl Drop for RenderTarget {
    fn drop(&mut self) {
        if let Ok(mut backend) = self.backend.lock() {
            backend.destroy_render_target(self.handle);
        }
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
    fn test_render_without_readback() {
        let device = create_test_device();
        let target = RenderTarget::new(&device, 100, 100, TextureFormat::Rgba8Unorm).unwrap();

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::RED);
        }

        // This should succeed without any CPU readback
        target.render(encoder).unwrap();

        // The test validates that render() works without requiring read_to_cpu()
    }

    #[test]
    fn test_explicit_readback() {
        let device = create_test_device();
        let target = RenderTarget::new(&device, 2, 2, TextureFormat::Rgba8Unorm).unwrap();

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::RED);
        }

        target.render(encoder).unwrap();

        // Now explicitly read
        let pixels = target.read_to_cpu().unwrap();

        assert_eq!(pixels.len(), 2 * 2 * 4);
        // Red color: R=255, G=0, B=0, A=255
        assert_eq!(pixels[0], 255);
        assert_eq!(pixels[1], 0);
        assert_eq!(pixels[2], 0);
        assert_eq!(pixels[3], 255);
    }

    #[test]
    fn test_read_to_buffer() {
        let device = create_test_device();
        let target = RenderTarget::new(&device, 2, 2, TextureFormat::Rgba8Unorm).unwrap();

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::GREEN);
        }

        target.render(encoder).unwrap();

        let mut buffer = vec![0u8; target.buffer_size()];
        target.read_to_buffer(&mut buffer).unwrap();

        // Green color: R=0, G=255, B=0, A=255
        assert_eq!(buffer[0], 0);
        assert_eq!(buffer[1], 255);
        assert_eq!(buffer[2], 0);
        assert_eq!(buffer[3], 255);
    }

    #[test]
    fn test_multiple_renders() {
        let device = create_test_device();
        let target = RenderTarget::new(&device, 10, 10, TextureFormat::Rgba8Unorm).unwrap();

        // Render multiple times to the same target
        for color in [Color::RED, Color::GREEN, Color::BLUE] {
            let mut encoder = CommandEncoder::new();
            {
                let mut pass = encoder.begin_render_pass();
                pass.clear(color);
            }
            target.render(encoder).unwrap();
        }

        // Final read should show blue
        let pixels = target.read_to_cpu().unwrap();
        assert_eq!(pixels[0], 0); // R
        assert_eq!(pixels[1], 0); // G
        assert_eq!(pixels[2], 255); // B
        assert_eq!(pixels[3], 255); // A
    }

    // Depth buffer tests
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

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::CORNFLOWER_BLUE);
            pass.clear_depth(1.0); // Clear depth to far plane
        }

        target.render(encoder).unwrap();

        // Should succeed without errors
    }
}
