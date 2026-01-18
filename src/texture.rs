//! GPU texture management.
//!
//! Textures are GPU images that can be sampled in shaders.
//! Unlike render targets, textures are typically created from CPU data
//! (images, procedural data) and then sampled during rendering.

use crate::backend::{GpuBackend, TextureHandle};
use crate::device::Device;
use crate::types::{TextureFormat, TextureUsage};
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A GPU texture that can be sampled in shaders.
///
/// Textures hold image data on the GPU and can be bound to shaders
/// for sampling operations (e.g., applying textures to 3D models).
pub struct Texture {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: TextureHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
}

impl Texture {
    /// Create a new empty texture.
    ///
    /// The texture is created with uninitialized data. Use `write()` to
    /// upload image data after creation.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device to create the texture on
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    /// * `format` - Pixel format
    /// * `usage` - How the texture will be used
    ///
    /// # Errors
    ///
    /// Returns an error if GPU resource allocation fails.
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Result<Self> {
        let handle = {
            let mut backend = device.backend.lock().unwrap();
            backend.create_texture(device.handle, width, height, format, usage)?
        };

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
            width,
            height,
            format,
        })
    }

    /// Create a texture initialized with data.
    ///
    /// The data must be in the correct format for the texture's pixel format.
    /// For RGBA8 textures, this is 4 bytes per pixel in RGBA order.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device to create the texture on
    /// * `data` - Raw pixel data (must match width * height * bytes_per_pixel)
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    /// * `format` - Pixel format
    /// * `usage` - How the texture will be used
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - GPU resource allocation fails
    /// - Data size doesn't match expected size
    pub fn with_data(
        device: &Device,
        data: &[u8],
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Result<Self> {
        let expected_size = (width * height * format.bytes_per_pixel()) as usize;
        if data.len() != expected_size {
            anyhow::bail!(
                "Data size mismatch: expected {} bytes, got {} bytes",
                expected_size,
                data.len()
            );
        }

        let texture = Self::new(device, width, height, format, usage)?;
        texture.write(data)?;
        Ok(texture)
    }

    /// Write pixel data to the texture.
    ///
    /// The data must match the texture's dimensions and format.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw pixel data (must match width * height * bytes_per_pixel)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data size doesn't match expected size
    /// - GPU upload fails
    pub fn write(&self, data: &[u8]) -> Result<()> {
        let expected_size = (self.width * self.height * self.format.bytes_per_pixel()) as usize;
        if data.len() != expected_size {
            anyhow::bail!(
                "Data size mismatch: expected {} bytes, got {} bytes",
                expected_size,
                data.len()
            );
        }

        let mut backend = self.backend.lock().unwrap();
        backend.write_texture(self.handle, data, self.width, self.height)
    }

    /// Get the width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the texture format.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Get the size of the texture data in bytes.
    pub fn byte_size(&self) -> usize {
        (self.width * self.height * self.format.bytes_per_pixel()) as usize
    }

    /// Get the backend handle for this texture.
    ///
    /// This is used for binding the texture to shaders via bind groups.
    pub fn handle(&self) -> TextureHandle {
        self.handle
    }

    /// Get the texture's index in the global bindless descriptor set.
    ///
    /// Returns `Some(index)` if bindless is enabled and this texture is registered.
    /// Returns `None` otherwise.
    ///
    /// Use this for fully bindless rendering where you pass resource indices
    /// directly via push constants instead of using bind groups.
    pub fn bindless_index(&self) -> Option<u32> {
        let backend = self.backend.lock().unwrap();
        backend.texture_bindless_index(self.handle)
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        if let Ok(mut backend) = self.backend.lock() {
            backend.destroy_texture(self.handle);
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
    fn test_texture_creation() {
        let device = create_test_device();
        let texture = Texture::new(
            &device,
            256,
            256,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        )
        .unwrap();

        assert_eq!(texture.width(), 256);
        assert_eq!(texture.height(), 256);
        assert_eq!(texture.format(), TextureFormat::Rgba8Unorm);
        assert_eq!(texture.byte_size(), 256 * 256 * 4);
    }

    #[test]
    fn test_texture_with_data() {
        let device = create_test_device();

        // Create a 2x2 RGBA texture
        let data = vec![
            255, 0, 0, 255, // Red
            0, 255, 0, 255, // Green
            0, 0, 255, 255, // Blue
            255, 255, 255, 255, // White
        ];

        let texture = Texture::with_data(
            &device,
            &data,
            2,
            2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        )
        .unwrap();

        assert_eq!(texture.width(), 2);
        assert_eq!(texture.height(), 2);
    }

    #[test]
    fn test_texture_with_data_size_mismatch() {
        let device = create_test_device();

        // Data is too small for a 2x2 RGBA texture
        let data = vec![255, 0, 0, 255]; // Only 1 pixel

        let result = Texture::with_data(
            &device,
            &data,
            2,
            2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_texture_write() {
        let device = create_test_device();
        let texture = Texture::new(
            &device,
            2,
            2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        )
        .unwrap();

        let data = vec![0u8; 2 * 2 * 4];
        texture.write(&data).unwrap();
    }

    #[test]
    fn test_texture_write_size_mismatch() {
        let device = create_test_device();
        let texture = Texture::new(
            &device,
            2,
            2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        )
        .unwrap();

        let data = vec![0u8; 4]; // Too small
        let result = texture.write(&data);

        assert!(result.is_err());
    }
}
