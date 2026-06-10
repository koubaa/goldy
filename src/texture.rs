//! GPU texture management.
//!
//! Textures are GPU images that can be sampled in shaders.
//! Unlike render targets, textures are typically created from CPU data
//! (images, procedural data) and then sampled during rendering.

use crate::backend::{GpuBackend, TextureHandle};
use crate::device::Device;
use crate::types::{ResourceAccess, ResourceCategory, ResourceHandle, TextureFlags, TextureFormat, TextureKind};
use crate::vram_allocator::{ParcelDeed, ParcelType};
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A GPU texture that can be sampled in shaders.
///
/// Textures hold image data on the GPU and can be bound to shaders
/// for sampling operations (e.g., applying textures to 3D models).
#[derive(Clone)]
pub struct Texture {
    _device: Option<Device>,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: TextureHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: TextureKind,
    flags: TextureFlags,
    owned: bool,
    /// Accounting deed for observer + allocator notification on drop.
    deed: Option<ParcelDeed>,
}

impl Texture {
    /// Attach the accounting deed (called from [`Device::alloc_texture`] only).
    pub(crate) fn set_deed(&mut self, deed: ParcelDeed) {
        self.deed = Some(deed);
    }

    /// Create a new empty texture with the specified access pattern.
    ///
    /// The texture is created with uninitialized data. Use `write()` to
    /// upload image data after creation.
    ///
    /// # Access Patterns
    ///
    /// - `TextureKind::Interpolated`: Hardware filtering between neighbors (texture units).
    ///   Use for textures sampled with bilinear/trilinear filtering.
    ///
    /// - `TextureKind::Direct`: Direct 2D indexing without filtering.
    ///   Use for storage images, compute output, or when you need exact pixel values.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device to create the texture on
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    /// * `format` - Pixel format
    /// * `access` - Spatial access pattern
    /// * `flags` - Additional texture flags (copy operations, render target)
    ///
    /// # Errors
    ///
    /// Returns an error if GPU resource allocation fails.
    pub(crate) fn new(
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<Self> {
        tracing::debug!(width, height, ?format, ?access, ?flags, "Creating texture");
        let handle = {
            let mut backend = device.inner.backend.lock().unwrap();
            backend.create_texture(device.inner.handle, width, height, format, access, flags)?
        };

        Ok(Self {
            _device: Some(device.clone()),
            backend: Arc::clone(&device.inner.backend),
            handle,
            width,
            height,
            format,
            access,
            flags,
            owned: true,
            deed: None,
        })
    }

    /// Create a texture initialized with data.
    ///
    /// The data must be in the correct format for the texture's pixel format.
    /// For RGBA8 textures, this is 4 bytes per pixel in RGBA order.
    ///
    /// See `Device::alloc_texture` for access pattern documentation.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device to create the texture on
    /// * `data` - Raw pixel data (must match width * height * bytes_per_pixel)
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    /// * `format` - Pixel format
    /// * `access` - Spatial access pattern
    /// * `flags` - Additional texture flags
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
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<Self> {
        let expected_size = (width * height * format.bytes_per_pixel()) as usize;
        if data.len() != expected_size {
            anyhow::bail!(
                "Data size mismatch: expected {} bytes, got {} bytes",
                expected_size,
                data.len()
            );
        }

        let texture = Self::new(device, width, height, format, access, flags)?;
        #[allow(deprecated)]
        texture.write(data)?;
        Ok(texture)
    }

    /// Write pixel data to a subregion of the texture.
    ///
    /// The data must match the specified width and height for the texture's format.
    /// The region must fit within the texture bounds.
    ///
    /// # Arguments
    ///
    /// * `x` - Left offset in pixels
    /// * `y` - Top offset in pixels
    /// * `width` - Width of the region in pixels
    /// * `height` - Height of the region in pixels
    /// * `data` - Raw pixel data (must match width * height * bytes_per_pixel)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Region is out of bounds
    /// - Data size doesn't match expected size
    /// - GPU upload fails
    #[deprecated(
        since = "0.1.0",
        note = "Use TaskGraph::write_texture_region() for batched, non-blocking uploads. \
                This method submits synchronously and stalls the GPU."
    )]
    pub fn write_region(&self, x: u32, y: u32, width: u32, height: u32, data: &[u8]) -> Result<()> {
        if x + width > self.width || y + height > self.height {
            anyhow::bail!(
                "Region out of bounds: {}x{} at ({},{}) exceeds {}x{} texture",
                width,
                height,
                x,
                y,
                self.width,
                self.height
            );
        }
        let expected_size = (width * height * self.format.bytes_per_pixel()) as usize;
        if data.len() != expected_size {
            anyhow::bail!(
                "Data size mismatch: expected {} bytes for {}x{} region, got {}",
                expected_size,
                width,
                height,
                data.len()
            );
        }
        let mut backend = self.backend.lock().unwrap();
        backend.write_texture_region(self.handle, x, y, width, height, data)
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
    #[deprecated(
        since = "0.1.0",
        note = "Use TaskGraph::write_texture() for batched, non-blocking uploads. \
                This method submits synchronously and stalls the GPU."
    )]
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
        let bytes = u64::from(self.width) * u64::from(self.height) * u64::from(self.format.bytes_per_pixel());
        usize::try_from(bytes).unwrap_or(usize::MAX)
    }

    /// Read texture contents to CPU memory.
    ///
    /// The texture must have been created with [`TextureFlags::COPY_SRC`].
    /// The output slice must be at least [`byte_size()`](Self::byte_size) bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Output buffer is too small
    /// - Texture was not created with COPY_SRC
    /// - GPU readback fails
    pub fn read_to_cpu(&self, output: &mut [u8]) -> Result<()> {
        let expected_size = self.byte_size();
        if output.len() < expected_size {
            anyhow::bail!("Output buffer too small: {} < {}", output.len(), expected_size);
        }
        let mut backend = self.backend.lock().unwrap();
        backend.read_texture_to_cpu(self.handle, output)
    }

    /// Get the backend handle for this texture.
    pub fn gpu_handle(&self) -> TextureHandle {
        self.handle
    }

    /// Resource descriptor index for how this texture will be accessed in the current dispatch.
    pub fn resource_index(&self, access: ResourceAccess) -> Option<u32> {
        let backend = self.backend.lock().unwrap();
        match (self.access, access) {
            (TextureKind::Interpolated, ResourceAccess::Read) => backend.texture_bindless_index(self.handle),
            (TextureKind::Interpolated, ResourceAccess::Write | ResourceAccess::ReadWrite) => None,
            (TextureKind::Direct, ResourceAccess::Read) => None,
            (TextureKind::Direct, ResourceAccess::Write | ResourceAccess::ReadWrite) => {
                backend.texture_bindless_index(self.handle)
            }
            (TextureKind::DirectInterpolated, ResourceAccess::Read) => {
                backend.texture_bindless_sampled_index(self.handle)
            }
            (TextureKind::DirectInterpolated, ResourceAccess::Write | ResourceAccess::ReadWrite) => {
                backend.texture_bindless_index(self.handle)
            }
        }
    }

    /// Typed resource descriptor handle for validation and dispatch wiring.
    pub fn handle(&self, access: ResourceAccess) -> Option<ResourceHandle> {
        self.resource_index(access).map(|i| {
            let category = match (self.access, access) {
                (TextureKind::DirectInterpolated, ResourceAccess::Read) => ResourceCategory::Texture,
                _ => ResourceCategory::from(self.access),
            };
            ResourceHandle::new(category, i)
        })
    }

    /// Get the access pattern this texture was created with.
    pub fn access(&self) -> TextureKind {
        self.access
    }
    /// Creation flags ([`TextureFlags`]) used when this texture was allocated.
    ///
    /// Views from [`Self::borrow`] keep the parent's flags. Non-owning textures
    /// that wrap externally owned GPU images (such as swapchain drawables)
    /// report [`TextureFlags::empty()`].
    pub fn flags(&self) -> TextureFlags {
        self.flags
    }

    /// Whether dropping this texture destroys the GPU resource (`true`) or not (`false`).
    ///
    /// Borrowed textures ([`Self::borrow`]) and other non-owning views of
    /// externally managed resources return `false`.
    pub fn is_owned(&self) -> bool {
        self.owned
    }

    /// Create a non-owning view of this texture.
    ///
    /// The returned `Texture` shares the same GPU resource and handle but does
    /// **not** destroy the underlying resource when dropped. Use this when you
    /// need to hand a reference into a system (e.g. a bind map) that may drop
    /// it before the original owner is done — for example to avoid a
    /// use-after-free when the bind map entry is evicted while the caller still
    /// holds the original `Texture`.
    pub fn borrow(&self) -> Self {
        Self {
            _device: self._device.clone(),
            backend: Arc::clone(&self.backend),
            handle: self.handle,
            width: self.width,
            height: self.height,
            format: self.format,
            access: self.access,
            flags: self.flags,
            owned: false,
            deed: None,
        }
    }

    /// Create a borrowed texture wrapping an externally-owned GPU resource.
    ///
    /// The returned `Texture` provides the same read/query API but does **not**
    /// destroy the underlying resource when dropped. Used for transient resources
    /// like surface frame drawables whose lifetime is managed elsewhere.
    ///
    /// Swapchain drawables on surfaces with compute-to-surface support are
    /// writable, so we tag them as `TextureKind::Direct` (storage image).
    pub(crate) fn borrowed(
        backend: Arc<Mutex<Box<dyn GpuBackend>>>,
        handle: TextureHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Self {
        Self {
            _device: None,
            backend,
            handle,
            width,
            height,
            format,
            access: TextureKind::Direct,
            flags: TextureFlags::empty(),
            owned: false,
            deed: None,
        }
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        tracing::trace!(
            width = self.width,
            height = self.height,
            format = ?self.format,
            "Destroying texture"
        );
        if let Ok(mut backend) = self.backend.lock() {
            backend.destroy_texture(self.handle);
        }
        if let Some(deed) = self.deed.as_ref() {
            let byte_size = self.byte_size() as u64;
            deed.notify_freed(byte_size, byte_size, ParcelType::Texture);
        }
    }
}

#[cfg(test)]
#[allow(deprecated)]
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
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
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
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
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
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
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
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
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
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
        )
        .unwrap();

        let data = vec![0u8; 4]; // Too small
        let result = texture.write(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_texture_r8_unorm() {
        let device = create_test_device();
        let data = vec![128u8; 64 * 64]; // 1 byte per pixel
        let texture = Texture::with_data(
            &device,
            &data,
            64,
            64,
            TextureFormat::R8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
        )
        .unwrap();

        assert_eq!(texture.width(), 64);
        assert_eq!(texture.height(), 64);
        assert_eq!(texture.format(), TextureFormat::R8Unorm);
        assert_eq!(texture.byte_size(), 64 * 64);
    }

    #[test]
    fn test_texture_rg8_unorm() {
        let device = create_test_device();
        let data = vec![0u8; 32 * 32 * 2]; // 2 bytes per pixel
        let texture = Texture::with_data(
            &device,
            &data,
            32,
            32,
            TextureFormat::Rg8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
        )
        .unwrap();

        assert_eq!(texture.width(), 32);
        assert_eq!(texture.height(), 32);
        assert_eq!(texture.format(), TextureFormat::Rg8Unorm);
        assert_eq!(texture.byte_size(), 32 * 32 * 2);
    }

    #[test]
    fn test_texture_r8_data_size_validation() {
        let device = create_test_device();
        let data = vec![0u8; 100]; // Wrong size for 64x64 R8
        let result = Texture::with_data(
            &device,
            &data,
            64,
            64,
            TextureFormat::R8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
        );
        assert!(result.is_err());
    }
}
