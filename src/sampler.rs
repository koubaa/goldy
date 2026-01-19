//! GPU sampler management.
//!
//! Samplers define how textures are sampled in shaders, including
//! filtering modes and addressing (wrapping) behavior.

use crate::backend::{GpuBackend, SamplerHandle};
use crate::device::Device;
use crate::types::SamplerDesc;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A GPU sampler for texture sampling.
///
/// Samplers control how texture data is read in shaders:
/// - **Filtering**: How to interpolate between texels (nearest or linear)
/// - **Addressing**: What happens when UVs are outside [0, 1] (clamp, repeat, mirror)
///
/// # Example
///
/// ```rust,no_run
/// use goldy::{Device, Sampler, SamplerDesc, FilterMode, AddressMode};
///
/// fn create_sampler(device: &Device) -> anyhow::Result<Sampler> {
///     Sampler::new(device, &SamplerDesc {
///         mag_filter: FilterMode::Linear,
///         min_filter: FilterMode::Linear,
///         address_mode_u: AddressMode::Repeat,
///         address_mode_v: AddressMode::Repeat,
///         ..Default::default()
///     })
/// }
/// ```
pub struct Sampler {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: SamplerHandle,
}

impl Sampler {
    /// Create a new sampler with the specified settings.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device to create the sampler on
    /// * `desc` - Sampler settings (filtering, addressing)
    ///
    /// # Errors
    ///
    /// Returns an error if GPU resource allocation fails.
    pub fn new(device: &Device, desc: &SamplerDesc) -> Result<Self> {
        let handle = {
            let mut backend = device.backend.lock().unwrap();
            backend.create_sampler(device.handle, desc)?
        };

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
        })
    }

    /// Create a sampler with default settings (nearest filtering, clamp to edge).
    pub fn default_sampler(device: &Device) -> Result<Self> {
        Self::new(device, &SamplerDesc::default())
    }

    /// Create a sampler with linear filtering and clamp to edge addressing.
    ///
    /// This is a common configuration for smooth texture sampling.
    pub fn linear(device: &Device) -> Result<Self> {
        use crate::types::FilterMode;
        Self::new(
            device,
            &SamplerDesc {
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Linear,
                ..Default::default()
            },
        )
    }

    /// Create a sampler with nearest filtering and clamp to edge addressing.
    ///
    /// This preserves hard pixel edges (useful for pixel art).
    pub fn nearest(device: &Device) -> Result<Self> {
        use crate::types::FilterMode;
        Self::new(
            device,
            &SamplerDesc {
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                mipmap_filter: FilterMode::Nearest,
                ..Default::default()
            },
        )
    }

    /// Create a sampler with linear filtering and repeat addressing.
    ///
    /// This is common for tiling textures.
    pub fn linear_repeat(device: &Device) -> Result<Self> {
        use crate::types::{AddressMode, FilterMode};
        Self::new(
            device,
            &SamplerDesc {
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Linear,
                address_mode_u: AddressMode::Repeat,
                address_mode_v: AddressMode::Repeat,
                address_mode_w: AddressMode::Repeat,
                max_anisotropy: 1.0,
                compare: None,
                lod_min_clamp: 0.0,
                lod_max_clamp: 32.0,
            },
        )
    }

    /// Get the backend handle for this sampler.
    pub fn handle(&self) -> SamplerHandle {
        self.handle
    }

    /// Get the sampler's index in the global descriptor set.
    ///
    /// Returns `Some(index)` if this sampler is registered.
    /// Returns `None` otherwise.
    pub fn bindless_index(&self) -> Option<u32> {
        let backend = self.backend.lock().unwrap();
        backend.sampler_bindless_index(self.handle)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        if let Ok(mut backend) = self.backend.lock() {
            backend.destroy_sampler(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::types::{AddressMode, FilterMode};

    fn create_test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn test_sampler_creation() {
        let device = create_test_device();
        let _sampler = Sampler::new(&device, &SamplerDesc::default()).unwrap();
    }

    #[test]
    fn test_sampler_default() {
        let device = create_test_device();
        let _sampler = Sampler::default_sampler(&device).unwrap();
    }

    #[test]
    fn test_sampler_linear() {
        let device = create_test_device();
        let _sampler = Sampler::linear(&device).unwrap();
    }

    #[test]
    fn test_sampler_nearest() {
        let device = create_test_device();
        let _sampler = Sampler::nearest(&device).unwrap();
    }

    #[test]
    fn test_sampler_linear_repeat() {
        let device = create_test_device();
        let _sampler = Sampler::linear_repeat(&device).unwrap();
    }

    #[test]
    fn test_sampler_custom() {
        let device = create_test_device();
        let desc = SamplerDesc {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::MirrorRepeat,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        };
        let _sampler = Sampler::new(&device, &desc).unwrap();
    }
}
