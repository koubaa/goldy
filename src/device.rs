//! GPU device management.
//!
//! # Thread Safety
//!
//! RAG uses a single-threaded command submission model with lock-free command recording:
//!
//! - **Command Recording**: [`CommandEncoder`](crate::CommandEncoder) is completely lock-free.
//!   You can create and record commands on any thread without any synchronization.
//!   
//! - **Resource Creation**: Creating resources ([`Buffer`](crate::Buffer),
//!   [`RenderPipeline`](crate::RenderPipeline), etc.) acquires the backend lock.
//!   These operations are safe from any thread but serialize internally.
//!
//! - **Command Submission**: Submitting commands via [`RenderTarget::render()`](crate::RenderTarget::render)
//!   or [`SurfaceFrame::render()`](crate::SurfaceFrame::render) acquires the backend lock.
//!
//! ## Best Practices
//!
//! For optimal performance:
//! 1. Create resources during initialization, not per-frame
//! 2. Record commands lock-free using `CommandEncoder` on any thread
//! 3. Submit commands from a single thread (typically the main/render thread)
//!
//! This model is sufficient for most applications. Future versions may add
//! multi-queue support for parallel command submission if needed.

use crate::backend::{self, AdapterInfo, DeviceHandle, GpuBackend};
use crate::types::*;
use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};

/// GPU instance - entry point for RAG.
///
/// Create an instance to enumerate adapters and create devices.
pub struct Instance {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
}

impl Instance {
    /// Create a new RAG instance.
    pub fn new() -> Result<Self> {
        let backend = backend::create_default_backend()?;
        Ok(Self {
            backend: Arc::new(Mutex::new(backend)),
        })
    }

    /// Enumerate available GPU adapters.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        let backend = self.backend.lock().unwrap();
        backend
            .enumerate_adapters()
            .into_iter()
            .map(|info| Adapter { info })
            .collect()
    }

    /// Create a device on the first adapter matching the given type.
    pub fn create_device(&self, preferred_type: DeviceType) -> Result<Device> {
        let adapters = self.enumerate_adapters();
        
        // Find preferred adapter
        let adapter = adapters
            .iter()
            .find(|a| a.info.device_type == preferred_type)
            .or_else(|| adapters.first())
            .context("No GPU adapters available")?;

        self.create_device_for_adapter(adapter.info.id)
    }

    /// Create a device on a specific adapter by ID.
    pub fn create_device_for_adapter(&self, adapter_id: u32) -> Result<Device> {
        let mut backend = self.backend.lock().unwrap();
        let handle = backend.create_device(adapter_id)?;
        
        Ok(Device {
            backend: Arc::clone(&self.backend),
            handle,
            adapter_id,
        })
    }

    /// Get the backend type (Vulkan, Metal, DX12).
    pub fn backend_type(&self) -> BackendType {
        self.backend.lock().unwrap().backend_type()
    }
}

/// Information about a GPU adapter.
#[derive(Debug, Clone)]
pub struct Adapter {
    pub info: AdapterInfo,
}

impl Adapter {
    /// Get the adapter ID.
    pub fn id(&self) -> u32 {
        self.info.id
    }

    /// Get the adapter name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Get the device type.
    pub fn device_type(&self) -> DeviceType {
        self.info.device_type
    }

    /// Get the vendor name.
    pub fn vendor(&self) -> &str {
        &self.info.vendor
    }
}

/// Device capabilities and format preferences.
///
/// Use this to query the optimal formats and limits for your use case.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Preferred format for window surfaces (swapchains).
    /// For windowed apps, use this for `RenderPipelineDesc::target_format`.
    pub preferred_surface_format: TextureFormat,
    
    /// Preferred format for off-screen render targets.
    /// For headless rendering (video encoding, CPU readback), use this format.
    pub preferred_render_target_format: TextureFormat,
    
    /// Formats supported for window surfaces.
    pub supported_surface_formats: Vec<TextureFormat>,
    
    /// Formats supported for render targets.
    pub supported_render_target_formats: Vec<TextureFormat>,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            preferred_surface_format: TextureFormat::Bgra8UnormSrgb,
            preferred_render_target_format: TextureFormat::Rgba8Unorm,
            supported_surface_formats: vec![
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Bgra8Unorm,
            ],
            supported_render_target_formats: vec![
                TextureFormat::Rgba8Unorm,
                TextureFormat::Rgba8UnormSrgb,
                TextureFormat::Bgra8Unorm,
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Rgba16Float,
                TextureFormat::Rgba32Float,
            ],
        }
    }
}

/// A GPU device - used to create resources and render.
///
/// The `Device` is the primary interface for GPU operations. It is `Send + Sync`,
/// so it can be safely shared across threads (typically via `Arc<Device>`).
///
/// # Thread Safety
///
/// Internally, `Device` uses a `Mutex` to serialize backend operations. This means:
/// - Resource creation is thread-safe but serializes internally
/// - Command recording via [`CommandEncoder`](crate::CommandEncoder) is lock-free
/// - Command submission acquires the lock
///
/// See the [module documentation](self) for best practices.
pub struct Device {
    pub(crate) backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: DeviceHandle,
    adapter_id: u32,
}

impl Device {
    /// Get the adapter ID this device was created on.
    pub fn adapter_id(&self) -> u32 {
        self.adapter_id
    }

    /// Check if the device is still valid.
    pub fn is_valid(&self) -> bool {
        self.backend.lock().unwrap().is_device_valid(self.handle)
    }

    /// Get device capabilities and format preferences.
    ///
    /// Use this to query optimal formats for your use case:
    /// - Windowed apps: use `preferred_surface_format` for pipelines
    /// - Headless/streaming: use `preferred_render_target_format` for RenderTarget
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use rag::{Instance, DeviceType};
    ///
    /// let instance = Instance::new()?;
    /// let device = instance.create_device(DeviceType::DiscreteGpu)?;
    /// let caps = device.capabilities();
    /// 
    /// println!("Surface format: {:?}", caps.preferred_surface_format);
    /// println!("RenderTarget format: {:?}", caps.preferred_render_target_format);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn capabilities(&self) -> DeviceCapabilities {
        // For now, return sensible defaults
        // Future: query actual device limits and capabilities
        DeviceCapabilities::default()
    }

    /// Create a device from a backend for testing purposes.
    #[cfg(test)]
    pub(crate) fn from_backend(backend: Box<dyn GpuBackend>) -> anyhow::Result<Self> {
        let backend = Arc::new(Mutex::new(backend));
        let handle = {
            let mut b = backend.lock().unwrap();
            b.create_device(0)?
        };
        Ok(Self {
            backend,
            handle,
            adapter_id: 0,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_device(self.handle);
    }
}

