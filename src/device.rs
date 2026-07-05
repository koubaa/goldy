//! GPU device management.
//!
//! # Thread Safety
//!
//! Goldy uses a single-threaded command submission model with lock-free command recording:
//!
//! - **Graph Recording**: [`RenderPassBuilder`](crate::RenderPassBuilder) and [`TaskGraph`](crate::TaskGraph)
//!   record commands without touching the GPU backend. You can build graphs on any thread.
//!   
//! - **Resource Creation**: Creating resources ([`crate::Buffer`],
//!   [`RenderPipeline`](crate::RenderPipeline), etc.) acquires the backend lock.
//!   These operations are safe from any thread but serialize internally.
//!
//! - **Command Submission**: Submitting via [`TaskGraph::dispatch`](crate::TaskGraph::dispatch) or
//!   [`Surface::submit_graph_to_frame`](crate::Surface::submit_graph_to_frame) acquires the backend lock.
//!
//! ## Best Practices
//!
//! For optimal performance:
//! 1. Create resources during initialization, not per-frame
//! 2. Build task graphs on any thread; declare buffer/parcel dependencies on each node
//! 3. Submit graphs from a single thread (typically the main/render thread)
//!
//! This model is sufficient for most applications. Future versions may add
//! multi-queue support for parallel command submission if needed.

use crate::backend::{self, AdapterInfo, DeviceHandle, GpuBackend};
use crate::error::GoldyError;
use crate::shader_library::ShaderLibrary;
use crate::slang::{ShaderTarget, SlangCompiler, StructLayout};
use crate::timeline::TimelineValue;
use crate::types::*;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Unique ID generator for temp directories
static REGISTRY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// GPU instance - entry point for Goldy.
///
/// Create an instance to enumerate adapters and create devices.
pub struct Instance {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
}

impl Instance {
    /// Create a new Goldy instance.
    pub fn new() -> Result<Self> {
        let backend = backend::create_shared_backend()?;
        let backend_type = backend.lock().unwrap().backend_type();
        tracing::info!(?backend_type, "Goldy instance created");
        Ok(Self { backend })
    }

    fn adapter_from_info(&self, info: AdapterInfo) -> Adapter {
        let caps = self.backend.lock().unwrap().adapter_capabilities(info.id);
        Adapter {
            inner: Arc::new(AdapterInner {
                backend: Arc::clone(&self.backend),
                info,
                caps,
            }),
        }
    }

    /// Enumerate available GPU adapters.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        let infos = self.backend.lock().unwrap().enumerate_adapters();
        let adapters: Vec<Adapter> = infos.into_iter().map(|info| self.adapter_from_info(info)).collect();
        tracing::debug!(count = adapters.len(), "Enumerated GPU adapters");
        for adapter in &adapters {
            tracing::debug!(
                id = adapter.inner.info.id,
                name = %adapter.inner.info.name,
                vendor = %adapter.inner.info.vendor,
                device_type = ?adapter.inner.info.device_type,
                "  adapter"
            );
        }
        adapters
    }

    /// Request an adapter matching the given options (wgpu-style).
    pub fn request_adapter(&self, opts: &RequestAdapterOptions) -> Result<Adapter> {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            if self.backend_type() == BackendType::Dx12 && opts.force_fallback_adapter {
                tracing::info!("Using WARP fallback adapter");
                return self.adapter_for_id(crate::backend::dx12::WARP_ADAPTER_ID);
            }
        }

        tracing::info!(?opts.power_preference, "Requesting GPU adapter");
        let adapters = self.enumerate_adapters();
        anyhow::ensure!(!adapters.is_empty(), "No GPU adapters available");

        let adapter = match opts.power_preference {
            PowerPreference::HighPerformance => adapters
                .iter()
                .find(|a| a.device_type() == DeviceType::DiscreteGpu)
                .or_else(|| adapters.iter().find(|a| a.device_type() == DeviceType::IntegratedGpu))
                .or_else(|| adapters.iter().find(|a| a.device_type() == DeviceType::Other))
                .or(adapters.first()),
            PowerPreference::LowPower => adapters
                .iter()
                .find(|a| a.device_type() == DeviceType::IntegratedGpu)
                .or_else(|| adapters.iter().find(|a| a.device_type() == DeviceType::Cpu))
                .or(adapters.first()),
            PowerPreference::None => adapters.first(),
        }
        .context("No GPU adapters available")?;

        tracing::info!(
            adapter_id = adapter.inner.info.id,
            adapter_name = %adapter.inner.info.name,
            adapter_type = ?adapter.inner.info.device_type,
            "Selected GPU adapter"
        );

        Ok(adapter.clone())
    }

    fn adapter_for_id(&self, adapter_id: u32) -> Result<Adapter> {
        let info = self
            .backend
            .lock()
            .unwrap()
            .enumerate_adapters()
            .into_iter()
            .find(|a| a.id == adapter_id)
            .with_context(|| format!("Invalid adapter ID: {adapter_id}"))?;
        Ok(self.adapter_from_info(info))
    }

    /// Create a device on the first adapter matching the given type.
    ///
    /// On Windows with the DX12 backend, set `GOLDY_DX12_FORCE_WARP=1` to create the device on
    /// the WARP software adapter instead, even if a real GPU is present (WARP is still listed via
    /// `GOLDY_DX12_ALLOW_WARP=1` or by setting `GOLDY_DX12_FORCE_WARP=1` alone, which also
    /// registers the WARP adapter). Ignored for non-DX12 backends.
    #[deprecated(
        since = "0.2.0",
        note = "use Instance::request_adapter(...).request_device(...) instead"
    )]
    pub fn create_device(&self, preferred_type: DeviceType) -> Result<Device> {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            if self.backend_type() == BackendType::Dx12 && crate::backend::dx12::env_force_warp() {
                tracing::info!("GOLDY_DX12_FORCE_WARP=1 — using WARP adapter");
                return self
                    .adapter_for_id(crate::backend::dx12::WARP_ADAPTER_ID)?
                    .request_device(&DeviceDescriptor::default());
            }
        }

        tracing::info!(?preferred_type, "Requesting GPU device");
        let adapters = self.enumerate_adapters();

        let adapter = adapters
            .iter()
            .find(|a| a.inner.info.device_type == preferred_type)
            .or_else(|| adapters.first())
            .context("No GPU adapters available")?;

        tracing::info!(
            adapter_id = adapter.inner.info.id,
            adapter_name = %adapter.inner.info.name,
            adapter_type = ?adapter.inner.info.device_type,
            "Selected GPU adapter"
        );

        adapter.request_device(&DeviceDescriptor::default())
    }

    /// Create a device on a specific adapter by ID.
    ///
    /// The device is automatically configured with the built-in `goldy_exp`
    /// (experimental) shader library registered. You can register additional
    /// libraries using [`Device::register_library`].
    #[deprecated(
        since = "0.2.0",
        note = "use Adapter::request_device(...) after enumerate_adapters or request_adapter"
    )]
    pub fn create_device_for_adapter(&self, adapter_id: u32) -> Result<Device> {
        self.adapter_for_id(adapter_id)?
            .request_device(&DeviceDescriptor::default())
    }

    /// Get the backend type (Vulkan, Metal, DX12).
    pub fn backend_type(&self) -> BackendType {
        self.backend.lock().unwrap().backend_type()
    }
}

/// Options for [`Instance::request_adapter`].
#[derive(Debug, Clone)]
pub struct RequestAdapterOptions {
    /// Prefer a high-performance or low-power adapter when multiple are available.
    pub power_preference: PowerPreference,
    /// When true on DX12, select the WARP software adapter.
    pub force_fallback_adapter: bool,
}

impl Default for RequestAdapterOptions {
    fn default() -> Self {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        let force_fallback_adapter = crate::backend::dx12::env_force_warp();
        #[cfg(not(all(feature = "dx12", target_os = "windows")))]
        let force_fallback_adapter = false;
        Self {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter,
        }
    }
}

/// Power preference for adapter selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerPreference {
    /// No preference — use the first enumerated adapter.
    None,
    /// Prefer integrated / low-power GPUs.
    LowPower,
    /// Prefer discrete / high-performance GPUs, with integrated and other fallbacks.
    HighPerformance,
}

/// Descriptor for [`Adapter::request_device`].
#[derive(Debug, Clone, Default)]
pub struct DeviceDescriptor {
    /// Optional debug label for the logical device.
    pub label: Option<String>,
}

pub(crate) struct AdapterInner {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    info: AdapterInfo,
    caps: DeviceCapabilities,
}

/// A physical GPU adapter with immutable capabilities.
#[derive(Clone)]
pub struct Adapter {
    pub(crate) inner: Arc<AdapterInner>,
}

impl std::fmt::Debug for Adapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Adapter")
            .field("info", &self.inner.info)
            .finish_non_exhaustive()
    }
}

impl Adapter {
    /// Immutable adapter metadata (name, vendor, device type, backend).
    pub fn get_info(&self) -> AdapterInfo {
        self.inner.info.clone()
    }

    /// Immutable capability snapshot for this adapter.
    pub fn capabilities(&self) -> DeviceCapabilities {
        self.inner.caps.clone()
    }

    /// Create a logical [`Device`] on this adapter.
    pub fn request_device(&self, desc: &DeviceDescriptor) -> Result<Device> {
        let _ = desc;
        tracing::debug!(adapter_id = self.inner.info.id, "Creating device for adapter");
        let mut backend = self.inner.backend.lock().unwrap();
        let handle = backend.create_device(self.inner.info.id)?;

        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            if self.inner.info.id == crate::backend::dx12::WARP_ADAPTER_ID
                && backend.backend_type() == BackendType::Dx12
            {
                crate::backend::dx12::log_warp_module_path_once();
            }
        }

        drop(backend);

        let device_timeline_reader = {
            let backend = self.inner.backend.lock().unwrap();
            backend
                .clone_device_timeline_reader(handle)
                .ok_or_else(|| anyhow::anyhow!("missing device timeline reader"))?
        };

        let mut registry = ShaderLibraryRegistry::new();
        registry.register(ShaderLibrary::goldy_experimental())?;

        tracing::info!(
            adapter_id = self.inner.info.id,
            device_type = ?self.inner.info.device_type,
            "GPU device created"
        );

        Ok(Device {
            inner: Arc::new(DeviceInner {
                backend: Arc::clone(&self.inner.backend),
                handle,
                adapter: self.clone(),
                library_registry: Arc::new(Mutex::new(registry)),
                vram_allocator: Arc::new(crate::vram_allocator::DefaultVramAllocator::new()),
                owns_backend_device: true,
                device_timeline_reader,
                context_readers: Arc::new(Mutex::new(HashMap::new())),
            }),
        })
    }

    /// Get the adapter ID.
    pub fn id(&self) -> u32 {
        self.inner.info.id
    }

    /// Get the adapter name.
    pub fn name(&self) -> &str {
        &self.inner.info.name
    }

    /// Get the device type.
    pub fn device_type(&self) -> DeviceType {
        self.inner.info.device_type
    }

    /// Get the vendor name.
    pub fn vendor(&self) -> &str {
        &self.inner.info.vendor
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

    /// Whether [`crate::types::BufferFlags::CPU_READABLE`] scattered buffers can be read from CPU
    /// without a GPU copy.
    ///
    /// `true` on Vulkan (`HOST_VISIBLE` storage) and Metal (Shared storage). `false` on Direct3D 12
    /// (requires GPU copy to a READBACK heap).
    pub has_zero_copy_storage_readback: bool,

    /// How costly in-place buffer resize (`resize_to`) is on this device.
    pub buffer_resize_cost: BufferResizeCost,

    /// Sparse / tile page size when applicable; informational for aligning resize hints.
    pub buffer_page_size: u64,

    /// Whether `hint_unused_above` on backing buffer allocations can return physical memory to the system.
    pub buffer_decommit_supported: bool,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            preferred_surface_format: TextureFormat::Bgra8UnormSrgb,
            preferred_render_target_format: TextureFormat::Rgba8Unorm,
            supported_surface_formats: vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm],
            supported_render_target_formats: vec![
                TextureFormat::Rgba8Unorm,
                TextureFormat::Rgba8UnormSrgb,
                TextureFormat::Bgra8Unorm,
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Rgba16Float,
                TextureFormat::Rgba32Float,
            ],
            has_zero_copy_storage_readback: true,
            buffer_resize_cost: BufferResizeCost::Copy,
            buffer_page_size: 64 * 1024,
            buffer_decommit_supported: false,
        }
    }
}

/// A GPU device - used to create resources and render.
///
/// `Device` is a lightweight, cloneable handle (internally reference-counted).
/// Cloning a `Device` is cheap (`Arc` bump) and gives you another handle to the
/// same underlying GPU device. The physical device is only torn down once every
/// `Device` handle **and** every resource created from it have been dropped.
///
/// # Thread Safety
///
/// Internally, `Device` uses a `Mutex` to serialize backend operations. This means:
/// - Resource creation is thread-safe but serializes internally
/// - Task graph recording via [`RenderPassBuilder`](crate::RenderPassBuilder) is lock-free
/// - Graph submission acquires the lock
///
/// See the [module documentation](self) for best practices.
///
/// # Shader Libraries
///
/// The device maintains a registry of shader libraries that are automatically
/// available to all shaders compiled for this device. The built-in `goldy`
/// library is registered by default.
///
/// ```rust,ignore
/// use goldy::ShaderLibrary;
///
/// // Register a custom library
/// device.register_library(ShaderLibrary::from_source("mylib", "module mylib;"))?;
///
/// // Check if a library is registered
/// assert!(device.has_library("goldy"));
/// ```
pub struct Device {
    pub(crate) inner: Arc<DeviceInner>,
}

pub(crate) struct DeviceInner {
    pub(crate) backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: DeviceHandle,
    adapter: Adapter,
    library_registry: Arc<Mutex<ShaderLibraryRegistry>>,
    vram_allocator: Arc<dyn crate::vram_allocator::VramAllocatorAlloc>,
    /// When `false`, this [`Device`] is a logical alias (e.g. [`Device::with_vram_allocator`]);
    /// dropping it must not call [`GpuBackend::destroy_device`] on the shared handle.
    pub(crate) owns_backend_device: bool,
    /// Device floor + sync primitive; per-context progress comes from [`Self::context_readers`].
    pub(crate) device_timeline_reader: Arc<dyn backend::DeviceTimelineReader>,
    /// Per-context timeline readers registered at [`crate::Context::new`], unregistered on drop.
    pub(crate) context_readers:
        Arc<Mutex<HashMap<crate::backend::ContextHandle, Arc<dyn backend::ContextTimelineReader>>>>,
}

impl Clone for Device {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Internal registry for shader libraries.
struct ShaderLibraryRegistry {
    libraries: HashMap<String, ShaderLibrary>,
    /// Temp directory for library sources (lazily created)
    temp_dir: Option<PathBuf>,
    /// Whether temp files are out of sync with libraries
    dirty: bool,
}

impl ShaderLibraryRegistry {
    fn new() -> Self {
        Self {
            libraries: HashMap::new(),
            temp_dir: None,
            dirty: true,
        }
    }

    fn register(&mut self, library: ShaderLibrary) -> Result<()> {
        let name = library.name().to_string();
        if self.libraries.contains_key(&name) {
            anyhow::bail!("Library '{}' is already registered", name);
        }
        self.libraries.insert(name, library);
        self.dirty = true;
        Ok(())
    }

    fn unregister(&mut self, name: &str) -> bool {
        if self.libraries.remove(name).is_some() {
            self.dirty = true;
            true
        } else {
            false
        }
    }

    fn has(&self, name: &str) -> bool {
        self.libraries.contains_key(name)
    }

    fn list(&self) -> Vec<&str> {
        self.libraries.keys().map(|s| s.as_str()).collect()
    }

    /// Ensure temp files are written and return search paths.
    fn get_search_paths(&mut self) -> Result<Vec<PathBuf>> {
        if self.libraries.is_empty() {
            return Ok(vec![]);
        }

        // Create temp directory if needed
        if self.temp_dir.is_none() {
            let unique_id = REGISTRY_COUNTER.fetch_add(1, Ordering::Relaxed);
            let temp_dir = std::env::temp_dir().join(format!("goldy-shaders-{}-{}", std::process::id(), unique_id));
            std::fs::create_dir_all(&temp_dir).context("Failed to create shader library temp directory")?;
            self.temp_dir = Some(temp_dir);
        }

        // Write library files if dirty
        if self.dirty {
            let temp_dir = self.temp_dir.as_ref().unwrap();

            for library in self.libraries.values() {
                for (module_path, source) in library.modules() {
                    // Convert module path (with forward slashes) to OS-appropriate file path
                    let mut file_path = temp_dir.clone();
                    for component in module_path.split('/') {
                        file_path = file_path.join(component);
                    }
                    file_path.set_extension("slang");

                    // Ensure parent directories exist
                    if let Some(parent) = file_path.parent() {
                        std::fs::create_dir_all(parent).context("Failed to create module directory")?;
                    }

                    std::fs::write(&file_path, source)
                        .with_context(|| format!("Failed to write module: {}", module_path))?;
                }
            }

            self.dirty = false;
        }

        Ok(vec![self.temp_dir.clone().unwrap()])
    }
}

impl Drop for ShaderLibraryRegistry {
    fn drop(&mut self) {
        // Clean up temp directory
        if let Some(temp_dir) = self.temp_dir.take() {
            let _ = std::fs::remove_dir_all(temp_dir);
        }
    }
}

impl Device {
    // =======================================================================
    // VramAllocator
    // =======================================================================

    pub(crate) fn defer_release(&self, epoch: TimelineValue, payload: crate::vram_allocator::DeferredPayload) {
        self.inner.vram_allocator.defer_release(epoch, payload);
    }

    /// Returns the currently installed [`VramAllocator`].
    ///
    /// The default is [`DefaultVramAllocator`] which delegates directly to the
    /// backend with no overhead.
    ///
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    /// [`DefaultVramAllocator`]: crate::vram_allocator::DefaultVramAllocator
    pub fn vram_allocator(&self) -> &dyn crate::vram_allocator::VramAllocator {
        self.inner.vram_allocator.as_ref()
    }

    /// Returns a clone of the [`Arc`] holding the current buffer allocator.
    ///
    /// [`VramAllocatorAlloc`]: crate::vram_allocator::VramAllocatorAlloc
    #[allow(dead_code)] // device alias tests
    pub(crate) fn vram_allocator_arc(&self) -> Arc<dyn crate::vram_allocator::VramAllocatorAlloc> {
        Arc::clone(&self.inner.vram_allocator)
    }

    /// Create a new `Device` handle sharing the same GPU device but using
    /// a different [`VramAllocator`].
    ///
    /// All resources created through the returned handle will go through the
    /// new allocator. Resources created through the original handle are
    /// unaffected.
    ///
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    #[allow(dead_code)] // device alias tests
    pub(crate) fn with_vram_allocator(&self, allocator: Arc<dyn crate::vram_allocator::VramAllocatorAlloc>) -> Self {
        Self {
            inner: Arc::new(DeviceInner {
                backend: Arc::clone(&self.inner.backend),
                handle: self.inner.handle,
                adapter: self.inner.adapter.clone(),
                library_registry: Arc::clone(&self.inner.library_registry),
                vram_allocator: allocator,
                owns_backend_device: false,
                device_timeline_reader: Arc::clone(&self.inner.device_timeline_reader),
                context_readers: Arc::clone(&self.inner.context_readers),
            }),
        }
    }

    /// Install an [`AllocationPolicy`](crate::allocation_policy::AllocationPolicy) on the
    /// device's [`DefaultVramAllocator`](crate::vram_allocator::DefaultVramAllocator).
    ///
    /// Fails if a policy is already installed.
    pub fn set_allocation_policy(
        &self,
        policy: Arc<dyn crate::allocation_policy::AllocationPolicy>,
    ) -> anyhow::Result<()> {
        self.inner.vram_allocator.set_allocation_policy(policy)
    }

    /// Install an allocation policy if the device still has the default [`NoPolicy`](crate::NoPolicy).
    pub fn ensure_allocation_policy(
        &self,
        policy: Arc<dyn crate::allocation_policy::AllocationPolicy>,
    ) -> anyhow::Result<()> {
        self.inner.vram_allocator.ensure_allocation_policy(policy)
    }

    fn parcel_deed(&self) -> crate::vram_allocator::ParcelDeed {
        crate::vram_allocator::ParcelDeed::new(Arc::downgrade(&self.inner.vram_allocator))
    }

    fn finish_buffer_alloc(&self, mut buf: crate::buffer::Allocation) -> anyhow::Result<crate::buffer::Allocation> {
        buf.set_deed(self.parcel_deed());
        Ok(buf)
    }

    fn finish_texture_alloc(
        &self,
        mut tex: crate::texture::TextureBacking,
    ) -> anyhow::Result<crate::texture::TextureBacking> {
        tex.set_deed(self.parcel_deed());
        Ok(tex)
    }

    /// Allocate a GPU buffer through the device's [`VramAllocator`].
    ///
    /// Crate-internal entry point for runtime allocators and pools. Application code should
    /// use [`RetainedPool::acquire_buffer`](crate::RetainedPool::acquire_buffer) instead.
    /// Allocations receive an accounting deed and honor the installed allocator's budget
    /// and telemetry.
    ///
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    /// [`VramAllocator::alloc_buffer`]: crate::vram_allocator::VramAllocator::alloc_buffer
    pub(crate) fn alloc_buffer(
        &self,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> anyhow::Result<crate::buffer::Allocation> {
        let buf = self
            .inner
            .vram_allocator
            .alloc_buffer(self, size, access, element_stride, flags)?;
        self.finish_buffer_alloc(buf)
    }

    /// Allocate a GPU buffer with a capacity hint through the device's [`VramAllocator`].
    ///
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    pub(crate) fn alloc_buffer_with_capacity(
        &self,
        initial_size: u64,
        expected_max: u64,
        access: BufferKind,
        flags: BufferFlags,
    ) -> anyhow::Result<crate::buffer::Allocation> {
        let buf =
            self.inner
                .vram_allocator
                .alloc_buffer_with_capacity(self, initial_size, expected_max, access, flags)?;
        self.finish_buffer_alloc(buf)
    }

    /// Allocate a buffer initialized with typed data (element stride from `T`).
    #[cfg(test)]
    pub(crate) fn alloc_buffer_with_data<T: crate::buffer::StructuredBufferElement>(
        &self,
        data: &[T],
        access: BufferKind,
    ) -> anyhow::Result<crate::buffer::Allocation> {
        let bytes = bytemuck::cast_slice(data);
        let stride = std::mem::size_of::<T>() as u32;
        self.alloc_buffer_with_bytes_stride(bytes, access, stride)
    }

    /// Allocate a buffer initialized with raw bytes and a custom element stride.
    #[cfg(test)]
    pub(crate) fn alloc_buffer_with_bytes_stride(
        &self,
        data: &[u8],
        access: BufferKind,
        element_stride: u32,
    ) -> anyhow::Result<crate::buffer::Allocation> {
        self.alloc_buffer_with_bytes_stride_and_flags(data, access, element_stride, BufferFlags::empty())
    }

    /// Like [`Self::alloc_buffer_with_bytes_stride`], with explicit [`BufferFlags`].
    pub(crate) fn alloc_buffer_with_bytes_stride_and_flags(
        &self,
        data: &[u8],
        access: BufferKind,
        element_stride: u32,
        flags: BufferFlags,
    ) -> anyhow::Result<crate::buffer::Allocation> {
        let buf = self.alloc_buffer(data.len() as u64, access, Some(element_stride), flags)?;
        buf.write(0, data)?;
        Ok(buf)
    }

    /// Allocate a GPU texture through the device's [`VramAllocator`].
    ///
    /// All public texture creation goes through this method. Allocations receive an
    /// accounting deed and honor the installed allocator's budget and telemetry.
    ///
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    /// [`VramAllocatorAlloc::alloc_texture`]: crate::vram_allocator::VramAllocatorAlloc::alloc_texture
    pub(crate) fn alloc_texture(
        &self,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> anyhow::Result<crate::texture::TextureBacking> {
        let tex = self
            .inner
            .vram_allocator
            .alloc_texture(self, width, height, format, access, flags)?;
        self.finish_texture_alloc(tex)
    }

    // =======================================================================
    // Device metadata
    // =======================================================================

    /// Physical adapter this device was created from.
    pub fn adapter(&self) -> &Adapter {
        &self.inner.adapter
    }

    /// Get the adapter ID this device was created on.
    pub fn adapter_id(&self) -> u32 {
        self.inner.adapter.id()
    }

    /// Get the device type (discrete GPU, integrated GPU, CPU/software, etc.).
    pub fn device_type(&self) -> DeviceType {
        self.inner.adapter.device_type()
    }

    /// Graphics backend used by this device (Vulkan, Dx12, Metal, ...).
    pub fn backend_type(&self) -> BackendType {
        self.inner.backend.lock().unwrap().backend_type()
    }

    /// Check if the device is still valid.
    pub fn is_valid(&self) -> bool {
        self.inner.backend.lock().unwrap().is_device_valid(self.inner.handle)
    }

    /// Create a submission/timeline context bound to this device.
    ///
    /// The context holds an `Arc` clone of the device substrate, so the device
    /// outlives the context. Submit, wait, signal, and reclamation APIs live on [`Context`].
    pub fn create_context(&self) -> Result<crate::context::Context, GoldyError> {
        crate::context::Context::new(self.clone())
    }

    /// Latest device-global submission sequence retired on the GPU.
    ///
    /// Epochs from any [`crate::context::Context::submit`] on this device share one value space; use this
    /// when reclaiming deferred frees keyed by timeline value (e.g. heap transient allocator).
    pub fn timeline_retired(&self) -> crate::timeline::TimelineValue {
        let horizon = self.inner.device_timeline_reader.device_horizon();
        let reader_clones: Vec<_> = {
            let readers = self.inner.context_readers.lock().unwrap();
            readers.values().cloned().collect()
        };
        let max_ctx = reader_clones.iter().map(|r| r.gpu_progress()).max().unwrap_or(0);
        horizon.max(max_ctx)
    }

    /// Lock-free GPU progress for a live context on this device (for ledger / parcel queries).
    pub(crate) fn context_gpu_progress(
        &self,
        ctx: crate::backend::ContextHandle,
    ) -> Option<crate::timeline::TimelineValue> {
        let reader = {
            let readers = self.inner.context_readers.lock().unwrap();
            readers.get(&ctx).cloned()
        }?;
        Some(reader.gpu_progress())
    }

    pub(crate) fn register_context_timeline_reader(
        &self,
        ctx: crate::backend::ContextHandle,
        reader: Arc<dyn backend::ContextTimelineReader>,
    ) {
        self.inner.context_readers.lock().unwrap().insert(ctx, reader);
    }

    pub(crate) fn unregister_context_timeline_reader(&self, ctx: crate::backend::ContextHandle) {
        self.inner.context_readers.lock().unwrap().remove(&ctx);
    }

    /// Block until the device-global timeline has retired at least `value`.
    ///
    /// Unlike [`Context::wait_until`], this does not require the caller to hold the same
    /// [`Context`] that submitted `value`. The backend searches across all live contexts
    /// on this device for the one that produced `value` and waits on its native primitive
    /// (Metal `MTLSharedEvent`, Vulkan timeline semaphore, DX12 fence).
    ///
    /// Use this from allocators and other device-scoped objects that receive epoch values
    /// from external contexts they do not own.
    ///
    /// [`Context::wait_until`]: crate::Context::wait_until
    pub fn wait_until_retired(&self, value: crate::timeline::TimelineValue) -> Result<(), GoldyError> {
        let mut backend = self.inner.backend.lock().unwrap();
        backend.device_wait_until(self.inner.handle, value).map_err(|e| {
            drop(backend);
            GoldyError::Backend(e)
        })
    }

    /// Returns `true` if the device has been permanently lost.
    ///
    /// After this returns `true`, all further submit / wait calls will fail with
    /// [`GoldyError::DeviceLost`]. The device should be dropped and re-created.
    pub fn is_device_lost(&self) -> bool {
        self.inner.backend.lock().unwrap().is_device_lost(self.inner.handle)
    }

    /// Number of bindless descriptor slots still available for allocation in
    /// the given `category`.
    ///
    /// Use this to check remaining capacity and make adaptive cleanup
    /// decisions (e.g. calling [`Context::flush_deferred_deletions`](crate::Context::flush_deferred_deletions)
    /// when slots are low) rather than relying on fixed heuristics.
    ///
    /// Returns `u32::MAX` on backends that don't enforce a per-category cap
    /// (currently Vulkan and DX12, which support 16 384+ per category).
    pub fn available_bindless_slots(&self, category: ResourceCategory) -> u32 {
        let backend = self.inner.backend.lock().unwrap();
        backend.available_bindless_slots(self.inner.handle, category)
    }

    /// Maximum number of bindless descriptor slots per category for this device.
    ///
    /// Returns `u32::MAX` on backends that don't enforce a meaningful per-category cap.
    pub fn max_bindless_slots_per_category(&self, category: ResourceCategory) -> u32 {
        let backend = self.inner.backend.lock().unwrap();
        backend.max_bindless_slots_per_category(self.inner.handle, category)
    }

    /// Snapshot of the Metal buffer heap allocator state.
    /// Returns `None` on non-Metal backends.
    #[doc(hidden)]
    pub fn buffer_heap_stats(&self) -> Option<crate::backend::BufferHeapStats> {
        let backend = self.inner.backend.lock().unwrap();
        backend.buffer_heap_stats(self.inner.handle)
    }

    /// Snapshot of the Metal texture heap allocator state.
    /// Returns `None` on non-Metal backends.
    #[doc(hidden)]
    pub fn texture_heap_stats(&self) -> Option<crate::backend::TextureHeapStats> {
        let backend = self.inner.backend.lock().unwrap();
        backend.texture_heap_stats(self.inner.handle)
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
    /// use goldy::{DeviceDescriptor, Instance, RequestAdapterOptions};
    ///
    /// let instance = Instance::new()?;
    /// let adapter = instance.request_adapter(&RequestAdapterOptions::default())?;
    /// let device = adapter.request_device(&DeviceDescriptor::default())?;
    /// let caps = device.capabilities();
    ///
    /// println!("Surface format: {:?}", caps.preferred_surface_format);
    /// println!("RenderTarget format: {:?}", caps.preferred_render_target_format);
    /// println!("Zero-copy CPU storage readback: {}", caps.has_zero_copy_storage_readback);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn capabilities(&self) -> DeviceCapabilities {
        self.inner.adapter.capabilities()
    }

    // --- Shader Library Management ---

    /// Register a shader library for use in shader imports.
    ///
    /// After registration, shaders can use `import <library_name>;` to access
    /// the library's modules.
    ///
    /// # Errors
    ///
    /// Returns an error if a library with the same name is already registered.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use goldy::ShaderLibrary;
    ///
    /// let my_lib = ShaderLibrary::from_source("myutils", r#"
    ///     module myutils;
    ///     public float3 custom_color() { return float3(1, 0, 0); }
    /// "#);
    ///
    /// device.register_library(my_lib)?;
    ///
    /// // Now shaders can use: import myutils;
    /// ```
    pub fn register_library(&self, library: ShaderLibrary) -> Result<()> {
        tracing::debug!(library_name = %library.name(), "Registering shader library");
        self.inner.library_registry.lock().unwrap().register(library)
    }

    /// Unregister a shader library.
    ///
    /// Returns `true` if the library was found and removed, `false` if it
    /// wasn't registered.
    ///
    /// # Note
    ///
    /// Unregistering the built-in `goldy` library is allowed but not recommended,
    /// as many shader utilities depend on it.
    pub fn unregister_library(&self, name: &str) -> bool {
        self.inner.library_registry.lock().unwrap().unregister(name)
    }

    /// Check if a shader library is registered.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // The goldy library is registered by default
    /// assert!(device.has_library("goldy"));
    /// ```
    pub fn has_library(&self, name: &str) -> bool {
        self.inner.library_registry.lock().unwrap().has(name)
    }

    /// List all registered shader libraries.
    ///
    /// Returns the names of all currently registered libraries.
    pub fn list_libraries(&self) -> Vec<String> {
        self.inner
            .library_registry
            .lock()
            .unwrap()
            .list()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Notify the backend that all transient buffers have been freed and the
    /// underlying heap/allocator bookkeeping should be rebalanced. On Metal
    /// this replaces the primary heap (right-sized to recent peak usage) and
    /// drops overflow heaps, so subsequent frames allocate from one contiguous
    /// heap instead of chasing overflow after overflow.
    ///
    /// The backend is responsible for making the call safe: it will block
    /// until in-flight GPU work finishes before touching the heaps, so
    /// callers do not need to issue their own `wait_fence` first. They must
    /// however have already dropped any Rust-side references to buffers
    /// allocated from these heaps (otherwise the underlying Metal allocation
    /// remains alive and the reset is a no-op for that range).
    ///
    /// Typical use is just before a large reallocation (e.g. recreating a
    /// long-lived pool backing buffer) or at a natural steady-state boundary
    /// such as a resize or scene change.
    pub fn reset_buffer_heaps(&self) {
        self.inner.backend.lock().unwrap().reset_buffer_heaps(self.inner.handle);
    }

    /// Ensure the internal heap can accommodate at least `min_capacity` bytes
    /// in a single allocation without overflow. Call after `reset_buffer_heaps`
    /// and before creating large pool backing buffers.
    pub fn ensure_buffer_heap_capacity(&self, min_capacity: u64) {
        self.inner
            .backend
            .lock()
            .unwrap()
            .ensure_buffer_heap_capacity(self.inner.handle, min_capacity);
    }

    /// Drop empty overflow heaps (both buffer and texture) after frame cleanup.
    ///
    /// Safe to call after retired buffers/textures have been dropped. On Metal
    /// this releases `MTLHeap` objects that accumulated during frames when the
    /// primary heaps couldn't satisfy all allocations.
    pub fn compact_overflow_heaps(&self) {
        self.inner
            .backend
            .lock()
            .unwrap()
            .compact_overflow_heaps(self.inner.handle);
    }

    /// Drop backend-held Slang / driver compiler session state to reduce host RSS.
    ///
    /// On Metal this frees the persistent Slang compiler that usually holds large
    /// IR caches. Call after all pipelines you need are created; any later lazy
    /// compile will re-instantiate the compiler.
    pub fn release_idle_shader_compiler(&self) {
        self.inner.backend.lock().unwrap().release_idle_shader_compiler();
    }

    /// No-op: texture uploads are scheduled via [`crate::task_graph::TaskGraph`].
    #[deprecated(
        since = "0.1.0",
        note = "Texture uploads are batched via TaskGraph::write_texture / write_texture_region; there is nothing to flush."
    )]
    pub fn flush_texture_uploads(&self) -> Result<()> {
        Ok(())
    }

    /// Query the platform row-pitch and staging buffer layout for an UPLOAD from a 2-D texture region.
    ///
    /// On DX12 rows are padded to 256-byte alignment; on Vulkan and Metal rows are tight
    /// (`width × bpp`).  Use the returned [`crate::backend::TextureCopyFootprint`] to allocate
    /// a `CPU_WRITABLE` buffer of `staging_bytes` capacity and write each row at `row_pitch`
    /// stride starting from byte `footprint_offset` — then pass `row_pitch` as the
    /// `src_row_pitch` argument to [`crate::Scheme::copy_buffer_to_texture_parcel`] so the
    /// backend can skip the intermediate repack step.
    pub fn texture_copy_footprint(
        &self,
        width: u32,
        height: u32,
        format: crate::types::TextureFormat,
    ) -> Result<crate::backend::TextureCopyFootprint, GoldyError> {
        let backend = self.inner.backend.lock().unwrap();
        backend
            .query_texture_copy_footprint(self.inner.handle, width, height, format)
            .map_err(GoldyError::Backend)
    }

    /// Get search paths for shader compilation (internal use).
    pub(crate) fn get_shader_search_paths(&self) -> Result<Vec<PathBuf>> {
        self.inner.library_registry.lock().unwrap().get_search_paths()
    }

    /// Reflect the memory layout of a Slang `struct` by compiling `shader_source` once for reflection.
    ///
    /// Search paths include registered shader libraries (same as [`ShaderModule::from_slang`](crate::ShaderModule::from_slang)). The
    /// active backend's codegen target (SPIR-V / DXIL / Metal) is used so reported layout matches
    /// real shader compilation.
    ///
    /// `shader_source` must declare a vertex entry point named **`vs_main`**.
    pub fn reflect_struct(&self, shader_source: &str, type_name: &str) -> Result<StructLayout> {
        let paths = self.get_shader_search_paths()?;
        let path_strings: Vec<String> = paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        let path_refs: Vec<&str> = path_strings.iter().map(|s| s.as_str()).collect();
        let target = match self.inner.backend.lock().unwrap().backend_type() {
            BackendType::Vulkan => ShaderTarget::Spirv,
            BackendType::Dx12 => ShaderTarget::Dxil,
            BackendType::Metal => ShaderTarget::Metal,
        };
        let compiler = SlangCompiler::new().context("Failed to create Slang compiler for reflect_struct")?;
        compiler.reflect_struct_layout(shader_source, target, &path_refs, type_name)
    }

    /// Create a device from a backend for testing purposes.
    #[doc(hidden)]
    pub(crate) fn from_backend(backend: Box<dyn GpuBackend>) -> anyhow::Result<Self> {
        let backend = Arc::new(Mutex::new(backend));
        let adapter_info = {
            let b = backend.lock().unwrap();
            b.enumerate_adapters().into_iter().next().unwrap_or(AdapterInfo {
                id: 0,
                name: "Test GPU".to_string(),
                vendor: "Goldy Test".to_string(),
                backend: BackendType::Vulkan,
                device_type: DeviceType::Other,
            })
        };
        let caps = backend.lock().unwrap().adapter_capabilities(adapter_info.id);
        let adapter = Adapter {
            inner: Arc::new(AdapterInner {
                backend: Arc::clone(&backend),
                info: adapter_info,
                caps,
            }),
        };
        let handle = {
            let mut b = backend.lock().unwrap();
            b.create_device(adapter.id())?
        };

        let device_timeline_reader = {
            let b = backend.lock().unwrap();
            b.clone_device_timeline_reader(handle)
                .expect("missing device timeline reader")
        };

        let mut registry = ShaderLibraryRegistry::new();
        registry.register(ShaderLibrary::goldy_experimental())?;

        Ok(Self {
            inner: Arc::new(DeviceInner {
                backend,
                handle,
                adapter,
                library_registry: Arc::new(Mutex::new(registry)),
                vram_allocator: Arc::new(crate::vram_allocator::DefaultVramAllocator::new()),
                owns_backend_device: true,
                device_timeline_reader,
                context_readers: Arc::new(Mutex::new(HashMap::new())),
            }),
        })
    }

    #[doc(hidden)]
    pub fn with_mock_backend<R>(&self, f: impl FnOnce(&mut crate::backend::mock::MockBackend) -> R) -> R {
        let mut guard = self.inner.backend.lock().unwrap();
        let mock = guard
            .as_mut()
            .as_any_mut()
            .downcast_mut::<crate::backend::mock::MockBackend>()
            .expect("Device::with_mock_backend: backend is not MockBackend");
        f(mock)
    }

    /// Access the inner [`MockBackend`] for test introspection.
    ///
    /// Panics if the device was not created with `Device::from_backend(Box::new(MockBackend::new()))`.
    #[cfg(test)]
    pub(crate) fn with_mock<R>(&self, f: impl FnOnce(&mut crate::backend::mock::MockBackend) -> R) -> R {
        self.with_mock_backend(f)
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        tracing::debug!(
            %self.handle,
            adapter_id = self.adapter.id(),
            device_type = ?self.adapter.device_type(),
            owns_backend = self.owns_backend_device,
            "Dropping GPU device handle"
        );
        // Wait for all GPU work on this device to complete before tearing down resources.
        // Contexts must be dropped before DeviceInner; device_wait_idle is the device-wide fence.
        // Skip if already lost: the hardware cannot make progress and destroy_device orders teardown.
        let already_lost = self.backend.lock().unwrap().is_device_lost(self.handle);
        if !already_lost {
            let mut backend = self.backend.lock().unwrap();
            let _ = backend.device_wait_idle(self.handle);
        }
        // Drop all deferred payloads after the idle wait.
        self.vram_allocator.drain();
        // The placement heap is owned per-`Context` and dropped in `ContextInner::drop`,
        // which runs before this (contexts hold a `Device` clone, so they outlive nothing
        // but are dropped first by ekrano/users tearing down renderers before devices).
        if self.owns_backend_device {
            let mut backend = self.backend.lock().unwrap();
            backend.destroy_device(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn with_vram_allocator_alias_does_not_destroy_backend_device() {
        use std::sync::Arc;

        let device = test_device();
        let alias = device.with_vram_allocator(Arc::new(crate::vram_allocator::DefaultVramAllocator::new()));
        assert!(device.is_valid());
        drop(alias);
        assert!(
            device.is_valid(),
            "dropping a with_vram_allocator alias must not destroy the backend device"
        );
    }

    #[test]
    fn test_goldy_library_registered_by_default() {
        let device = test_device();
        assert!(device.has_library("goldy_exp"));
    }

    #[test]
    fn test_register_custom_library() {
        let device = test_device();

        let lib = ShaderLibrary::from_source("custom", "module custom;");
        device.register_library(lib).unwrap();

        assert!(device.has_library("custom"));
    }

    #[test]
    fn test_register_duplicate_fails() {
        let device = test_device();

        let lib1 = ShaderLibrary::from_source("mylib", "module mylib;");
        let lib2 = ShaderLibrary::from_source("mylib", "module mylib;");

        device.register_library(lib1).unwrap();
        let result = device.register_library(lib2);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already registered"));
    }

    #[test]
    fn test_unregister_library() {
        let device = test_device();

        let lib = ShaderLibrary::from_source("temp", "module temp;");
        device.register_library(lib).unwrap();
        assert!(device.has_library("temp"));

        assert!(device.unregister_library("temp"));
        assert!(!device.has_library("temp"));
    }

    #[test]
    fn test_unregister_nonexistent_returns_false() {
        let device = test_device();
        assert!(!device.unregister_library("nonexistent"));
    }

    #[test]
    fn test_list_libraries() {
        let device = test_device();

        let libs = device.list_libraries();
        assert!(libs.contains(&"goldy_exp".to_string()));

        device
            .register_library(ShaderLibrary::from_source("extra", "module extra;"))
            .unwrap();
        let libs = device.list_libraries();
        assert!(libs.contains(&"goldy_exp".to_string()));
        assert!(libs.contains(&"extra".to_string()));
    }

    #[test]
    fn test_search_paths_writes_files() {
        let device = test_device();

        let paths = device.get_shader_search_paths().unwrap();
        assert_eq!(paths.len(), 1);

        // Verify goldy_exp files were written
        let goldy_file = paths[0].join("goldy_exp.slang");
        assert!(goldy_file.exists(), "goldy_exp.slang should exist at {:?}", goldy_file);

        let math_file = paths[0].join("goldy_exp/math.slang");
        assert!(
            math_file.exists(),
            "goldy_exp/math.slang should exist at {:?}",
            math_file
        );
    }
}
