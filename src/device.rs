//! GPU device management.
//!
//! # Thread Safety
//!
//! Goldy uses a single-threaded command submission model with lock-free command recording:
//!
//! - **Command Recording**: [`CommandEncoder`](crate::CommandEncoder) is completely lock-free.
//!   You can create and record commands on any thread without any synchronization.
//!   
//! - **Resource Creation**: Creating resources ([`crate::Buffer`],
//!   [`RenderPipeline`](crate::RenderPipeline), etc.) acquires the backend lock.
//!   These operations are safe from any thread but serialize internally.
//!
//! - **Command Submission**: Submitting commands via [`RenderTarget::render()`](crate::RenderTarget::render)
//!   or [`Frame::render`](crate::surface::Frame::render) acquires the backend lock.
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
use crate::error::GoldyError;
use crate::shader_library::ShaderLibrary;
use crate::slang::{ShaderTarget, SlangCompiler, StructLayout};
use crate::task_graph::TaskGraph;
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

    /// Enumerate available GPU adapters.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        let backend = self.backend.lock().unwrap();
        let adapters: Vec<Adapter> = backend
            .enumerate_adapters()
            .into_iter()
            .map(|info| Adapter { info })
            .collect();
        tracing::debug!(count = adapters.len(), "Enumerated GPU adapters");
        for adapter in &adapters {
            tracing::debug!(
                id = adapter.info.id,
                name = %adapter.info.name,
                vendor = %adapter.info.vendor,
                device_type = ?adapter.info.device_type,
                "  adapter"
            );
        }
        adapters
    }

    /// Create a device on the first adapter matching the given type.
    ///
    /// On Windows with the DX12 backend, set `GOLDY_DX12_FORCE_WARP=1` to create the device on
    /// the WARP software adapter instead, even if a real GPU is present (WARP is still listed via
    /// `GOLDY_DX12_ALLOW_WARP=1` or by setting `GOLDY_DX12_FORCE_WARP=1` alone, which also
    /// registers the WARP adapter). Ignored for non-DX12 backends.
    pub fn create_device(&self, preferred_type: DeviceType) -> Result<Device> {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            if self.backend_type() == BackendType::Dx12 && crate::backend::dx12::env_force_warp() {
                tracing::info!("GOLDY_DX12_FORCE_WARP=1 — using WARP adapter");
                return self.create_device_for_adapter(crate::backend::dx12::WARP_ADAPTER_ID);
            }
        }

        tracing::info!(?preferred_type, "Requesting GPU device");
        let adapters = self.enumerate_adapters();

        let adapter = adapters
            .iter()
            .find(|a| a.info.device_type == preferred_type)
            .or_else(|| adapters.first())
            .context("No GPU adapters available")?;

        tracing::info!(
            adapter_id = adapter.info.id,
            adapter_name = %adapter.info.name,
            adapter_type = ?adapter.info.device_type,
            "Selected GPU adapter"
        );

        self.create_device_for_adapter(adapter.info.id)
    }

    /// Create a device on a specific adapter by ID.
    ///
    /// The device is automatically configured with the built-in `goldy_exp`
    /// (experimental) shader library registered. You can register additional
    /// libraries using [`Device::register_library`].
    pub fn create_device_for_adapter(&self, adapter_id: u32) -> Result<Device> {
        tracing::debug!(adapter_id, "Creating device for adapter");
        let mut backend = self.backend.lock().unwrap();

        let device_type = backend
            .enumerate_adapters()
            .into_iter()
            .find(|a| a.id == adapter_id)
            .map(|a| a.device_type)
            .unwrap_or(DeviceType::Other);

        let handle = backend.create_device(adapter_id)?;

        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            if adapter_id == crate::backend::dx12::WARP_ADAPTER_ID
                && backend.backend_type() == BackendType::Dx12
            {
                crate::backend::dx12::log_warp_module_path_once();
            }
        }

        let mut registry = ShaderLibraryRegistry::new();
        registry.register(ShaderLibrary::goldy_experimental())?;

        tracing::info!(adapter_id, ?device_type, "GPU device created");

        Ok(Device {
            inner: Arc::new(DeviceInner {
                backend: Arc::clone(&self.backend),
                handle,
                adapter_id,
                device_type,
                library_registry: Arc::new(Mutex::new(registry)),
                vram_allocator: Arc::new(crate::vram_allocator::DefaultVramAllocator::new()),
                placement_heap: Mutex::new(None),
                high_water_timeline: AtomicU64::new(0),
            }),
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

    /// Whether [`crate::types::BufferFlags::CPU_READABLE`] scattered buffers can be read from CPU
    /// without a GPU copy.
    ///
    /// `true` on Vulkan (`HOST_VISIBLE` storage) and Metal (Shared storage). `false` on Direct3D 12
    /// (requires GPU copy to a READBACK heap).
    pub has_zero_copy_storage_readback: bool,

    /// How [`crate::Buffer::resize_to`] is implemented on this device.
    pub buffer_resize_cost: BufferResizeCost,

    /// Sparse / tile page size when applicable; informational for aligning resize hints.
    pub buffer_page_size: u64,

    /// Whether [`crate::Buffer::hint_unused_above`] can return physical memory to the system.
    pub buffer_decommit_supported: bool,
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
/// - Command recording via [`CommandEncoder`](crate::CommandEncoder) is lock-free
/// - Command submission acquires the lock
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
    adapter_id: u32,
    device_type: DeviceType,
    library_registry: Arc<Mutex<ShaderLibraryRegistry>>,
    vram_allocator: Arc<dyn crate::vram_allocator::VramAllocator>,
    pub(crate) placement_heap: Mutex<Option<crate::placement_heap::PlacementHeap>>,
    /// Largest TimelineValue ever returned from submit(). Used in Drop to ensure
    /// all submitted GPU work has completed before destroying the device.
    high_water_timeline: AtomicU64,
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
            let temp_dir = std::env::temp_dir().join(format!(
                "goldy-shaders-{}-{}",
                std::process::id(),
                unique_id
            ));
            std::fs::create_dir_all(&temp_dir)
                .context("Failed to create shader library temp directory")?;
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
                        std::fs::create_dir_all(parent)
                            .context("Failed to create module directory")?;
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

    /// Returns the currently installed [`VramAllocator`].
    ///
    /// The default is [`DefaultVramAllocator`] which delegates directly to the
    /// backend with no overhead.
    ///
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    /// [`DefaultVramAllocator`]: crate::vram_allocator::DefaultVramAllocator
    pub fn vram_allocator(&self) -> &dyn crate::vram_allocator::VramAllocator {
        &*self.inner.vram_allocator
    }

    /// Returns a clone of the [`Arc`] holding the current [`VramAllocator`].
    ///
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    pub fn vram_allocator_arc(&self) -> Arc<dyn crate::vram_allocator::VramAllocator> {
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
    pub fn with_vram_allocator(
        &self,
        allocator: Arc<dyn crate::vram_allocator::VramAllocator>,
    ) -> Self {
        Self {
            inner: Arc::new(DeviceInner {
                backend: Arc::clone(&self.inner.backend),
                handle: self.inner.handle,
                adapter_id: self.inner.adapter_id,
                device_type: self.inner.device_type,
                library_registry: Arc::clone(&self.inner.library_registry),
                vram_allocator: allocator,
                placement_heap: Mutex::new(None),
                high_water_timeline: AtomicU64::new(0),
            }),
        }
    }

    /// Allocate a GPU buffer through the device's [`VramAllocator`].
    ///
    /// Equivalent to calling [`VramAllocator::alloc_buffer`] on the installed
    /// allocator. Prefer this over [`Buffer::new`] when you want allocations to
    /// go through the unified allocator for tracking and budgeting.
    ///
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    /// [`VramAllocator::alloc_buffer`]: crate::vram_allocator::VramAllocator::alloc_buffer
    /// [`Buffer::new`]: crate::buffer::Buffer::new
    pub fn alloc_buffer(
        &self,
        size: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> anyhow::Result<crate::buffer::Buffer> {
        self.inner
            .vram_allocator
            .alloc_buffer(self, size, access, element_stride, flags)
    }

    /// Allocate a GPU buffer with a capacity hint through the device's [`VramAllocator`].
    ///
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    pub fn alloc_buffer_with_capacity(
        &self,
        initial_size: u64,
        expected_max: u64,
        access: DataAccess,
        flags: BufferFlags,
    ) -> anyhow::Result<crate::buffer::Buffer> {
        self.inner.vram_allocator.alloc_buffer_with_capacity(
            self,
            initial_size,
            expected_max,
            access,
            flags,
        )
    }

    /// Allocate a GPU texture through the device's [`VramAllocator`].
    ///
    /// Equivalent to calling [`VramAllocator::alloc_texture`] on the installed
    /// allocator. Prefer this over [`Texture::new`] when you want allocations to
    /// go through the unified allocator for tracking and budgeting.
    ///
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    /// [`VramAllocator::alloc_texture`]: crate::vram_allocator::VramAllocator::alloc_texture
    /// [`Texture::new`]: crate::texture::Texture::new
    pub fn alloc_texture(
        &self,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: SpatialAccess,
        flags: TextureFlags,
    ) -> anyhow::Result<crate::texture::Texture> {
        self.inner
            .vram_allocator
            .alloc_texture(self, width, height, format, access, flags)
    }

    // =======================================================================
    // Device metadata
    // =======================================================================

    /// Get the adapter ID this device was created on.
    pub fn adapter_id(&self) -> u32 {
        self.inner.adapter_id
    }

    /// Get the device type (discrete GPU, integrated GPU, CPU/software, etc.).
    pub fn device_type(&self) -> DeviceType {
        self.inner.device_type
    }

    /// Graphics backend used by this device (Vulkan, Dx12, Metal, ...).
    pub fn backend_type(&self) -> BackendType {
        self.inner.backend.lock().unwrap().backend_type()
    }

    /// Check if the device is still valid.
    pub fn is_valid(&self) -> bool {
        self.inner
            .backend
            .lock()
            .unwrap()
            .is_device_valid(self.inner.handle)
    }

    /// Latest GPU completion counter on this device's timeline (`wait_until(value)` is valid once
    /// `gpu_progress() >= value`).
    pub fn gpu_progress(&self) -> TimelineValue {
        let _tz = crate::tracy_zone!("device.gpu_progress");
        let backend = {
            let _lock = crate::tracy_zone!("device.gpu_progress.lock");
            self.inner.backend.lock().unwrap()
        };
        let _query = crate::tracy_zone!("device.gpu_progress.query");
        backend.gpu_progress(self.inner.handle)
    }

    /// The largest [`TimelineValue`] ever returned by [`submit`](Self::submit) on this device.
    ///
    /// Waiting on this value guarantees that all GPU work submitted through this device handle
    /// has completed. Primarily useful for diagnostics; `Drop for DeviceInner` uses this
    /// automatically before tearing down the backend device.
    pub fn high_water_timeline(&self) -> TimelineValue {
        self.inner.high_water_timeline.load(Ordering::Relaxed)
    }

    /// Returns `true` if the device has been permanently lost.
    ///
    /// After this returns `true`, all further submit / wait calls will fail with
    /// [`GoldyError::DeviceLost`]. The device should be dropped and re-created.
    pub fn is_device_lost(&self) -> bool {
        self.inner
            .backend
            .lock()
            .unwrap()
            .is_device_lost(self.inner.handle)
    }

    /// Map a backend [`anyhow::Error`] to the appropriate [`GoldyError`] variant.
    fn classify(&self, e: anyhow::Error) -> GoldyError {
        if self.is_device_lost() {
            return GoldyError::DeviceLost;
        }
        GoldyError::Backend(e)
    }

    /// Block until the device timeline reaches at least `value`.
    pub fn wait_until(&self, value: TimelineValue) -> Result<(), GoldyError> {
        let _tz = crate::tracy_zone!("device.wait_until");
        let mut backend = {
            let _lock = crate::tracy_zone!("device.wait_until.lock");
            self.inner.backend.lock().unwrap()
        };
        let _backend = crate::tracy_zone!("device.wait_until.backend");
        backend.wait_until(self.inner.handle, value).map_err(|e| {
            drop(backend);
            self.classify(e)
        })
    }

    /// Like [`wait_until`](Self::wait_until) but returns `Err(`[`GoldyError::SubmitTimeout`]`)` on timeout.
    pub fn wait_until_timeout(
        &self,
        value: TimelineValue,
        timeout_ms: u32,
    ) -> Result<(), GoldyError> {
        let mut backend = self.inner.backend.lock().unwrap();
        match backend.wait_until_timeout(self.inner.handle, value, timeout_ms) {
            Ok(true) => Ok(()),
            Ok(false) => Err(GoldyError::SubmitTimeout),
            Err(e) => {
                drop(backend);
                Err(self.classify(e))
            }
        }
    }

    /// Number of bindless descriptor slots still available for allocation in
    /// the given `category`.
    ///
    /// Use this to check remaining capacity and make adaptive cleanup
    /// decisions (e.g. calling [`flush_deferred_deletions`](Self::flush_deferred_deletions)
    /// when slots are low) rather than relying on fixed heuristics.
    ///
    /// Returns `u32::MAX` on backends that don't enforce a per-category cap
    /// (currently Vulkan and DX12, which support 16 384+ per category).
    pub fn available_bindless_slots(&self, category: BindlessCategory) -> u32 {
        let backend = self.inner.backend.lock().unwrap();
        backend.available_bindless_slots(self.inner.handle, category)
    }

    /// Maximum number of bindless descriptor slots per category for this device.
    ///
    /// Returns `u32::MAX` on backends that don't enforce a meaningful per-category cap.
    pub fn max_bindless_slots_per_category(&self, category: BindlessCategory) -> u32 {
        let backend = self.inner.backend.lock().unwrap();
        backend.max_bindless_slots_per_category(self.inner.handle, category)
    }

    /// Reclaim bindless descriptor slots and process deferred GPU deletions
    /// whose timeline barrier has been signaled.
    ///
    /// This is normally called internally at acquire / present / submit, but
    /// consumers that drop buffers between those points (e.g. during a
    /// non-blocking frame drain) can call this to reclaim slots immediately
    /// rather than waiting for the next internal call.
    ///
    /// Drives all epoch-based reclamation: `VramAllocator::reclaim` drops any
    /// [`DeferredPayload`]s registered via [`defer_release`] whose epoch has been
    /// reached. This includes:
    /// - `BufferView`s from the placement heap (for transient buffer lifetimes)
    /// - `RegionReclaimToken`s from `EpochRegionsAllocator`
    /// - `FreeRangeToken`s from `HeapTransientAllocator`
    /// - `ResetToken`s from `BumpResetAllocator`
    ///
    /// [`DeferredPayload`]: crate::vram_allocator::DeferredPayload
    /// [`defer_release`]: Self::defer_release
    pub fn flush_deferred_deletions(&self) {
        let _tz = crate::tracy_zone!("device.flush_deferred_deletions");
        let mut backend = self.inner.backend.lock().unwrap();
        backend.flush_deferred_deletions(self.inner.handle);
        let progress = backend.gpu_progress(self.inner.handle);
        drop(backend);
        self.inner.vram_allocator.reclaim(progress);
    }

    /// Register a [`DeferredPayload`] for deferred dropping after GPU timeline `epoch` retires.
    ///
    /// The device's [`VramAllocator`] holds the payload alive until a subsequent call to
    /// [`flush_deferred_deletions`] or device [`Drop`] observes `gpu_progress >= epoch`, at
    /// which point all resources in the payload are dropped.
    ///
    /// This is a lower-level primitive. Prefer [`GpuGuard`](crate::gpu_guard::GpuGuard) for
    /// ergonomic RAII-based resource lifetime management.
    ///
    /// [`DeferredPayload`]: crate::vram_allocator::DeferredPayload
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    /// [`flush_deferred_deletions`]: Self::flush_deferred_deletions
    pub fn defer_release(
        &self,
        epoch: TimelineValue,
        payload: crate::vram_allocator::DeferredPayload,
    ) {
        self.inner.vram_allocator.defer_release(epoch, payload);
    }

    /// Convenience wrapper around [`defer_release`]: defer a single resource until `epoch` retires.
    ///
    /// Wraps `resource` in a [`DeferredPayload`] and forwards to the device's [`VramAllocator`].
    ///
    /// [`defer_release`]: Self::defer_release
    /// [`DeferredPayload`]: crate::vram_allocator::DeferredPayload
    /// [`VramAllocator`]: crate::vram_allocator::VramAllocator
    pub fn defer_until<T: Send + 'static>(&self, epoch: TimelineValue, resource: T) {
        let mut payload = crate::vram_allocator::DeferredPayload::new();
        payload.push(resource);
        self.inner.vram_allocator.defer_release(epoch, payload);
    }

    #[doc(hidden)]
    pub fn deferred_deletion_pending_count(&self) -> usize {
        let backend = self.inner.backend.lock().unwrap();
        backend.deferred_deletion_pending_count(self.inner.handle)
    }

    /// Submit a compiled [`TaskGraph`] on the device timeline (standalone / non-surface compute).
    ///
    /// Graphs containing transient buffers are resolved automatically using a
    /// persistent [`PlacementHeap`](crate::placement_heap::PlacementHeap) owned
    /// by the device. The heap is lazily created on first use and reused across
    /// submissions — callers never need to manage transient buffer backing
    /// storage, view creation, or bindless index patching.
    pub fn submit(&self, graph: &mut TaskGraph) -> Result<TimelineValue, GoldyError> {
        if !graph.has_transient_resources() {
            let mut backend = self.inner.backend.lock().unwrap();
            let tv = graph
                .submit_with_backend(self, backend.as_mut(), None, &HashMap::new(), true)
                .map_err(|e| {
                    drop(backend);
                    self.classify(e)
                })?;
            self.inner
                .high_water_timeline
                .fetch_max(tv, Ordering::Relaxed);
            return Ok(tv);
        }

        // All transient resource cases go through the heap (cached views/textures).
        let tv = self
            .submit_with_placement_heap(graph, true)
            .map_err(|e| self.classify(e))?;
        self.inner
            .high_water_timeline
            .fetch_max(tv, Ordering::Relaxed);
        Ok(tv)
    }

    /// Like [`Self::submit`] but does **not** block the CPU until transient-buffer or
    /// transient-texture GPU work completes after the submission.
    ///
    /// Use this only when building a pipelined frame loop that records multiple
    /// consecutive graphs (for example [`FrameOrchestrator::flush`](crate::FrameOrchestrator::flush)) or when the
    /// caller tracks completion via [`TimelineValue`] / [`Self::gpu_progress`].
    ///
    /// [`Self::submit`] still waits on transient resources — that path remains the
    /// safe default for one-shot submission.
    pub fn submit_pipelined(&self, graph: &mut TaskGraph) -> Result<TimelineValue, GoldyError> {
        let _tz = crate::tracy_zone!("device.submit_pipelined");
        if !graph.has_transient_resources() {
            let mut backend = self.inner.backend.lock().unwrap();
            let tv = graph
                .submit_with_backend(self, backend.as_mut(), None, &HashMap::new(), false)
                .map_err(|e| {
                    drop(backend);
                    self.classify(e)
                })?;
            self.inner
                .high_water_timeline
                .fetch_max(tv, Ordering::Relaxed);
            return Ok(tv);
        }

        // All transient resource cases go through the heap (cached views/textures).
        let tv = self
            .submit_with_placement_heap(graph, false)
            .map_err(|e| self.classify(e))?;
        self.inner
            .high_water_timeline
            .fetch_max(tv, Ordering::Relaxed);
        Ok(tv)
    }

    /// Submit a task graph and retain the closed command list for potential reuse.
    ///
    /// Works like [`Self::submit_pipelined`] but also stores the compiled command list keyed
    /// by the graph's [`TaskGraph::compute_retention_fingerprint`].  Call
    /// [`Self::try_resubmit_retained`] on the next frame — passing the same fingerprint — to
    /// re-execute the same list without re-recording when the graph is unchanged.
    ///
    /// Graphs with transient resources, render passes, or upload nodes are not eligible for
    /// retention and fall back gracefully to a plain submit.
    pub fn submit_pipelined_and_retain(
        &self,
        graph: &mut TaskGraph,
    ) -> Result<TimelineValue, GoldyError> {
        if graph.has_transient_resources() {
            // Transient-buffer graphs cannot be safely retained; fall back.
            return self.submit_pipelined(graph);
        }
        let mut backend = self.inner.backend.lock().unwrap();
        let tv = graph
            .submit_with_backend_and_retain(self, backend.as_mut())
            .map_err(|e| {
                drop(backend);
                self.classify(e)
            })?;
        self.inner
            .high_water_timeline
            .fetch_max(tv, Ordering::Relaxed);
        Ok(tv)
    }

    /// Re-execute a previously retained command list without re-recording.
    ///
    /// Returns `Ok(Some(tv))` on success, `Ok(None)` if no matching retained list exists
    /// (caller should fall back to a full submit-and-retain cycle).
    pub fn try_resubmit_retained(&self, key: u64) -> Result<Option<TimelineValue>, GoldyError> {
        let mut backend = self.inner.backend.lock().unwrap();
        let result = backend
            .try_resubmit_retained(self.inner.handle, key)
            .map_err(|e| {
                drop(backend);
                self.classify(e)
            })?;
        if let Some(tv) = result {
            self.inner
                .high_water_timeline
                .fetch_max(tv, Ordering::Relaxed);
        }
        Ok(result)
    }

    /// Default pipeline depth for initial placement heap sizing.
    pub(crate) const DEFAULT_PIPELINE_DEPTH: u64 = 4;

    /// Resolve transient buffers via the device-owned placement heap and submit.
    ///
    /// 1. Compute wave-interval coloring layout
    /// 2. Lazily create / reclaim / acquire from the persistent placement heap
    /// 3. Create `BufferView`s at `base_offset + colored_offset`; collect bindless indices
    /// 4. Patch dispatch `resource_slots` with real bindless indices
    /// 5. Submit the resolved IR (views kept alive in the heap ring until GPU retires)
    ///
    /// Handles both transient-buffer and transient-texture cases, routing through
    /// the device's [`PlacementHeap`](crate::placement_heap::PlacementHeap) for
    /// stable-slot caching. Views and textures are owned by the heap across frames;
    /// only evicted entries go through `defer_release`.
    fn submit_with_placement_heap(
        &self,
        graph: &mut TaskGraph,
        wait_for_transient_completion: bool,
    ) -> Result<TimelineValue> {
        use crate::task_graph::{ResolvedTransientBuffer, ResolvedTransientTexture, SlotResolver};

        // Compute the schedule once; derive node_waves from it so the
        // transient layout and texture resolution paths don't re-run
        // build_edges + schedule_waves.
        let (schedule, _) = graph.schedule_and_split_wave();
        let node_waves =
            crate::task_graph::analysis::node_to_wave_map(&schedule, graph.ir().nodes.len());

        let has_buffers = graph.has_transient_buffers();

        let (alloc_size, base_align, layout_opt) = if has_buffers {
            let (ts, ba, lay) = graph.transient_heap_size_and_layout(&node_waves)?;
            let sz = (ts + ba - 1).max(256);
            (sz, ba, Some(lay))
        } else {
            (0u64, 1u64, None)
        };

        let mut heap_guard = self.inner.placement_heap.lock().unwrap();

        if heap_guard.is_none() {
            let cap = (256 * 1024 * 1024u64).max(alloc_size * Self::DEFAULT_PIPELINE_DEPTH);
            *heap_guard = Some(
                crate::placement_heap::PlacementHeap::with_capacity(self, cap)
                    .context("failed to create device placement heap")?,
            );
        }
        let heap = heap_guard.as_mut().unwrap();

        if has_buffers {
            let depth = Self::DEFAULT_PIPELINE_DEPTH as usize;
            heap.configure_pages(alloc_size, depth, self)?;
        }

        // ── Build the SlotResolver ───────────────────────────────────────────
        let mut resolver = SlotResolver::new();

        let tex_handles = graph.resolve_transient_textures_with_heap(self, heap, &node_waves)?;
        for (id, handle) in &tex_handles {
            resolver
                .textures
                .insert(*id, ResolvedTransientTexture { handle: *handle });
        }

        if has_buffers {
            let layout = layout_opt.unwrap();
            let raw_offset = heap.advance_page();
            let base_offset = raw_offset.div_ceil(base_align) * base_align;

            let buf_handle = heap.buffer().handle;
            for spec in graph.transient_specs() {
                let offset = base_offset + layout[&spec.id];
                let view_stride = spec.stride.max(1);
                let (uav, srv, _hit) =
                    heap.get_or_create_view(spec.id, offset, spec.size, view_stride, self)?;

                resolver.buffers.insert(
                    spec.id,
                    ResolvedTransientBuffer {
                        parent: buf_handle,
                        offset,
                        len: spec.size,
                        uav_index: uav,
                        srv_index: srv,
                    },
                );
            }
        }

        drop(heap_guard);

        let mut backend = self.inner.backend.lock().unwrap();
        let tv = graph.submit_ir_with_resolver(
            self,
            backend.as_mut(),
            &resolver,
            wait_for_transient_completion,
        )?;
        drop(backend);

        // Stamp paged-mode timeline.
        let mut heap_guard = self.inner.placement_heap.lock().unwrap();
        if let Some(heap) = heap_guard.as_mut() {
            heap.stamp_pending(tv);
        }

        Ok(tv)
    }

    /// Submit a task graph and block until it completes.
    pub fn dispatch(&self, graph: &mut TaskGraph) -> Result<(), GoldyError> {
        let v = self.submit(graph)?;
        self.wait_until(v)
    }

    /// Snapshot of the device-owned placement heap's state for diagnostics.
    ///
    /// Returns `None` if the heap hasn't been created yet (no transient-buffer
    /// graphs have been submitted).
    pub fn placement_heap_stats(&self) -> Option<crate::placement_heap::PlacementHeapStats> {
        let heap_guard = self.inner.placement_heap.lock().unwrap();
        heap_guard.as_ref().map(|h| h.stats())
    }

    /// Number of `BufferView`s and `Texture`s currently held in the placement heap's
    /// stable-slot view cache. Returns `(cached_views, cached_textures)`.
    ///
    /// Useful for the `[PERF]` log and diagnostics; non-zero values in steady state
    /// confirm that the cache is active and hitting.
    pub fn transient_cache_counts(&self) -> (usize, usize) {
        let heap_guard = self.inner.placement_heap.lock().unwrap();
        match heap_guard.as_ref() {
            Some(h) => (h.cached_view_count(), h.cached_texture_count()),
            None => (0, 0),
        }
    }

    /// Total number of `create_buffer_view` backend calls made by the placement heap's
    /// view cache since initialization. Monotonically increasing.
    ///
    /// Use this in tests to verify that steady-state frames produce zero new creates.
    pub fn transient_view_create_count(&self) -> usize {
        let heap_guard = self.inner.placement_heap.lock().unwrap();
        heap_guard
            .as_ref()
            .map(|h| h.view_create_count())
            .unwrap_or(0)
    }

    /// Total number of `Texture::new` calls made by the placement heap's texture cache
    /// since initialization. Monotonically increasing.
    pub fn transient_texture_create_count(&self) -> usize {
        let heap_guard = self.inner.placement_heap.lock().unwrap();
        heap_guard
            .as_ref()
            .map(|h| h.texture_create_count())
            .unwrap_or(0)
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
    /// use goldy::{Instance, DeviceType};
    ///
    /// let instance = Instance::new()?;
    /// let device = instance.create_device(DeviceType::DiscreteGpu)?;
    /// let caps = device.capabilities();
    ///
    /// println!("Surface format: {:?}", caps.preferred_surface_format);
    /// println!("RenderTarget format: {:?}", caps.preferred_render_target_format);
    /// println!("Zero-copy CPU storage readback: {}", caps.has_zero_copy_storage_readback);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn capabilities(&self) -> DeviceCapabilities {
        let backend = self.inner.backend.lock().unwrap();
        backend.device_capabilities(self.inner.handle)
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
        self.inner
            .library_registry
            .lock()
            .unwrap()
            .register(library)
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
        self.inner
            .backend
            .lock()
            .unwrap()
            .reset_buffer_heaps(self.inner.handle);
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
        self.inner
            .backend
            .lock()
            .unwrap()
            .release_idle_shader_compiler();
    }

    /// No-op: texture uploads are scheduled via [`crate::task_graph::TaskGraph`].
    #[deprecated(
        since = "0.1.0",
        note = "Texture uploads are batched via TaskGraph::write_texture / write_texture_region; there is nothing to flush."
    )]
    pub fn flush_texture_uploads(&self) -> Result<()> {
        Ok(())
    }

    /// Get search paths for shader compilation (internal use).
    pub(crate) fn get_shader_search_paths(&self) -> Result<Vec<PathBuf>> {
        self.inner
            .library_registry
            .lock()
            .unwrap()
            .get_search_paths()
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
        let path_strings: Vec<String> = paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let path_refs: Vec<&str> = path_strings.iter().map(|s| s.as_str()).collect();
        let target = match self.inner.backend.lock().unwrap().backend_type() {
            BackendType::Vulkan => ShaderTarget::Spirv,
            BackendType::Dx12 => ShaderTarget::Dxil,
            BackendType::Metal => ShaderTarget::Metal,
            BackendType::WebGPU => {
                anyhow::bail!("reflect_struct is not supported on the WebGPU backend yet");
            }
        };
        let compiler =
            SlangCompiler::new().context("Failed to create Slang compiler for reflect_struct")?;
        compiler.reflect_struct_layout(shader_source, target, &path_refs, type_name)
    }

    /// Create a device from a backend for testing purposes.
    #[cfg(test)]
    pub(crate) fn from_backend(backend: Box<dyn GpuBackend>) -> anyhow::Result<Self> {
        let backend = Arc::new(Mutex::new(backend));
        let handle = {
            let mut b = backend.lock().unwrap();
            b.create_device(0)?
        };

        let mut registry = ShaderLibraryRegistry::new();
        registry.register(ShaderLibrary::goldy_experimental())?;

        Ok(Self {
            inner: Arc::new(DeviceInner {
                backend,
                handle,
                adapter_id: 0,
                device_type: DeviceType::Other,
                library_registry: Arc::new(Mutex::new(registry)),
                vram_allocator: Arc::new(crate::vram_allocator::DefaultVramAllocator::new()),
                placement_heap: Mutex::new(None),
                high_water_timeline: AtomicU64::new(0),
            }),
        })
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        tracing::debug!(
            %self.handle,
            adapter_id = self.adapter_id,
            device_type = ?self.device_type,
            "Destroying GPU device"
        );
        // Wait for all submitted GPU work to complete. This covers any in-flight work not
        // covered by the placement-heap wait (e.g. non-transient submits held by callers).
        // Skip the wait if the device is already lost: the hardware cannot make progress
        // and backends handle per-object teardown ordering inside destroy_device.
        let high_water = self.high_water_timeline.load(Ordering::Relaxed);
        let already_lost = self.backend.lock().unwrap().is_device_lost(self.handle);
        if high_water > 0 && !already_lost {
            let mut backend = self.backend.lock().unwrap();
            let _ = backend.wait_until(self.handle, high_water);
        }
        // Drop all deferred payloads. The GPU has completed all submitted work (high-water
        // wait above), so it is safe to release any resources that were deferred.
        self.vram_allocator.drain();
        // Drop the placement heap (and its views/buffer) while the device is still alive.
        // The heap's in-flight work is already covered by the high-water wait above.
        if let Ok(mut heap_guard) = self.placement_heap.lock() {
            *heap_guard = None;
        }
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_device(self.handle);
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
    fn high_water_timeline_starts_at_zero() {
        let device = test_device();
        assert_eq!(device.high_water_timeline(), 0);
    }

    #[test]
    fn high_water_timeline_advances_after_submit() {
        use crate::task_graph::TaskGraph;
        let device = test_device();
        assert_eq!(device.high_water_timeline(), 0);
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).unwrap();
        assert!(tv > 0);
        assert_eq!(device.high_water_timeline(), tv);
        // Second submit advances it further.
        let tv2 = device.submit(&mut graph).unwrap();
        assert!(tv2 > tv);
        assert_eq!(device.high_water_timeline(), tv2);
    }

    #[test]
    fn defer_until_resources_are_not_dropped_before_epoch() {
        use crate::task_graph::TaskGraph;
        let device = test_device();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).unwrap();

        let alive = std::sync::Arc::new(99u32);
        let weak = std::sync::Arc::downgrade(&alive);

        // Defer with a far-future epoch — resource must not drop yet.
        device.defer_until(tv + 100, alive);

        // flush_deferred_deletions with current gpu_progress (tv) should not reclaim tv+100.
        device.flush_deferred_deletions();
        assert!(
            weak.upgrade().is_some(),
            "resource should still be alive after flush at tv"
        );
    }

    #[test]
    fn defer_until_resources_dropped_after_flush_at_epoch() {
        use crate::task_graph::TaskGraph;
        let device = test_device();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).unwrap();

        let alive = std::sync::Arc::new(42u32);
        let weak = std::sync::Arc::downgrade(&alive);

        device.defer_until(tv, alive);

        // After advancing GPU to tv and flushing, resource should be dropped.
        device.wait_until(tv).unwrap();
        device.flush_deferred_deletions();
        assert!(
            weak.upgrade().is_none(),
            "resource should be dropped after flush at epoch"
        );
    }

    #[test]
    fn defer_release_drops_all_on_device_drop() {
        use crate::task_graph::TaskGraph;
        let device = test_device();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).unwrap();

        let alive = std::sync::Arc::new(7u32);
        let weak = std::sync::Arc::downgrade(&alive);

        // Defer with a far-future epoch so normal flush won't reclaim it.
        device.defer_until(tv + 9999, alive);

        // Dropping the device should drain all deferred resources.
        drop(device);
        assert!(
            weak.upgrade().is_none(),
            "device drop should drain deferred resources"
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already registered"));
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
        assert!(
            goldy_file.exists(),
            "goldy_exp.slang should exist at {:?}",
            goldy_file
        );

        let math_file = paths[0].join("goldy_exp/math.slang");
        assert!(
            math_file.exists(),
            "goldy_exp/math.slang should exist at {:?}",
            math_file
        );
    }
}
