//! GPU device management.
//!
//! # Thread Safety
//!
//! Goldy uses a single-threaded command submission model with lock-free command recording:
//!
//! - **Command Recording**: [`CommandEncoder`](crate::CommandEncoder) is completely lock-free.
//!   You can create and record commands on any thread without any synchronization.
//!   
//! - **Resource Creation**: Creating resources ([`Buffer`],
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
use crate::gpu_future::GpuFuture;
use crate::shader_library::ShaderLibrary;
use crate::slang::{ShaderTarget, SlangCompiler, StructLayout};
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
                && self.backend_type() == BackendType::Dx12
            {
                crate::backend::dx12::log_warp_module_path_once();
            }
        }

        let mut registry = ShaderLibraryRegistry::new();
        registry.register(ShaderLibrary::goldy_experimental())?;

        tracing::info!(adapter_id, ?device_type, "GPU device created");

        Ok(Device {
            backend: Arc::clone(&self.backend),
            handle,
            adapter_id,
            device_type,
            library_registry: Arc::new(Mutex::new(registry)),
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
    pub(crate) backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: DeviceHandle,
    adapter_id: u32,
    device_type: DeviceType,
    /// Shader library registry
    library_registry: Arc<Mutex<ShaderLibraryRegistry>>,
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
    /// Get the adapter ID this device was created on.
    pub fn adapter_id(&self) -> u32 {
        self.adapter_id
    }

    /// Get the device type (discrete GPU, integrated GPU, CPU/software, etc.).
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Graphics backend used by this device (Vulkan, Dx12, Metal, ...).
    pub fn backend_type(&self) -> BackendType {
        self.backend.lock().unwrap().backend_type()
    }

    /// Check if the device is still valid.
    pub fn is_valid(&self) -> bool {
        self.backend.lock().unwrap().is_device_valid(self.handle)
    }

    /// Submit a full compute command stream (typically [`TaskGraph::compile_commands`]).
    /// Returns [`GpuFuture`] — does not block.
    pub(crate) fn submit_compute_commands(
        &self,
        commands: &[backend::ComputeCommand],
    ) -> Result<GpuFuture> {
        let mut backend = self.backend.lock().unwrap();
        let token = backend.submit_compute(self.handle, commands)?;
        Ok(GpuFuture {
            backend: Arc::clone(&self.backend),
            device: self.handle,
            fence_token: Some(token),
        })
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
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn capabilities(&self) -> DeviceCapabilities {
        // For now, return sensible defaults
        // Future: query actual device limits and capabilities
        DeviceCapabilities::default()
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
        self.library_registry.lock().unwrap().register(library)
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
        self.library_registry.lock().unwrap().unregister(name)
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
        self.library_registry.lock().unwrap().has(name)
    }

    /// List all registered shader libraries.
    ///
    /// Returns the names of all currently registered libraries.
    pub fn list_libraries(&self) -> Vec<String> {
        self.library_registry
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
        self.backend.lock().unwrap().reset_buffer_heaps(self.handle);
    }

    /// Flush any deferred texture uploads to the GPU.
    ///
    /// This is called automatically before compute submissions and texture
    /// readbacks, but may be called explicitly when immediate availability
    /// is required.
    pub fn flush_texture_uploads(&self) -> Result<()> {
        self.backend.lock().unwrap().flush_texture_uploads()
    }

    /// Get search paths for shader compilation (internal use).
    pub(crate) fn get_shader_search_paths(&self) -> Result<Vec<PathBuf>> {
        self.library_registry.lock().unwrap().get_search_paths()
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
        let target = match self.backend.lock().unwrap().backend_type() {
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

        // Create registry with built-in goldy_exp library
        let mut registry = ShaderLibraryRegistry::new();
        registry.register(ShaderLibrary::goldy_experimental())?;

        Ok(Self {
            backend,
            handle,
            adapter_id: 0,
            device_type: DeviceType::Other,
            library_registry: Arc::new(Mutex::new(registry)),
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        tracing::debug!(
            %self.handle,
            adapter_id = self.adapter_id,
            device_type = ?self.device_type,
            "Destroying GPU device"
        );
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
