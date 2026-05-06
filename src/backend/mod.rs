//! GPU backend abstraction.
//!
//! This module defines the `GpuBackend` trait that each graphics API
//! (Vulkan, Metal, DX12) must implement.
//!
//! ## Backend Selection
//!
//! By default, goldy selects the platform-preferred backend:
//! - **macOS**: Metal
//! - **Windows**: DX12
//! - **Linux**: Vulkan
//!
//! You can override this at runtime by setting the `GOLDY_BACKEND` environment variable:
//!
//! ```bash
//! # Use Vulkan on Windows (instead of DX12)
//! GOLDY_BACKEND=vulkan cargo run --example triangle
//!
//! # Valid values: vulkan, dx12, metal
//! ```

#[cfg(feature = "vulkan")]
pub mod vulkan;

// DX12 backend for Windows
#[cfg(all(feature = "dx12", target_os = "windows"))]
pub mod dx12;

// Mock backend for testing (always available)
pub mod mock;

// Metal backend for macOS (native Metal, not MoltenVK)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

use crate::types::{
    BackendType, BindlessHandle, BufferFlags, Color, DataAccess, DepthFormat, DepthStencilState,
    DeviceType, IndexFormat, PresentMode, PrimitiveTopology, SamplerDesc, SpatialAccess,
    TextureFlags, TextureFormat, VertexBufferLayout,
};
use anyhow::Result;

/// When set via `GOLDY_VALIDATION` (e.g. `api` or `all` in the token list), or loader
/// `VK_INSTANCE_LAYERS`, enables backend-specific GPU validation where supported:
/// Vulkan enables `VK_LAYER_KHRONOS_validation` and `VK_EXT_debug_utils` at instance creation;
/// Metal sets `MTL_SHADER_VALIDATION=1` before the first device is created if that variable is unset.
///
/// See the `validation_env` module for the full `GOLDY_VALIDATION` list syntax (`layout`, `api`, `all`, …).
///
/// For Vulkan, validation is also enabled when `VK_INSTANCE_LAYERS` includes
/// `VK_LAYER_KHRONOS_validation` (loader-driven workflow; see Vulkan backend `new()`).
#[cfg(any(feature = "vulkan", all(feature = "metal", target_os = "macos")))]
#[must_use]
pub(crate) fn goldy_validation_enabled() -> bool {
    crate::validation_env::gpu_api_validation_enabled()
}

#[cfg(any(
    test,
    feature = "vulkan",
    all(feature = "dx12", target_os = "windows"),
    all(feature = "metal", target_os = "macos"),
))]
use crate::types::BindlessCategory;

/// Validate a typed push-constant array against per-slot category expectations
/// reported by shader reflection.
///
/// `expectations[i]` is the category that slot `i` must have according to the
/// shader's `goldy_dyn_*` call with literal slot index `i`. `None` means
/// "reflection couldn't infer — skip validation for this slot" (e.g. the slot
/// is only accessed via a dynamic index that regex analysis can't resolve).
///
/// Returns an error naming the offending slot(s) on mismatch. Designed to run
/// on the dispatch hot path; allocates only on the error path.
#[cfg(any(
    test,
    feature = "vulkan",
    all(feature = "dx12", target_os = "windows"),
    all(feature = "metal", target_os = "macos"),
))]
pub(crate) fn validate_typed_push_constants(
    handles: &[BindlessHandle],
    expectations: &[Option<BindlessCategory>],
    shader_name: &str,
) -> Result<()> {
    let mut mismatches: Vec<String> = Vec::new();
    for (slot, handle) in handles.iter().enumerate() {
        let Some(expected) = expectations.get(slot).copied().flatten() else {
            continue;
        };
        if !handle.category().is_compatible_with(expected) {
            mismatches.push(format!(
                "slot {slot}: shader expects `{}` (via goldy_dyn_{}(_)) but got `{}` handle (index {})",
                expected.name(),
                expected.name(),
                handle.category().name(),
                handle.index()
            ));
        }
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        anyhow::bail!(
            "push-constant category mismatch in shader `{shader_name}`:\n  {}\n\
             Hint: use `Buffer::bindless_handle()` / `Texture::bindless_handle()` \
             rather than `.bindless_index()` so the resource's category flows \
             through to the push-constant setter.",
            mismatches.join("\n  ")
        );
    }
}

/// Pure stride validation — always runs, no env-var check.
///
/// `expected_strides[i]` is the byte stride the shader expects at slot `i`
/// (from Slang reflection). `None` skips that slot. `resolve_stride` must return
/// the host buffer's effective stride, or `None` if it cannot be determined.
#[cfg(any(
    test,
    feature = "vulkan",
    all(feature = "dx12", target_os = "windows"),
    all(feature = "metal", target_os = "macos"),
))]
fn check_push_constant_buffer_strides(
    handles: &[BindlessHandle],
    expected_strides: &[Option<u32>],
    shader_name: &str,
    mut resolve_stride: impl FnMut(BindlessHandle) -> Option<u32>,
) -> Result<()> {
    let mut mismatches: Vec<String> = Vec::new();
    for (slot, handle) in handles.iter().enumerate() {
        let Some(expected) = expected_strides.get(slot).copied().flatten() else {
            continue;
        };
        match handle.category() {
            BindlessCategory::Scattered | BindlessCategory::Broadcast => {}
            _ => continue,
        }
        let Some(actual) = resolve_stride(*handle) else {
            mismatches.push(format!(
                "slot {slot}: shader expects element stride {expected} bytes but no element stride could be resolved for bindless index {} ({})",
                handle.index(),
                handle.category().name()
            ));
            continue;
        };
        if actual != expected {
            mismatches.push(format!(
                "slot {slot}: shader expects element stride {expected} bytes but bound buffer has stride {actual} (bindless index {})",
                handle.index()
            ));
        }
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        anyhow::bail!(
            "push-constant buffer stride mismatch in shader `{shader_name}`:\n  {}\n\
             Hint: match `Buffer::with_data` / `with_bytes_stride` / `new_with_stride` to the `T` in `goldy_dyn_*<T>`. \
             See DEBUGGING.md (`GOLDY_VALIDATE_LAYOUTS`).",
            mismatches.join("\n  ")
        );
    }
}

/// Hot-path wrapper: skips validation unless `GOLDY_VALIDATE_LAYOUTS=1`.
/// Called by each backend's dispatch path.
#[cfg(any(
    test,
    feature = "vulkan",
    all(feature = "dx12", target_os = "windows"),
    all(feature = "metal", target_os = "macos"),
))]
pub(crate) fn validate_typed_push_constant_buffer_strides(
    handles: &[BindlessHandle],
    expected_strides: &[Option<u32>],
    shader_name: &str,
    resolve_stride: impl FnMut(BindlessHandle) -> Option<u32>,
) -> Result<()> {
    if !crate::slang::layout_validation_enabled() {
        return Ok(());
    }
    check_push_constant_buffer_strides(handles, expected_strides, shader_name, resolve_stride)
}

// Re-export raw_window_handle for Surface API users
pub use raw_window_handle;

/// Information about a GPU adapter (physical device).
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    /// Adapter index.
    pub id: u32,
    /// Device name.
    pub name: String,
    /// Vendor name.
    pub vendor: String,
    /// Backend type.
    pub backend: BackendType,
    /// Device type (discrete, integrated, etc.).
    pub device_type: DeviceType,
}

/// Opaque handle types.
pub type DeviceHandle = u64;
pub type BufferHandle = u64;
pub type ShaderHandle = u64;
pub type PipelineHandle = u64;
pub type ComputePipelineHandle = u64;
pub type RenderTargetHandle = u64;
pub type SurfaceHandle = u64;
pub type SwapchainImageHandle = u64;
pub type TextureHandle = u64;
pub type SamplerHandle = u64;

/// Fence token for non-blocking compute submission.
/// Backends use this to identify a specific GPU submission for polling and waiting.
pub type FenceToken = u64;

/// Render command for command buffer recording.
#[derive(Debug, Clone)]
pub enum RenderCommand {
    /// Clear the color render target.
    Clear(Color),
    /// Clear the depth buffer.
    ClearDepth(f32),
    /// Set the active pipeline.
    SetPipeline(PipelineHandle),
    /// Set a vertex buffer.
    SetVertexBuffer {
        slot: u32,
        buffer: BufferHandle,
        offset: u64,
    },
    /// Set an index buffer.
    SetIndexBuffer {
        buffer: BufferHandle,
        offset: u64,
        format: IndexFormat,
    },
    /// Bind resource slots directly with buffer handles (fully bindless mode).
    /// The backend will look up each buffer's bindless index and bind them.
    BindResources { buffers: Vec<BufferHandle> },
    /// Bind resource slots with raw u32 indices (fully bindless mode).
    /// Use this for textures/samplers or when you already have the indices.
    ///
    /// `indices` go to region A (bindless, packed as u16).
    /// `user` go to region B (user scalars, full u32).
    ///
    /// **Prefer [`RenderCommand::BindResourcesTyped`]** — the raw form
    /// bypasses per-slot category validation.
    BindResourcesRaw { indices: Vec<u32>, user: Vec<u32> },
    /// Bind resource slots with typed [`BindlessHandle`]s. Backends validate
    /// each handle's [`crate::types::BindlessCategory`]
    /// against the bound shader's reflection and emit the raw indices.
    BindResourcesTyped { handles: Vec<BindlessHandle> },
    /// Draw primitives (non-indexed).
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    /// Draw indexed primitives.
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        /// Offset added to each index value before fetching the vertex.
        base_vertex: i32,
        first_instance: u32,
    },
}

/// GPU command recorded into a command buffer / task-graph submission.
///
/// Includes compute dispatches, buffer upload/clear, texture uploads, and
/// scheduling barriers — not compute-only despite historical naming in call sites.
#[derive(Debug, Clone)]
pub enum GpuCommand {
    /// Set the active compute pipeline.
    SetPipeline(ComputePipelineHandle),
    /// Bind resource slots (fully bindless mode - buffer indices passed directly).
    BindResources { buffers: Vec<BufferHandle> },
    /// Bind resource slots with raw u32 indices (for textures/samplers or mixed resources).
    ///
    /// `indices` go to region A (bindless, packed as u16).
    /// `user` go to region B (user scalars, full u32).
    ///
    /// **Prefer [`GpuCommand::BindResourcesTyped`]** — the raw form
    /// bypasses per-slot category validation.
    BindResourcesRaw { indices: Vec<u32>, user: Vec<u32> },
    /// Bind resource slots with typed [`BindlessHandle`]s. Backends validate
    /// each handle's [`crate::types::BindlessCategory`]
    /// against the bound shader's reflection and emit the raw indices.
    BindResourcesTyped { handles: Vec<BindlessHandle> },
    /// Dispatch compute workgroups.
    Dispatch {
        workgroups_x: u32,
        workgroups_y: u32,
        workgroups_z: u32,
    },
    /// Indirect dispatch: workgroup counts read from buffer at offset (3× u32: x, y, z).
    DispatchIndirect { buffer: BufferHandle, offset: u64 },
    /// Fill a buffer region with zeros. Batched into the same command stream as dispatches.
    ClearBuffer {
        buffer: BufferHandle,
        offset: u64,
        size: u64,
    },
    /// Write CPU data into a buffer, batched onto the compute command list.
    ///
    /// For DEVICE_LOCAL storage buffers the backend writes to the staging area
    /// then records a GPU copy; for HOST_VISIBLE buffers it maps directly.
    /// This avoids a per-upload `queue_wait_idle` by deferring the GPU copy
    /// to the same submission as the dispatches that consume the data.
    WriteBuffer {
        buffer: BufferHandle,
        offset: u64,
        data: Vec<u8>,
    },
    /// Upload CPU pixel data into a texture (full image), batched with the same submission
    /// as surrounding GPU work.
    WriteTexture {
        texture: TextureHandle,
        data: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Upload a subrectangle of a texture from CPU data.
    WriteTextureRegion {
        texture: TextureHandle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
    /// Memory barrier between compute dispatches.
    /// Ensures all prior shader writes are visible to subsequent reads.
    Barrier,
    /// Per-resource memory barrier. Only the listed resources are synchronized.
    /// Emitted by the compute graph scheduler at dependency edges.
    ResourceBarrier {
        buffers: Vec<BufferHandle>,
        textures: Vec<TextureHandle>,
    },
}

/// Deprecated alias for [`GpuCommand`].
#[deprecated(since = "0.1.0", note = "renamed to GpuCommand")]
pub type ComputeCommand = GpuCommand;

/// GPU backend trait - implemented by Vulkan, Metal, DX12.
pub trait GpuBackend: Send + Sync {
    /// Get the backend type.
    fn backend_type(&self) -> BackendType;

    /// Enumerate available adapters.
    fn enumerate_adapters(&self) -> Vec<AdapterInfo>;

    // Device management
    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle>;
    fn destroy_device(&mut self, device: DeviceHandle);
    fn is_device_valid(&self, device: DeviceHandle) -> bool;

    // Buffer management
    fn create_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<BufferHandle>;
    fn destroy_buffer(&mut self, buffer: BufferHandle);
    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;
    /// Read buffer contents to CPU. Copies from offset 0 for length output.len().
    fn read_buffer_to_cpu(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        output: &mut [u8],
    ) -> Result<()>;
    /// Capability snapshot for `device` (surface formats, [`crate::device::DeviceCapabilities::has_zero_copy_storage_readback`], …).
    fn device_capabilities(&self, device: DeviceHandle) -> crate::device::DeviceCapabilities {
        let _ = device;
        crate::device::DeviceCapabilities::default()
    }
    /// Fill buffer region with zeros. If size is 0, clears from offset to end of buffer.
    fn clear_buffer(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<()>;
    fn buffer_size(&self, buffer: BufferHandle) -> u64;
    /// Get the buffer's index in the global bindless descriptor set.
    /// Returns None if the buffer is not registered.
    fn buffer_bindless_index(&self, buffer: BufferHandle) -> Option<u32>;
    /// Get the buffer's SRV (read-only) bindless index.
    /// For DX12, scattered buffers have both a UAV (write) and SRV (read-only) descriptor.
    /// Returns the SRV index for use with `StructuredBuffer<T>` / `goldy_buf_ro`.
    /// Falls back to the primary bindless index on backends with unified descriptors.
    fn buffer_bindless_srv_index(&self, buffer: BufferHandle) -> Option<u32>;

    /// Create a view into a sub-region of an existing buffer.
    ///
    /// The view gets its own bindless descriptor pointing at `[offset, offset+size)` of the
    /// parent buffer. The shader sees the sub-region as a zero-based buffer. The view does not
    /// own the underlying memory — dropping it unregisters its descriptor but does not free
    /// the parent's allocation.
    ///
    /// `element_stride` determines the structured buffer stride for the view's descriptor.
    fn create_buffer_view(
        &mut self,
        parent: BufferHandle,
        offset: u64,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferHandle>;

    // Shader management
    fn create_shader_with_paths(
        &mut self,
        device: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
    ) -> Result<ShaderHandle>;

    /// Like [`Self::create_shader_with_paths`], but when `layout_checks` is non-empty, each struct
    /// is validated against Slang reflection on the first per-stage compile (same compile as GPU IR).
    ///
    /// Default: only empty `layout_checks` is allowed; non-empty returns an error.
    fn create_shader_with_checks(
        &mut self,
        device: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
        layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
    ) -> Result<ShaderHandle> {
        if layout_checks.is_empty() {
            self.create_shader_with_paths(
                device,
                slang_source,
                search_paths,
                defines,
                optimization_level,
            )
        } else {
            anyhow::bail!("Layout validation requires the Vulkan, DX12, or Metal backend")
        }
    }

    fn destroy_shader(&mut self, shader: ShaderHandle);

    // Pipeline management
    fn create_pipeline(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
    ) -> Result<PipelineHandle>;
    fn destroy_pipeline(&mut self, pipeline: PipelineHandle);

    // Pipeline with depth stencil state
    #[allow(clippy::too_many_arguments)]
    fn create_pipeline_with_depth(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        depth_stencil: Option<&DepthStencilState>,
    ) -> Result<PipelineHandle>;

    // RenderTarget API - GPU buffer stays on GPU, readback is optional
    fn create_render_target(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<RenderTargetHandle>;
    /// Create a render target with an optional depth buffer.
    fn create_render_target_with_depth(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle>;
    fn destroy_render_target(&mut self, target: RenderTargetHandle);
    fn render_to_target(
        &mut self,
        device: DeviceHandle,
        target: RenderTargetHandle,
        commands: &[RenderCommand],
    ) -> Result<()>;
    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()>;

    // Texture management
    fn create_texture(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: SpatialAccess,
        flags: TextureFlags,
    ) -> Result<TextureHandle>;
    fn write_texture(
        &mut self,
        texture: TextureHandle,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()>;
    /// Write pixel data to a subregion of the texture.
    /// The data must match width*height*bpp for the texture's format.
    fn write_texture_region(
        &mut self,
        texture: TextureHandle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<()>;
    fn destroy_texture(&mut self, texture: TextureHandle);
    /// Read texture contents to CPU memory.
    /// The texture must have been created with TextureFlags::COPY_SRC.
    fn read_texture_to_cpu(&mut self, texture: TextureHandle, output: &mut [u8]) -> Result<()>;
    /// Get the texture's index in the global bindless descriptor set.
    /// Returns None if the texture is not registered.
    fn texture_bindless_index(&self, texture: TextureHandle) -> Option<u32>;

    // Sampler management
    fn create_sampler(&mut self, device: DeviceHandle, desc: &SamplerDesc)
        -> Result<SamplerHandle>;
    fn destroy_sampler(&mut self, sampler: SamplerHandle);
    /// Get the sampler's index in the global bindless descriptor set.
    /// Returns None if the sampler is not registered.
    fn sampler_bindless_index(&self, sampler: SamplerHandle) -> Option<u32>;

    // Surface API - zero-copy presentation to window
    /// Create a surface for presenting to a window.
    /// The window handle is platform-specific (HWND on Windows, wl_surface on Wayland, NSView on macOS).
    /// When `depth_format` is `Some`, a depth buffer is created for depth testing (e.g. 3D rendering).
    fn create_surface(
        &mut self,
        device: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        display: &dyn raw_window_handle::HasDisplayHandle,
        depth_format: Option<DepthFormat>,
    ) -> Result<SurfaceHandle>;

    /// Destroy a surface.
    fn destroy_surface(&mut self, surface: SurfaceHandle);

    /// Acquire the next swapchain image to render to.
    ///
    /// After acquire, the frame's texture is available via [`GpuBackend::surface_frame_texture`].
    fn surface_acquire(&mut self, surface: SurfaceHandle) -> Result<SwapchainImageHandle>;

    /// Get the texture handle for the currently acquired surface frame.
    ///
    /// Returns `None` if no frame is currently acquired (i.e. `surface_acquire`
    /// has not been called or `surface_present` has already been called).
    /// The returned texture is registered in the bindless descriptor set and
    /// can be used with compute or render passes.
    fn surface_frame_texture(&self, surface: SurfaceHandle) -> Option<TextureHandle>;

    /// Render commands to a swapchain image.
    fn surface_render(
        &mut self,
        surface: SurfaceHandle,
        image: SwapchainImageHandle,
        commands: &[RenderCommand],
    ) -> Result<()>;

    /// Present a swapchain image to the screen.
    fn surface_present(
        &mut self,
        surface: SurfaceHandle,
        image: SwapchainImageHandle,
    ) -> Result<()>;

    /// Resize the surface (recreates swapchain).
    fn surface_resize(&mut self, surface: SurfaceHandle, width: u32, height: u32) -> Result<()>;

    /// Get the current surface dimensions.
    fn surface_size(&self, surface: SurfaceHandle) -> (u32, u32);

    /// Get the texture format used by a surface's swapchain.
    /// Use this to ensure your render pipeline matches the surface format.
    fn surface_format(&self, surface: SurfaceHandle) -> TextureFormat;

    /// Set the present mode for a surface.
    /// Returns an error if the mode is not supported by the backend.
    fn surface_set_present_mode(
        &mut self,
        _surface: SurfaceHandle,
        _mode: PresentMode,
    ) -> Result<()> {
        Ok(())
    }

    /// Get the current present mode for a surface.
    fn surface_present_mode(&self, _surface: SurfaceHandle) -> PresentMode {
        PresentMode::Auto
    }

    // Compute pipeline management
    /// Create a compute pipeline from a compute shader.
    fn create_compute_pipeline(
        &mut self,
        device: DeviceHandle,
        compute_shader: ShaderHandle,
    ) -> Result<ComputePipelineHandle>;

    /// Destroy a compute pipeline.
    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle);

    /// Execute compute commands.
    /// This submits compute work to the GPU and waits for completion.
    fn dispatch_compute(&mut self, device: DeviceHandle, commands: &[GpuCommand]) -> Result<()> {
        let token = self.submit_compute(device, commands)?;
        self.wait_fence(device, token)
    }

    /// Submit compute commands without blocking. Returns a fence token for polling/waiting.
    fn submit_compute(
        &mut self,
        device: DeviceHandle,
        commands: &[GpuCommand],
    ) -> Result<FenceToken>;

    /// Check if the fence for the given token has signaled (work complete).
    fn is_fence_complete(&self, device: DeviceHandle, token: FenceToken) -> bool;

    /// Block until the fence signals. Returns an error if the device was lost.
    ///
    /// Implementations should remove and destroy the fence for `token` after waiting.
    fn wait_fence(&mut self, device: DeviceHandle, token: FenceToken) -> Result<()>;

    /// Wait with timeout. Returns Ok(true) if signaled, Ok(false) if timeout elapsed, Err if device lost.
    ///
    /// On success or unrecoverable error, implementations should remove and destroy the fence.
    /// On timeout the fence remains valid for a later wait.
    fn wait_fence_timeout(
        &mut self,
        device: DeviceHandle,
        token: FenceToken,
        timeout_ms: u32,
    ) -> Result<bool>;

    /// Notify the backend that a frame has completed and all transient buffers
    /// have been freed. Backends may use this to right-size internal heap
    /// allocations. No-op by default.
    fn reset_buffer_heaps(&mut self, _device: DeviceHandle) {}
}

/// Create the default backend for the current platform.
///
/// The backend can be overridden at runtime by setting the `GOLDY_BACKEND`
/// environment variable to one of: `vulkan`, `dx12`, `metal`.
///
/// Without the override, the platform default is used:
/// - macOS: Metal
/// - Windows: DX12  
/// - Linux: Vulkan
pub fn create_default_backend() -> Result<Box<dyn GpuBackend>> {
    // Check for runtime override via environment variable
    if let Ok(backend_str) = std::env::var("GOLDY_BACKEND") {
        let backend_type = match backend_str.to_lowercase().as_str() {
            "vulkan" | "vk" => BackendType::Vulkan,
            "dx12" | "d3d12" | "directx" => BackendType::Dx12,
            "metal" | "mtl" => BackendType::Metal,
            other => anyhow::bail!(
                "Unknown GOLDY_BACKEND value '{}'. Valid options: vulkan, dx12, metal",
                other
            ),
        };
        tracing::info!(
            "Using backend from GOLDY_BACKEND env var: {:?}",
            backend_type
        );
        return create_backend(backend_type);
    }

    // On macOS with metal feature, prefer Metal
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        tracing::info!("Creating Metal backend");
        Ok(Box::new(metal::MetalBackend::new()?))
    }

    // On Windows with dx12 feature, prefer DX12
    #[cfg(all(
        feature = "dx12",
        target_os = "windows",
        not(all(feature = "metal", target_os = "macos"))
    ))]
    {
        tracing::info!("Creating DX12 backend");
        Ok(Box::new(dx12::Dx12Backend::new()?))
    }

    // Vulkan fallback on non-DX12/non-Metal platforms
    #[cfg(all(
        feature = "vulkan",
        not(all(feature = "dx12", target_os = "windows")),
        not(all(feature = "metal", target_os = "macos"))
    ))]
    {
        tracing::info!("Creating Vulkan backend");
        Ok(Box::new(vulkan::VulkanBackend::new()?))
    }

    // No backend available
    #[cfg(not(any(
        all(feature = "metal", target_os = "macos"),
        all(feature = "dx12", target_os = "windows"),
        feature = "vulkan"
    )))]
    {
        anyhow::bail!("No GPU backend available - enable 'vulkan', 'dx12', or 'metal' feature")
    }
}

/// Create the default backend wrapped in an `Arc<Mutex<...>>`.
///
/// For DX12, returns a process-wide singleton so that all `Instance` objects
/// share one backend — the existing per-instance `Mutex` then naturally
/// serializes all D3D12 calls, preventing debug-layer access violations
/// when parallel test threads create independent instances.
///
/// For other backends, creates a fresh instance each time.
pub fn create_shared_backend() -> Result<std::sync::Arc<std::sync::Mutex<Box<dyn GpuBackend>>>> {
    use std::sync::{Arc, Mutex};

    #[cfg(all(feature = "dx12", target_os = "windows"))]
    {
        let wants_dx12 = match std::env::var("GOLDY_BACKEND") {
            Ok(v) => matches!(v.to_lowercase().as_str(), "dx12" | "d3d12" | "directx"),
            Err(_) => true, // DX12 is the Windows default
        };
        if wants_dx12 {
            return dx12::shared_backend();
        }
    }

    let backend = create_default_backend()?;
    Ok(Arc::new(Mutex::new(backend)))
}

/// Create a specific backend by type.
pub fn create_backend(backend_type: BackendType) -> Result<Box<dyn GpuBackend>> {
    match backend_type {
        #[cfg(feature = "vulkan")]
        BackendType::Vulkan => {
            tracing::info!("Creating Vulkan backend");
            Ok(Box::new(vulkan::VulkanBackend::new()?))
        }
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        BackendType::Dx12 => {
            tracing::info!("Creating DX12 backend");
            Ok(Box::new(dx12::Dx12Backend::new()?))
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        BackendType::Metal => {
            tracing::info!("Creating Metal backend");
            Ok(Box::new(metal::MetalBackend::new()?))
        }
        _ => anyhow::bail!("Backend {:?} not available on this platform", backend_type),
    }
}
