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

/// Shared primitives reused across Vulkan, DX12, and Metal backends.
#[cfg(any(
    feature = "vulkan",
    all(feature = "dx12", target_os = "windows"),
    all(feature = "metal", target_os = "macos")
))]
pub(crate) mod shared;

use crate::types::{
    BackendType, BindlessHandle, BufferFlags, Color, DataAccess, DepthFormat, DepthStencilState,
    DeviceType, IndexFormat, PresentMode, PrimitiveTopology, SamplerDesc, SpatialAccess,
    TextureFlags, TextureFormat, VertexBufferLayout,
};
use anyhow::Result;
use std::sync::Arc;

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
                "slot {slot}: shader expects `{}` but got `{}` handle (index {})",
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

/// Validate that each bound buffer's `element_stride` matches the shader's
/// reflected expectation for that push-constant slot.
///
/// `buffer_strides[i]` is the actual `element_stride` of the buffer bound to
/// slot `i`.  `expected[i]` is the stride the shader expects (from Slang
/// reflection).  `None` on either side means "unknown / not applicable" and
/// is silently skipped.
///
/// Only runs when layout validation is enabled; designed for the bind hot
/// path — allocates only on the error path.
#[cfg(any(
    test,
    feature = "vulkan",
    all(feature = "dx12", target_os = "windows"),
    all(feature = "metal", target_os = "macos"),
))]
pub(crate) fn validate_binding_strides(
    buffer_strides: &[Option<u32>],
    expected: &[Option<u32>],
    shader_name: &str,
) -> Result<()> {
    let mut mismatches: Vec<String> = Vec::new();
    for (slot, actual) in buffer_strides.iter().enumerate() {
        let Some(exp) = expected.get(slot).copied().flatten() else {
            continue;
        };
        let Some(act) = actual else {
            continue;
        };
        if *act != exp {
            mismatches.push(format!(
                "slot {slot}: shader expects element stride {exp} but buffer has {act}"
            ));
        }
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        anyhow::bail!(
            "buffer element-stride mismatch in shader `{shader_name}`:\n  {}\n\
             Hint: ensure the buffer's element_stride (set at creation) matches \
             the size of the shader's StructuredBuffer<T> element type.",
            mismatches.join("\n  ")
        );
    }
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

/// Opaque token tying surface work to an acquired swapchain frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameToken {
    pub surface: SurfaceHandle,
    pub image: SwapchainImageHandle,
}

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
        data: Arc<[u8]>,
    },
    /// Upload CPU pixel data into a texture (full image), batched with the same submission
    /// as surrounding GPU work.
    WriteTexture {
        texture: TextureHandle,
        data: Arc<[u8]>,
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
        data: Arc<[u8]>,
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

/// Mixed compute + offscreen render commands from [`crate::task_graph::TaskGraph`].
#[derive(Debug, Clone)]
pub enum GraphCommand {
    /// Compute / upload / barrier from [`GpuCommand`].
    Compute(GpuCommand),
    /// Graphics work recorded against a [`RenderTargetHandle`] (offscreen render target).
    Render {
        target: RenderTargetHandle,
        commands: Vec<RenderCommand>,
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

    /// Create a buffer with a reserved byte capacity (`capacity >= initial_size`).
    /// Returns `(handle, actual_allocated_bytes)`. Default ignores `capacity` beyond `initial_size`.
    fn create_buffer_with_capacity(
        &mut self,
        device: DeviceHandle,
        initial_size: u64,
        capacity: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        let _ = capacity;
        let handle = self.create_buffer(device, initial_size, access, element_stride, flags)?;
        Ok((handle, initial_size))
    }

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

    /// Bytes reserved for this buffer (>= [`Self::buffer_size`]). Used for oversize reservations.
    fn buffer_capacity(&self, buffer: BufferHandle) -> u64 {
        self.buffer_size(buffer)
    }

    /// Update logical size without changing physical storage (must be `<= buffer_capacity()`).
    fn set_buffer_logical_size(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        new_logical_size: u64,
    ) -> Result<()>;

    /// Hint that bytes at and above `offset` may be discarded by the system (see [`crate::Buffer::hint_unused_above`]).
    fn hint_buffer_unused_above(&mut self, buffer: BufferHandle, offset: u64) {
        let _ = (buffer, offset);
    }

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

    /// Resize buffer storage in place. Logical handle and bindless offsets stay stable.
    fn resize_buffer(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        new_size: u64,
        preserve_contents: bool,
    ) -> Result<()>;

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

    // --- Timeline + explicit frame bracket ---

    /// Latest GPU completion point on this device's timeline (`value` is done when
    /// `gpu_progress() >= value`).
    fn gpu_progress(&self, device: DeviceHandle) -> crate::timeline::TimelineValue;

    fn wait_until(
        &mut self,
        device: DeviceHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<()>;

    fn wait_until_timeout(
        &mut self,
        device: DeviceHandle,
        value: crate::timeline::TimelineValue,
        timeout_ms: u32,
    ) -> Result<bool>;

    /// Submit compute (and transfer) commands on the device timeline, not tied to a surface frame.
    fn submit_standalone(
        &mut self,
        device: DeviceHandle,
        commands: &[GpuCommand],
    ) -> Result<crate::timeline::TimelineValue>;

    /// Submit an analyzed task graph with optional offscreen [`GraphCommand::Render`] segments.
    ///
    /// The default implementation is correct but suboptimal: it calls
    /// [`wait_until`](Self::wait_until) (CPU stall) between each compute batch and render
    /// pass to ensure GPU ordering. Backends should override this to record all commands
    /// into a single command buffer/list with GPU-side barriers, eliminating CPU waits.
    /// Metal, Vulkan, and DX12 all provide such overrides.
    fn submit_graph(
        &mut self,
        device: DeviceHandle,
        commands: &[GraphCommand],
    ) -> Result<crate::timeline::TimelineValue> {
        let mut batch: Vec<GpuCommand> = Vec::new();
        let mut last_tv = self.gpu_progress(device);
        for cmd in commands {
            match cmd {
                GraphCommand::Compute(c) => batch.push(c.clone()),
                GraphCommand::Render {
                    target,
                    commands: render_cmds,
                } => {
                    if !batch.is_empty() {
                        last_tv = self.submit_standalone(device, &batch)?;
                        self.wait_until(device, last_tv)?;
                        batch.clear();
                    }
                    self.render_to_target(device, *target, render_cmds)?;
                    last_tv = self.submit_standalone(device, &[])?;
                }
            }
        }
        if !batch.is_empty() {
            last_tv = self.submit_standalone(device, &batch)?;
        }
        Ok(last_tv)
    }

    /// Acquire the next swapchain image and begin a frame bracket.
    fn begin_frame(&mut self, surface: SurfaceHandle) -> Result<(FrameToken, TextureHandle)>;

    fn record_render(&mut self, frame: &FrameToken, commands: &[RenderCommand]) -> Result<()>;

    /// Record GPU work that must be ordered with the active surface frame (e.g. compute into the swapchain).
    fn record_gpu_work(&mut self, frame: &FrameToken, commands: &[GpuCommand]) -> Result<()>;

    /// End the frame: submit all recorded work, present, return the timeline value when the frame completes on the GPU.
    fn end_frame(&mut self, frame: FrameToken) -> Result<crate::timeline::TimelineValue>;

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
        let v = self.submit_standalone(device, commands)?;
        self.wait_until(device, v)
    }

    /// Notify the backend that a frame has completed and all transient buffers
    /// have been freed. Backends may use this to right-size internal heap
    /// allocations. No-op by default.
    fn reset_buffer_heaps(&mut self, _device: DeviceHandle) {}

    /// Ensure the internal heap/allocator can service a single allocation of
    /// at least `min_capacity` bytes without overflow. Call *after*
    /// `reset_buffer_heaps` and *before* large allocations. No-op by default.
    fn ensure_buffer_heap_capacity(&mut self, _device: DeviceHandle, _min_capacity: u64) {}

    /// Drop empty overflow heaps (both buffer and texture) after frame cleanup.
    /// Safe to call after retired buffers/textures have been dropped. No-op by default.
    fn compact_overflow_heaps(&mut self, _device: DeviceHandle) {}

    /// Drop backend-held shader compiler state (Metal: Slang session) to reduce host memory.
    ///
    /// Safe after all lazily-compiled shader stages are resident. A later compile will
    /// recreate the compiler automatically.
    fn release_idle_shader_compiler(&mut self) {}

    /// Number of bindless descriptor slots still available for allocation in `category`.
    ///
    /// Backends report `max - live` where *live* = allocated-and-not-yet-freed
    /// slots. Slots that are pending GPU-side reclamation (Metal's deferred
    /// timeline release) count as live until they are actually recycled.
    ///
    /// Returns `u32::MAX` by default (unlimited / not tracked).
    fn available_bindless_slots(
        &self,
        _device: DeviceHandle,
        _category: crate::types::BindlessCategory,
    ) -> u32 {
        u32::MAX
    }

    /// Maximum number of bindless descriptor slots per category for this backend.
    ///
    /// Returns `u32::MAX` by default (unlimited / not tracked).
    fn max_bindless_slots_per_category(
        &self,
        _device: DeviceHandle,
        _category: crate::types::BindlessCategory,
    ) -> u32 {
        u32::MAX
    }

    /// Process pending GPU deletions and reclaim bindless descriptor slots
    /// whose GPU timeline barrier has been signaled.
    ///
    /// Normally called internally at acquire/present/submit, but consumers
    /// that drop buffers between those points (e.g. during a non-blocking
    /// frame drain) can call this explicitly to reclaim slots immediately.
    fn flush_deferred_deletions(&mut self, _device: DeviceHandle) {}

    /// Resources queued for destruction after the GPU timeline advances (for tests).
    #[doc(hidden)]
    fn deferred_deletion_pending_count(&self, _device: DeviceHandle) -> usize {
        0
    }
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

#[cfg(test)]
mod push_constant_validation_tests {
    use super::validate_typed_push_constants;
    use crate::types::{BindlessCategory, BindlessHandle};

    #[test]
    fn valid_categories_pass() {
        let handles = vec![
            BindlessHandle::new(BindlessCategory::Scattered, 0),
            BindlessHandle::new(BindlessCategory::Broadcast, 1),
        ];
        let expectations = vec![
            Some(BindlessCategory::Scattered),
            Some(BindlessCategory::Broadcast),
        ];
        validate_typed_push_constants(&handles, &expectations, "test_shader").unwrap();
    }

    #[test]
    fn none_expectations_are_skipped() {
        let handles = vec![
            BindlessHandle::new(BindlessCategory::Scattered, 0),
            BindlessHandle::new(BindlessCategory::Texture, 1),
        ];
        let expectations = vec![None, None];
        validate_typed_push_constants(&handles, &expectations, "test_shader").unwrap();
    }

    #[test]
    fn empty_expectations_passes_any_handles() {
        let handles = vec![
            BindlessHandle::new(BindlessCategory::Scattered, 5),
            BindlessHandle::new(BindlessCategory::Sampler, 2),
        ];
        validate_typed_push_constants(&handles, &[], "test_shader").unwrap();
    }

    #[test]
    fn scattered_where_broadcast_expected_fails() {
        let handles = vec![BindlessHandle::new(BindlessCategory::Scattered, 0)];
        let expectations = vec![Some(BindlessCategory::Broadcast)];
        let err = validate_typed_push_constants(&handles, &expectations, "my_shader")
            .unwrap_err()
            .to_string();
        assert!(err.contains("slot 0"), "error should name the slot: {err}");
        assert!(
            err.contains("broadcast"),
            "error should mention expected category: {err}"
        );
        assert!(
            err.contains("scattered"),
            "error should mention actual category: {err}"
        );
    }

    #[test]
    fn texture_where_scattered_expected_fails() {
        let handles = vec![
            BindlessHandle::new(BindlessCategory::Scattered, 0),
            BindlessHandle::new(BindlessCategory::Texture, 3),
        ];
        let expectations = vec![
            Some(BindlessCategory::Scattered),
            Some(BindlessCategory::Scattered),
        ];
        let err = validate_typed_push_constants(&handles, &expectations, "compute_cs")
            .unwrap_err()
            .to_string();
        assert!(err.contains("slot 1"), "error should name slot 1: {err}");
        assert!(
            err.contains("texture"),
            "error should mention actual: {err}"
        );
    }

    #[test]
    fn multiple_mismatches_reported() {
        let handles = vec![
            BindlessHandle::new(BindlessCategory::Texture, 0),
            BindlessHandle::new(BindlessCategory::Sampler, 1),
        ];
        let expectations = vec![
            Some(BindlessCategory::Scattered),
            Some(BindlessCategory::Broadcast),
        ];
        let err = validate_typed_push_constants(&handles, &expectations, "sh")
            .unwrap_err()
            .to_string();
        assert!(err.contains("slot 0"), "should report slot 0: {err}");
        assert!(err.contains("slot 1"), "should report slot 1: {err}");
    }
}

#[cfg(test)]
mod binding_stride_validation_tests {
    use super::validate_binding_strides;

    #[test]
    fn matching_strides_pass() {
        let actual = vec![Some(4), Some(16)];
        let expected = vec![Some(4), Some(16)];
        validate_binding_strides(&actual, &expected, "test").unwrap();
    }

    #[test]
    fn none_expected_skipped() {
        let actual = vec![Some(4), Some(8)];
        let expected: Vec<Option<u32>> = vec![None, None];
        validate_binding_strides(&actual, &expected, "test").unwrap();
    }

    #[test]
    fn none_actual_skipped() {
        let actual: Vec<Option<u32>> = vec![None, None];
        let expected = vec![Some(4), Some(16)];
        validate_binding_strides(&actual, &expected, "test").unwrap();
    }

    #[test]
    fn empty_expected_passes() {
        let actual = vec![Some(4), Some(8)];
        validate_binding_strides(&actual, &[], "test").unwrap();
    }

    #[test]
    fn stride_mismatch_detected() {
        let actual = vec![Some(4)];
        let expected = vec![Some(16)];
        let err = validate_binding_strides(&actual, &expected, "my_shader")
            .unwrap_err()
            .to_string();
        assert!(err.contains("slot 0"), "should name slot: {err}");
        assert!(err.contains("16"), "should mention expected stride: {err}");
        assert!(err.contains("4"), "should mention actual stride: {err}");
    }

    #[test]
    fn multiple_stride_mismatches_reported() {
        let actual = vec![Some(4), Some(8)];
        let expected = vec![Some(16), Some(32)];
        let err = validate_binding_strides(&actual, &expected, "cs")
            .unwrap_err()
            .to_string();
        assert!(err.contains("slot 0"), "should report slot 0: {err}");
        assert!(err.contains("slot 1"), "should report slot 1: {err}");
    }
}
