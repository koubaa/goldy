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

/// Shared primitives reused across Vulkan, DX12, and Metal backends, and by
/// task-graph command emission (e.g. `DispatchBatch` argument packing).
pub(crate) mod shared;

/// Fence/timeline polling threads for async [`crate::signal::Signal`] delivery (Vulkan, DX12).
#[cfg(any(feature = "vulkan", all(feature = "dx12", target_os = "windows")))]
pub(crate) mod signal_fence;

use crate::types::{
    BackendType, BufferFlags, BufferKind, Color, DepthFormat, DepthStencilState, DeviceType, IndexFormat, PresentMode,
    PrimitiveTopology, ResourceHandle, SamplerDesc, TextureFlags, TextureFormat, TextureKind, VertexBufferLayout,
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
use crate::types::ResourceCategory;

#[cfg(all(feature = "dx12", target_os = "windows"))]
use crate::types::BindlessSlotKind;

/// Gate for dispatch paths: run `f` only when layout validation is enabled.
///
/// Pure validators ([`validate_raw_binding_strides`], etc.) contain the check logic
/// and never read env vars. Unit tests call those directly; backends call them
/// through this wrapper.
#[cfg(any(
    test,
    feature = "vulkan",
    all(feature = "dx12", target_os = "windows"),
    all(feature = "metal", target_os = "macos"),
))]
#[inline]
pub(crate) fn with_layout_validation<F>(f: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    if crate::slang::layout_validation_enabled() {
        f()
    } else {
        Ok(())
    }
}

/// Validate raw bindless indices against per-slot SRV/UAV expectations (DX12).
///
#[cfg(all(feature = "dx12", target_os = "windows"))]
pub(crate) fn validate_bindless_slot_kinds(
    indices: &[u32],
    expectations: &[Option<BindlessSlotKind>],
    mut resolve: impl FnMut(u32) -> Option<BindlessSlotKind>,
    shader_name: &str,
) -> Result<()> {
    if expectations.is_empty() {
        return Ok(());
    }
    let mut mismatches: Vec<String> = Vec::new();
    for (slot, &index) in indices.iter().enumerate() {
        let Some(expected) = expectations.get(slot).copied().flatten() else {
            continue;
        };
        let Some(actual) = resolve(index) else {
            continue;
        };
        if actual != expected {
            mismatches.push(format!(
                "slot {slot}: shader expects {expected} bindless slot but index {index} resolves to {actual}",
                expected = expected.name(),
                actual = actual.name(),
            ));
        }
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        anyhow::bail!(
            "bindless SRV/UAV mismatch in shader `{shader_name}`:\n  {}\n\
             Hint: `Scattered<T>` / `StorageBuffer<T>` need `ResourceAccess::ReadWrite` or `Write` \
             (UAV index); `BufRO<T>` needs `ResourceAccess::Read` (SRV index). \
             Prefer `bind_resources(&[&buf])` or `Buffer::handle(access)` over raw indices.",
            mismatches.join("\n  ")
        );
    }
}

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
    handles: &[ResourceHandle],
    expectations: &[Option<ResourceCategory>],
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
             Hint: use `Buffer::handle(access)` / `Texture::handle(access)` \
             rather than raw backend indices so the resource's category flows \
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
             Hint: for Scattered<T> / BufRO<T> parameters, ensure the buffer's element_stride \
             matches sizeof(T) as declared in the shader.\n\
             For Broadcast (constant-buffer) parameters, use acquire_buffer_sized::<T>() where T \
             exactly matches the shader struct (e.g. #[repr(C)] with the same fields and order).",
            mismatches.join("\n  ")
        );
    }
}

/// Validate buffer element strides for a compute dispatch bound via raw bindless indices.
///
/// For each push-constant slot with a reflected stride expectation, resolves the bound
/// buffer's `element_stride` through `resolve_stride(index, category)` and delegates
/// to [`validate_binding_strides`].
///
/// Also reports any Scattered/Broadcast slot whose index is missing from `indices`.
///
#[cfg(any(
    test,
    feature = "vulkan",
    all(feature = "dx12", target_os = "windows"),
    all(feature = "metal", target_os = "macos"),
))]
pub(crate) fn validate_raw_binding_strides(
    indices: &[u32],
    categories: &[Option<crate::types::ResourceCategory>],
    expected: &[Option<u32>],
    mut resolve_stride: impl FnMut(u32, crate::types::ResourceCategory) -> Option<u32>,
    shader_name: &str,
) -> Result<()> {
    if expected.is_empty() {
        return Ok(());
    }
    let mut actual: Vec<Option<u32>> = vec![None; expected.len()];
    let mut missing_slots: Vec<usize> = Vec::new();
    for (slot, exp) in expected.iter().enumerate() {
        if exp.is_none() {
            continue;
        }
        let Some(cat) = categories.get(slot).and_then(|c| *c) else {
            continue;
        };
        if !matches!(
            cat,
            crate::types::ResourceCategory::Scattered | crate::types::ResourceCategory::Broadcast
        ) {
            continue;
        }
        match indices.get(slot) {
            Some(&idx) => actual[slot] = resolve_stride(idx, cat),
            None => missing_slots.push(slot),
        }
    }
    if !missing_slots.is_empty() {
        let slots: Vec<String> = missing_slots.iter().map(|s| s.to_string()).collect();
        anyhow::bail!(
            "bind_resources_raw for shader `{shader_name}` is missing indices for \
             slot(s) {}: shader has {} reflected binding slot(s) but only {} index/indices \
             were provided.\n\
             Hint: pass one raw bindless index per Scattered/Broadcast parameter in the \
             shader signature, in declaration order.",
            slots.join(", "),
            expected.len(),
            indices.len(),
        );
    }
    validate_binding_strides(&actual, expected, shader_name)
}

/// Validate buffer strides for legacy `RenderCommand::BindResources` before lowering.
///
/// Call from `prepare_render_commands` while buffer handles are still available.
/// Legacy bind commands bail at record time; this is the only place stride checks run
/// for standalone render passes.
#[cfg(any(
    test,
    feature = "vulkan",
    all(feature = "dx12", target_os = "windows"),
    all(feature = "metal", target_os = "macos"),
))]
pub(crate) fn validate_render_pass_bind_resources<F, G>(
    commands: &[RenderCommand],
    mut pipeline_strides: F,
    mut buffer_stride: G,
) -> Result<()>
where
    F: FnMut(PipelineHandle) -> Option<(Vec<Option<u32>>, String)>,
    G: FnMut(BufferHandle) -> Option<u32>,
{
    let mut current_pipeline: Option<PipelineHandle> = None;
    for cmd in commands {
        match cmd {
            RenderCommand::SetPipeline(h) => current_pipeline = Some(*h),
            RenderCommand::BindResources { buffers } => {
                if let Some(ph) = current_pipeline {
                    if let Some((expected, name)) = pipeline_strides(ph) {
                        if expected.is_empty() {
                            continue;
                        }
                        let actual: Vec<Option<u32>> = buffers.iter().map(|h| buffer_stride(*h)).collect();
                        validate_binding_strides(&actual, &expected, &name)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
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
pub type ContextHandle = u64;
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
    /// WSI swapchain image index (which drawable will be presented).
    pub image: SwapchainImageHandle,
    /// Submission context that owns this frame's timeline (set by [`crate::surface::Frame`]).
    pub context: ContextHandle,
    /// In-flight slot index for the compute/scratch texture bound this frame.
    ///
    /// Used for present-lease retention keys: must match the physical backing that
    /// shader dispatches and copies target, not necessarily [`Self::image`].
    /// On Vulkan this is `current_frame`; on DX12 it equals the swapchain image index.
    pub frame_slot: u32,
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
    BindResourcesRaw {
        indices: Vec<u32>,
        user: Vec<u32>,
        /// Offset within the frame-table row where this draw's indices live.
        frame_table_base: u32,
    },
    /// Bind resource slots with typed [`ResourceHandle`]s. Backends validate
    /// each handle's [`crate::types::ResourceCategory`]
    /// against the bound shader's reflection and emit the raw indices.
    BindResourcesTyped { handles: Vec<ResourceHandle> },
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

/// Layout of a grant-readback staging buffer for a 2D texture copy.
///
/// `logical_bytes` is the tight linear size clients observe (`width * height * bpp`).
/// `staging_bytes` and `row_pitch` describe the GPU/readback layout (may include padding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureReadbackLayout {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub logical_bytes: u64,
    pub staging_bytes: u64,
    pub row_pitch: u32,
    /// Byte offset of the subresource footprint within the staging buffer (DX12 placed copy).
    pub footprint_offset: u64,
}

impl TextureReadbackLayout {
    pub fn tight_row_bytes(&self) -> u32 {
        self.width.saturating_mul(self.format.bytes_per_pixel())
    }
}

/// Cross-submission synchronization derived from the resource epoch ledger.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubmitSync {
    /// Same-context scoped memory barrier prepended before command execution.
    pub prologue: crate::task_graph::BarrierSet,
    /// Cross-context GPU queue-waits on producer timeline values.
    pub waits: Vec<crate::timeline::Epoch>,
}

impl SubmitSync {
    pub fn is_empty(&self) -> bool {
        self.prologue.is_empty() && self.waits.is_empty()
    }

    pub fn from_cross_submit(cross: &crate::task_graph::CrossSubmitSync) -> Self {
        Self {
            prologue: cross.prologue.clone(),
            waits: cross.waits.clone(),
        }
    }

    /// True when the backend should emit the legacy blanket cross-submission acquire.
    pub fn use_legacy_acquire(&self) -> bool {
        false
    }
}

/// GPU queue-wait epochs before submit. Mock uses CPU `device_wait_until`; real backends
/// should prefer native queue-waits and treat this as a fallback.
pub(crate) fn apply_submit_sync_waits(
    backend: &mut dyn GpuBackend,
    device: DeviceHandle,
    sync: Option<&SubmitSync>,
) -> Result<()> {
    if let Some(s) = sync {
        for epoch in &s.waits {
            backend.device_wait_until(device, epoch.value)?;
        }
    }
    Ok(())
}

/// Prepend cross-submit prologue commands when executing a submit with sync info.
pub(crate) fn commands_with_sync_prologue(commands: &[GpuCommand], sync: Option<&SubmitSync>) -> Vec<GpuCommand> {
    if let Some(s) = sync {
        if !s.prologue.is_empty() {
            return crate::task_graph::cross_submit::prepend_prologue(commands, &s.prologue);
        }
    }
    commands.to_vec()
}

/// GPU command recorded into a command buffer / task-graph submission.
///
/// Includes compute dispatches, buffer upload/clear, texture uploads, and
/// scheduling barriers — not compute-only despite historical naming in call sites.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuCommand {
    /// Set the active compute pipeline.
    SetPipeline(ComputePipelineHandle),
    /// Bind resource slots with raw u32 indices (for textures/samplers or mixed resources).
    ///
    /// `indices` go to region A (bindless, packed as u16).
    /// `user` go to region B (user scalars, full u32).
    ///
    /// **Prefer [`GpuCommand::BindResourcesTyped`]** — the raw form
    /// bypasses per-slot category validation.
    BindResourcesRaw {
        indices: Vec<u32>,
        user: Vec<u32>,
        /// Offset within the frame-table row where this dispatch's indices live.
        frame_table_base: u32,
    },
    /// Bind resource slots with typed [`ResourceHandle`]s. Backends validate
    /// each handle's [`crate::types::ResourceCategory`]
    /// against the bound shader's reflection and emit the raw indices.
    BindResourcesTyped { handles: Vec<ResourceHandle> },
    /// Dispatch compute workgroups.
    Dispatch {
        /// Debug label from [`crate::task_graph::TaskGraph::node`] when emitted by the analyzer.
        label: Option<&'static str>,
        workgroups_x: u32,
        workgroups_y: u32,
        workgroups_z: u32,
    },
    /// Indirect dispatch: workgroup counts read from buffer at offset (3× u32: x, y, z).
    DispatchIndirect {
        label: Option<&'static str>,
        buffer: BufferHandle,
        offset: u64,
    },
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
    /// Copy the entire contents of `src` texture into `dst`.
    ///
    /// Both textures must have compatible formats and identical dimensions.
    /// The backend inserts appropriate layout transitions and memory barriers.
    CopyTexture { src: TextureHandle, dst: TextureHandle },
    /// Copy the color attachment of an offscreen render target into a texture.
    ///
    /// The source must have been rendered to earlier in the same submission (or
    /// a prior submission whose timeline has completed). The backend transitions
    /// the source from its post-render layout and copies into `dst`.
    CopyRenderTarget {
        src: RenderTargetHandle,
        dst: TextureHandle,
    },
    /// Copy bytes from `src` buffer into `dst` buffer (grant-read staging path).
    CopyBuffer {
        src: BufferHandle,
        dst: BufferHandle,
        size: u64,
    },
    /// Copy a texture subresource into a grant-readback staging buffer (placed footprint).
    CopyTextureToReadback {
        src: TextureHandle,
        dst: BufferHandle,
        layout: TextureReadbackLayout,
    },
    /// Batched indirect dispatch: multiple consecutive dispatches sharing the same pipeline.
    ///
    /// Each entry packs `[PushLayout bytes | wg_x u32 | wg_y u32 | wg_z u32]` into
    /// `arg_data` (`DISPATCH_BATCH_STRIDE` bytes per entry).  The backend records a
    /// single `ExecuteIndirect` on DX12 or iterates on Vulkan/Metal.
    ///
    /// Only emitted by `emit_commands` when consecutive same-pipeline dispatches
    /// are detected within a wave.  Falls back to individual `Dispatch` commands
    /// if no grouping is possible.
    DispatchBatch {
        label: Option<&'static str>,
        /// Pre-filled argument data: `count` entries of `DISPATCH_BATCH_STRIDE` bytes each.
        arg_data: Arc<[u8]>,
        count: u32,
    },
    /// Frame-table staging payload — written to the upload staging buffer and copied
    /// to the device-local table by the prologue at the start of each submission.
    FrameTableStaging { data: std::sync::Arc<[u32]> },
    ///
    /// Within a [`crate::task_graph::TaskGraph`] submission, prefer
    /// `ResourceBarrier` which is produced by the scheduler with precise
    /// `src_usage` / `dst_usage` derived from the dependency graph.
    Barrier,
    /// Per-resource memory barrier with full access semantics.
    ///
    /// Emitted by the compute graph scheduler at dependency edges.
    /// Each `(handle, BarrierUsage)` pair describes what kind of GPU work produced
    /// and will consume that specific resource.  Each backend lowers them to its
    /// native synchronization primitives without needing to infer access from
    /// surrounding commands.
    ResourceBarrier {
        buffers: Vec<(BufferHandle, crate::task_graph::BarrierUsage)>,
        textures: Vec<(TextureHandle, crate::task_graph::BarrierUsage)>,
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
    /// Downcast to `&mut dyn std::any::Any` for test introspection.
    #[doc(hidden)]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Get the backend type.
    fn backend_type(&self) -> BackendType;

    /// Enumerate available adapters.
    fn enumerate_adapters(&self) -> Vec<AdapterInfo>;

    /// Immutable capability snapshot for a physical adapter (no logical device required).
    fn adapter_capabilities(&self, adapter_id: u32) -> crate::device::DeviceCapabilities {
        let _ = adapter_id;
        crate::device::DeviceCapabilities::default()
    }

    // Device management
    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle>;
    fn destroy_device(&mut self, device: DeviceHandle);
    fn is_device_valid(&self, device: DeviceHandle) -> bool;

    /// Block until all GPU work on this device has completed (teardown primitive).
    fn device_wait_idle(&mut self, device: DeviceHandle) -> Result<()>;

    // Submission context (timeline / submit / reclaim API is keyed by context)
    fn create_context(&mut self, device: DeviceHandle) -> Result<ContextHandle>;
    fn destroy_context(&mut self, ctx: ContextHandle);

    /// Returns `true` if the device has been permanently lost (TDR, hardware hang, etc.).
    ///
    /// Backends set this flag atomically when they detect device loss so that
    /// [`Device::is_device_lost`](crate::Device::is_device_lost) can be polled from the
    /// render loop without acquiring any lock. The default returns `false` for
    /// backends that have not yet wired up the flag.
    fn is_device_lost(&self, _device: DeviceHandle) -> bool {
        false
    }

    // Buffer management
    fn create_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        access: BufferKind,
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
        access: BufferKind,
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
    fn read_buffer_to_cpu(&mut self, device: DeviceHandle, buffer: BufferHandle, output: &mut [u8]) -> Result<()>;
    /// Allocate a persistently mapped READBACK staging buffer for grant readback (no bindless slot).
    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle>;
    /// Read bytes from a buffer created by [`Self::alloc_readback_buffer`].
    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()>;
    /// Release a grant-readback staging buffer.
    fn free_readback_buffer(&mut self, buffer: BufferHandle);
    /// Query copy/readback layout for a 2D texture grant (uncompressed formats only in v1).
    fn query_texture_readback_layout(
        &self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<TextureReadbackLayout>;
    /// Allocate a persistently mapped READBACK staging buffer for a texture grant.
    fn alloc_texture_readback_staging(
        &mut self,
        device: DeviceHandle,
        layout: TextureReadbackLayout,
    ) -> Result<BufferHandle>;
    /// Read tight linear bytes from a texture grant staging buffer.
    fn read_texture_readback_staging(
        &self,
        buffer: BufferHandle,
        layout: TextureReadbackLayout,
        output: &mut [u8],
    ) -> Result<()>;

    /// Mock-backend grant readback allocation counter (tests only).
    #[doc(hidden)]
    #[cfg(test)]
    fn test_readback_alloc_count(&self) -> usize {
        let _ = self;
        0
    }

    /// Mock-backend grant readback free counter (tests only).
    #[doc(hidden)]
    #[cfg(test)]
    fn test_readback_free_count(&self) -> usize {
        let _ = self;
        0
    }

    /// Mock-backend surface present counter (tests only).
    #[doc(hidden)]
    #[cfg(test)]
    fn test_surface_present_count(&self) -> usize {
        let _ = self;
        0
    }

    /// Mock-backend wait_until call counter (tests only).
    #[doc(hidden)]
    #[cfg(test)]
    fn test_wait_until_count(&self) -> usize {
        let _ = self;
        0
    }

    /// Capability snapshot for `device` (surface formats, [`crate::device::DeviceCapabilities::has_zero_copy_storage_readback`], …).
    fn device_capabilities(&self, device: DeviceHandle) -> crate::device::DeviceCapabilities {
        let _ = device;
        crate::device::DeviceCapabilities::default()
    }
    /// Fill buffer region with zeros. If size is 0, clears from offset to end of buffer.
    fn clear_buffer(&mut self, device: DeviceHandle, buffer: BufferHandle, offset: u64, size: u64) -> Result<()>;
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
            self.create_shader_with_paths(device, slang_source, search_paths, defines, optimization_level)
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
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<TextureHandle>;
    fn write_texture(&mut self, texture: TextureHandle, data: &[u8], width: u32, height: u32) -> Result<()>;
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

    /// For `TextureKind::DirectInterpolated` textures, return the sampled-texture-pool
    /// (SRV) bindless index.  Returns `None` for textures without a secondary SRV slot.
    fn texture_bindless_sampled_index(&self, texture: TextureHandle) -> Option<u32>;

    // Sampler management
    fn create_sampler(&mut self, device: DeviceHandle, desc: &SamplerDesc) -> Result<SamplerHandle>;
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
    fn surface_set_present_mode(&mut self, _surface: SurfaceHandle, _mode: PresentMode) -> Result<()> {
        Ok(())
    }

    /// Get the current present mode for a surface.
    fn surface_present_mode(&self, _surface: SurfaceHandle) -> PresentMode {
        PresentMode::Auto
    }

    // --- Timeline + explicit frame bracket ---

    /// Latest GPU completion point on this context's timeline (`value` is done when
    /// `gpu_progress() >= value`).
    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue;

    /// Latest device-global submission sequence retired on the GPU (shared queue / seq space).
    ///
    /// `value` is done when `device_timeline_retired() >= value`. This is the max over live
    /// context completion primitives, the device sync fence (DX12), and the post-destroy floor.
    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue;

    /// Block until the device-global timeline has retired at least `value`.
    ///
    /// Unlike [`Self::wait_until`] (which is per-context), this searches across all live contexts on
    /// `device` for the one that signaled `value` and waits on its native primitive. Use this
    /// when the `TimelineValue` was produced by an arbitrary context — e.g. from outside the
    /// allocator — so you don't need a matching `ContextHandle`.
    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> Result<()>;

    /// Drain pending backend signals for this context (async queue + synchronous oversubscribed).
    fn poll_signals(&mut self, ctx: ContextHandle) -> Vec<crate::signal::Signal>;

    /// Oldest timeline ticket not yet retired by the GPU, if any work is still in flight.
    fn peek_oldest_in_flight(&self, ctx: ContextHandle) -> Option<crate::timeline::TimelineValue>;

    /// Number of swapchain drawables held by the client / GPU and not yet returned by the compositor.
    fn pending_acquire_count(&self, surface: SurfaceHandle) -> u32;

    fn wait_until(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()>;

    fn wait_until_timeout(
        &mut self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
        timeout_ms: u32,
    ) -> Result<bool>;

    /// Submit compute (and transfer) commands on the context timeline, not tied to a surface frame.
    ///
    /// When `sync` is `Some`, the backend emits the scoped prologue barrier and GPU-side
    /// queue-waits instead of the legacy blanket cross-submission acquire.
    fn submit_standalone(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
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
        ctx: ContextHandle,
        commands: &[GraphCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        let mut batch: Vec<GpuCommand> = Vec::new();
        let mut last_tv = self.gpu_progress(ctx);
        for cmd in commands {
            match cmd {
                GraphCommand::Compute(c) => batch.push(c.clone()),
                GraphCommand::Render {
                    target,
                    commands: render_cmds,
                } => {
                    if !batch.is_empty() {
                        last_tv = self.submit_standalone(ctx, &batch, sync)?;
                        self.wait_until(ctx, last_tv)?;
                        batch.clear();
                    }
                    let device = self.context_device(ctx);
                    self.render_to_target(device, *target, render_cmds)?;
                    last_tv = self.submit_standalone(ctx, &[], sync)?;
                }
            }
        }
        if !batch.is_empty() {
            last_tv = self.submit_standalone(ctx, &batch, sync)?;
        }
        Ok(last_tv)
    }

    /// Resolve the logical device for a submission context.
    fn context_device(&self, ctx: ContextHandle) -> DeviceHandle;

    /// Submit commands and retain the closed command list for potential reuse on the next
    /// cache-hit frame.
    ///
    /// When `key` matches the key passed on the previous call AND the GPU resources
    /// referenced by those commands have stable bindless handles, callers may call
    /// [`Self::try_resubmit_retained`] instead of re-recording.
    ///
    /// The DX12 backend implements this by keeping the closed `ID3D12GraphicsCommandList`
    /// alive (without `Reset`) and re-executing it via `ExecuteCommandLists`.  Other
    /// backends default to a normal [`Self::submit_graph`] (safe fallback, no caching).
    fn submit_graph_and_retain(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        let _ = key;
        self.submit_graph(ctx, commands, sync)
    }

    /// Re-execute a previously retained command list without re-recording.
    ///
    /// Returns `Ok(Some(tv))` if the retained list for `key` was found and resubmitted.
    /// Returns `Ok(None)` if no matching retained list exists (caller should fall back to
    /// a full [`Self::submit_graph_and_retain`]).  Returns `Err` on a backend error.
    fn try_resubmit_retained(
        &mut self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<crate::timeline::TimelineValue>> {
        let _ = (ctx, key, sync);
        Ok(None)
    }

    /// Whether the backend can retain present-touching partitions across submits.
    ///
    /// Backends that resolve the drawable to a fresh [`TextureHandle`] each frame
    /// (Metal) return `false` so the present partition is re-resolved and re-recorded
    /// each submit rather than resubmitting a stale cached command list.
    fn retains_present_partitions(&self) -> bool {
        true
    }

    /// Drop the retained command list associated with `key`, marking its allocator slot
    /// as available for re-use.  No-op if no retained list exists.
    fn evict_retained(&mut self, _ctx: ContextHandle, _key: u64) {}

    /// Acquire the next swapchain image and begin a frame bracket.
    ///
    /// `ctx` is the submission context that owns this surface's timeline; frame
    /// submit/present and swapchain signals are routed through it.
    fn begin_frame(&mut self, surface: SurfaceHandle, ctx: ContextHandle) -> Result<(FrameToken, TextureHandle)>;

    fn record_render(&mut self, frame: &FrameToken, commands: &[RenderCommand]) -> Result<()>;

    /// Record GPU work that must be ordered with the active surface frame (e.g. compute into the swapchain).
    fn record_gpu_work(&mut self, frame: &FrameToken, commands: &[GpuCommand]) -> Result<()>;

    /// Submit all recorded GPU work for this frame bracket. Does not present.
    ///
    /// Returns the timeline value signaled when the frame's compute (and transfer) work
    /// completes on the GPU. When no work was recorded, returns the latest completed
    /// or scheduled compute timeline appropriate for the backend.
    fn submit_frame(&mut self, frame: &FrameToken) -> Result<crate::timeline::TimelineValue>;

    /// Present the swapchain image for this frame after [`Self::submit_frame`].
    ///
    /// `submit_tv` is the value returned by [`Self::submit_frame`] so backends that use
    /// separate present queues can wait for compute before presenting.
    fn present_frame(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue>;

    /// Submit recorded work and present. Convenience for callers that do not split the bracket.
    ///
    /// Default implementation calls [`Self::submit_frame`] then [`Self::present_frame`].
    /// The returned timeline is from present (when present allocates its own signal) or
    /// from submit when present reuses the submit timeline.
    fn end_frame(&mut self, frame: FrameToken) -> Result<crate::timeline::TimelineValue> {
        let submit_tv = self.submit_frame(&frame)?;
        self.present_frame(frame, submit_tv)
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
    fn available_bindless_slots(&self, _device: DeviceHandle, _category: crate::types::ResourceCategory) -> u32 {
        u32::MAX
    }

    /// Maximum number of bindless descriptor slots per category for this backend.
    ///
    /// Returns `u32::MAX` by default (unlimited / not tracked).
    fn max_bindless_slots_per_category(&self, _device: DeviceHandle, _category: crate::types::ResourceCategory) -> u32 {
        u32::MAX
    }

    /// Process pending GPU deletions and reclaim bindless descriptor slots
    /// whose GPU timeline barrier has been signaled.
    ///
    /// Normally called internally at acquire/present/submit, but consumers
    /// that drop buffers between those points (e.g. during a non-blocking
    /// frame drain) can call this explicitly to reclaim slots immediately.
    fn flush_deferred_deletions(&mut self, _ctx: ContextHandle) {}

    /// Install a per-thread reclamation epoch for the next deferred-payload drop window.
    ///
    /// Metal uses this so `Buffer::drop` during [`crate::Context::boundary_crossed`] queues heap
    /// frees with the already-retired epoch instead of `timeline_scheduled_max`. Only the
    /// installing thread observes the override; other threads keep conservative barriers.
    fn set_reclamation_context(&mut self, _ctx: ContextHandle, _epoch: Option<crate::timeline::TimelineValue>) {}

    /// Resources queued for destruction after the GPU timeline advances (for tests).
    #[doc(hidden)]
    fn deferred_deletion_pending_count(&self, _ctx: ContextHandle) -> usize {
        0
    }

    /// Snapshot of the buffer heap allocator state (overflow count, buffer count, etc.).
    /// Only meaningful on Metal; other backends return `None`.
    #[doc(hidden)]
    fn buffer_heap_stats(&self, _device: DeviceHandle) -> Option<BufferHeapStats> {
        None
    }

    /// Snapshot of the texture heap allocator state.
    /// Only meaningful on Metal; other backends return `None`.
    #[doc(hidden)]
    fn texture_heap_stats(&self, _device: DeviceHandle) -> Option<TextureHeapStats> {
        None
    }

    /// Number of in-flight command buffers tracked by the backend for wait-reclaim.
    /// Only meaningful on Metal; other backends return 0.
    #[doc(hidden)]
    fn in_flight_command_buffer_count(&self, _ctx: ContextHandle) -> usize {
        0
    }
}

/// Snapshot of a Metal buffer heap allocator's state.
#[derive(Debug, Clone, Copy, Default)]
pub struct BufferHeapStats {
    /// Total number of buffers ever allocated from the heap hierarchy (monotonically increasing).
    /// This counter does NOT decrease when buffers are freed.
    pub buffer_count: u32,
    /// Number of overflow heaps currently alive (0 in steady state).
    pub overflow_count: usize,
    /// Peak total bytes used across all heaps since last reset.
    pub high_water_bytes: u64,
    /// Size of the primary heap in bytes.
    pub primary_heap_bytes: u64,
}

/// Snapshot of a Metal texture heap allocator's state.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextureHeapStats {
    /// Number of live textures currently allocated from the heap hierarchy.
    pub texture_count: u32,
    /// Number of overflow heaps currently alive (0 in steady state).
    pub overflow_count: usize,
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
        tracing::info!("Using backend from GOLDY_BACKEND env var: {:?}", backend_type);
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
    use crate::types::{ResourceCategory, ResourceHandle};

    #[test]
    fn valid_categories_pass() {
        let handles = vec![
            ResourceHandle::new(ResourceCategory::Scattered, 0),
            ResourceHandle::new(ResourceCategory::Broadcast, 1),
        ];
        let expectations = vec![Some(ResourceCategory::Scattered), Some(ResourceCategory::Broadcast)];
        validate_typed_push_constants(&handles, &expectations, "test_shader").unwrap();
    }

    #[test]
    fn none_expectations_are_skipped() {
        let handles = vec![
            ResourceHandle::new(ResourceCategory::Scattered, 0),
            ResourceHandle::new(ResourceCategory::Texture, 1),
        ];
        let expectations = vec![None, None];
        validate_typed_push_constants(&handles, &expectations, "test_shader").unwrap();
    }

    #[test]
    fn empty_expectations_passes_any_handles() {
        let handles = vec![
            ResourceHandle::new(ResourceCategory::Scattered, 5),
            ResourceHandle::new(ResourceCategory::Sampler, 2),
        ];
        validate_typed_push_constants(&handles, &[], "test_shader").unwrap();
    }

    #[test]
    fn scattered_where_broadcast_expected_fails() {
        let handles = vec![ResourceHandle::new(ResourceCategory::Scattered, 0)];
        let expectations = vec![Some(ResourceCategory::Broadcast)];
        let err = validate_typed_push_constants(&handles, &expectations, "my_shader")
            .unwrap_err()
            .to_string();
        assert!(err.contains("slot 0"), "error should name the slot: {err}");
        assert!(
            err.contains("broadcast"),
            "error should mention expected category: {err}"
        );
        assert!(err.contains("scattered"), "error should mention actual category: {err}");
    }

    #[test]
    fn texture_where_scattered_expected_fails() {
        let handles = vec![
            ResourceHandle::new(ResourceCategory::Scattered, 0),
            ResourceHandle::new(ResourceCategory::Texture, 3),
        ];
        let expectations = vec![Some(ResourceCategory::Scattered), Some(ResourceCategory::Scattered)];
        let err = validate_typed_push_constants(&handles, &expectations, "compute_cs")
            .unwrap_err()
            .to_string();
        assert!(err.contains("slot 1"), "error should name slot 1: {err}");
        assert!(err.contains("texture"), "error should mention actual: {err}");
    }

    #[test]
    fn multiple_mismatches_reported() {
        let handles = vec![
            ResourceHandle::new(ResourceCategory::Texture, 0),
            ResourceHandle::new(ResourceCategory::Sampler, 1),
        ];
        let expectations = vec![Some(ResourceCategory::Scattered), Some(ResourceCategory::Broadcast)];
        let err = validate_typed_push_constants(&handles, &expectations, "sh")
            .unwrap_err()
            .to_string();
        assert!(err.contains("slot 0"), "should report slot 0: {err}");
        assert!(err.contains("slot 1"), "should report slot 1: {err}");
    }
}

#[cfg(all(test, feature = "dx12", target_os = "windows"))]
mod bindless_slot_validation_tests {
    use super::validate_bindless_slot_kinds;
    use crate::types::BindlessSlotKind;

    #[test]
    fn matching_srv_uav_passes() {
        let expectations = vec![Some(BindlessSlotKind::StorageUav), Some(BindlessSlotKind::ReadOnlySrv)];
        validate_bindless_slot_kinds(
            &[10, 20],
            &expectations,
            |idx| {
                Some(match idx {
                    10 => BindlessSlotKind::StorageUav,
                    20 => BindlessSlotKind::ReadOnlySrv,
                    _ => return None,
                })
            },
            "test_shader",
        )
        .unwrap();
    }

    #[test]
    fn srv_where_uav_expected_fails() {
        let expectations = vec![Some(BindlessSlotKind::StorageUav)];
        let err = validate_bindless_slot_kinds(
            &[5],
            &expectations,
            |_| Some(BindlessSlotKind::ReadOnlySrv),
            "game_of_life_render",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("SRV/UAV mismatch"), "{err}");
        assert!(err.contains("slot 0"), "{err}");
        assert!(err.contains("storage UAV"), "{err}");
        assert!(err.contains("read-only SRV"), "{err}");
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

    // ----- GOLDY_VALIDATION=all: improved hint + missing-index detection -----

    /// The hint message must mention both Scattered and Broadcast so developers
    /// understand why a constant-buffer parameter can fail stride validation.
    #[test]
    fn hint_message_mentions_both_scattered_and_broadcast() {
        let actual = vec![Some(4)];
        let expected = vec![Some(16)];
        let err = validate_binding_strides(&actual, &expected, "cs")
            .unwrap_err()
            .to_string();
        assert!(
            err.to_lowercase().contains("scattered") || err.to_lowercase().contains("broadcast"),
            "hint must mention both buffer kinds: {err}"
        );
    }

    /// When `validate_raw_binding_strides` is called with fewer indices than
    /// the shader's reflected Scattered/Broadcast slot count, it must report an
    /// error naming the missing slots — not silently succeed.
    ///
    /// This is the `GOLDY_VALIDATION=all` guard for "too few indices" bugs.
    #[test]
    fn missing_index_for_required_slot_is_an_error() {
        use super::validate_raw_binding_strides;
        use crate::types::ResourceCategory;

        // Shader has 2 Scattered slots but caller only provides 1 index.
        let indices = vec![42u32]; // only slot 0 provided
        let categories = vec![
            Some(ResourceCategory::Scattered), // slot 0 — has index
            Some(ResourceCategory::Scattered), // slot 1 — missing
        ];
        let expected = vec![Some(16u32), Some(16u32)];

        let err = validate_raw_binding_strides(
            &indices,
            &categories,
            &expected,
            |_idx, _cat| Some(16),
            "my_compute_shader",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("slot"), "error must name missing slot: {err}");
        assert!(err.contains("my_compute_shader"), "error must name the shader: {err}");
        assert!(err.contains('1'), "error must mention slot index 1: {err}");
    }

    /// Providing exactly the right number of indices for all Scattered/Broadcast
    /// slots must pass cleanly.
    #[test]
    fn exact_index_count_passes() {
        use super::validate_raw_binding_strides;
        use crate::types::ResourceCategory;

        let indices = vec![0u32, 1u32];
        let categories = vec![Some(ResourceCategory::Scattered), Some(ResourceCategory::Broadcast)];
        let expected = vec![Some(16u32), Some(4u32)];

        validate_raw_binding_strides(
            &indices,
            &categories,
            &expected,
            |_idx, cat| match cat {
                ResourceCategory::Scattered => Some(16),
                ResourceCategory::Broadcast => Some(4),
                _ => None,
            },
            "ok_shader",
        )
        .unwrap();
    }

    /// Broadcast slot correct, scattered slot wrong — only slot 1 reported.
    #[test]
    fn multi_binding_second_slot_mismatch() {
        use super::validate_raw_binding_strides;
        use crate::types::ResourceCategory;

        let indices = vec![0u32, 1u32];
        let categories = vec![Some(ResourceCategory::Broadcast), Some(ResourceCategory::Scattered)];
        let expected = vec![Some(16u32), Some(16u32)];

        let err = validate_raw_binding_strides(
            &indices,
            &categories,
            &expected,
            |idx, cat| match (idx, cat) {
                (0, ResourceCategory::Broadcast) => Some(16),
                (1, ResourceCategory::Scattered) => Some(4),
                _ => None,
            },
            "struct_shader",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("slot 1"), "error must identify slot 1: {err}");
        assert!(err.contains("stride"), "error must mention stride: {err}");
    }

    /// Broadcast 16 + Scattered<float4> 16 both match.
    #[test]
    fn multi_binding_all_correct_passes() {
        use super::validate_raw_binding_strides;
        use crate::types::ResourceCategory;

        let indices = vec![0u32, 1u32];
        let categories = vec![Some(ResourceCategory::Broadcast), Some(ResourceCategory::Scattered)];
        let expected = vec![Some(16u32), Some(16u32)];

        validate_raw_binding_strides(&indices, &categories, &expected, |_idx, _cat| Some(16), "struct_shader").unwrap();
    }

    /// SimParams natural stride 4 (not cbuffer-padded 16).
    #[test]
    fn broadcast_single_float_natural_stride_passes() {
        use super::validate_raw_binding_strides;
        use crate::types::ResourceCategory;

        let indices = vec![0u32, 1u32];
        let categories = vec![Some(ResourceCategory::Scattered), Some(ResourceCategory::Broadcast)];
        let expected = vec![Some(4u32), Some(4u32)];

        validate_raw_binding_strides(
            &indices,
            &categories,
            &expected,
            |idx, cat| match (idx, cat) {
                (0, ResourceCategory::Scattered) => Some(4),
                (1, ResourceCategory::Broadcast) => Some(4),
                _ => None,
            },
            "sim_params_shader",
        )
        .unwrap();
    }

    /// Broadcast buffer with stride 16 must fail against natural stride 4.
    #[test]
    fn broadcast_single_float_cbuffer_stride_fails() {
        use super::validate_raw_binding_strides;
        use crate::types::ResourceCategory;

        let indices = vec![0u32, 1u32];
        let categories = vec![Some(ResourceCategory::Scattered), Some(ResourceCategory::Broadcast)];
        let expected = vec![Some(4u32), Some(4u32)];

        let err = validate_raw_binding_strides(
            &indices,
            &categories,
            &expected,
            |idx, cat| match (idx, cat) {
                (0, ResourceCategory::Scattered) => Some(4),
                (1, ResourceCategory::Broadcast) => Some(16),
                _ => None,
            },
            "sim_params_shader",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("slot 1"), "params is slot 1: {err}");
        assert!(err.contains("stride"), "error must mention stride: {err}");
    }

    /// Stride validation for render passes runs at prepare time (before lowering).
    #[test]
    fn validate_render_pass_bind_resources_catches_mismatch() {
        use super::validate_render_pass_bind_resources;
        use crate::backend::{BufferHandle, PipelineHandle, RenderCommand};

        let pipeline: PipelineHandle = 1;
        let buf: BufferHandle = 1;
        let commands = vec![
            RenderCommand::SetPipeline(pipeline),
            RenderCommand::BindResources { buffers: vec![buf] },
        ];

        let err = validate_render_pass_bind_resources(
            &commands,
            |h| {
                if h == 1 {
                    Some((vec![Some(4)], "test_shader".to_string()))
                } else {
                    None
                }
            },
            |h| {
                if h == 1 {
                    Some(16)
                } else {
                    None
                }
            },
        )
        .expect_err("mismatched render bind must fail");
        assert!(err.to_string().contains("slot 0"));
    }
}
