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
pub(crate) mod vulkan;

// DX12 backend for Windows
#[cfg(all(feature = "dx12", target_os = "windows"))]
pub(crate) mod dx12;

// Mock backend for testing (always available)
pub(crate) mod mock;

// Metal backend for macOS (native Metal, not MoltenVK)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) mod metal;

// Portable WebGPU backend (compute-only prototype).
#[cfg(feature = "webgpu")]
pub(crate) mod webgpu;

// NVIDIA CUDA backend (compute-only prototype).
#[cfg(feature = "cuda")]
pub(crate) mod cuda;

pub(crate) use crate::device::{AdapterInfo, BufferHeapStats, TextureHeapStats, VideoMemoryInfo};
pub(crate) use crate::handles::{
    BufferHandle, ComputePipelineHandle, ContextHandle, DeviceHandle, PipelineHandle, RenderTargetHandle,
    SamplerHandle, ShaderHandle, TextureHandle,
};
#[cfg(feature = "graphics")]
pub(crate) use crate::handles::{SurfaceHandle, SwapchainImageHandle};
pub(crate) use crate::texture::TextureCopyFootprint;

/// Shared primitives reused across Vulkan, DX12, and Metal backends, and by
/// task-graph command emission (e.g. `DispatchBatch` argument packing).
pub(crate) mod shared;

/// Per-device FIFO worker for async GPU queue submission.
pub(crate) mod submission_worker;

/// Helpers for relocating reuse waits and deferred host writes to the submission worker.
pub(crate) mod host_sidecar;

/// Fence/timeline polling threads for async [`crate::signal::Signal`] delivery (Vulkan, DX12, CUDA).
#[cfg(any(feature = "vulkan", all(feature = "dx12", target_os = "windows"), feature = "cuda"))]
pub(crate) mod signal_fence;

use crate::types::{
    BackendType, BufferFlags, BufferKind, IndexFormat, ResourceAccess, ResourceHandle, SamplerDesc, TextureFlags,
    TextureFormat, TextureKind,
};
#[cfg(feature = "graphics")]
use crate::types::{DepthFormat, DepthStencilState, PresentMode, PrimitiveTopology, VertexBufferLayout};
use anyhow::Result;
use std::sync::Arc;

/// When set via `GOLDY_VALIDATION` (e.g. `api` or `all` in the token list), or loader
/// `VK_INSTANCE_LAYERS`, enables backend-specific GPU validation where supported:
/// Vulkan enables `VK_LAYER_KHRONOS_validation` and `VK_EXT_debug_utils` at instance creation;
/// Metal sets `MTL_SHADER_VALIDATION=1` before the first device is created if that variable is unset;
/// CUDA enables Driver diagnostics (PTX JIT logs, eager sync, launch-limit checks) and may set
/// `CUDA_LAUNCH_BLOCKING=1` when unset.
///
/// See the `validation_env` module for the full `GOLDY_VALIDATION` list syntax (`layout`, `api`, `all`, …).
///
/// For Vulkan, validation is also enabled when `VK_INSTANCE_LAYERS` includes
/// `VK_LAYER_KHRONOS_validation` (loader-driven workflow; see Vulkan backend `new()`).
#[cfg(any(feature = "vulkan", feature = "cuda", all(feature = "metal", target_os = "macos"),))]
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

/// Opaque token tying surface work to an acquired swapchain frame.
#[cfg(feature = "graphics")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FrameToken {
    pub surface: SurfaceHandle,
    /// WSI swapchain image index (which drawable will be presented).
    pub image: SwapchainImageHandle,
    /// Submission context that owns this frame's timeline (set when the surface frame is acquired).
    pub context: ContextHandle,
    /// In-flight slot index for the compute/scratch texture bound this frame.
    ///
    /// Used for present-lease retention keys: must match the physical backing that
    /// shader dispatches and copies target, not necessarily [`Self::image`].
    /// On Vulkan this is `current_frame`; on DX12 it equals the swapchain image index.
    pub frame_slot: u32,
    /// Command-allocator / frame-sync slot claimed at acquire for this frame.
    ///
    /// `present()` reads [`Self::image`], scratch, and `frame_sync[present_slot]` from
    /// this token only — not from single-valued surface fields — so the next acquire
    /// can overlap an in-flight present.
    ///
    /// On Vulkan this equals [`Self::frame_slot`]. On DX12 it is the rotating
    /// `current_frame` index (distinct from [`Self::image`]).
    pub present_slot: u32,
}

/// Render command for command buffer recording.
#[allow(dead_code)] // fields matched by GPU backends behind feature flags
#[derive(Debug, Clone)]
pub(crate) enum RenderCommand {
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

/// CPU-side write deferred to the submission worker, after [`SubmitSync::host_observed_waits`]
/// retire on the host.
///
/// Currently applied by the DX12, Vulkan, and Metal submission workers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeferredHostWrite {
    pub buffer: BufferHandle,
    pub offset: u64,
    pub data: Arc<[u8]>,
}

/// Cross-submission synchronization derived from the runtime's ledger (spec §5).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SubmitSync {
    /// Same-context scoped memory barrier prepended before command execution.
    pub prologue: crate::task_graph::BarrierSet,
    /// Cross-context GPU queue-waits on producer timeline values.
    pub waits: Vec<crate::timeline::Epoch>,
    /// Cross-context WAR hazards resolved on the CPU before enqueueing GPU work.
    ///
    /// A live GPU `Wait` for a cross-context WAR can form opposing queue dependencies
    /// with a concurrent cross-context RAW wait (producer upload vs consumer reader),
    /// wedging per-context queues. CPU-side retirement breaks the cycle.
    pub cpu_waits: Vec<crate::timeline::Epoch>,
    /// CPU-observed fence waits before host-visible writes (not ordered by GPU queue-wait alone).
    pub host_observed_waits: Vec<crate::timeline::Epoch>,
    /// Host writes performed on the submission worker after `host_observed_waits` retire.
    pub deferred_host_writes: Vec<DeferredHostWrite>,
}

impl SubmitSync {
    pub fn is_empty(&self) -> bool {
        self.prologue.is_empty()
            && self.waits.is_empty()
            && self.cpu_waits.is_empty()
            && self.host_observed_waits.is_empty()
            && self.deferred_host_writes.is_empty()
    }

    /// Merge `extra` queue-order epochs into `self.waits` (max per context).
    pub fn merge_queue_waits(&mut self, extra: &[crate::timeline::Epoch]) {
        crate::backend::host_sidecar::merge_epochs(&mut self.waits, extra);
    }

    /// Merge host-observed epochs into `self.host_observed_waits` (max per context).
    pub fn merge_host_observed_waits(&mut self, extra: &[crate::timeline::Epoch]) {
        crate::backend::host_sidecar::merge_epochs(&mut self.host_observed_waits, extra);
    }

    /// True when this submit should use the legacy blanket cross-submission acquire
    /// instead of epoch-driven scoped barriers.
    ///
    /// Epoch-driven scheme/task-graph submits pass `Some(SubmitSync)` with a scoped
    /// prologue and/or cross-context waits; they must not also emit the blanket acquire.
    #[cfg_attr(not(any(test, feature = "vulkan")), allow(dead_code))]
    pub fn use_legacy_acquire(&self) -> bool {
        false
    }

    /// Whether `sync` selects the legacy blanket acquire path.
    #[cfg_attr(not(any(test, feature = "vulkan")), allow(dead_code))]
    pub fn use_legacy_acquire_from(sync: Option<&SubmitSync>) -> bool {
        match sync {
            None => true,
            Some(s) => s.use_legacy_acquire(),
        }
    }
}

#[cfg(test)]
mod submit_sync_tests {
    use super::SubmitSync;

    #[test]
    fn use_legacy_acquire_from_none_means_legacy() {
        assert!(SubmitSync::use_legacy_acquire_from(None));
    }

    #[test]
    fn use_legacy_acquire_from_some_means_scoped() {
        assert!(!SubmitSync::use_legacy_acquire_from(Some(&SubmitSync::default())));
    }
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
pub(crate) enum GpuCommand {
    /// Set the active compute pipeline.
    SetPipeline(ComputePipelineHandle),
    /// Bind resource slots with raw u32 indices (for textures/samplers or mixed resources).
    ///
    /// `indices` go to region A (bindless, packed as u16).
    /// `user` go to region B (user scalars, full u32).
    BindResourcesRaw {
        indices: Vec<u32>,
        user: Vec<u32>,
        /// Offset within the frame-table row where this dispatch's indices live.
        frame_table_base: u32,
    },
    /// Dispatch compute workgroups.
    Dispatch {
        /// Debug label from [`crate::Scheme::node`] when emitted by the analyzer.
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
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    },
    /// Copy pixels from a CPU-writable buffer into a texture subregion.
    ///
    /// When `src_row_pitch == 0` the source is tightly packed and the backend will repack
    /// into a footprint-aligned intermediate buffer.  When `src_row_pitch > 0` the source
    /// is already footprint-aligned (allocated with `staging_bytes` capacity and rows
    /// written at `src_row_pitch` stride), and the backend copies directly.
    CopyBufferToTexture {
        src: BufferHandle,
        src_offset: u64,
        /// 0 = tight source (backend repacks); >0 = footprint pitch, copy directly.
        src_row_pitch: u32,
        dst: TextureHandle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    /// Copy a texture subresource into a withdraw-staging staging buffer (placed footprint).
    CopyTextureToReadback {
        src: TextureHandle,
        dst: BufferHandle,
        layout: TextureCopyFootprint,
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

/// Mixed compute + offscreen render commands from [`crate::Scheme`].
#[derive(Debug, Clone)]
pub(crate) enum GraphCommand {
    /// Compute / upload / barrier from [`GpuCommand`].
    Compute(GpuCommand),
    /// Graphics work recorded against a [`RenderTargetHandle`] (offscreen render target).
    Render {
        target: RenderTargetHandle,
        color_load: crate::types::TargetLoad,
        commands: Vec<RenderCommand>,
    },
}

/// Blocking GPU timeline wait, cloned out of the backend under the global lock so
/// [`TimelineBlockingWait::block`] can run without holding it.
pub(crate) trait TimelineBlockingWait: Send {
    fn block(self: Box<Self>) -> Result<()>;

    /// Like [`Self::block`] but returns `Ok(false)` on timeout instead of blocking forever.
    fn block_timeout(self: Box<Self>, _timeout_ms: u32) -> Result<bool> {
        self.block()?;
        Ok(true)
    }
}

/// Context teardown detached from live lookup tables. [`Self::wait`] and [`Self::finish`]
/// run without holding the global backend mutex.
#[doc(hidden)]
pub(crate) trait ContextDestroyHandle: Send {
    fn wait(&self) -> Result<()>;
    fn finish(self: Box<Self>) -> Result<()>;
}

/// Drain GPU work then release resources for a detached context.
pub(crate) fn run_context_destroy(handle: Box<dyn ContextDestroyHandle>) {
    if let Err(e) = handle.wait() {
        tracing::error!("context destroy wait failed: {e:#}");
    }
    if let Err(e) = handle.finish() {
        tracing::error!("context destroy finish failed: {e:#}");
    }
}

/// Like [`destroy_context`] for an already-unlocked concrete backend (tests, `destroy_device`).
#[cfg(any(
    test,
    feature = "vulkan",
    all(feature = "dx12", target_os = "windows"),
    all(feature = "metal", target_os = "macos"),
    feature = "cuda",
))]
pub(crate) fn destroy_context_mut(backend: &mut dyn GpuBackend, ctx: ContextHandle) {
    if let Some(handle) = backend.detach_context_for_destroy(ctx) {
        run_context_destroy(handle);
    }
}
/// Destroy `ctx` without holding the global backend lock across blocking GPU work.
pub(crate) fn destroy_context(backend: &std::sync::Arc<std::sync::Mutex<Box<dyn GpuBackend>>>, ctx: ContextHandle) {
    if let Some(handle) = {
        let mut guard = backend.lock().unwrap();
        guard.detach_context_for_destroy(ctx)
    } {
        run_context_destroy(handle);
    }
}

/// Per-context deferred GPU deletion flush cloned out of the backend so
/// [`Context::boundary_crossed`](crate::Context::boundary_crossed) does not need the global
/// backend mutex for `flush_deferred_deletions`.
///
/// Each implementation is responsible for computing whatever device-scope timeline value it
/// needs internally; the trait no longer takes `device_retired` as a parameter so callers
/// do not need to produce it.
#[doc(hidden)]
pub(crate) trait ContextDeferredDeletionFlush: Send + Sync {
    fn flush(&self);
}

/// Lock-free per-context GPU timeline progress query (Vulkan/DX12 fence or semaphore value).
#[doc(hidden)]
pub(crate) trait ContextGpuProgress: Send + Sync {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue;
}

/// Per-context reclamation epoch scope (Metal heap routing during `boundary_crossed`).
#[doc(hidden)]
pub(crate) trait ContextReclamationScope: Send + Sync {
    fn set_epoch(&self, epoch: Option<crate::timeline::TimelineValue>);
}

pub(crate) struct NoOpReclamationScope;

impl ContextReclamationScope for NoOpReclamationScope {
    fn set_epoch(&self, _epoch: Option<crate::timeline::TimelineValue>) {}
}

pub(crate) struct NoOpDeferredDeletionFlush;

impl ContextDeferredDeletionFlush for NoOpDeferredDeletionFlush {
    fn flush(&self) {}
}

/// Bookkeeping applied after [`PresentGpuWork::run`] completes without the global lock.
#[cfg(feature = "graphics")]
#[allow(dead_code)] // fields read by present-split impls behind backend feature flags
pub(crate) struct PresentFinishState {
    pub frame: FrameToken,
    /// Fence/timeline guarding swapchain image return (0 when immediate return).
    pub return_fence: crate::timeline::TimelineValue,
    pub scratch_texture: Option<TextureHandle>,
    pub scratch_layout_updated: bool,
    pub present_timeline: crate::timeline::TimelineValue,
    /// Vulkan: timeline signalled by the present-copy submit (if any).
    pub copy_timeline: Option<crate::timeline::TimelineValue>,
    /// Vulkan: compute timeline stored on the surface slot (easement semantics).
    pub frame_compute_timeline: Option<crate::timeline::TimelineValue>,
    /// Vulkan: context `last_submitted_seq` update from the copy submit.
    pub signal_timeline: Option<crate::timeline::TimelineValue>,
    pub render_pass_submitted: bool,
    /// Whether WSI `queue_present` / equivalent succeeded (modulo OUT_OF_DATE).
    pub present_ok: bool,
}

/// GPU-side present work (copy + queue present) cloned out of the backend under the
/// global lock so [`crate::surface::Frame::present`] can drop it during execution.
#[cfg(feature = "graphics")]
pub(crate) trait PresentGpuWork: Send {
    fn run(self: Box<Self>) -> Result<PresentFinishState>;
}

/// Lock-free submit surface cloned at [`crate::Context::new`].
///
/// Holds `Arc` handles to per-context submission state, device ledger, and resource
/// tables so IR lowering + command recording + queue submit can run without the global
/// backend mutex. Resource create/destroy still takes the global lock (write access).
pub(crate) trait ContextSubmitSession: Send + Sync {
    /// When true, compute and render partitions use separate GPU queues and must not merge.
    fn separate_graphics_queue(&self) -> bool {
        false
    }

    /// Synthetic context handle for device-queue producer epochs (DX12 compute style).
    fn device_queue_owner(&self, _ctx: ContextHandle) -> Option<ContextHandle> {
        None
    }

    fn retains_present_partitions(&self) -> bool;
    fn submit_standalone(
        &self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue>;
    fn submit_graph(
        &self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue>;
    fn submit_graph_and_retain(
        &self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue>;
    fn try_resubmit_retained(
        &self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<crate::timeline::TimelineValue>>;
    fn evict_retained(&self, ctx: ContextHandle, key: u64);
}

/// Clone a per-context submit session at [`crate::Context::new`].
///
/// Real GPU backends return lock-free sessions; Metal and mock use [`LockedSubmitSession`].
pub(crate) trait GpuBackendSubmitSession {
    fn clone_context_submit_session(
        &self,
        ctx: ContextHandle,
        backend: std::sync::Arc<std::sync::Mutex<Box<dyn GpuBackend>>>,
    ) -> std::sync::Arc<dyn ContextSubmitSession>;
}

/// Per-context submit session that acquires the global backend mutex around every
/// submit call. Used by Metal (until lock-free recording lands) and [`mock::MockBackend`].
pub(crate) struct LockedSubmitSession {
    backend: std::sync::Arc<std::sync::Mutex<Box<dyn GpuBackend>>>,
    backend_type: BackendType,
}

impl LockedSubmitSession {
    /// Build a session while the global backend lock is already held (must not re-lock).
    pub fn with_backend_type(
        backend: std::sync::Arc<std::sync::Mutex<Box<dyn GpuBackend>>>,
        backend_type: BackendType,
    ) -> std::sync::Arc<dyn ContextSubmitSession> {
        std::sync::Arc::new(Self { backend, backend_type })
    }
}

impl ContextSubmitSession for LockedSubmitSession {
    fn retains_present_partitions(&self) -> bool {
        !matches!(self.backend_type, crate::types::BackendType::Metal)
    }

    fn submit_standalone(
        &self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        let mut guard = {
            let _tz = crate::tracy_zone!("goldy.submit_session.lock");
            self.backend.lock().unwrap()
        };
        let _tz = crate::tracy_zone!("goldy.submit_session.submit_standalone");
        let tv = guard.submit_standalone(ctx, commands, sync)?;
        drop(guard);
        Ok(tv)
    }

    fn submit_graph(
        &self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        let mut guard = {
            let _tz = crate::tracy_zone!("goldy.submit_session.lock");
            self.backend.lock().unwrap()
        };
        let _tz = crate::tracy_zone!("goldy.submit_session.submit_graph");
        guard.submit_graph(ctx, commands, sync)
    }

    fn submit_graph_and_retain(
        &self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        let mut guard = {
            let _tz = crate::tracy_zone!("goldy.submit_session.lock");
            self.backend.lock().unwrap()
        };
        let _tz = crate::tracy_zone!("goldy.submit_session.submit_graph_and_retain");
        guard.submit_graph_and_retain(ctx, commands, key, sync)
    }

    fn try_resubmit_retained(
        &self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<crate::timeline::TimelineValue>> {
        let mut guard = {
            let _tz = crate::tracy_zone!("goldy.submit_session.lock");
            self.backend.lock().unwrap()
        };
        let _tz = crate::tracy_zone!("goldy.submit_session.try_resubmit_retained");
        guard.try_resubmit_retained(ctx, key, sync)
    }

    fn evict_retained(&self, ctx: ContextHandle, key: u64) {
        self.backend.lock().unwrap().evict_retained(ctx, key);
    }
}

/// Split present hooks used by [`crate::surface::Frame::present`] to drop the
/// global backend lock during copy + WSI present.
#[cfg(feature = "graphics")]
pub(crate) trait GpuBackendPresentSplit {
    fn take_present_gpu_work(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn PresentGpuWork>>;

    fn finish_present(
        &mut self,
        finish: PresentFinishState,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue>;
}

#[cfg(feature = "graphics")]
trait GpuBackendGraphics: GpuBackendPresentSplit {}

#[cfg(feature = "graphics")]
impl<T: GpuBackendPresentSplit + ?Sized> GpuBackendGraphics for T {}

#[cfg(not(feature = "graphics"))]
trait GpuBackendGraphics {}

#[cfg(not(feature = "graphics"))]
impl<T: ?Sized> GpuBackendGraphics for T {}

/// Split timeline wait hooks used by [`Context::wait_until`](crate::Context::wait_until)
/// to drop the global backend lock during blocking GPU waits.
pub(crate) trait GpuBackendTimelineWait {
    /// Returns a worker wait handle when `value` may still be in the submission queue.
    /// The caller must run [`submission_worker::SubmissionEpochWait::wait`] without holding
    /// the backend mutex so parallel submits cannot deadlock with `wait_until`.
    fn take_timeline_submission_epoch_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<submission_worker::SubmissionEpochWait>>;

    fn take_timeline_blocking_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn TimelineBlockingWait>>>;

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()>;
}

/// GPU backend trait - implemented by Vulkan, Metal, DX12.
#[allow(private_bounds)]
pub(crate) trait GpuBackend:
    Send + Sync + GpuBackendTimelineWait + GpuBackendGraphics + GpuBackendSubmitSession
{
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

    /// Detach `ctx` from live lookup tables under a brief global-backend lock.
    ///
    /// The caller should release the lock before [`ContextDestroyHandle::wait`] /
    /// [`ContextDestroyHandle::finish`]. Prefer [`destroy_context`] when destroying
    /// from code that holds the wrapping `Arc<Mutex<Box<dyn GpuBackend>>>`.
    #[doc(hidden)]
    fn detach_context_for_destroy(&mut self, ctx: ContextHandle) -> Option<Box<dyn ContextDestroyHandle>>;

    /// Clone the per-context deferred-deletion flusher for [`Context::boundary_crossed`].
    #[doc(hidden)]
    fn clone_context_deletion_flush(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn ContextDeferredDeletionFlush>>;

    /// Clone a lock-free GPU progress query for [`Context::gpu_progress`].
    #[doc(hidden)]
    fn clone_context_gpu_progress(&self, ctx: ContextHandle) -> Option<std::sync::Arc<dyn ContextGpuProgress>> {
        let _ = ctx;
        None
    }

    /// Clone the per-context reclamation scope for [`Context::boundary_crossed`].
    #[doc(hidden)]
    fn clone_context_reclamation_scope(&self, ctx: ContextHandle) -> std::sync::Arc<dyn ContextReclamationScope> {
        let _ = ctx;
        std::sync::Arc::new(NoOpReclamationScope)
    }

    /// Returns `true` if the device has been permanently lost (TDR, hardware hang, etc.).
    ///
    /// Backends set this flag atomically when they detect device loss so that
    /// [`Device::is_device_lost`](crate::Device::is_device_lost) can be polled from the
    /// render loop without acquiring any lock. The default returns `false` for
    /// backends that have not yet wired up the flag.
    fn is_device_lost(&self, _device: DeviceHandle) -> bool {
        false
    }

    /// OS/driver video-memory usage for `device`, when the backend can query it.
    ///
    /// DX12 returns DXGI local/non-local segment info. Other backends default to `None`.
    fn query_video_memory(&self, _device: DeviceHandle) -> Option<VideoMemoryInfo> {
        None
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
    /// Allocate a persistently mapped READBACK staging buffer for withdraw staging (no bindless slot).
    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle>;
    /// Read bytes from a buffer created by [`Self::alloc_readback_buffer`].
    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()>;
    /// Release a withdraw-staging staging buffer.
    fn free_readback_buffer(&mut self, buffer: BufferHandle);
    /// Query copy/readback layout for a 2D texture grant (uncompressed formats only in v1).
    fn query_texture_copy_footprint(
        &self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<TextureCopyFootprint>;
    /// Allocate a persistently mapped READBACK staging buffer for a texture grant.
    fn alloc_texture_readback_staging(
        &mut self,
        device: DeviceHandle,
        layout: TextureCopyFootprint,
    ) -> Result<BufferHandle>;
    /// Read tight linear bytes from a texture grant staging buffer.
    fn read_texture_readback_staging(
        &self,
        buffer: BufferHandle,
        layout: TextureCopyFootprint,
        output: &mut [u8],
    ) -> Result<()>;

    /// Barrier-layout tag for retained [`GpuCommand::CopyBufferToTexture`] fingerprinting.
    ///
    /// Must change when the texture's copy-source barrier state changes such that a
    /// retained command buffer's baked barriers would be invalid (typically once,
    /// COMMON → shader-read after the first upload).
    fn texture_copy_retention_tag(&self, texture: TextureHandle) -> u64;

    /// Mock-backend withdraw staging allocation counter (tests only).
    #[doc(hidden)]
    #[cfg(test)]
    fn test_readback_alloc_count(&self) -> usize {
        let _ = self;
        0
    }

    /// Mock-backend withdraw staging free counter (tests only).
    #[doc(hidden)]
    #[cfg(test)]
    fn test_readback_free_count(&self) -> usize {
        let _ = self;
        0
    }

    /// Mock-backend surface present counter (tests only).
    #[doc(hidden)]
    #[cfg(all(test, feature = "graphics"))]
    fn test_surface_present_count(&self) -> usize {
        let _ = self;
        0
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

    /// Hint that bytes at and above `offset` may be discarded by the system (see `hint_unused_above` on the backing allocation).
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
    #[cfg(feature = "graphics")]
    fn create_pipeline(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
    ) -> Result<PipelineHandle>;
    #[cfg(feature = "graphics")]
    fn destroy_pipeline(&mut self, pipeline: PipelineHandle);

    // Pipeline with depth stencil state
    #[cfg(feature = "graphics")]
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

    // RenderTarget API - GPU-only; no CPU readback
    /// Create a render target with an optional depth buffer.
    #[cfg(feature = "graphics")]
    fn create_render_target_with_depth(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle>;
    /// Destroy a render target created by [`Self::create_render_target_with_depth`].
    ///
    /// Must free backend GPU resources and recycle descriptor-heap slots (DX12 RTV/DSV).
    /// Omitting this leaks slots until the heap overflows and the driver AVs.
    #[cfg(feature = "graphics")]
    fn destroy_render_target(&mut self, target: RenderTargetHandle);
    #[cfg(feature = "graphics")]
    fn render_to_target(
        &mut self,
        device: DeviceHandle,
        target: RenderTargetHandle,
        color_load: crate::types::TargetLoad,
        commands: &[RenderCommand],
    ) -> Result<()>;

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
    /// Attach a human-readable debug name to a texture resource.
    ///
    /// The name is forwarded to the graphics API's debug-naming facility
    /// (e.g. `vkSetDebugUtilsObjectNameEXT` on Vulkan) so that validation
    /// messages and GPU debuggers show the name instead of a raw handle.
    ///
    /// Backends that do not support debug naming provide a default no-op
    /// implementation, so callers do not need to feature-gate the call.
    fn set_texture_debug_name(&mut self, _handle: TextureHandle, _name: &str) {}
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
    #[cfg(feature = "graphics")]
    fn create_surface(
        &mut self,
        device: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        display: &dyn raw_window_handle::HasDisplayHandle,
        depth_format: Option<DepthFormat>,
    ) -> Result<SurfaceHandle>;

    /// Destroy a surface.
    #[cfg(feature = "graphics")]
    fn destroy_surface(&mut self, surface: SurfaceHandle);

    /// Resize the surface (recreates swapchain).
    #[cfg(feature = "graphics")]
    fn surface_resize(&mut self, surface: SurfaceHandle, width: u32, height: u32) -> Result<()>;

    /// Get the current surface dimensions.
    #[cfg(feature = "graphics")]
    fn surface_size(&self, surface: SurfaceHandle) -> (u32, u32);

    /// Get the texture format used by a surface's swapchain.
    /// Use this to ensure your render pipeline matches the surface format.
    #[cfg(feature = "graphics")]
    fn surface_format(&self, surface: SurfaceHandle) -> TextureFormat;

    /// Set the present mode for a surface.
    /// Returns an error if the mode is not supported by the backend.
    #[cfg(feature = "graphics")]
    fn surface_set_present_mode(&mut self, _surface: SurfaceHandle, _mode: PresentMode) -> Result<()> {
        Ok(())
    }

    // --- Timeline + explicit frame bracket ---

    /// Latest GPU completion point on this context's timeline (`value` is done when
    /// `gpu_progress() >= value`).
    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue;

    /// Latest device-global submission sequence retired on the GPU (shared queue / seq space).
    ///
    /// `value` is done when `device_timeline_retired() >= value`. This is the highest
    /// contiguous prefix of attributed timeline values whose owning semaphore has completed,
    /// floored by post-destroy retirement — not a max over independent context queues.
    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue;

    /// Block until the device-global timeline has retired at least `value`.
    ///
    /// Unlike [`Self::wait_until`] (which is per-context), this waits on the native semaphore
    /// that was signalled for `value` at submit time. Use this when the `TimelineValue` was
    /// produced by an arbitrary context so you don't need a matching `ContextHandle`.
    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> Result<()>;

    /// Drain pending backend signals for this context (async queue + synchronous oversubscribed).
    ///
    /// `progress` is the caller's lock-free [`Context::gpu_progress`](crate::Context::gpu_progress)
    /// so this path does not re-query the timeline under the global backend mutex.
    fn poll_signals(
        &mut self,
        ctx: ContextHandle,
        progress: crate::timeline::TimelineValue,
    ) -> Vec<crate::signal::QueuedSignal>;

    fn wait_until(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        if let Some(wait) = self.take_timeline_blocking_wait(ctx, value)? {
            wait.block()?;
        }
        self.finish_timeline_wait(ctx, value)
    }

    /// Submit compute (and transfer) commands on the context timeline, not tied to a surface frame.
    ///
    /// When `sync` is `Some`, the backend emits the scoped prologue barrier and GPU-side
    /// queue-waits instead of the legacy blanket cross-submission acquire
    /// ([`SubmitSync::use_legacy_acquire_from`]).
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
                    color_load,
                    commands: render_cmds,
                } => {
                    #[cfg(feature = "graphics")]
                    {
                        if !batch.is_empty() {
                            last_tv = self.submit_standalone(ctx, &batch, sync)?;
                            self.wait_until(ctx, last_tv)?;
                            batch.clear();
                        }
                        let device = self.context_device(ctx);
                        self.render_to_target(device, *target, *color_load, render_cmds)?;
                        last_tv = self.submit_standalone(ctx, &[], sync)?;
                    }
                    #[cfg(not(feature = "graphics"))]
                    {
                        let _ = (target, color_load, render_cmds);
                        anyhow::bail!("render graph commands require the `graphics` feature");
                    }
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

    /// Drop the retained command list associated with `key`, marking its allocator slot
    /// as available for re-use.  No-op if no retained list exists.
    fn evict_retained(&mut self, _ctx: ContextHandle, _key: u64) {}

    /// Acquire the next swapchain image and begin a frame bracket.
    ///
    /// `ctx` is the submission context that owns this surface's timeline; frame
    /// submit/present and swapchain signals are routed through it.
    #[cfg(feature = "graphics")]
    fn begin_frame(&mut self, surface: SurfaceHandle, ctx: ContextHandle) -> Result<(FrameToken, TextureHandle)>;

    /// Submit all recorded GPU work for this frame bracket. Does not present.
    ///
    /// Returns the timeline value signaled when the frame's compute (and transfer) work
    /// completes on the GPU. When no work was recorded, returns the latest completed
    /// or scheduled compute timeline appropriate for the backend.
    #[cfg(feature = "graphics")]
    fn submit_frame(&mut self, frame: &FrameToken) -> Result<crate::timeline::TimelineValue>;

    // Compute pipeline management
    /// Create a compute pipeline from a compute shader.
    ///
    /// `debug_name` is an optional human-readable label (e.g. `"fine_area"`) used for
    /// GPU debugger / Instruments identification. Backends that support object labels
    /// apply it to the underlying PSO (and Metal function). When `None`, backends use
    /// a generic fallback such as `compute_shader#N`.
    fn create_compute_pipeline(
        &mut self,
        device: DeviceHandle,
        compute_shader: ShaderHandle,
        debug_name: Option<&str>,
    ) -> Result<ComputePipelineHandle>;

    /// Destroy a compute pipeline.
    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle);

    /// Per push-constant resource slot (in shader-signature order), the descriptor
    /// access the shader *signature* requires — independent of the graph access used
    /// for barriers.
    ///
    /// `Some(ResourceAccess::Read)` for read-only SRV params (`BufRO<T>`),
    /// `Some(ResourceAccess::ReadWrite)` for storage UAV params (`Scattered<T>`),
    /// and `None` for slots with no reflected preference (callers fall back to the
    /// graph access). The default returns an empty vec, meaning "no reflection
    /// available"; backends where the read/write descriptor split is irrelevant
    /// (e.g. Metal argument buffers) can leave it unimplemented.
    fn compute_pipeline_slot_access(&self, _pipeline: ComputePipelineHandle) -> Vec<Option<ResourceAccess>> {
        Vec::new()
    }

    /// Like [`Self::compute_pipeline_slot_access`] but for a graphics pipeline.
    #[cfg(feature = "graphics")]
    fn render_pipeline_slot_access(&self, _pipeline: PipelineHandle) -> Vec<Option<ResourceAccess>> {
        Vec::new()
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
    fn available_bindless_slots(&self, _device: DeviceHandle, _category: crate::types::ResourceCategory) -> u32 {
        u32::MAX
    }

    /// Maximum number of bindless descriptor slots per category for this backend.
    ///
    /// Returns `u32::MAX` by default (unlimited / not tracked).
    fn max_bindless_slots_per_category(&self, _device: DeviceHandle, _category: crate::types::ResourceCategory) -> u32 {
        u32::MAX
    }

    /// Maximum concurrent submission contexts this device can create.
    ///
    /// Vulkan pre-allocates a fixed compute-queue pool at device create; DX12/Metal
    /// create queues on demand and report [`u32::MAX`].
    fn max_submission_contexts(&self, _device: DeviceHandle) -> u32 {
        u32::MAX
    }

    /// Resources queued for destruction after the GPU timeline advances (for tests).
    #[doc(hidden)]
    fn deferred_deletion_pending_count(&self, _ctx: ContextHandle) -> usize {
        0
    }

    /// Device-level deferred deletions (bindless buffer/texture destroys that may span
    /// contexts). Per-context [`Self::deferred_deletion_pending_count`] stays separate.
    #[doc(hidden)]
    fn device_deferred_deletion_pending_count(&self, _device: DeviceHandle) -> usize {
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

/// Create the default backend for the current platform.
///
/// The backend can be overridden at runtime by setting the `GOLDY_BACKEND`
/// environment variable to one of: `vulkan`, `dx12`, `metal`, `webgpu`, `cuda`.
///
/// Without the override, the platform default is used when a native backend is
/// compiled in:
/// - macOS: Metal
/// - Windows: DX12
/// - Linux: Vulkan
///
/// CUDA and WebGPU are **not** platform defaults. They are selected automatically
/// only when the build enables `cuda` or `webgpu` **and** no native backend
/// (`vulkan`, `dx12`, `metal`) is compiled in — e.g.
/// `--no-default-features --features cuda`. In a normal default build, use
/// `GOLDY_BACKEND=cuda` or `GOLDY_BACKEND=webgpu` to opt in.
pub(crate) fn create_default_backend() -> Result<Box<dyn GpuBackend>> {
    // Check for runtime override via environment variable
    if let Ok(backend_str) = std::env::var("GOLDY_BACKEND") {
        let backend_type = match backend_str.to_lowercase().as_str() {
            "vulkan" | "vk" => BackendType::Vulkan,
            "dx12" | "d3d12" | "directx" => BackendType::Dx12,
            "metal" | "mtl" => BackendType::Metal,
            "webgpu" | "wgpu" => BackendType::WebGpu,
            "cuda" => BackendType::Cuda,
            other => anyhow::bail!(
                "Unknown GOLDY_BACKEND value '{}'. Valid options: vulkan, dx12, metal, webgpu, cuda",
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

    // Compute-only prototypes: only when no native graphics backend is compiled in.
    #[cfg(all(
        feature = "cuda",
        not(all(feature = "metal", target_os = "macos")),
        not(all(feature = "dx12", target_os = "windows")),
        not(feature = "vulkan")
    ))]
    {
        tracing::info!("Creating CUDA backend (compute-only build, no native backend compiled in)");
        Ok(Box::new(cuda::CudaBackend::new()?))
    }

    #[cfg(all(
        feature = "webgpu",
        not(feature = "cuda"),
        not(all(feature = "metal", target_os = "macos")),
        not(all(feature = "dx12", target_os = "windows")),
        not(feature = "vulkan")
    ))]
    {
        tracing::info!("Creating WebGPU backend (compute-only build, no native backend compiled in)");
        Ok(Box::new(webgpu::WebGpuBackend::new()?))
    }

    // No backend available
    #[cfg(not(any(
        all(feature = "metal", target_os = "macos"),
        all(feature = "dx12", target_os = "windows"),
        feature = "vulkan",
        feature = "cuda",
        feature = "webgpu"
    )))]
    {
        anyhow::bail!("No GPU backend available — enable 'vulkan', 'dx12', 'metal', 'cuda', or 'webgpu'")
    }
}

/// Create the default backend wrapped in an `Arc<Mutex<...>>`.
///
/// For DX12, each [`crate::Instance`] gets its own backend with independent mutable state;
/// DXGI factory/adapters are shared process-wide. Other backends create a fresh instance.
pub(crate) fn create_shared_backend() -> Result<std::sync::Arc<std::sync::Mutex<Box<dyn GpuBackend>>>> {
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
pub(crate) fn create_backend(backend_type: BackendType) -> Result<Box<dyn GpuBackend>> {
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
        #[cfg(feature = "webgpu")]
        BackendType::WebGpu => {
            tracing::info!("Creating WebGPU backend (compute-only)");
            Ok(Box::new(webgpu::WebGpuBackend::new()?))
        }
        #[cfg(feature = "cuda")]
        BackendType::Cuda => {
            tracing::info!("Creating CUDA backend");
            Ok(Box::new(cuda::CudaBackend::new()?))
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
