//! Compute-focused CUDA backend (Slang → PTX → Driver API).
//!
//! With `cuda` + `graphics` + `dx12` on Windows, presentation and a first-slice
//! raster path are enabled via a DX12 companion: CUDA writes shared RGBA8 scratch
//! textures and DX12 presents them with `CopyResource`; offscreen `Rgba32Float` / `Rgba8Unorm` render
//! targets and indexed / non-indexed graphics pipelines (point/line/triangle list+strip)
//! are also supported, including optional DX12-only depth attachments and
//! depth-stencil PSOs (depth is not CUDA-imported). Buffer handles are late-physicalized:
//! acquire reserves identity only; scheme usage chooses Shared (deposit→IA), Native
//! (compute), or NativeAndTwin (compute→IA). Bindless render bindings use the
//! companion's SM 6.6 descriptor heaps with CUDA registry slots as DX12 indices.
//!
//! Slang compiles `[goldy_compute]` (and plain compute) shaders to PTX. Launch
//! arguments use Slang's CUDA ABI:
//! - buffers → `{T* data; size_t count}`
//! - sampled textures → `CUtexObject`
//! - storage textures → `CUsurfObject`
//! - samplers → ignored pointer-sized `SamplerState` (filtering is baked into each
//!   `CUtexObject` from the bound Goldy sampler)
//! interleaved with bare `uniform uint` scalars from [`GpuCommand::BindResourcesRaw::user`].
//! Single-dispatch registry keys come from [`GpuCommand::BindResourcesRaw`]; batched
//! dispatches resolve keys from [`GpuCommand::FrameTableStaging`] in shader
//! parameter order — there is no bindless heap or device-side frame-table routing.
//!
//! Submits are host-asynchronous: each context owns a CUDA stream, each timeline
//! value owns a completion event, and a per-device [`SubmissionWorker`] records work.
//!
//! Retainable compute partitions are split into alternating graph-safe islands and
//! stream-replayed boundary segments (`submit_graph_and_retain`). Graph islands are
//! captured into CUDA graphs and relaunched on cache hits (`try_resubmit_retained`);
//! pinned host→device copies, CUDA-owned memsets/DtoD copies, and kernel launches
//! share graph islands. Format-specialized launches, imported-surface copies, and
//! external fences stay on the stream path. Schemes write a CUDA-owned staging
//! texture (`out_image`) and export via `CopyTexture` into D3D12-imported RGBA8
//! scratch before present's `CopyResource`. Indirect dispatches use CUDA 13.1
//! device-updatable kernel nodes plus an in-graph updater. Inline `WriteBuffer`
//! payloads (pageable `Vec`) still fall back to command replay. Dynamic waits,
//! deferred host writes, and completion events stay outside the captured graph.
//!
//! Windows CUDA+DX12 presentation, WDDM context-switch costs, and API gaps:
//! see [`WDDM_INTEROP.md`](WDDM_INTEROP.md) in this directory (internal; not user docs).

mod capture_gate;
mod pending_submit;
mod pinned_host;
mod retained_graph;
mod runtime_module;
mod texture;
mod timeline;

#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
mod buffer_phys;
#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
mod dx12_bindless;
#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
mod dx12_companion;
#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
mod dx12_interop;
#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
mod raster;
#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
mod surface;

use super::*;
use crate::backend::shared::{PushLayout, DISPATCH_BATCH_STRIDE, MAX_USER_SLOTS, TOTAL_PUSH_BYTES};
use crate::backend::submission_worker::{self, SubmissionWorker};
use crate::frame_table::dispatch_table_base_word_index;
use crate::slang::virtual_main::{CudaLaunchArgKind, CudaStorageTextureSpec};
use crate::types::{BufferResizeCost, DeviceType};
use crate::{goldy_event, goldy_span};
use anyhow::{Context as _, Result};
use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use pending_submit::{CudaOp, CudaPendingSubmit, CudaSubmitBody};
use pinned_host::CudaPinnedHost;
use retained_graph::GraphRegistry;
pub use retained_graph::{CudaGraphStats, CudaGraphStatsSnapshot};
use std::collections::{BTreeMap, HashMap};
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::thread::JoinHandle;
use texture::{memcpy_htod_array, storage_shader_convertible, CudaSamplerKey, CudaTextureResource};
use timeline::{EventLedger, LedgerCompletion, LedgerEntry};

/// Logical retained entry under the backend lock (graphs themselves live on the worker).
#[derive(Clone)]
enum RetainedEntry {
    /// Alternating graph islands + stream-replayed boundary segments.
    ///
    /// Graph-safe kernel runs are captured into the worker registry (one island per
    /// [`pending_submit::CudaOpSegment::Graph`]); stream segments are stored here and
    /// re-executed with `execute_ops` on each resubmit.
    Segmented {
        segments: Arc<Vec<pending_submit::CudaOpSegment>>,
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        scratch_images: Vec<(SurfaceHandle, usize)>,
        /// NativeAndTwin buffers any island/stream segment may write; dirty on each launch.
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        twin_dirty: Vec<BufferHandle>,
        /// Buffer handles written by stream segments; bumped on each relaunch without scanning.
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        written_buffers: Vec<BufferHandle>,
    },
    /// Fully pre-materialized operations. Replay never rematerializes commands.
    Ops {
        ops: Arc<Vec<CudaOp>>,
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        written_buffers: Vec<BufferHandle>,
    },
    /// Render partitions retain their high-level commands; DX12 list reuse happens in raster.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    Render(Vec<GraphCommand>),
    /// Dynamic present boundary: blit this raster target for `surface`'s *current*
    /// swapchain image. Scratch textures rotate per frame; do not pin a fixed scratch.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    PresentRenderTarget {
        target: RenderTargetHandle,
        surface: SurfaceHandle,
    },
}

/// Soft cap on concurrent submission contexts per CUDA device.
const MAX_CUDA_SUBMISSION_CONTEXTS: u32 = 32;

static CUDA_VALIDATION_INIT: Once = Once::new();

/// Cached device launch limits queried once at [`CudaBackend::create_device`].
#[derive(Clone, Copy, Debug)]
pub(super) struct CudaDeviceLimits {
    pub max_grid_dim_x: u32,
    pub max_grid_dim_y: u32,
    pub max_grid_dim_z: u32,
    pub max_threads_per_block: u32,
    pub max_shared_memory_per_block: u32,
}

/// Slang CUDA structured-buffer descriptor: `{ T* data; size_t count }`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct CudaBufferArg {
    data: u64,
    count: usize,
}

// SAFETY: plain POD matching Slang's CUDA StructuredBuffer / RWStructuredBuffer ABI.
unsafe impl DeviceRepr for CudaBufferArg {}

pub(crate) struct CudaBackend {
    adapter_info: Vec<AdapterInfo>,
    devices: HashMap<DeviceHandle, CudaDevice>,
    contexts: HashMap<ContextHandle, Arc<CudaSubmitContext>>,
    buffers: HashMap<BufferHandle, CudaBuffer>,
    buffer_slots: HashMap<u32, BufferHandle>,
    textures: HashMap<TextureHandle, Arc<CudaTextureResource>>,
    /// Registry keys for both sampled and storage texture views.
    texture_slots: HashMap<u32, TextureHandle>,
    samplers: HashMap<SamplerHandle, CudaSampler>,
    sampler_slots: HashMap<u32, SamplerHandle>,
    shaders: HashMap<ShaderHandle, CudaShader>,
    compute_pipelines: HashMap<ComputePipelineHandle, CudaComputePipeline>,
    retained: HashMap<(ContextHandle, u64), RetainedEntry>,
    graph_stats: Arc<CudaGraphStats>,
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    surfaces: HashMap<SurfaceHandle, surface::CudaSurfaceState>,
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    pipelines: HashMap<PipelineHandle, raster::CudaGraphicsPipeline>,
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    render_targets: HashMap<RenderTargetHandle, raster::CudaRenderTarget>,
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    raster_list_cache: HashMap<RenderTargetHandle, raster::RasterListCache>,
    /// D3D12 resources backing imported CUDA textures (bindless SRV/UAV writes).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    texture_dx12: HashMap<TextureHandle, windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    /// CUDA external-memory imports that must outlive [`Self::textures`] views.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    texture_imports: HashMap<TextureHandle, dx12_interop::CudaImportedTexture>,
    slang_compiler: crate::slang::SlangCompiler,
    next_device: DeviceHandle,
    next_context: ContextHandle,
    next_buffer: BufferHandle,
    next_texture: TextureHandle,
    next_sampler: SamplerHandle,
    next_slot: u32,
    next_shader: ShaderHandle,
    next_compute_pipeline: ComputePipelineHandle,
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    next_surface: SurfaceHandle,
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    next_pipeline: PipelineHandle,
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    next_render_target: RenderTargetHandle,
}

struct CudaDevice {
    ctx: Arc<CudaContext>,
    /// Stream used for host-driven alloc / immediate buffer APIs.
    alloc_stream: Arc<CudaStream>,
    submission_worker: Arc<SubmissionWorker>,
    next_timeline: Arc<AtomicU64>,
    retired: Arc<AtomicU64>,
    event_ledger: EventLedger,
    /// Recycled CUDA completion events (shared with contexts / present).
    event_pool: Arc<timeline::EventPool>,
    deletion_queue: Arc<Mutex<Vec<CudaDeferredDrop>>>,
    /// Worker-owned CUDA GraphExec registry (serialized via the submission worker).
    graph_registry: Arc<Mutex<GraphRegistry>>,
    graph_stats: Arc<CudaGraphStats>,
    limits: CudaDeviceLimits,
    /// NVRTC-compiled updater for device-updatable indirect dispatch.
    indirect_updater: Arc<runtime_module::IndirectUpdater>,
    /// DX12 presentation companion (cuda+graphics+dx12 on Windows only).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    dx12: Option<Arc<dx12_companion::Dx12Companion>>,
}

pub(super) struct CudaSubmitContext {
    handle: ContextHandle,
    device: DeviceHandle,
    stream: Arc<CudaStream>,
    completed: AtomicU64,
    last_emitted: AtomicU64,
    signal_queue: crate::signal::SignalQueue,
    device_retired: Arc<AtomicU64>,
    event_ledger: EventLedger,
    event_pool: Arc<timeline::EventPool>,
    deletion_queue: Arc<Mutex<Vec<CudaDeferredDrop>>>,
    fence_shutdown: Arc<AtomicBool>,
    fence_thread: Mutex<Option<JoinHandle<()>>>,
}

impl CudaSubmitContext {
    /// Poll completion markers and recycle retired CUDA events into the device pool.
    pub(super) fn poll_retire_events(&self) {
        timeline::poll_retire_events(
            &self.event_ledger,
            &self.completed,
            self.handle,
            &self.device_retired,
            &self.signal_queue,
            &self.last_emitted,
            &self.event_pool,
        );
    }
}

pub(super) enum CudaDeferredDrop {
    Buffer {
        retire_at: u64,
        /// Native CUDA alloc, or `None` when already leaked / deferred-unmaterialized.
        #[allow(dead_code)]
        memory: Option<Arc<Mutex<CudaSlice<u8>>>>,
        /// When set with [`CudaBuffer::memory_is_external`], drop order is leak-slice then twin.
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        #[allow(dead_code)]
        shared: Option<Arc<dx12_interop::SharedBufferBacking>>,
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        memory_is_external: bool,
    },
    /// Imported shared textures: field order is load-bearing (CUDA views → import → D3D12).
    Texture {
        retire_at: u64,
        #[allow(dead_code)]
        resource: Arc<CudaTextureResource>,
        /// External memory + mipmapped array; must outlive [`Self::Texture::resource`].
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        #[allow(dead_code)]
        import: Option<dx12_interop::CudaImportedTexture>,
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        #[allow(dead_code)]
        d3d12_resource: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    },
    Pipeline {
        retire_at: u64,
        #[allow(dead_code)]
        module: Arc<CudaModule>,
        #[allow(dead_code)]
        function: CudaFunction,
    },
    /// Offscreen raster target: CUDA views before import before D3D12 (drop order).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    RenderTarget {
        retire_at: u64,
        #[allow(dead_code)]
        cuda_texture: Arc<CudaTextureResource>,
        #[allow(dead_code)]
        import: dx12_interop::CudaImportedTexture,
        #[allow(dead_code)]
        d3d12_resource: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        /// DX12-only depth; dropped with the color resource after timeline retire.
        #[allow(dead_code)]
        depth_texture: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    },
}

struct CudaProgress {
    context: Arc<CudaSubmitContext>,
}

impl ContextGpuProgress for CudaProgress {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue {
        self.context.poll_retire_events();
        let completed = self.context.completed.load(Ordering::Acquire);
        let retired = self.context.device_retired.load(Ordering::Acquire);
        // Device-contiguous retirement covers pruned entries this context may have
        // already lost from `completed` due to a racing poller snapshot.
        completed.max(retired)
    }
}

struct CudaDestroyContext {
    cuda_ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    worker: Arc<SubmissionWorker>,
    fence_shutdown: Arc<AtomicBool>,
    /// Joined in [`Self::wait`] before stream sync so the poller cannot race
    /// `bind_to_thread` / `cuStreamSynchronize` (cudarc records Drop failures into
    /// sticky `CudaContext::error_state`, which then poisons later teardown binds).
    fence_thread: Mutex<Option<JoinHandle<()>>>,
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    companion: Option<Arc<dx12_companion::Dx12Companion>>,
}

/// After DX12 external-semaphore interop, `cuStreamSynchronize` can return
/// `CUDA_ERROR_NOT_SUPPORTED` on some WDDM stacks. Drain sticky context state and
/// treat that case as drained (companion fence already waited).
pub(super) fn cuda_context_stream_sync_after_interop(
    cuda_ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    label: &str,
) -> Result<()> {
    if let Err(e) = cuda_ctx.check_err() {
        tracing::debug!("CUDA: cleared sticky context error before {label}: {e:?}");
    }
    cuda_ctx
        .bind_to_thread()
        .with_context(|| format!("CUDA: bind context before {label}"))?;
    match stream.synchronize() {
        Ok(()) => Ok(()),
        Err(e) if e.0 == sys::CUresult::CUDA_ERROR_NOT_SUPPORTED => {
            tracing::debug!("CUDA: {label} skipped (NOT_SUPPORTED on WDDM+D3D12 interop stack)");
            let _ = cuda_ctx.check_err();
            Ok(())
        }
        Err(e) => Err(e).with_context(|| format!("CUDA: {label}")),
    }
}

impl ContextDestroyHandle for CudaDestroyContext {
    fn wait(&self) -> Result<()> {
        self.fence_shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.fence_thread.lock().unwrap().take() {
            let _ = handle.join();
        }
        self.worker
            .flush()
            .context("CUDA: flush submission worker before context destroy")?;
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        if let Some(companion) = self.companion.as_ref() {
            // Unblock any stream-side WaitExternalFence before cuStreamSynchronize.
            companion
                .wait_idle()
                .context("CUDA/DX12: companion wait_idle before context destroy sync")?;
        }
        cuda_context_stream_sync_after_interop(&self.cuda_ctx, &self.stream, "context destroy stream sync")?;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        // Poller was joined in `wait`; clear any residual handle.
        if let Some(handle) = self.fence_thread.lock().unwrap().take() {
            crate::backend::signal_fence::join_fence_poller(&self.fence_shutdown, Some(handle));
        }
        Ok(())
    }
}

struct CudaDeferredDeletionFlush {
    context: Arc<CudaSubmitContext>,
}

impl ContextDeferredDeletionFlush for CudaDeferredDeletionFlush {
    fn flush(&self) {
        self.context.poll_retire_events();
        drain_deletion_queue_up_to(
            &self.context.deletion_queue,
            self.context.device_retired.load(Ordering::Acquire),
        );
    }
}

fn drain_deletion_queue_up_to(queue: &Mutex<Vec<CudaDeferredDrop>>, retired: u64) {
    let mut guard = queue.lock().unwrap();
    let mut kept = Vec::new();
    for entry in guard.drain(..) {
        let retire_at = match &entry {
            CudaDeferredDrop::Buffer { retire_at, .. }
            | CudaDeferredDrop::Texture { retire_at, .. }
            | CudaDeferredDrop::Pipeline { retire_at, .. } => *retire_at,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            CudaDeferredDrop::RenderTarget { retire_at, .. } => *retire_at,
        };
        if retire_at > retired {
            kept.push(entry);
            continue;
        }
        if let CudaDeferredDrop::Buffer {
            memory,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            shared,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            memory_is_external,
            ..
        } = entry
        {
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            if memory_is_external {
                if let Some(mem) = memory {
                    if let Ok(mutex) = Arc::try_unwrap(mem) {
                        buffer_phys::leak_shared_buffer_slice(mutex.into_inner().unwrap_or_else(|e| e.into_inner()));
                    }
                }
                drop(shared);
                continue;
            }
            drop(memory);
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            drop(shared);
            continue;
        }
        drop(entry);
    }
    *guard = kept;
}

struct CudaBuffer {
    device: DeviceHandle,
    /// `None` while [`CudaPhysKind::Deferred`] (graphics+dx12); always `Some` otherwise.
    memory: Option<Arc<Mutex<CudaSlice<u8>>>>,
    offset: u64,
    size: u64,
    capacity: u64,
    element_stride: Option<u32>,
    /// Access kind from create — Broadcast needs a CBV on the DX12 companion heap.
    kind: BufferKind,
    flags: BufferFlags,
    slot: Option<u32>,
    readback: bool,
    /// Bumped on every host/GPU write that changes contents (retained raster fingerprint).
    content_epoch: u64,
    /// Host-side staging for [`BufferFlags::CPU_WRITABLE`] (DX12 UPLOAD analogue).
    /// `write_buffer` memcpys here without GPU sync; Copy materialization records a
    /// [`CudaOp`] that HtoDs from this Arc at execute time (so retained resubmits see
    /// fresh bytes). Page-locked so memcpy nodes can be CUDA-graph-captured.
    host_staging: Option<Arc<Mutex<CudaPinnedHost>>>,
    /// Parent allocation for [`GpuBackend::create_buffer_view`] slices (shares memory).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    parent: Option<BufferHandle>,
    /// DX12-shareable twin for vertex IA, or the sole backing when phys is Shared.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    shared: Option<Arc<dx12_interop::SharedBufferBacking>>,
    /// `content_epoch` last copied into [`Self::shared`] (NativeAndTwin only).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    shared_epoch: u64,
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    phys_kind: buffer_phys::CudaPhysKind,
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    requirements: buffer_phys::CudaBufferReq,
    /// Host bytes staged before first materialization (`acquire_buffer_with_data`).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    pending_init: Option<Vec<u8>>,
    /// `memory` aliases [`Self::shared`] import and must be leaked on drop.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    memory_is_external: bool,
}

impl CudaBuffer {
    fn memory_arc(&self) -> Result<&Arc<Mutex<CudaSlice<u8>>>> {
        self.memory
            .as_ref()
            .context("CUDA: buffer has no physical memory yet (deferred)")
    }

    fn has_host_staging(&self) -> bool {
        self.host_staging.is_some()
    }
}

#[derive(Clone)]
struct CudaShader {
    device: DeviceHandle,
    source: String,
    search_paths: Vec<String>,
    defines: Vec<(String, String)>,
    optimization_level: crate::types::OptimizationLevel,
}

struct CudaSampler {
    device: DeviceHandle,
    #[allow(dead_code)]
    desc: SamplerDesc,
    slot: u32,
    key: CudaSamplerKey,
}

struct CudaComputeKernel {
    module: Arc<CudaModule>,
    #[allow(dead_code)]
    function: CudaFunction,
    /// From `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` at module load.
    max_threads_per_block: u32,
}

struct CudaComputePipeline {
    device: DeviceHandle,
    shader_handle: ShaderHandle,
    /// Shader snapshot for lazy format specialization (`ShaderModule` may already be destroyed).
    shader: CudaShader,
    workgroup_size: [u32; 3],
    slot_access: Vec<Option<ResourceAccess>>,
    /// Author param order for `[goldy_compute]`; empty for plain Slang compute (all-buffer fallback).
    launch_layout: Vec<CudaLaunchArgKind>,
    /// Default size-matched (`Identity`) specialization.
    identity: CudaComputeKernel,
    /// Non-identity DirectSpatial format specializations (e.g. float4↔Rgba8Unorm pack view).
    variants: Mutex<HashMap<Vec<CudaStorageTextureSpec>, CudaComputeKernel>>,
}

/// Host-side values pushed to `cuLaunchKernel` in shader parameter order.
#[derive(Clone)]
pub(super) enum CudaLaunchArg {
    Buffer(CudaBufferArg),
    /// `CUtexObject`, `CUsurfObject`, or ignored `SamplerState` word.
    Handle(u64),
    Scalar(u32),
}

#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
fn op_writes_imported_scratch(op: &CudaOp, scratch: &Arc<CudaTextureResource>) -> bool {
    match op {
        CudaOp::Launch {
            keep_alive_textures, ..
        }
        | CudaOp::LaunchIndirect {
            keep_alive_textures, ..
        } => keep_alive_textures.iter().any(|texture| Arc::ptr_eq(texture, scratch)),
        CudaOp::WriteTexture { texture, .. }
        | CudaOp::WriteTextureFromHost { texture, .. }
        | CudaOp::CopyBufferToTexture { texture, .. } => Arc::ptr_eq(texture, scratch),
        CudaOp::CopyTexture { dst, .. } => Arc::ptr_eq(dst, scratch),
        _ => false,
    }
}

fn cuda_spec_dump_tag(specs: &[CudaStorageTextureSpec]) -> String {
    if specs.is_empty()
        || specs
            .iter()
            .all(|spec| matches!(spec, CudaStorageTextureSpec::Identity))
    {
        "id".to_owned()
    } else {
        specs
            .iter()
            .map(|spec| match spec {
                CudaStorageTextureSpec::Identity => "id",
                CudaStorageTextureSpec::Float4Rgba8Unorm => "f4rgba8",
                CudaStorageTextureSpec::Float4Bgra8Unorm => "f4bgra8",
            })
            .collect::<Vec<_>>()
            .join("-")
    }
}

fn write_dump_file(dir: &std::path::Path, name: &str, contents: &str) {
    use std::io::Write;
    let path = dir.join(name);
    match std::fs::File::create(&path) {
        Ok(mut file) => {
            if let Err(error) = file.write_all(contents.as_bytes()) {
                tracing::warn!("CUDA: GOLDY_DUMP_SHADERS write {} failed: {error}", path.display());
            } else {
                tracing::info!("Dumped CUDA shader to {}", path.display());
            }
        }
        Err(error) => tracing::warn!("CUDA: GOLDY_DUMP_SHADERS create {} failed: {error}", path.display()),
    }
}

fn maybe_dump_cuda_shaders(
    compiler: &crate::slang::SlangCompiler,
    shader: &CudaShader,
    shader_handle: ShaderHandle,
    storage_specs: &[CudaStorageTextureSpec],
    cuda_source: &str,
    ptx: &str,
    paths: &[&str],
    defines: &[(&str, &str)],
) {
    let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") else {
        return;
    };
    let dir = std::path::Path::new(&dump_dir);
    if let Err(error) = std::fs::create_dir_all(dir) {
        tracing::warn!("CUDA: GOLDY_DUMP_SHADERS create_dir_all {dump_dir} failed: {error}");
        return;
    }
    let tag = cuda_spec_dump_tag(storage_specs);
    let stem = format!("cs_main_h{shader_handle}_{tag}_cuda");
    write_dump_file(dir, &format!("{stem}.ptx"), ptx);
    match compiler.compile_bindless_with_reflection_and_defines(
        cuda_source,
        crate::slang::ShaderTarget::CudaSource,
        &[("cs_main", crate::slang::SlangStage::Compute)],
        paths,
        defines,
        &[],
        shader.optimization_level,
    ) {
        Ok(compiled) => match compiled.shader.as_str() {
            Some(cu) => write_dump_file(dir, &format!("{stem}.cu"), cu),
            None => tracing::warn!("CUDA: GOLDY_DUMP_SHADERS Slang CUDA C++ was not text"),
        },
        Err(error) => tracing::warn!("CUDA: GOLDY_DUMP_SHADERS Slang CUDA C++ compile failed: {error:#}"),
    }
}

impl CudaBackend {
    pub(crate) fn graph_stats_snapshot(&self) -> CudaGraphStatsSnapshot {
        self.graph_stats.snapshot()
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    pub(crate) fn buffer_phys_kind_for_test(&self, buffer: BufferHandle) -> Option<&'static str> {
        let buf = self.buffers.get(&buffer)?;
        Some(match buf.phys_kind {
            buffer_phys::CudaPhysKind::Deferred => "deferred",
            buffer_phys::CudaPhysKind::Native => "native",
            buffer_phys::CudaPhysKind::Shared => "shared",
            buffer_phys::CudaPhysKind::NativeAndTwin => "native_and_twin",
        })
    }

    pub(crate) fn new() -> Result<Self> {
        let _span = goldy_span!("backend.cuda.init").entered();
        tracing::info!("Initializing CUDA backend");
        ensure_cuda_toolkit_on_path();
        // `CUDA_LAUNCH_BLOCKING` must be set before driver work begins. Use a
        // process-wide Once so parallel test threads do not race on `set_var`.
        CUDA_VALIDATION_INIT.call_once(|| {
            if crate::backend::goldy_validation_enabled() && std::env::var_os("CUDA_LAUNCH_BLOCKING").is_none() {
                // SAFETY: called exactly once per process, before device enumeration below.
                unsafe { std::env::set_var("CUDA_LAUNCH_BLOCKING", "1") };
                tracing::info!("Set CUDA_LAUNCH_BLOCKING=1 (GOLDY_VALIDATION api)");
            }
        });
        cudarc::driver::result::init().context("CUDA: driver init failed")?;
        let driver_version = ensure_cuda_driver_at_least_13_1()?;
        let count = CudaContext::device_count().context("CUDA: enumerate devices")?;
        if count <= 0 {
            anyhow::bail!("CUDA: no devices found");
        }
        let mut adapter_info = Vec::with_capacity(count as usize);
        for ordinal in 0..count {
            let ctx = CudaContext::new(ordinal as usize).with_context(|| format!("CUDA: open device {ordinal}"))?;
            let name = ctx.name().unwrap_or_else(|_| format!("CUDA device {ordinal}"));
            let major = ctx
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .unwrap_or(0);
            let minor = ctx
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .unwrap_or(0);
            tracing::info!("  [{ordinal}] {name} (DiscreteGpu) - compute capability {major}.{minor}");
            adapter_info.push(AdapterInfo {
                id: ordinal as u32,
                name,
                vendor: "NVIDIA".to_string(),
                backend: BackendType::Cuda,
                device_type: DeviceType::DiscreteGpu,
            });
        }
        tracing::info!("Found {count} CUDA device(s) (driver {driver_version})");
        goldy_event!(
            "backend.cuda.init",
            device_count = count,
            driver_version = driver_version,
            success = true
        );
        let slang_compiler = crate::slang::SlangCompiler::new().context("CUDA: initialize Slang")?;
        Ok(Self {
            adapter_info,
            devices: HashMap::new(),
            contexts: HashMap::new(),
            buffers: HashMap::new(),
            buffer_slots: HashMap::new(),
            textures: HashMap::new(),
            texture_slots: HashMap::new(),
            samplers: HashMap::new(),
            sampler_slots: HashMap::new(),
            shaders: HashMap::new(),
            compute_pipelines: HashMap::new(),
            retained: HashMap::new(),
            graph_stats: Arc::new(CudaGraphStats::default()),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            surfaces: HashMap::new(),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            pipelines: HashMap::new(),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            render_targets: HashMap::new(),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            raster_list_cache: HashMap::new(),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            texture_dx12: HashMap::new(),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            texture_imports: HashMap::new(),
            slang_compiler,
            next_device: 1,
            next_context: 1,
            next_buffer: 1,
            next_texture: 1,
            next_sampler: 1,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            next_slot: dx12_bindless::USER_SLOT_BASE,
            #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
            next_slot: 0,
            next_shader: 1,
            next_compute_pipeline: 1,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            next_surface: 1,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            next_pipeline: 1,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            next_render_target: 1,
        })
    }

    pub(crate) fn graph_stats(&self) -> Arc<CudaGraphStats> {
        Arc::clone(&self.graph_stats)
    }

    fn device(&self, handle: DeviceHandle) -> Result<&CudaDevice> {
        self.devices.get(&handle).context("CUDA: invalid device handle")
    }

    fn context(&self, handle: ContextHandle) -> Result<&Arc<CudaSubmitContext>> {
        self.contexts.get(&handle).context("CUDA: invalid context handle")
    }

    fn sync_device_streams_for_immediate_api(&mut self, device: DeviceHandle) -> Result<()> {
        // Immediate host writes (deposit staging, clear, resize) run on `alloc_stream`.
        // Do **not** host-synchronize submission context streams here: on WDDM+D3D12 they
        // carry WaitExternalFence/SignalExternalFence and `cuStreamSynchronize` deposits
        // sticky CUDA_ERROR_NOT_SUPPORTED (or AVs). Deposit pools already epoch-gate
        // staging reuse via gpu_progress; surface teardown waits DX12 before those syncs.
        let (worker, alloc_stream, cuda_ctx) = {
            let gpu = self.device(device)?;
            (
                Arc::clone(&gpu.submission_worker),
                Arc::clone(&gpu.alloc_stream),
                Arc::clone(&gpu.ctx),
            )
        };
        worker.flush()?;
        cuda_context_stream_sync_after_interop(&cuda_ctx, &alloc_stream, "sync alloc stream for immediate API")?;
        for context in self.contexts.values().filter(|context| context.device == device) {
            context.poll_retire_events();
        }
        Ok(())
    }

    #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
    fn unsupported<T>(operation: &str) -> Result<T> {
        anyhow::bail!("CUDA compute-only backend does not support {operation}")
    }

    fn create_storage_buffer(
        &mut self,
        device: DeviceHandle,
        logical_size: u64,
        capacity: u64,
        element_stride: Option<u32>,
        kind: BufferKind,
        flags: BufferFlags,
    ) -> Result<BufferHandle> {
        let mut capacity = capacity.max(logical_size).max(4);
        // D3D12 CBVs require a 256-byte aligned range that fits in the resource.
        if kind == BufferKind::Broadcast {
            capacity = capacity.max((logical_size + 255) & !255);
        }
        let gpu = self.device(device)?;
        let _gate = capture_gate::lock_capture_alloc_gate();
        // Graphics+DX12: defer physical backing until scheme usage is known (Shared vs
        // Native vs NativeAndTwin). Compute-only builds still allocate eagerly.
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        let memory = None;
        #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
        let memory = Some(Arc::new(Mutex::new(
            gpu.alloc_stream
                .alloc_zeros::<u8>(capacity as usize)
                .context("CUDA: alloc buffer")?,
        )));
        let host_staging = if flags.contains(BufferFlags::CPU_WRITABLE) {
            Some(Arc::new(Mutex::new(CudaPinnedHost::alloc(
                &gpu.ctx,
                capacity as usize,
            )?)))
        } else {
            None
        };
        let _ = &gpu; // used in non-graphics branch for `alloc_stream`
        let handle = self.next_buffer;
        self.next_buffer += 1;
        let slot = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .context("CUDA buffer registry exhausted")?;
        self.buffer_slots.insert(slot, handle);
        self.buffers.insert(
            handle,
            CudaBuffer {
                device,
                memory,
                offset: 0,
                size: logical_size,
                capacity,
                element_stride,
                kind,
                flags,
                host_staging,
                slot: Some(slot),
                readback: false,
                content_epoch: 0,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                parent: None,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                shared: None,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                shared_epoch: 0,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                phys_kind: buffer_phys::CudaPhysKind::Deferred,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                requirements: buffer_phys::CudaBufferReq::empty(),
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                pending_init: None,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                memory_is_external: false,
            },
        );
        Ok(handle)
    }

    fn compile_compute_ptx(
        &self,
        shader: &CudaShader,
        shader_handle: ShaderHandle,
    ) -> Result<(String, Vec<Option<ResourceAccess>>, [u32; 3], Vec<CudaLaunchArgKind>)> {
        self.compile_compute_ptx_with_specs(shader, shader_handle, &[])
    }

    fn compile_compute_ptx_with_specs(
        &self,
        shader: &CudaShader,
        shader_handle: ShaderHandle,
        storage_specs: &[CudaStorageTextureSpec],
    ) -> Result<(String, Vec<Option<ResourceAccess>>, [u32; 3], Vec<CudaLaunchArgKind>)> {
        ensure_cuda_toolkit_on_path();
        let paths: Vec<&str> = shader.search_paths.iter().map(String::as_str).collect();
        let defines: Vec<(&str, &str)> = shader
            .defines
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        let launch_layout = crate::slang::virtual_main::extract_cuda_compute_launch_layout(&shader.source, &defines)
            .map_err(|error| anyhow::anyhow!("CUDA launch layout failed: {error}"))?;
        let cuda_source = if storage_specs.is_empty()
            || storage_specs
                .iter()
                .all(|spec| matches!(spec, CudaStorageTextureSpec::Identity))
        {
            crate::slang::virtual_main::transform_virtual_main_cuda_compute(&shader.source, &defines)
                .map_err(|error| anyhow::anyhow!("CUDA shader lowering failed: {error}"))?
        } else {
            crate::slang::virtual_main::transform_virtual_main_cuda_compute_specialized(
                &shader.source,
                &defines,
                storage_specs,
            )
            .map_err(|error| anyhow::anyhow!("CUDA shader specialization failed: {error}"))?
        };
        let workgroup_size = crate::slang::parse_numthreads(&shader.source).unwrap_or([1, 1, 1]);
        let compiled = self.slang_compiler.compile_bindless_with_reflection_and_defines(
            &cuda_source,
            crate::slang::ShaderTarget::Ptx,
            &[("cs_main", crate::slang::SlangStage::Compute)],
            &paths,
            &defines,
            &[],
            shader.optimization_level,
        )?;
        let mut ptx = compiled
            .shader
            .as_str()
            .context("CUDA: Slang returned non-text PTX output")?
            .to_owned();
        while ptx.ends_with('\0') {
            ptx.pop();
        }
        maybe_dump_cuda_shaders(
            &self.slang_compiler,
            shader,
            shader_handle,
            storage_specs,
            &cuda_source,
            &ptx,
            &paths,
            &defines,
        );
        let access = crate::slang::virtual_main::extract_push_constant_categories(&shader.source)
            .iter()
            .map(|category| {
                category.map(|category| match category {
                    crate::types::ResourceCategory::Broadcast
                    | crate::types::ResourceCategory::Texture
                    | crate::types::ResourceCategory::Sampler | crate::types::ResourceCategory::Accel => {
                        ResourceAccess::Read
                    }
                    crate::types::ResourceCategory::Scattered | crate::types::ResourceCategory::StorageImage => {
                        ResourceAccess::ReadWrite
                    }
                })
            })
            .collect();
        Ok((ptx, access, workgroup_size, launch_layout))
    }

    fn load_compute_kernel(&self, device: DeviceHandle, ptx: &str) -> Result<CudaComputeKernel> {
        let gpu = self.device(device)?;
        let _gate = capture_gate::lock_capture_alloc_gate();
        let module = load_ptx_module(&gpu.ctx, ptx)?;
        let function = module
            .load_function("cs_main")
            .context("CUDA: cuModuleGetFunction(cs_main) failed")?;
        let max_threads_per_block = function
            .max_threads_per_block()
            .context("CUDA: query CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK failed")?
            .max(0) as u32;
        Ok(CudaComputeKernel {
            module,
            function,
            max_threads_per_block,
        })
    }

    /// Resolve per-`DirectSpatial` format specs from bound textures for this launch.
    fn resolve_storage_texture_specs(
        &self,
        launch_layout: &[CudaLaunchArgKind],
        indices: &[u32],
    ) -> Result<Vec<CudaStorageTextureSpec>> {
        if launch_layout.is_empty() {
            return Ok(Vec::new());
        }
        let mut formats = Vec::new();
        let mut index_i = 0usize;
        for kind in launch_layout {
            match kind {
                CudaLaunchArgKind::StorageTexture { .. } => {
                    let index = *indices.get(index_i).with_context(|| {
                        format!("CUDA: missing registry index for storage texture at binding {index_i}")
                    })?;
                    let tex = self.resolve_texture(index_i, index)?;
                    formats.push(tex.format);
                    index_i += 1;
                }
                CudaLaunchArgKind::Buffer | CudaLaunchArgKind::SampledTexture { .. } | CudaLaunchArgKind::Sampler => {
                    index_i += 1;
                }
                CudaLaunchArgKind::Scalar => {}
            }
        }
        crate::slang::virtual_main::derive_cuda_storage_texture_specs(launch_layout, &formats)
            .map_err(|error| anyhow::anyhow!("{error}"))
    }

    /// Identity kernel, or a lazily compiled format-specialized PTX variant.
    fn ensure_compute_kernel(
        &self,
        pipeline_handle: ComputePipelineHandle,
        specs: &[CudaStorageTextureSpec],
    ) -> Result<(Arc<CudaModule>, u32)> {
        let pipeline = self
            .compute_pipelines
            .get(&pipeline_handle)
            .context("CUDA: invalid compute pipeline")?;
        let identity_specs = specs
            .iter()
            .all(|spec| matches!(spec, CudaStorageTextureSpec::Identity));
        if specs.is_empty() || identity_specs {
            return Ok((
                Arc::clone(&pipeline.identity.module),
                pipeline.identity.max_threads_per_block,
            ));
        }

        {
            let variants = pipeline.variants.lock().unwrap();
            if let Some(kernel) = variants.get(specs) {
                return Ok((Arc::clone(&kernel.module), kernel.max_threads_per_block));
            }
        }

        let (ptx, _, _, _) = self.compile_compute_ptx_with_specs(&pipeline.shader, pipeline.shader_handle, specs)?;
        let kernel = self.load_compute_kernel(pipeline.device, &ptx)?;

        let mut variants = pipeline.variants.lock().unwrap();
        let entry = variants.entry(specs.to_vec()).or_insert(kernel);
        Ok((Arc::clone(&entry.module), entry.max_threads_per_block))
    }

    fn buffer_arg(&self, stream: &Arc<CudaStream>, buffer: &CudaBuffer) -> Result<CudaBufferArg> {
        let _gate = capture_gate::lock_capture_alloc_gate();
        let memory = buffer.memory_arc()?.lock().unwrap();
        let start = buffer.offset as usize;
        let end = (buffer.offset + buffer.size) as usize;
        let view = memory.try_slice(start..end).context("CUDA: buffer view out of range")?;
        let (ptr, _sync) = view.device_ptr(stream);
        let stride = buffer.element_stride.unwrap_or(1).max(1) as u64;
        let count = if buffer.size == 0 {
            0
        } else {
            (buffer.size / stride) as usize
        };
        if crate::backend::goldy_validation_enabled() {
            if ptr == 0 {
                anyhow::bail!("CUDA validation: StructuredBuffer device pointer is null");
            }
            let expected = if buffer.size == 0 {
                0usize
            } else {
                (buffer.size / stride) as usize
            };
            if count != expected {
                anyhow::bail!(
                    "CUDA validation: StructuredBuffer count {count} != logical_size/stride {expected} \
                     (size={}, stride={stride})",
                    buffer.size
                );
            }
        }
        Ok(CudaBufferArg { data: ptr, count })
    }

    fn resolve_buffer_arg(&self, stream: &Arc<CudaStream>, binding: usize, index: u32) -> Result<CudaBufferArg> {
        let handle = self
            .buffer_slots
            .get(&index)
            .with_context(|| format!("CUDA: binding {binding} references unknown registry key {index}"))?;
        let buffer = self
            .buffers
            .get(handle)
            .with_context(|| format!("CUDA: registry key {index} references a destroyed buffer"))?;
        self.buffer_arg(stream, buffer)
    }

    /// Build launch args in shader parameter order.
    ///
    /// Empty `launch_layout` means plain (non-`[goldy_compute]`) Slang: one buffer arg
    /// per registry index and no scalars.
    fn build_launch_args(
        &self,
        stream: &Arc<CudaStream>,
        launch_layout: &[CudaLaunchArgKind],
        indices: &[u32],
        user: &[u32],
    ) -> Result<Vec<CudaLaunchArg>> {
        if launch_layout.is_empty() {
            if !user.is_empty() {
                anyhow::bail!(
                    "CUDA: scalar user params require a [goldy_compute] entry; got {} user word(s)",
                    user.len()
                );
            }
            let mut args = Vec::with_capacity(indices.len());
            for (binding, index) in indices.iter().copied().enumerate() {
                args.push(CudaLaunchArg::Buffer(self.resolve_buffer_arg(stream, binding, index)?));
            }
            return Ok(args);
        }

        let expected_indices = launch_layout
            .iter()
            .filter(|kind| kind.consumes_registry_index())
            .count();
        let expected_scalars = launch_layout
            .iter()
            .filter(|kind| matches!(kind, CudaLaunchArgKind::Scalar))
            .count();
        if indices.len() != expected_indices {
            anyhow::bail!(
                "CUDA: dispatch bound {} resource index(es) but shader expects {expected_indices}",
                indices.len()
            );
        }
        if user.len() != expected_scalars {
            anyhow::bail!(
                "CUDA: dispatch provided {} scalar user word(s) but shader expects {expected_scalars}",
                user.len()
            );
        }

        // Resolve the single effective sampler configuration for this dispatch.
        let mut sampler_keys = Vec::new();
        let mut index_i = 0usize;
        for kind in launch_layout {
            match kind {
                CudaLaunchArgKind::Sampler => {
                    let index = indices[index_i];
                    index_i += 1;
                    let handle = self
                        .sampler_slots
                        .get(&index)
                        .with_context(|| format!("CUDA: sampler binding references unknown registry key {index}"))?;
                    let sampler = self
                        .samplers
                        .get(handle)
                        .with_context(|| format!("CUDA: registry key {index} references a destroyed sampler"))?;
                    sampler_keys.push(sampler.key);
                }
                CudaLaunchArgKind::Scalar => {}
                CudaLaunchArgKind::Buffer
                | CudaLaunchArgKind::SampledTexture { .. }
                | CudaLaunchArgKind::StorageTexture { .. } => {
                    index_i += 1;
                }
            }
        }
        let effective_sampler = match sampler_keys.as_slice() {
            [] => CudaSamplerKey::nearest_clamp(),
            [first] => *first,
            many => {
                let first = many[0];
                if many.iter().any(|key| *key != first) {
                    anyhow::bail!(
                        "CUDA: at most one distinct Filter configuration per dispatch \
                         (CUDA bakes sampler state into each CUtexObject)"
                    );
                }
                first
            }
        };

        let mut args = Vec::with_capacity(launch_layout.len());
        let mut index_i = 0usize;
        let mut user_i = 0usize;
        for kind in launch_layout {
            match kind {
                CudaLaunchArgKind::Buffer => {
                    let index = indices[index_i];
                    args.push(CudaLaunchArg::Buffer(self.resolve_buffer_arg(stream, index_i, index)?));
                    index_i += 1;
                }
                CudaLaunchArgKind::SampledTexture { .. } => {
                    let index = indices[index_i];
                    let tex = self.resolve_texture(index_i, index)?;
                    let handle = tex.tex_object(effective_sampler)?;
                    args.push(CudaLaunchArg::Handle(handle));
                    index_i += 1;
                }
                CudaLaunchArgKind::StorageTexture { element } => {
                    let index = indices[index_i];
                    let tex = self.resolve_texture(index_i, index)?;
                    if !storage_shader_convertible(element, tex.format) {
                        anyhow::bail!(
                            "CUDA: DirectSpatial<{element}> cannot access {:?}; \
                             supported: float4↔Rgba32Float|Rgba8Unorm (packed), \
                             half4↔Rgba16Float, uint8_t4↔Rgba8Unorm",
                            tex.format
                        );
                    }
                    let handle = tex.surf_object()?;
                    args.push(CudaLaunchArg::Handle(handle));
                    index_i += 1;
                }
                CudaLaunchArgKind::Sampler => {
                    // Slang emits an unused SamplerState parameter (pointer-sized).
                    let _index = indices[index_i];
                    args.push(CudaLaunchArg::Handle(0));
                    index_i += 1;
                }
                CudaLaunchArgKind::Scalar => {
                    args.push(CudaLaunchArg::Scalar(user[user_i]));
                    user_i += 1;
                }
            }
        }
        Ok(args)
    }

    fn resolve_texture(&self, binding: usize, index: u32) -> Result<&Arc<CudaTextureResource>> {
        let handle = self
            .texture_slots
            .get(&index)
            .with_context(|| format!("CUDA: binding {binding} references unknown texture registry key {index}"))?;
        self.textures
            .get(handle)
            .with_context(|| format!("CUDA: registry key {index} references a destroyed texture"))
    }

    fn alloc_registry_slot(&mut self) -> u32 {
        let slot = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .expect("CUDA: registry slot counter overflow");
        slot
    }

    /// Create a shareable D3D12 texture imported into CUDA and register bindless descriptors.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    #[allow(clippy::too_many_arguments)]
    fn create_shared_texture_resource(
        &mut self,
        device: DeviceHandle,
        companion: &Arc<dx12_companion::Dx12Companion>,
        cuda_ctx: &Arc<CudaContext>,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
        storage_slot: Option<u32>,
        sampled_slot: Option<u32>,
    ) -> Result<TextureHandle> {
        use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL};
        use windows::Win32::Graphics::Direct3D12::*;

        let dxgi = dx12_bindless::texture_format_to_dxgi(format)?;
        let mut resource_flags = D3D12_RESOURCE_FLAG_NONE;
        if matches!(access, TextureKind::Direct | TextureKind::DirectInterpolated) {
            resource_flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }
        let (d3d12_resource, allocation_size) =
            dx12_interop::create_shared_texture(&companion.device, width, height, dxgi, resource_flags)?;
        let handle_nt = unsafe {
            companion
                .device
                .CreateSharedHandle(&d3d12_resource, None, GENERIC_ALL.0, None)
        }
        .context("CUDA/DX12: CreateSharedHandle(texture) failed")?;
        let import = dx12_interop::import_shared_texture(cuda_ctx, handle_nt, allocation_size, width, height, format)?;
        unsafe {
            let _ = CloseHandle(handle_nt);
        }
        if matches!(access, TextureKind::Direct | TextureKind::DirectInterpolated) {
            dx12_interop::init_resource_state(companion, &d3d12_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
        }
        let resource = CudaTextureResource::from_imported_array(
            cuda_ctx,
            import.level0(),
            width,
            height,
            format,
            access,
            flags,
            storage_slot,
            sampled_slot,
        )?;
        // Keep import alive by leaking into a side table keyed by the texture — the
        // D3D12 resource map holds the graphics object; import must outlive CUDA views.
        // Store import on a dedicated map by retaining the SharedScratchTexture pattern
        // via texture_dx12 + dropping import last through deferred drop of the Arc texture
        // (array is borrowed; import must live in texture_imports).
        let handle = self.next_texture;
        self.next_texture += 1;
        if let Some(slot) = storage_slot {
            self.texture_slots.insert(slot, handle);
            companion
                .bindless
                .write_texture_uav(&companion.device, slot, &d3d12_resource, format)?;
        }
        if let Some(slot) = sampled_slot {
            self.texture_slots.insert(slot, handle);
            companion
                .bindless
                .write_texture_srv(&companion.device, slot, &d3d12_resource, format)?;
        }
        self.texture_dx12.insert(handle, d3d12_resource);
        // Pin import for the lifetime of the texture handle.
        self.texture_imports.insert(handle, import);
        let _ = device;
        self.textures.insert(handle, resource);
        Ok(handle)
    }

    fn texture_device_is(&self, tex: &CudaTextureResource, device: DeviceHandle) -> bool {
        self.devices
            .get(&device)
            .map(|d| Arc::ptr_eq(&d.ctx, &tex.ctx))
            .unwrap_or(false)
    }

    fn write_buffer_region(stream: &Arc<CudaStream>, buffer: &CudaBuffer, offset: u64, data: &[u8]) -> Result<()> {
        if offset + data.len() as u64 > buffer.size {
            anyhow::bail!("CUDA: write exceeds logical buffer size");
        }
        let memory = buffer
            .memory
            .as_ref()
            .context("CUDA: write_buffer_region on deferred buffer")?;
        let mut memory = memory.lock().unwrap();
        let start = (buffer.offset + offset) as usize;
        let end = start + data.len();
        let mut view = memory
            .try_slice_mut(start..end)
            .context("CUDA: write range out of bounds")?;
        stream.memcpy_htod(data, &mut view).context("CUDA: HtoD write failed")?;
        // Deposit staging uses alloc_stream; submit copies on the context stream.
        stream
            .synchronize()
            .context("CUDA: sync alloc stream after host write")?;
        pending_submit::maybe_validate_sync(stream, "immediate WriteBuffer")
    }

    fn clear_buffer_region(stream: &Arc<CudaStream>, buffer: &CudaBuffer, offset: u64, size: u64) -> Result<()> {
        let clear_size = if size == 0 {
            buffer.size.saturating_sub(offset)
        } else {
            size
        };
        let memory = buffer
            .memory
            .as_ref()
            .context("CUDA: clear_buffer_region on deferred buffer")?;
        let mut memory = memory.lock().unwrap();
        let start = (buffer.offset + offset) as usize;
        let end = start + clear_size as usize;
        let mut view = memory
            .try_slice_mut(start..end)
            .context("CUDA: clear range out of bounds")?;
        stream.memset_zeros(&mut view).context("CUDA: memset failed")?;
        pending_submit::maybe_validate_sync(stream, "immediate ClearBuffer")
    }

    #[allow(dead_code)]
    fn copy_buffer_region(
        stream: &Arc<CudaStream>,
        src: &CudaBuffer,
        src_offset: u64,
        dst: &CudaBuffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        if src.device != dst.device {
            anyhow::bail!("CUDA: copy across devices is not supported");
        }
        if src_offset + size > src.size {
            anyhow::bail!("CUDA: copy source range exceeds logical buffer size");
        }
        if dst_offset + size > dst.size {
            anyhow::bail!("CUDA: copy destination range exceeds logical buffer size");
        }

        let src_abs = src.offset + src_offset;
        let dst_abs = dst.offset + dst_offset;
        let byte_len = size as usize;

        if Arc::ptr_eq(src.memory_arc()?, dst.memory_arc()?) {
            // Same allocation: avoid simultaneous &/&mut CudaSlice views. A device temp
            // keeps both overlapping and non-overlapping self-copies memmove-safe.
            let mut temp = stream
                .alloc_zeros::<u8>(byte_len)
                .context("CUDA: alloc overlapping-copy scratch")?;
            {
                let memory = src.memory_arc()?.lock().unwrap();
                let src_view = memory
                    .try_slice(src_abs as usize..src_abs as usize + byte_len)
                    .context("CUDA: copy source out of bounds")?;
                stream
                    .memcpy_dtod(&src_view, &mut temp)
                    .context("CUDA: same-alloc copy to scratch")?;
            }
            {
                let mut memory = dst.memory_arc()?.lock().unwrap();
                let mut dst_view = memory
                    .try_slice_mut(dst_abs as usize..dst_abs as usize + byte_len)
                    .context("CUDA: copy destination out of bounds")?;
                stream
                    .memcpy_dtod(&temp, &mut dst_view)
                    .context("CUDA: same-alloc copy from scratch")?;
            }
            return Ok(());
        }

        // Distinct allocations: lock in pointer order to avoid AB/BA deadlocks.
        let src_arc = Arc::clone(src.memory_arc()?);
        let dst_arc = Arc::clone(dst.memory_arc()?);
        let src_ptr = Arc::as_ptr(&src_arc);
        let dst_ptr = Arc::as_ptr(&dst_arc);
        if src_ptr < dst_ptr {
            let src_guard = src_arc.lock().unwrap();
            let mut dst_guard = dst_arc.lock().unwrap();
            let src_view = src_guard
                .try_slice(src_abs as usize..src_abs as usize + byte_len)
                .context("CUDA: copy source out of bounds")?;
            let mut dst_view = dst_guard
                .try_slice_mut(dst_abs as usize..dst_abs as usize + byte_len)
                .context("CUDA: copy destination out of bounds")?;
            stream
                .memcpy_dtod(&src_view, &mut dst_view)
                .context("CUDA: device-to-device copy failed")?;
        } else {
            let mut dst_guard = dst_arc.lock().unwrap();
            let src_guard = src_arc.lock().unwrap();
            let src_view = src_guard
                .try_slice(src_abs as usize..src_abs as usize + byte_len)
                .context("CUDA: copy source out of bounds")?;
            let mut dst_view = dst_guard
                .try_slice_mut(dst_abs as usize..dst_abs as usize + byte_len)
                .context("CUDA: copy destination out of bounds")?;
            stream
                .memcpy_dtod(&src_view, &mut dst_view)
                .context("CUDA: device-to-device copy failed")?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn launch_compute(
        &self,
        stream: &Arc<CudaStream>,
        pipeline_handle: ComputePipelineHandle,
        indices: &[u32],
        user: &[u32],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        let pipeline = self
            .compute_pipelines
            .get(&pipeline_handle)
            .context("CUDA: invalid compute pipeline")?;
        let specs = self.resolve_storage_texture_specs(&pipeline.launch_layout, indices)?;
        let (module, max_threads_per_block) = self.ensure_compute_kernel(pipeline_handle, &specs)?;
        let limits = self.device(pipeline.device)?.limits;
        validate_launch_config(
            &limits,
            max_threads_per_block,
            workgroups,
            pipeline.workgroup_size,
            0,
            None,
        )?;
        let launch_args = self.build_launch_args(stream, &pipeline.launch_layout, indices, user)?;
        let function = module
            .load_function("cs_main")
            .context("CUDA: cuModuleGetFunction(cs_main) failed")?;
        let cfg = LaunchConfig {
            grid_dim: workgroups,
            block_dim: (
                pipeline.workgroup_size[0],
                pipeline.workgroup_size[1],
                pipeline.workgroup_size[2],
            ),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order/types match the Slang CUDA entry signature.
        unsafe {
            let mut builder = stream.launch_builder(&function);
            for arg in &launch_args {
                match arg {
                    CudaLaunchArg::Buffer(buffer) => {
                        builder.arg(buffer);
                    }
                    CudaLaunchArg::Handle(handle) => {
                        builder.arg(handle);
                    }
                    CudaLaunchArg::Scalar(word) => {
                        builder.arg(word);
                    }
                }
            }
            builder.launch(cfg).context("CUDA: cuLaunchKernel failed")?;
        }
        Ok(())
    }

    fn launch_layout_index_count(launch_layout: &[CudaLaunchArgKind]) -> Option<usize> {
        if launch_layout.is_empty() {
            None
        } else {
            Some(
                launch_layout
                    .iter()
                    .filter(|kind| kind.consumes_registry_index())
                    .count(),
            )
        }
    }

    fn launch_layout_scalar_count(launch_layout: &[CudaLaunchArgKind]) -> usize {
        launch_layout
            .iter()
            .filter(|kind| matches!(kind, CudaLaunchArgKind::Scalar))
            .count()
    }

    fn materialize_dispatch_batch(
        &self,
        stream: &Arc<CudaStream>,
        pipeline_handle: ComputePipelineHandle,
        frame_table: Option<&[u32]>,
        arg_data: &[u8],
        count: u32,
        label: Option<&'static str>,
    ) -> Result<Vec<CudaOp>> {
        let pipeline = self
            .compute_pipelines
            .get(&pipeline_handle)
            .context("CUDA: invalid compute pipeline")?;
        let entry_count = count as usize;
        if entry_count == 0 {
            return Ok(Vec::new());
        }
        let needed = entry_count
            .checked_mul(DISPATCH_BATCH_STRIDE)
            .context("CUDA: DispatchBatch stride overflow")?;
        anyhow::ensure!(
            arg_data.len() >= needed,
            "CUDA: DispatchBatch arg_data len {} < {} entries × stride {}",
            arg_data.len(),
            entry_count,
            DISPATCH_BATCH_STRIDE
        );

        let mut bases = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let base = i * DISPATCH_BATCH_STRIDE;
            let layout: PushLayout = *bytemuck::from_bytes(&arg_data[base..base + TOTAL_PUSH_BYTES]);
            bases.push(layout._reserved[dispatch_table_base_word_index()]);
        }

        let n_buffers = match Self::launch_layout_index_count(&pipeline.launch_layout) {
            Some(n) => n,
            None => {
                anyhow::ensure!(
                    entry_count >= 2,
                    "CUDA: DispatchBatch with empty launch layout requires at least 2 entries"
                );
                let delta = bases[1]
                    .checked_sub(bases[0])
                    .context("CUDA: invalid frame-table bases")?;
                for window in bases.windows(2) {
                    anyhow::ensure!(
                        window[1].saturating_sub(window[0]) == delta,
                        "CUDA: DispatchBatch frame-table bases are not uniformly spaced ({bases:?})"
                    );
                }
                delta as usize
            }
        };
        let n_scalars = Self::launch_layout_scalar_count(&pipeline.launch_layout);

        if n_buffers > 0 {
            let table =
                frame_table.context("CUDA: DispatchBatch requires FrameTableStaging when bindings are present")?;
            for (i, &table_base) in bases.iter().enumerate() {
                let start = table_base as usize;
                let end = start
                    .checked_add(n_buffers)
                    .context("CUDA: frame-table range overflow")?;
                anyhow::ensure!(
                    end <= table.len(),
                    "CUDA: DispatchBatch entry {i} frame-table range [{start}, {end}) exceeds staging len {}",
                    table.len()
                );
            }
        }

        let mut ops = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let base = i * DISPATCH_BATCH_STRIDE;
            let layout: PushLayout = *bytemuck::from_bytes(&arg_data[base..base + TOTAL_PUSH_BYTES]);
            let wg_off = base + TOTAL_PUSH_BYTES;
            let wg_x = u32::from_ne_bytes(arg_data[wg_off..wg_off + 4].try_into().unwrap());
            let wg_y = u32::from_ne_bytes(arg_data[wg_off + 4..wg_off + 8].try_into().unwrap());
            let wg_z = u32::from_ne_bytes(arg_data[wg_off + 8..wg_off + 12].try_into().unwrap());

            let indices: Vec<u32> = if n_buffers == 0 {
                Vec::new()
            } else {
                let table = frame_table.expect("validated above");
                let start = bases[i] as usize;
                table[start..start + n_buffers].to_vec()
            };
            let user = if n_scalars == 0 {
                Vec::new()
            } else {
                anyhow::ensure!(
                    n_scalars <= MAX_USER_SLOTS,
                    "CUDA: DispatchBatch entry {i} expects {n_scalars} scalars (max {MAX_USER_SLOTS})"
                );
                layout.user[..n_scalars].to_vec()
            };

            ops.push(
                self.materialize_launch(stream, pipeline_handle, &indices, &user, (wg_x, wg_y, wg_z), label)
                    .with_context(|| format!("CUDA: DispatchBatch entry {i} launch failed"))?,
            );
        }
        Ok(ops)
    }

    fn materialize_launch(
        &self,
        stream: &Arc<CudaStream>,
        pipeline_handle: ComputePipelineHandle,
        indices: &[u32],
        user: &[u32],
        workgroups: (u32, u32, u32),
        label: Option<&'static str>,
    ) -> Result<CudaOp> {
        let pipeline = self
            .compute_pipelines
            .get(&pipeline_handle)
            .context("CUDA: invalid compute pipeline")?;
        let specs = self.resolve_storage_texture_specs(&pipeline.launch_layout, indices)?;
        let (module, max_threads_per_block) = self.ensure_compute_kernel(pipeline_handle, &specs)?;
        // Format-specialized PTX variants stay on the stream-replay path (multi-island
        // stream segments). Identity DirectSpatial stays capturable.
        let graph_capture_ok = specs
            .iter()
            .all(|spec| matches!(spec, CudaStorageTextureSpec::Identity));
        let limits = self.device(pipeline.device)?.limits;
        validate_launch_config(
            &limits,
            max_threads_per_block,
            workgroups,
            pipeline.workgroup_size,
            0,
            label,
        )?;
        let launch_args = self.build_launch_args(stream, &pipeline.launch_layout, indices, user)?;
        let (keep_alive_buffers, keep_alive_textures) = self.collect_launch_pins(indices)?;
        let function = module
            .load_function("cs_main")
            .context("CUDA: cuModuleGetFunction(cs_main) failed")?;
        Ok(CudaOp::Launch {
            label,
            function,
            module,
            workgroup_size: pipeline.workgroup_size,
            grid: workgroups,
            args: launch_args,
            keep_alive_buffers,
            keep_alive_textures,
            graph_capture_ok,
        })
    }

    fn materialize_launch_indirect(
        &self,
        stream: &Arc<CudaStream>,
        pipeline_handle: ComputePipelineHandle,
        indices: &[u32],
        user: &[u32],
        shape_buffer: BufferHandle,
        shape_offset: u64,
        label: Option<&'static str>,
    ) -> Result<CudaOp> {
        let pipeline = self
            .compute_pipelines
            .get(&pipeline_handle)
            .context("CUDA: invalid compute pipeline")?;
        let shape_buf = self
            .buffers
            .get(&shape_buffer)
            .context("CUDA: invalid indirect shape buffer")?;
        if shape_offset
            .checked_add(12)
            .filter(|end| *end <= shape_buf.size)
            .is_none()
        {
            anyhow::bail!(
                "CUDA: indirect shape range [{shape_offset}, {}) exceeds buffer size {}",
                shape_offset + 12,
                shape_buf.size
            );
        }
        let specs = self.resolve_storage_texture_specs(&pipeline.launch_layout, indices)?;
        let (module, max_threads_per_block) = self.ensure_compute_kernel(pipeline_handle, &specs)?;
        let graph_capture_ok = specs
            .iter()
            .all(|spec| matches!(spec, CudaStorageTextureSpec::Identity));
        let launch_args = self.build_launch_args(stream, &pipeline.launch_layout, indices, user)?;
        let (mut keep_alive_buffers, keep_alive_textures) = self.collect_launch_pins(indices)?;
        keep_alive_buffers.push(Arc::clone(shape_buf.memory_arc()?));

        let _gate = pending_submit::lock_capture_alloc_gate();
        let shape_abs_offset = shape_buf.offset + shape_offset;
        let shape_ptr = {
            let memory = shape_buf.memory_arc()?.lock().unwrap();
            let (base, _sync) = memory.device_ptr(stream);
            base + shape_abs_offset
        };

        let device = self.device(pipeline.device)?;
        let node_slot = Arc::new(Mutex::new(
            device
                .alloc_stream
                .alloc_zeros::<u64>(1)
                .context("CUDA: alloc device-updatable node slot")?,
        ));
        let status_memory = Arc::new(Mutex::new(
            device
                .alloc_stream
                .alloc_zeros::<i32>(1)
                .context("CUDA: alloc indirect status word")?,
        ));
        let node_slot_ptr = {
            let slot = node_slot.lock().unwrap();
            let (ptr, _sync) = slot.device_ptr(stream);
            ptr
        };
        let status_ptr = {
            let status = status_memory.lock().unwrap();
            let (ptr, _sync) = status.device_ptr(stream);
            ptr
        };
        drop(_gate);

        let function = module
            .load_function("cs_main")
            .context("CUDA: cuModuleGetFunction(cs_main) failed")?;
        let updater = device.indirect_updater.function.clone();
        let updater_module = Arc::clone(&device.indirect_updater.module);
        let max_grid = (
            device.limits.max_grid_dim_x,
            device.limits.max_grid_dim_y,
            device.limits.max_grid_dim_z,
        );
        let limits = device.limits;
        let workgroup_size = pipeline.workgroup_size;

        Ok(CudaOp::LaunchIndirect {
            label,
            function,
            module,
            workgroup_size,
            args: launch_args,
            keep_alive_buffers,
            keep_alive_textures,
            graph_capture_ok,
            shape_ptr,
            shape_memory: Arc::clone(shape_buf.memory_arc()?),
            shape_abs_offset,
            node_slot_ptr,
            node_slot,
            status_ptr,
            status_memory,
            updater,
            updater_module,
            max_grid,
            max_threads_per_block,
            limits,
        })
    }

    /// Pin buffers/textures referenced by registry keys for the lifetime of a launch/graph.
    fn collect_launch_pins(
        &self,
        indices: &[u32],
    ) -> Result<(Vec<Arc<Mutex<CudaSlice<u8>>>>, Vec<Arc<CudaTextureResource>>)> {
        let mut buffers = Vec::new();
        let mut textures = Vec::new();
        for (binding, index) in indices.iter().copied().enumerate() {
            if let Some(handle) = self.buffer_slots.get(&index) {
                let buffer = self
                    .buffers
                    .get(handle)
                    .with_context(|| format!("CUDA: registry key {index} references a destroyed buffer"))?;
                buffers.push(Arc::clone(buffer.memory_arc()?));
            } else if let Some(handle) = self.texture_slots.get(&index) {
                let texture = self
                    .textures
                    .get(handle)
                    .with_context(|| format!("CUDA: registry key {index} references a destroyed texture"))?;
                textures.push(Arc::clone(texture));
            } else if self.sampler_slots.contains_key(&index) {
                // Samplers are CPU-side descriptors; nothing to pin.
            } else {
                anyhow::bail!("CUDA: binding {binding} references unknown registry key {index}");
            }
        }
        Ok((buffers, textures))
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn ensure_requirements_for_commands(&mut self, commands: &[GpuCommand]) -> Result<()> {
        use buffer_phys::CudaBufferReq;
        // Collect (handle, req) then apply — avoids borrow issues while scanning.
        let mut updates: Vec<(BufferHandle, CudaBufferReq)> = Vec::new();
        let mut current_indices: Vec<u32> = Vec::new();
        let mut frame_table: Option<&[u32]> = None;
        for command in commands {
            match command {
                GpuCommand::FrameTableStaging { data } => {
                    frame_table = Some(data.as_ref());
                }
                GpuCommand::BindResourcesRaw { indices, .. } => {
                    current_indices.clone_from(indices);
                }
                GpuCommand::Dispatch { .. } => {
                    for index in &current_indices {
                        if let Some(handle) = self.buffer_slots.get(index) {
                            updates.push((*handle, CudaBufferReq::KERNEL));
                        }
                    }
                }
                GpuCommand::DispatchBatch { arg_data, count, .. } => {
                    // DispatchBatch does not emit BindResourcesRaw; indices live in FrameTableStaging.
                    let entry_count = *count as usize;
                    if entry_count == 0 {
                        continue;
                    }
                    let needed = entry_count
                        .checked_mul(DISPATCH_BATCH_STRIDE)
                        .context("CUDA: DispatchBatch stride overflow")?;
                    anyhow::ensure!(
                        arg_data.len() >= needed,
                        "CUDA: DispatchBatch arg_data len {} < {} entries × stride {}",
                        arg_data.len(),
                        entry_count,
                        DISPATCH_BATCH_STRIDE
                    );
                    let mut bases = Vec::with_capacity(entry_count);
                    for i in 0..entry_count {
                        let base = i * DISPATCH_BATCH_STRIDE;
                        let layout: PushLayout = *bytemuck::from_bytes(&arg_data[base..base + TOTAL_PUSH_BYTES]);
                        bases.push(layout._reserved[dispatch_table_base_word_index()]);
                    }
                    // Infer buffer count from uniform frame-table spacing (same as materialize).
                    let n_buffers = if entry_count >= 2 {
                        bases[1].saturating_sub(bases[0]) as usize
                    } else if let Some(table) = frame_table {
                        // Single-entry batch should not occur, but fall back to remaining row words.
                        table.len().saturating_sub(bases[0] as usize)
                    } else {
                        0
                    };
                    if n_buffers > 0 {
                        let table = frame_table
                            .context("CUDA: DispatchBatch requires FrameTableStaging when bindings are present")?;
                        for &table_base in &bases {
                            let start = table_base as usize;
                            let end = start
                                .checked_add(n_buffers)
                                .context("CUDA: frame-table range overflow")?;
                            anyhow::ensure!(
                                end <= table.len(),
                                "CUDA: DispatchBatch frame-table range [{start}, {end}) exceeds staging len {}",
                                table.len()
                            );
                            for &index in &table[start..end] {
                                if let Some(handle) = self.buffer_slots.get(&index) {
                                    updates.push((*handle, CudaBufferReq::KERNEL));
                                }
                            }
                        }
                    }
                    // Also honor any lingering BindResourcesRaw indices.
                    for index in &current_indices {
                        if let Some(handle) = self.buffer_slots.get(index) {
                            updates.push((*handle, CudaBufferReq::KERNEL));
                        }
                    }
                }
                GpuCommand::DispatchIndirect { buffer, .. } => {
                    updates.push((*buffer, CudaBufferReq::KERNEL | CudaBufferReq::TRANSFER));
                    for index in &current_indices {
                        if let Some(handle) = self.buffer_slots.get(index) {
                            updates.push((*handle, CudaBufferReq::KERNEL));
                        }
                    }
                }
                GpuCommand::ClearBuffer { buffer, .. } => {
                    updates.push((*buffer, CudaBufferReq::TRANSFER | CudaBufferReq::HOST_WRITE));
                }
                GpuCommand::WriteBuffer { buffer, .. } => {
                    updates.push((*buffer, CudaBufferReq::HOST_WRITE));
                }
                GpuCommand::CopyBuffer { src, dst, .. } => {
                    // CPU_WRITABLE deposit staging stays host-only; Copy materializes as HtoD
                    // into dst — do not force TRANSFER materialization of the staging parcel.
                    let src_host = self.buffers.get(src).is_some_and(|b| b.has_host_staging());
                    if !src_host {
                        updates.push((*src, CudaBufferReq::TRANSFER));
                    }
                    updates.push((*dst, CudaBufferReq::TRANSFER | CudaBufferReq::HOST_WRITE));
                }
                GpuCommand::CopyBufferToTexture { src, .. } => {
                    let src_host = self.buffers.get(src).is_some_and(|b| b.has_host_staging());
                    if !src_host {
                        updates.push((*src, CudaBufferReq::TRANSFER));
                    }
                }
                _ => {}
            }
        }
        for (handle, req) in updates {
            self.ensure_buffer_requirements(handle, req)?;
        }
        Ok(())
    }

    /// Buffers whose `memory` is the imported Shared primary (CUDA writes are DX12-visible).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn shared_primary_memories_written_by_ops(&self, ops: &[CudaOp]) -> Vec<Arc<dx12_interop::SharedBufferBacking>> {
        let mut out = Vec::new();
        for op in ops {
            let written = match op {
                CudaOp::Clear { memory, .. } | CudaOp::Write { memory, .. } | CudaOp::WriteFromHost { memory, .. } => {
                    Some(memory)
                }
                CudaOp::Copy { dst, .. } | CudaOp::CopyTextureToBuffer { dst, .. } => Some(dst),
                _ => None,
            };
            let Some(written) = written else { continue };
            for buf in self.buffers.values() {
                if buf.phys_kind != buffer_phys::CudaPhysKind::Shared {
                    continue;
                }
                let Some(mem) = buf.memory.as_ref() else { continue };
                if Arc::ptr_eq(mem, written) {
                    if let Some(shared) = buf.shared.as_ref() {
                        if !out.iter().any(|s| Arc::ptr_eq(s, shared)) {
                            out.push(Arc::clone(shared));
                        }
                    }
                }
            }
        }
        out
    }

    /// Retained Ops/Graph replays mutate buffer bytes without rematerializing; bump epochs
    /// so raster fingerprints and NativeAndTwin DtoD refresh see dirty content.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn bump_content_epochs_for_handles(&mut self, handles: &[BufferHandle]) {
        for &handle in handles {
            if let Some(buf) = self.buffers.get_mut(&handle) {
                buf.bump_content_epoch();
            }
        }
    }

    /// Retained Ops/Graph replays mutate buffer bytes without rematerializing; bump epochs
    /// so raster fingerprints and NativeAndTwin DtoD refresh see dirty content.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn bump_content_epochs_for_retained_writes(&mut self, ops: &[CudaOp]) {
        let handles = self.buffer_handles_written_by_ops(ops);
        self.bump_content_epochs_for_handles(&handles);
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn bump_content_epochs_for_native_twin_writes(&mut self, ops: &[CudaOp]) {
        self.bump_content_epochs_for_retained_writes(ops);
    }

    /// All buffers whose device memory is written by `ops` (for retained relaunch dirty lists).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn buffer_handles_written_by_ops(&self, ops: &[CudaOp]) -> Vec<BufferHandle> {
        let mut memories = Vec::new();
        for op in ops {
            match op {
                CudaOp::Clear { memory, .. } | CudaOp::Write { memory, .. } | CudaOp::WriteFromHost { memory, .. } => {
                    memories.push(Arc::clone(memory));
                }
                CudaOp::Copy { dst, .. } | CudaOp::CopyTextureToBuffer { dst, .. } => {
                    memories.push(Arc::clone(dst));
                }
                CudaOp::Launch { keep_alive_buffers, .. } | CudaOp::LaunchIndirect { keep_alive_buffers, .. } => {
                    memories.extend(keep_alive_buffers.iter().cloned());
                }
                _ => {}
            }
        }
        if memories.is_empty() {
            return Vec::new();
        }
        let written: std::collections::HashSet<*const Mutex<CudaSlice<u8>>> =
            memories.iter().map(Arc::as_ptr).collect();
        self.buffers
            .iter()
            .filter_map(|(handle, buf)| {
                let mem = buf.memory.as_ref()?;
                written.contains(&Arc::as_ptr(mem)).then_some(*handle)
            })
            .collect()
    }

    /// NativeAndTwin buffers whose memory is written by `ops` (for retained graph dirty lists).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn native_twin_buffers_written_by_ops(&self, ops: &[CudaOp]) -> Vec<BufferHandle> {
        let mut memories = Vec::new();
        for op in ops {
            match op {
                CudaOp::Clear { memory, .. } | CudaOp::Write { memory, .. } | CudaOp::WriteFromHost { memory, .. } => {
                    memories.push(Arc::clone(memory));
                }
                CudaOp::Copy { dst, .. } | CudaOp::CopyTextureToBuffer { dst, .. } => {
                    memories.push(Arc::clone(dst));
                }
                CudaOp::Launch { keep_alive_buffers, .. } | CudaOp::LaunchIndirect { keep_alive_buffers, .. } => {
                    memories.extend(keep_alive_buffers.iter().cloned())
                }
                _ => {}
            }
        }
        if memories.is_empty() {
            return Vec::new();
        }
        let written: std::collections::HashSet<*const Mutex<CudaSlice<u8>>> =
            memories.iter().map(Arc::as_ptr).collect();
        self.buffers
            .iter()
            .filter_map(|(handle, buf)| {
                if buf.phys_kind != buffer_phys::CudaPhysKind::NativeAndTwin {
                    return None;
                }
                let mem = buf.memory.as_ref()?;
                written.contains(&Arc::as_ptr(mem)).then_some(*handle)
            })
            .collect()
    }

    fn materialize_ops(&mut self, stream: &Arc<CudaStream>, commands: &[GpuCommand]) -> Result<Vec<CudaOp>> {
        let _gate = capture_gate::lock_capture_alloc_gate();
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        {
            // Late-physicalize from this submit's usage before building CudaOps.
            self.ensure_requirements_for_commands(commands)?;
        }
        let mut ops = Vec::new();
        let mut current_pipeline: Option<ComputePipelineHandle> = None;
        let mut current_indices: Vec<u32> = Vec::new();
        let mut current_user: Vec<u32> = Vec::new();
        let mut frame_table: Option<Arc<[u32]>> = None;

        for command in commands {
            match command {
                GpuCommand::SetPipeline(pipeline) => current_pipeline = Some(*pipeline),
                GpuCommand::BindResourcesRaw { indices, user, .. } => {
                    if user.len() > MAX_USER_SLOTS {
                        anyhow::bail!(
                            "CUDA: at most {MAX_USER_SLOTS} scalar user params per dispatch, got {}",
                            user.len()
                        );
                    }
                    current_indices.clone_from(indices);
                    current_user.clone_from(user);
                }
                GpuCommand::Dispatch {
                    label,
                    workgroups_x,
                    workgroups_y,
                    workgroups_z,
                } => {
                    let pipeline_handle = current_pipeline.context("CUDA: dispatch without a compute pipeline")?;
                    ops.push(self.materialize_launch(
                        stream,
                        pipeline_handle,
                        &current_indices,
                        &current_user,
                        (*workgroups_x, *workgroups_y, *workgroups_z),
                        *label,
                    )?);
                }
                GpuCommand::DispatchIndirect { label, buffer, offset } => {
                    let pipeline_handle =
                        current_pipeline.context("CUDA: indirect dispatch without a compute pipeline")?;
                    ops.push(self.materialize_launch_indirect(
                        stream,
                        pipeline_handle,
                        &current_indices,
                        &current_user,
                        *buffer,
                        *offset,
                        *label,
                    )?);
                }
                GpuCommand::ClearBuffer { buffer, offset, size } => {
                    let buffer = self.buffers.get_mut(buffer).context("CUDA: invalid clear buffer")?;
                    let clear_size = if *size == 0 {
                        buffer.size.saturating_sub(*offset)
                    } else {
                        *size
                    };
                    // Invalidate shared DX12 VB twins used by raster.
                    buffer.bump_content_epoch();
                    let memory = Arc::clone(buffer.memory_arc()?);
                    let abs_offset = buffer.offset + *offset;
                    let device_ptr = pending_submit::bake_device_ptr(stream, &memory, abs_offset);
                    ops.push(CudaOp::Clear {
                        memory,
                        abs_offset,
                        size: clear_size,
                        device_ptr,
                    });
                }
                GpuCommand::WriteBuffer { buffer, offset, data } => {
                    let buffer = self.buffers.get_mut(buffer).context("CUDA: invalid write buffer")?;
                    if *offset + data.len() as u64 > buffer.size {
                        anyhow::bail!("CUDA: write exceeds logical buffer size");
                    }
                    // Invalidate shared DX12 VB twins used by raster (deposit / staging path).
                    buffer.bump_content_epoch();
                    ops.push(CudaOp::Write {
                        memory: Arc::clone(buffer.memory_arc()?),
                        abs_offset: buffer.offset + *offset,
                        data: data.to_vec(),
                    });
                }
                GpuCommand::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let src_buf = self.buffers.get(src).context("CUDA: invalid copy source")?;
                    if src_buf.device != self.buffers.get(dst).context("CUDA: invalid copy destination")?.device {
                        anyhow::bail!("CUDA: copy across devices is not supported");
                    }
                    if *src_offset + *size > src_buf.size {
                        anyhow::bail!("CUDA: copy source range exceeds logical buffer size");
                    }
                    if let Some(staging) = src_buf.host_staging.as_ref() {
                        let start = (src_buf.offset + *src_offset) as usize;
                        let host = Arc::clone(staging);
                        let dst_buf = self.buffers.get_mut(dst).context("CUDA: invalid copy destination")?;
                        if *dst_offset + *size > dst_buf.size {
                            anyhow::bail!("CUDA: copy destination range exceeds logical buffer size");
                        }
                        dst_buf.bump_content_epoch();
                        let memory = Arc::clone(dst_buf.memory_arc()?);
                        let abs_offset = dst_buf.offset + *dst_offset;
                        let device_ptr = pending_submit::bake_device_ptr(stream, &memory, abs_offset);
                        ops.push(CudaOp::WriteFromHost {
                            memory,
                            abs_offset,
                            device_ptr,
                            host,
                            host_offset: start,
                            len: *size as usize,
                        });
                    } else {
                        let src_memory = Arc::clone(src_buf.memory_arc()?);
                        let src_abs = src_buf.offset + *src_offset;
                        let dst_buf = self.buffers.get_mut(dst).context("CUDA: invalid copy destination")?;
                        if *dst_offset + *size > dst_buf.size {
                            anyhow::bail!("CUDA: copy destination range exceeds logical buffer size");
                        }
                        dst_buf.bump_content_epoch();
                        let dst_memory = Arc::clone(dst_buf.memory_arc()?);
                        let dst_abs = dst_buf.offset + *dst_offset;
                        let src_ptr = pending_submit::bake_device_ptr(stream, &src_memory, src_abs);
                        let dst_ptr = pending_submit::bake_device_ptr(stream, &dst_memory, dst_abs);
                        ops.push(CudaOp::Copy {
                            src: src_memory,
                            src_abs,
                            src_ptr,
                            dst: dst_memory,
                            dst_abs,
                            dst_ptr,
                            size: *size,
                        });
                    }
                }
                GpuCommand::FrameTableStaging { data } => {
                    frame_table = Some(Arc::clone(data));
                }
                GpuCommand::ResourceBarrier { .. } => {}
                GpuCommand::BuildAccelerationStructure(_) => {
                    anyhow::bail!("CUDA backend does not support acceleration structures");
                }
                GpuCommand::DispatchBatch { label, arg_data, count } => {
                    let pipeline_handle = current_pipeline.context("CUDA: DispatchBatch without a compute pipeline")?;
                    let batch_ops = self.materialize_dispatch_batch(
                        stream,
                        pipeline_handle,
                        frame_table.as_deref(),
                        arg_data.as_ref(),
                        *count,
                        *label,
                    )?;
                    ops.extend(batch_ops);
                }
                GpuCommand::WriteTexture {
                    texture,
                    data,
                    width,
                    height,
                } => {
                    let tex = self
                        .textures
                        .get(texture)
                        .context("CUDA: invalid WriteTexture handle")?;
                    if *width != tex.width || *height != tex.height {
                        anyhow::bail!(
                            "CUDA: WriteTexture size {}x{} does not match texture {}x{}",
                            width,
                            height,
                            tex.width,
                            tex.height
                        );
                    }
                    ops.push(CudaOp::WriteTexture {
                        texture: Arc::clone(tex),
                        x: 0,
                        y: 0,
                        width: *width,
                        height: *height,
                        data: data.to_vec(),
                        src_row_pitch: 0,
                    });
                }
                GpuCommand::WriteTextureRegion {
                    texture,
                    x,
                    y,
                    width,
                    height,
                    data,
                } => {
                    let tex = self
                        .textures
                        .get(texture)
                        .context("CUDA: invalid WriteTextureRegion handle")?;
                    ops.push(CudaOp::WriteTexture {
                        texture: Arc::clone(tex),
                        x: *x,
                        y: *y,
                        width: *width,
                        height: *height,
                        data: data.to_vec(),
                        src_row_pitch: 0,
                    });
                }
                GpuCommand::CopyTexture { src, dst } => {
                    let src_tex = self.textures.get(src).context("CUDA: invalid CopyTexture source")?;
                    let dst_tex = self
                        .textures
                        .get(dst)
                        .context("CUDA: invalid CopyTexture destination")?;
                    ops.push(CudaOp::CopyTexture {
                        src: Arc::clone(src_tex),
                        dst: Arc::clone(dst_tex),
                        src_x: 0,
                        src_y: 0,
                        dst_x: 0,
                        dst_y: 0,
                        width: src_tex.width,
                        height: src_tex.height,
                    });
                }
                GpuCommand::CopyTextureRegion {
                    src,
                    dst,
                    src_x,
                    src_y,
                    dst_x,
                    dst_y,
                    width,
                    height,
                } => {
                    let src_tex = self
                        .textures
                        .get(src)
                        .context("CUDA: invalid CopyTextureRegion source")?;
                    let dst_tex = self
                        .textures
                        .get(dst)
                        .context("CUDA: invalid CopyTextureRegion destination")?;
                    ops.push(CudaOp::CopyTexture {
                        src: Arc::clone(src_tex),
                        dst: Arc::clone(dst_tex),
                        src_x: *src_x,
                        src_y: *src_y,
                        dst_x: *dst_x,
                        dst_y: *dst_y,
                        width: *width,
                        height: *height,
                    });
                }
                GpuCommand::CopyBufferToTexture {
                    src,
                    src_offset,
                    src_row_pitch,
                    dst,
                    x,
                    y,
                    width,
                    height,
                } => {
                    let src_buf = self
                        .buffers
                        .get(src)
                        .context("CUDA: invalid CopyBufferToTexture source")?;
                    let dst_tex = self
                        .textures
                        .get(dst)
                        .context("CUDA: invalid CopyBufferToTexture destination")?;
                    if let Some(staging) = src_buf.host_staging.as_ref() {
                        let bpp = dst_tex.format.bytes_per_pixel() as u32;
                        let row_pitch = if *src_row_pitch == 0 {
                            width.saturating_mul(bpp)
                        } else {
                            *src_row_pitch
                        };
                        let nbytes = (row_pitch as usize).saturating_mul(*height as usize);
                        let start = (src_buf.offset + *src_offset) as usize;
                        ops.push(CudaOp::WriteTextureFromHost {
                            texture: Arc::clone(dst_tex),
                            x: *x,
                            y: *y,
                            width: *width,
                            height: *height,
                            host: Arc::clone(staging),
                            host_offset: start,
                            len: nbytes,
                            src_row_pitch: *src_row_pitch,
                        });
                    } else {
                        let src_memory = Arc::clone(src_buf.memory_arc()?);
                        let src_abs = src_buf.offset + *src_offset;
                        let src_ptr = pending_submit::bake_device_ptr(stream, &src_memory, src_abs);
                        ops.push(CudaOp::CopyBufferToTexture {
                            src: src_memory,
                            src_abs,
                            src_ptr,
                            src_row_pitch: *src_row_pitch,
                            texture: Arc::clone(dst_tex),
                            x: *x,
                            y: *y,
                            width: *width,
                            height: *height,
                        });
                    }
                }
                GpuCommand::CopyTextureToReadback { src, dst, layout } => {
                    let src_tex = self
                        .textures
                        .get(src)
                        .context("CUDA: invalid CopyTextureToReadback source")?;
                    let dst_buf = self
                        .buffers
                        .get(dst)
                        .context("CUDA: invalid CopyTextureToReadback destination")?;
                    if layout.width != src_tex.width || layout.height != src_tex.height {
                        anyhow::bail!(
                            "CUDA: CopyTextureToReadback footprint {}x{} != texture {}x{}",
                            layout.width,
                            layout.height,
                            src_tex.width,
                            src_tex.height
                        );
                    }
                    let dst_memory = Arc::clone(dst_buf.memory_arc()?);
                    let dst_abs = dst_buf.offset + layout.footprint_offset;
                    let dst_ptr = pending_submit::bake_device_ptr(stream, &dst_memory, dst_abs);
                    ops.push(CudaOp::CopyTextureToBuffer {
                        texture: Arc::clone(src_tex),
                        x: 0,
                        y: 0,
                        width: layout.width,
                        height: layout.height,
                        dst: dst_memory,
                        dst_abs,
                        dst_ptr,
                        dst_row_pitch: layout.row_pitch,
                    });
                }
                GpuCommand::CopyRenderTarget { src, dst } => {
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    {
                        // Fast path: copy into surface present scratch → stash the DX12 RT
                        // so present can blit it directly (no CUDA array round-trip).
                        if let Some((surf, image_index)) = surface::scratch_slot_for_texture(self, *dst) {
                            let (d3d12_resource, fence, format) = {
                                let rt = self
                                    .render_targets
                                    .get(src)
                                    .context("CUDA/DX12: invalid CopyRenderTarget source")?;
                                let dst_tex = self
                                    .textures
                                    .get(dst)
                                    .context("CUDA/DX12: invalid CopyRenderTarget destination")?;
                                if rt.cuda_texture.format != dst_tex.format {
                                    anyhow::bail!(
                                        "CUDA/DX12: CopyRenderTarget format mismatch ({:?} → {:?})",
                                        rt.cuda_texture.format,
                                        dst_tex.format
                                    );
                                }
                                if rt.width != dst_tex.width || rt.height != dst_tex.height {
                                    anyhow::bail!(
                                        "CUDA/DX12: CopyRenderTarget size mismatch ({}x{} → {}x{})",
                                        rt.width,
                                        rt.height,
                                        dst_tex.width,
                                        dst_tex.height
                                    );
                                }
                                (rt.d3d12_resource.clone(), rt.last_dx12_fence, rt.format)
                            };
                            if let Some(slot) = self
                                .surfaces
                                .get_mut(&surf)
                                .and_then(|s| s.scratch.get_mut(image_index))
                                .and_then(|s| s.as_mut())
                            {
                                slot.present_source = Some(surface::PresentSource::Dx12Raster {
                                    resource: d3d12_resource,
                                    fence,
                                    format,
                                });
                            }
                        } else {
                            let (cuda_src, fence, device) = {
                                let rt = self
                                    .render_targets
                                    .get(src)
                                    .context("CUDA/DX12: invalid CopyRenderTarget source")?;
                                (Arc::clone(&rt.cuda_texture), rt.last_dx12_fence, rt.device)
                            };
                            let cuda_dst = Arc::clone(
                                self.textures
                                    .get(dst)
                                    .context("CUDA/DX12: invalid CopyRenderTarget destination")?,
                            );
                            if cuda_src.format != cuda_dst.format {
                                anyhow::bail!(
                                    "CUDA/DX12: CopyRenderTarget format mismatch ({:?} → {:?})",
                                    cuda_src.format,
                                    cuda_dst.format
                                );
                            }
                            if cuda_src.width != cuda_dst.width || cuda_src.height != cuda_dst.height {
                                anyhow::bail!(
                                    "CUDA/DX12: CopyRenderTarget size mismatch ({}x{} → {}x{})",
                                    cuda_src.width,
                                    cuda_src.height,
                                    cuda_dst.width,
                                    cuda_dst.height
                                );
                            }
                            if fence > 0 {
                                let companion = self
                                    .devices
                                    .get(&device)
                                    .context("CUDA: invalid device")?
                                    .dx12
                                    .as_ref()
                                    .context("CUDA/DX12: companion required for CopyRenderTarget")?;
                                ops.push(CudaOp::WaitExternalFence {
                                    cuda_ctx: Arc::clone(&companion.cuda_ctx),
                                    semaphore: pending_submit::SendExternalSemaphore(companion.cuda_semaphore),
                                    value: fence,
                                });
                            }
                            ops.push(CudaOp::CopyTexture {
                                src: Arc::clone(&cuda_src),
                                dst: cuda_dst,
                                src_x: 0,
                                src_y: 0,
                                dst_x: 0,
                                dst_y: 0,
                                width: cuda_src.width,
                                height: cuda_src.height,
                            });
                        }
                    }
                    #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
                    {
                        let _ = (src, dst);
                        anyhow::bail!("CUDA: CopyRenderTarget requires cuda+graphics+dx12 on Windows");
                    }
                }
                GpuCommand::BuildAccelerationStructure(_) => {
                    anyhow::bail!("CUDA backend does not support acceleration structures");
                }
            }
        }
        Ok(ops)
    }

    fn submit_commands(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        let _tz = crate::tracy_zone!("cuda.submit");
        let effective = {
            let _tz = crate::tracy_zone!("cuda.submit.prologue");
            commands_with_sync_prologue(commands, sync)
        };
        let stream = Arc::clone(&self.context(ctx)?.stream);
        let ops = {
            let _tz = crate::tracy_zone!("cuda.submit.materialize");
            self.materialize_ops(&stream, &effective)?
        };
        let _tz = crate::tracy_zone!("cuda.submit.enqueue");
        self.enqueue_submit(
            ctx,
            sync,
            CudaSubmitBody::Ops {
                ops,
                bump_content_epochs: true,
            },
        )
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn first_imported_scratch_write_index(&self, ops: &[CudaOp], touched: &[(SurfaceHandle, usize)]) -> Option<usize> {
        let scratches: Vec<_> = touched
            .iter()
            .filter_map(|(surface, image)| {
                self.surfaces
                    .get(surface)
                    .and_then(|state| state.scratch.get(*image))
                    .and_then(|slot| slot.as_ref())
                    .map(|slot| Arc::clone(&slot.shared.cuda_texture))
            })
            .collect();
        ops.iter()
            .position(|op| scratches.iter().any(|scratch| op_writes_imported_scratch(op, scratch)))
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn scratch_images_touched_by_ops(&self, ops: &[CudaOp]) -> Vec<(SurfaceHandle, usize)> {
        let mut touched = Vec::new();
        for (surface_handle, state) in &self.surfaces {
            for (image_index, slot) in state.scratch.iter().enumerate() {
                let Some(slot) = slot.as_ref() else {
                    continue;
                };
                let scratch = &slot.shared.cuda_texture;
                let writes_scratch = ops.iter().any(|op| op_writes_imported_scratch(op, scratch));
                if writes_scratch {
                    touched.push((*surface_handle, image_index));
                }
            }
        }
        touched
    }

    /// CUDA graphs cannot capture launches that touch D3D12-imported external memory.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn ops_touch_external_buffers(&self, ops: &[CudaOp]) -> bool {
        !self.shared_buffers_touched_by_ops(ops).is_empty()
    }

    #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
    fn ops_touch_external_buffers(&self, _ops: &[CudaOp]) -> bool {
        false
    }

    /// Collect buffer handles whose `memory` Arc is referenced by `ops` as a write
    /// destination (or conservatively as a launch keep-alive).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn shared_buffers_touched_by_ops(&self, ops: &[CudaOp]) -> Vec<BufferHandle> {
        let mut memories = Vec::new();
        for op in ops {
            match op {
                CudaOp::Clear { memory, .. } | CudaOp::Write { memory, .. } | CudaOp::WriteFromHost { memory, .. } => {
                    memories.push(Arc::clone(memory));
                }
                CudaOp::Copy { dst, .. } => memories.push(Arc::clone(dst)),
                CudaOp::Launch { keep_alive_buffers, .. } | CudaOp::LaunchIndirect { keep_alive_buffers, .. } => {
                    memories.extend(keep_alive_buffers.iter().cloned());
                }
                CudaOp::CopyTextureToBuffer { dst, .. } => memories.push(Arc::clone(dst)),
                _ => {}
            }
        }
        if memories.is_empty() {
            return Vec::new();
        }
        self.buffers
            .iter()
            .filter_map(|(handle, buf)| {
                // Only Shared-primary memory is an external import. NativeAndTwin's
                // `shared` twin is separate from `memory` and is graph-safe to write.
                if !buf.memory_is_external {
                    return None;
                }
                let mem = buf.memory.as_ref()?;
                memories.iter().any(|m| Arc::ptr_eq(m, mem)).then_some(*handle)
            })
            .collect()
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn prepare_surface_submit_ops_for_retained(
        &mut self,
        ctx: ContextHandle,
        scratch_images: &[(SurfaceHandle, usize)],
        ops: &mut Vec<CudaOp>,
        bump_content_epochs: bool,
    ) -> Result<()> {
        let touched = if scratch_images.is_empty() {
            self.scratch_images_touched_by_ops(ops)
        } else {
            scratch_images.to_vec()
        };
        // Signal the companion fence when CUDA wrote surface scratch (present).
        // Deposit-only Shared-primary writes (e.g. solid_cube upload scheme) do not
        // publish a CUDA→DX12 fence — raster syncs alloc_stream before IA instead
        // (see SharedBufferBacking::pending_host_sync in dx12_interop.rs).
        if bump_content_epochs {
            self.bump_content_epochs_for_native_twin_writes(ops);
        }
        let shared_primaries = self.shared_primary_memories_written_by_ops(ops);

        let device_handle = self.context(ctx)?.device;
        if touched.is_empty() && shared_primaries.is_empty() {
            return Ok(());
        }

        let companion = Arc::clone(
            self.device(device_handle)?
                .dx12
                .as_ref()
                .context("CUDA/DX12: interop submit missing companion")?,
        );
        let worker = Arc::clone(&self.device(device_handle)?.submission_worker);
        let mut waits = Vec::new();
        for (surface_handle, image_index) in &touched {
            let slot = self
                .surfaces
                .get_mut(surface_handle)
                .and_then(|state| state.scratch.get_mut(*image_index))
                .and_then(|slot| slot.as_mut())
                .context("CUDA/DX12: scratch slot disappeared during submit")?;
            let reuse_fence = slot.pending_scratch_reuse_fence.max(slot.dx12_complete);
            if reuse_fence > 0 {
                waits.push(CudaOp::WaitExternalFence {
                    cuda_ctx: Arc::clone(&companion.cuda_ctx),
                    semaphore: pending_submit::SendExternalSemaphore(companion.recycle_semaphore),
                    value: reuse_fence,
                });
                slot.pending_scratch_reuse_fence = 0;
                slot.dx12_complete = 0;
            }
        }
        // Shared-primary rewrites must not race an in-flight DX12 IA draw.
        let mut ia_wait = 0u64;
        for shared in &shared_primaries {
            ia_wait = ia_wait.max(shared.last_dx12_ia_fence.load(Ordering::Acquire));
        }
        if ia_wait > 0 {
            worker.flush().context("CUDA/DX12: flush worker before IA cpu_wait")?;
            companion
                .cpu_wait(ia_wait)
                .context("CUDA/DX12: wait IA fence before Shared buffer rewrite")?;
        }
        if !waits.is_empty() {
            // Wait immediately before the first imported-scratch write (typically
            // CopyTexture export). Prepending to the whole last stream segment would
            // stall `cs_main` on the previous present's DX12 fence.
            let insert_at = self.first_imported_scratch_write_index(ops, &touched).unwrap_or(0);
            let mut suffix = ops.split_off(insert_at);
            let mut prefix = std::mem::take(ops);
            prefix.append(&mut waits);
            prefix.append(&mut suffix);
            *ops = prefix;
        }

        if touched.is_empty() {
            // Deposit-only: copy ops stay in `ops`; raster prologue syncs alloc_stream.
            return Ok(());
        }

        // Scratch present needs a CUDA→DX12 fence Signal so the DIRECT queue can Wait
        // before reading imported scratch. Publishing `last_cuda_fence` before the
        // worker runs is safe as long as DX12 always `wait_queue`s that value before
        // Signal'ing past it (see refresh_shared_vertex_backing).
        let cuda_complete = companion.next_fence_value();
        ops.push(CudaOp::SignalExternalFence {
            cuda_ctx: Arc::clone(&companion.cuda_ctx),
            semaphore: pending_submit::SendExternalSemaphore(companion.cuda_semaphore),
            value: cuda_complete,
        });
        for (surface_handle, image_index) in touched {
            if let Some(slot) = self
                .surfaces
                .get_mut(&surface_handle)
                .and_then(|state| state.scratch.get_mut(image_index))
                .and_then(|slot| slot.as_mut())
            {
                slot.present_source = Some(surface::PresentSource::CudaScratch { cuda_complete });
            }
        }
        for shared in shared_primaries {
            shared.last_cuda_fence.store(cuda_complete, Ordering::Release);
            shared.pending_host_sync.store(false, Ordering::Release);
        }
        Ok(())
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn prepare_surface_submit_ops(&mut self, ctx: ContextHandle, ops: &mut Vec<CudaOp>) -> Result<()> {
        self.prepare_surface_submit_ops_for_retained(ctx, &[], ops, true)
    }

    #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
    fn prepare_surface_submit_ops(&mut self, _ctx: ContextHandle, _ops: &mut Vec<CudaOp>) -> Result<()> {
        Ok(())
    }

    fn flatten_graph_commands(commands: &[GraphCommand]) -> Result<Vec<GpuCommand>> {
        let mut out = Vec::with_capacity(commands.len());
        for cmd in commands {
            match cmd {
                GraphCommand::Compute(c) => out.push(c.clone()),
                GraphCommand::Render { .. } => {
                    anyhow::bail!(
                        "CUDA: GraphCommand::Render cannot be flattened into compute ops; \
                         use submit_graph (blocking render_to_target between batches)"
                    )
                }
            }
        }
        Ok(out)
    }

    /// True when the graph contains an offscreen render partition.
    fn graph_has_render(commands: &[GraphCommand]) -> bool {
        commands.iter().any(|c| matches!(c, GraphCommand::Render { .. }))
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn retarget_surface_scratch_commands(&self, commands: &mut [GraphCommand]) {
        let mut replacements = HashMap::new();
        for state in self.surfaces.values() {
            let Some(current) = state.current_texture_handle else {
                continue;
            };
            for slot in state.scratch.iter().flatten() {
                replacements.insert(slot.texture_handle, current);
            }
        }
        for command in commands {
            let GraphCommand::Compute(GpuCommand::CopyRenderTarget { dst, .. }) = command else {
                continue;
            };
            if let Some(current) = replacements.get(dst) {
                *dst = *current;
            }
        }
    }

    fn submit_graph_with_renders(
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
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    {
                        // Pre-declare VERTEX for deposit-only VBs/IBs so Copy lands Shared.
                        // If the batch has a dispatch, skip — KERNEL+VERTEX must materialize
                        // Native (then twin) without a Shared→NativeAndTwin promote mid-submit.
                        let batch_has_dispatch = batch.iter().any(|c| {
                            matches!(
                                c,
                                GpuCommand::Dispatch { .. }
                                    | GpuCommand::DispatchIndirect { .. }
                                    | GpuCommand::DispatchBatch { .. }
                            )
                        });
                        if !batch_has_dispatch {
                            for render in render_cmds.iter() {
                                match render {
                                    RenderCommand::SetVertexBuffer { buffer, .. }
                                    | RenderCommand::SetIndexBuffer { buffer, .. } => {
                                        self.ensure_buffer_requirements(*buffer, buffer_phys::CudaBufferReq::VERTEX)?;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        if !batch.is_empty() {
                            let written_ia_buffer = render_cmds.iter().any(|render| {
                                let buffer = match render {
                                    RenderCommand::SetVertexBuffer { buffer, .. }
                                    | RenderCommand::SetIndexBuffer { buffer, .. } => buffer,
                                    _ => return false,
                                };
                                batch.iter().any(|compute| match compute {
                                    GpuCommand::ClearBuffer { buffer: dst, .. }
                                    | GpuCommand::WriteBuffer { buffer: dst, .. } => dst == buffer,
                                    GpuCommand::CopyBuffer { dst, .. } => dst == buffer,
                                    // Conservatively sync when any dispatch precedes an IA draw —
                                    // the dispatch may have written VB/IB through bindless indices.
                                    GpuCommand::Dispatch { .. } | GpuCommand::DispatchIndirect { .. } => true,
                                    _ => false,
                                })
                            });
                            // Capture before clear: Scheme emits FrameTableStaging in this batch.
                            let graph_staging = batch.iter().find_map(|c| match c {
                                GpuCommand::FrameTableStaging { data } => Some(Arc::clone(data)),
                                _ => None,
                            });
                            last_tv = self.submit_commands(ctx, &batch, sync)?;
                            if written_ia_buffer {
                                // Compute may have rewritten native IA storage without a
                                // WriteBuffer op; invalidate shared twins so raster DtoDs.
                                for render in render_cmds.iter() {
                                    let buffer = match render {
                                        RenderCommand::SetVertexBuffer { buffer, .. }
                                        | RenderCommand::SetIndexBuffer { buffer, .. } => buffer,
                                        _ => continue,
                                    };
                                    if let Some(buf) = self.buffers.get_mut(buffer) {
                                        if buf.phys_kind == buffer_phys::CudaPhysKind::NativeAndTwin {
                                            buf.bump_content_epoch();
                                        }
                                    }
                                }
                                let needs_twin_sync = render_cmds.iter().any(|render| {
                                    let buffer = match render {
                                        RenderCommand::SetVertexBuffer { buffer, .. }
                                        | RenderCommand::SetIndexBuffer { buffer, .. } => buffer,
                                        _ => return false,
                                    };
                                    self.buffers.get(buffer).is_some_and(|buf| {
                                        buf.phys_kind == buffer_phys::CudaPhysKind::NativeAndTwin
                                            && buf.content_epoch != buf.shared_epoch
                                    })
                                });
                                if needs_twin_sync {
                                    let device = self.context_device(ctx);
                                    let worker = Arc::clone(&self.device(device)?.submission_worker);
                                    worker.flush().context("CUDA/DX12: flush before raster VB wait")?;
                                    self.graph_stats.worker_flushes.fetch_add(1, Ordering::Relaxed);
                                    self.context(ctx)?
                                        .stream
                                        .synchronize()
                                        .context("CUDA/DX12: synchronize compute before raster")?;
                                }
                            }
                            batch.clear();
                            let device = self.context_device(ctx);
                            raster::render_to_target(
                                self,
                                device,
                                *target,
                                *color_load,
                                render_cmds,
                                graph_staging.as_deref(),
                            )?;
                        } else {
                            let device = self.context_device(ctx);
                            raster::render_to_target(self, device, *target, *color_load, render_cmds, None)?;
                        }
                    }
                    #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
                    {
                        let _ = (target, color_load, render_cmds);
                        anyhow::bail!("CUDA: render graph commands require cuda+graphics+dx12 on Windows");
                    }
                }
            }
        }
        if !batch.is_empty() {
            last_tv = self.submit_commands(ctx, &batch, sync)?;
        }
        Ok(last_tv)
    }

    /// True when an empty Ops submit's sync only references DX12 fence ledger entries
    /// (or has no CUDA-side host work). Such submits need no CUDA completion event.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn empty_ops_sync_is_dx12_fence_only(&self, ctx: ContextHandle, sync: Option<&SubmitSync>) -> Result<bool> {
        let Some(sync) = sync else {
            return Ok(true);
        };
        if !sync.cpu_waits.is_empty() || !sync.host_observed_waits.is_empty() || !sync.deferred_host_writes.is_empty() {
            return Ok(false);
        }
        if sync.waits.is_empty() {
            return Ok(true);
        }
        let device_handle = self.context(ctx)?.device;
        let device = self.device(device_handle)?;
        let ledger = &device.event_ledger;
        let retired = &device.retired;
        for epoch in &sync.waits {
            match timeline::completion_for_wait(ledger, retired, epoch.context, epoch.value) {
                timeline::WaitCompletion::Pending(LedgerCompletion::Dx12Fence { .. }) => {}
                timeline::WaitCompletion::AlreadyComplete => {}
                timeline::WaitCompletion::Pending(LedgerCompletion::CudaEvent(_))
                | timeline::WaitCompletion::Missing => return Ok(false),
            }
        }
        Ok(true)
    }

    fn enqueue_submit(
        &mut self,
        ctx: ContextHandle,
        sync: Option<&SubmitSync>,
        mut body: CudaSubmitBody,
    ) -> Result<crate::timeline::TimelineValue> {
        let _tz = crate::tracy_zone!("cuda.enqueue_submit");
        {
            let _tz = crate::tracy_zone!("cuda.enqueue_submit.prepare_surface");
            match &mut body {
                CudaSubmitBody::Ops {
                    ops,
                    bump_content_epochs,
                } => {
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    self.prepare_surface_submit_ops_for_retained(ctx, &[], ops, *bump_content_epochs)?;
                    #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
                    {
                        let _ = bump_content_epochs;
                        self.prepare_surface_submit_ops(ctx, ops)?;
                    }
                }
                CudaSubmitBody::CaptureAndLaunch { segments, .. } => {
                    let stream_ops = pending_submit::last_stream_segment_mut(segments);
                    self.prepare_surface_submit_ops(ctx, stream_ops)?;
                }
                CudaSubmitBody::LaunchRetained {
                    segments,
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    scratch_images,
                    ..
                } => {
                    let stream_ops = pending_submit::last_launch_stream_segment_mut(segments);
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    self.prepare_surface_submit_ops_for_retained(ctx, scratch_images, stream_ops, false)?;
                    #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
                    self.prepare_surface_submit_ops(ctx, stream_ops)?;
                }
            }
        }
        let sync_needs_work = sync.is_some_and(|sync| {
            !sync.waits.is_empty()
                || !sync.cpu_waits.is_empty()
                || !sync.host_observed_waits.is_empty()
                || !sync.deferred_host_writes.is_empty()
        });
        if matches!(&body, CudaSubmitBody::Ops { ops, .. } if ops.is_empty()) && !sync_needs_work {
            self.graph_stats.empty_submits_elided.fetch_add(1, Ordering::Relaxed);
            return Ok(self.gpu_progress(ctx));
        }
        // Empty body + DX12-fence-only waits: nothing runs on the CUDA stream, so a
        // completion event / stream join is pure overhead (raster-direct present path).
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        if matches!(&body, CudaSubmitBody::Ops { ops, .. } if ops.is_empty())
            && self.empty_ops_sync_is_dx12_fence_only(ctx, sync)?
        {
            self.graph_stats.empty_submits_elided.fetch_add(1, Ordering::Relaxed);
            return Ok(self.gpu_progress(ctx));
        }

        let context = Arc::clone(self.context(ctx)?);
        let device_handle = context.device;
        let device = self.device(device_handle)?;
        let stream = Arc::clone(&context.stream);
        let worker = Arc::clone(&device.submission_worker);
        let next_timeline = Arc::clone(&device.next_timeline);
        let event_ledger = Arc::clone(&device.event_ledger);
        let event_pool = Arc::clone(&device.event_pool);

        worker.check_error()?;

        let (fence_value, completion_event) = {
            let _tz = crate::tracy_zone!("cuda.enqueue_submit.alloc_timeline");
            let fence_value = {
                let _tz = crate::tracy_zone!("cuda.enqueue_submit.alloc_timeline.counter");
                submission_worker::allocate_timeline_value(&next_timeline)
            };
            let completion_event = {
                let _tz = crate::tracy_zone!("cuda.enqueue_submit.alloc_timeline.create_event");
                event_pool.acquire()?
            };
            self.graph_stats.completion_events.fetch_add(1, Ordering::Relaxed);
            {
                let _tz = crate::tracy_zone!("cuda.enqueue_submit.alloc_timeline.ledger_lock");
                let mut guard = event_ledger.lock().unwrap();
                let _tz = crate::tracy_zone!("cuda.enqueue_submit.alloc_timeline.ledger_insert");
                guard.insert(
                    fence_value,
                    LedgerEntry {
                        context: ctx,
                        completion: LedgerCompletion::CudaEvent(Arc::clone(&completion_event)),
                        recorded: false,
                    },
                );
            }
            (fence_value, completion_event)
        };

        let device_retired = Arc::clone(&device.retired);
        let mut stream_waits = Vec::new();
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        let dx12_stream_fence_waits = Vec::new();
        let mut host_waits = Vec::new();
        let mut deferred_writes = Vec::new();
        if let Some(sync) = sync {
            let _tz = crate::tracy_zone!("cuda.enqueue_submit.resolve_sync");
            for epoch in &sync.waits {
                match timeline::completion_for_wait(&event_ledger, &device_retired, epoch.context, epoch.value) {
                    timeline::WaitCompletion::Pending(LedgerCompletion::CudaEvent(event)) => stream_waits.push(event),
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    timeline::WaitCompletion::Pending(LedgerCompletion::Dx12Fence { .. }) => {
                        // Present completion is a companion fence. Joining it here
                        // (`cuWaitExternalSemaphoresAsync`) serializes worker kernels
                        // behind the CUDA↔DX12 present tail and wakes CUDA after Present.
                    }
                    timeline::WaitCompletion::AlreadyComplete => {}
                    timeline::WaitCompletion::Missing => {
                        anyhow::bail!(
                            "CUDA: cross-context wait missing event for context {:?} value {}",
                            epoch.context,
                            epoch.value
                        );
                    }
                    #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
                    timeline::WaitCompletion::Pending(_) => unreachable!("only CudaEvent on non-dx12"),
                }
            }
            for epoch in sync.cpu_waits.iter().chain(sync.host_observed_waits.iter()) {
                match timeline::completion_for_wait(&event_ledger, &device_retired, epoch.context, epoch.value) {
                    timeline::WaitCompletion::Pending(LedgerCompletion::CudaEvent(event)) => host_waits.push(event),
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    timeline::WaitCompletion::Pending(LedgerCompletion::Dx12Fence { .. }) => {
                        // Same skip as sync.waits: do not wake CUDA on present-fence epochs.
                    }
                    timeline::WaitCompletion::AlreadyComplete => {}
                    timeline::WaitCompletion::Missing => {
                        anyhow::bail!(
                            "CUDA: host wait missing event for context {:?} value {}",
                            epoch.context,
                            epoch.value
                        );
                    }
                    #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
                    timeline::WaitCompletion::Pending(_) => unreachable!("only CudaEvent on non-dx12"),
                }
            }
            deferred_writes = pending_submit::materialize_deferred_writes(&sync.deferred_host_writes, |handle| {
                let buffer = self
                    .buffers
                    .get(&handle)
                    .with_context(|| format!("CUDA: deferred write invalid buffer {handle}"))?;
                Ok((Arc::clone(buffer.memory_arc()?), buffer.offset))
            })?;
        }

        let pending = CudaPendingSubmit {
            stream,
            context,
            fence_value,
            completion_event,
            event_ledger,
            stream_waits,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            dx12_stream_fence_waits,
            host_waits,
            deferred_writes,
            body,
        };
        {
            let _tz = crate::tracy_zone!("cuda.enqueue_submit.worker_enqueue");
            worker.enqueue(fence_value, Box::new(pending))?;
        }
        Ok(fence_value)
    }

    fn enqueue_evict_retained(&mut self, ctx: ContextHandle, key: u64) {
        let Some(context) = self.contexts.get(&ctx).cloned() else {
            return;
        };
        let Some(device) = self.devices.get(&context.device) else {
            return;
        };
        let retire_fallback = submission_worker::submission_horizon(&device.next_timeline);
        let job = pending_submit::CudaEvictRetained {
            ctx,
            key,
            registry: Arc::clone(&device.graph_registry),
            stats: Arc::clone(&device.graph_stats),
            device_retired: Arc::clone(&device.retired),
            retire_fallback,
        };
        // Best-effort: eviction is ordered after prior launches on the same worker.
        if let Err(error) = device.submission_worker.enqueue(0, Box::new(job)) {
            tracing::warn!(?error, ctx, key, "CUDA: failed to enqueue retained-graph eviction");
        }
    }

    /// Drop a retained CUDA graph immediately (caller must have drained GPU work).
    pub(super) fn destroy_retained_graph_sync(&mut self, ctx: ContextHandle, key: u64) {
        let device_handle = match self.context(ctx) {
            Ok(context) => context.device,
            Err(_) => return,
        };
        let device = match self.devices.get(&device_handle) {
            Some(device) => device,
            None => return,
        };
        let mut guard = device.graph_registry.lock().unwrap();
        guard.drain_retired(device.retired.load(Ordering::Acquire));
        if let Some(program) = guard.remove(ctx, key) {
            drop(program);
            device.graph_stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn clear_device_retained_sync(&mut self, device_handle: crate::backend::DeviceHandle) {
        let ctxs: Vec<_> = self
            .contexts
            .iter()
            .filter_map(|(handle, context)| (context.device == device_handle).then_some(*handle))
            .collect();
        for ctx in &ctxs {
            let keys: Vec<_> = self.retained.keys().filter(|(c, _)| *c == *ctx).copied().collect();
            for (c, key) in keys {
                self.retained.remove(&(c, key));
                self.destroy_retained_graph_sync(c, key);
            }
        }
        let device = match self.devices.get(&device_handle) {
            Some(device) => device,
            None => return,
        };
        let mut guard = device.graph_registry.lock().unwrap();
        guard.drain_retired(device.retired.load(Ordering::Acquire));
        for ctx in &ctxs {
            for program in guard.remove_context(*ctx) {
                drop(program);
            }
        }
        guard.clear_pending_drops();
    }

    pub(super) fn drop_retained_graphs_holding_memory(
        &mut self,
        device_handle: crate::backend::DeviceHandle,
        memory: &Arc<Mutex<CudaSlice<u8>>>,
        stream: &CudaStream,
        target_device_ptr: u64,
    ) {
        let device = match self.devices.get(&device_handle) {
            Some(device) => device,
            None => return,
        };
        let mut guard = device.graph_registry.lock().unwrap();
        guard.drain_retired(device.retired.load(Ordering::Acquire));
        guard.drop_graphs_holding_memory(memory, stream, target_device_ptr);
    }
}

/// Soft clone of buffer metadata + shared allocation (for copy that needs both ends).
impl CudaBuffer {
    fn clone_meta(&self) -> Self {
        Self {
            device: self.device,
            memory: self.memory.as_ref().map(Arc::clone),
            offset: self.offset,
            size: self.size,
            capacity: self.capacity,
            element_stride: self.element_stride,
            kind: self.kind,
            flags: self.flags,
            host_staging: self.host_staging.as_ref().map(Arc::clone),
            slot: self.slot,
            readback: self.readback,
            content_epoch: self.content_epoch,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            parent: self.parent,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            shared: self.shared.as_ref().map(Arc::clone),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            shared_epoch: self.shared_epoch,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            phys_kind: self.phys_kind,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            requirements: self.requirements,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            pending_init: self.pending_init.clone(),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            memory_is_external: self.memory_is_external,
        }
    }

    fn bump_content_epoch(&mut self) {
        self.content_epoch = self.content_epoch.wrapping_add(1);
    }
}

fn ensure_cuda_toolkit_on_path() {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let candidates = [
        std::env::var_os("CUDA_PATH")
            .map(PathBuf::from)
            .map(|p| p.join("bin/x64")),
        std::env::var_os("CUDA_PATH").map(PathBuf::from).map(|p| p.join("bin")),
        Some(PathBuf::from(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64",
        )),
        Some(PathBuf::from(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin",
        )),
        Some(PathBuf::from("/usr/local/cuda/bin")),
    ];
    for cand in candidates.into_iter().flatten() {
        if !cand.is_dir() {
            continue;
        }
        let cand_os = cand.as_os_str();
        if path.to_string_lossy().contains(cand.to_string_lossy().as_ref()) {
            return;
        }
        let mut new_path = cand_os.to_os_string();
        #[cfg(windows)]
        new_path.push(";");
        #[cfg(not(windows))]
        new_path.push(":");
        new_path.push(&path);
        // SAFETY: called before concurrent Slang/NVRTC work in this process for the PoC.
        unsafe { std::env::set_var("PATH", new_path) };
        return;
    }
}

/// CUDA 13.1 floor for device-updatable kernel nodes (`1000 * major + 10 * minor`).
const MIN_CUDA_DRIVER_VERSION: i32 = 13010;

fn cuda_driver_version_string(encoding: i32) -> String {
    let major = encoding / 1000;
    let minor = (encoding % 1000) / 10;
    format!("{major}.{minor}")
}

fn ensure_cuda_driver_at_least_13_1() -> Result<String> {
    let mut version = 0i32;
    let r = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        anyhow::bail!("CUDA: cuDriverGetVersion failed: {r:?}");
    }
    let driver_version = cuda_driver_version_string(version);
    tracing::info!("CUDA driver version: {driver_version} (encoding {version}; need >= {MIN_CUDA_DRIVER_VERSION})");
    if version < MIN_CUDA_DRIVER_VERSION {
        anyhow::bail!(
            "CUDA: goldy requires CUDA driver 13.1+ for device-updatable graph nodes \
             (got driver version encoding {version}; need >= {MIN_CUDA_DRIVER_VERSION})"
        );
    }
    Ok(driver_version)
}

fn query_device_limits(ctx: &CudaContext) -> Result<CudaDeviceLimits> {
    use cudarc::driver::sys::CUdevice_attribute;
    let attr = |a: CUdevice_attribute| -> Result<u32> {
        Ok(ctx.attribute(a).context("CUDA: cuDeviceGetAttribute failed")?.max(0) as u32)
    };
    Ok(CudaDeviceLimits {
        max_grid_dim_x: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
        max_grid_dim_y: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
        max_grid_dim_z: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        max_threads_per_block: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
        max_shared_memory_per_block: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
    })
}

/// Host-side launch-config checks when `GOLDY_VALIDATION=api` (or `all`) is set.
pub(super) fn validate_launch_config(
    limits: &CudaDeviceLimits,
    function_max_threads: u32,
    grid: (u32, u32, u32),
    block: [u32; 3],
    shared_mem_bytes: u32,
    label: Option<&'static str>,
) -> Result<()> {
    if !crate::backend::goldy_validation_enabled() {
        return Ok(());
    }
    validate_launch_config_unchecked(limits, function_max_threads, grid, block, shared_mem_bytes, label)
}

/// Unconditional launch-config checks (used by api validation and unit tests).
fn validate_launch_config_unchecked(
    limits: &CudaDeviceLimits,
    function_max_threads: u32,
    grid: (u32, u32, u32),
    block: [u32; 3],
    shared_mem_bytes: u32,
    label: Option<&'static str>,
) -> Result<()> {
    let where_ = label.unwrap_or("<unnamed>");
    let (gx, gy, gz) = grid;
    if gx == 0 || gy == 0 || gz == 0 {
        anyhow::bail!("CUDA validation: dispatch '{where_}' has zero grid dim ({gx},{gy},{gz})");
    }
    if gx > limits.max_grid_dim_x || gy > limits.max_grid_dim_y || gz > limits.max_grid_dim_z {
        anyhow::bail!(
            "CUDA validation: dispatch '{where_}' grid ({gx},{gy},{gz}) exceeds device max \
             ({},{},{})",
            limits.max_grid_dim_x,
            limits.max_grid_dim_y,
            limits.max_grid_dim_z
        );
    }
    let [bx, by, bz] = block;
    if bx == 0 || by == 0 || bz == 0 {
        anyhow::bail!("CUDA validation: dispatch '{where_}' has zero block dim ({bx},{by},{bz})");
    }
    let threads = bx as u64 * by as u64 * bz as u64;
    let max_fn = function_max_threads.max(1);
    let max_dev = limits.max_threads_per_block.max(1);
    if threads > max_fn as u64 {
        anyhow::bail!("CUDA validation: dispatch '{where_}' block threads {threads} exceeds function max {max_fn}");
    }
    if threads > max_dev as u64 {
        anyhow::bail!("CUDA validation: dispatch '{where_}' block threads {threads} exceeds device max {max_dev}");
    }
    if shared_mem_bytes > limits.max_shared_memory_per_block {
        anyhow::bail!(
            "CUDA validation: dispatch '{where_}' shared_mem {shared_mem_bytes} exceeds device max {}",
            limits.max_shared_memory_per_block
        );
    }
    Ok(())
}

fn c_string_log(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).trim().to_owned()
}

/// When api validation is on: JIT with error/info logs via `cuModuleLoadDataEx`, unload, then
/// load through cudarc's safe `load_module` (no public `CudaModule` constructor).
fn load_ptx_module(ctx: &Arc<CudaContext>, ptx: &str) -> Result<Arc<CudaModule>> {
    if crate::backend::goldy_validation_enabled() {
        load_ptx_module_validated(ctx, ptx)?;
    }
    ctx.load_module(Ptx::from_src(ptx.to_owned()))
        .context("CUDA: cuModuleLoadData failed")
}

/// Probe-load PTX with JIT log buffers; unload on success. Failures include driver log text.
fn load_ptx_module_validated(ctx: &Arc<CudaContext>, ptx: &str) -> Result<()> {
    use cudarc::driver::sys::{cuModuleLoadDataEx, cuModuleUnload, CUjit_option, CUmodule, CUresult};
    use std::mem::MaybeUninit;
    use std::os::raw::c_void;

    ctx.bind_to_thread()
        .context("CUDA: bind context for PTX JIT validation")?;
    let c_src = CString::new(ptx).context("CUDA: PTX source contains interior NUL")?;

    let mut error_log = vec![0u8; 16 * 1024];
    let mut info_log = vec![0u8; 16 * 1024];
    let error_size = error_log.len();
    let info_size = info_log.len();
    let verbose: usize = 1;
    let line_info: usize = 1;

    let mut options = [
        CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
        CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CUjit_option::CU_JIT_INFO_LOG_BUFFER,
        CUjit_option::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CUjit_option::CU_JIT_LOG_VERBOSE,
        CUjit_option::CU_JIT_GENERATE_LINE_INFO,
    ];
    let mut values: [*mut c_void; 6] = [
        error_log.as_mut_ptr().cast(),
        error_size as *mut c_void,
        info_log.as_mut_ptr().cast(),
        info_size as *mut c_void,
        verbose as *mut c_void,
        line_info as *mut c_void,
    ];

    let mut module = MaybeUninit::<CUmodule>::uninit();
    let status = unsafe {
        cuModuleLoadDataEx(
            module.as_mut_ptr(),
            c_src.as_ptr().cast(),
            options.len() as u32,
            options.as_mut_ptr(),
            values.as_mut_ptr(),
        )
    };
    let error_text = c_string_log(&error_log);
    let info_text = c_string_log(&info_log);
    if status != CUresult::CUDA_SUCCESS {
        anyhow::bail!("CUDA: PTX JIT failed ({status:?})\nerror log:\n{error_text}\ninfo log:\n{info_text}");
    }
    let module = unsafe { module.assume_init() };
    let unload = unsafe { cuModuleUnload(module) };
    if unload != CUresult::CUDA_SUCCESS {
        anyhow::bail!("CUDA: cuModuleUnload after validated JIT failed ({unload:?})");
    }
    if !info_text.is_empty() {
        tracing::debug!("CUDA PTX JIT info log:\n{info_text}");
    }
    Ok(())
}

impl GpuBackendSubmitSession for CudaBackend {
    fn clone_context_submit_session(
        &self,
        _ctx: ContextHandle,
        backend: std::sync::Arc<std::sync::Mutex<Box<dyn GpuBackend>>>,
    ) -> std::sync::Arc<dyn ContextSubmitSession> {
        LockedSubmitSession::with_backend_type(backend, BackendType::Cuda)
    }
}

impl GpuBackendTimelineWait for CudaBackend {
    fn take_timeline_submission_epoch_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<submission_worker::SubmissionEpochWait>> {
        if self.gpu_progress(ctx) >= value {
            return Ok(None);
        }
        let device_handle = self.context_device(ctx);
        let Some(device) = self.devices.get(&device_handle) else {
            return Ok(None);
        };
        let horizon = submission_worker::submission_horizon(&device.next_timeline);
        if value == 0 || value > horizon {
            return Ok(None);
        }
        Ok(Some(submission_worker::SubmissionEpochWait::new(
            Arc::clone(&device.submission_worker),
            value,
            horizon,
        )))
    }

    fn take_timeline_blocking_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn TimelineBlockingWait>>> {
        if self.gpu_progress(ctx) >= value {
            return Ok(None);
        }
        let device_handle = self.context_device(ctx);
        let device = self.device(device_handle)?;
        match timeline::completion_for_wait(&device.event_ledger, &device.retired, ctx, value) {
            timeline::WaitCompletion::AlreadyComplete => Ok(None),
            timeline::WaitCompletion::Pending(LedgerCompletion::CudaEvent(event)) => {
                Ok(Some(Box::new(timeline::CudaTimelineBlockingWait { event })))
            }
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            timeline::WaitCompletion::Pending(LedgerCompletion::Dx12Fence {
                companion,
                value: fence_value,
                recycle,
            }) => {
                if fence_value == 0 {
                    // Present TV reserved but not yet Signal'd — treat as not submitted.
                    Ok(Some(Box::new(timeline::CudaAbsentTimelineWait { context: ctx, value })))
                } else {
                    Ok(Some(Box::new(timeline::Dx12FenceTimelineBlockingWait {
                        companion,
                        value: fence_value,
                        recycle,
                    })))
                }
            }
            // Match DX12: waiting on a never-submitted value still yields a timeout-capable
            // wait object (not an immediate Err that used to deadlock under classify+lock).
            timeline::WaitCompletion::Missing => {
                Ok(Some(Box::new(timeline::CudaAbsentTimelineWait { context: ctx, value })))
            }
            #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
            timeline::WaitCompletion::Pending(_) => unreachable!("only CudaEvent on non-dx12"),
        }
    }

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        let context = Arc::clone(self.context(ctx)?);
        let device_handle = context.device;
        // Device-contiguous retirement is authoritative once prune has dropped ledger
        // entries: a wait target at or below `device_retired` is already complete even
        // if a racing poller briefly lagged `context.completed`.
        if value > 0 && value <= context.device_retired.load(Ordering::Acquire) {
            let retired = context.device_retired.load(Ordering::Acquire);
            drain_deletion_queue_up_to(&context.deletion_queue, retired);
            return Ok(());
        }
        // When the timeline has already retired, skip worker flush. Flush joins the
        // submission thread; if it is blocked in cuWaitExternalSemaphoresAsync (common
        // at multi-window teardown), Scheme::drop would hang forever even though
        // gpu_progress >= value.
        if self.gpu_progress(ctx) < value {
            if let Some(device) = self.devices.get(&device_handle) {
                device.submission_worker.flush()?;
            }
        }
        context.poll_retire_events();
        if value > 0 && value <= context.device_retired.load(Ordering::Acquire) {
            let retired = context.device_retired.load(Ordering::Acquire);
            drain_deletion_queue_up_to(&context.deletion_queue, retired);
            return Ok(());
        }
        if self.gpu_progress(ctx) < value {
            anyhow::bail!("CUDA: timeline value {value} was not submitted on context {ctx}");
        }
        let retired = context.device_retired.load(Ordering::Acquire);
        drain_deletion_queue_up_to(&context.deletion_queue, retired);
        Ok(())
    }
}

#[cfg(feature = "graphics")]
#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
impl GpuBackendPresentSplit for CudaBackend {
    fn take_present_gpu_work(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn PresentGpuWork>> {
        surface::take_present_gpu_work(self, frame, submit_tv)
    }

    fn finish_present(
        &mut self,
        finish: PresentFinishState,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        surface::finish_present(self, finish, submit_tv)
    }
}

#[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
impl GpuBackendPresentSplit for CudaBackend {
    fn take_present_gpu_work(
        &mut self,
        _frame: FrameToken,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn PresentGpuWork>> {
        Self::unsupported("presentation (requires cuda+graphics+dx12 on Windows)")
    }

    fn finish_present(
        &mut self,
        _finish: PresentFinishState,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        Self::unsupported("presentation (requires cuda+graphics+dx12 on Windows)")
    }
}

impl GpuBackend for CudaBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cuda
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        self.adapter_info.clone()
    }

    fn adapter_capabilities(&self, _adapter_id: u32) -> crate::device::DeviceCapabilities {
        crate::device::DeviceCapabilities {
            // Surfaces expose shared Rgba8Unorm scratch (DirectSpatial<float4> packs);
            // swapchain is matching R8G8B8A8 for a single CopyResource present.
            preferred_surface_format: TextureFormat::Rgba8Unorm,
            preferred_render_target_format: TextureFormat::Rgba8Unorm,
            supported_surface_formats: vec![TextureFormat::Rgba8Unorm],
            supported_render_target_formats: vec![TextureFormat::Rgba32Float, TextureFormat::Rgba8Unorm],
            has_zero_copy_storage_readback: false,
            buffer_resize_cost: BufferResizeCost::Copy,
            buffer_decommit_supported: false,
            host_sidecar_on_submit_worker: true,
            split_compute_partitions_on_barrier_cost: false,
            fuse_upload_with_compute_partitions: true,
            ..crate::device::DeviceCapabilities::default()
        }
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        ensure_cuda_toolkit_on_path();
        let ctx = CudaContext::new(adapter_id as usize)
            .with_context(|| format!("CUDA: create device for adapter {adapter_id}"))?;
        let limits =
            query_device_limits(&ctx).with_context(|| format!("CUDA: query device limits for adapter {adapter_id}"))?;
        let major = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .context("CUDA: query compute capability major")?;
        let minor = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .context("CUDA: query compute capability minor")?;
        let indirect_updater = Arc::new(
            runtime_module::load_indirect_updater(&ctx, (major, minor))
                .with_context(|| format!("CUDA: load indirect updater for adapter {adapter_id}"))?,
        );
        // Dedicated non-blocking stream — never the legacy default stream. Default-stream
        // allocs implicitly wait on every other stream, including a THREAD_LOCAL graph
        // capture on the submit worker, which yields CUDA_ERROR_STREAM_CAPTURE_ISOLATION
        // and invalidates in-flight upload graphs.
        let alloc_stream = ctx
            .new_stream()
            .with_context(|| format!("CUDA: create alloc stream for adapter {adapter_id}"))?;
        let event_pool = Arc::new(timeline::EventPool::new(Arc::clone(&ctx)));
        event_pool
            .prewarm()
            .with_context(|| format!("CUDA: prewarm event pool for adapter {adapter_id}"))?;
        let handle = self.next_device;
        self.next_device += 1;
        let mut gpu = CudaDevice {
            ctx,
            alloc_stream,
            submission_worker: Arc::new(SubmissionWorker::new(submission_worker::SUBMISSION_QUEUE_CAPACITY)),
            next_timeline: Arc::new(AtomicU64::new(1)),
            retired: Arc::new(AtomicU64::new(0)),
            event_ledger: Arc::new(Mutex::new(BTreeMap::new())),
            event_pool,
            deletion_queue: Arc::new(Mutex::new(Vec::new())),
            graph_registry: Arc::new(Mutex::new(GraphRegistry::default())),
            graph_stats: Arc::clone(&self.graph_stats),
            limits,
            indirect_updater,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            dx12: None,
        };
        let dx12_companion = {
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            {
                surface::attach_companion(&mut gpu)
                    .with_context(|| format!("CUDA: attach DX12 presentation companion for adapter {adapter_id}"))?;
                gpu.dx12.is_some()
            }
            #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
            {
                false
            }
        };
        tracing::info!(
            device = handle,
            adapter_id,
            cc = %format!("{major}.{minor}"),
            max_grid = ?(limits.max_grid_dim_x, limits.max_grid_dim_y, limits.max_grid_dim_z),
            max_threads_per_block = limits.max_threads_per_block,
            max_shared_mem_per_block = limits.max_shared_memory_per_block,
            dx12_companion,
            "Created CUDA device"
        );
        self.devices.insert(handle, gpu);
        Ok(handle)
    }

    fn destroy_device(&mut self, device: DeviceHandle) {
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        {
            let surfaces: Vec<_> = self
                .surfaces
                .iter()
                .filter_map(|(h, s)| (s.device == device).then_some(*h))
                .collect();
            for surface in surfaces {
                surface::destroy_surface(self, surface);
            }
        }
        let contexts: Vec<_> = self
            .contexts
            .iter()
            .filter_map(|(handle, context)| (context.device == device).then_some(*handle))
            .collect();
        for ctx in contexts {
            let _ = destroy_context_mut(self, ctx);
        }
        if let Some(mut gpu) = self.devices.remove(&device) {
            let _ = gpu.submission_worker.flush();
            let _ = gpu.alloc_stream.synchronize();
            gpu.submission_worker.shutdown();
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            {
                gpu.dx12 = None;
            }
        }
        self.buffers.retain(|_, buffer| buffer.device != device);
        self.shaders.retain(|_, shader| shader.device != device);
        self.compute_pipelines.retain(|_, pipeline| pipeline.device != device);
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        {
            self.pipelines.retain(|_, p| p.device != device);
            let rt_tex: Vec<_> = self
                .render_targets
                .iter()
                .filter(|(_, rt)| rt.device == device)
                .map(|(_, rt)| {
                    // Find texture handles that alias the RT cuda_texture via slot map.
                    rt.cuda_texture.storage_slot
                })
                .collect();
            self.render_targets.retain(|_, rt| rt.device != device);
            self.raster_list_cache.clear();
            for slot in rt_tex.into_iter().flatten() {
                if let Some(tex) = self.texture_slots.remove(&slot) {
                    self.textures.remove(&tex);
                }
            }
        }
        self.buffer_slots.retain(|_, handle| self.buffers.contains_key(handle));
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn device_wait_idle(&mut self, device: DeviceHandle) -> Result<()> {
        let worker = Arc::clone(&self.device(device)?.submission_worker);
        let alloc_stream = Arc::clone(&self.device(device)?.alloc_stream);
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        let dx12 = self.device(device)?.dx12.as_ref().map(Arc::clone);
        worker.flush()?;
        for context in self.contexts.values().filter(|context| context.device == device) {
            context
                .stream
                .synchronize()
                .context("CUDA: context stream synchronize failed")?;
            context.poll_retire_events();
        }
        alloc_stream.synchronize().context("CUDA: device wait idle failed")?;
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        if let Some(companion) = dx12 {
            companion.wait_idle()?;
        }
        Ok(())
    }

    fn create_context(&mut self, device: DeviceHandle) -> Result<ContextHandle> {
        let existing = self
            .contexts
            .values()
            .filter(|context| context.device == device)
            .count() as u32;
        if existing >= MAX_CUDA_SUBMISSION_CONTEXTS {
            anyhow::bail!("CUDA: submission context limit reached ({MAX_CUDA_SUBMISSION_CONTEXTS} per device)");
        }
        let (stream, event_ledger, event_pool, device_retired, deletion_queue) = {
            let gpu = self.device(device)?;
            (
                gpu.ctx.new_stream().context("CUDA: create context stream failed")?,
                Arc::clone(&gpu.event_ledger),
                Arc::clone(&gpu.event_pool),
                Arc::clone(&gpu.retired),
                Arc::clone(&gpu.deletion_queue),
            )
        };
        let handle = self.next_context;
        self.next_context += 1;
        let fence_shutdown = Arc::new(AtomicBool::new(false));
        let context = Arc::new(CudaSubmitContext {
            handle,
            device,
            stream,
            completed: AtomicU64::new(0),
            last_emitted: AtomicU64::new(0),
            signal_queue: crate::signal::SignalQueue::new(),
            device_retired,
            event_ledger,
            event_pool,
            deletion_queue,
            fence_shutdown: Arc::clone(&fence_shutdown),
            fence_thread: Mutex::new(None),
        });

        let poller_context = Arc::clone(&context);
        let shutdown = Arc::clone(&fence_shutdown);
        let handle_thread = std::thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                poller_context.poll_retire_events();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });
        *context.fence_thread.lock().unwrap() = Some(handle_thread);

        self.contexts.insert(handle, context);
        Ok(handle)
    }

    fn detach_context_for_destroy(&mut self, ctx: ContextHandle) -> Option<Box<dyn ContextDestroyHandle>> {
        let context = self.contexts.remove(&ctx)?;
        let keys: Vec<u64> = self
            .retained
            .keys()
            .filter_map(|(c, k)| (*c == ctx).then_some(*k))
            .collect();
        for key in keys {
            self.retained.remove(&(ctx, key));
        }
        let worker = self
            .devices
            .get(&context.device)
            .map(|device| Arc::clone(&device.submission_worker));
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        let companion = self
            .devices
            .get(&context.device)
            .and_then(|d| d.dx12.as_ref().map(Arc::clone));
        if let Some(device) = self.devices.get(&context.device) {
            let retire_fallback = submission_worker::submission_horizon(&device.next_timeline);
            let job = pending_submit::CudaEvictContextGraphs {
                ctx,
                registry: Arc::clone(&device.graph_registry),
                stats: Arc::clone(&device.graph_stats),
                device_retired: Arc::clone(&device.retired),
                retire_fallback,
            };
            let _ = device.submission_worker.enqueue(0, Box::new(job));
        }
        let fence_thread = context.fence_thread.lock().unwrap().take();
        context.fence_shutdown.store(true, Ordering::Relaxed);
        let cuda_ctx = Arc::clone(&self.devices.get(&context.device)?.ctx);
        Some(Box::new(CudaDestroyContext {
            cuda_ctx,
            stream: Arc::clone(&context.stream),
            worker: worker.unwrap_or_else(|| Arc::new(SubmissionWorker::new(1))),
            fence_shutdown: Arc::clone(&context.fence_shutdown),
            fence_thread: Mutex::new(fence_thread),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            companion,
        }))
    }

    fn clone_context_deletion_flush(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn ContextDeferredDeletionFlush>> {
        Some(Arc::new(CudaDeferredDeletionFlush {
            context: Arc::clone(self.contexts.get(&ctx)?),
        }))
    }

    fn clone_context_gpu_progress(&self, ctx: ContextHandle) -> Option<std::sync::Arc<dyn ContextGpuProgress>> {
        let context = Arc::clone(self.contexts.get(&ctx)?);
        Some(Arc::new(CudaProgress { context }))
    }

    fn context_device(&self, ctx: ContextHandle) -> DeviceHandle {
        self.contexts.get(&ctx).map(|context| context.device).unwrap_or(0)
    }

    fn create_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<BufferHandle> {
        self.create_storage_buffer(device, size, size, element_stride, access, flags)
    }

    fn create_buffer_with_capacity(
        &mut self,
        device: DeviceHandle,
        initial_size: u64,
        capacity: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        let capacity = capacity.max(initial_size);
        let handle = self.create_storage_buffer(device, initial_size, capacity, element_stride, access, flags)?;
        let stored = self.buffers.get(&handle).map(|b| b.capacity).unwrap_or(capacity);
        Ok((handle, stored))
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) {
        if let Some(buffer) = self.buffers.remove(&buffer) {
            if let Some(slot) = buffer.slot {
                self.buffer_slots.remove(&slot);
            }
            if let Some(device) = self.devices.get(&buffer.device) {
                let retire_at = submission_worker::submission_horizon(&device.next_timeline);
                device.deletion_queue.lock().unwrap().push(CudaDeferredDrop::Buffer {
                    retire_at,
                    memory: buffer.memory,
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    shared: buffer.shared,
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    memory_is_external: buffer.memory_is_external,
                });
            }
        }
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        // CPU_WRITABLE / deposit staging: host memcpy only (DX12 UPLOAD model). HtoD is
        // performed at Copy/CopyBufferToTexture materialization on the context stream —
        // never flush the submission worker or sync alloc_stream here.
        {
            let (target, stage_offset, logical_size, has_staging) = {
                let buf = self.buffers.get(&buffer).context("CUDA: invalid buffer handle")?;
                let self_has = buf.has_host_staging();
                let logical_size = buf.size;
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                let parent = buf.parent;
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                let view_abs = buf.offset + offset;
                let _ = buf;

                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                {
                    if let Some(parent) = parent {
                        let parent_has = self.buffers.get(&parent).is_some_and(|p| p.has_host_staging());
                        if parent_has {
                            (parent, view_abs, logical_size, true)
                        } else if self_has {
                            (buffer, offset, logical_size, true)
                        } else {
                            (buffer, offset, logical_size, false)
                        }
                    } else if self_has {
                        (buffer, offset, logical_size, true)
                    } else {
                        (buffer, offset, logical_size, false)
                    }
                }
                #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
                {
                    if self_has {
                        (buffer, offset, logical_size, true)
                    } else {
                        (buffer, offset, logical_size, false)
                    }
                }
            };
            if has_staging {
                if offset + data.len() as u64 > logical_size {
                    anyhow::bail!("CUDA: write exceeds logical buffer size");
                }
                let buf = self.buffers.get_mut(&target).context("CUDA: invalid buffer handle")?;
                let staging = buf.host_staging.as_ref().context("CUDA: missing host staging")?;
                {
                    let mut staging = staging.lock().unwrap();
                    let end = stage_offset as usize + data.len();
                    if end > staging.len() {
                        anyhow::bail!("CUDA: write exceeds host staging capacity");
                    }
                    staging.as_mut_slice()[stage_offset as usize..end].copy_from_slice(data);
                }
                buf.bump_content_epoch();
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                if target != buffer {
                    if let Some(v) = self.buffers.get_mut(&buffer) {
                        v.bump_content_epoch();
                    }
                }
                return Ok(());
            }
        }

        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        {
            // Stage host bytes on deferred buffers — do not force Native materialization
            // before scheme usage is known (deposit→Shared path). Views stage into the parent.
            let (is_deferred, parent, abs_offset, logical_size) = {
                let buf = self.buffers.get(&buffer).context("CUDA: invalid buffer handle")?;
                (buf.phys_kind.is_deferred(), buf.parent, buf.offset + offset, buf.size)
            };
            if is_deferred {
                if offset + data.len() as u64 > logical_size {
                    anyhow::bail!("CUDA: write exceeds logical buffer size");
                }
                let target = parent.unwrap_or(buffer);
                let stage_offset = if parent.is_some() { abs_offset } else { offset };
                let buf = self.buffers.get_mut(&target).context("CUDA: invalid buffer handle")?;
                let pending_len = if parent.is_some() {
                    buf.size as usize
                } else {
                    buf.size as usize
                };
                if stage_offset + data.len() as u64 > buf.size {
                    anyhow::bail!("CUDA: write exceeds parent buffer size");
                }
                let mut pending = buf.pending_init.take().unwrap_or_else(|| vec![0u8; pending_len]);
                if pending.len() < pending_len {
                    pending.resize(pending_len, 0);
                }
                let start = stage_offset as usize;
                pending[start..start + data.len()].copy_from_slice(data);
                buf.pending_init = Some(pending);
                buf.requirements |= buffer_phys::CudaBufferReq::HOST_WRITE;
                buf.bump_content_epoch();
                if let Some(view) = parent.map(|_| buffer) {
                    // Keep view epoch in sync for retained fingerprints.
                    if let Some(v) = self.buffers.get_mut(&view) {
                        v.bump_content_epoch();
                    }
                }
                return Ok(());
            }
            self.ensure_buffer_requirements(buffer, buffer_phys::CudaBufferReq::HOST_WRITE)?;
        }
        self.sync_device_streams_for_immediate_api(
            self.buffers
                .get(&buffer)
                .map(|buffer| buffer.device)
                .context("CUDA: invalid buffer handle")?,
        )?;
        let device = {
            let buffer = self.buffers.get_mut(&buffer).context("CUDA: invalid buffer handle")?;
            buffer.bump_content_epoch();
            buffer.device
        };
        let stream = Arc::clone(&self.device(device)?.alloc_stream);
        let buffer_ref = self.buffers.get(&buffer).context("CUDA: invalid buffer handle")?;
        Self::write_buffer_region(&stream, buffer_ref, offset, data)?;
        Ok(())
    }

    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle> {
        let gpu = self.device(device)?;
        let _gate = capture_gate::lock_capture_alloc_gate();
        let capacity = size.max(4);
        let memory = Arc::new(Mutex::new(
            gpu.alloc_stream
                .alloc_zeros::<u8>(capacity as usize)
                .context("CUDA: alloc readback")?,
        ));
        let handle = self.next_buffer;
        self.next_buffer += 1;
        self.buffers.insert(
            handle,
            CudaBuffer {
                device,
                memory: Some(memory),
                offset: 0,
                size,
                capacity,
                element_stride: None,
                kind: BufferKind::Scattered,
                flags: BufferFlags::empty(),
                host_staging: None,
                slot: None,
                readback: true,
                content_epoch: 0,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                parent: None,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                shared: None,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                shared_epoch: 0,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                phys_kind: buffer_phys::CudaPhysKind::Native,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                requirements: buffer_phys::CudaBufferReq::empty(),
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                pending_init: None,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                memory_is_external: false,
            },
        );
        Ok(handle)
    }

    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        let buffer = self.buffers.get(&buffer).context("CUDA: invalid readback buffer")?;
        if !buffer.readback {
            anyhow::bail!("CUDA: buffer is not readback staging");
        }
        if output.len() as u64 > buffer.size {
            anyhow::bail!("CUDA: read exceeds readback buffer size");
        }
        let device = buffer.device;
        // Ensure any context-stream copy into this staging buffer has retired.
        let worker = Arc::clone(&self.device(device)?.submission_worker);
        worker.flush()?;
        self.graph_stats.worker_flushes.fetch_add(1, Ordering::Relaxed);
        for context in self.contexts.values().filter(|context| context.device == device) {
            context
                .stream
                .synchronize()
                .context("CUDA: readback context stream sync failed")?;
        }
        let stream = Arc::clone(&self.device(device)?.alloc_stream);
        let memory = buffer.memory_arc()?.lock().unwrap();
        let view = memory
            .try_slice(buffer.offset as usize..(buffer.offset as usize + output.len()))
            .context("CUDA: readback range out of bounds")?;
        stream
            .memcpy_dtoh(&view, output)
            .context("CUDA: DtoH readback failed")?;
        self.graph_stats.dtoh_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn free_readback_buffer(&mut self, buffer: BufferHandle) {
        self.destroy_buffer(buffer);
    }

    fn query_texture_copy_footprint(
        &self,
        _device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<TextureCopyFootprint> {
        // Reject unsupported formats early (e.g. BGRA).
        texture::format_info(format)?;
        let row_pitch = width.saturating_mul(format.bytes_per_pixel());
        let logical_bytes = row_pitch as u64 * height as u64;
        Ok(TextureCopyFootprint {
            width,
            height,
            format,
            logical_bytes,
            staging_bytes: logical_bytes,
            row_pitch,
            footprint_offset: 0,
        })
    }

    fn alloc_texture_readback_staging(
        &mut self,
        device: DeviceHandle,
        layout: TextureCopyFootprint,
    ) -> Result<BufferHandle> {
        self.alloc_readback_buffer(device, layout.staging_bytes)
    }

    fn read_texture_readback_staging(
        &self,
        buffer: BufferHandle,
        layout: TextureCopyFootprint,
        output: &mut [u8],
    ) -> Result<()> {
        if output.len() as u64 != layout.logical_bytes {
            anyhow::bail!(
                "CUDA: read_texture_readback_staging size mismatch: expected {}, got {}",
                layout.logical_bytes,
                output.len()
            );
        }
        let buf = self
            .buffers
            .get(&buffer)
            .context("CUDA: invalid texture readback staging buffer")?;
        if !buf.readback {
            anyhow::bail!("CUDA: read_texture_readback_staging requires a withdraw staging buffer");
        }
        // Tight footprint: staging == logical.
        let mut staging = vec![0u8; layout.staging_bytes as usize];
        self.read_readback_buffer(buffer, &mut staging)?;
        if layout.row_pitch == layout.tight_row_bytes() && layout.footprint_offset == 0 {
            output.copy_from_slice(&staging[..layout.logical_bytes as usize]);
            return Ok(());
        }
        // Pitched unpack (defensive; CUDA uses tight rows today).
        let tight = layout.tight_row_bytes() as usize;
        let pitch = layout.row_pitch as usize;
        let base = layout.footprint_offset as usize;
        for row in 0..layout.height as usize {
            let src = base + row * pitch;
            let dst = row * tight;
            output[dst..dst + tight].copy_from_slice(&staging[src..src + tight]);
        }
        Ok(())
    }

    fn texture_copy_retention_tag(&self, _texture: TextureHandle) -> u64 {
        // CUDA arrays have no layout transitions; a constant tag keeps pitched
        // buffer→texture partition fingerprints stable across submits.
        0
    }

    fn clear_buffer(&mut self, device: DeviceHandle, buffer: BufferHandle, offset: u64, size: u64) -> Result<()> {
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        {
            let (is_deferred, parent, abs_offset, logical_size) = {
                let buf = self.buffers.get(&buffer).context("CUDA: invalid buffer handle")?;
                (buf.phys_kind.is_deferred(), buf.parent, buf.offset + offset, buf.size)
            };
            if is_deferred {
                let clear_size = if size == 0 {
                    logical_size.saturating_sub(offset)
                } else {
                    size
                };
                if offset.saturating_add(clear_size) > logical_size {
                    anyhow::bail!("CUDA: clear exceeds logical buffer size");
                }
                let target = parent.unwrap_or(buffer);
                let stage_offset = if parent.is_some() { abs_offset } else { offset };
                let buf = self.buffers.get_mut(&target).context("CUDA: invalid buffer handle")?;
                let pending_len = buf.size as usize;
                let mut pending = buf.pending_init.take().unwrap_or_else(|| vec![0u8; pending_len]);
                if pending.len() < pending_len {
                    pending.resize(pending_len, 0);
                }
                let start = stage_offset as usize;
                let end = start + clear_size as usize;
                pending[start..end].fill(0);
                buf.pending_init = Some(pending);
                buf.requirements |= buffer_phys::CudaBufferReq::HOST_WRITE;
                buf.bump_content_epoch();
                if parent.is_some() {
                    if let Some(v) = self.buffers.get_mut(&buffer) {
                        v.bump_content_epoch();
                    }
                }
                return Ok(());
            }
            self.ensure_buffer_requirements(
                buffer,
                buffer_phys::CudaBufferReq::TRANSFER | buffer_phys::CudaBufferReq::HOST_WRITE,
            )?;
        }
        self.sync_device_streams_for_immediate_api(device)?;
        let stream = Arc::clone(&self.device(device)?.alloc_stream);
        let target = self.buffers.get_mut(&buffer).context("CUDA: invalid buffer handle")?;
        target.bump_content_epoch();
        Self::clear_buffer_region(&stream, target, offset, size)
    }

    fn buffer_size(&self, buffer: BufferHandle) -> u64 {
        self.buffers.get(&buffer).map(|buffer| buffer.size).unwrap_or(0)
    }

    fn buffer_capacity(&self, buffer: BufferHandle) -> u64 {
        self.buffers.get(&buffer).map(|buffer| buffer.capacity).unwrap_or(0)
    }

    fn set_buffer_logical_size(
        &mut self,
        _device: DeviceHandle,
        buffer: BufferHandle,
        new_logical_size: u64,
    ) -> Result<()> {
        let buffer = self.buffers.get_mut(&buffer).context("CUDA: invalid buffer handle")?;
        if new_logical_size == 0 || new_logical_size > buffer.capacity {
            anyhow::bail!("CUDA: logical size must be in 1..=capacity");
        }
        buffer.size = new_logical_size;
        Ok(())
    }

    fn buffer_bindless_index(&self, buffer: BufferHandle) -> Option<u32> {
        self.buffers.get(&buffer)?.slot
    }

    fn buffer_bindless_srv_index(&self, buffer: BufferHandle) -> Option<u32> {
        self.buffer_bindless_index(buffer)
    }

    fn create_buffer_view(
        &mut self,
        parent: BufferHandle,
        offset: u64,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        let parent_handle = parent;
        let parent = self
            .buffers
            .get(&parent_handle)
            .context("CUDA: invalid parent buffer")?
            .clone_meta();
        if offset + size > parent.size {
            anyhow::bail!("CUDA: buffer view exceeds parent");
        }
        let handle = self.next_buffer;
        self.next_buffer += 1;
        let slot = self.next_slot;
        self.next_slot += 1;
        self.buffer_slots.insert(slot, handle);
        self.buffers.insert(
            handle,
            CudaBuffer {
                device: parent.device,
                memory: parent.memory,
                offset: parent.offset + offset,
                size,
                capacity: size,
                element_stride: element_stride.or(parent.element_stride),
                kind: parent.kind,
                flags: parent.flags,
                // Views write through the parent; no separate host staging.
                host_staging: None,
                slot: Some(slot),
                readback: false,
                content_epoch: parent.content_epoch,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                parent: Some(parent_handle),
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                shared: parent.shared,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                shared_epoch: parent.shared_epoch,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                phys_kind: parent.phys_kind,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                requirements: parent.requirements,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                pending_init: None,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                memory_is_external: parent.memory_is_external,
            },
        );
        Ok(handle)
    }

    fn resize_buffer(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        new_size: u64,
        preserve_contents: bool,
    ) -> Result<()> {
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        {
            let is_deferred = self
                .buffers
                .get(&buffer)
                .map(|b| b.phys_kind.is_deferred())
                .unwrap_or(false);
            if is_deferred {
                let buf = self.buffers.get_mut(&buffer).context("CUDA: invalid buffer handle")?;
                if buf.parent.is_some() {
                    anyhow::bail!("CUDA: cannot resize a buffer view");
                }
                if buf.device != device {
                    anyhow::bail!("CUDA: buffer belongs to another device");
                }
                let new_cap = new_size.max(4);
                match &mut buf.pending_init {
                    Some(pending) if preserve_contents => {
                        pending.resize(new_size as usize, 0);
                    }
                    Some(pending) => {
                        *pending = vec![0u8; new_size as usize];
                    }
                    None if preserve_contents || new_size > 0 => {
                        // Unwritten deferred bytes are zeros; keep that contract after resize.
                        buf.pending_init = Some(vec![0u8; new_size as usize]);
                    }
                    None => {}
                }
                if let Some(staging) = buf.host_staging.as_ref() {
                    let mut staging = staging.lock().unwrap();
                    staging.resize(new_cap as usize, preserve_contents)?;
                }
                buf.size = new_size;
                buf.capacity = new_cap;
                buf.bump_content_epoch();
                return Ok(());
            }
        }
        let old = self
            .buffers
            .get(&buffer)
            .context("CUDA: invalid buffer handle")?
            .clone_meta();
        if old.device != device {
            anyhow::bail!("CUDA: buffer belongs to another device");
        }
        self.sync_device_streams_for_immediate_api(device)?;
        let stream = Arc::clone(&self.device(device)?.alloc_stream);
        let capacity = new_size.max(4);
        let mut replacement = stream
            .alloc_zeros::<u8>(capacity as usize)
            .context("CUDA: resize alloc")?;
        if preserve_contents {
            let copy_size = old.size.min(new_size);
            if copy_size > 0 {
                let memory = old.memory_arc()?.lock().unwrap();
                let src = memory
                    .try_slice(old.offset as usize..(old.offset + copy_size) as usize)
                    .context("CUDA: resize src")?;
                let mut dst = replacement
                    .try_slice_mut(0..copy_size as usize)
                    .context("CUDA: resize dst")?;
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .context("CUDA: resize device-to-device copy")?;
            }
        }

        if let Some(device) = self.devices.get(&device) {
            let retire_at = submission_worker::submission_horizon(&device.next_timeline);
            device.deletion_queue.lock().unwrap().push(CudaDeferredDrop::Buffer {
                retire_at,
                memory: old.memory,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                shared: old.shared,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                memory_is_external: old.memory_is_external,
            });
        }

        let target = self.buffers.get_mut(&buffer).expect("validated above");
        target.memory = Some(Arc::new(Mutex::new(replacement)));
        target.offset = 0;
        target.size = new_size;
        target.capacity = capacity;
        if let Some(staging) = target.host_staging.as_ref() {
            staging.lock().unwrap().resize(capacity as usize, preserve_contents)?;
        }
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        {
            // Shared twin is size-specific; drop and recreate on next VB bind.
            target.shared = None;
            target.shared_epoch = 0;
            target.phys_kind = buffer_phys::CudaPhysKind::Native;
            target.memory_is_external = false;
        }
        target.bump_content_epoch();
        Ok(())
    }

    fn create_shader_with_paths(
        &mut self,
        device: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
    ) -> Result<ShaderHandle> {
        self.device(device)?;
        let handle = self.next_shader;
        self.next_shader += 1;
        self.shaders.insert(
            handle,
            CudaShader {
                device,
                source: slang_source.to_owned(),
                search_paths: search_paths.iter().map(|value| (*value).to_owned()).collect(),
                defines: defines
                    .iter()
                    .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                    .collect(),
                optimization_level,
            },
        );
        Ok(handle)
    }

    fn destroy_shader(&mut self, shader: ShaderHandle) {
        self.shaders.remove(&shader);
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn create_pipeline(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        raster::create_pipeline(
            self,
            device,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
            None,
        )
    }

    #[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
    fn create_pipeline(
        &mut self,
        _device: DeviceHandle,
        _vertex_shader: ShaderHandle,
        _fragment_shader: ShaderHandle,
        _vertex_layout: &VertexBufferLayout,
        _topology: PrimitiveTopology,
        _target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        Self::unsupported("graphics pipelines (requires cuda+graphics+dx12 on Windows)")
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn render_pipeline_slot_access(&self, pipeline: PipelineHandle) -> Vec<Option<ResourceAccess>> {
        self.pipelines
            .get(&pipeline)
            .map(|p| {
                p.push_constant_slot_kinds
                    .iter()
                    .map(|kind| match kind {
                        Some(crate::types::BindlessSlotKind::StorageUav) => Some(ResourceAccess::ReadWrite),
                        Some(crate::types::BindlessSlotKind::ReadOnlySrv) => Some(ResourceAccess::Read),
                        Some(crate::types::BindlessSlotKind::UniformCbv) | None => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    #[cfg(feature = "graphics")]
    fn destroy_pipeline(&mut self, pipeline: PipelineHandle) {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        raster::destroy_pipeline(self, pipeline);
        #[cfg(not(all(feature = "dx12", target_os = "windows")))]
        let _ = pipeline;
    }

    #[cfg(feature = "graphics")]
    fn create_pipeline_with_depth(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        depth_stencil: Option<&DepthStencilState>,
    ) -> Result<PipelineHandle> {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            return raster::create_pipeline(
                self,
                device,
                vertex_shader,
                fragment_shader,
                vertex_layout,
                topology,
                target_format,
                depth_stencil,
            );
        }
        #[cfg(not(all(feature = "dx12", target_os = "windows")))]
        {
            let _ = (
                device,
                vertex_shader,
                fragment_shader,
                vertex_layout,
                topology,
                target_format,
                depth_stencil,
            );
            Self::unsupported("graphics pipelines (requires cuda+graphics+dx12 on Windows)")
        }
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn create_render_target_with_depth(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        raster::create_render_target(self, device, width, height, color_format, depth_format)
    }

    #[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
    fn create_render_target_with_depth(
        &mut self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _color_format: TextureFormat,
        _depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        Self::unsupported("render targets (requires cuda+graphics+dx12 on Windows)")
    }

    #[cfg(feature = "graphics")]
    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            raster::destroy_render_target(self, target);
        }
        #[cfg(not(all(feature = "dx12", target_os = "windows")))]
        {
            let _ = target;
        }
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn render_to_target(
        &mut self,
        device: DeviceHandle,
        target: RenderTargetHandle,
        color_load: crate::types::TargetLoad,
        commands: &[RenderCommand],
    ) -> Result<()> {
        raster::render_to_target(self, device, target, color_load, commands, None)
    }

    #[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
    fn render_to_target(
        &mut self,
        _device: DeviceHandle,
        _target: RenderTargetHandle,
        _color_load: crate::types::TargetLoad,
        _commands: &[RenderCommand],
    ) -> Result<()> {
        Self::unsupported("rendering (requires cuda+graphics+dx12 on Windows)")
    }

    fn create_texture(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<TextureHandle> {
        let ctx = Arc::clone(&self.device(device)?.ctx);
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        let companion = self.device(device)?.dx12.clone();
        let (storage_slot, sampled_slot) = match access {
            TextureKind::Interpolated => (None, Some(self.alloc_registry_slot())),
            TextureKind::Direct => (Some(self.alloc_registry_slot()), None),
            TextureKind::DirectInterpolated => (Some(self.alloc_registry_slot()), Some(self.alloc_registry_slot())),
        };

        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        if let Some(companion) = companion {
            return self.create_shared_texture_resource(
                device,
                &companion,
                &ctx,
                width,
                height,
                format,
                access,
                flags,
                storage_slot,
                sampled_slot,
            );
        }

        let resource =
            CudaTextureResource::create(&ctx, width, height, format, access, flags, storage_slot, sampled_slot)?;
        let handle = self.next_texture;
        self.next_texture += 1;
        if let Some(slot) = storage_slot {
            self.texture_slots.insert(slot, handle);
        }
        if let Some(slot) = sampled_slot {
            self.texture_slots.insert(slot, handle);
        }
        self.textures.insert(handle, resource);
        Ok(handle)
    }

    fn write_texture(&mut self, texture: TextureHandle, data: &[u8], width: u32, height: u32) -> Result<()> {
        let tex = self.textures.get(&texture).context("CUDA: invalid texture handle")?;
        let device = {
            // Find owning device via context of the array's CudaContext — textures store ctx Arc.
            // Use any device whose ctx matches; fall back to syncing all isn't needed if we sync
            // via the texture's own context stream.
            tex.ctx.clone()
        };
        // Sync all devices that share this context (one CUDA context per device).
        let device_handle = self
            .devices
            .iter()
            .find(|(_, d)| Arc::ptr_eq(&d.ctx, &device))
            .map(|(h, _)| *h)
            .context("CUDA: texture device not found")?;
        self.sync_device_streams_for_immediate_api(device_handle)?;
        let stream = Arc::clone(&self.device(device_handle)?.alloc_stream);
        let tex = self.textures.get(&texture).context("CUDA: invalid texture handle")?;
        if width != tex.width || height != tex.height {
            anyhow::bail!(
                "CUDA: write_texture size {}x{} does not match texture {}x{}",
                width,
                height,
                tex.width,
                tex.height
            );
        }
        memcpy_htod_array(&stream, tex, 0, 0, width, height, data, 0)?;
        stream.synchronize().context("CUDA: synchronize after write_texture")?;
        Ok(())
    }

    fn write_texture_region(
        &mut self,
        texture: TextureHandle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<()> {
        let tex = self.textures.get(&texture).context("CUDA: invalid texture handle")?;
        let device_handle = self
            .devices
            .iter()
            .find(|(_, d)| Arc::ptr_eq(&d.ctx, &tex.ctx))
            .map(|(h, _)| *h)
            .context("CUDA: texture device not found")?;
        self.sync_device_streams_for_immediate_api(device_handle)?;
        let stream = Arc::clone(&self.device(device_handle)?.alloc_stream);
        let tex = self.textures.get(&texture).context("CUDA: invalid texture handle")?;
        memcpy_htod_array(&stream, tex, x, y, width, height, data, 0)?;
        stream
            .synchronize()
            .context("CUDA: synchronize after write_texture_region")?;
        Ok(())
    }

    fn destroy_texture(&mut self, texture: TextureHandle) {
        if let Some(resource) = self.textures.remove(&texture) {
            if let Some(slot) = resource.storage_slot {
                self.texture_slots.remove(&slot);
            }
            if let Some(slot) = resource.sampled_slot {
                self.texture_slots.remove(&slot);
            }
            // Pull import/D3D12 out of the live maps now, but keep them alive until after
            // `resource` drops — the CUarray is borrowed from the mapped mipmapped array.
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            let import = self.texture_imports.remove(&texture);
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            let d3d12_resource = self.texture_dx12.remove(&texture);
            let device_handle = self
                .devices
                .iter()
                .find(|(_, d)| Arc::ptr_eq(&d.ctx, &resource.ctx))
                .map(|(h, _)| *h);
            if let Some(device_handle) = device_handle {
                if let Some(device) = self.devices.get(&device_handle) {
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    if let Some(companion) = device.dx12.as_ref() {
                        let retire_at = companion.companion_fence_high_water().max(1);
                        if let Some(slot) = resource.storage_slot {
                            companion.bindless.defer_reclaim_resource(slot, retire_at);
                        }
                        if let Some(slot) = resource.sampled_slot {
                            companion.bindless.defer_reclaim_resource(slot, retire_at);
                        }
                    }
                    let retire_at = submission_worker::submission_horizon(&device.next_timeline);
                    device.deletion_queue.lock().unwrap().push(CudaDeferredDrop::Texture {
                        retire_at,
                        resource,
                        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                        import,
                        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                        d3d12_resource,
                    });
                    return;
                }
            }
            // No device / deletion queue: drop CUDA views before import before D3D12.
            drop(resource);
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            {
                drop(import);
                drop(d3d12_resource);
            }
        }
    }

    fn texture_bindless_index(&self, texture: TextureHandle) -> Option<u32> {
        self.textures.get(&texture)?.storage_slot.or_else(|| {
            // Interpolated-only textures expose their sampled slot as the primary index.
            self.textures.get(&texture)?.sampled_slot
        })
    }

    fn texture_bindless_sampled_index(&self, texture: TextureHandle) -> Option<u32> {
        self.textures.get(&texture)?.sampled_slot
    }

    fn create_sampler(&mut self, device: DeviceHandle, desc: &SamplerDesc) -> Result<SamplerHandle> {
        self.device(device)?;
        let key = CudaSamplerKey::from_desc(desc)?;
        let slot = self.alloc_registry_slot();
        let handle = self.next_sampler;
        self.next_sampler += 1;
        self.sampler_slots.insert(slot, handle);
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        if let Some(companion) = self.device(device)?.dx12.as_ref() {
            companion.bindless.write_sampler(&companion.device, slot, desc)?;
        }
        self.samplers.insert(
            handle,
            CudaSampler {
                device,
                desc: desc.clone(),
                slot,
                key,
            },
        );
        Ok(handle)
    }

    fn destroy_sampler(&mut self, sampler: SamplerHandle) {
        if let Some(sampler) = self.samplers.remove(&sampler) {
            self.sampler_slots.remove(&sampler.slot);
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            if let Some(device) = self.devices.get(&sampler.device) {
                if let Some(companion) = device.dx12.as_ref() {
                    let retire_at = companion.companion_fence_high_water().max(1);
                    companion.bindless.defer_reclaim_sampler(sampler.slot, retire_at);
                }
            }
        }
    }

    fn sampler_bindless_index(&self, sampler: SamplerHandle) -> Option<u32> {
        self.samplers.get(&sampler).map(|s| s.slot)
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn create_surface(
        &mut self,
        device: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        display: &dyn raw_window_handle::HasDisplayHandle,
        depth_format: Option<DepthFormat>,
    ) -> Result<SurfaceHandle> {
        surface::create_surface(self, device, window, display, depth_format)
    }

    #[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
    fn create_surface(
        &mut self,
        _device: DeviceHandle,
        _window: &dyn raw_window_handle::HasWindowHandle,
        _display: &dyn raw_window_handle::HasDisplayHandle,
        _depth_format: Option<DepthFormat>,
    ) -> Result<SurfaceHandle> {
        Self::unsupported("surfaces (requires cuda+graphics+dx12 on Windows)")
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn destroy_surface(&mut self, surface: SurfaceHandle) {
        surface::destroy_surface(self, surface);
    }

    #[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
    fn destroy_surface(&mut self, _surface: SurfaceHandle) {}

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn surface_resize(&mut self, surface: SurfaceHandle, width: u32, height: u32) -> Result<()> {
        surface::surface_resize(self, surface, width, height)
    }

    #[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
    fn surface_resize(&mut self, _surface: SurfaceHandle, _width: u32, _height: u32) -> Result<()> {
        Self::unsupported("surfaces (requires cuda+graphics+dx12 on Windows)")
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn surface_size(&self, surface: SurfaceHandle) -> (u32, u32) {
        surface::surface_size(self, surface)
    }

    #[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
    fn surface_size(&self, _surface: SurfaceHandle) -> (u32, u32) {
        (0, 0)
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn surface_format(&self, surface: SurfaceHandle) -> TextureFormat {
        surface::surface_format(self, surface)
    }

    #[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
    fn surface_format(&self, _surface: SurfaceHandle) -> TextureFormat {
        TextureFormat::Bgra8UnormSrgb
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn surface_set_present_mode(&mut self, surface: SurfaceHandle, mode: crate::types::PresentMode) -> Result<()> {
        surface::surface_set_present_mode(self, surface, mode)
    }

    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue {
        let Some(context) = self.contexts.get(&ctx) else {
            return 0;
        };
        context.poll_retire_events();
        let completed = context.completed.load(Ordering::Acquire);
        let retired = context.device_retired.load(Ordering::Acquire);
        completed.max(retired)
    }

    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue {
        let Some(device) = self.devices.get(&device) else {
            return 0;
        };
        timeline::advance_device_retired(&device.event_ledger, &device.retired);
        device.retired.load(Ordering::Acquire)
    }

    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        if value == 0 {
            return Ok(());
        }
        let gpu = self.device(device)?;
        gpu.submission_worker
            .wait_submitted_if_scheduled(value, submission_worker::submission_horizon(&gpu.next_timeline))?;
        {
            let ledger = gpu.event_ledger.lock().unwrap();
            let entry = ledger
                .get(&value)
                .with_context(|| format!("CUDA: timeline value {value} has not been submitted"))?;
            match &entry.completion {
                LedgerCompletion::CudaEvent(event) => event
                    .synchronize()
                    .context("CUDA: device_wait_until event sync failed")?,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                LedgerCompletion::Dx12Fence {
                    companion,
                    value: fence_value,
                    recycle,
                } => {
                    if *fence_value == 0 {
                        anyhow::bail!("CUDA: timeline value {value} present fence not yet signaled");
                    }
                    companion
                        .cpu_wait_timeline(*fence_value, *recycle)
                        .context("CUDA: device_wait_until DX12 fence wait failed")?;
                }
            }
        }
        timeline::advance_device_retired(&gpu.event_ledger, &gpu.retired);
        for context in self.contexts.values().filter(|context| context.device == device) {
            context.poll_retire_events();
        }
        Ok(())
    }

    fn poll_signals(
        &mut self,
        ctx: ContextHandle,
        _progress: crate::timeline::TimelineValue,
    ) -> Vec<crate::signal::QueuedSignal> {
        if let Some(context) = self.contexts.get(&ctx) {
            context.poll_retire_events();
            crate::signal::drain_all_queued_signals(&context.signal_queue)
        } else {
            Vec::new()
        }
    }

    fn submit_standalone(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        self.submit_commands(ctx, commands, sync)
    }

    fn submit_graph(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        if Self::graph_has_render(commands) {
            return self.submit_graph_with_renders(ctx, commands, sync);
        }
        let gpu_commands = Self::flatten_graph_commands(commands)?;
        self.submit_commands(ctx, &gpu_commands, sync)
    }

    fn submit_graph_and_retain(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        // Render partitions are not CUDA-graph-capturable; fall back to the blocking path.
        if Self::graph_has_render(commands) {
            if self.retained.remove(&(ctx, key)).is_some() {
                self.enqueue_evict_retained(ctx, key);
            }
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            {
                self.retained
                    .insert((ctx, key), RetainedEntry::Render(commands.to_vec()));
            }
            return self.submit_graph_with_renders(ctx, commands, sync);
        }

        // Evict any previous artifact for this key before recording a replacement.
        if self.retained.remove(&(ctx, key)).is_some() {
            self.enqueue_evict_retained(ctx, key);
        }

        let gpu_commands = Self::flatten_graph_commands(commands)?;
        let effective = commands_with_sync_prologue(&gpu_commands, sync);
        let stream = Arc::clone(&self.context(ctx)?.stream);
        let ops = self.materialize_ops(&stream, &effective)?;
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        let direct_present = effective.iter().find_map(|command| {
            let GpuCommand::CopyRenderTarget { src, dst } = command else {
                return None;
            };
            surface::scratch_slot_for_texture(self, *dst).map(|(surf, _)| (*src, surf))
        });
        let stripped = pending_submit::strip_external_fence_ops(ops);
        let mut segments = pending_submit::partition_ops_into_segments(stripped);
        // Demote graph islands that touch D3D12 shared-primary memory to stream replay.
        for segment in &mut segments {
            if let pending_submit::CudaOpSegment::Graph(ops) = segment {
                if self.ops_touch_external_buffers(ops) {
                    *segment = pending_submit::CudaOpSegment::Stream(std::mem::take(ops));
                }
            }
        }
        // Coalesce adjacent stream segments created by demotion (do not reclassify).
        segments = pending_submit::coalesce_adjacent_stream_segments(segments);

        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        let all_ops = pending_submit::flatten_segment_ops(&segments);
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        let scratch_images = self.scratch_images_touched_by_ops(&all_ops);
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        let twin_dirty = self.native_twin_buffers_written_by_ops(&all_ops);

        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        if let Some((target, surface)) = direct_present {
            self.retained
                .insert((ctx, key), RetainedEntry::PresentRenderTarget { target, surface });
            // No CUDA stream work / completion event — present uses the DX12 fence ledger.
            // Sync waits on prior DX12 present fences are redundant with queue/raster ordering.
            let _ = sync;
            return Ok(self.gpu_progress(ctx));
        }

        let graph_islands = pending_submit::graph_island_count(&segments);
        // Capture when at least one graph-safe island remains. Stream segments (clears,
        // specialized kernels, present CopyTexture export + fence) stay as replayed ops
        // interleaved with island launches on the same stream.
        if graph_islands > 0 && !retained_graph::cuda_launch_blocking_active() {
            let device_handle = self.context(ctx)?.device;
            let device = self.device(device_handle)?;
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            let written_buffers = self.buffer_handles_written_by_ops(&pending_submit::collect_stream_ops(&segments));
            let segments = Arc::new(segments);
            let body = CudaSubmitBody::CaptureAndLaunch {
                key,
                segments: (*segments).clone(),
                registry: Arc::clone(&device.graph_registry),
                stats: Arc::clone(&device.graph_stats),
            };
            self.retained.insert(
                (ctx, key),
                RetainedEntry::Segmented {
                    segments,
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    scratch_images,
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    twin_dirty,
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    written_buffers,
                },
            );
            tracing::trace!(
                key,
                graph_islands,
                "CUDA: capturing retainable partition into multi-island CudaGraph program"
            );
            self.enqueue_submit(ctx, sync, body)
        } else {
            self.graph_stats.fallbacks.fetch_add(1, Ordering::Relaxed);
            let ops = pending_submit::flatten_segment_ops(&segments);
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            let written_buffers = self.buffer_handles_written_by_ops(&ops);
            self.retained.insert(
                (ctx, key),
                RetainedEntry::Ops {
                    ops: Arc::new(ops.clone()),
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    written_buffers,
                },
            );
            tracing::trace!(
                key,
                blocking = retained_graph::cuda_launch_blocking_active(),
                "CUDA: retainable partition uses pre-materialized op fallback"
            );
            self.enqueue_submit(
                ctx,
                sync,
                CudaSubmitBody::Ops {
                    ops,
                    bump_content_epochs: true,
                },
            )
        }
    }

    fn try_resubmit_retained(
        &mut self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<crate::timeline::TimelineValue>> {
        let _tz = crate::tracy_zone!("cuda.resubmit_retained");
        let entry = {
            let _tz = crate::tracy_zone!("cuda.resubmit_retained.lookup");
            self.retained.get(&(ctx, key)).cloned()
        };
        match entry {
            Some(RetainedEntry::Segmented {
                segments,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                scratch_images,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                twin_dirty,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                written_buffers,
            }) => {
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                {
                    let _tz = crate::tracy_zone!("cuda.resubmit_retained.bump_epochs");
                    self.bump_content_epochs_for_handles(&twin_dirty);
                    self.bump_content_epochs_for_handles(&written_buffers);
                }
                let device_handle = self.context(ctx)?.device;
                let device = self.device(device_handle)?;
                let body = CudaSubmitBody::LaunchRetained {
                    key,
                    segments: pending_submit::to_launch_segments(segments.as_ref()),
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    scratch_images,
                    registry: Arc::clone(&device.graph_registry),
                    stats: Arc::clone(&device.graph_stats),
                };
                tracing::trace!(key, "CUDA: launching retained multi-island CudaGraph program");
                let _tz = crate::tracy_zone!("cuda.resubmit_retained.enqueue");
                self.enqueue_submit(ctx, sync, body).map(Some)
            }
            Some(RetainedEntry::Ops {
                ops,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                written_buffers,
            }) => {
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                {
                    let _tz = crate::tracy_zone!("cuda.resubmit_retained.bump_epochs");
                    self.bump_content_epochs_for_handles(&written_buffers);
                }
                tracing::trace!(key, "CUDA: replaying retained pre-materialized ops");
                let _tz = crate::tracy_zone!("cuda.resubmit_retained.enqueue");
                self.enqueue_submit(
                    ctx,
                    sync,
                    CudaSubmitBody::Ops {
                        ops: (*ops).clone(),
                        bump_content_epochs: false,
                    },
                )
                .map(Some)
            }
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            Some(RetainedEntry::Render(mut commands)) => {
                let _tz = crate::tracy_zone!("cuda.resubmit_retained.render");
                self.retarget_surface_scratch_commands(&mut commands);
                tracing::trace!(key, "CUDA: replaying retained render partition");
                self.submit_graph_with_renders(ctx, &commands, sync).map(Some)
            }
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            Some(RetainedEntry::PresentRenderTarget { target, surface }) => {
                let _tz = crate::tracy_zone!("cuda.resubmit_retained.present_rt");
                let (resource, fence, format) = {
                    let target = self
                        .render_targets
                        .get(&target)
                        .context("CUDA/DX12: retained present target disappeared")?;
                    (target.d3d12_resource.clone(), target.last_dx12_fence, target.format)
                };
                let Some(state) = self.surfaces.get_mut(&surface) else {
                    return Ok(None);
                };
                let Some(current) = state.current_texture_handle else {
                    return Ok(None);
                };
                let image_index = state
                    .scratch
                    .iter()
                    .position(|slot| slot.as_ref().is_some_and(|s| s.texture_handle == current));
                let Some(image_index) = image_index else {
                    return Ok(None);
                };
                if let Some(slot) = state.scratch.get_mut(image_index).and_then(|s| s.as_mut()) {
                    slot.present_source = Some(surface::PresentSource::Dx12Raster {
                        resource,
                        fence,
                        format,
                    });
                    return Ok(Some(self.gpu_progress(ctx)));
                }
                Ok(None)
            }
            None => Ok(None),
        }
    }

    fn evict_retained(&mut self, ctx: ContextHandle, key: u64) {
        if self.retained.remove(&(ctx, key)).is_some() {
            self.enqueue_evict_retained(ctx, key);
        }
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn begin_frame(&mut self, surface: SurfaceHandle, ctx: ContextHandle) -> Result<(FrameToken, TextureHandle)> {
        surface::begin_frame(self, surface, ctx)
    }

    #[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
    fn begin_frame(&mut self, _surface: SurfaceHandle, _ctx: ContextHandle) -> Result<(FrameToken, TextureHandle)> {
        Self::unsupported("frames (requires cuda+graphics+dx12 on Windows)")
    }

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn submit_frame(&mut self, frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        surface::submit_frame(self, frame)
    }

    #[cfg(all(feature = "graphics", not(all(feature = "dx12", target_os = "windows"))))]
    fn submit_frame(&mut self, _frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        Self::unsupported("frames (requires cuda+graphics+dx12 on Windows)")
    }

    fn create_compute_pipeline(
        &mut self,
        device: DeviceHandle,
        compute_shader: ShaderHandle,
        _debug_name: Option<&str>,
    ) -> Result<ComputePipelineHandle> {
        let shader = self
            .shaders
            .get(&compute_shader)
            .context("CUDA: invalid shader handle")?;
        if shader.device != device {
            anyhow::bail!("CUDA: shader belongs to another device");
        }
        let shader_snapshot = shader.clone();
        let (ptx, slot_access, workgroup_size, launch_layout) = self.compile_compute_ptx(shader, compute_shader)?;
        let identity = self.load_compute_kernel(device, &ptx)?;

        // Preload float4↔Rgba8Unorm specialization when every DirectSpatial slot is float4.
        // Lazy cuModuleLoad on first specialized launch can deadlock / fault under CUDA's
        // default lazy module loading while other kernels are in flight on the stream.
        let mut variants = HashMap::new();
        let storage_elements: Vec<&str> = launch_layout
            .iter()
            .filter_map(|kind| match kind {
                CudaLaunchArgKind::StorageTexture { element } => Some(element.as_str()),
                _ => None,
            })
            .collect();
        if !storage_elements.is_empty() && storage_elements.iter().all(|element| *element == "float4") {
            let specs = vec![CudaStorageTextureSpec::Float4Rgba8Unorm; storage_elements.len()];
            let (spec_ptx, _, _, _) = self.compile_compute_ptx_with_specs(&shader_snapshot, compute_shader, &specs)?;
            let kernel = self.load_compute_kernel(device, &spec_ptx)?;
            variants.insert(specs, kernel);
        }

        let handle = self.next_compute_pipeline;
        self.next_compute_pipeline += 1;
        self.compute_pipelines.insert(
            handle,
            CudaComputePipeline {
                device,
                shader_handle: compute_shader,
                shader: shader_snapshot,
                workgroup_size,
                slot_access,
                launch_layout,
                identity,
                variants: Mutex::new(variants),
            },
        );
        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        if let Some(pipeline) = self.compute_pipelines.remove(&pipeline) {
            if let Some(device) = self.devices.get(&pipeline.device) {
                let retire_at = submission_worker::submission_horizon(&device.next_timeline);
                let mut queue = device.deletion_queue.lock().unwrap();
                queue.push(CudaDeferredDrop::Pipeline {
                    retire_at,
                    module: pipeline.identity.module,
                    function: pipeline.identity.function,
                });
                for (_specs, kernel) in pipeline.variants.into_inner().unwrap() {
                    queue.push(CudaDeferredDrop::Pipeline {
                        retire_at,
                        module: kernel.module,
                        function: kernel.function,
                    });
                }
            }
        }
    }

    fn compute_pipeline_slot_access(&self, pipeline: ComputePipelineHandle) -> Vec<Option<ResourceAccess>> {
        self.compute_pipelines
            .get(&pipeline)
            .map(|pipeline| pipeline.slot_access.clone())
            .unwrap_or_default()
    }

    fn max_bindless_slots_per_category(&self, _device: DeviceHandle, category: crate::types::ResourceCategory) -> u32 {
        match category {
            crate::types::ResourceCategory::Scattered
            | crate::types::ResourceCategory::Broadcast
            | crate::types::ResourceCategory::Texture
            | crate::types::ResourceCategory::StorageImage
            | crate::types::ResourceCategory::Sampler | crate::types::ResourceCategory::Accel => 4096,
        }
    }

    fn available_bindless_slots(&self, device: DeviceHandle, category: crate::types::ResourceCategory) -> u32 {
        let used = match category {
            crate::types::ResourceCategory::Scattered | crate::types::ResourceCategory::Broadcast => {
                self.buffers
                    .values()
                    .filter(|buffer| buffer.device == device && buffer.slot.is_some())
                    .count() as u32
            }
            crate::types::ResourceCategory::Texture => self
                .textures
                .values()
                .filter(|tex| tex.sampled_slot.is_some() && self.texture_device_is(tex, device))
                .count() as u32,
            crate::types::ResourceCategory::StorageImage => self
                .textures
                .values()
                .filter(|tex| tex.storage_slot.is_some() && self.texture_device_is(tex, device))
                .count() as u32,
            crate::types::ResourceCategory::Sampler => self
                .samplers
                .values()
                .filter(|sampler| sampler.device == device)
                .count() as u32,
            crate::types::ResourceCategory::Accel => 0,
        };
        self.max_bindless_slots_per_category(device, category)
            .saturating_sub(used)
    }

    fn max_submission_contexts(&self, _device: DeviceHandle) -> u32 {
        MAX_CUDA_SUBMISSION_CONTEXTS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{RwLockReadGuard, RwLockWriteGuard};

    #[allow(dead_code)] // held for RAII lock lifetime
    enum CudaTestGate {
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        Shared(RwLockReadGuard<'static, ()>),
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        Exclusive(RwLockWriteGuard<'static, ()>),
        #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
        None,
    }

    struct CudaTestDevice {
        device: Arc<crate::Device>,
        _gate: CudaTestGate,
    }

    impl CudaTestDevice {
        fn arc(&self) -> Arc<crate::Device> {
            Arc::clone(&self.device)
        }
    }

    impl std::ops::Deref for CudaTestDevice {
        type Target = crate::Device;

        fn deref(&self) -> &crate::Device {
            &self.device
        }
    }

    fn cuda_exclusive_guard() -> CudaTestGate {
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        {
            CudaTestGate::Exclusive(crate::test_support::cuda_lib_exclusive_gate())
        }
        #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
        {
            CudaTestGate::None
        }
    }

    const DOUBLE_SLANG: &str = r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uniform RWStructuredBuffer<uint> values, uint3 id : SV_DispatchThreadID) {
    values[id.x] = values[id.x] * 2;
}
"#;

    const DOUBLE_GOLDY_SLANG: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> values, ThreadId id) {
    values[id.x] = values[id.x] * 2;
}
"#;

    const DOUBLE_GOLDY_TWO_BUFFER_SLANG: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(BufRO<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x] * 2;
}
"#;

    fn wait_for(backend: &mut CudaBackend, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        if let Some(wait) = backend.take_timeline_submission_epoch_wait(ctx, value)? {
            wait.wait()?;
        }
        if let Some(wait) = backend.take_timeline_blocking_wait(ctx, value)? {
            wait.block()?;
        }
        backend.finish_timeline_wait(ctx, value)?;
        Ok(())
    }

    fn run_compute_dispatch_and_readback(shader_source: &str) -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA compute test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        backend.write_buffer(buffer, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))?;
        let shader = backend.create_shader_with_paths(
            device,
            shader_source,
            &[],
            &[],
            crate::types::OptimizationLevel::Default,
        )?;
        let pipeline = backend.create_compute_pipeline(device, shader, Some("double"))?;
        let slot = backend.buffer_bindless_index(buffer).context("missing registry key")?;
        let submitted = backend.submit_standalone(
            ctx,
            &[
                GpuCommand::SetPipeline(pipeline),
                GpuCommand::BindResourcesRaw {
                    indices: vec![slot],
                    user: vec![],
                    frame_table_base: 0,
                },
                GpuCommand::Dispatch {
                    label: Some("double"),
                    workgroups_x: 4,
                    workgroups_y: 1,
                    workgroups_z: 1,
                },
            ],
            None,
        )?;
        assert!(
            backend.gpu_progress(ctx) < submitted || backend.gpu_progress(ctx) == submitted,
            "progress must not exceed the submitted timeline value"
        );
        wait_for(&mut backend, ctx, submitted)?;
        assert_eq!(backend.gpu_progress(ctx), submitted);

        let readback = backend.alloc_readback_buffer(device, 16)?;
        let copied = backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: buffer,
                src_offset: 0,
                dst: readback,
                dst_offset: 0,
                size: 16,
            }],
            None,
        )?;
        wait_for(&mut backend, ctx, copied)?;
        let mut bytes = [0u8; 16];
        backend.read_readback_buffer(readback, &mut bytes)?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn slang_compute_dispatch_and_readback() -> Result<()> {
        run_compute_dispatch_and_readback(DOUBLE_SLANG)
    }

    fn run_scheme_compute_and_withdraw(shader_source: &str) -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let buffer = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let shader = crate::ShaderModule::from_slang(&device, shader_source)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double", &pipeline)
            .with_parcel(&buffer, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &buffer)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn scheme_dispatches_goldy_virtual_compute_and_withdraws() -> Result<()> {
        run_scheme_compute_and_withdraw(DOUBLE_GOLDY_SLANG)
    }

    #[test]
    fn scheme_binds_two_goldy_buffers_in_parameter_order() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let input = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let output = pool.acquire_buffer_sized::<u32>(4, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(&device, DOUBLE_GOLDY_TWO_BUFFER_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double", &pipeline)
            .with_parcel(&input, crate::NodeAccess::Read)
            .with_parcel(&output, crate::NodeAccess::Write)
            .dispatch(4, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &output)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[2, 4, 6, 8]);
        Ok(())
    }

    fn try_cuda_device() -> Result<Option<CudaTestDevice>> {
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        {
            let Some(device) = crate::test_support::shared_cuda_lib_device() else {
                eprintln!("skipping CUDA scheme test: no shared CUDA device");
                return Ok(None);
            };
            return Ok(Some(CudaTestDevice {
                device,
                _gate: CudaTestGate::Shared(crate::test_support::cuda_lib_shared_gate()),
            }));
        }
        #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
        {
            match CudaBackend::new() {
                Ok(backend) => Ok(Some(CudaTestDevice {
                    device: Arc::new(crate::Device::from_backend(Box::new(backend))?),
                    _gate: CudaTestGate::None,
                })),
                Err(error) => {
                    eprintln!("skipping CUDA scheme test: {error:#}");
                    Ok(None)
                }
            }
        }
    }

    fn try_cuda_device_with_stats() -> Result<Option<(CudaTestDevice, Arc<CudaGraphStats>)>> {
        // Quiet counter window: exclusive against other shared-device / raw-backend tests.
        let gate = cuda_exclusive_guard();
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        {
            let Some(device) = crate::test_support::shared_cuda_lib_device() else {
                eprintln!("skipping CUDA scheme test: no shared CUDA device");
                return Ok(None);
            };
            let stats = device
                .cuda_graph_stats_for_test()
                .context("CUDA: shared device missing graph stats")?;
            stats.reset();
            return Ok(Some((CudaTestDevice { device, _gate: gate }, stats)));
        }
        #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
        {
            let _ = gate;
            match CudaBackend::new() {
                Ok(backend) => {
                    let stats = backend.graph_stats();
                    stats.reset();
                    Ok(Some((
                        CudaTestDevice {
                            device: Arc::new(crate::Device::from_backend(Box::new(backend))?),
                            _gate: CudaTestGate::None,
                        },
                        stats,
                    )))
                }
                Err(error) => {
                    eprintln!("skipping CUDA scheme test: {error:#}");
                    Ok(None)
                }
            }
        }
    }

    const WITH_PARAM_UINT_SLANG: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint value, ThreadId id) {
    out[0] = value;
}
"#;

    #[test]
    fn scheme_with_param_uint_roundtrip() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let out = pool.acquire_buffer_sized::<u32>(1, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(&device, WITH_PARAM_UINT_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        const EXPECTED: u32 = 42;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_uint", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(EXPECTED)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[EXPECTED]);
        Ok(())
    }

    #[test]
    fn scheme_with_param_uint_zero() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let out = pool.acquire_buffer_with_data(&[0xDEAD_BEEFu32], BufferKind::Scattered)?;
        let shader = crate::ShaderModule::from_slang(&device, WITH_PARAM_UINT_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_zero", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(0u32)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[0]);
        Ok(())
    }

    #[test]
    fn scheme_with_param_uint_max() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let out = pool.acquire_buffer_sized::<u32>(1, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(&device, WITH_PARAM_UINT_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_max", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(u32::MAX)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[u32::MAX]);
        Ok(())
    }

    #[test]
    fn scheme_with_param_float_reinterpret() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let out = pool.acquire_buffer_sized::<u32>(1, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(
            &device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<float> out, float value, ThreadId id) {
    out[0] = value;
}
"#,
        )?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        #[allow(clippy::approx_constant)]
        let value: f32 = 3.14159;
        let bits = value.to_bits();

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_float", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(bits)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[bits]);
        Ok(())
    }

    #[test]
    fn scheme_with_param_two_scalars() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let out = pool.acquire_buffer_sized::<u32>(2, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(
            &device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint a, uint b, ThreadId id) {
    out[0] = a;
    out[1] = b;
}
"#,
        )?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        const A: u32 = 0xABCD;
        const B: u32 = 0x1234;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_two", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(A)
            .with_param(B)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[A, B]);
        Ok(())
    }

    #[test]
    fn scheme_with_param_after_two_buffers() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        const N: usize = 64;
        let input: Vec<u32> = (0..N as u32).collect();
        let inp = pool.acquire_buffer_with_data(&input, BufferKind::Scattered)?;
        let out = pool.acquire_buffer_sized::<u32>(N as u64, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(
            &device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> inp, Scattered<uint> out, uint offset, ThreadId id) {
    out[id.x] = inp[id.x] + offset;
}
"#,
        )?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        const OFFSET: u32 = 100;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_offset", &pipeline)
            .with_parcel(&inp, crate::NodeAccess::Read)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(OFFSET)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        let expected: Vec<u32> = input.iter().map(|v| v + OFFSET).collect();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), expected.as_slice());
        Ok(())
    }

    #[test]
    fn scheme_broadcast_parcel_struct_mul() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            mul: u32,
        }
        impl crate::StructuredBufferElement for Params {}
        let cfg = pool.acquire_buffer_with_data(&[Params { mul: 3 }], BufferKind::Broadcast)?;
        let values = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let shader = crate::ShaderModule::from_slang(
            &device,
            r#"
import goldy_exp;

struct Params { uint mul; };

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Params cfg, Scattered<uint> values, ThreadId id) {
    values[id.x] = values[id.x] * cfg.mul;
}
"#,
        )?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("broadcast_mul", &pipeline)
            .with_parcel(&cfg, crate::NodeAccess::Read)
            .with_parcel(&values, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &values)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[3, 6, 9, 12]);
        Ok(())
    }

    #[test]
    fn slang_emits_ptx_for_compute() -> Result<()> {
        ensure_cuda_toolkit_on_path();
        let compiler = match crate::slang::SlangCompiler::new() {
            Ok(compiler) => compiler,
            Err(error) => {
                eprintln!("skipping CUDA PTX emission test: {error:#}");
                return Ok(());
            }
        };
        let compiled = match compiler.compile_bindless_with_reflection(
            DOUBLE_SLANG,
            crate::slang::ShaderTarget::Ptx,
            &[("cs_main", crate::slang::SlangStage::Compute)],
            &[],
        ) {
            Ok(compiled) => compiled,
            Err(error) => {
                eprintln!("skipping CUDA PTX emission test (Slang/NVRTC): {error:#}");
                return Ok(());
            }
        };
        let ptx = compiled.shader.as_str().context("expected text PTX")?;
        assert!(
            ptx.contains(".entry") || ptx.contains("cs_main"),
            "Slang output did not look like PTX:\n{ptx}"
        );
        Ok(())
    }

    #[test]
    fn ptx_cache_key_differs_from_wgsl() {
        use crate::shader_cache::compile_cache_key;
        use crate::slang::{ffi::SlangStage, ShaderTarget};
        use crate::types::OptimizationLevel;

        let src = "void cs_main() {}";
        let eps = [("cs_main", SlangStage::Compute)];
        let ptx = compile_cache_key(src, ShaderTarget::Ptx, &eps, &[], &[], &[], OptimizationLevel::Default);
        let wgsl = compile_cache_key(src, ShaderTarget::Wgsl, &eps, &[], &[], &[], OptimizationLevel::Default);
        assert_ne!(ptx, wgsl);
    }

    // ─── M2: multi-node command coverage ───────────────────────────────────

    const M2_FILL_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, uint value, ThreadId id) {
    data[id.x] = value;
}
"#;

    const M2_DOUBLE_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x] * 2;
}
"#;

    const M2_ADD_TEN_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + 10;
}
"#;

    const M2_IN_PLACE_DOUBLE_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] * 2;
}
"#;

    const M2_COPY_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x];
}
"#;

    const M2_SUM_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> a, Scattered<uint> b, Scattered<uint> out, ThreadId id) {
    out[id.x] = a[id.x] + b[id.x];
}
"#;

    const M2_FILL_INDEX_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = id.x;
}
"#;

    #[test]
    fn scheme_same_pipeline_batch_two_fills() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let a = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let b = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(&device, M2_FILL_SHADER)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("fill_a", &pipeline)
            .with_parcel(&a, crate::NodeAccess::Write)
            .with_param(7u32)
            .dispatch(1, 1, 1);
        scheme
            .node("fill_b", &pipeline)
            .with_parcel(&b, crate::NodeAccess::Write)
            .with_param(9u32)
            .dispatch(1, 1, 1);
        let withdraw_a = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &a)?;
        let withdraw_b = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &b)?;
        let mut submission = scheme.submit()?;
        let bytes_a = withdraw_a.claim(&mut submission)?.consume()?;
        let bytes_b = withdraw_b.claim(&mut submission)?.consume()?;
        assert!(bytemuck::cast_slice::<u8, u32>(&bytes_a).iter().all(|&v| v == 7));
        assert!(bytemuck::cast_slice::<u8, u32>(&bytes_b).iter().all(|&v| v == 9));
        Ok(())
    }

    #[test]
    fn scheme_graph_linear_chain() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let src = pool.acquire_buffer_with_data(&(0..64u32).collect::<Vec<_>>(), BufferKind::Scattered)?;
        let dst = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let double_pipe =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_DOUBLE_SHADER)?)?;
        let add_pipe =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_ADD_TEN_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double", &double_pipe)
            .with_parcel(&src, crate::NodeAccess::Read)
            .with_parcel(&dst, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("add_ten", &add_pipe)
            .with_parcel(&dst, crate::NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &dst)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        for (i, &val) in bytemuck::cast_slice::<u8, u32>(&bytes).iter().enumerate() {
            assert_eq!(val, (i as u32) * 2 + 10, "element {i}");
        }
        Ok(())
    }

    #[test]
    fn scheme_graph_independent_dispatches() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let a = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let b = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        // Distinct pipeline objects so analysis emits two Dispatch commands, not DispatchBatch.
        let fill_a = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_FILL_SHADER)?)?;
        let fill_b = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_FILL_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("fill_a", &fill_a)
            .with_parcel(&a, crate::NodeAccess::Write)
            .with_param(42u32)
            .dispatch(1, 1, 1);
        scheme
            .node("fill_b", &fill_b)
            .with_parcel(&b, crate::NodeAccess::Write)
            .with_param(99u32)
            .dispatch(1, 1, 1);
        let withdraw_a = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &a)?;
        let withdraw_b = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &b)?;
        let mut submission = scheme.submit()?;
        assert!(
            bytemuck::cast_slice::<u8, u32>(&withdraw_a.claim(&mut submission)?.consume()?)
                .iter()
                .all(|&v| v == 42)
        );
        assert!(
            bytemuck::cast_slice::<u8, u32>(&withdraw_b.claim(&mut submission)?.consume()?)
                .iter()
                .all(|&v| v == 99)
        );
        Ok(())
    }

    #[test]
    fn scheme_graph_diamond_dependency() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let src = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let y = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let z = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let out = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let fill = crate::ComputePipeline::new(
            &device,
            &crate::ShaderModule::from_slang(&device, M2_FILL_INDEX_SHADER)?,
        )?;
        let double =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_DOUBLE_SHADER)?)?;
        let sum = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_SUM_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("fill_src", &fill)
            .with_parcel(&src, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("double_to_y", &double)
            .with_parcel(&src, crate::NodeAccess::Read)
            .with_parcel(&y, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("double_to_z", &double)
            .with_parcel(&src, crate::NodeAccess::Read)
            .with_parcel(&z, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("sum_yz", &sum)
            .with_parcel(&y, crate::NodeAccess::Read)
            .with_parcel(&z, crate::NodeAccess::Read)
            .with_parcel(&out, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        for (i, &val) in bytemuck::cast_slice::<u8, u32>(&bytes).iter().enumerate() {
            assert_eq!(val, (i as u32) * 4, "element {i}");
        }
        Ok(())
    }

    #[test]
    fn scheme_clear_then_dispatch() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let buf = pool.acquire_buffer_with_data(&vec![0xDEAD_BEEFu32; 64], BufferKind::Scattered)?;
        let pipe = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_ADD_TEN_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme.clear_parcel(&buf, 0, 64 * 4)?;
        scheme
            .node("add_ten", &pipe)
            .with_parcel(&buf, crate::NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &buf)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert!(bytemuck::cast_slice::<u8, u32>(&bytes).iter().all(|&v| v == 10));
        Ok(())
    }

    #[test]
    fn scheme_write_copy_dispatch_chain() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let src = pool.acquire_buffer_with_data(&(0..64u32).collect::<Vec<_>>(), BufferKind::Scattered)?;
        let mid = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let dst = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let copy = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_COPY_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme.copy_buffer_parcel(&src, 0, &mid, 0, 64 * 4)?;
        scheme
            .node("copy_mid_to_dst", &copy)
            .with_parcel(&mid, crate::NodeAccess::Read)
            .with_parcel(&dst, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &dst)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        for (i, &val) in bytemuck::cast_slice::<u8, u32>(&bytes).iter().enumerate() {
            assert_eq!(val, i as u32, "element {i}");
        }
        Ok(())
    }

    #[test]
    fn scheme_buffer_view_copy_and_isolation() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        const N: usize = 64;
        let src: Vec<u32> = (1..=N as u32).collect();
        let dst = vec![0u32; N];
        let cells = pool.acquire_record([
            crate::ordinal(crate::Init::data(&src)),
            crate::ordinal(crate::Init::data(&dst)),
        ])?;
        let copy = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_COPY_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("copy_fields", &copy)
            .with_parcel(&cells[0], crate::NodeAccess::Read)
            .with_parcel(&cells[1], crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &cells[1])?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), src.as_slice());

        // Isolation: doubling one field must leave the sibling untouched.
        let sentinel = vec![100u32; N];
        let work: Vec<u32> = (1..=N as u32).collect();
        let cells = pool.acquire_record([
            crate::ordinal(crate::Init::data(&sentinel)),
            crate::ordinal(crate::Init::data(&work)),
        ])?;
        let double = crate::ComputePipeline::new(
            &device,
            &crate::ShaderModule::from_slang(&device, M2_IN_PLACE_DOUBLE_SHADER)?,
        )?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double_work", &double)
            .with_parcel(&cells[1], crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant_sentinel = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &cells[0])?;
        let grant_work = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &cells[1])?;
        let mut submission = scheme.submit()?;
        assert!(
            bytemuck::cast_slice::<u8, u32>(&grant_sentinel.claim(&mut submission)?.consume()?)
                .iter()
                .all(|&v| v == 100)
        );
        for (i, &val) in bytemuck::cast_slice::<u8, u32>(&grant_work.claim(&mut submission)?.consume()?)
            .iter()
            .enumerate()
        {
            assert_eq!(val, (i as u32 + 1) * 2, "work[{i}]");
        }
        Ok(())
    }

    #[test]
    fn adapter_capabilities_publish_cuda_format_limits() -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA capabilities test: {error:#}");
                return Ok(());
            }
        };
        let caps = backend.adapter_capabilities(0);
        assert_eq!(caps.preferred_surface_format, TextureFormat::Rgba8Unorm);
        assert_eq!(caps.supported_surface_formats, vec![TextureFormat::Rgba8Unorm]);
        assert_eq!(caps.preferred_render_target_format, TextureFormat::Rgba8Unorm);
        assert_eq!(
            caps.supported_render_target_formats,
            vec![TextureFormat::Rgba32Float, TextureFormat::Rgba8Unorm]
        );
        assert!(
            !caps
                .supported_surface_formats
                .iter()
                .any(|f| matches!(f, TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb)),
            "CUDA must not advertise BGRA surface formats"
        );
        assert!(
            !caps
                .supported_render_target_formats
                .iter()
                .any(|f| matches!(f, TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb)),
            "CUDA must not advertise BGRA render-target formats"
        );
        assert!(
            !caps.ray_query && !caps.ray_tracing_pipelines && !caps.mesh_shaders && !caps.amplification_shaders,
            "CUDA does not advertise RT or mesh shaders"
        );
        Ok(())
    }

    #[test]
    fn overlapping_self_copy_is_memmove_safe() -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA overlapping copy test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            32,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        // [1,2,3,4,5,6,7,8] → copy first 16 bytes onto offset 8 (overlap).
        backend.write_buffer(buffer, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4, 5, 6, 7, 8]))?;
        backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: buffer,
                src_offset: 0,
                dst: buffer,
                dst_offset: 8,
                size: 16,
            }],
            None,
        )?;
        let readback = backend.alloc_readback_buffer(device, 32)?;
        backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: buffer,
                src_offset: 0,
                dst: readback,
                dst_offset: 0,
                size: 32,
            }],
            None,
        )?;
        let mut bytes = [0u8; 32];
        backend.read_readback_buffer(readback, &mut bytes)?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[1, 2, 1, 2, 3, 4, 7, 8]);
        Ok(())
    }

    #[test]
    fn resize_buffer_preserves_contents_on_device() -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA resize test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        backend.write_buffer(buffer, 0, bytemuck::cast_slice(&[10u32, 20, 30, 40]))?;
        backend.resize_buffer(device, buffer, 32, true)?;
        assert_eq!(backend.buffer_size(buffer), 32);
        assert!(backend.buffer_capacity(buffer) >= 32);

        let readback = backend.alloc_readback_buffer(device, 32)?;
        backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: buffer,
                src_offset: 0,
                dst: readback,
                dst_offset: 0,
                size: 32,
            }],
            None,
        )?;
        let mut bytes = [0u8; 32];
        backend.read_readback_buffer(readback, &mut bytes)?;
        let words = bytemuck::cast_slice::<u8, u32>(&bytes);
        assert_eq!(&words[..4], &[10, 20, 30, 40]);
        // Newly grown tail is zero-filled by alloc_zeros.
        assert_eq!(&words[4..], &[0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn async_submit_wait_until_advances_progress() -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA async timeline test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        backend.write_buffer(buffer, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))?;
        let shader = backend.create_shader_with_paths(
            device,
            DOUBLE_SLANG,
            &[],
            &[],
            crate::types::OptimizationLevel::Default,
        )?;
        let pipeline = backend.create_compute_pipeline(device, shader, Some("double"))?;
        let slot = backend.buffer_bindless_index(buffer).context("missing registry key")?;
        let submitted = backend.submit_standalone(
            ctx,
            &[
                GpuCommand::SetPipeline(pipeline),
                GpuCommand::BindResourcesRaw {
                    indices: vec![slot],
                    user: vec![],
                    frame_table_base: 0,
                },
                GpuCommand::Dispatch {
                    label: Some("double"),
                    workgroups_x: 4,
                    workgroups_y: 1,
                    workgroups_z: 1,
                },
            ],
            None,
        )?;
        // Submit must return a timeline value without requiring GPU completion first.
        assert!(submitted >= 1);
        wait_for(&mut backend, ctx, submitted)?;
        assert_eq!(backend.gpu_progress(ctx), submitted);
        assert!(backend.device_timeline_retired(device) >= submitted);
        Ok(())
    }

    #[test]
    fn two_contexts_submit_and_complete_independently() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx_a = device.create_context()?;
        let ctx_b = device.create_context()?;
        let shader = crate::ShaderModule::from_slang(&device, DOUBLE_GOLDY_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let buf_a = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let buf_b = pool.acquire_buffer_with_data(&[10u32, 20, 30, 40], BufferKind::Scattered)?;

        let mut scheme_a = crate::Scheme::new(&ctx_a);
        scheme_a
            .node("a", &pipeline)
            .with_parcel(&buf_a, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        let withdraw_a = crate::MemoryExchange::new(&ctx_a).bind_withdraw(&mut scheme_a, &buf_a)?;
        let mut submission_a = scheme_a.submit()?;

        let mut scheme_b = crate::Scheme::new(&ctx_b);
        scheme_b
            .node("b", &pipeline)
            .with_parcel(&buf_b, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        let withdraw_b = crate::MemoryExchange::new(&ctx_b).bind_withdraw(&mut scheme_b, &buf_b)?;
        let mut submission_b = scheme_b.submit()?;

        let bytes_a = withdraw_a.claim(&mut submission_a)?.consume()?;
        let bytes_b = withdraw_b.claim(&mut submission_b)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes_a), &[2, 4, 6, 8]);
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes_b), &[20, 40, 60, 80]);
        Ok(())
    }

    #[test]
    fn scheme_clear_parcel_partial_preserves_edges() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        const N: usize = 64;
        let mut init = Vec::with_capacity(N);
        for i in 0..N {
            if i < 16 {
                init.push(0xAAAA_AAAAu32);
            } else if i < 48 {
                init.push(0xBBBB_BBBBu32);
            } else {
                init.push(0xCCCC_CCCCu32);
            }
        }
        let mut pool = crate::RetainedPool::new(device.arc());
        let buf = pool.acquire_buffer_with_data(&init, BufferKind::Scattered)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme.clear_parcel(&buf, 16 * 4, 32 * 4)?;
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &buf)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        let words = bytemuck::cast_slice::<u8, u32>(&bytes);
        assert!(words[..16].iter().all(|&v| v == 0xAAAA_AAAA));
        assert!(words[16..48].iter().all(|&v| v == 0));
        assert!(words[48..].iter().all(|&v| v == 0xCCCC_CCCC));
        Ok(())
    }

    #[test]
    fn validated_ptx_load_rejects_invalid_ptx_with_jit_log() -> Result<()> {
        ensure_cuda_toolkit_on_path();
        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("skipping CUDA JIT validation test: {error:#}");
                return Ok(());
            }
        };
        let err = match load_ptx_module_validated(&ctx, "this is not valid PTX !!!") {
            Ok(()) => {
                panic!("expected invalid PTX to fail JIT validation");
            }
            Err(error) => format!("{error:#}"),
        };
        assert!(
            err.contains("PTX JIT failed") || err.contains("error log") || err.contains("JIT"),
            "expected JIT failure diagnostics, got:\n{err}"
        );
        // Error log from the driver should be non-trivial for garbage PTX.
        assert!(err.len() > 32, "expected non-empty JIT diagnostic text, got:\n{err}");
        Ok(())
    }

    #[test]
    fn launch_config_validation_rejects_oversize_grid() {
        let limits = CudaDeviceLimits {
            max_grid_dim_x: 1024,
            max_grid_dim_y: 1024,
            max_grid_dim_z: 64,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 48 * 1024,
        };
        let err = validate_launch_config_unchecked(&limits, 1024, (u32::MAX, 1, 1), [1, 1, 1], 0, Some("too_wide"))
            .expect_err("oversize grid must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("too_wide") && msg.contains("exceeds device max"),
            "unexpected error: {msg}"
        );

        let err = validate_launch_config_unchecked(&limits, 256, (1, 1, 1), [512, 1, 1], 0, Some("fat_block"))
            .expect_err("oversize block must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("fat_block") && msg.contains("exceeds function max"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn goldy_validation_api_gate_compiles_for_cuda() {
        // Ensures `feature = "cuda"` keeps `goldy_validation_enabled` linked.
        let _ = crate::backend::goldy_validation_enabled();
    }

    #[test]
    fn retained_partition_captures_once_then_graph_launches() -> Result<()> {
        let Some((device, stats)) = try_cuda_device_with_stats()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let buffer = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let pipeline =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, DOUBLE_GOLDY_SLANG)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double", &pipeline)
            .with_parcel(&buffer, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);

        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &buffer)?;
        let mut submission = scheme.submit()?;
        let bytes1 = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes1), &[2, 4, 6, 8]);

        let after_first = stats.snapshot();
        assert!(
            after_first.captures >= 1,
            "first retainable submit should capture at least one graph: {after_first:?}"
        );
        assert!(
            after_first.launches >= 1,
            "first retainable submit should launch the captured graph: {after_first:?}"
        );
        let captures_after_first = after_first.captures;
        let launches_after_first = after_first.launches;

        // Stable resubmit without rebinding withdraw (would dirty IR).
        let mut submission = scheme.submit()?;
        let bytes2 = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes2), &[4, 8, 12, 16]);

        let after_second = stats.snapshot();
        assert_eq!(
            after_second.captures, captures_after_first,
            "stable resubmit must not recapture: first={after_first:?} second={after_second:?}"
        );
        assert!(
            after_second.launches > launches_after_first,
            "stable resubmit must graph-launch: first={after_first:?} second={after_second:?}"
        );
        assert_eq!(
            scheme.replay_stats().records,
            1,
            "stable resubmit should not re-record the partition"
        );
        Ok(())
    }

    #[test]
    fn scalar_param_change_recaptures_distinct_partition() -> Result<()> {
        let Some((device, stats)) = try_cuda_device_with_stats()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let out = pool.acquire_buffer_sized::<u32>(1, BufferKind::Scattered, BufferFlags::empty())?;
        let pipeline = crate::ComputePipeline::new(
            &device,
            &crate::ShaderModule::from_slang(&device, WITH_PARAM_UINT_SLANG)?,
        )?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("fill", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(7u32)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        assert_eq!(
            bytemuck::cast_slice::<u8, u32>(&withdraw.claim(&mut submission)?.consume()?)[0],
            7
        );
        let captures_after_first = stats.snapshot().captures;
        assert!(captures_after_first >= 1);

        // Mutate scalar param → dirty scheme → new partition fingerprint / recapture.
        scheme
            .node("fill", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(9u32)
            .dispatch(1, 1, 1);
        // Rebuilding the node above may not clear prior nodes; use a fresh scheme instead.
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("fill", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(9u32)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        assert_eq!(
            bytemuck::cast_slice::<u8, u32>(&withdraw.claim(&mut submission)?.consume()?)[0],
            9
        );
        assert!(
            stats.snapshot().captures > captures_after_first,
            "distinct scalar binding should produce a new capture"
        );
        Ok(())
    }

    #[test]
    fn cpu_writable_pinned_staging_copy_roundtrip() -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA pinned staging test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let staging = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::CPU_WRITABLE | BufferFlags::COPY_SRC,
        )?;
        let dst = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        backend.write_buffer(staging, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))?;
        let tv = backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: staging,
                src_offset: 0,
                dst,
                dst_offset: 0,
                size: 16,
            }],
            None,
        )?;
        wait_for(&mut backend, ctx, tv)?;
        let readback = backend.alloc_readback_buffer(device, 16)?;
        let tv = backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: dst,
                src_offset: 0,
                dst: readback,
                dst_offset: 0,
                size: 16,
            }],
            None,
        )?;
        wait_for(&mut backend, ctx, tv)?;
        let mut bytes = vec![0u8; 16];
        backend.read_readback_buffer(readback, &mut bytes)?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[1, 2, 3, 4]);

        backend.write_buffer(staging, 0, bytemuck::cast_slice(&[5u32, 6, 7, 8]))?;
        let tv = backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: staging,
                src_offset: 0,
                dst,
                dst_offset: 0,
                size: 16,
            }],
            None,
        )?;
        wait_for(&mut backend, ctx, tv)?;
        let tv = backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: dst,
                src_offset: 0,
                dst: readback,
                dst_offset: 0,
                size: 16,
            }],
            None,
        )?;
        wait_for(&mut backend, ctx, tv)?;
        backend.read_readback_buffer(readback, &mut bytes)?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[5, 6, 7, 8]);
        Ok(())
    }

    #[test]
    fn cpu_writable_copy_captures_once_then_graph_launches() -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA pinned capture test: {error:#}");
                return Ok(());
            }
        };
        let stats = backend.graph_stats();
        stats.reset();
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let staging = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::CPU_WRITABLE | BufferFlags::COPY_SRC,
        )?;
        let dst = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        backend.write_buffer(staging, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))?;
        const KEY: u64 = 0xC0_91_ED;
        let commands = [GraphCommand::Compute(GpuCommand::CopyBuffer {
            src: staging,
            src_offset: 0,
            dst,
            dst_offset: 0,
            size: 16,
        })];
        let tv = backend.submit_graph_and_retain(ctx, &commands, KEY, None)?;
        wait_for(&mut backend, ctx, tv)?;
        let after_first = stats.snapshot();
        assert!(
            after_first.captures >= 1,
            "pinned WriteFromHost copy must capture: {after_first:?}"
        );
        assert_eq!(
            after_first.fallbacks, 0,
            "pinned host copy must not use Ops fallback: {after_first:?}"
        );

        backend.write_buffer(staging, 0, bytemuck::cast_slice(&[9u32, 8, 7, 6]))?;
        let tv = backend
            .try_resubmit_retained(ctx, KEY, None)?
            .context("expected retained upload graph")?;
        wait_for(&mut backend, ctx, tv)?;
        let after_second = stats.snapshot();
        assert_eq!(
            after_second.captures, after_first.captures,
            "resubmit must not recapture: first={after_first:?} second={after_second:?}"
        );
        assert!(
            after_second.launches > after_first.launches,
            "resubmit must graph-launch: first={after_first:?} second={after_second:?}"
        );

        let readback = backend.alloc_readback_buffer(device, 16)?;
        let tv = backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: dst,
                src_offset: 0,
                dst: readback,
                dst_offset: 0,
                size: 16,
            }],
            None,
        )?;
        wait_for(&mut backend, ctx, tv)?;
        let mut bytes = vec![0u8; 16];
        backend.read_readback_buffer(readback, &mut bytes)?;
        assert_eq!(
            bytemuck::cast_slice::<u8, u32>(&bytes),
            &[9, 8, 7, 6],
            "graph relaunch must observe CPU writes into pinned staging"
        );
        Ok(())
    }

    #[test]
    fn cpu_writable_texture_copy_captures_once_then_graph_launches() -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA pinned texture capture test: {error:#}");
                return Ok(());
            }
        };
        let stats = backend.graph_stats();
        stats.reset();
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let staging = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::CPU_WRITABLE | BufferFlags::COPY_SRC,
        )?;
        let tex = backend.create_texture(
            device,
            2,
            2,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
        )?;
        let imported = backend.textures.get(&tex).is_some_and(|t| t.is_imported());
        backend.write_buffer(staging, 0, &[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])?;
        const KEY: u64 = 0xC0_91_7E;
        let commands = [GraphCommand::Compute(GpuCommand::CopyBufferToTexture {
            src: staging,
            src_offset: 0,
            src_row_pitch: 0,
            dst: tex,
            x: 0,
            y: 0,
            width: 2,
            height: 2,
        })];
        let tv = backend.submit_graph_and_retain(ctx, &commands, KEY, None)?;
        wait_for(&mut backend, ctx, tv)?;
        let after_first = stats.snapshot();
        if imported {
            assert_eq!(
                after_first.captures, 0,
                "imported-surface texture upload must not capture: {after_first:?}"
            );
        } else {
            assert!(
                after_first.captures >= 1,
                "pinned WriteTextureFromHost must capture: {after_first:?}"
            );
            assert_eq!(
                after_first.fallbacks, 0,
                "pinned texture upload must not use Ops fallback: {after_first:?}"
            );
        }

        backend.write_buffer(staging, 0, &[16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])?;
        let tv = backend
            .try_resubmit_retained(ctx, KEY, None)?
            .context("expected retained texture upload")?;
        wait_for(&mut backend, ctx, tv)?;
        let after_second = stats.snapshot();
        assert_eq!(
            after_second.captures, after_first.captures,
            "resubmit must not recapture: first={after_first:?} second={after_second:?}"
        );
        if !imported {
            assert!(
                after_second.launches > after_first.launches,
                "resubmit must graph-launch: first={after_first:?} second={after_second:?}"
            );
        }

        let readback = backend.alloc_readback_buffer(device, 16)?;
        let layout = crate::TextureCopyFootprint {
            width: 2,
            height: 2,
            format: TextureFormat::Rgba8Unorm,
            logical_bytes: 16,
            staging_bytes: 16,
            row_pitch: 8,
            footprint_offset: 0,
        };
        let tv = backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyTextureToReadback {
                src: tex,
                dst: readback,
                layout,
            }],
            None,
        )?;
        wait_for(&mut backend, ctx, tv)?;
        let mut bytes = vec![0u8; 16];
        backend.read_readback_buffer(readback, &mut bytes)?;
        assert_eq!(
            bytes.as_slice(),
            &[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "relaunch must observe CPU writes into pinned staging"
        );
        Ok(())
    }

    #[test]
    fn upload_write_commands_use_command_fallback_not_graph_capture() -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA fallback retention test: {error:#}");
                return Ok(());
            }
        };
        let stats = backend.graph_stats();
        stats.reset();
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        let commands = [GraphCommand::Compute(GpuCommand::WriteBuffer {
            buffer,
            offset: 0,
            data: Arc::<[u8]>::from(bytemuck::cast_slice(&[1u32, 2, 3, 4]).to_vec()),
        })];
        let tv = backend.submit_graph_and_retain(ctx, &commands, 0xDEAD_F001, None)?;
        wait_for(&mut backend, ctx, tv)?;
        let snap = stats.snapshot();
        assert_eq!(snap.captures, 0, "WriteBuffer must not capture: {snap:?}");
        assert!(
            snap.fallbacks >= 1,
            "WriteBuffer retain path should count as fallback: {snap:?}"
        );

        // Resubmit still works via command replay.
        let tv = backend
            .try_resubmit_retained(ctx, 0xDEAD_F001, None)?
            .context("expected retained fallback entry")?;
        wait_for(&mut backend, ctx, tv)?;
        let replay = stats.snapshot();
        assert_eq!(
            replay.fallbacks, snap.fallbacks,
            "pre-materialized fallback replay must not count another fallback"
        );
        assert_eq!(
            replay.rematerialize_fallbacks, 0,
            "pre-materialized fallback replay must not rematerialize"
        );
        Ok(())
    }

    #[test]
    fn destroy_context_evicts_retained_graphs() -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA eviction test: {error:#}");
                return Ok(());
            }
        };
        let stats = backend.graph_stats();
        stats.reset();
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        backend.write_buffer(buffer, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))?;
        let shader = backend.create_shader_with_paths(
            device,
            DOUBLE_SLANG,
            &[],
            &[],
            crate::types::OptimizationLevel::Default,
        )?;
        let pipeline = backend.create_compute_pipeline(device, shader, Some("double"))?;
        let slot = backend.buffer_bindless_index(buffer).context("missing registry key")?;
        let commands = [
            GraphCommand::Compute(GpuCommand::SetPipeline(pipeline)),
            GraphCommand::Compute(GpuCommand::BindResourcesRaw {
                indices: vec![slot],
                user: vec![],
                frame_table_base: 0,
            }),
            GraphCommand::Compute(GpuCommand::Dispatch {
                label: Some("double"),
                workgroups_x: 4,
                workgroups_y: 1,
                workgroups_z: 1,
            }),
        ];
        const KEY: u64 = 0xE11C7;
        let tv = backend.submit_graph_and_retain(ctx, &commands, KEY, None)?;
        wait_for(&mut backend, ctx, tv)?;
        assert!(stats.snapshot().captures >= 1);

        backend.evict_retained(ctx, KEY);
        backend.device_wait_idle(device)?;
        assert!(
            stats.snapshot().evictions >= 1,
            "evict_retained + idle should destroy the graph: {:?}",
            stats.snapshot()
        );

        // Resubmit after eviction must miss and return None.
        assert!(backend.try_resubmit_retained(ctx, KEY, None)?.is_none());
        Ok(())
    }

    #[test]
    fn ops_are_graph_safe_rejects_empty() {
        assert!(!pending_submit::ops_are_graph_safe(&[]));
    }

    #[test]
    fn indirect_scheme_captures_and_relaunches_with_gpu_shape() -> Result<()> {
        let Some((device, stats)) = try_cuda_device_with_stats()? else {
            return Ok(());
        };
        stats.reset();
        let ctx = device.create_context()?;
        let write_shape_slang = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<DispatchShape> shape, ThreadId id) {
    DispatchShape s;
    s.x = 4;
    s.y = 1;
    s.z = 1;
    shape[0] = s;
}
"#;
        let write_pipe =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, write_shape_slang)?)?;
        let work_pipe =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, DOUBLE_GOLDY_SLANG)?)?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let shape =
            pool.acquire_buffer_sized::<crate::types::DispatchShape>(1, BufferKind::Scattered, BufferFlags::empty())?;
        let work = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("write_shape", &write_pipe)
            .with_parcel(&shape, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("work", &work_pipe)
            .with_parcel(&work, crate::NodeAccess::ReadWrite)
            .dispatch_shape_parcel(&*shape)?;

        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &work)?;
        let mut submission = scheme.submit()?;
        let bytes1 = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes1), &[2, 4, 6, 8]);

        let after_first = stats.snapshot();
        assert!(
            after_first.captures >= 1,
            "indirect launch-only partition should capture: {after_first:?}"
        );
        let captures_after_first = after_first.captures;
        let launches_after_first = after_first.launches;

        let mut submission = scheme.submit()?;
        let bytes2 = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes2), &[4, 8, 12, 16]);
        let after_second = stats.snapshot();
        assert_eq!(
            after_second.captures, captures_after_first,
            "stable indirect resubmit must not recapture: {after_first:?} vs {after_second:?}"
        );
        assert!(
            after_second.launches > launches_after_first,
            "stable indirect resubmit must graph-launch: {after_first:?} vs {after_second:?}"
        );
        assert_eq!(scheme.replay_stats().records, 1);
        #[cfg(not(feature = "metal"))]
        assert_eq!(scheme.replay_stats().resubmit_hits, 1);
        Ok(())
    }

    #[test]
    fn indirect_with_clear_captures_graph_islands() -> Result<()> {
        let Some((device, stats)) = try_cuda_device_with_stats()? else {
            return Ok(());
        };
        stats.reset();
        let ctx = device.create_context()?;
        let write_shape_slang = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<DispatchShape> shape, ThreadId id) {
    DispatchShape s;
    s.x = 4;
    s.y = 1;
    s.z = 1;
    shape[0] = s;
}
"#;
        let write_pipe =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, write_shape_slang)?)?;
        let work_pipe =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, DOUBLE_GOLDY_SLANG)?)?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let shape =
            pool.acquire_buffer_sized::<crate::types::DispatchShape>(1, BufferKind::Scattered, BufferFlags::empty())?;
        let work = pool.acquire_buffer_with_data(&[5u32, 6, 7, 8], BufferKind::Scattered)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme.clear_parcel(&work, 0, 0)?;
        scheme
            .node("write_shape", &write_pipe)
            .with_parcel(&shape, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("work", &work_pipe)
            .with_parcel(&work, crate::NodeAccess::ReadWrite)
            .dispatch_shape_parcel(&*shape)?;

        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &work)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[0, 0, 0, 0]);
        let snap = stats.snapshot();
        assert!(snap.captures >= 1, "clear+indirect launches must capture: {snap:?}");
        assert_eq!(
            snap.fallbacks, 0,
            "clear+indirect should use multi-island capture, not full Ops fallback: {snap:?}"
        );

        // Stable resubmit: no recapture, graph relaunches.
        let captures_after_first = snap.captures;
        let launches_after_first = snap.launches;
        let mut submission = scheme.submit()?;
        let bytes2 = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes2), &[0, 0, 0, 0]);
        let after = stats.snapshot();
        assert_eq!(
            after.captures, captures_after_first,
            "stable multi-island resubmit must not recapture: {after:?}"
        );
        assert!(
            after.launches > launches_after_first,
            "stable multi-island resubmit must graph-launch: {after:?}"
        );
        Ok(())
    }

    #[test]
    fn multi_island_launch_clear_launch_captures_two_islands() -> Result<()> {
        let Some((device, stats)) = try_cuda_device_with_stats()? else {
            return Ok(());
        };
        stats.reset();
        let ctx = device.create_context()?;
        let pipeline =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, DOUBLE_GOLDY_SLANG)?)?;
        let mut pool = crate::RetainedPool::new(device.arc());
        let a = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let b = pool.acquire_buffer_with_data(&[10u32, 20, 30, 40], BufferKind::Scattered)?;

        // Launch(a) → Clear(b) → Launch(b): all graph-safe, one captured island.
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double_a", &pipeline)
            .with_parcel(&a, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        scheme.clear_parcel(&b, 0, 0)?;
        scheme
            .node("double_b", &pipeline)
            .with_parcel(&b, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);

        let withdraw_a = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &a)?;
        let withdraw_b = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &b)?;
        let mut submission = scheme.submit()?;
        let bytes_a = withdraw_a.claim(&mut submission)?.consume()?;
        let bytes_b = withdraw_b.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes_a), &[2, 4, 6, 8]);
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes_b), &[0, 0, 0, 0]);

        let after_first = stats.snapshot();
        assert!(
            after_first.captures >= 1,
            "launch→clear→launch must capture a graph island: {after_first:?}"
        );
        assert_eq!(
            after_first.fallbacks, 0,
            "multi-island path must not fall back to Ops: {after_first:?}"
        );
        let captures_after_first = after_first.captures;
        let launches_after_first = after_first.launches;

        let mut submission = scheme.submit()?;
        let bytes_a2 = withdraw_a.claim(&mut submission)?.consume()?;
        let bytes_b2 = withdraw_b.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes_a2), &[4, 8, 12, 16]);
        // Cleared every resubmit, then doubled from zeros → still zeros.
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes_b2), &[0, 0, 0, 0]);

        let after_second = stats.snapshot();
        assert_eq!(
            after_second.captures, captures_after_first,
            "stable multi-island resubmit must not recapture: {after_second:?}"
        );
        assert!(
            after_second.launches >= launches_after_first + 1,
            "stable resubmit must relaunch the captured island: first={after_first:?} second={after_second:?}"
        );
        assert_eq!(scheme.replay_stats().records, 1);
        Ok(())
    }

    /// Regression for premature DX12↔CUDA import teardown on texture replace.
    ///
    /// `destroy_texture` must keep the external-memory import alive with the deferred
    /// CUDA views. Dropping the import first left a borrowed `CUarray` dangling; the
    /// next shared-texture create then failed with
    /// `cuMipmappedArrayGetLevel` / `CUDA_ERROR_ILLEGAL_ADDRESS` (ekrano mask-atlas
    /// 1×1 → 64×64 resize).
    #[test]
    fn shared_texture_replace_keeps_import_alive_until_views_drop() -> Result<()> {
        let _exclusive = cuda_exclusive_guard();
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA shared texture replace test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;

        let small = backend.create_texture(
            device,
            1,
            1,
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
        )?;
        backend.write_texture(small, &[255, 255, 255, 255], 1, 1)?;
        backend.destroy_texture(small);

        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        if backend.device(device)?.dx12.is_some() {
            let queue = backend.device(device)?.deletion_queue.lock().unwrap();
            let deferred_with_import = queue.iter().any(|entry| {
                matches!(
                    entry,
                    CudaDeferredDrop::Texture {
                        import: Some(_),
                        d3d12_resource: Some(_),
                        ..
                    }
                )
            });
            assert!(
                deferred_with_import,
                "destroy_texture must defer DX12 import+resource with the CUDA views"
            );
        }

        // Immediate recreate+upload is the failure mode for an early import drop.
        let large = backend.create_texture(
            device,
            64,
            64,
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
        )?;
        let pixels = vec![128u8; 64 * 64 * 4];
        backend
            .write_texture(large, &pixels, 64, 64)
            .context("write after replace (import must still be valid)")?;
        backend.destroy_texture(large);

        let again = backend.create_texture(
            device,
            32,
            32,
            TextureFormat::Rgba8Unorm,
            TextureKind::DirectInterpolated,
            TextureFlags::COPY_DST,
        )?;
        let pixels = vec![64u8; 32 * 32 * 4];
        backend.write_texture(again, &pixels, 32, 32)?;
        backend.destroy_texture(again);
        backend.device_wait_idle(device)?;
        Ok(())
    }
}
