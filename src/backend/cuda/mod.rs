//! Compute-focused CUDA backend (Slang → PTX → Driver API).
//!
//! With `cuda` + `graphics` + `dx12` on Windows, presentation and a first-slice
//! raster path are enabled via a DX12 companion: CUDA writes shared float4 scratch
//! textures and DX12 presents them; offscreen `Rgba32Float` render targets and
//! TriangleList graphics pipelines are also supported. Depth, indexed draws, and
//! bindless render bindings remain unsupported in this slice.
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
//! Retainable compute partitions whose bodies are kernel-launch-only are captured into
//! CUDA graphs on first submit (`submit_graph_and_retain`) and relaunched on cache hits
//! (`try_resubmit_retained`). Indirect dispatches use CUDA 13.1 device-updatable kernel
//! nodes plus an in-graph updater; uploads, clears, copies, and other graph-unsafe ops fall
//! back to Goldy command-list replay (with a worker-side DtoH resolve for indirect grids).
//! Dynamic waits, deferred host writes, and completion events stay outside the captured graph.

mod pending_submit;
mod retained_graph;
mod runtime_module;
mod texture;
mod timeline;

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
use crate::slang::virtual_main::CudaLaunchArgKind;
use crate::types::{BufferResizeCost, DeviceType};
use anyhow::{Context as _, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use pending_submit::{CudaOp, CudaPendingSubmit, CudaSubmitBody};
use retained_graph::{CudaGraphStats, GraphRegistry};
use std::collections::{BTreeMap, HashMap};
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::thread::JoinHandle;
use texture::{memcpy_htod_array, storage_shader_compatible, CudaSamplerKey, CudaTextureResource};
use timeline::{EventLedger, LedgerEntry};

/// Logical retained entry under the backend lock (graphs themselves live on the worker).
enum RetainedEntry {
    /// Partition was captured; worker registry holds the [`OwnedCudaGraph`](retained_graph::OwnedCudaGraph).
    Graph,
    /// Graph-unsafe partition: replay stored Goldy commands each resubmit.
    Commands(Vec<GraphCommand>),
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
    vb_mirrors: HashMap<BufferHandle, raster::Dx12VertexMirror>,
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
    deletion_queue: Arc<Mutex<Vec<CudaDeferredDrop>>>,
    fence_shutdown: Arc<AtomicBool>,
    fence_thread: Mutex<Option<JoinHandle<()>>>,
}

pub(super) enum CudaDeferredDrop {
    Buffer {
        retire_at: u64,
        #[allow(dead_code)]
        memory: Arc<Mutex<CudaSlice<u8>>>,
    },
    Texture {
        retire_at: u64,
        #[allow(dead_code)]
        resource: Arc<CudaTextureResource>,
    },
    Pipeline {
        retire_at: u64,
        #[allow(dead_code)]
        module: Arc<CudaModule>,
        #[allow(dead_code)]
        function: CudaFunction,
    },
}

struct CudaProgress {
    context: Arc<CudaSubmitContext>,
    event_ledger: EventLedger,
}

impl ContextGpuProgress for CudaProgress {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue {
        timeline::poll_retire_events(
            &self.event_ledger,
            &self.context.completed,
            self.context.handle,
            &self.context.device_retired,
            &self.context.signal_queue,
            &self.context.last_emitted,
        );
        self.context.completed.load(Ordering::Acquire)
    }
}

struct CudaDestroyContext {
    stream: Arc<CudaStream>,
    worker: Arc<SubmissionWorker>,
    fence_shutdown: Arc<AtomicBool>,
    fence_thread: Option<JoinHandle<()>>,
}

impl ContextDestroyHandle for CudaDestroyContext {
    fn wait(&self) -> Result<()> {
        let _ = self.worker.flush();
        self.stream.synchronize().context("CUDA: context destroy stream sync")
    }

    fn finish(self: Box<Self>) -> Result<()> {
        crate::backend::signal_fence::join_fence_poller(&self.fence_shutdown, self.fence_thread);
        Ok(())
    }
}

struct CudaDeferredDeletionFlush {
    context: Arc<CudaSubmitContext>,
}

impl ContextDeferredDeletionFlush for CudaDeferredDeletionFlush {
    fn flush(&self) {
        timeline::poll_retire_events(
            &self.context.event_ledger,
            &self.context.completed,
            self.context.handle,
            &self.context.device_retired,
            &self.context.signal_queue,
            &self.context.last_emitted,
        );
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
        };
        if retire_at > retired {
            kept.push(entry);
        }
        // else drop entry (and its Arc payload) here
    }
    *guard = kept;
}

struct CudaBuffer {
    device: DeviceHandle,
    memory: Arc<Mutex<CudaSlice<u8>>>,
    offset: u64,
    size: u64,
    capacity: u64,
    element_stride: Option<u32>,
    slot: Option<u32>,
    readback: bool,
    /// Bumped on every host/GPU write that changes contents (VB mirror cache key).
    content_epoch: u64,
}

struct CudaShader {
    device: DeviceHandle,
    source: String,
    search_paths: Vec<String>,
    defines: Vec<(String, String)>,
    optimization_level: crate::types::OptimizationLevel,
}

struct CudaSampler {
    device: DeviceHandle,
    desc: SamplerDesc,
    slot: u32,
    key: CudaSamplerKey,
}

struct CudaComputePipeline {
    device: DeviceHandle,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    function: CudaFunction,
    workgroup_size: [u32; 3],
    /// From `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` at pipeline create.
    max_threads_per_block: u32,
    slot_access: Vec<Option<ResourceAccess>>,
    /// Author param order for `[goldy_compute]`; empty for plain Slang compute (all-buffer fallback).
    launch_layout: Vec<CudaLaunchArgKind>,
}

/// Host-side values pushed to `cuLaunchKernel` in shader parameter order.
pub(super) enum CudaLaunchArg {
    Buffer(CudaBufferArg),
    /// `CUtexObject`, `CUsurfObject`, or ignored `SamplerState` word.
    Handle(u64),
    Scalar(u32),
}

impl CudaBackend {
    pub(crate) fn new() -> Result<Self> {
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
        ensure_cuda_driver_at_least_13_1()?;
        let count = CudaContext::device_count().context("CUDA: enumerate devices")?;
        if count <= 0 {
            anyhow::bail!("CUDA: no devices found");
        }
        let mut adapter_info = Vec::with_capacity(count as usize);
        for ordinal in 0..count {
            let ctx = CudaContext::new(ordinal as usize).with_context(|| format!("CUDA: open device {ordinal}"))?;
            let name = ctx.name().unwrap_or_else(|_| format!("CUDA device {ordinal}"));
            adapter_info.push(AdapterInfo {
                id: ordinal as u32,
                name,
                vendor: "NVIDIA".to_string(),
                backend: BackendType::Cuda,
                device_type: DeviceType::DiscreteGpu,
            });
        }
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
            vb_mirrors: HashMap::new(),
            next_device: 1,
            next_context: 1,
            next_buffer: 1,
            next_texture: 1,
            next_sampler: 1,
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

    #[cfg(test)]
    fn graph_stats(&self) -> Arc<CudaGraphStats> {
        Arc::clone(&self.graph_stats)
    }

    fn device(&self, handle: DeviceHandle) -> Result<&CudaDevice> {
        self.devices.get(&handle).context("CUDA: invalid device handle")
    }

    fn context(&self, handle: ContextHandle) -> Result<&Arc<CudaSubmitContext>> {
        self.contexts.get(&handle).context("CUDA: invalid context handle")
    }

    fn sync_device_streams_for_immediate_api(&mut self, device: DeviceHandle) -> Result<()> {
        let worker = Arc::clone(&self.device(device)?.submission_worker);
        worker.flush()?;
        for context in self.contexts.values().filter(|context| context.device == device) {
            context
                .stream
                .synchronize()
                .context("CUDA: sync context stream for immediate API")?;
            timeline::poll_retire_events(
                &context.event_ledger,
                &context.completed,
                context.handle,
                &context.device_retired,
                &context.signal_queue,
                &context.last_emitted,
            );
        }
        Ok(())
    }

    fn unsupported<T>(operation: &str) -> Result<T> {
        anyhow::bail!("CUDA compute-only backend does not support {operation}")
    }

    fn create_storage_buffer(
        &mut self,
        device: DeviceHandle,
        logical_size: u64,
        capacity: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        let capacity = capacity.max(logical_size).max(4);
        let gpu = self.device(device)?;
        let memory = Arc::new(Mutex::new(
            gpu.alloc_stream
                .alloc_zeros::<u8>(capacity as usize)
                .context("CUDA: alloc buffer")?,
        ));
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
                slot: Some(slot),
                readback: false,
                content_epoch: 0,
            },
        );
        Ok(handle)
    }

    fn compile_compute_ptx(
        &self,
        shader: &CudaShader,
    ) -> Result<(String, Vec<Option<ResourceAccess>>, [u32; 3], Vec<CudaLaunchArgKind>)> {
        ensure_cuda_toolkit_on_path();
        let compiler = crate::slang::SlangCompiler::new().context("CUDA: initialize Slang")?;
        let paths: Vec<&str> = shader.search_paths.iter().map(String::as_str).collect();
        let defines: Vec<(&str, &str)> = shader
            .defines
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        let launch_layout = crate::slang::virtual_main::extract_cuda_compute_launch_layout(&shader.source)
            .map_err(|error| anyhow::anyhow!("CUDA launch layout failed: {error}"))?;
        let cuda_source = crate::slang::virtual_main::transform_virtual_main_cuda_compute(&shader.source)
            .map_err(|error| anyhow::anyhow!("CUDA shader lowering failed: {error}"))?;
        let workgroup_size = crate::slang::parse_numthreads(&shader.source).unwrap_or([1, 1, 1]);
        let compiled = compiler.compile_bindless_with_reflection_and_defines(
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
        let access = crate::slang::virtual_main::extract_push_constant_categories(&shader.source)
            .iter()
            .map(|category| {
                category.map(|category| match category {
                    crate::types::ResourceCategory::Broadcast
                    | crate::types::ResourceCategory::Texture
                    | crate::types::ResourceCategory::Sampler => ResourceAccess::Read,
                    crate::types::ResourceCategory::Scattered | crate::types::ResourceCategory::StorageImage => {
                        ResourceAccess::ReadWrite
                    }
                })
            })
            .collect();
        Ok((ptx, access, workgroup_size, launch_layout))
    }

    fn buffer_arg(&self, stream: &Arc<CudaStream>, buffer: &CudaBuffer) -> Result<CudaBufferArg> {
        let memory = buffer.memory.lock().unwrap();
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
                    if !storage_shader_compatible(element, tex.format) {
                        anyhow::bail!(
                            "CUDA: DirectSpatial<{element}> writable access requires a \
                             storage-compatible format (float4 ↔ Rgba32Float); got {:?}",
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
        let mut memory = buffer.memory.lock().unwrap();
        let start = (buffer.offset + offset) as usize;
        let end = start + data.len();
        let mut view = memory
            .try_slice_mut(start..end)
            .context("CUDA: write range out of bounds")?;
        stream.memcpy_htod(data, &mut view).context("CUDA: HtoD write failed")?;
        pending_submit::maybe_validate_sync(stream, "immediate WriteBuffer")
    }

    fn clear_buffer_region(stream: &Arc<CudaStream>, buffer: &CudaBuffer, offset: u64, size: u64) -> Result<()> {
        let clear_size = if size == 0 {
            buffer.size.saturating_sub(offset)
        } else {
            size
        };
        let mut memory = buffer.memory.lock().unwrap();
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

        if Arc::ptr_eq(&src.memory, &dst.memory) {
            // Same allocation: avoid simultaneous &/&mut CudaSlice views. A device temp
            // keeps both overlapping and non-overlapping self-copies memmove-safe.
            let mut temp = stream
                .alloc_zeros::<u8>(byte_len)
                .context("CUDA: alloc overlapping-copy scratch")?;
            {
                let memory = src.memory.lock().unwrap();
                let src_view = memory
                    .try_slice(src_abs as usize..src_abs as usize + byte_len)
                    .context("CUDA: copy source out of bounds")?;
                stream
                    .memcpy_dtod(&src_view, &mut temp)
                    .context("CUDA: same-alloc copy to scratch")?;
            }
            {
                let mut memory = dst.memory.lock().unwrap();
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
        let src_arc = Arc::clone(&src.memory);
        let dst_arc = Arc::clone(&dst.memory);
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
        pipeline: &CudaComputePipeline,
        indices: &[u32],
        user: &[u32],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        let limits = self.device(pipeline.device)?.limits;
        validate_launch_config(
            &limits,
            pipeline.max_threads_per_block,
            workgroups,
            pipeline.workgroup_size,
            0,
            None,
        )?;
        let launch_args = self.build_launch_args(stream, &pipeline.launch_layout, indices, user)?;
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
            let mut builder = stream.launch_builder(&pipeline.function);
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
        let limits = self.device(pipeline.device)?.limits;
        validate_launch_config(
            &limits,
            pipeline.max_threads_per_block,
            workgroups,
            pipeline.workgroup_size,
            0,
            label,
        )?;
        let launch_args = self.build_launch_args(stream, &pipeline.launch_layout, indices, user)?;
        let (keep_alive_buffers, keep_alive_textures) = self.collect_launch_pins(indices)?;
        let function = pipeline
            .module
            .load_function("cs_main")
            .context("CUDA: cuModuleGetFunction(cs_main) failed")?;
        Ok(CudaOp::Launch {
            label,
            function,
            module: Arc::clone(&pipeline.module),
            workgroup_size: pipeline.workgroup_size,
            grid: workgroups,
            args: launch_args,
            keep_alive_buffers,
            keep_alive_textures,
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
        let launch_args = self.build_launch_args(stream, &pipeline.launch_layout, indices, user)?;
        let (mut keep_alive_buffers, keep_alive_textures) = self.collect_launch_pins(indices)?;
        keep_alive_buffers.push(Arc::clone(&shape_buf.memory));

        let shape_abs_offset = shape_buf.offset + shape_offset;
        let shape_ptr = {
            let memory = shape_buf.memory.lock().unwrap();
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

        let function = pipeline
            .module
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
        let max_threads_per_block = pipeline.max_threads_per_block;
        let workgroup_size = pipeline.workgroup_size;
        let module = Arc::clone(&pipeline.module);

        Ok(CudaOp::LaunchIndirect {
            label,
            function,
            module,
            workgroup_size,
            args: launch_args,
            keep_alive_buffers,
            keep_alive_textures,
            shape_ptr,
            shape_memory: Arc::clone(&shape_buf.memory),
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
                buffers.push(Arc::clone(&buffer.memory));
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

    fn materialize_ops(&mut self, stream: &Arc<CudaStream>, commands: &[GpuCommand]) -> Result<Vec<CudaOp>> {
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
                    let buffer = self.buffers.get(buffer).context("CUDA: invalid clear buffer")?;
                    let clear_size = if *size == 0 {
                        buffer.size.saturating_sub(*offset)
                    } else {
                        *size
                    };
                    ops.push(CudaOp::Clear {
                        memory: Arc::clone(&buffer.memory),
                        abs_offset: buffer.offset + *offset,
                        size: clear_size,
                    });
                }
                GpuCommand::WriteBuffer { buffer, offset, data } => {
                    let buffer = self.buffers.get(buffer).context("CUDA: invalid write buffer")?;
                    if *offset + data.len() as u64 > buffer.size {
                        anyhow::bail!("CUDA: write exceeds logical buffer size");
                    }
                    ops.push(CudaOp::Write {
                        memory: Arc::clone(&buffer.memory),
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
                    let dst_buf = self.buffers.get(dst).context("CUDA: invalid copy destination")?;
                    if src_buf.device != dst_buf.device {
                        anyhow::bail!("CUDA: copy across devices is not supported");
                    }
                    if *src_offset + *size > src_buf.size {
                        anyhow::bail!("CUDA: copy source range exceeds logical buffer size");
                    }
                    if *dst_offset + *size > dst_buf.size {
                        anyhow::bail!("CUDA: copy destination range exceeds logical buffer size");
                    }
                    ops.push(CudaOp::Copy {
                        src: Arc::clone(&src_buf.memory),
                        src_abs: src_buf.offset + *src_offset,
                        dst: Arc::clone(&dst_buf.memory),
                        dst_abs: dst_buf.offset + *dst_offset,
                        size: *size,
                    });
                }
                GpuCommand::FrameTableStaging { data } => {
                    frame_table = Some(Arc::clone(data));
                }
                GpuCommand::ResourceBarrier { .. } => {}
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
                    ops.push(CudaOp::CopyBufferToTexture {
                        src: Arc::clone(&src_buf.memory),
                        src_abs: src_buf.offset + *src_offset,
                        src_row_pitch: *src_row_pitch,
                        texture: Arc::clone(dst_tex),
                        x: *x,
                        y: *y,
                        width: *width,
                        height: *height,
                    });
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
                    ops.push(CudaOp::CopyTextureToBuffer {
                        texture: Arc::clone(src_tex),
                        x: 0,
                        y: 0,
                        width: layout.width,
                        height: layout.height,
                        dst: Arc::clone(&dst_buf.memory),
                        dst_abs: dst_buf.offset + layout.footprint_offset,
                        dst_row_pitch: layout.row_pitch,
                    });
                }
                GpuCommand::CopyRenderTarget { src, dst } => {
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    {
                        // Fast path: copy into surface present scratch → stash the DX12 RT
                        // so present can blit it directly (no CUDA array round-trip).
                        if let Some((surf, image_index)) =
                            surface::scratch_slot_for_texture(self, *dst)
                        {
                            let (d3d12_resource, fence) = {
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
                                (rt.d3d12_resource.clone(), rt.last_dx12_fence)
                            };
                            if let Some(slot) = self
                                .surfaces
                                .get_mut(&surf)
                                .and_then(|s| s.scratch.get_mut(image_index))
                                .and_then(|s| s.as_mut())
                            {
                                slot.dx12_present_src = Some((d3d12_resource, fence));
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
                            if cuda_src.width != cuda_dst.width || cuda_src.height != cuda_dst.height
                            {
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
                                    .context(
                                        "CUDA/DX12: companion required for CopyRenderTarget",
                                    )?;
                                ops.push(CudaOp::WaitExternalFence {
                                    cuda_ctx: Arc::clone(&companion.cuda_ctx),
                                    semaphore: pending_submit::SendExternalSemaphore(
                                        companion.cuda_semaphore,
                                    ),
                                    value: fence,
                                });
                            }
                            ops.push(CudaOp::CopyTexture {
                                src: cuda_src,
                                dst: cuda_dst,
                            });
                        }
                    }
                    #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
                    {
                        let _ = (src, dst);
                        anyhow::bail!(
                            "CUDA: CopyRenderTarget requires cuda+graphics+dx12 on Windows"
                        );
                    }
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
        let effective = commands_with_sync_prologue(commands, sync);
        let stream = Arc::clone(&self.context(ctx)?.stream);
        let ops = self.materialize_ops(&stream, &effective)?;
        self.enqueue_submit(ctx, sync, CudaSubmitBody::Ops(ops))
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
        commands
            .iter()
            .any(|c| matches!(c, GraphCommand::Render { .. }))
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
                        if !batch.is_empty() {
                            last_tv = self.submit_commands(ctx, &batch, sync)?;
                            self.wait_until(ctx, last_tv)?;
                            batch.clear();
                        }
                        let device = self.context_device(ctx);
                        raster::render_to_target(self, device, *target, *color_load, render_cmds)?;
                        last_tv = self.submit_commands(ctx, &[], sync)?;
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

    fn enqueue_submit(
        &mut self,
        ctx: ContextHandle,
        sync: Option<&SubmitSync>,
        body: CudaSubmitBody,
    ) -> Result<crate::timeline::TimelineValue> {
        let context = Arc::clone(self.context(ctx)?);
        let device_handle = context.device;
        let device = self.device(device_handle)?;
        let stream = Arc::clone(&context.stream);
        let worker = Arc::clone(&device.submission_worker);
        let next_timeline = Arc::clone(&device.next_timeline);
        let event_ledger = Arc::clone(&device.event_ledger);
        let cuda_ctx = Arc::clone(&device.ctx);

        worker.check_error()?;

        let fence_value = submission_worker::allocate_timeline_value(&next_timeline);
        let completion_event = Arc::new(
            cuda_ctx
                .new_event(None)
                .context("CUDA: create completion event failed")?,
        );
        event_ledger.lock().unwrap().insert(
            fence_value,
            LedgerEntry {
                context: ctx,
                event: Arc::clone(&completion_event),
                recorded: false,
            },
        );

        let mut stream_waits = Vec::new();
        let mut host_waits = Vec::new();
        let mut deferred_writes = Vec::new();
        if let Some(sync) = sync {
            for epoch in &sync.waits {
                let event = timeline::lookup_event(&event_ledger, epoch.context, epoch.value).with_context(|| {
                    format!(
                        "CUDA: cross-context wait missing event for context {:?} value {}",
                        epoch.context, epoch.value
                    )
                })?;
                stream_waits.push(event);
            }
            for epoch in sync.cpu_waits.iter().chain(sync.host_observed_waits.iter()) {
                let event = timeline::lookup_event(&event_ledger, epoch.context, epoch.value).with_context(|| {
                    format!(
                        "CUDA: host wait missing event for context {:?} value {}",
                        epoch.context, epoch.value
                    )
                })?;
                host_waits.push(event);
            }
            deferred_writes = pending_submit::materialize_deferred_writes(&sync.deferred_host_writes, |handle| {
                let buffer = self
                    .buffers
                    .get(&handle)
                    .with_context(|| format!("CUDA: deferred write invalid buffer {handle}"))?;
                Ok((Arc::clone(&buffer.memory), buffer.offset))
            })?;
        }

        let pending = CudaPendingSubmit {
            stream,
            context,
            fence_value,
            completion_event,
            event_ledger,
            stream_waits,
            host_waits,
            deferred_writes,
            body,
        };
        worker.enqueue(fence_value, Box::new(pending))?;
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
}

/// Soft clone of buffer metadata + shared allocation (for copy that needs both ends).
impl CudaBuffer {
    fn clone_meta(&self) -> Self {
        Self {
            device: self.device,
            memory: Arc::clone(&self.memory),
            offset: self.offset,
            size: self.size,
            capacity: self.capacity,
            element_stride: self.element_stride,
            slot: self.slot,
            readback: self.readback,
            content_epoch: self.content_epoch,
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

fn ensure_cuda_driver_at_least_13_1() -> Result<()> {
    let mut version = 0i32;
    let r = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        anyhow::bail!("CUDA: cuDriverGetVersion failed: {r:?}");
    }
    if version < MIN_CUDA_DRIVER_VERSION {
        anyhow::bail!(
            "CUDA: goldy requires CUDA driver 13.1+ for device-updatable graph nodes \
             (got driver version encoding {version}; need >= {MIN_CUDA_DRIVER_VERSION})"
        );
    }
    Ok(())
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
        match timeline::lookup_event(&device.event_ledger, ctx, value) {
            Some(event) => Ok(Some(Box::new(timeline::CudaTimelineBlockingWait { event }))),
            // Match DX12: waiting on a never-submitted value still yields a timeout-capable
            // wait object (not an immediate Err that used to deadlock under classify+lock).
            None => Ok(Some(Box::new(timeline::CudaAbsentTimelineWait { context: ctx, value }))),
        }
    }

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        let context = Arc::clone(self.context(ctx)?);
        let device_handle = context.device;
        if let Some(device) = self.devices.get(&device_handle) {
            device.submission_worker.flush()?;
        }
        timeline::poll_retire_events(
            &context.event_ledger,
            &context.completed,
            context.handle,
            &context.device_retired,
            &context.signal_queue,
            &context.last_emitted,
        );
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
        let alloc_stream = ctx.default_stream();
        let handle = self.next_device;
        self.next_device += 1;
        let mut gpu = CudaDevice {
            ctx,
            alloc_stream,
            submission_worker: Arc::new(SubmissionWorker::new(submission_worker::SUBMISSION_QUEUE_CAPACITY)),
            next_timeline: Arc::new(AtomicU64::new(1)),
            retired: Arc::new(AtomicU64::new(0)),
            event_ledger: Arc::new(Mutex::new(BTreeMap::new())),
            deletion_queue: Arc::new(Mutex::new(Vec::new())),
            graph_registry: Arc::new(Mutex::new(GraphRegistry::default())),
            graph_stats: Arc::clone(&self.graph_stats),
            limits,
            indirect_updater,
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            dx12: None,
        };
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        surface::attach_companion(&mut gpu)
            .with_context(|| format!("CUDA: attach DX12 presentation companion for adapter {adapter_id}"))?;
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
            self.vb_mirrors.clear();
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
        let (present_stream, dx12) = {
            let gpu = self.device(device)?;
            (
                gpu.dx12.as_ref().map(|c| Arc::clone(&c.present_stream)),
                gpu.dx12.as_ref().map(Arc::clone),
            )
        };
        worker.flush()?;
        for context in self.contexts.values().filter(|context| context.device == device) {
            context
                .stream
                .synchronize()
                .context("CUDA: context stream synchronize failed")?;
            timeline::poll_retire_events(
                &context.event_ledger,
                &context.completed,
                context.handle,
                &context.device_retired,
                &context.signal_queue,
                &context.last_emitted,
            );
        }
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        if let Some(stream) = present_stream {
            stream
                .synchronize()
                .context("CUDA: present stream synchronize failed")?;
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
        let (stream, event_ledger, device_retired, deletion_queue) = {
            let gpu = self.device(device)?;
            (
                gpu.ctx.new_stream().context("CUDA: create context stream failed")?,
                Arc::clone(&gpu.event_ledger),
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
            event_ledger: Arc::clone(&event_ledger),
            deletion_queue,
            fence_shutdown: Arc::clone(&fence_shutdown),
            fence_thread: Mutex::new(None),
        });

        let poller_context = Arc::clone(&context);
        let poller_ledger = Arc::clone(&event_ledger);
        let shutdown = Arc::clone(&fence_shutdown);
        let handle_thread = std::thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                timeline::poll_retire_events(
                    &poller_ledger,
                    &poller_context.completed,
                    poller_context.handle,
                    &poller_context.device_retired,
                    &poller_context.signal_queue,
                    &poller_context.last_emitted,
                );
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
        Some(Box::new(CudaDestroyContext {
            stream: Arc::clone(&context.stream),
            worker: worker.unwrap_or_else(|| Arc::new(SubmissionWorker::new(1))),
            fence_shutdown: Arc::clone(&context.fence_shutdown),
            fence_thread,
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
        Some(Arc::new(CudaProgress {
            event_ledger: Arc::clone(&context.event_ledger),
            context,
        }))
    }

    fn context_device(&self, ctx: ContextHandle) -> DeviceHandle {
        self.contexts.get(&ctx).map(|context| context.device).unwrap_or(0)
    }

    fn create_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        _access: BufferKind,
        element_stride: Option<u32>,
        _flags: BufferFlags,
    ) -> Result<BufferHandle> {
        self.create_storage_buffer(device, size, size, element_stride)
    }

    fn create_buffer_with_capacity(
        &mut self,
        device: DeviceHandle,
        initial_size: u64,
        capacity: u64,
        _access: BufferKind,
        element_stride: Option<u32>,
        _flags: BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        let capacity = capacity.max(initial_size);
        Ok((
            self.create_storage_buffer(device, initial_size, capacity, element_stride)?,
            capacity,
        ))
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
                });
            }
        }
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        self.sync_device_streams_for_immediate_api(
            self.buffers
                .get(&buffer)
                .map(|buffer| buffer.device)
                .context("CUDA: invalid buffer handle")?,
        )?;
        let device = {
            let buffer = self
                .buffers
                .get_mut(&buffer)
                .context("CUDA: invalid buffer handle")?;
            buffer.bump_content_epoch();
            buffer.device
        };
        let stream = Arc::clone(&self.device(device)?.alloc_stream);
        let buffer = self
            .buffers
            .get(&buffer)
            .context("CUDA: invalid buffer handle")?;
        Self::write_buffer_region(&stream, buffer, offset, data)
    }

    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle> {
        let gpu = self.device(device)?;
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
                memory,
                offset: 0,
                size,
                capacity,
                element_stride: None,
                slot: None,
                readback: true,
                content_epoch: 0,
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
        for context in self.contexts.values().filter(|context| context.device == device) {
            context
                .stream
                .synchronize()
                .context("CUDA: readback context stream sync failed")?;
        }
        let stream = Arc::clone(&self.device(device)?.alloc_stream);
        let memory = buffer.memory.lock().unwrap();
        let view = memory
            .try_slice(buffer.offset as usize..(buffer.offset as usize + output.len()))
            .context("CUDA: readback range out of bounds")?;
        stream.memcpy_dtoh(&view, output).context("CUDA: DtoH readback failed")
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
        let parent = self
            .buffers
            .get(&parent)
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
                slot: Some(slot),
                readback: false,
                content_epoch: parent.content_epoch,
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
                let memory = old.memory.lock().unwrap();
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
        let target = self.buffers.get_mut(&buffer).expect("validated above");
        target.memory = Arc::new(Mutex::new(replacement));
        target.offset = 0;
        target.size = new_size;
        target.capacity = capacity;
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
        if depth_stencil.is_some() {
            return Self::unsupported("graphics pipelines with depth (first CUDA raster slice)");
        }
        self.create_pipeline(
            device,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
        )
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

    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn render_to_target(
        &mut self,
        device: DeviceHandle,
        target: RenderTargetHandle,
        color_load: crate::types::TargetLoad,
        commands: &[RenderCommand],
    ) -> Result<()> {
        raster::render_to_target(self, device, target, color_load, commands)
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
        let gpu = self.device(device)?;
        let ctx = Arc::clone(&gpu.ctx);
        let (storage_slot, sampled_slot) = match access {
            TextureKind::Interpolated => (None, Some(self.alloc_registry_slot())),
            TextureKind::Direct => (Some(self.alloc_registry_slot()), None),
            TextureKind::DirectInterpolated => (Some(self.alloc_registry_slot()), Some(self.alloc_registry_slot())),
        };
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
            let device_handle = self
                .devices
                .iter()
                .find(|(_, d)| Arc::ptr_eq(&d.ctx, &resource.ctx))
                .map(|(h, _)| *h);
            if let Some(device_handle) = device_handle {
                if let Some(device) = self.devices.get(&device_handle) {
                    let retire_at = submission_worker::submission_horizon(&device.next_timeline);
                    device
                        .deletion_queue
                        .lock()
                        .unwrap()
                        .push(CudaDeferredDrop::Texture { retire_at, resource });
                }
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
    fn surface_set_present_mode(
        &mut self,
        surface: SurfaceHandle,
        mode: crate::types::PresentMode,
    ) -> Result<()> {
        surface::surface_set_present_mode(self, surface, mode)
    }

    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue {
        let Some(context) = self.contexts.get(&ctx) else {
            return 0;
        };
        timeline::poll_retire_events(
            &context.event_ledger,
            &context.completed,
            context.handle,
            &context.device_retired,
            &context.signal_queue,
            &context.last_emitted,
        );
        context.completed.load(Ordering::Acquire)
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
        let event = {
            let ledger = gpu.event_ledger.lock().unwrap();
            ledger
                .get(&value)
                .map(|entry| Arc::clone(&entry.event))
                .with_context(|| format!("CUDA: timeline value {value} has not been submitted"))?
        };
        event
            .synchronize()
            .context("CUDA: device_wait_until event sync failed")?;
        timeline::advance_device_retired(&gpu.event_ledger, &gpu.retired);
        for context in self.contexts.values().filter(|context| context.device == device) {
            timeline::poll_retire_events(
                &context.event_ledger,
                &context.completed,
                context.handle,
                &context.device_retired,
                &context.signal_queue,
                &context.last_emitted,
            );
        }
        Ok(())
    }

    fn poll_signals(
        &mut self,
        ctx: ContextHandle,
        _progress: crate::timeline::TimelineValue,
    ) -> Vec<crate::signal::QueuedSignal> {
        if let Some(context) = self.contexts.get(&ctx) {
            timeline::poll_retire_events(
                &context.event_ledger,
                &context.completed,
                context.handle,
                &context.device_retired,
                &context.signal_queue,
                &context.last_emitted,
            );
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

        if pending_submit::ops_are_graph_safe(&ops) && !retained_graph::cuda_launch_blocking_active() {
            let device_handle = self.context(ctx)?.device;
            let device = self.device(device_handle)?;
            let body = CudaSubmitBody::CaptureAndLaunch {
                key,
                ops,
                registry: Arc::clone(&device.graph_registry),
                stats: Arc::clone(&device.graph_stats),
            };
            self.retained.insert((ctx, key), RetainedEntry::Graph);
            tracing::trace!(key, "CUDA: capturing retainable partition into CudaGraph");
            self.enqueue_submit(ctx, sync, body)
        } else {
            self.graph_stats.fallbacks.fetch_add(1, Ordering::Relaxed);
            self.retained
                .insert((ctx, key), RetainedEntry::Commands(commands.to_vec()));
            tracing::trace!(
                key,
                blocking = retained_graph::cuda_launch_blocking_active(),
                "CUDA: retainable partition uses command-replay fallback"
            );
            self.enqueue_submit(ctx, sync, CudaSubmitBody::Ops(ops))
        }
    }

    fn try_resubmit_retained(
        &mut self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<crate::timeline::TimelineValue>> {
        match self.retained.get(&(ctx, key)) {
            Some(RetainedEntry::Graph) => {
                let device_handle = self.context(ctx)?.device;
                let device = self.device(device_handle)?;
                let body = CudaSubmitBody::LaunchRetained {
                    key,
                    registry: Arc::clone(&device.graph_registry),
                    stats: Arc::clone(&device.graph_stats),
                };
                tracing::trace!(key, "CUDA: launching retained CudaGraph");
                self.enqueue_submit(ctx, sync, body).map(Some)
            }
            Some(RetainedEntry::Commands(commands)) => {
                let commands = commands.clone();
                let gpu_commands = Self::flatten_graph_commands(&commands)?;
                let effective = commands_with_sync_prologue(&gpu_commands, sync);
                let stream = Arc::clone(&self.context(ctx)?.stream);
                let ops = self.materialize_ops(&stream, &effective)?;
                self.graph_stats.fallbacks.fetch_add(1, Ordering::Relaxed);
                tracing::trace!(key, "CUDA: replaying retained GraphCommands");
                self.enqueue_submit(ctx, sync, CudaSubmitBody::Ops(ops)).map(Some)
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
        let (ptx, slot_access, workgroup_size, launch_layout) = self.compile_compute_ptx(shader)?;
        let gpu = self.device(device)?;
        let module = load_ptx_module(&gpu.ctx, &ptx)?;
        let function = module
            .load_function("cs_main")
            .context("CUDA: cuModuleGetFunction(cs_main) failed")?;
        let max_threads_per_block = function
            .max_threads_per_block()
            .context("CUDA: query CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK failed")?
            .max(0) as u32;
        let handle = self.next_compute_pipeline;
        self.next_compute_pipeline += 1;
        self.compute_pipelines.insert(
            handle,
            CudaComputePipeline {
                device,
                module,
                function,
                workgroup_size,
                max_threads_per_block,
                slot_access,
                launch_layout,
            },
        );
        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        if let Some(pipeline) = self.compute_pipelines.remove(&pipeline) {
            if let Some(device) = self.devices.get(&pipeline.device) {
                let retire_at = submission_worker::submission_horizon(&device.next_timeline);
                device.deletion_queue.lock().unwrap().push(CudaDeferredDrop::Pipeline {
                    retire_at,
                    module: pipeline.module,
                    function: pipeline.function,
                });
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
            | crate::types::ResourceCategory::Sampler => 4096,
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
        let backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA scheme test: {error:#}");
                return Ok(());
            }
        };
        let device = Arc::new(crate::Device::from_backend(Box::new(backend))?);
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA scheme test: {error:#}");
                return Ok(());
            }
        };
        let device = Arc::new(crate::Device::from_backend(Box::new(backend))?);
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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

    fn try_cuda_device() -> Result<Option<Arc<crate::Device>>> {
        match CudaBackend::new() {
            Ok(backend) => Ok(Some(Arc::new(crate::Device::from_backend(Box::new(backend))?))),
            Err(error) => {
                eprintln!("skipping CUDA scheme test: {error:#}");
                Ok(None)
            }
        }
    }

    fn try_cuda_device_with_stats() -> Result<Option<(Arc<crate::Device>, Arc<CudaGraphStats>)>> {
        match CudaBackend::new() {
            Ok(backend) => {
                let stats = backend.graph_stats();
                stats.reset();
                Ok(Some((Arc::new(crate::Device::from_backend(Box::new(backend))?), stats)))
            }
            Err(error) => {
                eprintln!("skipping CUDA scheme test: {error:#}");
                Ok(None)
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
    fn overlapping_self_copy_is_memmove_safe() -> Result<()> {
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
    fn upload_write_commands_use_command_fallback_not_graph_capture() -> Result<()> {
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
        assert!(stats.snapshot().fallbacks >= 2);
        Ok(())
    }

    #[test]
    fn destroy_context_evicts_retained_graphs() -> Result<()> {
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        assert_eq!(scheme.replay_stats().resubmit_hits, 1);
        Ok(())
    }

    #[test]
    fn indirect_with_clear_uses_command_fallback() -> Result<()> {
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
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
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
        assert_eq!(snap.captures, 0, "clear+indirect must not capture: {snap:?}");
        assert!(snap.fallbacks >= 1, "clear+indirect should fallback: {snap:?}");
        Ok(())
    }
}
