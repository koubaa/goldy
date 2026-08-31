//! Owned CUDA submits executed on the per-device submission worker.

#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
use super::dx12_companion::{cuda_wait_fence, Dx12Companion};
use super::retained_graph::{self, CudaGraphStats, GraphRegistry};
use super::timeline::{self, EventLedger};
use super::{CudaBufferArg, CudaLaunchArg, CudaSubmitContext};
use crate::backend::submission_worker::PendingSubmit;
use crate::backend::{ContextHandle, DeferredHostWrite};
use crate::timeline::TimelineValue;
use crate::types::DispatchShape;
use anyhow::{Context as _, Result};
use cudarc::driver::{
    CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
};
use std::sync::{Arc, Mutex};

pub(super) use super::capture_gate::lock_capture_alloc_gate;

#[repr(C)]
#[derive(Clone, Copy)]
struct U64Word(u64);
// SAFETY: plain POD matching a device pointer / handle word.
unsafe impl DeviceRepr for U64Word {}

#[repr(C)]
#[derive(Clone, Copy)]
struct U32Word(u32);
// SAFETY: plain POD matching a CUDA `unsigned int` kernel parameter.
unsafe impl DeviceRepr for U32Word {}

/// GPU/host work recorded under the backend lock, executed without it.
pub(super) struct CudaPendingSubmit {
    pub stream: Arc<CudaStream>,
    pub context: Arc<CudaSubmitContext>,
    pub fence_value: TimelineValue,
    pub completion_event: Arc<CudaEvent>,
    pub event_ledger: EventLedger,
    /// Cross-context GPU waits (`SubmitSync.waits`).
    pub stream_waits: Vec<Arc<CudaEvent>>,
    /// DX12 fence values to wait on the submission stream before GPU work.
    ///
    /// Also used for demoted WAR/`host_observed` Dx12Fence epochs — never CPU-wait those
    /// on the worker (matches DX12's measured FPS policy).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    pub dx12_stream_fence_waits: Vec<(Arc<Dx12Companion>, u64)>,
    /// Host waits before deferred writes / GPU enqueue.
    pub host_waits: Vec<Arc<CudaEvent>>,
    pub deferred_writes: Vec<MaterializedHostWrite>,
    pub body: CudaSubmitBody,
}

/// One alternating segment of a retainable CUDA partition.
///
/// Graph-safe kernel runs and pinned host/device copies are captured into
/// [`CudaOpSegment::Graph`] islands; remaining boundary ops (inline WriteBuffer,
/// imported-surface copies, external fences) stay as stream-replayed ops.
/// Multiple islands and stream segments may interleave; FIFO stream order preserves
/// dependencies without inter-island events.
#[derive(Clone)]
pub(super) enum CudaOpSegment {
    /// Contiguous graph-safe kernel launches (captured / relaunched).
    Graph(Vec<CudaOp>),
    /// Graph-unsafe boundary ops executed with `execute_ops`.
    Stream(Vec<CudaOp>),
}

impl CudaOpSegment {
    pub fn is_graph(&self) -> bool {
        matches!(self, Self::Graph(_))
    }

    pub fn ops(&self) -> &[CudaOp] {
        match self {
            Self::Graph(ops) | Self::Stream(ops) => ops,
        }
    }
}

/// Retained relaunch plan: graph islands are markers only (payload lives in the registry).
#[derive(Clone)]
pub(super) enum CudaLaunchSegment {
    Graph,
    Stream(Vec<CudaOp>),
}

/// Body executed after the dynamic sync prefix and before the completion event.
pub(super) enum CudaSubmitBody {
    /// Immediate stream ops (standalone submits and command-replay fallback).
    Ops {
        ops: Vec<CudaOp>,
        /// When false, surface prepare skips content-epoch bumps (already done by retained replay).
        bump_content_epochs: bool,
    },
    /// Capture each [`CudaOpSegment::Graph`] into a retained island, then execute the
    /// full segment list once (graph launch + stream replay).
    CaptureAndLaunch {
        key: u64,
        segments: Vec<CudaOpSegment>,
        registry: Arc<Mutex<GraphRegistry>>,
        stats: Arc<CudaGraphStats>,
    },
    /// Relaunch retained graph islands and replay stream segments in order.
    LaunchRetained {
        key: u64,
        segments: Vec<CudaLaunchSegment>,
        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
        scratch_images: Vec<(crate::backend::SurfaceHandle, usize)>,
        registry: Arc<Mutex<GraphRegistry>>,
        stats: Arc<CudaGraphStats>,
    },
}

pub(super) struct MaterializedHostWrite {
    pub memory: Arc<Mutex<CudaSlice<u8>>>,
    pub abs_offset: u64,
    pub data: Arc<[u8]>,
}

#[derive(Clone)]
pub(super) enum CudaOp {
    Launch {
        label: Option<&'static str>,
        function: CudaFunction,
        module: Arc<CudaModule>,
        workgroup_size: [u32; 3],
        grid: (u32, u32, u32),
        args: Vec<CudaLaunchArg>,
        keep_alive_buffers: Vec<Arc<Mutex<CudaSlice<u8>>>>,
        keep_alive_textures: Vec<Arc<super::texture::CudaTextureResource>>,
        /// False for format-specialized PTX variants. Those kernels stay in
        /// [`CudaOpSegment::Stream`]; capturing them into a graph has faulted on relaunch.
        graph_capture_ok: bool,
    },
    /// GPU-driven dispatch: graph path uses a device-updatable consumer node; fallback
    /// path resolves the shape via DtoH on the worker stream.
    LaunchIndirect {
        label: Option<&'static str>,
        function: CudaFunction,
        module: Arc<CudaModule>,
        workgroup_size: [u32; 3],
        args: Vec<CudaLaunchArg>,
        keep_alive_buffers: Vec<Arc<Mutex<CudaSlice<u8>>>>,
        keep_alive_textures: Vec<Arc<super::texture::CudaTextureResource>>,
        /// See [`CudaOp::Launch::graph_capture_ok`].
        graph_capture_ok: bool,
        /// Absolute device address of the 12-byte [`DispatchShape`].
        shape_ptr: u64,
        shape_memory: Arc<Mutex<CudaSlice<u8>>>,
        shape_abs_offset: u64,
        /// Device slot written with `CUgraphDeviceNode` after capture finalize.
        node_slot_ptr: u64,
        node_slot: Arc<Mutex<CudaSlice<u64>>>,
        /// Diagnostic status word (0 = ok, -1 = oversized, else CUDA error).
        status_ptr: u64,
        /// Retains the status allocation; only [`status_ptr`] is read after record.
        #[allow(dead_code)]
        status_memory: Arc<Mutex<CudaSlice<i32>>>,
        updater: CudaFunction,
        updater_module: Arc<CudaModule>,
        max_grid: (u32, u32, u32),
        /// Function max threads for host-side validation on the fallback path.
        max_threads_per_block: u32,
        limits: super::CudaDeviceLimits,
    },
    Clear {
        memory: Arc<Mutex<CudaSlice<u8>>>,
        abs_offset: u64,
        size: u64,
        /// Device pointer of the clear range, baked at materialize (capture-safe).
        device_ptr: u64,
    },
    Write {
        memory: Arc<Mutex<CudaSlice<u8>>>,
        abs_offset: u64,
        data: Vec<u8>,
    },
    /// HtoD from CPU_WRITABLE pinned host staging, read at execute time (retained-safe).
    WriteFromHost {
        memory: Arc<Mutex<CudaSlice<u8>>>,
        abs_offset: u64,
        /// Device pointer of the destination range, baked at materialize (capture-safe).
        device_ptr: u64,
        host: Arc<Mutex<super::pinned_host::CudaPinnedHost>>,
        host_offset: usize,
        len: usize,
    },
    Copy {
        src: Arc<Mutex<CudaSlice<u8>>>,
        src_abs: u64,
        src_ptr: u64,
        dst: Arc<Mutex<CudaSlice<u8>>>,
        dst_abs: u64,
        dst_ptr: u64,
        size: u64,
    },
    WriteTexture {
        texture: Arc<super::texture::CudaTextureResource>,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: Vec<u8>,
        src_row_pitch: u32,
    },
    /// Texture upload from CPU_WRITABLE pinned host staging, read at execute time (retained-safe).
    WriteTextureFromHost {
        texture: Arc<super::texture::CudaTextureResource>,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        host: Arc<Mutex<super::pinned_host::CudaPinnedHost>>,
        host_offset: usize,
        len: usize,
        src_row_pitch: u32,
    },
    CopyBufferToTexture {
        src: Arc<Mutex<CudaSlice<u8>>>,
        src_abs: u64,
        src_ptr: u64,
        src_row_pitch: u32,
        texture: Arc<super::texture::CudaTextureResource>,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    CopyTexture {
        src: Arc<super::texture::CudaTextureResource>,
        dst: Arc<super::texture::CudaTextureResource>,
    },
    /// Wait on a D3D12 fence imported as a CUDA external semaphore (raster → compute handoff).
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    WaitExternalFence {
        cuda_ctx: Arc<cudarc::driver::CudaContext>,
        semaphore: SendExternalSemaphore,
        value: u64,
    },
    /// Signal the companion fence after all preceding CUDA scratch writes complete.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    SignalExternalFence {
        cuda_ctx: Arc<cudarc::driver::CudaContext>,
        semaphore: SendExternalSemaphore,
        value: u64,
    },
    CopyTextureToBuffer {
        texture: Arc<super::texture::CudaTextureResource>,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        dst: Arc<Mutex<CudaSlice<u8>>>,
        dst_abs: u64,
        dst_ptr: u64,
        dst_row_pitch: u32,
    },
}

/// `CUexternalSemaphore` is a driver handle; Goldy only uses it from the submission worker.
#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
#[derive(Clone, Copy)]
pub(super) struct SendExternalSemaphore(pub cudarc::driver::sys::CUexternalSemaphore);
#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
// SAFETY: handle is process-local and only touched under Goldy's submit serialization.
unsafe impl Send for SendExternalSemaphore {}
#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
unsafe impl Sync for SendExternalSemaphore {}

/// True when `ops` can be recorded into a CUDA graph.
///
/// Kernel launches (direct and indirect) are capture-safe when they only touch
/// driver-owned allocations. Pinned host→device copies, CUDA-owned memsets, and
/// DtoD copies are also capture-safe: the captured node bakes the host/device
/// pointer, matching D3D12's retained `CopyBufferRegion` from an UPLOAD heap.
/// Launches that would write imported D3D12/external surface scratch are rewritten
/// onto CUDA-owned staging before this check (with a `CopyTexture` export left in
/// the non-capturable tail).
pub(super) fn op_is_graph_safe(op: &CudaOp) -> bool {
    match op {
        CudaOp::Launch {
            keep_alive_textures,
            graph_capture_ok,
            ..
        }
        | CudaOp::LaunchIndirect {
            keep_alive_textures,
            graph_capture_ok,
            ..
        } => *graph_capture_ok && !keep_alive_textures.iter().any(|tex| tex.is_imported()),
        CudaOp::Clear { .. } | CudaOp::WriteFromHost { .. } => true,
        // Same-allocation copies allocate scratch during execute, which is not capturable.
        CudaOp::Copy { src, dst, .. } => !Arc::ptr_eq(src, dst),
        CudaOp::WriteTextureFromHost { texture, .. } | CudaOp::CopyBufferToTexture { texture, .. } => {
            !texture.is_imported()
        }
        CudaOp::CopyTexture { src, dst } => !src.is_imported() && !dst.is_imported(),
        // Inline `Write`/`WriteTexture` own a `Vec<u8>` whose address is not a
        // stable pinned allocation. External fences are frame-varying.
        _ => false,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn ops_are_graph_safe(ops: &[CudaOp]) -> bool {
    !ops.is_empty() && ops.iter().all(op_is_graph_safe)
}

/// Split ops into maximal alternating graph-safe and stream-replay segments.
///
/// Contiguous graph-safe launches become [`CudaOpSegment::Graph`]; everything else
/// coalesces into [`CudaOpSegment::Stream`]. Empty input yields an empty vec.
pub(super) fn partition_ops_into_segments(ops: Vec<CudaOp>) -> Vec<CudaOpSegment> {
    if ops.is_empty() {
        return Vec::new();
    }
    let mut segments = Vec::new();
    let mut current: Option<(bool, Vec<CudaOp>)> = None;
    for op in ops {
        let safe = op_is_graph_safe(&op);
        match current.as_mut() {
            Some((is_safe, buf)) if *is_safe == safe => buf.push(op),
            _ => {
                if let Some((was_safe, buf)) = current.take() {
                    segments.push(if was_safe {
                        CudaOpSegment::Graph(buf)
                    } else {
                        CudaOpSegment::Stream(buf)
                    });
                }
                current = Some((safe, vec![op]));
            }
        }
    }
    if let Some((was_safe, buf)) = current {
        segments.push(if was_safe {
            CudaOpSegment::Graph(buf)
        } else {
            CudaOpSegment::Stream(buf)
        });
    }
    segments
}

/// Flatten every stream segment's ops (for epoch bumps / memory touch checks).
pub(super) fn collect_stream_ops(segments: &[CudaOpSegment]) -> Vec<CudaOp> {
    segments
        .iter()
        .filter_map(|seg| match seg {
            CudaOpSegment::Stream(ops) => Some(ops.iter().cloned()),
            CudaOpSegment::Graph(_) => None,
        })
        .flatten()
        .collect()
}

/// Build a retained relaunch plan without cloning graph-island op payloads.
pub(super) fn to_launch_segments(segments: &[CudaOpSegment]) -> Vec<CudaLaunchSegment> {
    segments
        .iter()
        .map(|segment| match segment {
            CudaOpSegment::Graph(_) => CudaLaunchSegment::Graph,
            CudaOpSegment::Stream(ops) => CudaLaunchSegment::Stream(ops.clone()),
        })
        .collect()
}

/// Flatten all ops across segments (graph + stream), preserving order.
pub(super) fn flatten_segment_ops(segments: &[CudaOpSegment]) -> Vec<CudaOp> {
    segments.iter().flat_map(|seg| seg.ops().iter().cloned()).collect()
}

/// Merge adjacent [`CudaOpSegment::Stream`] runs without reclassifying graph islands.
///
/// Used after demoting externally-touched graph islands to stream so demoted launches
/// stay on the stream path even though `op_is_graph_safe` would still return true.
pub(super) fn coalesce_adjacent_stream_segments(segments: Vec<CudaOpSegment>) -> Vec<CudaOpSegment> {
    let mut out: Vec<CudaOpSegment> = Vec::with_capacity(segments.len());
    for segment in segments {
        match (out.last_mut(), segment) {
            (Some(CudaOpSegment::Stream(prev)), CudaOpSegment::Stream(next)) => {
                prev.extend(next);
            }
            (_, segment) => out.push(segment),
        }
    }
    out
}

/// Number of graph islands in `segments`.
pub(super) fn graph_island_count(segments: &[CudaOpSegment]) -> usize {
    segments.iter().filter(|s| s.is_graph()).count()
}

/// Mutable handle to the last stream segment (for DX12 fence injection). Creates an
/// empty trailing stream segment if the program ends on a graph island.
pub(super) fn last_stream_segment_mut(segments: &mut Vec<CudaOpSegment>) -> &mut Vec<CudaOp> {
    if !matches!(segments.last(), Some(CudaOpSegment::Stream(_))) {
        segments.push(CudaOpSegment::Stream(Vec::new()));
    }
    match segments.last_mut() {
        Some(CudaOpSegment::Stream(ops)) => ops,
        _ => unreachable!("just ensured a trailing Stream segment"),
    }
}

/// Same as [`last_stream_segment_mut`] for retained relaunch plans.
pub(super) fn last_launch_stream_segment_mut(segments: &mut Vec<CudaLaunchSegment>) -> &mut Vec<CudaOp> {
    if !matches!(segments.last(), Some(CudaLaunchSegment::Stream(_))) {
        segments.push(CudaLaunchSegment::Stream(Vec::new()));
    }
    match segments.last_mut() {
        Some(CudaLaunchSegment::Stream(ops)) => ops,
        _ => unreachable!("just ensured a trailing Stream segment"),
    }
}

pub(super) fn strip_external_fence_ops(ops: Vec<CudaOp>) -> Vec<CudaOp> {
    ops.into_iter()
        .filter(|op| {
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            {
                !matches!(
                    op,
                    CudaOp::WaitExternalFence { .. } | CudaOp::SignalExternalFence { .. }
                )
            }
            #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
            {
                let _ = op;
                true
            }
        })
        .collect()
}

#[cfg(test)]
mod graph_safe_tests {
    use super::*;

    #[test]
    fn empty_ops_are_not_graph_safe() {
        assert!(!ops_are_graph_safe(&[]));
    }

    #[test]
    fn clear_ops_are_graph_safe() {
        let Ok(ctx) = cudarc::driver::CudaContext::new(0) else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let stream = ctx.default_stream();
        let Ok(slice) = stream.alloc_zeros::<u8>(16) else {
            eprintln!("skip: alloc failed");
            return;
        };
        let op = CudaOp::Clear {
            memory: Arc::new(Mutex::new(slice)),
            abs_offset: 0,
            size: 16,
            device_ptr: 0,
        };
        assert!(ops_are_graph_safe(&[op]));
    }

    fn make_clear() -> CudaOp {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device");
        let stream = ctx.default_stream();
        let slice = stream.alloc_zeros::<u8>(16).expect("alloc");
        CudaOp::Clear {
            memory: Arc::new(Mutex::new(slice)),
            abs_offset: 0,
            size: 16,
            device_ptr: 0,
        }
    }

    /// Build a Launch marked graph-safe without needing a real module load.
    /// Uses a clear as a stand-in "unsafe" op and synthesizes safe/unsafe via Clear only
    /// for partitioning tests that don't need real launches.
    #[test]
    fn partition_empty_yields_empty() {
        assert!(partition_ops_into_segments(Vec::new()).is_empty());
    }

    #[test]
    fn partition_clears_form_one_graph_island() {
        let Ok(_) = cudarc::driver::CudaContext::new(0) else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let segments = partition_ops_into_segments(vec![make_clear(), make_clear()]);
        assert_eq!(segments.len(), 1);
        assert!(matches!(&segments[0], CudaOpSegment::Graph(ops) if ops.len() == 2));
    }

    #[test]
    fn partition_clear_and_copy_form_one_graph_island() {
        let Ok(ctx) = cudarc::driver::CudaContext::new(0) else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let stream = ctx.default_stream();
        let Ok(a) = stream.alloc_zeros::<u8>(16) else {
            return;
        };
        let Ok(b) = stream.alloc_zeros::<u8>(16) else {
            return;
        };
        let mem_a = Arc::new(Mutex::new(a));
        let mem_b = Arc::new(Mutex::new(b));
        let clear = CudaOp::Clear {
            memory: Arc::clone(&mem_a),
            abs_offset: 0,
            size: 16,
            device_ptr: 0,
        };
        let copy = CudaOp::Copy {
            src: Arc::clone(&mem_a),
            src_abs: 0,
            src_ptr: 0,
            dst: Arc::clone(&mem_b),
            dst_abs: 0,
            dst_ptr: 0,
            size: 16,
        };
        let segments = partition_ops_into_segments(vec![clear, copy]);
        assert_eq!(
            segments.len(),
            1,
            "adjacent graph-safe ops must coalesce into one island"
        );
        assert!(matches!(&segments[0], CudaOpSegment::Graph(ops) if ops.len() == 2));
    }

    #[test]
    fn last_stream_segment_mut_appends_when_ending_on_graph() {
        // Graph segment with empty ops is unusual but exercises the helper.
        let mut segments = vec![CudaOpSegment::Graph(Vec::new())];
        let stream = last_stream_segment_mut(&mut segments);
        assert!(stream.is_empty());
        assert_eq!(segments.len(), 2);
        assert!(matches!(segments[1], CudaOpSegment::Stream(_)));
    }
}

pub(super) fn ops_contain_indirect(ops: &[CudaOp]) -> bool {
    ops.iter().any(|op| matches!(op, CudaOp::LaunchIndirect { .. }))
}

pub(super) fn collect_pins(
    ops: &[CudaOp],
) -> (
    Vec<Arc<Mutex<CudaSlice<u8>>>>,
    Vec<Arc<CudaModule>>,
    Vec<Arc<super::texture::CudaTextureResource>>,
    Vec<Arc<Mutex<super::pinned_host::CudaPinnedHost>>>,
) {
    let mut buffers = Vec::new();
    let mut modules = Vec::new();
    let mut textures = Vec::new();
    let mut hosts = Vec::new();
    for op in ops {
        match op {
            CudaOp::Launch {
                module,
                keep_alive_buffers,
                keep_alive_textures,
                ..
            } => {
                modules.push(Arc::clone(module));
                buffers.extend(keep_alive_buffers.iter().cloned());
                textures.extend(keep_alive_textures.iter().cloned());
            }
            CudaOp::LaunchIndirect {
                module,
                updater_module,
                keep_alive_buffers,
                keep_alive_textures,
                shape_memory,
                ..
            } => {
                modules.push(Arc::clone(module));
                modules.push(Arc::clone(updater_module));
                buffers.extend(keep_alive_buffers.iter().cloned());
                buffers.push(Arc::clone(shape_memory));
                textures.extend(keep_alive_textures.iter().cloned());
            }
            CudaOp::Clear { memory, .. } | CudaOp::Write { memory, .. } => {
                buffers.push(Arc::clone(memory));
            }
            CudaOp::WriteFromHost { memory, host, .. } => {
                buffers.push(Arc::clone(memory));
                hosts.push(Arc::clone(host));
            }
            CudaOp::Copy { src, dst, .. } => {
                buffers.push(Arc::clone(src));
                buffers.push(Arc::clone(dst));
            }
            CudaOp::WriteTexture { texture, .. } => {
                textures.push(Arc::clone(texture));
            }
            CudaOp::WriteTextureFromHost { texture, host, .. } => {
                textures.push(Arc::clone(texture));
                hosts.push(Arc::clone(host));
            }
            CudaOp::CopyBufferToTexture { src, texture, .. } => {
                buffers.push(Arc::clone(src));
                textures.push(Arc::clone(texture));
            }
            CudaOp::CopyTexture { src, dst } => {
                textures.push(Arc::clone(src));
                textures.push(Arc::clone(dst));
            }
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            CudaOp::WaitExternalFence { .. } | CudaOp::SignalExternalFence { .. } => {}
            CudaOp::CopyTextureToBuffer { texture, dst, .. } => {
                textures.push(Arc::clone(texture));
                buffers.push(Arc::clone(dst));
            }
        }
    }
    (buffers, modules, textures, hosts)
}

pub(super) fn maybe_validate_sync(stream: &Arc<CudaStream>, op: &str) -> Result<()> {
    if !crate::backend::goldy_validation_enabled() {
        return Ok(());
    }
    stream
        .synchronize()
        .with_context(|| format!("CUDA validation: {op} synchronize failed"))
}

/// Execute materialized ops on `stream`.
///
/// When `validate` is false, per-op stream synchronization is skipped (required during
/// CUDA graph capture, where host sync is illegal). Indirect launches in capture mode
/// record updater + placeholder consumer; the consumer is made device-updatable after
/// `end_capture` via [`finalize_indirect_capture`].
pub(super) fn execute_ops(stream: &Arc<CudaStream>, ops: &[CudaOp], validate: bool) -> Result<()> {
    let capturing = !validate;
    for op in ops {
        match op {
            CudaOp::Launch {
                label,
                function,
                workgroup_size,
                grid,
                args,
                ..
            } => {
                launch_direct(stream, *label, function, *workgroup_size, *grid, args, validate)?;
            }
            CudaOp::LaunchIndirect {
                label,
                function,
                workgroup_size,
                args,
                shape_ptr,
                shape_memory,
                shape_abs_offset,
                node_slot_ptr,
                status_ptr,
                updater,
                max_grid,
                max_threads_per_block,
                limits,
                ..
            } => {
                if capturing {
                    launch_indirect_for_capture(
                        stream,
                        updater,
                        function,
                        *workgroup_size,
                        args,
                        *shape_ptr,
                        *node_slot_ptr,
                        *status_ptr,
                        *max_grid,
                    )?;
                } else {
                    launch_indirect_fallback(
                        stream,
                        *label,
                        function,
                        *workgroup_size,
                        args,
                        shape_memory,
                        *shape_abs_offset,
                        *max_grid,
                        *max_threads_per_block,
                        *limits,
                        validate,
                    )?;
                }
            }
            CudaOp::Clear {
                memory,
                abs_offset,
                size,
                device_ptr,
            } => {
                if capturing {
                    capture_memset_zeros(stream, *device_ptr, *size)?;
                } else {
                    let mut guard = memory.lock().unwrap();
                    let start = *abs_offset as usize;
                    let end = start + *size as usize;
                    let mut view = guard
                        .try_slice_mut(start..end)
                        .context("CUDA: clear range out of bounds")?;
                    stream.memset_zeros(&mut view).context("CUDA: memset failed")?;
                }
                if validate {
                    maybe_validate_sync(stream, "ClearBuffer")?;
                }
            }
            CudaOp::Write {
                memory,
                abs_offset,
                data,
            } => {
                let mut guard = memory.lock().unwrap();
                let start = *abs_offset as usize;
                let end = start + data.len();
                let mut view = guard
                    .try_slice_mut(start..end)
                    .context("CUDA: write range out of bounds")?;
                stream.memcpy_htod(data, &mut view).context("CUDA: HtoD write failed")?;
                if validate {
                    maybe_validate_sync(stream, "WriteBuffer")?;
                }
            }
            CudaOp::WriteFromHost {
                memory,
                abs_offset,
                device_ptr,
                host,
                host_offset,
                len,
            } => {
                let _tz = crate::tracy_zone!("cuda.execute_op.write_from_host.lock");
                let host = host.lock().unwrap();
                let host_end = *host_offset + *len;
                if host_end > host.len() {
                    anyhow::bail!("CUDA: WriteFromHost exceeds host staging");
                }
                let src = &host.as_slice()[*host_offset..host_end];
                {
                    let _tz = crate::tracy_zone!("cuda.execute_op.write_from_host.htod");
                    if capturing {
                        capture_memcpy_htod(stream, *device_ptr, src).context("CUDA: WriteFromHost HtoD failed")?;
                    } else {
                        let mut guard = memory.lock().unwrap();
                        let start = *abs_offset as usize;
                        let end = start + *len;
                        let mut view = guard
                            .try_slice_mut(start..end)
                            .context("CUDA: WriteFromHost range out of bounds")?;
                        stream
                            .memcpy_htod(src, &mut view)
                            .context("CUDA: WriteFromHost HtoD failed")?;
                    }
                }
                if validate {
                    maybe_validate_sync(stream, "WriteFromHost")?;
                }
            }
            CudaOp::Copy {
                src,
                src_abs,
                src_ptr,
                dst,
                dst_abs,
                dst_ptr,
                size,
            } => {
                if capturing {
                    capture_memcpy_dtod(stream, *src_ptr, *dst_ptr, *size)?;
                } else {
                    execute_copy(stream, src, *src_abs, dst, *dst_abs, *size)?;
                }
                if validate {
                    maybe_validate_sync(stream, "CopyBuffer")?;
                }
            }
            CudaOp::WriteTexture {
                texture,
                x,
                y,
                width,
                height,
                data,
                src_row_pitch,
            } => {
                super::texture::memcpy_htod_array(stream, texture, *x, *y, *width, *height, data, *src_row_pitch)?;
                if validate {
                    maybe_validate_sync(stream, "WriteTexture")?;
                }
            }
            CudaOp::WriteTextureFromHost {
                texture,
                x,
                y,
                width,
                height,
                host,
                host_offset,
                len,
                src_row_pitch,
            } => {
                let _tz = crate::tracy_zone!("cuda.execute_op.write_texture_from_host.lock");
                let host = host.lock().unwrap();
                let end = *host_offset + *len;
                if end > host.len() {
                    anyhow::bail!("CUDA: WriteTextureFromHost exceeds host staging");
                }
                let data = &host.as_slice()[*host_offset..end];
                {
                    let _tz = crate::tracy_zone!("cuda.execute_op.write_texture_from_host.htod");
                    super::texture::memcpy_htod_array(stream, texture, *x, *y, *width, *height, data, *src_row_pitch)?;
                }
                if validate {
                    maybe_validate_sync(stream, "WriteTextureFromHost")?;
                }
            }
            CudaOp::CopyBufferToTexture {
                src,
                src_abs,
                src_ptr,
                src_row_pitch,
                texture,
                x,
                y,
                width,
                height,
            } => {
                let src_ptr = if capturing {
                    *src_ptr
                } else {
                    let memory = src.lock().unwrap();
                    let (base, _sync) = memory.device_ptr(stream);
                    base + *src_abs
                };
                super::texture::memcpy_dtod_array(stream, src_ptr, *src_row_pitch, texture, *x, *y, *width, *height)?;
                if validate {
                    maybe_validate_sync(stream, "CopyBufferToTexture")?;
                }
            }
            CudaOp::CopyTexture { src, dst } => {
                super::texture::memcpy_array_to_array(stream, src, dst)?;
                if validate {
                    maybe_validate_sync(stream, "CopyTexture")?;
                }
            }
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            CudaOp::WaitExternalFence {
                cuda_ctx,
                semaphore,
                value,
            } => {
                super::dx12_companion::cuda_wait_fence(cuda_ctx, semaphore.0, stream.cu_stream(), *value)?;
                if validate {
                    maybe_validate_sync(stream, "WaitExternalFence")?;
                }
            }
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            CudaOp::SignalExternalFence {
                cuda_ctx,
                semaphore,
                value,
            } => {
                super::dx12_companion::cuda_signal_fence(cuda_ctx, semaphore.0, stream.cu_stream(), *value)?;
                if validate {
                    maybe_validate_sync(stream, "SignalExternalFence")?;
                }
            }
            CudaOp::CopyTextureToBuffer {
                texture,
                x,
                y,
                width,
                height,
                dst,
                dst_abs,
                dst_ptr,
                dst_row_pitch,
            } => {
                let dst_ptr = if capturing {
                    *dst_ptr
                } else {
                    let memory = dst.lock().unwrap();
                    let (base, _sync) = memory.device_ptr(stream);
                    base + *dst_abs
                };
                super::texture::memcpy_array_to_device(
                    stream,
                    texture,
                    *x,
                    *y,
                    *width,
                    *height,
                    dst_ptr,
                    *dst_row_pitch,
                )?;
                if validate {
                    maybe_validate_sync(stream, "CopyTextureToReadback")?;
                }
            }
        }
    }
    Ok(())
}

fn launch_direct(
    stream: &Arc<CudaStream>,
    label: Option<&'static str>,
    function: &CudaFunction,
    workgroup_size: [u32; 3],
    grid: (u32, u32, u32),
    args: &[CudaLaunchArg],
    validate: bool,
) -> Result<()> {
    let where_ = label.unwrap_or("<unnamed>");
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (workgroup_size[0], workgroup_size[1], workgroup_size[2]),
        shared_mem_bytes: 0,
    };
    // SAFETY: argument order/types match the Slang CUDA entry signature.
    unsafe {
        let mut builder = stream.launch_builder(function);
        for arg in args {
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
        builder
            .launch(cfg)
            .with_context(|| format!("CUDA: cuLaunchKernel failed for dispatch '{where_}'"))?;
    }
    if validate {
        maybe_validate_sync(stream, &format!("dispatch '{where_}'"))?;
    }
    Ok(())
}

fn launch_indirect_for_capture(
    stream: &Arc<CudaStream>,
    updater: &CudaFunction,
    consumer: &CudaFunction,
    workgroup_size: [u32; 3],
    args: &[CudaLaunchArg],
    shape_ptr: u64,
    node_slot_ptr: u64,
    status_ptr: u64,
    max_grid: (u32, u32, u32),
) -> Result<()> {
    // Pass baked device pointers as POD words so cudarc does not create
    // cross-stream wait edges that invalidate THREAD_LOCAL capture.
    let shape_arg = U64Word(shape_ptr);
    let slot_arg = U64Word(node_slot_ptr);
    let status_arg = U64Word(status_ptr);
    let max_x = U32Word(max_grid.0);
    let max_y = U32Word(max_grid.1);
    let max_z = U32Word(max_grid.2);
    // SAFETY: updater signature matches goldy_apply_dispatch_shape.
    unsafe {
        stream
            .launch_builder(updater)
            .arg(&shape_arg)
            .arg(&slot_arg)
            .arg(&max_x)
            .arg(&max_y)
            .arg(&max_z)
            .arg(&status_arg)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
            .context("CUDA: updater launch failed during graph capture")?;
    }
    // Placeholder consumer grid; the updater sets the real grid before each graph launch.
    launch_direct(
        stream,
        Some("<indirect-capture>"),
        consumer,
        workgroup_size,
        (1, 1, 1),
        args,
        false,
    )
}

fn launch_indirect_fallback(
    stream: &Arc<CudaStream>,
    label: Option<&'static str>,
    function: &CudaFunction,
    workgroup_size: [u32; 3],
    args: &[CudaLaunchArg],
    shape_memory: &Arc<Mutex<CudaSlice<u8>>>,
    shape_abs_offset: u64,
    max_grid: (u32, u32, u32),
    max_threads_per_block: u32,
    limits: super::CudaDeviceLimits,
    validate: bool,
) -> Result<()> {
    let where_ = label.unwrap_or("<unnamed>");
    let mut host = [0u8; 12];
    {
        let memory = shape_memory.lock().unwrap();
        let start = shape_abs_offset as usize;
        let end = start + 12;
        let view = memory
            .try_slice(start..end)
            .context("CUDA: indirect shape range out of bounds")?;
        stream
            .memcpy_dtoh(&view, &mut host[..])
            .context("CUDA: indirect shape DtoH failed")?;
    }
    // Ensure the copy completes before reading host bytes for the launch config.
    stream
        .synchronize()
        .context("CUDA: synchronize after indirect shape DtoH failed")?;
    let shape = DispatchShape {
        x: u32::from_le_bytes(host[0..4].try_into().unwrap()),
        y: u32::from_le_bytes(host[4..8].try_into().unwrap()),
        z: u32::from_le_bytes(host[8..12].try_into().unwrap()),
    };
    let grid = (shape.x, shape.y, shape.z);
    if shape.x == 0 || shape.y == 0 || shape.z == 0 {
        tracing::trace!(
            label = where_,
            grid = ?(shape.x, shape.y, shape.z),
            "CUDA: indirect dispatch zero grid — no-op"
        );
        return Ok(());
    }
    if shape.x > max_grid.0 || shape.y > max_grid.1 || shape.z > max_grid.2 {
        anyhow::bail!(
            "CUDA: indirect dispatch '{where_}' grid ({},{},{}) exceeds device max ({},{},{})",
            shape.x,
            shape.y,
            shape.z,
            max_grid.0,
            max_grid.1,
            max_grid.2
        );
    }
    super::validate_launch_config(&limits, max_threads_per_block, grid, workgroup_size, 0, label)?;
    launch_direct(stream, label, function, workgroup_size, grid, args, validate)
}

/// After stream capture, opt each indirect consumer into device-updatable mode and
/// write the returned handles into the corresponding device slots.
pub(super) fn finalize_indirect_capture(
    stream: &Arc<CudaStream>,
    cu_graph: cudarc::driver::sys::CUgraph,
    ops: &[CudaOp],
) -> Result<()> {
    // Kernel-node ordinal among all captured kernel nodes.
    let mut kernel_ordinal = 0usize;
    for op in ops {
        match op {
            CudaOp::Launch { .. } => {
                kernel_ordinal += 1;
            }
            CudaOp::LaunchIndirect { node_slot, .. } => {
                // Updater is kernel_ordinal; consumer is kernel_ordinal + 1.
                let consumer_ordinal = kernel_ordinal + 1;
                let dev_node = retained_graph::make_kernel_node_device_updatable(cu_graph, consumer_ordinal)?;
                let mut slot = node_slot.lock().unwrap();
                stream
                    .memcpy_htod(&[dev_node as u64], &mut *slot)
                    .context("CUDA: write CUgraphDeviceNode into updater slot failed")?;
                kernel_ordinal += 2;
            }
            // Memcpy/memset nodes are not kernel nodes; skip them when counting
            // device-updatable consumers.
            CudaOp::Clear { .. }
            | CudaOp::Write { .. }
            | CudaOp::WriteFromHost { .. }
            | CudaOp::Copy { .. }
            | CudaOp::WriteTexture { .. }
            | CudaOp::WriteTextureFromHost { .. }
            | CudaOp::CopyBufferToTexture { .. }
            | CudaOp::CopyTexture { .. }
            | CudaOp::CopyTextureToBuffer { .. } => {}
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            CudaOp::WaitExternalFence { .. } | CudaOp::SignalExternalFence { .. } => {}
        }
    }
    Ok(())
}

/// Capture `ops` into a retained CUDA graph. When any op is [`CudaOp::LaunchIndirect`],
/// opts consumer nodes into device-updatable mode before instantiate.
pub(super) fn capture_partition_graph(
    stream: &Arc<CudaStream>,
    ops: &[CudaOp],
) -> Result<retained_graph::OwnedCudaGraph> {
    use cudarc::driver::sys;
    // Allocations live on the device alloc stream. `device_ptr(submit_stream)`
    // inserts `cuStreamWaitEvent` for those alloc events. Doing that *during*
    // THREAD_LOCAL capture invalidates the graph, so prime waits first.
    prime_submit_waits_for_ops(stream, ops)?;
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .context("CUDA: begin_capture failed")?;
    let capture_result = execute_ops(stream, ops, false);
    let end = unsafe { cudarc::driver::result::stream::end_capture(stream.cu_stream()) };
    if let Err(error) = &capture_result {
        let _ = stream.synchronize();
        return Err(anyhow::anyhow!("CUDA: graph capture recording failed: {error:#}"));
    }
    let cu_graph = end.context("CUDA: end_capture failed")?;
    if cu_graph.is_null() {
        anyhow::bail!("CUDA: end_capture returned an empty graph");
    }
    if ops_contain_indirect(ops) {
        if let Err(error) = finalize_indirect_capture(stream, cu_graph, ops) {
            let _ = unsafe { cudarc::driver::result::graph::destroy(cu_graph) };
            return Err(error).context("CUDA: graph finalize (device-updatable) failed");
        }
    }
    retained_graph::instantiate_owned(stream, cu_graph)
}

fn run_dynamic_prefix(
    stream: &Arc<CudaStream>,
    host_waits: &[Arc<CudaEvent>],
    deferred_writes: &[MaterializedHostWrite],
    stream_waits: &[Arc<CudaEvent>],
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))] dx12_stream_fence_waits: &[(
        Arc<Dx12Companion>,
        u64,
    )],
) -> Result<()> {
    for event in host_waits {
        timeline::host_wait_event(event)?;
    }
    for write in deferred_writes {
        let mut memory = write.memory.lock().unwrap();
        let start = write.abs_offset as usize;
        let end = start + write.data.len();
        let mut view = memory
            .try_slice_mut(start..end)
            .context("CUDA: deferred host write out of bounds")?;
        stream
            .memcpy_htod(write.data.as_ref(), &mut view)
            .context("CUDA: deferred host write HtoD failed")?;
        maybe_validate_sync(stream, "deferred host write")?;
    }
    for event in stream_waits {
        stream
            .wait(event)
            .context("CUDA: stream wait on producer event failed")?;
    }
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    for (companion, value) in dx12_stream_fence_waits {
        cuda_wait_fence(
            &companion.cuda_ctx,
            companion.cuda_semaphore,
            stream.cu_stream(),
            *value,
        )
        .with_context(|| format!("CUDA/DX12: stream wait on fence {value} failed"))?;
    }
    Ok(())
}

fn finish_submit(
    stream: &Arc<CudaStream>,
    context: &CudaSubmitContext,
    fence_value: TimelineValue,
    completion_event: &CudaEvent,
    event_ledger: &EventLedger,
) -> Result<()> {
    completion_event
        .record(stream)
        .context("CUDA: record completion event failed")?;
    timeline::mark_recorded(event_ledger, fence_value);
    context.poll_retire_events();
    Ok(())
}

impl PendingSubmit for CudaPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.cuda");
        let CudaPendingSubmit {
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
        } = *self;
        let body_result: Result<(), anyhow::Error> = (|| {
            // Hold across prefix + capture: API-thread `device_ptr` / alloc on another
            // stream during THREAD_LOCAL capture is CUDA_ERROR_STREAM_CAPTURE_ISOLATION.
            let _capture_gate = matches!(&body, CudaSubmitBody::CaptureAndLaunch { .. }).then(lock_capture_alloc_gate);
            {
                let _tz = crate::tracy_zone!("goldy.submit_worker.cuda.prefix");
                run_dynamic_prefix(
                    &stream,
                    &host_waits,
                    &deferred_writes,
                    &stream_waits,
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    &dx12_stream_fence_waits,
                )?;
            }

            match body {
                CudaSubmitBody::Ops { ops, .. } => {
                    let _tz = crate::tracy_zone!("goldy.submit_worker.cuda.execute_ops");
                    execute_ops(&stream, &ops, true)?;
                }
                CudaSubmitBody::CaptureAndLaunch {
                    key,
                    segments,
                    registry,
                    stats,
                } => {
                    let _tz = crate::tracy_zone!("goldy.submit_worker.cuda.capture_and_launch");
                    let ctx = context.handle;
                    let mut islands = Vec::new();
                    for segment in &segments {
                        match segment {
                            CudaOpSegment::Graph(ops) => {
                                let (buffers, modules, textures, hosts) = collect_pins(ops);
                                let needs_indirect = ops_contain_indirect(ops);
                                let graph = {
                                    let _tz = crate::tracy_zone!("goldy.submit_worker.cuda.capture");
                                    capture_partition_graph(&stream, ops)?
                                };
                                stats.captures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if needs_indirect {
                                    graph
                                        .upload()
                                        .context("CUDA: cuGraphUpload failed after indirect capture")?;
                                }
                                graph.launch().context("CUDA: cuGraphLaunch failed after capture")?;
                                stats.launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                maybe_validate_sync(&stream, "graph launch after capture")?;
                                islands.push(retained_graph::CudaRetainedPartition {
                                    graph,
                                    buffers,
                                    modules,
                                    textures,
                                    hosts,
                                    last_launch_tv: fence_value,
                                });
                            }
                            CudaOpSegment::Stream(ops) => {
                                if !ops.is_empty() {
                                    let _tz = crate::tracy_zone!("goldy.submit_worker.cuda.stream_segment");
                                    execute_ops(&stream, ops, true)?;
                                }
                            }
                        }
                    }
                    {
                        let _tz = crate::tracy_zone!("goldy.submit_worker.cuda.registry_insert");
                        let mut guard = registry.lock().unwrap();
                        guard.drain_retired(context.device_retired.load(std::sync::atomic::Ordering::Acquire));
                        if let Some(old) = guard.remove(ctx, key) {
                            let retire_at = old.last_launch_tv().max(fence_value);
                            guard.defer_drop(retire_at, old);
                        }
                        guard.insert(
                            ctx,
                            key,
                            retained_graph::CudaRetainedProgram {
                                islands,
                                last_launch_tv: fence_value,
                            },
                        );
                    }
                }
                CudaSubmitBody::LaunchRetained {
                    key,
                    segments,
                    registry,
                    stats,
                    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                        scratch_images: _,
                } => {
                    let _tz = crate::tracy_zone!("goldy.submit_worker.cuda.launch_retained");
                    let ctx = context.handle;
                    let mut island_idx = 0usize;
                    for segment in &segments {
                        match segment {
                            CudaLaunchSegment::Graph => {
                                let mut guard = registry.lock().unwrap();
                                guard.drain_retired(context.device_retired.load(std::sync::atomic::Ordering::Acquire));
                                let program = guard.get_mut(ctx, key).with_context(|| {
                                    format!("CUDA: retained graph missing for context {ctx} key {key:#x}")
                                })?;
                                let island = program.islands.get_mut(island_idx).with_context(|| {
                                    format!("CUDA: retained island {island_idx} missing for context {ctx} key {key:#x}")
                                })?;
                                island.graph.launch().context("CUDA: cuGraphLaunch failed")?;
                                island.last_launch_tv = fence_value;
                                program.last_launch_tv = fence_value;
                                drop(guard);
                                stats.launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                maybe_validate_sync(&stream, "retained graph launch")?;
                                island_idx += 1;
                            }
                            CudaLaunchSegment::Stream(ops) => {
                                if !ops.is_empty() {
                                    let _tz = crate::tracy_zone!("goldy.submit_worker.cuda.stream_segment");
                                    execute_ops(&stream, ops, true)?;
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        })();
        // Always publish the completion event so wait_until cannot observe
        // submitted_epoch >= tv with recorded=false (skipped/`?` before record).
        let finish_result = {
            let _tz = crate::tracy_zone!("goldy.submit_worker.cuda.finish");
            finish_submit(&stream, &context, fence_value, &completion_event, &event_ledger)
        };
        match (body_result, finish_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), _) | (_, Err(e)) => Err(e),
        }
    }
}

/// Remove a retained graph from the worker registry and defer its destruction.
pub(super) struct CudaEvictRetained {
    pub ctx: ContextHandle,
    pub key: u64,
    pub registry: Arc<Mutex<GraphRegistry>>,
    pub stats: Arc<CudaGraphStats>,
    pub device_retired: Arc<std::sync::atomic::AtomicU64>,
    /// Minimum retirement value if the graph was never launched.
    pub retire_fallback: u64,
}

impl PendingSubmit for CudaEvictRetained {
    fn execute(self: Box<Self>) -> Result<()> {
        let mut guard = self.registry.lock().unwrap();
        guard.drain_retired(self.device_retired.load(std::sync::atomic::Ordering::Acquire));
        if let Some(program) = guard.remove(self.ctx, self.key) {
            let retire_at = program.last_launch_tv().max(self.retire_fallback);
            guard.defer_drop(retire_at, program);
            self.stats.evictions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Drop all retained graphs for a context (used during context teardown).
pub(super) struct CudaEvictContextGraphs {
    pub ctx: ContextHandle,
    pub registry: Arc<Mutex<GraphRegistry>>,
    pub stats: Arc<CudaGraphStats>,
    pub device_retired: Arc<std::sync::atomic::AtomicU64>,
    pub retire_fallback: u64,
}

impl PendingSubmit for CudaEvictContextGraphs {
    fn execute(self: Box<Self>) -> Result<()> {
        let mut guard = self.registry.lock().unwrap();
        guard.drain_retired(self.device_retired.load(std::sync::atomic::Ordering::Acquire));
        for program in guard.remove_context(self.ctx) {
            let retire_at = program.last_launch_tv().max(self.retire_fallback);
            guard.defer_drop(retire_at, program);
            self.stats.evictions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Wait on the submit stream for alloc-stream buffer events *before* graph capture.
fn prime_submit_waits_for_ops(stream: &Arc<CudaStream>, ops: &[CudaOp]) -> Result<()> {
    let (buffers, _, _, _) = collect_pins(ops);
    for memory in buffers {
        let guard = memory.lock().unwrap();
        let (_ptr, _sync) = guard.device_ptr(stream);
    }
    Ok(())
}

/// Device pointer baked on the API thread before capture (no alloc-stream CUDA
/// calls from the capturing worker).
pub(super) fn bake_device_ptr(stream: &Arc<CudaStream>, memory: &Arc<Mutex<CudaSlice<u8>>>, abs_offset: u64) -> u64 {
    let _gate = lock_capture_alloc_gate();
    let guard = memory.lock().unwrap();
    let (base, _sync) = guard.device_ptr(stream);
    base + abs_offset
}

fn capture_memset_zeros(stream: &Arc<CudaStream>, device_ptr: u64, size: u64) -> Result<()> {
    if size == 0 {
        return Ok(());
    }
    unsafe { cudarc::driver::result::memset_d8_async(device_ptr, 0, size as usize, stream.cu_stream()) }
        .context("CUDA: capture memset failed")
}

fn capture_memcpy_htod(stream: &Arc<CudaStream>, device_ptr: u64, src: &[u8]) -> Result<()> {
    unsafe { cudarc::driver::result::memcpy_htod_async(device_ptr, src, stream.cu_stream()) }
        .context("CUDA: capture HtoD failed")
}

fn capture_memcpy_dtod(stream: &Arc<CudaStream>, src_ptr: u64, dst_ptr: u64, size: u64) -> Result<()> {
    if size == 0 {
        return Ok(());
    }
    unsafe { cudarc::driver::result::memcpy_dtod_async(dst_ptr, src_ptr, size as usize, stream.cu_stream()) }
        .context("CUDA: capture DtoD failed")
}

fn execute_copy(
    stream: &Arc<CudaStream>,
    src: &Arc<Mutex<CudaSlice<u8>>>,
    src_abs: u64,
    dst: &Arc<Mutex<CudaSlice<u8>>>,
    dst_abs: u64,
    size: u64,
) -> Result<()> {
    if size == 0 {
        return Ok(());
    }
    let byte_len = size as usize;
    if Arc::ptr_eq(src, dst) {
        let mut temp = stream
            .alloc_zeros::<u8>(byte_len)
            .context("CUDA: alloc overlapping-copy scratch")?;
        {
            let memory = src.lock().unwrap();
            let src_view = memory
                .try_slice(src_abs as usize..src_abs as usize + byte_len)
                .context("CUDA: copy source out of bounds")?;
            stream
                .memcpy_dtod(&src_view, &mut temp)
                .context("CUDA: same-alloc copy to scratch")?;
        }
        {
            let mut memory = dst.lock().unwrap();
            let mut dst_view = memory
                .try_slice_mut(dst_abs as usize..dst_abs as usize + byte_len)
                .context("CUDA: same-alloc copy from scratch")?;
            stream
                .memcpy_dtod(&temp, &mut dst_view)
                .context("CUDA: same-alloc copy from scratch")?;
        }
        return Ok(());
    }

    let src_ptr = Arc::as_ptr(src);
    let dst_ptr = Arc::as_ptr(dst);
    if src_ptr < dst_ptr {
        let src_guard = src.lock().unwrap();
        let mut dst_guard = dst.lock().unwrap();
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
        let mut dst_guard = dst.lock().unwrap();
        let src_guard = src.lock().unwrap();
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

/// Resolve a buffer device pointer for launch args while retaining the allocation.
#[allow(dead_code)]
pub(super) fn buffer_device_arg(
    stream: &Arc<CudaStream>,
    memory: &Arc<Mutex<CudaSlice<u8>>>,
    offset: u64,
    size: u64,
    element_stride: Option<u32>,
) -> Result<(CudaBufferArg, Arc<Mutex<CudaSlice<u8>>>)> {
    let guard = memory.lock().unwrap();
    let start = offset as usize;
    let end = (offset + size) as usize;
    let view = guard.try_slice(start..end).context("CUDA: buffer view out of range")?;
    let (ptr, _sync) = view.device_ptr(stream);
    let stride = element_stride.unwrap_or(1).max(1) as u64;
    let count = if size == 0 { 0 } else { (size / stride) as usize };
    Ok((CudaBufferArg { data: ptr, count }, Arc::clone(memory)))
}

pub(super) fn materialize_deferred_writes(
    writes: &[DeferredHostWrite],
    resolve: impl Fn(crate::backend::BufferHandle) -> Result<(Arc<Mutex<CudaSlice<u8>>>, u64)>,
) -> Result<Vec<MaterializedHostWrite>> {
    let mut out = Vec::with_capacity(writes.len());
    for write in writes {
        let (memory, base_offset) = resolve(write.buffer)?;
        out.push(MaterializedHostWrite {
            memory,
            abs_offset: base_offset + write.offset,
            data: Arc::clone(&write.data),
        });
    }
    Ok(out)
}
