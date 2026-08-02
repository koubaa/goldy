//! Owned CUDA submits executed on the per-device submission worker.

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
    /// Host waits before deferred writes / GPU enqueue.
    pub host_waits: Vec<Arc<CudaEvent>>,
    pub deferred_writes: Vec<MaterializedHostWrite>,
    pub body: CudaSubmitBody,
}

/// Body executed after the dynamic sync prefix and before the completion event.
pub(super) enum CudaSubmitBody {
    /// Immediate stream ops (standalone submits and command-replay fallback).
    Ops(Vec<CudaOp>),
    /// Capture ops into a retained CUDA graph, then launch it once.
    CaptureAndLaunch {
        key: u64,
        ops: Vec<CudaOp>,
        tail: Vec<CudaOp>,
        registry: Arc<Mutex<GraphRegistry>>,
        stats: Arc<CudaGraphStats>,
    },
    /// Launch a previously retained CUDA graph.
    LaunchRetained {
        key: u64,
        tail: Vec<CudaOp>,
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
        /// Absolute device address of the 12-byte [`DispatchShape`].
        shape_ptr: u64,
        shape_memory: Arc<Mutex<CudaSlice<u8>>>,
        shape_abs_offset: u64,
        /// Device slot written with `CUgraphDeviceNode` after capture finalize.
        node_slot_ptr: u64,
        node_slot: Arc<Mutex<CudaSlice<u64>>>,
        /// Diagnostic status word (0 = ok, -1 = oversized, else CUDA error).
        status_ptr: u64,
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
    },
    Write {
        memory: Arc<Mutex<CudaSlice<u8>>>,
        abs_offset: u64,
        data: Vec<u8>,
    },
    Copy {
        src: Arc<Mutex<CudaSlice<u8>>>,
        src_abs: u64,
        dst: Arc<Mutex<CudaSlice<u8>>>,
        dst_abs: u64,
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
    CopyBufferToTexture {
        src: Arc<Mutex<CudaSlice<u8>>>,
        src_abs: u64,
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

/// True when `ops` can be recorded into a CUDA graph without host allocation or HtoD.
///
/// Kernel launches (direct and indirect) are capture-safe when they only touch
/// driver-owned allocations. Launches that write imported D3D12/external surface
/// scratch (`cuImportExternalMemory`) are not — capture often fails at `end_capture`.
pub(super) fn op_is_graph_safe(op: &CudaOp) -> bool {
    match op {
        CudaOp::Launch {
            keep_alive_textures, ..
        }
        | CudaOp::LaunchIndirect {
            keep_alive_textures, ..
        } => !keep_alive_textures.iter().any(|tex| tex.is_imported()),
        _ => false,
    }
}

pub(super) fn ops_are_graph_safe(ops: &[CudaOp]) -> bool {
    !ops.is_empty() && ops.iter().all(op_is_graph_safe)
}

pub(super) fn split_graph_core_and_tail(ops: Vec<CudaOp>) -> (Vec<CudaOp>, Vec<CudaOp>) {
    let split = ops.iter().position(|op| !op_is_graph_safe(op)).unwrap_or(ops.len());
    let mut core = ops;
    let tail = core.split_off(split);
    (core, tail)
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
    fn clear_ops_are_not_graph_safe() {
        // Graph capture rejects clears/copies; only kernel launches without imported
        // textures are safe. A Clear with a dummy Arc is enough to hit the `_ => false` arm.
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
        };
        assert!(!ops_are_graph_safe(&[op]));
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
) {
    let mut buffers = Vec::new();
    let mut modules = Vec::new();
    let mut textures = Vec::new();
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
            CudaOp::Copy { src, dst, .. } => {
                buffers.push(Arc::clone(src));
                buffers.push(Arc::clone(dst));
            }
            CudaOp::WriteTexture { texture, .. } => {
                textures.push(Arc::clone(texture));
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
    (buffers, modules, textures)
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
            } => {
                let mut guard = memory.lock().unwrap();
                let start = *abs_offset as usize;
                let end = start + *size as usize;
                let mut view = guard
                    .try_slice_mut(start..end)
                    .context("CUDA: clear range out of bounds")?;
                stream.memset_zeros(&mut view).context("CUDA: memset failed")?;
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
            CudaOp::Copy {
                src,
                src_abs,
                dst,
                dst_abs,
                size,
            } => {
                execute_copy(stream, src, *src_abs, dst, *dst_abs, *size)?;
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
            CudaOp::CopyBufferToTexture {
                src,
                src_abs,
                src_row_pitch,
                texture,
                x,
                y,
                width,
                height,
            } => {
                let src_ptr = {
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
                dst_row_pitch,
            } => {
                let dst_ptr = {
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
            CudaOp::Clear { .. }
            | CudaOp::Write { .. }
            | CudaOp::Copy { .. }
            | CudaOp::WriteTexture { .. }
            | CudaOp::CopyBufferToTexture { .. }
            | CudaOp::CopyTexture { .. }
            | CudaOp::CopyTextureToBuffer { .. } => {
                anyhow::bail!("CUDA: finalize_indirect_capture called on a graph-unsafe op set");
            }
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            CudaOp::WaitExternalFence { .. } | CudaOp::SignalExternalFence { .. } => {
                anyhow::bail!("CUDA: finalize_indirect_capture called on a graph-unsafe op set");
            }
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
    timeline::poll_retire_events(
        event_ledger,
        &context.completed,
        context.handle,
        &context.device_retired,
        &context.signal_queue,
        &context.last_emitted,
    );
    Ok(())
}

impl PendingSubmit for CudaPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        run_dynamic_prefix(
            &self.stream,
            &self.host_waits,
            &self.deferred_writes,
            &self.stream_waits,
        )?;

        match self.body {
            CudaSubmitBody::Ops(ops) => {
                execute_ops(&self.stream, &ops, true)?;
            }
            CudaSubmitBody::CaptureAndLaunch {
                key,
                ops,
                tail,
                registry,
                stats,
            } => {
                let ctx = self.context.handle;
                let (buffers, modules, textures) = collect_pins(&ops);
                let needs_indirect = ops_contain_indirect(&ops);
                let graph = capture_partition_graph(&self.stream, &ops)?;
                stats.captures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                {
                    let mut guard = registry.lock().unwrap();
                    guard.drain_retired(self.context.device_retired.load(std::sync::atomic::Ordering::Acquire));
                    if let Some(old) = guard.remove(ctx, key) {
                        let retire_at = old.last_launch_tv.max(self.fence_value);
                        guard.defer_drop(retire_at, old);
                    }
                    guard.insert(
                        ctx,
                        key,
                        retained_graph::CudaRetainedPartition {
                            graph,
                            buffers,
                            modules,
                            textures,
                            last_launch_tv: self.fence_value,
                        },
                    );
                    let partition = guard
                        .get_mut(ctx, key)
                        .context("CUDA: retained graph missing after capture")?;
                    if needs_indirect {
                        partition
                            .graph
                            .upload()
                            .context("CUDA: cuGraphUpload failed after indirect capture")?;
                    }
                    partition
                        .graph
                        .launch()
                        .context("CUDA: cuGraphLaunch failed after capture")?;
                }
                stats.launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                maybe_validate_sync(&self.stream, "graph launch after capture")?;
                execute_ops(&self.stream, &tail, true)?;
            }
            CudaSubmitBody::LaunchRetained {
                key,
                tail,
                registry,
                stats,
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                    scratch_images: _,
            } => {
                let ctx = self.context.handle;
                {
                    let mut guard = registry.lock().unwrap();
                    guard.drain_retired(self.context.device_retired.load(std::sync::atomic::Ordering::Acquire));
                    let partition = guard
                        .get_mut(ctx, key)
                        .with_context(|| format!("CUDA: retained graph missing for context {ctx} key {key:#x}"))?;
                    partition.graph.launch().context("CUDA: cuGraphLaunch failed")?;
                    partition.last_launch_tv = self.fence_value;
                }
                stats.launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                maybe_validate_sync(&self.stream, "retained graph launch")?;
                execute_ops(&self.stream, &tail, true)?;
            }
        }

        finish_submit(
            &self.stream,
            &self.context,
            self.fence_value,
            &self.completion_event,
            &self.event_ledger,
        )
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
        if let Some(partition) = guard.remove(self.ctx, self.key) {
            let retire_at = partition.last_launch_tv.max(self.retire_fallback);
            guard.defer_drop(retire_at, partition);
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
        for partition in guard.remove_context(self.ctx) {
            let retire_at = partition.last_launch_tv.max(self.retire_fallback);
            guard.defer_drop(retire_at, partition);
            self.stats.evictions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
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
