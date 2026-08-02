//! Owned CUDA submits executed on the per-device submission worker.

use super::retained_graph::{self, CudaGraphStats, GraphRegistry};
use super::timeline::{self, EventLedger};
use super::{CudaBufferArg, CudaLaunchArg, CudaSubmitContext};
use crate::backend::submission_worker::PendingSubmit;
use crate::backend::{ContextHandle, DeferredHostWrite};
use crate::timeline::TimelineValue;
use anyhow::{Context as _, Result};
use cudarc::driver::{
    CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use std::sync::{Arc, Mutex};

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
        registry: Arc<Mutex<GraphRegistry>>,
        stats: Arc<CudaGraphStats>,
    },
    /// Launch a previously retained CUDA graph.
    LaunchRetained {
        key: u64,
        registry: Arc<Mutex<GraphRegistry>>,
        stats: Arc<CudaGraphStats>,
    },
}

pub(super) struct MaterializedHostWrite {
    pub memory: Arc<Mutex<CudaSlice<u8>>>,
    pub abs_offset: u64,
    pub data: Arc<[u8]>,
}

pub(super) enum CudaOp {
    Launch {
        label: Option<&'static str>,
        function: CudaFunction,
        module: Arc<CudaModule>,
        workgroup_size: [u32; 3],
        grid: (u32, u32, u32),
        args: Vec<CudaLaunchArg>,
        keep_alive: Vec<Arc<Mutex<CudaSlice<u8>>>>,
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
}

/// True when `ops` can be recorded into a CUDA graph without host allocation or HtoD.
///
/// Only kernel launches are captured initially. `memset` / `memcpy` via cudarc have been
/// observed to invalidate `THREAD_LOCAL` stream capture on this driver stack, so clears and
/// copies stay on the command-replay path until an explicit-graph or capture-safe copy path lands.
pub(super) fn ops_are_graph_safe(ops: &[CudaOp]) -> bool {
    if ops.is_empty() {
        return false;
    }
    ops.iter().all(|op| matches!(op, CudaOp::Launch { .. }))
}

pub(super) fn collect_pins(
    ops: &[CudaOp],
) -> (Vec<Arc<Mutex<CudaSlice<u8>>>>, Vec<Arc<CudaModule>>) {
    let mut buffers = Vec::new();
    let mut modules = Vec::new();
    for op in ops {
        match op {
            CudaOp::Launch {
                module,
                keep_alive,
                ..
            } => {
                modules.push(Arc::clone(module));
                buffers.extend(keep_alive.iter().cloned());
            }
            CudaOp::Clear { memory, .. } | CudaOp::Write { memory, .. } => {
                buffers.push(Arc::clone(memory));
            }
            CudaOp::Copy { src, dst, .. } => {
                buffers.push(Arc::clone(src));
                buffers.push(Arc::clone(dst));
            }
        }
    }
    (buffers, modules)
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
/// CUDA graph capture, where host sync is illegal).
pub(super) fn execute_ops(stream: &Arc<CudaStream>, ops: &[CudaOp], validate: bool) -> Result<()> {
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
                let where_ = label.unwrap_or("<unnamed>");
                let cfg = LaunchConfig {
                    grid_dim: *grid,
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
                stream
                    .memcpy_htod(data, &mut view)
                    .context("CUDA: HtoD write failed")?;
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
        }
    }
    Ok(())
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
                registry,
                stats,
            } => {
                let ctx = self.context.handle;
                let (buffers, modules) = collect_pins(&ops);
                let graph = retained_graph::capture_ops_to_graph(&self.stream, || {
                    execute_ops(&self.stream, &ops, false)
                })?;
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
                            last_launch_tv: self.fence_value,
                        },
                    );
                    let partition = guard
                        .get_mut(ctx, key)
                        .context("CUDA: retained graph missing after capture")?;
                    partition
                        .graph
                        .launch()
                        .context("CUDA: cuGraphLaunch failed after capture")?;
                }
                stats.launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                maybe_validate_sync(&self.stream, "graph launch after capture")?;
            }
            CudaSubmitBody::LaunchRetained {
                key,
                registry,
                stats,
            } => {
                let ctx = self.context.handle;
                {
                    let mut guard = registry.lock().unwrap();
                    guard.drain_retired(self.context.device_retired.load(std::sync::atomic::Ordering::Acquire));
                    let partition = guard.get_mut(ctx, key).with_context(|| {
                        format!("CUDA: retained graph missing for context {ctx} key {key:#x}")
                    })?;
                    partition
                        .graph
                        .launch()
                        .context("CUDA: cuGraphLaunch failed")?;
                    partition.last_launch_tv = self.fence_value;
                }
                stats.launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                maybe_validate_sync(&self.stream, "retained graph launch")?;
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
            self.stats
                .evictions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
            self.stats
                .evictions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    Ok((
        CudaBufferArg { data: ptr, count },
        Arc::clone(memory),
    ))
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
