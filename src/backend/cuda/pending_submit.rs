//! Owned CUDA submits executed on the per-device submission worker.

use super::timeline::{self, EventLedger};
use super::{CudaBufferArg, CudaLaunchArg, CudaSubmitContext};
use crate::backend::submission_worker::PendingSubmit;
use crate::backend::DeferredHostWrite;
use crate::timeline::TimelineValue;
use anyhow::{Context as _, Result};
use cudarc::driver::{CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
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
    pub ops: Vec<CudaOp>,
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
        #[allow(dead_code)]
        module: Arc<CudaModule>,
        workgroup_size: [u32; 3],
        grid: (u32, u32, u32),
        args: Vec<CudaLaunchArg>,
        #[allow(dead_code)]
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

pub(super) fn maybe_validate_sync(stream: &Arc<CudaStream>, op: &str) -> Result<()> {
    if !crate::backend::goldy_validation_enabled() {
        return Ok(());
    }
    stream
        .synchronize()
        .with_context(|| format!("CUDA validation: {op} synchronize failed"))
}

impl PendingSubmit for CudaPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        for event in &self.host_waits {
            timeline::host_wait_event(event)?;
        }
        for write in &self.deferred_writes {
            let mut memory = write.memory.lock().unwrap();
            let start = write.abs_offset as usize;
            let end = start + write.data.len();
            let mut view = memory
                .try_slice_mut(start..end)
                .context("CUDA: deferred host write out of bounds")?;
            self.stream
                .memcpy_htod(write.data.as_ref(), &mut view)
                .context("CUDA: deferred host write HtoD failed")?;
            maybe_validate_sync(&self.stream, "deferred host write")?;
        }
        for event in &self.stream_waits {
            self.stream
                .wait(event)
                .context("CUDA: stream wait on producer event failed")?;
        }

        for op in &self.ops {
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
                        let mut builder = self.stream.launch_builder(function);
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
                    maybe_validate_sync(&self.stream, &format!("dispatch '{where_}'"))?;
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
                    self.stream.memset_zeros(&mut view).context("CUDA: memset failed")?;
                    maybe_validate_sync(&self.stream, "ClearBuffer")?;
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
                    self.stream
                        .memcpy_htod(data, &mut view)
                        .context("CUDA: HtoD write failed")?;
                    maybe_validate_sync(&self.stream, "WriteBuffer")?;
                }
                CudaOp::Copy {
                    src,
                    src_abs,
                    dst,
                    dst_abs,
                    size,
                } => {
                    execute_copy(&self.stream, src, *src_abs, dst, *dst_abs, *size)?;
                    maybe_validate_sync(&self.stream, "CopyBuffer")?;
                }
            }
        }

        self.completion_event
            .record(&self.stream)
            .context("CUDA: record completion event failed")?;
        timeline::mark_recorded(&self.event_ledger, self.fence_value);

        // Eager poll so same-thread follow-up waits see progress without relying solely on the poller.
        timeline::poll_retire_events(
            &self.event_ledger,
            &self.context.completed,
            self.context.handle,
            &self.context.device_retired,
            &self.context.signal_queue,
            &self.context.last_emitted,
        );
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
