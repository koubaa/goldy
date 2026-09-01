//! Compute-only CPU backend (`GOLDY_BACKEND=cpu`).
//!
//! Compiles `[goldy_compute]` kernels with the same host-callable JIT as
//! [`crate::cpu_shaders`] and executes scheme submits against host parcels.
//! Textures, samplers, and graphics stages are not supported.

use super::*;
use crate::cpu_shaders::{compile_shader, CpuComputeKernel, CpuHostBufferView, CpuParamSlot};
use crate::slang::compiler::SlangCompiler;
use crate::types::{BufferFlags, BufferKind, DeviceType, ResourceAccess, ResourceCategory};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

fn cpu_graphics_unsupported<T>() -> Result<T> {
    anyhow::bail!("CPU backend is compute-only (no textures, samplers, surfaces, or raster)")
}

struct CpuDevice {
    timeline_next: Arc<std::sync::atomic::AtomicU64>,
    submission_worker: Arc<crate::backend::submission_worker::SubmissionWorker>,
}

struct CpuPendingSubmit {
    tv: u64,
    context_state: Arc<Mutex<CpuContextState>>,
}

impl crate::backend::submission_worker::PendingSubmit for CpuPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let mut state = self.context_state.lock().unwrap();
        state.completed = self.tv;
        state.signal_queue.push_boundary_crossed(self.tv);
        Ok(())
    }
}

struct CpuContextState {
    device: DeviceHandle,
    completed: u64,
    last_submitted_seq: u64,
    signal_queue: crate::signal::SignalQueue,
}

struct CpuContextDestroyHandle;

impl ContextDestroyHandle for CpuContextDestroyHandle {
    fn wait(&self) -> Result<()> {
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

struct CpuContextGpuProgress {
    state: Arc<Mutex<CpuContextState>>,
}

impl crate::backend::ContextGpuProgress for CpuContextGpuProgress {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue {
        self.state.lock().unwrap().completed
    }
}

struct CpuBuffer {
    device_handle: DeviceHandle,
    size: u64,
    alloc_size: u64,
    data: Option<Vec<u8>>,
    parent: Option<(BufferHandle, u64)>,
    bindless_index: u32,
    is_withdraw_staging: bool,
}

struct CpuShader {
    device_handle: DeviceHandle,
    source: String,
    search_paths: Vec<String>,
    defines: Vec<(String, String)>,
    optimization_level: crate::types::OptimizationLevel,
}

struct CpuComputePipeline {
    device_handle: DeviceHandle,
    kernel: Arc<CpuComputeKernel>,
    slot_access: Vec<Option<ResourceAccess>>,
}

/// Host-callable compute backend. Never selected as a platform default.
pub(crate) struct CpuBackend {
    adapters: Vec<AdapterInfo>,
    devices: HashMap<DeviceHandle, CpuDevice>,
    next_device_handle: DeviceHandle,
    buffers: HashMap<BufferHandle, CpuBuffer>,
    next_buffer_handle: BufferHandle,
    shaders: HashMap<ShaderHandle, CpuShader>,
    next_shader_handle: ShaderHandle,
    compute_pipelines: HashMap<ComputePipelineHandle, CpuComputePipeline>,
    next_compute_pipeline_handle: ComputePipelineHandle,
    next_bindless_index: u32,
    bindless_to_buffer: HashMap<u32, BufferHandle>,
    contexts: HashMap<ContextHandle, Arc<Mutex<CpuContextState>>>,
    next_context_id: ContextHandle,
    device_retired_floor: HashMap<DeviceHandle, Arc<std::sync::atomic::AtomicU64>>,
    retained_graphs: HashMap<(ContextHandle, u64), Vec<GraphCommand>>,
    slang: SlangCompiler,
}

impl CpuBackend {
    pub fn new() -> Result<Self> {
        Ok(Self {
            adapters: vec![AdapterInfo {
                id: 0,
                name: "Goldy CPU (host-callable)".to_string(),
                vendor: "Goldy".to_string(),
                backend: BackendType::Cpu,
                device_type: DeviceType::Cpu,
            }],
            devices: HashMap::new(),
            next_device_handle: 1,
            buffers: HashMap::new(),
            next_buffer_handle: 1,
            shaders: HashMap::new(),
            next_shader_handle: 1,
            compute_pipelines: HashMap::new(),
            next_compute_pipeline_handle: 1,
            next_bindless_index: 0,
            bindless_to_buffer: HashMap::new(),
            contexts: HashMap::new(),
            next_context_id: 1,
            device_retired_floor: HashMap::new(),
            retained_graphs: HashMap::new(),
            slang: SlangCompiler::new().context("CPU backend: failed to load Slang")?,
        })
    }

    fn context_state(&self, ctx: ContextHandle) -> std::sync::MutexGuard<'_, CpuContextState> {
        self.contexts.get(&ctx).expect("invalid context handle").lock().unwrap()
    }

    fn device_retired(&self, device: DeviceHandle) -> u64 {
        let floor = self
            .device_retired_floor
            .get(&device)
            .map(|f| f.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0);
        let max_ctx = self
            .contexts
            .values()
            .filter(|c| c.lock().unwrap().device == device)
            .map(|c| c.lock().unwrap().completed)
            .max()
            .unwrap_or(0);
        floor.max(max_ctx)
    }

    fn complete_device_seq_on_all_contexts(&mut self, device: DeviceHandle, seq: u64) {
        for ctx in self.contexts.values() {
            let mut state = ctx.lock().unwrap();
            if state.device == device {
                state.completed = seq;
                state.last_submitted_seq = seq;
            }
        }
    }

    fn pending_submit(&self, ctx: ContextHandle, tv: u64) -> Result<CpuPendingSubmit> {
        let context_state = Arc::clone(
            self.contexts
                .get(&ctx)
                .ok_or_else(|| anyhow::anyhow!("Invalid context handle"))?,
        );
        Ok(CpuPendingSubmit { tv, context_state })
    }

    fn execute_submit_immediately(&self, ctx: ContextHandle, tv: u64) -> Result<()> {
        let device = self.context_device(ctx);
        let dev = self
            .devices
            .get(&device)
            .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?;
        dev.submission_worker
            .execute_immediately(tv, Box::new(self.pending_submit(ctx, tv)?))
    }

    fn await_submit(&self, ctx: ContextHandle, tv: u64) -> Result<()> {
        let device = self.context_device(ctx);
        let dev = self
            .devices
            .get(&device)
            .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?;
        dev.submission_worker.wait_submitted(tv)
    }

    fn scheduled_horizon(&self, device: DeviceHandle) -> u64 {
        self.devices
            .get(&device)
            .map(|d| {
                d.timeline_next
                    .load(std::sync::atomic::Ordering::Acquire)
                    .saturating_sub(1)
            })
            .unwrap_or(0)
    }

    fn apply_submit_sync(&mut self, sync: Option<&SubmitSync>) -> Result<()> {
        if let Some(s) = sync {
            for epoch in &s.waits {
                self.wait_until(epoch.context, epoch.value)?;
            }
            for epoch in &s.host_observed_waits {
                self.wait_until(epoch.context, epoch.value)?;
            }
            for write in &s.deferred_host_writes {
                self.write_buffer(write.buffer, write.offset, &write.data)?;
            }
        }
        Ok(())
    }

    fn resolve_root(&self, handle: BufferHandle) -> Result<(BufferHandle, u64, u64)> {
        let buf = self
            .buffers
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;
        let size = buf.size;
        if let Some((parent, offset)) = buf.parent {
            let (root, parent_off, _) = self.resolve_root(parent)?;
            Ok((root, parent_off + offset, size))
        } else {
            Ok((handle, 0, size))
        }
    }

    fn bytes_mut(&mut self, handle: BufferHandle) -> Result<&mut [u8]> {
        let (root, offset, size) = self.resolve_root(handle)?;
        let buf = self
            .buffers
            .get_mut(&root)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;
        let data = buf
            .data
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("CPU: buffer view has no owned storage"))?;
        let start = offset as usize;
        let end = start + size as usize;
        if end > data.len() {
            anyhow::bail!("CPU: buffer range exceeds allocation");
        }
        Ok(&mut data[start..end])
    }

    fn bytes(&self, handle: BufferHandle) -> Result<&[u8]> {
        let (root, offset, size) = self.resolve_root(handle)?;
        let buf = self
            .buffers
            .get(&root)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;
        let data = buf
            .data
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CPU: buffer view has no owned storage"))?;
        let start = offset as usize;
        let end = start + size as usize;
        Ok(&data[start..end])
    }

    fn host_view_for_bindless(&mut self, index: u32, stride: u32) -> Result<CpuHostBufferView> {
        let handle = *self
            .bindless_to_buffer
            .get(&index)
            .with_context(|| format!("CPU: unknown bindless index {index}"))?;
        let bytes = self.bytes_mut(handle)?;
        Ok(CpuHostBufferView {
            data: bytes.as_mut_ptr(),
            len: bytes.len(),
            stride,
        })
    }

    fn alloc_owned_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        alloc_size: u64,
        bindless: bool,
        is_withdraw_staging: bool,
    ) -> Result<BufferHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;
        let bindless_index = if bindless {
            let index = self.next_bindless_index;
            self.next_bindless_index += 1;
            self.bindless_to_buffer.insert(index, handle);
            index
        } else {
            0
        };
        let cap = alloc_size.max(size);
        self.buffers.insert(
            handle,
            CpuBuffer {
                device_handle: device,
                size,
                alloc_size: cap,
                data: Some(vec![0u8; cap as usize]),
                parent: None,
                bindless_index,
                is_withdraw_staging,
            },
        );
        Ok(handle)
    }

    fn execute_copy_buffer(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        let copy_len = size as usize;
        let src_data = {
            let src_bytes = self.bytes(src)?;
            if src_offset.saturating_add(copy_len as u64) > src_bytes.len() as u64 {
                anyhow::bail!("CPU CopyBuffer: size exceeds src bounds");
            }
            let start = src_offset as usize;
            src_bytes[start..start + copy_len].to_vec()
        };
        let dst_bytes = self.bytes_mut(dst)?;
        if dst_offset.saturating_add(copy_len as u64) > dst_bytes.len() as u64 {
            anyhow::bail!("CPU CopyBuffer: size exceeds dst bounds");
        }
        let dst_start = dst_offset as usize;
        dst_bytes[dst_start..dst_start + copy_len].copy_from_slice(&src_data);
        Ok(())
    }

    fn dispatch_kernel(
        &mut self,
        pipeline: ComputePipelineHandle,
        indices: &[u32],
        user: &[u32],
        groups: [u32; 3],
    ) -> Result<()> {
        let kernel = {
            let p = self
                .compute_pipelines
                .get(&pipeline)
                .context("CPU: invalid compute pipeline")?;
            Arc::clone(&p.kernel)
        };
        let layout = kernel.layout();
        let mut views = Vec::new();
        if layout.is_empty() {
            for &index in indices {
                views.push(self.host_view_for_bindless(index, 4)?);
            }
            if !user.is_empty() {
                anyhow::bail!(
                    "CPU: scalar user params require a [goldy_compute] entry; got {} word(s)",
                    user.len()
                );
            }
        } else {
            let mut index_i = 0usize;
            for slot in layout {
                match slot {
                    CpuParamSlot::Buffer { stride } => {
                        let index = *indices
                            .get(index_i)
                            .with_context(|| format!("CPU: missing resource index for buffer binding {index_i}"))?;
                        views.push(self.host_view_for_bindless(index, *stride)?);
                        index_i += 1;
                    }
                    CpuParamSlot::Scalar => {}
                }
            }
            if index_i != indices.len() {
                anyhow::bail!(
                    "CPU: dispatch bound {} resource index(es) but shader expects {index_i}",
                    indices.len()
                );
            }
            let expected_scalars = layout.iter().filter(|s| matches!(s, CpuParamSlot::Scalar)).count();
            if user.len() != expected_scalars {
                anyhow::bail!(
                    "CPU: dispatch provided {} scalar user word(s) but shader expects {expected_scalars}",
                    user.len()
                );
            }
        }
        kernel.dispatch_host(groups, &views, user)
    }

    fn execute_commands(&mut self, commands: &[GpuCommand]) -> Result<()> {
        let mut current_pipeline: Option<ComputePipelineHandle> = None;
        let mut current_indices: Vec<u32> = Vec::new();
        let mut current_user: Vec<u32> = Vec::new();
        let mut frame_table: Option<Arc<[u32]>> = None;

        for cmd in commands {
            match cmd {
                GpuCommand::SetPipeline(p) => current_pipeline = Some(*p),
                GpuCommand::BindResourcesRaw { indices, user, .. } => {
                    current_indices = indices.clone();
                    current_user = user.clone();
                }
                GpuCommand::FrameTableStaging { data } => {
                    frame_table = Some(Arc::clone(data));
                }
                GpuCommand::Dispatch {
                    workgroups_x,
                    workgroups_y,
                    workgroups_z,
                    ..
                } => {
                    let pipeline = current_pipeline.context("CPU: Dispatch without a compute pipeline")?;
                    self.dispatch_kernel(
                        pipeline,
                        &current_indices,
                        &current_user,
                        [*workgroups_x, *workgroups_y, *workgroups_z],
                    )?;
                }
                GpuCommand::DispatchIndirect { buffer, offset, .. } => {
                    let pipeline = current_pipeline.context("CPU: DispatchIndirect without a compute pipeline")?;
                    let bytes = self.bytes(*buffer)?;
                    let start = *offset as usize;
                    if start + 12 > bytes.len() {
                        anyhow::bail!("CPU: DispatchIndirect reads past buffer end");
                    }
                    let wg_x = u32::from_ne_bytes(bytes[start..start + 4].try_into().unwrap());
                    let wg_y = u32::from_ne_bytes(bytes[start + 4..start + 8].try_into().unwrap());
                    let wg_z = u32::from_ne_bytes(bytes[start + 8..start + 12].try_into().unwrap());
                    self.dispatch_kernel(pipeline, &current_indices, &current_user, [wg_x, wg_y, wg_z])?;
                }
                GpuCommand::DispatchBatch { arg_data, count, .. } => {
                    let pipeline = current_pipeline.context("CPU: DispatchBatch without a compute pipeline")?;
                    self.execute_dispatch_batch(pipeline, frame_table.as_deref(), arg_data, *count)?;
                }
                GpuCommand::WriteBuffer { buffer, offset, data } => {
                    self.write_buffer(*buffer, *offset, data)?;
                }
                GpuCommand::ClearBuffer { buffer, offset, size } => {
                    let device = self
                        .buffers
                        .get(buffer)
                        .map(|b| b.device_handle)
                        .context("CPU: invalid ClearBuffer handle")?;
                    self.clear_buffer(device, *buffer, *offset, *size)?;
                }
                GpuCommand::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    self.execute_copy_buffer(*src, *src_offset, *dst, *dst_offset, *size)?;
                }
                GpuCommand::ResourceBarrier { .. } => {}
                GpuCommand::WriteTexture { .. }
                | GpuCommand::WriteTextureRegion { .. }
                | GpuCommand::CopyTexture { .. }
                | GpuCommand::CopyRenderTarget { .. }
                | GpuCommand::CopyBufferToTexture { .. }
                | GpuCommand::CopyTextureToReadback { .. } => {
                    cpu_graphics_unsupported()?;
                }
            }
        }
        Ok(())
    }

    fn execute_dispatch_batch(
        &mut self,
        pipeline: ComputePipelineHandle,
        frame_table: Option<&[u32]>,
        arg_data: &[u8],
        count: u32,
    ) -> Result<()> {
        use crate::backend::shared::{PushLayout, DISPATCH_BATCH_STRIDE, MAX_USER_SLOTS, TOTAL_PUSH_BYTES};
        use crate::frame_table::dispatch_table_base_word_index;

        let kernel = {
            let p = self
                .compute_pipelines
                .get(&pipeline)
                .context("CPU: invalid compute pipeline")?;
            Arc::clone(&p.kernel)
        };
        let n_buffers = kernel
            .layout()
            .iter()
            .filter(|s| matches!(s, CpuParamSlot::Buffer { .. }))
            .count();
        let n_scalars = kernel
            .layout()
            .iter()
            .filter(|s| matches!(s, CpuParamSlot::Scalar))
            .count();
        let entry_count = count as usize;
        let needed = entry_count
            .checked_mul(DISPATCH_BATCH_STRIDE)
            .context("CPU: DispatchBatch stride overflow")?;
        anyhow::ensure!(arg_data.len() >= needed, "CPU: DispatchBatch arg_data too small");

        for i in 0..entry_count {
            let base = i * DISPATCH_BATCH_STRIDE;
            let layout: PushLayout = *bytemuck::from_bytes(&arg_data[base..base + TOTAL_PUSH_BYTES]);
            let wg_off = base + TOTAL_PUSH_BYTES;
            let wg_x = u32::from_ne_bytes(arg_data[wg_off..wg_off + 4].try_into().unwrap());
            let wg_y = u32::from_ne_bytes(arg_data[wg_off + 4..wg_off + 8].try_into().unwrap());
            let wg_z = u32::from_ne_bytes(arg_data[wg_off + 8..wg_off + 12].try_into().unwrap());
            let table_base = layout._reserved[dispatch_table_base_word_index()] as usize;
            let indices = if n_buffers == 0 {
                Vec::new()
            } else {
                let table = frame_table.context("CPU: DispatchBatch requires FrameTableStaging")?;
                let end = table_base
                    .checked_add(n_buffers)
                    .context("CPU: frame-table range overflow")?;
                anyhow::ensure!(
                    end <= table.len(),
                    "CPU: DispatchBatch frame-table range exceeds staging"
                );
                table[table_base..end].to_vec()
            };
            let user = if n_scalars == 0 {
                Vec::new()
            } else {
                anyhow::ensure!(n_scalars <= MAX_USER_SLOTS, "CPU: too many scalar words");
                layout.user[..n_scalars].to_vec()
            };
            self.dispatch_kernel(pipeline, &indices, &user, [wg_x, wg_y, wg_z])?;
        }
        Ok(())
    }
}

impl crate::backend::GpuBackendTimelineWait for CpuBackend {
    fn take_timeline_submission_epoch_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<crate::backend::submission_worker::SubmissionEpochWait>> {
        if self.gpu_progress(ctx) >= value {
            return Ok(None);
        }
        let device = self.context_device(ctx);
        let Some(dev) = self.devices.get(&device) else {
            return Ok(None);
        };
        let horizon = self.scheduled_horizon(device);
        if value == 0 || value > horizon {
            return Ok(None);
        }
        Ok(Some(crate::backend::submission_worker::SubmissionEpochWait::new(
            Arc::clone(&dev.submission_worker),
            value,
            horizon,
        )))
    }

    fn take_timeline_blocking_wait(
        &self,
        _ctx: ContextHandle,
        _value: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn crate::backend::TimelineBlockingWait>>> {
        Ok(None)
    }

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        let device = self.context_device(ctx);
        if let Some(dev) = self.devices.get(&device) {
            dev.submission_worker.flush()?;
            let horizon = self.scheduled_horizon(device);
            dev.submission_worker.wait_submitted_if_scheduled(value, horizon)?;
        }
        let cur = self.gpu_progress(ctx);
        if value > cur {
            self.context_state(ctx).completed = value;
        }
        Ok(())
    }
}

#[cfg(feature = "graphics")]
impl crate::backend::GpuBackendPresentSplit for CpuBackend {
    fn take_present_gpu_work(
        &mut self,
        _frame: FrameToken,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn crate::backend::PresentGpuWork>> {
        cpu_graphics_unsupported()
    }

    fn finish_present(
        &mut self,
        _finish: crate::backend::PresentFinishState,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        cpu_graphics_unsupported()
    }
}

impl crate::backend::GpuBackendSubmitSession for CpuBackend {
    fn clone_context_submit_session(
        &self,
        _ctx: ContextHandle,
        backend: std::sync::Arc<std::sync::Mutex<Box<dyn crate::backend::GpuBackend>>>,
    ) -> std::sync::Arc<dyn crate::backend::ContextSubmitSession> {
        crate::backend::LockedSubmitSession::with_backend_type(backend, BackendType::Cpu)
    }
}

impl GpuBackend for CpuBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        self.adapters.clone()
    }

    fn adapter_capabilities(&self, _adapter_id: u32) -> crate::device::DeviceCapabilities {
        crate::device::DeviceCapabilities {
            host_sidecar_on_submit_worker: true,
            ..crate::device::DeviceCapabilities::default()
        }
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        if adapter_id as usize >= self.adapters.len() {
            anyhow::bail!("Invalid adapter id: {adapter_id}");
        }
        let handle = self.next_device_handle;
        self.next_device_handle += 1;
        self.devices.insert(
            handle,
            CpuDevice {
                timeline_next: Arc::new(std::sync::atomic::AtomicU64::new(1)),
                submission_worker: Arc::new(crate::backend::submission_worker::SubmissionWorker::new(
                    crate::backend::submission_worker::SUBMISSION_QUEUE_CAPACITY,
                )),
            },
        );
        self.device_retired_floor
            .insert(handle, Arc::new(std::sync::atomic::AtomicU64::new(0)));
        Ok(handle)
    }

    fn destroy_device(&mut self, device: DeviceHandle) {
        if let Some(dev) = self.devices.remove(&device) {
            let _ = dev.submission_worker.flush();
        }
        self.contexts.retain(|_, c| c.lock().unwrap().device != device);
        self.device_retired_floor.remove(&device);
        self.buffers.retain(|_, b| b.device_handle != device);
        self.shaders.retain(|_, s| s.device_handle != device);
        self.compute_pipelines.retain(|_, p| p.device_handle != device);
        self.bindless_to_buffer.retain(|_, h| self.buffers.contains_key(h));
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn device_wait_idle(&mut self, device: DeviceHandle) -> Result<()> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        let scheduled = self.scheduled_horizon(device);
        if scheduled > 0 {
            if let Some(dev) = self.devices.get(&device) {
                dev.submission_worker.flush()?;
                dev.submission_worker.wait_submitted(scheduled)?;
            }
            self.complete_device_seq_on_all_contexts(device, scheduled);
        }
        Ok(())
    }

    fn create_context(&mut self, device: DeviceHandle) -> Result<ContextHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        let id = self.next_context_id;
        self.next_context_id = self.next_context_id.saturating_add(1);
        self.contexts.insert(
            id,
            Arc::new(Mutex::new(CpuContextState {
                device,
                completed: 0,
                last_submitted_seq: 0,
                signal_queue: crate::signal::SignalQueue::new(),
            })),
        );
        Ok(id)
    }

    fn detach_context_for_destroy(&mut self, ctx: ContextHandle) -> Option<Box<dyn ContextDestroyHandle>> {
        if let Some(state) = self.contexts.remove(&ctx) {
            let state = state.lock().unwrap();
            let retired_horizon = state.completed.max(state.last_submitted_seq);
            if let Some(floor) = self.device_retired_floor.get(&state.device) {
                floor.fetch_max(retired_horizon, std::sync::atomic::Ordering::Relaxed);
            }
            Some(Box::new(CpuContextDestroyHandle) as Box<dyn ContextDestroyHandle>)
        } else {
            None
        }
    }

    fn clone_context_deletion_flush(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn crate::backend::ContextDeferredDeletionFlush>> {
        let _ = ctx;
        Some(std::sync::Arc::new(crate::backend::NoOpDeferredDeletionFlush))
    }

    fn clone_context_gpu_progress(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn crate::backend::ContextGpuProgress>> {
        Some(std::sync::Arc::new(CpuContextGpuProgress {
            state: std::sync::Arc::clone(self.contexts.get(&ctx)?),
        }))
    }

    fn context_device(&self, ctx: ContextHandle) -> DeviceHandle {
        self.context_state(ctx).device
    }

    fn create_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        _access: BufferKind,
        _element_stride: Option<u32>,
        _flags: BufferFlags,
    ) -> Result<BufferHandle> {
        self.alloc_owned_buffer(device, size, size, true, false)
    }

    fn create_buffer_with_capacity(
        &mut self,
        device: DeviceHandle,
        initial_size: u64,
        capacity: u64,
        _access: BufferKind,
        _element_stride: Option<u32>,
        _flags: BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        let cap = capacity.max(initial_size);
        let handle = self.alloc_owned_buffer(device, initial_size, cap, true, false)?;
        Ok((handle, cap))
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) {
        if let Some(buf) = self.buffers.remove(&buffer) {
            self.bindless_to_buffer.remove(&buf.bindless_index);
        }
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let bytes = self.bytes_mut(buffer)?;
        let start = offset as usize;
        let end = start + data.len();
        if end > bytes.len() {
            anyhow::bail!("Write exceeds buffer size");
        }
        bytes[start..end].copy_from_slice(data);
        Ok(())
    }

    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle> {
        self.alloc_owned_buffer(device, size, size, false, true)
    }

    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        let buf = self
            .buffers
            .get(&buffer)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;
        if !buf.is_withdraw_staging {
            anyhow::bail!("read_readback_buffer requires a withdraw staging buffer");
        }
        let src = self.bytes(buffer)?;
        let len = output.len().min(src.len());
        output[..len].copy_from_slice(&src[..len]);
        Ok(())
    }

    fn free_readback_buffer(&mut self, buffer: BufferHandle) {
        self.buffers.remove(&buffer);
    }

    fn query_texture_copy_footprint(
        &self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _format: TextureFormat,
    ) -> Result<crate::backend::TextureCopyFootprint> {
        cpu_graphics_unsupported()
    }

    fn alloc_texture_readback_staging(
        &mut self,
        _device: DeviceHandle,
        _layout: crate::backend::TextureCopyFootprint,
    ) -> Result<BufferHandle> {
        cpu_graphics_unsupported()
    }

    fn read_texture_readback_staging(
        &self,
        _buffer: BufferHandle,
        _layout: crate::backend::TextureCopyFootprint,
        _output: &mut [u8],
    ) -> Result<()> {
        cpu_graphics_unsupported()
    }

    fn texture_copy_retention_tag(&self, _texture: TextureHandle) -> u64 {
        0
    }

    fn clear_buffer(&mut self, _device: DeviceHandle, buffer: BufferHandle, offset: u64, size: u64) -> Result<()> {
        let bytes = self.bytes_mut(buffer)?;
        let clear_size = if size == 0 {
            bytes.len().saturating_sub(offset as usize)
        } else {
            size as usize
        };
        let start = offset as usize;
        let end = (start + clear_size).min(bytes.len());
        bytes[start..end].fill(0);
        Ok(())
    }

    fn buffer_size(&self, buffer: BufferHandle) -> u64 {
        self.buffers.get(&buffer).map(|b| b.size).unwrap_or(0)
    }

    fn buffer_capacity(&self, buffer: BufferHandle) -> u64 {
        self.buffers.get(&buffer).map(|b| b.alloc_size).unwrap_or(0)
    }

    fn set_buffer_logical_size(
        &mut self,
        _device: DeviceHandle,
        buffer: BufferHandle,
        new_logical_size: u64,
    ) -> Result<()> {
        let buf = self
            .buffers
            .get_mut(&buffer)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;
        if new_logical_size > buf.alloc_size {
            anyhow::bail!("logical size exceeds allocation");
        }
        if new_logical_size == 0 {
            anyhow::bail!("buffer size must be non-zero");
        }
        buf.size = new_logical_size;
        Ok(())
    }

    fn buffer_bindless_index(&self, buffer: BufferHandle) -> Option<u32> {
        self.buffers.get(&buffer).map(|b| b.bindless_index)
    }

    fn buffer_bindless_srv_index(&self, buffer: BufferHandle) -> Option<u32> {
        self.buffer_bindless_index(buffer)
    }

    fn create_buffer_view(
        &mut self,
        parent: BufferHandle,
        offset: u64,
        size: u64,
        _element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        let parent_buf = self.buffers.get(&parent).context("Invalid parent buffer handle")?;
        if offset + size > parent_buf.size {
            anyhow::bail!("View exceeds parent buffer size");
        }
        let device_handle = parent_buf.device_handle;
        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;
        let index = self.next_bindless_index;
        self.next_bindless_index += 1;
        self.bindless_to_buffer.insert(index, handle);
        self.buffers.insert(
            handle,
            CpuBuffer {
                device_handle,
                size,
                alloc_size: size,
                data: None,
                parent: Some((parent, offset)),
                bindless_index: index,
                is_withdraw_staging: false,
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
        let buf = self
            .buffers
            .get_mut(&buffer)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;
        if buf.device_handle != device {
            anyhow::bail!("Buffer belongs to a different device");
        }
        if buf.parent.is_some() {
            anyhow::bail!("CPU: cannot resize a buffer view");
        }
        let new_len = new_size as usize;
        let data = buf.data.as_mut().context("CPU: missing owned storage")?;
        if preserve_contents {
            data.resize(new_len, 0);
        } else {
            *data = vec![0u8; new_len];
        }
        buf.size = new_size;
        buf.alloc_size = new_size;
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
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        if slang_source.contains("[goldy_vertex]") || slang_source.contains("[goldy_fragment]") {
            anyhow::bail!("CPU backend is compute-only: vertex/fragment shaders are not supported");
        }
        let handle = self.next_shader_handle;
        self.next_shader_handle += 1;
        self.shaders.insert(
            handle,
            CpuShader {
                device_handle: device,
                source: slang_source.to_string(),
                search_paths: search_paths.iter().map(|s| (*s).to_string()).collect(),
                defines: defines
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                    .collect(),
                optimization_level,
            },
        );
        Ok(handle)
    }

    fn destroy_shader(&mut self, shader: ShaderHandle) {
        self.shaders.remove(&shader);
    }

    #[cfg(feature = "graphics")]
    fn create_pipeline(
        &mut self,
        _device: DeviceHandle,
        _vertex_shader: ShaderHandle,
        _fragment_shader: ShaderHandle,
        _vertex_layout: &crate::types::VertexBufferLayout,
        _topology: crate::types::PrimitiveTopology,
        _target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        cpu_graphics_unsupported()
    }

    #[cfg(feature = "graphics")]
    fn destroy_pipeline(&mut self, _pipeline: PipelineHandle) {}

    #[cfg(feature = "graphics")]
    fn create_pipeline_with_depth(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &crate::types::VertexBufferLayout,
        topology: crate::types::PrimitiveTopology,
        target_format: TextureFormat,
        _depth_stencil: Option<&crate::types::DepthStencilState>,
    ) -> Result<PipelineHandle> {
        self.create_pipeline(
            device,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
        )
    }

    #[cfg(feature = "graphics")]
    fn create_render_target_with_depth(
        &mut self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _color_format: TextureFormat,
        _depth_format: Option<crate::types::DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        cpu_graphics_unsupported()
    }

    #[cfg(feature = "graphics")]
    fn destroy_render_target(&mut self, _target: RenderTargetHandle) {}

    #[cfg(feature = "graphics")]
    fn render_to_target(
        &mut self,
        _device: DeviceHandle,
        _target: RenderTargetHandle,
        _color_load: crate::types::TargetLoad,
        _commands: &[RenderCommand],
    ) -> Result<()> {
        cpu_graphics_unsupported()
    }

    fn create_texture(
        &mut self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _format: TextureFormat,
        _access: crate::types::TextureKind,
        _flags: crate::types::TextureFlags,
    ) -> Result<TextureHandle> {
        cpu_graphics_unsupported()
    }

    fn write_texture(&mut self, _texture: TextureHandle, _data: &[u8], _width: u32, _height: u32) -> Result<()> {
        cpu_graphics_unsupported()
    }

    fn write_texture_region(
        &mut self,
        _texture: TextureHandle,
        _x: u32,
        _y: u32,
        _width: u32,
        _height: u32,
        _data: &[u8],
    ) -> Result<()> {
        cpu_graphics_unsupported()
    }

    fn destroy_texture(&mut self, _texture: TextureHandle) {}

    fn texture_bindless_index(&self, _texture: TextureHandle) -> Option<u32> {
        None
    }

    fn texture_bindless_sampled_index(&self, _texture: TextureHandle) -> Option<u32> {
        None
    }

    fn create_sampler(&mut self, _device: DeviceHandle, _desc: &crate::types::SamplerDesc) -> Result<SamplerHandle> {
        cpu_graphics_unsupported()
    }

    fn destroy_sampler(&mut self, _sampler: SamplerHandle) {}

    fn sampler_bindless_index(&self, _sampler: SamplerHandle) -> Option<u32> {
        None
    }

    #[cfg(feature = "graphics")]
    fn create_surface(
        &mut self,
        _device: DeviceHandle,
        _window: &dyn raw_window_handle::HasWindowHandle,
        _display: &dyn raw_window_handle::HasDisplayHandle,
        _depth_format: Option<crate::types::DepthFormat>,
    ) -> Result<crate::handles::SurfaceHandle> {
        cpu_graphics_unsupported()
    }

    #[cfg(feature = "graphics")]
    fn destroy_surface(&mut self, _surface: crate::handles::SurfaceHandle) {}

    #[cfg(feature = "graphics")]
    fn surface_resize(&mut self, _surface: crate::handles::SurfaceHandle, _width: u32, _height: u32) -> Result<()> {
        cpu_graphics_unsupported()
    }

    #[cfg(feature = "graphics")]
    fn surface_size(&self, _surface: crate::handles::SurfaceHandle) -> (u32, u32) {
        (0, 0)
    }

    #[cfg(feature = "graphics")]
    fn surface_format(&self, _surface: crate::handles::SurfaceHandle) -> TextureFormat {
        TextureFormat::Rgba8Unorm
    }

    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue {
        self.context_state(ctx).completed
    }

    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue {
        self.device_retired(device)
    }

    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        if let Some(dev) = self.devices.get(&device) {
            dev.submission_worker.flush()?;
            let horizon = self.scheduled_horizon(device);
            dev.submission_worker.wait_submitted_if_scheduled(value, horizon)?;
        }
        let ctx_ids: Vec<_> = self
            .contexts
            .iter()
            .filter(|(_, c)| c.lock().unwrap().device == device)
            .map(|(id, _)| *id)
            .collect();
        for id in ctx_ids {
            let ctx = self.contexts.get(&id).unwrap();
            let mut state = ctx.lock().unwrap();
            if state.completed < value {
                state.completed = value;
            }
        }
        Ok(())
    }

    fn poll_signals(
        &mut self,
        ctx: ContextHandle,
        _progress: crate::timeline::TimelineValue,
    ) -> Vec<crate::signal::QueuedSignal> {
        crate::signal::drain_all_queued_signals(&self.context_state(ctx).signal_queue)
    }

    fn submit_standalone(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        let device = self.context_device(ctx);
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        self.apply_submit_sync(sync)?;
        let effective = super::commands_with_sync_prologue(commands, sync);
        self.execute_commands(&effective)?;

        let dev = self
            .devices
            .get(&device)
            .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?;
        let tv = crate::backend::submission_worker::allocate_timeline_value(&dev.timeline_next);
        self.context_state(ctx).last_submitted_seq = tv;
        self.execute_submit_immediately(ctx, tv)?;
        self.await_submit(ctx, tv)?;
        Ok(tv)
    }

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
                GraphCommand::Render { .. } => {
                    anyhow::bail!("CPU backend is compute-only: render graph commands are not supported");
                }
            }
        }
        if !batch.is_empty() {
            last_tv = self.submit_standalone(ctx, &batch, sync)?;
        }
        Ok(last_tv)
    }

    fn submit_graph_and_retain(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        self.retained_graphs.insert((ctx, key), commands.to_vec());
        self.submit_graph(ctx, commands, sync)
    }

    fn try_resubmit_retained(
        &mut self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<crate::timeline::TimelineValue>> {
        let Some(commands) = self.retained_graphs.get(&(ctx, key)).cloned() else {
            return Ok(None);
        };
        self.submit_graph(ctx, &commands, sync).map(Some)
    }

    fn evict_retained(&mut self, ctx: ContextHandle, key: u64) {
        self.retained_graphs.remove(&(ctx, key));
    }

    #[cfg(feature = "graphics")]
    fn begin_frame(
        &mut self,
        _surface: crate::handles::SurfaceHandle,
        _ctx: ContextHandle,
    ) -> Result<(FrameToken, TextureHandle)> {
        cpu_graphics_unsupported()
    }

    #[cfg(feature = "graphics")]
    fn submit_frame(&mut self, _frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        cpu_graphics_unsupported()
    }

    fn create_compute_pipeline(
        &mut self,
        device: DeviceHandle,
        compute_shader: ShaderHandle,
        _debug_name: Option<&str>,
    ) -> Result<ComputePipelineHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        let shader = self
            .shaders
            .get(&compute_shader)
            .context("CPU: invalid shader handle")?;
        if shader.device_handle != device {
            anyhow::bail!("Shader belongs to a different device");
        }
        let paths: Vec<&str> = shader.search_paths.iter().map(String::as_str).collect();
        let defines: Vec<(&str, &str)> = shader.defines.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let kernel = compile_shader(&self.slang, &shader.source, &paths, &defines, shader.optimization_level)?;
        let slot_access = crate::slang::virtual_main::extract_push_constant_categories(&shader.source)
            .iter()
            .map(|category| {
                category.map(|category| match category {
                    ResourceCategory::Broadcast | ResourceCategory::Texture | ResourceCategory::Sampler => {
                        ResourceAccess::Read
                    }
                    ResourceCategory::Scattered | ResourceCategory::StorageImage => ResourceAccess::ReadWrite,
                })
            })
            .collect();
        let handle = self.next_compute_pipeline_handle;
        self.next_compute_pipeline_handle += 1;
        self.compute_pipelines.insert(
            handle,
            CpuComputePipeline {
                device_handle: device,
                kernel: Arc::new(kernel),
                slot_access,
            },
        );
        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        self.compute_pipelines.remove(&pipeline);
    }

    fn compute_pipeline_slot_access(&self, pipeline: ComputePipelineHandle) -> Vec<Option<ResourceAccess>> {
        self.compute_pipelines
            .get(&pipeline)
            .map(|p| p.slot_access.clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::{BufferKind, MemoryExchange, NodeAccess, RetainedPool, Scheme, ShaderModule};
    use std::sync::Arc;

    #[test]
    fn cpu_backend_scheme_double_u32() {
        let device = Device::from_backend(Box::new(CpuBackend::new().expect("cpu backend"))).expect("device");
        assert_eq!(device.backend_type(), BackendType::Cpu);
        let ctx = device.create_context().expect("ctx");
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let n = 64usize;
        let input: Vec<u32> = (0..n as u32).collect();
        let data = pool
            .acquire_buffer_with_data(&input, BufferKind::Scattered)
            .expect("buffer");

        let src = r#"
            import goldy_exp;
            [goldy_compute]
            [numthreads(64, 1, 1)]
            void cs_main(Scattered<uint> data, ThreadId id) {
                if (id.x < goldy_buf_len(data)) {
                    data[id.x] = data[id.x] * 2u;
                }
            }
        "#;
        let shader = ShaderModule::from_slang(&device, src).expect("compile");
        let pipeline = crate::ComputePipeline::new(&device, &shader).expect("pipeline");
        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("double", &pipeline)
            .with_parcel(&data, NodeAccess::ReadWrite)
            .dispatch((n as u32).div_ceil(64), 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &data)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let bytes = grant.claim(&mut frame).expect("claim").consume().expect("consume");
        let out: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();
        assert_eq!(out.len(), n);
        for i in 0..n {
            assert_eq!(out[i], (i as u32) * 2, "index {i}");
        }
    }
}
