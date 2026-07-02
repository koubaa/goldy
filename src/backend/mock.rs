//! Mock backend for testing.
//!
//! This backend validates command sequences without requiring GPU hardware,
//! enabling unit tests to run in CI environments.

use super::*;
use crate::types::*;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock GPU backend for testing.
///
/// Tracks resource creation/destruction and command recording
/// without actually performing GPU operations.
pub struct MockBackend {
    adapters: Vec<AdapterInfo>,
    devices: HashMap<DeviceHandle, MockDevice>,
    next_device_handle: DeviceHandle,
    buffers: HashMap<BufferHandle, MockBuffer>,
    next_buffer_handle: BufferHandle,
    shaders: HashMap<ShaderHandle, MockShader>,
    next_shader_handle: ShaderHandle,
    pipelines: HashMap<PipelineHandle, MockPipeline>,
    next_pipeline_handle: PipelineHandle,
    compute_pipelines: HashMap<ComputePipelineHandle, MockComputePipeline>,
    next_compute_pipeline_handle: ComputePipelineHandle,
    render_targets: HashMap<RenderTargetHandle, MockRenderTarget>,
    next_render_target_handle: RenderTargetHandle,
    surfaces: HashMap<SurfaceHandle, MockSurface>,
    next_surface_handle: SurfaceHandle,
    textures: HashMap<TextureHandle, MockTexture>,
    next_texture_handle: TextureHandle,
    samplers: HashMap<SamplerHandle, MockSampler>,
    next_sampler_handle: SamplerHandle,
    /// Next bindless index to assign (shared across buffers, textures, samplers)
    next_bindless_index: u32,
    /// Render commands recorded during render operations
    pub recorded_commands: Vec<Vec<RenderCommand>>,
    /// Compute commands recorded during dispatch operations
    pub recorded_compute_commands: Vec<Vec<GpuCommand>>,
    /// Targets that were created (for verification)
    pub targets_created: Vec<(u32, u32, TextureFormat)>,
    /// Targets with depth buffer that were created (for verification)
    pub targets_with_depth_created: Vec<(u32, u32, TextureFormat, Option<DepthFormat>)>,
    /// Count of CPU readbacks performed
    pub cpu_readback_count: usize,
    /// Count of surface presents performed
    pub surface_present_count: usize,
    /// Count of textures created
    pub textures_created: usize,
    /// Count of samplers created
    pub samplers_created: usize,
    /// Count of compute/graph submits performed
    pub compute_dispatch_count: usize,
    /// Cross-context epoch waits recorded per standalone/graph submit (mock tests).
    pub recorded_waits: Vec<Vec<crate::timeline::Epoch>>,
    /// Whether each `submit_graph` call received `sync = Some(...)`.
    ///
    /// When `Some`, the backend knows to suppress its legacy blanket-acquire barrier and
    /// rely on the scoped prologue that was folded into the command list.
    pub recorded_graph_syncs: Vec<bool>,
    /// Retained command lists keyed by `(ctx, retention_key)`.
    retained_graphs: HashMap<(ContextHandle, u64), Vec<GraphCommand>>,
    /// Count of zero-record resubmits served from `retained_graphs`.
    pub retained_resubmit_count: usize,
    /// Count of `wait_until` calls (for verifying no CPU waits in unified paths)
    pub wait_until_count: usize,
    /// Count of grant readback staging buffer allocations
    pub readback_alloc_count: usize,
    /// Grant readback buffers freed via [`GpuBackend::free_readback_buffer`].
    pub readback_free_count: usize,
    /// Count of `create_buffer_view` calls (for verifying transient view cache hit rate)
    pub buffer_view_create_count: usize,
    /// Default format for new surfaces (simulates GPU/display preference)
    pub default_surface_format: TextureFormat,
    /// When true, [`Scheme::submit`] enqueues present on the per-device submission worker.
    schedules_present_on_submit_worker: bool,
    /// When true, [`GpuBackend::supports_lazy_present_finish`] returns true (DX12-style).
    lazy_present_finish: bool,
    /// Device-global submission sequence (shared value space across contexts on one queue).
    device_retired_floor: HashMap<DeviceHandle, Arc<std::sync::atomic::AtomicU64>>,
    surface_pending_acquire: HashMap<SurfaceHandle, u32>,
    contexts: HashMap<ContextHandle, Arc<Mutex<MockContextState>>>,
    next_context_id: ContextHandle,
}

#[allow(dead_code)]
struct MockDevice {
    adapter_id: u32,
    timeline_next: std::sync::Arc<std::sync::atomic::AtomicU64>,
    submission_worker: std::sync::Arc<crate::backend::submission_worker::SubmissionWorker>,
}

struct MockPendingSubmit {
    tv: u64,
    context_state: std::sync::Arc<std::sync::Mutex<MockContextState>>,
}

impl crate::backend::submission_worker::PendingSubmit for MockPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.mock");
        let mut state = self.context_state.lock().unwrap();
        state.completed = self.tv;
        state
            .signal_queue
            .push(crate::signal::Signal::BoundaryCrossed { epoch: self.tv });
        Ok(())
    }
}

/// Per-context submission stream state (mock timeline + signals).
struct MockContextState {
    device: DeviceHandle,
    completed: u64,
    last_submitted_seq: u64,
    signal_queue: crate::signal::SignalQueue,
}

struct MockContextTimelineReader {
    state: Arc<Mutex<MockContextState>>,
}

impl crate::backend::ContextTimelineReader for MockContextTimelineReader {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue {
        self.state.lock().unwrap().completed
    }

    fn peek_oldest_in_flight(&self) -> Option<crate::timeline::TimelineValue> {
        let state = self.state.lock().unwrap();
        if state.completed < state.last_submitted_seq {
            Some(state.completed.saturating_add(1))
        } else {
            None
        }
    }
}

struct MockDeviceTimelineReader {
    floor: Arc<std::sync::atomic::AtomicU64>,
}

impl crate::backend::DeviceTimelineReader for MockDeviceTimelineReader {
    fn device_horizon(&self) -> crate::timeline::TimelineValue {
        self.floor.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[allow(dead_code)]
struct MockBuffer {
    device_handle: DeviceHandle,
    /// Logical byte size.
    size: u64,
    /// Backing storage (`data.len()`).
    alloc_size: u64,
    data: Vec<u8>,
    bindless_index: u32,
    flags: BufferFlags,
    /// Grant-read staging buffer (persistently CPU-readable; no bindless slot).
    is_grant_readback: bool,
    texture_copy_footprint: Option<crate::backend::TextureCopyFootprint>,
}

#[allow(dead_code)]
struct MockShader {
    device_handle: DeviceHandle,
    source: String,
}

#[allow(dead_code)]
struct MockPipeline {
    device_handle: DeviceHandle,
}

#[allow(dead_code)]
struct MockComputePipeline {
    device_handle: DeviceHandle,
}

#[allow(dead_code)]
struct MockRenderTarget {
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    depth_format: Option<DepthFormat>,
    has_rendered: bool,
    /// Simulated pixel data (all zeros by default)
    data: Vec<u8>,
}

#[allow(dead_code)]
struct MockTexture {
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    data: Vec<u8>,
    bindless_index: u32,
    /// For `TextureKind::DirectInterpolated`, a second bindless index in the
    /// sampled-texture pool. `None` for all other access modes.
    sampled_bindless_index: Option<u32>,
}

#[allow(dead_code)]
struct MockSampler {
    device_handle: DeviceHandle,
    #[allow(dead_code)]
    desc: SamplerDesc,
    bindless_index: u32,
}

#[allow(dead_code)]
struct MockSurface {
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    next_image: SwapchainImageHandle,
    current_texture_handle: Option<TextureHandle>,
    pending_frame_compute: Vec<GpuCommand>,
}

impl MockBackend {
    /// Create a new mock backend with one simulated adapter.
    pub fn new() -> Self {
        Self {
            adapters: vec![AdapterInfo {
                id: 0,
                name: "Mock GPU".to_string(),
                vendor: "Goldy Test".to_string(),
                backend: BackendType::Vulkan, // Pretend to be Vulkan
                device_type: DeviceType::DiscreteGpu,
            }],
            devices: HashMap::new(),
            next_device_handle: 1,
            buffers: HashMap::new(),
            next_buffer_handle: 1,
            shaders: HashMap::new(),
            next_shader_handle: 1,
            pipelines: HashMap::new(),
            next_pipeline_handle: 1,
            compute_pipelines: HashMap::new(),
            next_compute_pipeline_handle: 1,
            render_targets: HashMap::new(),
            next_render_target_handle: 1,
            surfaces: HashMap::new(),
            next_surface_handle: 1,
            textures: HashMap::new(),
            next_texture_handle: 1,
            samplers: HashMap::new(),
            next_sampler_handle: 1,
            next_bindless_index: 0,
            recorded_commands: Vec::new(),
            recorded_compute_commands: Vec::new(),
            targets_created: Vec::new(),
            targets_with_depth_created: Vec::new(),
            cpu_readback_count: 0,
            surface_present_count: 0,
            textures_created: 0,
            samplers_created: 0,
            compute_dispatch_count: 0,
            recorded_waits: Vec::new(),
            recorded_graph_syncs: Vec::new(),
            retained_graphs: HashMap::new(),
            retained_resubmit_count: 0,
            wait_until_count: 0,
            readback_alloc_count: 0,
            readback_free_count: 0,
            buffer_view_create_count: 0,
            default_surface_format: TextureFormat::Bgra8UnormSrgb,
            schedules_present_on_submit_worker: true,
            lazy_present_finish: false,
            device_retired_floor: HashMap::new(),
            surface_pending_acquire: HashMap::new(),
            contexts: HashMap::new(),
            next_context_id: 1,
        }
    }

    fn context_state(&self, ctx: ContextHandle) -> std::sync::MutexGuard<'_, MockContextState> {
        self.contexts.get(&ctx).expect("invalid context handle").lock().unwrap()
    }

    fn context_state_mut(&mut self, ctx: ContextHandle) -> std::sync::MutexGuard<'_, MockContextState> {
        self.context_state(ctx)
    }

    /// Max completed over live contexts on `device`, floored by `retired_floor`.
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

    /// Device-scoped idle: advance every context on `device` to the scheduled horizon.
    fn complete_device_seq_on_all_contexts(&mut self, device: DeviceHandle, seq: u64) {
        for ctx in self.contexts.values() {
            let mut state = ctx.lock().unwrap();
            if state.device == device {
                state.completed = seq;
                state.last_submitted_seq = seq;
            }
        }
    }

    fn push_context_signal(&self, ctx: ContextHandle, signal: crate::signal::Signal) {
        self.context_state(ctx).signal_queue.push(signal);
    }

    fn mock_pending_submit(
        _ctx: ContextHandle,
        tv: u64,
        context_state: Arc<Mutex<MockContextState>>,
    ) -> Box<dyn crate::backend::submission_worker::PendingSubmit> {
        Box::new(MockPendingSubmit { tv, context_state })
    }

    fn mock_submit_worker(
        &self,
        ctx: ContextHandle,
    ) -> Result<Arc<crate::backend::submission_worker::SubmissionWorker>> {
        let device = self.context_device(ctx);
        Ok(Arc::clone(
            &self
                .devices
                .get(&device)
                .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?
                .submission_worker,
        ))
    }

    fn mock_context_state(&self, ctx: ContextHandle) -> Result<Arc<Mutex<MockContextState>>> {
        Ok(Arc::clone(
            self.contexts
                .get(&ctx)
                .ok_or_else(|| anyhow::anyhow!("Invalid context handle"))?,
        ))
    }

    /// Enqueue present work on the worker without completing it (FIFO-at-submit path).
    fn enqueue_mock_submit_deferred(&self, ctx: ContextHandle, tv: u64) -> Result<()> {
        let worker = self.mock_submit_worker(ctx)?;
        worker.check_error()?;
        let context_state = self.mock_context_state(ctx)?;
        worker.enqueue(tv, Self::mock_pending_submit(ctx, tv, context_state))
    }

    /// Complete a mock submit synchronously on the calling thread.
    fn run_mock_submit_now(&self, ctx: ContextHandle, tv: u64) -> Result<()> {
        let worker = self.mock_submit_worker(ctx)?;
        worker.check_error()?;
        let context_state = self.mock_context_state(ctx)?;
        worker.execute_immediately(tv, Self::mock_pending_submit(ctx, tv, context_state))
    }

    fn mock_scheduled_horizon(&self, device: DeviceHandle) -> u64 {
        self.devices
            .get(&device)
            .map(|d| {
                d.timeline_next
                    .load(std::sync::atomic::Ordering::Acquire)
                    .saturating_sub(1)
            })
            .unwrap_or(0)
    }

    fn record_submit_sync(&mut self, sync: Option<&SubmitSync>) -> Result<()> {
        if let Some(s) = sync {
            self.recorded_waits.push(s.waits.clone());
            for epoch in &s.waits {
                self.wait_until(epoch.context, epoch.value)?;
            }
        } else {
            self.recorded_waits.push(Vec::new());
        }
        Ok(())
    }

    /// Set the default surface format for testing different GPU preferences.
    ///
    /// Use this to verify that your code correctly uses `Surface::format()`
    /// rather than assuming a hardcoded format.
    pub fn set_default_surface_format(&mut self, format: TextureFormat) {
        self.default_surface_format = format;
    }

    /// Control whether [`crate::Scheme::submit`] schedules present on the submission worker.
    pub fn set_schedules_present_on_submit_worker(&mut self, enabled: bool) {
        self.schedules_present_on_submit_worker = enabled;
    }

    /// Control whether the mock reports lazy present finish (DX12-style speculative acquire path).
    pub fn set_lazy_present_finish(&mut self, enabled: bool) {
        self.lazy_present_finish = enabled;
    }

    /// Number of live retained command lists (test introspection).
    pub fn retained_graph_count(&self) -> usize {
        self.retained_graphs.len()
    }

    /// Reset recorded state for a new test.
    pub fn reset_tracking(&mut self) {
        self.recorded_commands.clear();
        self.recorded_compute_commands.clear();
        self.targets_created.clear();
        self.targets_with_depth_created.clear();
        self.cpu_readback_count = 0;
        self.surface_present_count = 0;
        self.textures_created = 0;
        self.samplers_created = 0;
        self.compute_dispatch_count = 0;
        self.recorded_waits.clear();
        self.recorded_graph_syncs.clear();
        self.wait_until_count = 0;
        self.buffer_view_create_count = 0;
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
            let src_buf = self
                .buffers
                .get(&src)
                .ok_or_else(|| anyhow::anyhow!("CopyBuffer: invalid src"))?;
            if src_offset.saturating_add(copy_len as u64) > src_buf.size {
                anyhow::bail!("CopyBuffer: size exceeds src bounds");
            }
            let start = src_offset as usize;
            src_buf.data[start..start + copy_len].to_vec()
        };
        let dst_buf = self
            .buffers
            .get_mut(&dst)
            .ok_or_else(|| anyhow::anyhow!("CopyBuffer: invalid dst"))?;
        if dst_offset.saturating_add(copy_len as u64) > dst_buf.size {
            anyhow::bail!("CopyBuffer: size exceeds dst bounds");
        }
        let dst_start = dst_offset as usize;
        dst_buf.data[dst_start..dst_start + copy_len].copy_from_slice(&src_data);
        Ok(())
    }

    fn execute_copy_texture_to_readback(
        &mut self,
        src: TextureHandle,
        dst: BufferHandle,
        layout: crate::backend::TextureCopyFootprint,
    ) -> Result<()> {
        let tex = self
            .textures
            .get(&src)
            .ok_or_else(|| anyhow::anyhow!("CopyTextureToReadback: invalid src"))?;
        let row_bytes = layout.tight_row_bytes() as usize;
        let pitch = layout.row_pitch as usize;
        let dst_buf = self
            .buffers
            .get_mut(&dst)
            .ok_or_else(|| anyhow::anyhow!("CopyTextureToReadback: invalid dst"))?;
        if dst_buf.size < layout.staging_bytes {
            anyhow::bail!("CopyTextureToReadback: dst too small");
        }
        for row in 0..layout.height as usize {
            let src_off = row * row_bytes;
            let dst_off = layout.footprint_offset as usize + row * pitch;
            dst_buf.data[dst_off..dst_off + row_bytes].copy_from_slice(&tex.data[src_off..src_off + row_bytes]);
        }
        Ok(())
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::backend::GpuBackendTimelineWait for MockBackend {
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
        let horizon = self.mock_scheduled_horizon(device);
        if value == 0 || value > horizon {
            return Ok(None);
        }
        Ok(Some(crate::backend::submission_worker::SubmissionEpochWait::new(
            std::sync::Arc::clone(&dev.submission_worker),
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

    fn check_submission_worker_for_context(&self, ctx: ContextHandle) -> Result<()> {
        let device = self.context_device(ctx);
        if let Some(dev) = self.devices.get(&device) {
            dev.submission_worker.check_error()
        } else {
            Ok(())
        }
    }

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        let device = self.context_device(ctx);
        if let Some(dev) = self.devices.get(&device) {
            dev.submission_worker.flush()?;
            let horizon = self.mock_scheduled_horizon(device);
            dev.submission_worker.wait_submitted_if_scheduled(value, horizon)?;
        }
        self.wait_until_count += 1;
        let cur = self.gpu_progress(ctx);
        if value > cur {
            self.context_state_mut(ctx).completed = value;
        }
        Ok(())
    }
}

impl crate::backend::GpuBackendPresentSplit for MockBackend {
    fn take_present_gpu_work(
        &mut self,
        frame: FrameToken,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn crate::backend::PresentGpuWork>> {
        if let Some(tex_handle) = self
            .surfaces
            .get_mut(&frame.surface)
            .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?
            .current_texture_handle
            .take()
        {
            self.textures.remove(&tex_handle);
        }
        Ok(Box::new(MockPresentGpuWork { frame }))
    }

    fn finish_present(
        &mut self,
        finish: crate::backend::PresentFinishState,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        let device = self
            .surfaces
            .get(&finish.frame.surface)
            .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?
            .device_handle;
        self.surface_present_count += 1;
        let image_index = finish.frame.image as u32;
        self.push_context_signal(
            finish.frame.context,
            crate::signal::Signal::SwapchainReturned { image_index },
        );
        if let Some(count) = self.surface_pending_acquire.get_mut(&finish.frame.surface) {
            *count = count.saturating_sub(1);
        }
        let dev = self
            .devices
            .get(&device)
            .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?;
        let tv = crate::backend::submission_worker::allocate_timeline_value(&dev.timeline_next);
        self.context_state_mut(finish.frame.context).last_submitted_seq = tv;
        self.run_mock_submit_now(finish.frame.context, tv)?;
        Ok(tv)
    }

    fn schedules_present_on_submit_worker(&self) -> bool {
        self.schedules_present_on_submit_worker
    }

    fn supports_lazy_present_finish(&self) -> bool {
        self.lazy_present_finish
    }

    fn schedule_present_on_submission_worker(
        &mut self,
        frame: FrameToken,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::backend::SchedulePresentOnWorkerResult> {
        if let Some(tex_handle) = self
            .surfaces
            .get_mut(&frame.surface)
            .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?
            .current_texture_handle
            .take()
        {
            self.textures.remove(&tex_handle);
        }
        let device = self
            .surfaces
            .get(&frame.surface)
            .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?
            .device_handle;
        let dev = self
            .devices
            .get(&device)
            .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?;
        let tv = crate::backend::submission_worker::allocate_timeline_value(&dev.timeline_next);
        self.enqueue_mock_submit_deferred(frame.context, tv)?;
        Ok(crate::backend::SchedulePresentOnWorkerResult {
            present_tv: tv,
            consume_job: None,
        })
    }

    fn take_scheduled_present_blocking_wait(
        &self,
        frame: FrameToken,
        present_tv: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn crate::backend::ScheduledPresentBlockingWait>>> {
        if !self.schedules_present_on_submit_worker {
            return Ok(None);
        }
        let device = self.context_device(frame.context);
        let dev = self
            .devices
            .get(&device)
            .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?;
        Ok(Some(Box::new(MockScheduledPresentWait {
            frame,
            present_tv,
            worker: std::sync::Arc::clone(&dev.submission_worker),
        })))
    }

    fn apply_scheduled_present_bookkeeping(
        &mut self,
        outcome: crate::backend::ScheduledPresentWaitOutcome,
    ) -> Result<()> {
        self.surface_present_count += 1;
        let image_index = outcome.frame.image as u32;
        self.push_context_signal(
            outcome.frame.context,
            crate::signal::Signal::SwapchainReturned { image_index },
        );
        if let Some(count) = self.surface_pending_acquire.get_mut(&outcome.frame.surface) {
            *count = count.saturating_sub(1);
        }
        self.context_state_mut(outcome.frame.context).last_submitted_seq = outcome.present_tv;
        Ok(())
    }
}

struct MockPresentGpuWork {
    frame: FrameToken,
}

struct MockScheduledPresentWait {
    frame: FrameToken,
    present_tv: crate::timeline::TimelineValue,
    worker: std::sync::Arc<crate::backend::submission_worker::SubmissionWorker>,
}

impl crate::backend::ScheduledPresentBlockingWait for MockScheduledPresentWait {
    fn run(self: Box<Self>) -> Result<crate::backend::ScheduledPresentWaitOutcome> {
        self.worker.wait_submitted(self.present_tv)?;
        self.worker.check_error()?;
        Ok(crate::backend::ScheduledPresentWaitOutcome {
            frame: self.frame,
            present_tv: self.present_tv,
            finish: None,
            return_fence: self.present_tv,
        })
    }
}

impl crate::backend::PresentGpuWork for MockPresentGpuWork {
    fn run(self: Box<Self>) -> Result<crate::backend::PresentFinishState> {
        Ok(crate::backend::PresentFinishState {
            frame: self.frame,
            return_fence: 0,
            scratch_texture: None,
            scratch_layout_updated: false,
            present_timeline: 0,
            copy_timeline: None,
            frame_compute_timeline: None,
            signal_timeline: None,
            render_pass_submitted: false,
            present_ok: true,
        })
    }
}

impl crate::backend::GpuBackendSubmitSession for MockBackend {
    fn clone_context_submit_session(
        &self,
        _ctx: ContextHandle,
        backend: std::sync::Arc<std::sync::Mutex<Box<dyn crate::backend::GpuBackend>>>,
    ) -> std::sync::Arc<dyn crate::backend::ContextSubmitSession> {
        crate::backend::LockedSubmitSession::with_backend_type(backend, self.backend_type())
    }
}

impl GpuBackend for MockBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Vulkan
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        self.adapters.clone()
    }

    fn adapter_capabilities(&self, _adapter_id: u32) -> crate::device::DeviceCapabilities {
        crate::device::DeviceCapabilities::default()
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        if adapter_id as usize >= self.adapters.len() {
            anyhow::bail!("Invalid adapter id: {}", adapter_id);
        }

        let handle = self.next_device_handle;
        self.next_device_handle += 1;

        self.devices.insert(
            handle,
            MockDevice {
                adapter_id,
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

        // Clean up resources owned by this device
        self.buffers.retain(|_, b| b.device_handle != device);
        self.shaders.retain(|_, s| s.device_handle != device);
        self.pipelines.retain(|_, p| p.device_handle != device);
        self.compute_pipelines.retain(|_, p| p.device_handle != device);
        self.render_targets.retain(|_, t| t.device_handle != device);
        self.textures.retain(|_, t| t.device_handle != device);
        self.samplers.retain(|_, s| s.device_handle != device);
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn device_wait_idle(&mut self, device: DeviceHandle) -> Result<()> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        let scheduled = self.mock_scheduled_horizon(device);
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
            Arc::new(Mutex::new(MockContextState {
                device,
                completed: 0,
                last_submitted_seq: 0,
                signal_queue: crate::signal::SignalQueue::new(),
            })),
        );
        Ok(id)
    }

    fn destroy_context(&mut self, ctx: ContextHandle) {
        if let Some(state) = self.contexts.remove(&ctx) {
            let state = state.lock().unwrap();
            let retired_horizon = state.completed.max(state.last_submitted_seq);
            if let Some(floor) = self.device_retired_floor.get(&state.device) {
                floor.fetch_max(retired_horizon, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    fn clone_context_timeline_reader(
        &self,
        ctx: ContextHandle,
    ) -> Option<Arc<dyn crate::backend::ContextTimelineReader>> {
        Some(Arc::new(MockContextTimelineReader {
            state: Arc::clone(self.contexts.get(&ctx)?),
        }))
    }

    fn clone_device_timeline_reader(
        &self,
        device: DeviceHandle,
    ) -> Option<Arc<dyn crate::backend::DeviceTimelineReader>> {
        Some(Arc::new(MockDeviceTimelineReader {
            floor: Arc::clone(self.device_retired_floor.get(&device)?),
        }))
    }

    fn clone_context_deletion_flush(
        &self,
        ctx: ContextHandle,
        _context_readers: std::sync::Arc<
            std::sync::Mutex<
                std::collections::HashMap<ContextHandle, std::sync::Arc<dyn crate::backend::ContextTimelineReader>>,
            >,
        >,
    ) -> Option<std::sync::Arc<dyn crate::backend::ContextDeferredDeletionFlush>> {
        let _ = ctx;
        Some(std::sync::Arc::new(crate::backend::NoOpDeferredDeletionFlush))
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
        flags: BufferFlags,
    ) -> Result<BufferHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;

        let bindless_index = self.next_bindless_index;
        self.next_bindless_index += 1;

        self.buffers.insert(
            handle,
            MockBuffer {
                device_handle: device,
                size,
                alloc_size: size,
                data: vec![0u8; size as usize],
                bindless_index,
                flags,
                is_grant_readback: false,
                texture_copy_footprint: None,
            },
        );

        Ok(handle)
    }

    fn create_buffer_with_capacity(
        &mut self,
        device: DeviceHandle,
        initial_size: u64,
        capacity: u64,
        _access: BufferKind,
        _element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;

        let bindless_index = self.next_bindless_index;
        self.next_bindless_index += 1;

        let cap = capacity.max(initial_size);
        self.buffers.insert(
            handle,
            MockBuffer {
                device_handle: device,
                size: initial_size,
                alloc_size: cap,
                data: vec![0u8; cap as usize],
                bindless_index,
                flags,
                is_grant_readback: false,
                texture_copy_footprint: None,
            },
        );

        Ok((handle, cap))
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) {
        self.buffers.remove(&buffer);
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buf = self
            .buffers
            .get_mut(&buffer)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;

        let start = offset as usize;
        let end = start + data.len();
        if end > buf.size as usize {
            anyhow::bail!("Write exceeds buffer size");
        }

        buf.data[start..end].copy_from_slice(data);
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
        // Mock backend uses a unified bindless index
        self.buffers.get(&buffer).map(|b| b.bindless_index)
    }

    fn create_buffer_view(
        &mut self,
        parent: BufferHandle,
        offset: u64,
        size: u64,
        _element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        self.buffer_view_create_count += 1;
        let parent_buf = self.buffers.get(&parent).context("Invalid parent buffer handle")?;
        if offset + size > parent_buf.size {
            anyhow::bail!("View exceeds parent buffer size");
        }
        let device_handle = parent_buf.device_handle;

        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;
        let index = self.next_bindless_index;
        self.next_bindless_index += 1;

        self.buffers.insert(
            handle,
            MockBuffer {
                device_handle,
                size,
                alloc_size: size,
                data: vec![0; size as usize],
                bindless_index: index,
                flags: BufferFlags::empty(),
                is_grant_readback: false,
                texture_copy_footprint: None,
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
        let new_len = new_size as usize;
        if preserve_contents {
            buf.data.resize(new_len, 0);
        } else {
            buf.data = vec![0u8; new_len];
        }
        buf.size = new_size;
        buf.alloc_size = new_size;
        Ok(())
    }

    fn read_buffer_to_cpu(&mut self, _device: DeviceHandle, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        let buf = self
            .buffers
            .get(&buffer)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;

        let len_u64 = output.len() as u64;
        if len_u64 > buf.size {
            anyhow::bail!("Read would exceed buffer bounds");
        }
        let len = len_u64 as usize;
        output[..len].copy_from_slice(&buf.data[..len]);
        Ok(())
    }

    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;
        self.readback_alloc_count += 1;
        self.buffers.insert(
            handle,
            MockBuffer {
                device_handle: device,
                size,
                alloc_size: size,
                data: vec![0u8; size as usize],
                bindless_index: 0,
                flags: BufferFlags::empty(),
                is_grant_readback: true,
                texture_copy_footprint: None,
            },
        );
        Ok(handle)
    }

    fn query_texture_copy_footprint(
        &self,
        _device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<crate::backend::TextureCopyFootprint> {
        let row_pitch = width.saturating_mul(format.bytes_per_pixel());
        let logical_bytes = row_pitch as u64 * height as u64;
        Ok(crate::backend::TextureCopyFootprint {
            width,
            height,
            format,
            logical_bytes,
            staging_bytes: logical_bytes,
            row_pitch,
            footprint_offset: 0,
        })
    }

    fn texture_copy_retention_tag(&self, texture: TextureHandle) -> u64 {
        let _ = texture;
        0
    }

    fn alloc_texture_readback_staging(
        &mut self,
        device: DeviceHandle,
        layout: crate::backend::TextureCopyFootprint,
    ) -> Result<BufferHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;
        self.readback_alloc_count += 1;
        self.buffers.insert(
            handle,
            MockBuffer {
                device_handle: device,
                size: layout.staging_bytes,
                alloc_size: layout.staging_bytes,
                data: vec![0u8; layout.staging_bytes as usize],
                bindless_index: 0,
                flags: BufferFlags::empty(),
                is_grant_readback: true,
                texture_copy_footprint: Some(layout),
            },
        );
        Ok(handle)
    }

    fn read_texture_readback_staging(
        &self,
        buffer: BufferHandle,
        layout: crate::backend::TextureCopyFootprint,
        output: &mut [u8],
    ) -> Result<()> {
        if output.len() as u64 != layout.logical_bytes {
            anyhow::bail!(
                "read_texture_readback_staging size mismatch: expected {}, got {}",
                layout.logical_bytes,
                output.len()
            );
        }
        let buf = self
            .buffers
            .get(&buffer)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;
        if !buf.is_grant_readback {
            anyhow::bail!("read_texture_readback_staging requires a grant readback buffer");
        }
        let row_bytes = layout.tight_row_bytes() as usize;
        let pitch = layout.row_pitch as usize;
        for row in 0..layout.height as usize {
            let src = row * pitch;
            let dst = row * row_bytes;
            output[dst..dst + row_bytes].copy_from_slice(&buf.data[src..src + row_bytes]);
        }
        Ok(())
    }

    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        let buf = self
            .buffers
            .get(&buffer)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;
        if !buf.is_grant_readback {
            anyhow::bail!("read_readback_buffer requires a grant readback buffer");
        }
        let len = output.len().min(buf.data.len());
        output[..len].copy_from_slice(&buf.data[..len]);
        Ok(())
    }

    fn free_readback_buffer(&mut self, buffer: BufferHandle) {
        self.buffers.remove(&buffer);
        self.readback_free_count += 1;
    }

    #[cfg(test)]
    fn test_readback_alloc_count(&self) -> usize {
        self.readback_alloc_count
    }

    #[cfg(test)]
    fn test_readback_free_count(&self) -> usize {
        self.readback_free_count
    }

    #[cfg(test)]
    fn test_surface_present_count(&self) -> usize {
        self.surface_present_count
    }

    #[cfg(test)]
    fn test_wait_until_count(&self) -> usize {
        self.wait_until_count
    }

    fn clear_buffer(&mut self, _device: DeviceHandle, buffer: BufferHandle, offset: u64, size: u64) -> Result<()> {
        let buf = self
            .buffers
            .get_mut(&buffer)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;

        let clear_size = if size == 0 {
            buf.size.saturating_sub(offset) as usize
        } else {
            size as usize
        };

        let start = offset as usize;
        let end = (start + clear_size).min(buf.size as usize);
        buf.data[start..end].fill(0);
        Ok(())
    }

    fn create_shader_with_paths(
        &mut self,
        device: DeviceHandle,
        slang_source: &str,
        _search_paths: &[&str],
        _defines: &[(&str, &str)],
        _optimization_level: crate::types::OptimizationLevel,
    ) -> Result<ShaderHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_shader_handle;
        self.next_shader_handle += 1;

        self.shaders.insert(
            handle,
            MockShader {
                device_handle: device,
                source: slang_source.to_string(),
            },
        );

        Ok(handle)
    }

    fn destroy_shader(&mut self, shader: ShaderHandle) {
        self.shaders.remove(&shader);
    }

    fn create_pipeline(
        &mut self,
        device: DeviceHandle,
        _vertex_shader: ShaderHandle,
        _fragment_shader: ShaderHandle,
        _vertex_layout: &VertexBufferLayout,
        _topology: PrimitiveTopology,
        _target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_pipeline_handle;
        self.next_pipeline_handle += 1;

        self.pipelines.insert(handle, MockPipeline { device_handle: device });

        Ok(handle)
    }

    fn destroy_pipeline(&mut self, pipeline: PipelineHandle) {
        self.pipelines.remove(&pipeline);
    }

    fn create_pipeline_with_depth(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        _depth_stencil: Option<&DepthStencilState>,
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

    // RenderTarget API
    fn create_render_target(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<RenderTargetHandle> {
        self.create_render_target_with_depth(device, width, height, format, None)
    }

    fn create_render_target_with_depth(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_render_target_handle;
        self.next_render_target_handle += 1;

        let size = (width * height * color_format.bytes_per_pixel()) as usize;
        self.render_targets.insert(
            handle,
            MockRenderTarget {
                device_handle: device,
                width,
                height,
                format: color_format,
                depth_format,
                has_rendered: false,
                data: vec![0u8; size],
            },
        );

        self.targets_created.push((width, height, color_format));
        self.targets_with_depth_created
            .push((width, height, color_format, depth_format));

        Ok(handle)
    }

    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        self.render_targets.remove(&target);
    }

    fn render_to_target(
        &mut self,
        device: DeviceHandle,
        target: RenderTargetHandle,
        commands: &[RenderCommand],
    ) -> Result<()> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let render_target = self
            .render_targets
            .get_mut(&target)
            .ok_or_else(|| anyhow::anyhow!("Invalid render target handle"))?;

        if render_target.device_handle != device {
            anyhow::bail!("Render target belongs to a different device");
        }

        // Record commands
        self.recorded_commands.push(commands.to_vec());

        // Simulate rendering by filling with a pattern based on clear color
        let clear_color = commands
            .iter()
            .find_map(|c| match c {
                RenderCommand::Clear(color) => Some(*color),
                _ => None,
            })
            .unwrap_or(Color::BLACK);

        // Fill with the clear color
        let r = (clear_color.r * 255.0) as u8;
        let g = (clear_color.g * 255.0) as u8;
        let b = (clear_color.b * 255.0) as u8;
        let a = (clear_color.a * 255.0) as u8;

        for i in (0..render_target.data.len()).step_by(4) {
            if i + 3 < render_target.data.len() {
                render_target.data[i] = r;
                render_target.data[i + 1] = g;
                render_target.data[i + 2] = b;
                render_target.data[i + 3] = a;
            }
        }

        render_target.has_rendered = true;

        Ok(())
    }

    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()> {
        let render_target = self
            .render_targets
            .get(&target)
            .ok_or_else(|| anyhow::anyhow!("Invalid render target handle"))?;

        if !render_target.has_rendered {
            anyhow::bail!("Cannot read from render target that hasn't been rendered to");
        }

        let expected_size = render_target.data.len();
        if output.len() < expected_size {
            anyhow::bail!("Output buffer too small: {} < {}", output.len(), expected_size);
        }

        output[..expected_size].copy_from_slice(&render_target.data);
        self.cpu_readback_count += 1;

        Ok(())
    }

    // Surface API (mock implementation)
    fn create_surface(
        &mut self,
        device: DeviceHandle,
        _window: &dyn raw_window_handle::HasWindowHandle,
        _display: &dyn raw_window_handle::HasDisplayHandle,
        _depth_format: Option<DepthFormat>,
    ) -> Result<SurfaceHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_surface_handle;
        self.next_surface_handle += 1;

        self.surfaces.insert(
            handle,
            MockSurface {
                device_handle: device,
                width: 800, // Default size
                height: 600,
                format: self.default_surface_format, // Use configured format
                next_image: 1,
                current_texture_handle: None,
                pending_frame_compute: Vec::new(),
            },
        );
        self.surface_pending_acquire.insert(handle, 0);

        Ok(handle)
    }

    fn destroy_surface(&mut self, surface: SurfaceHandle) {
        self.surfaces.remove(&surface);
        self.surface_pending_acquire.remove(&surface);
    }

    fn begin_frame(&mut self, surface: SurfaceHandle, ctx: ContextHandle) -> Result<(FrameToken, TextureHandle)> {
        let surf = self
            .surfaces
            .get_mut(&surface)
            .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?;

        let image = surf.next_image;
        surf.next_image += 1;
        surf.pending_frame_compute.clear();

        let tex_handle = self.next_texture_handle;
        self.next_texture_handle += 1;
        let bindless_index = self.next_bindless_index;
        self.next_bindless_index += 1;
        let width = surf.width;
        let height = surf.height;
        let format = surf.format;
        let device_handle = surf.device_handle;
        surf.current_texture_handle = Some(tex_handle);

        self.textures.insert(
            tex_handle,
            MockTexture {
                device_handle,
                width,
                height,
                format,
                data: vec![0; (width * height * format.bytes_per_pixel()) as usize],
                bindless_index,
                sampled_bindless_index: None,
            },
        );

        *self.surface_pending_acquire.entry(surface).or_insert(0) += 1;
        self.push_context_signal(
            ctx,
            crate::signal::Signal::SwapchainAcquired {
                image_index: image as u32,
            },
        );

        Ok((
            FrameToken {
                surface,
                image,
                context: ctx,
                frame_slot: image as u32,
                present_slot: image as u32,
            },
            tex_handle,
        ))
    }

    fn cancel_frame(&mut self, frame: FrameToken) -> Result<()> {
        if self.surface_pending_acquire.get(&frame.surface).copied().unwrap_or(0) == 0 {
            tracing::warn!(
                surface = frame.surface,
                image = frame.image,
                "cancel_frame: pending_acquire_count was already 0"
            );
        } else if let Some(count) = self.surface_pending_acquire.get_mut(&frame.surface) {
            *count = count.saturating_sub(1);
        }
        if let Some(surf) = self.surfaces.get_mut(&frame.surface) {
            surf.pending_frame_compute.clear();
            if surf.current_texture_handle.is_some() {
                surf.current_texture_handle = None;
            }
        }
        Ok(())
    }

    fn record_render(&mut self, frame: &FrameToken, commands: &[RenderCommand]) -> Result<()> {
        if !self.surfaces.contains_key(&frame.surface) {
            anyhow::bail!("Invalid surface handle");
        }

        self.recorded_commands.push(commands.to_vec());
        Ok(())
    }

    fn surface_resize(&mut self, surface: SurfaceHandle, width: u32, height: u32) -> Result<()> {
        let surf = self
            .surfaces
            .get_mut(&surface)
            .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?;

        surf.width = width;
        surf.height = height;
        if let Some(count) = self.surface_pending_acquire.get_mut(&surface) {
            *count = 0;
        }
        Ok(())
    }

    fn surface_size(&self, surface: SurfaceHandle) -> (u32, u32) {
        self.surfaces
            .get(&surface)
            .map(|s| (s.width, s.height))
            .unwrap_or((0, 0))
    }

    fn surface_format(&self, surface: SurfaceHandle) -> TextureFormat {
        self.surfaces
            .get(&surface)
            .map(|s| s.format)
            .unwrap_or(TextureFormat::Bgra8UnormSrgb)
    }

    // Texture management
    fn create_texture(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        _flags: TextureFlags,
    ) -> Result<TextureHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_texture_handle;
        self.next_texture_handle += 1;

        let bindless_index = self.next_bindless_index;
        self.next_bindless_index += 1;

        // For DirectInterpolated, allocate a second slot for the sampled-texture pool.
        let sampled_bindless_index = if matches!(access, TextureKind::DirectInterpolated) {
            let idx = self.next_bindless_index;
            self.next_bindless_index += 1;
            Some(idx)
        } else {
            None
        };

        let size = (width * height * format.bytes_per_pixel()) as usize;
        self.textures.insert(
            handle,
            MockTexture {
                device_handle: device,
                width,
                height,
                format,
                data: vec![0u8; size],
                bindless_index,
                sampled_bindless_index,
            },
        );

        self.textures_created += 1;
        Ok(handle)
    }

    fn write_texture(&mut self, texture: TextureHandle, data: &[u8], width: u32, height: u32) -> Result<()> {
        let tex = self
            .textures
            .get_mut(&texture)
            .ok_or_else(|| anyhow::anyhow!("Invalid texture handle"))?;

        if tex.width != width || tex.height != height {
            anyhow::bail!(
                "Texture dimensions mismatch: expected {}x{}, got {}x{}",
                tex.width,
                tex.height,
                width,
                height
            );
        }

        let expected_size = (width * height * tex.format.bytes_per_pixel()) as usize;
        if data.len() != expected_size {
            anyhow::bail!("Data size mismatch: expected {}, got {}", expected_size, data.len());
        }

        tex.data.copy_from_slice(data);
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
        let tex = self
            .textures
            .get_mut(&texture)
            .ok_or_else(|| anyhow::anyhow!("Invalid texture handle"))?;

        if x + width > tex.width || y + height > tex.height {
            anyhow::bail!(
                "Region out of bounds: {}x{} at ({},{}) exceeds {}x{} texture",
                width,
                height,
                x,
                y,
                tex.width,
                tex.height
            );
        }

        let bpp = tex.format.bytes_per_pixel() as usize;
        let expected_size = (width * height) as usize * bpp;
        if data.len() != expected_size {
            anyhow::bail!("Data size mismatch: expected {}, got {}", expected_size, data.len());
        }

        let tex_row_bytes = (tex.width * tex.format.bytes_per_pixel()) as usize;
        for row in 0..(height as usize) {
            let src_offset = row * (width as usize) * bpp;
            let dst_offset = ((y as usize + row) * tex_row_bytes) + (x as usize * bpp);
            let row_bytes = (width as usize) * bpp;
            tex.data[dst_offset..dst_offset + row_bytes].copy_from_slice(&data[src_offset..src_offset + row_bytes]);
        }
        Ok(())
    }

    fn destroy_texture(&mut self, texture: TextureHandle) {
        self.textures.remove(&texture);
    }

    fn read_texture_to_cpu(&mut self, texture: TextureHandle, output: &mut [u8]) -> Result<()> {
        let tex = self
            .textures
            .get(&texture)
            .ok_or_else(|| anyhow::anyhow!("Invalid texture handle"))?;

        let expected_size = tex.data.len();
        if output.len() < expected_size {
            anyhow::bail!("Output buffer too small: {} < {}", output.len(), expected_size);
        }
        output[..expected_size].copy_from_slice(&tex.data);
        Ok(())
    }

    fn texture_bindless_index(&self, texture: TextureHandle) -> Option<u32> {
        self.textures.get(&texture).map(|t| t.bindless_index)
    }

    fn texture_bindless_sampled_index(&self, texture: TextureHandle) -> Option<u32> {
        self.textures.get(&texture).and_then(|t| t.sampled_bindless_index)
    }

    // Sampler management
    fn create_sampler(&mut self, device: DeviceHandle, desc: &SamplerDesc) -> Result<SamplerHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_sampler_handle;
        self.next_sampler_handle += 1;

        let bindless_index = self.next_bindless_index;
        self.next_bindless_index += 1;

        self.samplers.insert(
            handle,
            MockSampler {
                device_handle: device,
                desc: desc.clone(),
                bindless_index,
            },
        );

        self.samplers_created += 1;
        Ok(handle)
    }

    fn destroy_sampler(&mut self, sampler: SamplerHandle) {
        self.samplers.remove(&sampler);
    }

    fn sampler_bindless_index(&self, sampler: SamplerHandle) -> Option<u32> {
        self.samplers.get(&sampler).map(|s| s.bindless_index)
    }

    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue {
        self.context_state(ctx).completed
    }

    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue {
        self.device_retired(device)
    }

    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> anyhow::Result<()> {
        if let Some(dev) = self.devices.get(&device) {
            dev.submission_worker.flush()?;
            let horizon = self.mock_scheduled_horizon(device);
            dev.submission_worker.wait_submitted_if_scheduled(value, horizon)?;
        }
        // On the mock, advance every context on this device to at least `value`
        // so that device_retired() >= value after the call.
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
    ) -> Vec<crate::signal::Signal> {
        crate::signal::drain_all_signals(&self.context_state(ctx).signal_queue)
    }

    fn peek_oldest_in_flight(&self, ctx: ContextHandle) -> Option<crate::timeline::TimelineValue> {
        let state = self.context_state(ctx);
        if state.completed < state.last_submitted_seq {
            Some(state.completed.saturating_add(1))
        } else {
            None
        }
    }

    fn pending_acquire_count(&self, surface: SurfaceHandle) -> u32 {
        self.surface_pending_acquire.get(&surface).copied().unwrap_or(0)
    }

    fn wait_until_timeout(
        &mut self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
        timeout_ms: u32,
    ) -> Result<bool> {
        if self.gpu_progress(ctx) >= value {
            return Ok(true);
        }
        let device = self.context_device(ctx);
        if let Some(dev) = self.devices.get(&device) {
            let horizon = self.mock_scheduled_horizon(device);
            if !dev
                .submission_worker
                .wait_submitted_if_scheduled_timeout(value, horizon, timeout_ms)?
            {
                return Ok(false);
            }
        }
        self.finish_timeline_wait(ctx, value)?;
        Ok(true)
    }

    fn submit_standalone(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&super::SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        let device = self.context_device(ctx);
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        self.record_submit_sync(sync)?;

        let effective = super::commands_with_sync_prologue(commands, sync);
        self.recorded_compute_commands.push(effective.clone());
        self.compute_dispatch_count += 1;

        for cmd in &effective {
            match cmd {
                GpuCommand::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    self.execute_copy_buffer(*src, *src_offset, *dst, *dst_offset, *size)?;
                }
                GpuCommand::CopyTextureToReadback { src, dst, layout } => {
                    self.execute_copy_texture_to_readback(*src, *dst, *layout)?;
                }
                _ => {}
            }
        }

        let dev = self
            .devices
            .get(&device)
            .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?;
        let tv = crate::backend::submission_worker::allocate_timeline_value(&dev.timeline_next);
        {
            let mut state = self.context_state_mut(ctx);
            state.last_submitted_seq = tv;
        }
        self.run_mock_submit_now(ctx, tv)?;
        Ok(tv)
    }

    fn submit_graph(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        sync: Option<&super::SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        let device = self.context_device(ctx);
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        self.recorded_graph_syncs.push(sync.is_some());

        let mut batch: Vec<GpuCommand> = Vec::new();
        let mut last_tv;
        last_tv = self.gpu_progress(ctx);
        for cmd in commands {
            match cmd {
                GraphCommand::Compute(c) => batch.push(c.clone()),
                GraphCommand::Render {
                    target,
                    commands: render_cmds,
                } => {
                    if !batch.is_empty() {
                        #[allow(unused_assignments)]
                        {
                            last_tv = self.submit_standalone(ctx, &batch, sync)?;
                        }
                        batch.clear();
                    }
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

    fn submit_graph_and_retain(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&super::SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        self.retained_graphs.insert((ctx, key), commands.to_vec());
        self.submit_graph(ctx, commands, sync)
    }

    fn try_resubmit_retained(
        &mut self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&super::SubmitSync>,
    ) -> Result<Option<crate::timeline::TimelineValue>> {
        let Some(commands) = self.retained_graphs.get(&(ctx, key)).cloned() else {
            return Ok(None);
        };
        self.retained_resubmit_count += 1;
        self.submit_graph(ctx, &commands, sync).map(Some)
    }

    fn record_gpu_work(&mut self, frame: &FrameToken, commands: &[GpuCommand]) -> Result<()> {
        let surf = self
            .surfaces
            .get_mut(&frame.surface)
            .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?;
        surf.pending_frame_compute.extend_from_slice(commands);
        Ok(())
    }

    fn submit_frame(&mut self, frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        let device = self
            .surfaces
            .get(&frame.surface)
            .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?
            .device_handle;

        let pending = {
            let surf = self
                .surfaces
                .get_mut(&frame.surface)
                .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?;
            std::mem::take(&mut surf.pending_frame_compute)
        };
        if !pending.is_empty() {
            self.recorded_compute_commands.push(pending);
            self.compute_dispatch_count += 1;
        }

        let dev = self
            .devices
            .get(&device)
            .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?;
        let tv = crate::backend::submission_worker::allocate_timeline_value(&dev.timeline_next);
        self.context_state_mut(frame.context).last_submitted_seq = tv;
        self.run_mock_submit_now(frame.context, tv)?;
        Ok(tv)
    }

    fn present_frame(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        let work = self.take_present_gpu_work(frame, submit_tv)?;
        let finish = work.run()?;
        self.finish_present(finish, submit_tv)
    }

    // Compute pipeline management
    fn create_compute_pipeline(
        &mut self,
        device: DeviceHandle,
        _compute_shader: ShaderHandle,
    ) -> Result<ComputePipelineHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_compute_pipeline_handle;
        self.next_compute_pipeline_handle += 1;

        self.compute_pipelines
            .insert(handle, MockComputePipeline { device_handle: device });

        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        self.compute_pipelines.remove(&pipeline);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_backend_creation() {
        let backend = MockBackend::new();
        assert_eq!(backend.enumerate_adapters().len(), 1);
        assert_eq!(backend.enumerate_adapters()[0].name, "Mock GPU");
    }

    #[test]
    fn test_device_creation() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        assert!(backend.is_device_valid(device));

        backend.destroy_device(device);
        assert!(!backend.is_device_valid(device));
    }

    #[test]
    fn test_render_target_creation() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();

        let target = backend
            .create_render_target(device, 800, 600, TextureFormat::Rgba8Unorm)
            .unwrap();

        assert_eq!(backend.targets_created.len(), 1);
        assert_eq!(backend.targets_created[0], (800, 600, TextureFormat::Rgba8Unorm));

        backend.destroy_render_target(target);
    }

    #[test]
    fn test_render_without_readback() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend
            .create_render_target(device, 100, 100, TextureFormat::Rgba8Unorm)
            .unwrap();

        let commands = vec![RenderCommand::Clear(Color::RED)];
        backend.render_to_target(device, target, &commands).unwrap();

        // No CPU readback should have occurred
        assert_eq!(backend.cpu_readback_count, 0);

        // Commands should be recorded
        assert_eq!(backend.recorded_commands.len(), 1);
    }

    #[test]
    fn test_explicit_readback() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend
            .create_render_target(device, 2, 2, TextureFormat::Rgba8Unorm)
            .unwrap();

        let commands = vec![RenderCommand::Clear(Color::RED)];
        backend.render_to_target(device, target, &commands).unwrap();

        // Now explicitly read back
        let mut output = vec![0u8; 2 * 2 * 4];
        backend.read_target_to_cpu(target, &mut output).unwrap();

        assert_eq!(backend.cpu_readback_count, 1);

        // Check the clear color was applied (RED = 255, 0, 0, 255)
        assert_eq!(output[0], 255); // R
        assert_eq!(output[1], 0); // G
        assert_eq!(output[2], 0); // B
        assert_eq!(output[3], 255); // A
    }

    #[test]
    fn test_readback_requires_render() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend
            .create_render_target(device, 10, 10, TextureFormat::Rgba8Unorm)
            .unwrap();

        // Try to read without rendering first
        let mut output = vec![0u8; 10 * 10 * 4];
        let result = backend.read_target_to_cpu(target, &mut output);

        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_renders_same_target() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend
            .create_render_target(device, 10, 10, TextureFormat::Rgba8Unorm)
            .unwrap();

        // Render multiple times to the same target
        backend
            .render_to_target(device, target, &[RenderCommand::Clear(Color::RED)])
            .unwrap();
        backend
            .render_to_target(device, target, &[RenderCommand::Clear(Color::GREEN)])
            .unwrap();
        backend
            .render_to_target(device, target, &[RenderCommand::Clear(Color::BLUE)])
            .unwrap();

        assert_eq!(backend.recorded_commands.len(), 3);

        // Only one target was created
        assert_eq!(backend.targets_created.len(), 1);
    }

    #[test]
    fn test_indexed_drawing_commands() {
        use crate::types::IndexFormat;

        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend
            .create_render_target(device, 100, 100, TextureFormat::Rgba8Unorm)
            .unwrap();

        // Create an index buffer
        let index_buffer = backend
            .create_buffer(device, 12, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();

        // Write some indices (6 u16 indices for 2 triangles)
        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];
        backend
            .write_buffer(index_buffer, 0, bytemuck::cast_slice(&indices))
            .unwrap();

        // Record indexed drawing commands
        let commands = vec![
            RenderCommand::Clear(Color::BLACK),
            RenderCommand::SetIndexBuffer {
                buffer: index_buffer,
                offset: 0,
                format: IndexFormat::Uint16,
            },
            RenderCommand::DrawIndexed {
                index_count: 6,
                instance_count: 1,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            },
        ];

        backend.render_to_target(device, target, &commands).unwrap();

        // Verify commands were recorded
        assert_eq!(backend.recorded_commands.len(), 1);
        assert_eq!(backend.recorded_commands[0].len(), 3);

        // Check SetIndexBuffer was recorded correctly
        match &backend.recorded_commands[0][1] {
            RenderCommand::SetIndexBuffer { buffer, offset, format } => {
                assert_eq!(*buffer, index_buffer);
                assert_eq!(*offset, 0);
                assert_eq!(*format, IndexFormat::Uint16);
            }
            _ => panic!("Expected SetIndexBuffer command"),
        }

        // Check DrawIndexed was recorded correctly
        match &backend.recorded_commands[0][2] {
            RenderCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => {
                assert_eq!(*index_count, 6);
                assert_eq!(*instance_count, 1);
                assert_eq!(*first_index, 0);
                assert_eq!(*base_vertex, 0);
                assert_eq!(*first_instance, 0);
            }
            _ => panic!("Expected DrawIndexed command"),
        }
    }

    #[test]
    fn test_indexed_drawing_with_offset() {
        use crate::types::IndexFormat;

        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend
            .create_render_target(device, 100, 100, TextureFormat::Rgba8Unorm)
            .unwrap();
        let index_buffer = backend
            .create_buffer(device, 24, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();

        // Test with offset and base_vertex
        let commands = vec![
            RenderCommand::SetIndexBuffer {
                buffer: index_buffer,
                offset: 12, // Skip first 3 u32 indices
                format: IndexFormat::Uint32,
            },
            RenderCommand::DrawIndexed {
                index_count: 3,
                instance_count: 10,
                first_index: 0,
                base_vertex: 100, // Offset into vertex buffer
                first_instance: 5,
            },
        ];

        backend.render_to_target(device, target, &commands).unwrap();

        // Verify the offset and base_vertex were preserved
        match &backend.recorded_commands[0][0] {
            RenderCommand::SetIndexBuffer { offset, format, .. } => {
                assert_eq!(*offset, 12);
                assert_eq!(*format, IndexFormat::Uint32);
            }
            _ => panic!("Expected SetIndexBuffer command"),
        }

        match &backend.recorded_commands[0][1] {
            RenderCommand::DrawIndexed {
                base_vertex,
                first_instance,
                instance_count,
                ..
            } => {
                assert_eq!(*base_vertex, 100);
                assert_eq!(*first_instance, 5);
                assert_eq!(*instance_count, 10);
            }
            _ => panic!("Expected DrawIndexed command"),
        }
    }

    #[test]
    fn test_surface_format_default() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();

        // Create a mock window handle for surface creation
        struct MockWindow;
        impl raw_window_handle::HasWindowHandle for MockWindow {
            fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
                // Return a null handle - mock backend doesn't use it
                Ok(unsafe {
                    raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Web(
                        raw_window_handle::WebWindowHandle::new(0),
                    ))
                })
            }
        }
        impl raw_window_handle::HasDisplayHandle for MockWindow {
            fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Web(
                        raw_window_handle::WebDisplayHandle::new(),
                    ))
                })
            }
        }

        let surface = backend.create_surface(device, &MockWindow, &MockWindow, None).unwrap();

        // Default format should be Bgra8UnormSrgb
        assert_eq!(backend.surface_format(surface), TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_surface_format_configurable() {
        let mut backend = MockBackend::new();

        // Configure a different format (simulating a GPU that prefers RGBA)
        backend.set_default_surface_format(TextureFormat::Rgba8Unorm);

        let device = backend.create_device(0).unwrap();

        struct MockWindow;
        impl raw_window_handle::HasWindowHandle for MockWindow {
            fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Web(
                        raw_window_handle::WebWindowHandle::new(0),
                    ))
                })
            }
        }
        impl raw_window_handle::HasDisplayHandle for MockWindow {
            fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Web(
                        raw_window_handle::WebDisplayHandle::new(),
                    ))
                })
            }
        }

        let surface = backend.create_surface(device, &MockWindow, &MockWindow, None).unwrap();

        // Should return the configured format, not the default
        assert_eq!(backend.surface_format(surface), TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_surface_format_multiple_formats() {
        // Test that different surfaces can have different formats
        // (when configured between creations)
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();

        struct MockWindow;
        impl raw_window_handle::HasWindowHandle for MockWindow {
            fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Web(
                        raw_window_handle::WebWindowHandle::new(0),
                    ))
                })
            }
        }
        impl raw_window_handle::HasDisplayHandle for MockWindow {
            fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Web(
                        raw_window_handle::WebDisplayHandle::new(),
                    ))
                })
            }
        }

        // Create first surface with default format
        let surface1 = backend.create_surface(device, &MockWindow, &MockWindow, None).unwrap();
        assert_eq!(backend.surface_format(surface1), TextureFormat::Bgra8UnormSrgb);

        // Change default and create second surface
        backend.set_default_surface_format(TextureFormat::Rgba8UnormSrgb);
        let surface2 = backend.create_surface(device, &MockWindow, &MockWindow, None).unwrap();

        // First surface should retain its original format
        assert_eq!(backend.surface_format(surface1), TextureFormat::Bgra8UnormSrgb);
        // Second surface should have the new format
        assert_eq!(backend.surface_format(surface2), TextureFormat::Rgba8UnormSrgb);
    }

    #[test]
    fn test_buffer_bindless_index() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();

        // Create multiple buffers and verify they get sequential bindless indices
        let buffer1 = backend
            .create_buffer(device, 64, BufferKind::Broadcast, None, BufferFlags::empty())
            .unwrap();
        let buffer2 = backend
            .create_buffer(device, 128, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();
        let buffer3 = backend
            .create_buffer(device, 256, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();

        assert_eq!(backend.buffer_bindless_index(buffer1), Some(0));
        assert_eq!(backend.buffer_bindless_index(buffer2), Some(1));
        assert_eq!(backend.buffer_bindless_index(buffer3), Some(2));
    }

    #[test]
    fn test_texture_bindless_index() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();

        // Create a buffer first to verify textures share the same index namespace
        let _buffer = backend
            .create_buffer(device, 64, BufferKind::Broadcast, None, BufferFlags::empty())
            .unwrap();

        let texture1 = backend
            .create_texture(
                device,
                256,
                256,
                TextureFormat::Rgba8Unorm,
                TextureKind::Interpolated,
                TextureFlags::empty(),
            )
            .unwrap();
        let texture2 = backend
            .create_texture(
                device,
                512,
                512,
                TextureFormat::Rgba8Unorm,
                TextureKind::Interpolated,
                TextureFlags::empty(),
            )
            .unwrap();

        // Textures should continue from where buffers left off
        assert_eq!(backend.texture_bindless_index(texture1), Some(1));
        assert_eq!(backend.texture_bindless_index(texture2), Some(2));
    }

    #[test]
    fn test_sampler_bindless_index() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();

        let sampler1 = backend.create_sampler(device, &SamplerDesc::default()).unwrap();
        let sampler2 = backend.create_sampler(device, &SamplerDesc::default()).unwrap();

        assert_eq!(backend.sampler_bindless_index(sampler1), Some(0));
        assert_eq!(backend.sampler_bindless_index(sampler2), Some(1));
    }

    #[test]
    fn test_bindless_indices_shared_namespace() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();

        // Create resources in interleaved order to verify shared namespace
        let buffer1 = backend
            .create_buffer(device, 64, BufferKind::Broadcast, None, BufferFlags::empty())
            .unwrap();
        let texture1 = backend
            .create_texture(
                device,
                256,
                256,
                TextureFormat::Rgba8Unorm,
                TextureKind::Interpolated,
                TextureFlags::empty(),
            )
            .unwrap();
        let sampler1 = backend.create_sampler(device, &SamplerDesc::default()).unwrap();
        let buffer2 = backend
            .create_buffer(device, 128, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();

        // All resources share a single incrementing index
        assert_eq!(backend.buffer_bindless_index(buffer1), Some(0));
        assert_eq!(backend.texture_bindless_index(texture1), Some(1));
        assert_eq!(backend.sampler_bindless_index(sampler1), Some(2));
        assert_eq!(backend.buffer_bindless_index(buffer2), Some(3));
    }

    #[test]
    fn test_bindless_index_invalid_handle() {
        let backend = MockBackend::new();

        // Invalid handles should return None
        assert_eq!(backend.buffer_bindless_index(999), None);
        assert_eq!(backend.texture_bindless_index(999), None);
        assert_eq!(backend.sampler_bindless_index(999), None);
    }

    #[test]
    fn test_bind_resources_command_recording() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend
            .create_render_target(device, 100, 100, TextureFormat::Rgba8Unorm)
            .unwrap();

        let buffer1 = backend
            .create_buffer(device, 64, BufferKind::Broadcast, None, BufferFlags::empty())
            .unwrap();
        let buffer2 = backend
            .create_buffer(device, 128, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();

        let commands = vec![
            RenderCommand::Clear(Color::BLACK),
            RenderCommand::BindResources {
                buffers: vec![buffer1, buffer2],
            },
        ];

        backend.render_to_target(device, target, &commands).unwrap();

        assert_eq!(backend.recorded_commands.len(), 1);
        assert_eq!(backend.recorded_commands[0].len(), 2);

        match &backend.recorded_commands[0][1] {
            RenderCommand::BindResources { buffers } => {
                assert_eq!(buffers.len(), 2);
                assert_eq!(buffers[0], buffer1);
                assert_eq!(buffers[1], buffer2);
            }
            _ => panic!("Expected BindResources command"),
        }
    }

    #[test]
    fn test_bind_resources_raw_command_recording() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend
            .create_render_target(device, 100, 100, TextureFormat::Rgba8Unorm)
            .unwrap();

        let commands = vec![
            RenderCommand::Clear(Color::BLACK),
            RenderCommand::BindResourcesRaw {
                indices: vec![0, 1, 2, 3],
                user: vec![],
                frame_table_base: 0,
            },
        ];

        backend.render_to_target(device, target, &commands).unwrap();

        assert_eq!(backend.recorded_commands.len(), 1);
        assert_eq!(backend.recorded_commands[0].len(), 2);

        match &backend.recorded_commands[0][1] {
            RenderCommand::BindResourcesRaw { indices, .. } => {
                assert_eq!(*indices, vec![0, 1, 2, 3]);
            }
            _ => panic!("Expected BindResourcesRaw command"),
        }
    }

    #[test]
    fn test_compute_bind_resources_recording() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();

        let buffer1 = backend
            .create_buffer(device, 64, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();
        let buffer2 = backend
            .create_buffer(device, 128, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();

        let idx1 = backend.buffer_bindless_index(buffer1).unwrap();
        let idx2 = backend.buffer_bindless_index(buffer2).unwrap();
        let commands = vec![
            GpuCommand::BindResourcesRaw {
                indices: vec![idx1, idx2],
                user: Vec::new(),
                frame_table_base: 0,
            },
            GpuCommand::Dispatch {
                label: None,
                workgroups_x: 8,
                workgroups_y: 8,
                workgroups_z: 1,
            },
        ];

        let ctx = backend.create_context(device).unwrap();
        let tv = backend.submit_standalone(ctx, &commands, None).unwrap();
        backend.wait_until(ctx, tv).unwrap();

        assert_eq!(backend.recorded_compute_commands.len(), 1);
        assert_eq!(backend.recorded_compute_commands[0].len(), 2);

        match &backend.recorded_compute_commands[0][0] {
            GpuCommand::BindResourcesRaw { indices, .. } => {
                assert_eq!(indices.as_slice(), &[idx1, idx2]);
            }
            _ => panic!("Expected BindResourcesRaw command"),
        }
    }

    #[test]
    fn submit_graph_does_not_cpu_wait() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend
            .create_render_target(device, 8, 8, TextureFormat::Rgba8Unorm)
            .unwrap();

        let commands = vec![
            GraphCommand::Compute(GpuCommand::SetPipeline(0)),
            GraphCommand::Compute(GpuCommand::Dispatch {
                label: None,
                workgroups_x: 1,
                workgroups_y: 1,
                workgroups_z: 1,
            }),
            GraphCommand::Render {
                target,
                commands: vec![RenderCommand::Clear(Color::RED)],
            },
            GraphCommand::Compute(GpuCommand::SetPipeline(0)),
            GraphCommand::Compute(GpuCommand::Dispatch {
                label: None,
                workgroups_x: 1,
                workgroups_y: 1,
                workgroups_z: 1,
            }),
        ];

        assert_eq!(backend.wait_until_count, 0);
        let ctx = backend.create_context(device).unwrap();
        backend.submit_graph(ctx, &commands, None).unwrap();
        assert_eq!(
            backend.wait_until_count, 0,
            "submit_graph should not call wait_until (no CPU waits)"
        );
    }

    #[test]
    fn acquire_release_pairs_on_mock() {
        struct MockWindow;
        impl raw_window_handle::HasWindowHandle for MockWindow {
            fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Web(
                        raw_window_handle::WebWindowHandle::new(0),
                    ))
                })
            }
        }
        impl raw_window_handle::HasDisplayHandle for MockWindow {
            fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Web(
                        raw_window_handle::WebDisplayHandle::new(),
                    ))
                })
            }
        }

        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let ctx = backend.create_context(device).unwrap();
        let surface = backend.create_surface(device, &MockWindow, &MockWindow, None).unwrap();

        let (frame, _tex) = backend.begin_frame(surface, ctx).unwrap();
        assert_eq!(backend.pending_acquire_count(surface), 1);

        backend.present_frame(frame, 0).unwrap();
        assert_eq!(backend.pending_acquire_count(surface), 0);

        let signals = backend.poll_signals(ctx, backend.gpu_progress(ctx));
        assert!(signals
            .iter()
            .any(|s| matches!(s, crate::signal::Signal::SwapchainAcquired { .. })));
        assert!(signals
            .iter()
            .any(|s| matches!(s, crate::signal::Signal::SwapchainReturned { .. })));
    }

    #[test]
    fn surface_frame_signals_stay_on_presenting_context() {
        struct MockWindow;
        impl raw_window_handle::HasWindowHandle for MockWindow {
            fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Web(
                        raw_window_handle::WebWindowHandle::new(0),
                    ))
                })
            }
        }
        impl raw_window_handle::HasDisplayHandle for MockWindow {
            fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Web(
                        raw_window_handle::WebDisplayHandle::new(),
                    ))
                })
            }
        }

        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let ctx_a = backend.create_context(device).unwrap();
        let ctx_b = backend.create_context(device).unwrap();
        let surface = backend.create_surface(device, &MockWindow, &MockWindow, None).unwrap();

        let (frame, _tex) = backend.begin_frame(surface, ctx_a).unwrap();
        assert!(backend.poll_signals(ctx_b, backend.gpu_progress(ctx_b)).is_empty());

        let tv = backend.submit_frame(&frame).unwrap();
        assert_eq!(backend.gpu_progress(ctx_a), tv);
        assert_eq!(backend.gpu_progress(ctx_b), 0);

        backend.present_frame(frame, tv).unwrap();
        let a_signals = backend.poll_signals(ctx_a, backend.gpu_progress(ctx_a));
        assert!(a_signals
            .iter()
            .any(|s| matches!(s, crate::signal::Signal::SwapchainReturned { .. })));
        assert!(backend.poll_signals(ctx_b, backend.gpu_progress(ctx_b)).is_empty());
    }

    #[test]
    fn counter_zero_after_resize() {
        struct MockWindow;
        impl raw_window_handle::HasWindowHandle for MockWindow {
            fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Web(
                        raw_window_handle::WebWindowHandle::new(0),
                    ))
                })
            }
        }
        impl raw_window_handle::HasDisplayHandle for MockWindow {
            fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Web(
                        raw_window_handle::WebDisplayHandle::new(),
                    ))
                })
            }
        }

        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let ctx = backend.create_context(device).unwrap();
        let surface = backend.create_surface(device, &MockWindow, &MockWindow, None).unwrap();

        let (_frame, _tex) = backend.begin_frame(surface, ctx).unwrap();
        assert_eq!(backend.pending_acquire_count(surface), 1);

        backend.surface_resize(surface, 1024, 768).unwrap();
        assert_eq!(backend.pending_acquire_count(surface), 0);

        let signals = backend.poll_signals(ctx, backend.gpu_progress(ctx));
        assert!(!signals
            .iter()
            .any(|s| matches!(s, crate::signal::Signal::SwapchainReturned { .. })));
    }

    #[test]
    fn destroy_context_floors_device_retired_when_signaled_lags_submitted() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let ctx = backend.create_context(device).unwrap();
        {
            let mut state = backend.context_state_mut(ctx);
            state.completed = 7;
            state.last_submitted_seq = 10;
        }
        backend
            .devices
            .get(&device)
            .unwrap()
            .timeline_next
            .store(10, std::sync::atomic::Ordering::Relaxed);

        backend.destroy_context(ctx);

        assert_eq!(
            backend.device_retired(device),
            10,
            "retired floor must include last_submitted_seq after destroy removes the context"
        );
    }
}
