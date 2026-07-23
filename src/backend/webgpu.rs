//! Compute-only WebGPU backend prototype.
//!
//! This backend deliberately does not emulate Goldy's native bindless heap. The
//! raw indices in [`GpuCommand::BindResourcesRaw`] are interpreted as backend
//! registry keys and packed into one fixed bind group in shader-parameter order.

use super::*;
use crate::types::{BufferResizeCost, DeviceType};
use anyhow::{Context as _, Result};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const RAW_WGSL_MARKER: &str = "// @goldy-wgsl";

pub(crate) struct WebGpuBackend {
    adapters: Vec<wgpu::Adapter>,
    adapter_info: Vec<AdapterInfo>,
    devices: HashMap<DeviceHandle, WebGpuDevice>,
    contexts: HashMap<ContextHandle, Arc<WebGpuContext>>,
    buffers: HashMap<BufferHandle, WebGpuBuffer>,
    buffer_slots: HashMap<u32, BufferHandle>,
    shaders: HashMap<ShaderHandle, WebGpuShader>,
    compute_pipelines: HashMap<ComputePipelineHandle, WebGpuComputePipeline>,
    next_device: DeviceHandle,
    next_context: ContextHandle,
    next_buffer: BufferHandle,
    next_slot: u32,
    next_shader: ShaderHandle,
    next_compute_pipeline: ComputePipelineHandle,
}

struct WebGpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    next_timeline: Arc<AtomicU64>,
    retired: Arc<AtomicU64>,
}

struct WebGpuContext {
    device: DeviceHandle,
    completed: AtomicU64,
    signal_queue: crate::signal::SignalQueue,
}

struct WebGpuProgress {
    context: Arc<WebGpuContext>,
}

impl ContextGpuProgress for WebGpuProgress {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue {
        self.context.completed.load(Ordering::Acquire)
    }
}

struct WebGpuDestroyContext;

impl ContextDestroyHandle for WebGpuDestroyContext {
    fn wait(&self) -> Result<()> {
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone)]
struct WebGpuBuffer {
    device: DeviceHandle,
    buffer: wgpu::Buffer,
    offset: u64,
    size: u64,
    capacity: u64,
    slot: Option<u32>,
    readback: bool,
}

struct WebGpuShader {
    device: DeviceHandle,
    source: String,
    search_paths: Vec<String>,
    defines: Vec<(String, String)>,
    optimization_level: crate::types::OptimizationLevel,
}

struct WebGpuComputePipeline {
    device: DeviceHandle,
    pipeline: wgpu::ComputePipeline,
    slot_access: Vec<Option<ResourceAccess>>,
}

impl WebGpuBackend {
    pub(crate) fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        if adapters.is_empty() {
            anyhow::bail!("WebGPU: no compatible adapters found");
        }
        let adapter_info = adapters
            .iter()
            .enumerate()
            .map(|(id, adapter)| {
                let info = adapter.get_info();
                AdapterInfo {
                    id: id as u32,
                    name: info.name,
                    vendor: format!("0x{:04x}", info.vendor),
                    backend: BackendType::WebGpu,
                    device_type: map_device_type(info.device_type),
                }
            })
            .collect();
        Ok(Self {
            adapters,
            adapter_info,
            devices: HashMap::new(),
            contexts: HashMap::new(),
            buffers: HashMap::new(),
            buffer_slots: HashMap::new(),
            shaders: HashMap::new(),
            compute_pipelines: HashMap::new(),
            next_device: 1,
            next_context: 1,
            next_buffer: 1,
            next_slot: 0,
            next_shader: 1,
            next_compute_pipeline: 1,
        })
    }

    fn device(&self, handle: DeviceHandle) -> Result<&WebGpuDevice> {
        self.devices.get(&handle).context("WebGPU: invalid device handle")
    }

    fn context(&self, handle: ContextHandle) -> Result<&Arc<WebGpuContext>> {
        self.contexts.get(&handle).context("WebGPU: invalid context handle")
    }

    fn unsupported<T>(operation: &str) -> Result<T> {
        anyhow::bail!("WebGPU compute-only backend does not support {operation}")
    }

    fn create_storage_buffer(
        &mut self,
        device: DeviceHandle,
        logical_size: u64,
        capacity: u64,
    ) -> Result<BufferHandle> {
        let capacity = capacity.max(logical_size).max(4);
        let gpu = self.device(device)?;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("goldy-webgpu-buffer"),
            size: capacity,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let handle = self.next_buffer;
        self.next_buffer += 1;
        let slot = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .context("WebGPU buffer registry exhausted")?;
        self.buffer_slots.insert(slot, handle);
        self.buffers.insert(
            handle,
            WebGpuBuffer {
                device,
                buffer,
                offset: 0,
                size: logical_size,
                capacity,
                slot: Some(slot),
                readback: false,
            },
        );
        Ok(handle)
    }

    fn compile_compute_wgsl(&self, shader: &WebGpuShader) -> Result<(String, Vec<Option<ResourceAccess>>)> {
        if let Some(wgsl) = shader.source.strip_prefix(RAW_WGSL_MARKER) {
            return Ok((wgsl.trim_start().to_owned(), Vec::new()));
        }

        let compiler = crate::slang::SlangCompiler::new().context("WebGPU: initialize Slang")?;
        let paths: Vec<&str> = shader.search_paths.iter().map(String::as_str).collect();
        let defines: Vec<(&str, &str)> = shader
            .defines
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        let webgpu_source = crate::slang::virtual_main::transform_virtual_main_webgpu_compute(&shader.source)
            .map_err(|error| anyhow::anyhow!("WebGPU shader lowering failed: {error}"))?;
        let compiled = compiler.compile_bindless_with_reflection_and_defines(
            &webgpu_source,
            crate::slang::ShaderTarget::Wgsl,
            &[("cs_main", crate::slang::SlangStage::Compute)],
            &paths,
            &defines,
            &[],
            shader.optimization_level,
        )?;
        let source = compiled
            .shader
            .as_str()
            .context("WebGPU: Slang returned non-text WGSL output")?
            .to_owned();
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
        Ok((source, access))
    }

    fn create_bind_group(
        &self,
        device_handle: DeviceHandle,
        pipeline: &wgpu::ComputePipeline,
        indices: &[u32],
    ) -> Result<Option<wgpu::BindGroup>> {
        if indices.is_empty() {
            return Ok(None);
        }
        let mut entries = Vec::with_capacity(indices.len());
        for (binding, index) in indices.iter().copied().enumerate() {
            let handle = self
                .buffer_slots
                .get(&index)
                .with_context(|| format!("WebGPU: binding {binding} references unknown registry key {index}"))?;
            let buffer = self
                .buffers
                .get(handle)
                .with_context(|| format!("WebGPU: registry key {index} references a destroyed buffer"))?;
            entries.push(wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer.buffer,
                    offset: buffer.offset,
                    size: NonZeroU64::new(buffer.size),
                }),
            });
        }
        let layout = pipeline.get_bind_group_layout(0);
        let device = &self.device(device_handle)?.device;
        Ok(Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("goldy-webgpu-dispatch-bindings"),
            layout: &layout,
            entries: &entries,
        })))
    }

    fn submit_commands(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
    ) -> Result<crate::timeline::TimelineValue> {
        let context = Arc::clone(self.context(ctx)?);
        let device_handle = context.device;
        let gpu = self.device(device_handle)?;
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        let next_timeline = Arc::clone(&gpu.next_timeline);
        let retired = Arc::clone(&gpu.retired);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("goldy-webgpu-compute"),
        });
        let mut current_pipeline: Option<ComputePipelineHandle> = None;
        let mut current_indices: Vec<u32> = Vec::new();

        for command in commands {
            match command {
                GpuCommand::SetPipeline(pipeline) => current_pipeline = Some(*pipeline),
                GpuCommand::BindResourcesRaw { indices, user, .. } => {
                    if !user.is_empty() {
                        anyhow::bail!(
                            "WebGPU: user scalar dispatch parameters are not implemented; use a bound uniform buffer"
                        );
                    }
                    current_indices.clone_from(indices);
                }
                GpuCommand::Dispatch {
                    label,
                    workgroups_x,
                    workgroups_y,
                    workgroups_z,
                } => {
                    let pipeline_handle = current_pipeline.context("WebGPU: dispatch without a compute pipeline")?;
                    let pipeline = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .context("WebGPU: invalid compute pipeline")?;
                    let bind_group = self.create_bind_group(pipeline.device, &pipeline.pipeline, &current_indices)?;
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: *label,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline.pipeline);
                    if let Some(bind_group) = bind_group.as_ref() {
                        pass.set_bind_group(0, bind_group, &[]);
                    }
                    pass.dispatch_workgroups(*workgroups_x, *workgroups_y, *workgroups_z);
                }
                GpuCommand::DispatchIndirect { buffer, offset, label } => {
                    let pipeline_handle = current_pipeline.context("WebGPU: indirect dispatch without a pipeline")?;
                    let pipeline = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .context("WebGPU: invalid compute pipeline")?;
                    let args = self.buffers.get(buffer).context("WebGPU: invalid indirect buffer")?;
                    let bind_group = self.create_bind_group(pipeline.device, &pipeline.pipeline, &current_indices)?;
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: *label,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline.pipeline);
                    if let Some(bind_group) = bind_group.as_ref() {
                        pass.set_bind_group(0, bind_group, &[]);
                    }
                    pass.dispatch_workgroups_indirect(&args.buffer, args.offset + offset);
                }
                GpuCommand::ClearBuffer { buffer, offset, size } => {
                    let buffer = self.buffers.get(buffer).context("WebGPU: invalid clear buffer")?;
                    let size = (*size != 0).then_some(*size);
                    encoder.clear_buffer(&buffer.buffer, buffer.offset + offset, size);
                }
                GpuCommand::WriteBuffer { buffer, offset, data } => {
                    let buffer = self.buffers.get(buffer).context("WebGPU: invalid write buffer")?;
                    queue.write_buffer(&buffer.buffer, buffer.offset + offset, data);
                }
                GpuCommand::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let src = self.buffers.get(src).context("WebGPU: invalid copy source")?;
                    let dst = self.buffers.get(dst).context("WebGPU: invalid copy destination")?;
                    encoder.copy_buffer_to_buffer(
                        &src.buffer,
                        src.offset + src_offset,
                        &dst.buffer,
                        dst.offset + dst_offset,
                        *size,
                    );
                }
                GpuCommand::FrameTableStaging { .. } | GpuCommand::ResourceBarrier { .. } => {
                    // The portable path binds resources directly. WebGPU tracks resource
                    // transitions and ordering within a submitted command buffer.
                }
                GpuCommand::DispatchBatch { .. } => {
                    anyhow::bail!("WebGPU: native dispatch batching is not supported")
                }
                GpuCommand::WriteTexture { .. }
                | GpuCommand::WriteTextureRegion { .. }
                | GpuCommand::CopyTexture { .. }
                | GpuCommand::CopyRenderTarget { .. }
                | GpuCommand::CopyBufferToTexture { .. }
                | GpuCommand::CopyTextureToReadback { .. } => {
                    anyhow::bail!("WebGPU compute-only backend: texture command is not supported")
                }
            }
        }

        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| anyhow::anyhow!("WebGPU device poll failed: {error}"))?;

        let value = next_timeline.fetch_add(1, Ordering::AcqRel);
        context.completed.store(value, Ordering::Release);
        retired.fetch_max(value, Ordering::AcqRel);
        context.signal_queue.push_boundary_crossed(value);
        Ok(value)
    }
}

fn map_device_type(device_type: wgpu::DeviceType) -> DeviceType {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
        wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
        wgpu::DeviceType::Cpu => DeviceType::Cpu,
        wgpu::DeviceType::Other | wgpu::DeviceType::VirtualGpu => DeviceType::Other,
    }
}

impl GpuBackendSubmitSession for WebGpuBackend {
    fn clone_context_submit_session(
        &self,
        _ctx: ContextHandle,
        backend: std::sync::Arc<std::sync::Mutex<Box<dyn GpuBackend>>>,
    ) -> std::sync::Arc<dyn ContextSubmitSession> {
        LockedSubmitSession::with_backend_type(backend, BackendType::WebGpu)
    }
}

impl GpuBackendTimelineWait for WebGpuBackend {
    fn take_timeline_submission_epoch_wait(
        &self,
        _ctx: ContextHandle,
        _value: crate::timeline::TimelineValue,
    ) -> Result<Option<submission_worker::SubmissionEpochWait>> {
        Ok(None)
    }

    fn take_timeline_blocking_wait(
        &self,
        _ctx: ContextHandle,
        _value: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn TimelineBlockingWait>>> {
        Ok(None)
    }

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        if self.gpu_progress(ctx) < value {
            anyhow::bail!("WebGPU: timeline value {value} was not submitted on context {ctx}");
        }
        Ok(())
    }
}

impl GpuBackendPresentSplit for WebGpuBackend {
    fn take_present_gpu_work(
        &mut self,
        _frame: FrameToken,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn PresentGpuWork>> {
        Self::unsupported("presentation")
    }

    fn finish_present(
        &mut self,
        _finish: PresentFinishState,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        Self::unsupported("presentation")
    }
}

impl GpuBackend for WebGpuBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn backend_type(&self) -> BackendType {
        BackendType::WebGpu
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        self.adapter_info.clone()
    }

    fn adapter_capabilities(&self, _adapter_id: u32) -> crate::device::DeviceCapabilities {
        crate::device::DeviceCapabilities {
            has_zero_copy_storage_readback: false,
            buffer_resize_cost: BufferResizeCost::Copy,
            buffer_decommit_supported: false,
            host_sidecar_on_submit_worker: false,
            split_compute_partitions_on_barrier_cost: false,
            fuse_upload_with_compute_partitions: true,
            ..crate::device::DeviceCapabilities::default()
        }
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        let adapter = self
            .adapters
            .get(adapter_id as usize)
            .with_context(|| format!("WebGPU: invalid adapter id {adapter_id}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("goldy-webgpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .context("WebGPU: request device")?;
        let handle = self.next_device;
        self.next_device += 1;
        self.devices.insert(
            handle,
            WebGpuDevice {
                device,
                queue,
                next_timeline: Arc::new(AtomicU64::new(1)),
                retired: Arc::new(AtomicU64::new(0)),
            },
        );
        Ok(handle)
    }

    fn destroy_device(&mut self, device: DeviceHandle) {
        if let Some(gpu) = self.devices.remove(&device) {
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.contexts.retain(|_, context| context.device != device);
        self.buffers.retain(|_, buffer| buffer.device != device);
        self.shaders.retain(|_, shader| shader.device != device);
        self.compute_pipelines.retain(|_, pipeline| pipeline.device != device);
        self.buffer_slots.retain(|_, handle| self.buffers.contains_key(handle));
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn device_wait_idle(&mut self, device: DeviceHandle) -> Result<()> {
        self.device(device)?
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| anyhow::anyhow!("WebGPU device poll failed: {error}"))
    }

    fn create_context(&mut self, device: DeviceHandle) -> Result<ContextHandle> {
        self.device(device)?;
        if self.contexts.values().any(|context| context.device == device) {
            anyhow::bail!("WebGPU prototype supports one submission context per device");
        }
        let handle = self.next_context;
        self.next_context += 1;
        self.contexts.insert(
            handle,
            Arc::new(WebGpuContext {
                device,
                completed: AtomicU64::new(0),
                signal_queue: crate::signal::SignalQueue::new(),
            }),
        );
        Ok(handle)
    }

    fn detach_context_for_destroy(&mut self, ctx: ContextHandle) -> Option<Box<dyn ContextDestroyHandle>> {
        self.contexts.remove(&ctx)?;
        Some(Box::new(WebGpuDestroyContext))
    }

    fn clone_context_deletion_flush(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn ContextDeferredDeletionFlush>> {
        self.contexts
            .contains_key(&ctx)
            .then(|| Arc::new(NoOpDeferredDeletionFlush) as Arc<dyn ContextDeferredDeletionFlush>)
    }

    fn clone_context_gpu_progress(&self, ctx: ContextHandle) -> Option<std::sync::Arc<dyn ContextGpuProgress>> {
        Some(Arc::new(WebGpuProgress {
            context: Arc::clone(self.contexts.get(&ctx)?),
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
        _element_stride: Option<u32>,
        _flags: BufferFlags,
    ) -> Result<BufferHandle> {
        self.create_storage_buffer(device, size, size)
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
        let capacity = capacity.max(initial_size);
        Ok((self.create_storage_buffer(device, initial_size, capacity)?, capacity))
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) {
        if let Some(buffer) = self.buffers.remove(&buffer) {
            if let Some(slot) = buffer.slot {
                self.buffer_slots.remove(&slot);
            }
        }
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buffer = self.buffers.get(&buffer).context("WebGPU: invalid buffer handle")?;
        if offset + data.len() as u64 > buffer.size {
            anyhow::bail!("WebGPU: write exceeds logical buffer size");
        }
        self.device(buffer.device)?
            .queue
            .write_buffer(&buffer.buffer, buffer.offset + offset, data);
        Ok(())
    }

    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle> {
        let gpu = self.device(device)?;
        let capacity = size.max(4);
        let raw = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("goldy-webgpu-readback"),
            size: capacity,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let handle = self.next_buffer;
        self.next_buffer += 1;
        self.buffers.insert(
            handle,
            WebGpuBuffer {
                device,
                buffer: raw,
                offset: 0,
                size,
                capacity,
                slot: None,
                readback: true,
            },
        );
        Ok(handle)
    }

    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        let buffer = self.buffers.get(&buffer).context("WebGPU: invalid readback buffer")?;
        if !buffer.readback {
            anyhow::bail!("WebGPU: buffer is not readback staging");
        }
        if output.len() as u64 > buffer.size {
            anyhow::bail!("WebGPU: read exceeds readback buffer size");
        }
        let slice = buffer.buffer.slice(0..output.len() as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device(buffer.device)?
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| anyhow::anyhow!("WebGPU readback poll failed: {error}"))?;
        rx.recv().context("WebGPU readback callback dropped")??;
        output.copy_from_slice(&slice.get_mapped_range());
        buffer.buffer.unmap();
        Ok(())
    }

    fn free_readback_buffer(&mut self, buffer: BufferHandle) {
        self.destroy_buffer(buffer);
    }

    fn query_texture_copy_footprint(
        &self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _format: TextureFormat,
    ) -> Result<TextureCopyFootprint> {
        Self::unsupported("texture readback")
    }

    fn alloc_texture_readback_staging(
        &mut self,
        _device: DeviceHandle,
        _layout: TextureCopyFootprint,
    ) -> Result<BufferHandle> {
        Self::unsupported("texture readback")
    }

    fn read_texture_readback_staging(
        &self,
        _buffer: BufferHandle,
        _layout: TextureCopyFootprint,
        _output: &mut [u8],
    ) -> Result<()> {
        Self::unsupported("texture readback")
    }

    fn texture_copy_retention_tag(&self, _texture: TextureHandle) -> u64 {
        0
    }

    fn clear_buffer(&mut self, device: DeviceHandle, buffer: BufferHandle, offset: u64, size: u64) -> Result<()> {
        let gpu = self.device(device)?;
        let target = self.buffers.get(&buffer).context("WebGPU: invalid buffer handle")?;
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("goldy-webgpu-clear"),
        });
        encoder.clear_buffer(&target.buffer, target.offset + offset, (size != 0).then_some(size));
        gpu.queue.submit([encoder.finish()]);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| anyhow::anyhow!("WebGPU clear poll failed: {error}"))
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
        let buffer = self.buffers.get_mut(&buffer).context("WebGPU: invalid buffer handle")?;
        if new_logical_size == 0 || new_logical_size > buffer.capacity {
            anyhow::bail!("WebGPU: logical size must be in 1..=capacity");
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
        _element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        let parent = self
            .buffers
            .get(&parent)
            .context("WebGPU: invalid parent buffer")?
            .clone();
        if offset + size > parent.size {
            anyhow::bail!("WebGPU: buffer view exceeds parent");
        }
        let handle = self.next_buffer;
        self.next_buffer += 1;
        let slot = self.next_slot;
        self.next_slot += 1;
        self.buffer_slots.insert(slot, handle);
        self.buffers.insert(
            handle,
            WebGpuBuffer {
                device: parent.device,
                buffer: parent.buffer,
                offset: parent.offset + offset,
                size,
                capacity: size,
                slot: Some(slot),
                readback: false,
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
            .context("WebGPU: invalid buffer handle")?
            .clone();
        if old.device != device {
            anyhow::bail!("WebGPU: buffer belongs to another device");
        }
        let gpu = self.device(device)?;
        let capacity = new_size.max(4);
        let replacement = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("goldy-webgpu-resized-buffer"),
            size: capacity,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        if preserve_contents {
            let copy_size = old.size.min(new_size);
            if copy_size > 0 {
                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("goldy-webgpu-resize-copy"),
                });
                encoder.copy_buffer_to_buffer(&old.buffer, old.offset, &replacement, 0, copy_size);
                gpu.queue.submit([encoder.finish()]);
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|error| anyhow::anyhow!("WebGPU resize poll failed: {error}"))?;
            }
        }
        let target = self.buffers.get_mut(&buffer).expect("validated above");
        target.buffer = replacement;
        target.offset = 0;
        target.size = new_size;
        target.capacity = capacity;
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
            WebGpuShader {
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

    fn create_pipeline(
        &mut self,
        _device: DeviceHandle,
        _vertex_shader: ShaderHandle,
        _fragment_shader: ShaderHandle,
        _vertex_layout: &VertexBufferLayout,
        _topology: PrimitiveTopology,
        _target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        Self::unsupported("graphics pipelines")
    }

    fn destroy_pipeline(&mut self, _pipeline: PipelineHandle) {}

    fn create_pipeline_with_depth(
        &mut self,
        _device: DeviceHandle,
        _vertex_shader: ShaderHandle,
        _fragment_shader: ShaderHandle,
        _vertex_layout: &VertexBufferLayout,
        _topology: PrimitiveTopology,
        _target_format: TextureFormat,
        _depth_stencil: Option<&DepthStencilState>,
    ) -> Result<PipelineHandle> {
        Self::unsupported("graphics pipelines")
    }

    fn create_render_target_with_depth(
        &mut self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _color_format: TextureFormat,
        _depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        Self::unsupported("render targets")
    }

    fn render_to_target(
        &mut self,
        _device: DeviceHandle,
        _target: RenderTargetHandle,
        _color_load: crate::types::TargetLoad,
        _commands: &[RenderCommand],
    ) -> Result<()> {
        Self::unsupported("rendering")
    }

    fn create_texture(
        &mut self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _format: TextureFormat,
        _access: TextureKind,
        _flags: TextureFlags,
    ) -> Result<TextureHandle> {
        Self::unsupported("textures")
    }

    fn write_texture(&mut self, _texture: TextureHandle, _data: &[u8], _width: u32, _height: u32) -> Result<()> {
        Self::unsupported("textures")
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
        Self::unsupported("textures")
    }

    fn destroy_texture(&mut self, _texture: TextureHandle) {}

    fn texture_bindless_index(&self, _texture: TextureHandle) -> Option<u32> {
        None
    }

    fn texture_bindless_sampled_index(&self, _texture: TextureHandle) -> Option<u32> {
        None
    }

    fn create_sampler(&mut self, _device: DeviceHandle, _desc: &SamplerDesc) -> Result<SamplerHandle> {
        Self::unsupported("samplers")
    }

    fn destroy_sampler(&mut self, _sampler: SamplerHandle) {}

    fn sampler_bindless_index(&self, _sampler: SamplerHandle) -> Option<u32> {
        None
    }

    fn create_surface(
        &mut self,
        _device: DeviceHandle,
        _window: &dyn raw_window_handle::HasWindowHandle,
        _display: &dyn raw_window_handle::HasDisplayHandle,
        _depth_format: Option<DepthFormat>,
    ) -> Result<SurfaceHandle> {
        Self::unsupported("surfaces")
    }

    fn destroy_surface(&mut self, _surface: SurfaceHandle) {}

    fn surface_resize(&mut self, _surface: SurfaceHandle, _width: u32, _height: u32) -> Result<()> {
        Self::unsupported("surfaces")
    }

    fn surface_size(&self, _surface: SurfaceHandle) -> (u32, u32) {
        (0, 0)
    }

    fn surface_format(&self, _surface: SurfaceHandle) -> TextureFormat {
        TextureFormat::Bgra8UnormSrgb
    }

    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue {
        self.contexts
            .get(&ctx)
            .map(|context| context.completed.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue {
        self.devices
            .get(&device)
            .map(|device| device.retired.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        self.device_wait_idle(device)?;
        if self.device_timeline_retired(device) < value {
            anyhow::bail!("WebGPU: timeline value {value} has not been submitted");
        }
        Ok(())
    }

    fn poll_signals(
        &mut self,
        ctx: ContextHandle,
        _progress: crate::timeline::TimelineValue,
    ) -> Vec<crate::signal::QueuedSignal> {
        self.contexts
            .get(&ctx)
            .map(|context| crate::signal::drain_all_queued_signals(&context.signal_queue))
            .unwrap_or_default()
    }

    fn submit_standalone(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        if let Some(sync) = sync {
            for epoch in sync
                .waits
                .iter()
                .chain(sync.cpu_waits.iter())
                .chain(sync.host_observed_waits.iter())
            {
                self.device_wait_until(self.context_device(ctx), epoch.value)?;
            }
            for write in &sync.deferred_host_writes {
                self.write_buffer(write.buffer, write.offset, &write.data)?;
            }
        }
        let effective = commands_with_sync_prologue(commands, sync);
        self.submit_commands(ctx, &effective)
    }

    fn begin_frame(&mut self, _surface: SurfaceHandle, _ctx: ContextHandle) -> Result<(FrameToken, TextureHandle)> {
        Self::unsupported("frames")
    }

    fn submit_frame(&mut self, _frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        Self::unsupported("frames")
    }

    fn create_compute_pipeline(
        &mut self,
        device: DeviceHandle,
        compute_shader: ShaderHandle,
        debug_name: Option<&str>,
    ) -> Result<ComputePipelineHandle> {
        let shader = self
            .shaders
            .get(&compute_shader)
            .context("WebGPU: invalid shader handle")?;
        if shader.device != device {
            anyhow::bail!("WebGPU: shader belongs to another device");
        }
        let (wgsl, slot_access) = self.compile_compute_wgsl(shader)?;
        let gpu = self.device(device)?;
        let error_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: debug_name,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: debug_name,
            layout: None,
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            anyhow::bail!("WebGPU shader/pipeline validation failed: {error}");
        }
        let handle = self.next_compute_pipeline;
        self.next_compute_pipeline += 1;
        self.compute_pipelines.insert(
            handle,
            WebGpuComputePipeline {
                device,
                pipeline,
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
            .map(|pipeline| pipeline.slot_access.clone())
            .unwrap_or_default()
    }

    fn max_bindless_slots_per_category(&self, device: DeviceHandle, category: crate::types::ResourceCategory) -> u32 {
        if !matches!(
            category,
            crate::types::ResourceCategory::Scattered | crate::types::ResourceCategory::Broadcast
        ) {
            return 0;
        }
        self.devices
            .get(&device)
            .map(|device| device.device.limits().max_storage_buffers_per_shader_stage)
            .unwrap_or(0)
    }

    fn available_bindless_slots(&self, device: DeviceHandle, category: crate::types::ResourceCategory) -> u32 {
        self.max_bindless_slots_per_category(device, category).saturating_sub(
            self.buffers
                .values()
                .filter(|buffer| buffer.device == device && buffer.slot.is_some())
                .count() as u32,
        )
    }

    fn max_submission_contexts(&self, _device: DeviceHandle) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOUBLE_WGSL: &str = r#"// @goldy-wgsl
@group(0) @binding(0)
var<storage, read_write> values: array<u32>;

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    values[id.x] = values[id.x] * 2u;
}
"#;

    const DOUBLE_SLANG: &str = r#"
RWStructuredBuffer<uint> values : register(u0);

[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
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

    fn run_compute_dispatch_and_readback(shader_source: &str) -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU compute test: {error:#}");
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
        assert_eq!(backend.gpu_progress(ctx), submitted);

        let readback = backend.alloc_readback_buffer(device, 16)?;
        backend.submit_standalone(
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
        let mut bytes = [0u8; 16];
        backend.read_readback_buffer(readback, &mut bytes)?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn wgsl_compute_dispatch_and_readback() -> Result<()> {
        run_compute_dispatch_and_readback(DOUBLE_WGSL)
    }

    #[test]
    fn slang_compute_dispatch_and_readback() -> Result<()> {
        run_compute_dispatch_and_readback(DOUBLE_SLANG)
    }

    fn run_scheme_compute_and_withdraw(shader_source: &str) -> Result<()> {
        let backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU scheme test: {error:#}");
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
    fn scheme_dispatches_slang_compute_and_withdraws() -> Result<()> {
        run_scheme_compute_and_withdraw(DOUBLE_SLANG)
    }

    #[test]
    fn scheme_dispatches_goldy_virtual_compute_and_withdraws() -> Result<()> {
        run_scheme_compute_and_withdraw(DOUBLE_GOLDY_SLANG)
    }

    #[test]
    fn scheme_binds_two_goldy_buffers_in_parameter_order() -> Result<()> {
        let backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU scheme test: {error:#}");
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

    #[test]
    fn slang_emits_wgsl_for_compute() -> Result<()> {
        let compiler = crate::slang::SlangCompiler::new()?;
        let compiled = compiler.compile_bindless_with_reflection(
            DOUBLE_SLANG,
            crate::slang::ShaderTarget::Wgsl,
            &[("cs_main", crate::slang::SlangStage::Compute)],
            &[],
        )?;
        let wgsl = compiled.shader.as_str().context("expected text WGSL")?;
        assert!(
            wgsl.contains("@compute"),
            "Slang output did not contain a WGSL compute entry:\n{wgsl}"
        );
        assert!(
            wgsl.contains("@binding(0)"),
            "Slang output did not preserve the storage binding:\n{wgsl}"
        );
        Ok(())
    }
}
