//! Metal backend implementation for macOS.
//!
//! Targets Metal 3.0+ on macOS 13+.
//! Uses Slang for shader compilation (Slang -> MSL).
//!
//! ## Module Structure
//!
//! Domain modules mirror Vulkan/DX12 for cross-backend navigability:
//! - `device`, `buffer`, `shader`, `pipeline`, `render_target`, `render_commands`
//! - `texture`, `sampler`, `surface`, `compute`
//! - `types`: Internal state structs
//! - `utils`: Format conversion and helpers

pub(super) mod api_log;
mod buffer;
mod compute;
mod context;
mod device;
mod frame_table;
mod pending_submit;
mod pipeline;
mod render_commands;
mod render_target;
mod sampler;
mod shader;
pub(super) mod staging;
mod surface;
mod texture;
mod types;
mod utils;

use super::*;
use crate::{goldy_event, goldy_span};
use ::metal as mtl;
use anyhow::{Context, Result};
use types::MetalState;

/// Returns `true` when each device's GPU timeline has caught up to all scheduled work.
pub(in crate::backend::metal) fn gpu_is_idle(state: &MetalState) -> bool {
    state.devices.iter().all(|(device, ld)| {
        ld.timeline_scheduled_max.load(std::sync::atomic::Ordering::Relaxed) == 0
            || context::device_retired(state, *device)
                >= ld.timeline_scheduled_max.load(std::sync::atomic::Ordering::Relaxed)
    })
}

/// Move every entry in each device's pending-slot list to its free list.
///
/// Called after [`GpuBackend::wait_until`] has confirmed GPU completion so slots
/// parked pending while work was in flight can be recycled.
pub(in crate::backend::metal) fn drain_all_pending_slots(state: &mut MetalState) {
    for device in state.devices.values() {
        device
            .descriptors
            .lock()
            .unwrap()
            .resource_registry
            .drain_pending_slots();
    }
}

/// Drop all entries from the front of a context's in-flight CB deque whose timeline
/// value is <= the current retirement horizon.  Safe to call at any time.
pub(in crate::backend::metal) fn drain_completed_cbs(sc: &mut types::MetalSubmissionContext) {
    let progress = context::context_gpu_progress(sc);
    while sc
        .in_flight_command_buffers
        .front()
        .is_some_and(|(tv, _)| *tv <= progress)
    {
        sc.in_flight_command_buffers.pop_front();
    }
}

/// Block until scheduled timeline values have been signaled on one device, or timeout.
pub(in crate::backend::metal) fn wait_device_idle(state: &MetalState, device: DeviceHandle) -> Result<()> {
    use std::sync::atomic::Ordering;
    if state.device_lost.load(Ordering::Relaxed) {
        anyhow::bail!("GPU device is lost; refusing to wait for in-flight work");
    }
    let ld = state
        .devices
        .get(&device)
        .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?;
    let target = ld.timeline_scheduled_max.load(std::sync::atomic::Ordering::Relaxed);
    if target == 0 {
        return Ok(());
    }
    ld.submission_worker
        .wait_submitted_if_scheduled(target, target)?;
    ld.submission_worker.check_error()?;
    const IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(5000);
    let reached = context::wait_until_device_seq_at_least(state, device, target, IDLE_TIMEOUT);
    if !reached {
        state.device_lost.store(true, Ordering::Relaxed);
        anyhow::bail!(
            "GPU wait_device_idle timed out after {}ms waiting for timeline {target}",
            IDLE_TIMEOUT.as_millis()
        );
    }
    Ok(())
}

/// Block until scheduled timeline values have been signaled on every device, or timeout.
pub(in crate::backend::metal) fn wait_all_in_flight(state: &MetalState) -> Result<()> {
    use std::sync::atomic::Ordering;
    if state.device_lost.load(Ordering::Relaxed) {
        anyhow::bail!("GPU device is lost; refusing to wait for in-flight work");
    }
    for device in state.devices.keys().copied().collect::<Vec<_>>() {
        wait_device_idle(state, device)?;
    }
    Ok(())
}

static METAL_VALIDATION_INIT: std::sync::Once = std::sync::Once::new();

/// Metal backend for macOS.
pub struct MetalBackend {
    state: MetalState,
}

impl MetalBackend {
    /// Create a new Metal backend.
    pub fn new() -> Result<Self> {
        let _span = goldy_span!("backend.metal.init").entered();
        tracing::info!("Initializing Metal backend");

        // Initialise API call logger (GOLDY_API_LOG) as early as possible so
        // even device-creation calls can be captured if desired.
        api_log::init();

        // `MTL_SHADER_VALIDATION` must be set before the first MTLDevice is created.
        // Use a process-wide Once so parallel test threads do not race on `setenv`.
        METAL_VALIDATION_INIT.call_once(|| {
            if crate::backend::goldy_validation_enabled() && std::env::var_os("MTL_SHADER_VALIDATION").is_none() {
                // SAFETY: called exactly once per process, before `Device::all()` below.
                unsafe { std::env::set_var("MTL_SHADER_VALIDATION", "1") };
                tracing::info!("Set MTL_SHADER_VALIDATION=1 (GOLDY_VALIDATION api)");
            }
        });

        let slang_compiler = crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?;

        let adapters: Vec<types::MetalAdapterInfo> = ::metal::Device::all()
            .into_iter()
            .enumerate()
            .map(|(idx, device)| types::MetalAdapterInfo {
                device,
                adapter_id: idx as u32,
            })
            .collect();

        goldy_event!("backend.metal.init", success = true);

        Ok(Self {
            state: MetalState {
                adapters,
                device_lost: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                devices: std::collections::HashMap::new(),
                next_device_handle: 1,
                contexts: std::collections::HashMap::new(),
                next_context_id: 1,
                buffers: std::collections::HashMap::new(),
                next_buffer_handle: 1,
                shaders: std::collections::HashMap::new(),
                next_shader_handle: 1,
                pipelines: std::collections::HashMap::new(),
                next_pipeline_handle: 1,
                compute_pipelines: std::collections::HashMap::new(),
                next_compute_pipeline_handle: 1,
                render_targets: std::collections::HashMap::new(),
                next_render_target_handle: 1,
                surfaces: std::collections::HashMap::new(),
                next_surface_handle: 1,
                textures: std::collections::HashMap::new(),
                next_texture_handle: 1,
                samplers: std::collections::HashMap::new(),
                next_sampler_handle: 1,
                slang_compiler: Some(slang_compiler),
            },
        })
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        tracing::info!("Shutting down Metal backend");
        let device_handles: Vec<_> = self.state.devices.keys().copied().collect();
        for handle in device_handles {
            device::destroy(&mut self.state, handle);
        }
    }
}

impl crate::backend::GpuBackendTimelineWait for MetalBackend {
    fn take_timeline_blocking_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn crate::backend::TimelineBlockingWait>>> {
        use std::sync::atomic::Ordering;

        if self.state.device_lost.load(Ordering::Relaxed) {
            anyhow::bail!("Metal device lost");
        }

        if self.gpu_progress(ctx) >= value {
            return Ok(None);
        }

        let cb_to_wait = self.state.contexts.get(&ctx).and_then(|sc_arc| {
            let sc = sc_arc.lock().unwrap();
            sc.in_flight_command_buffers
                .iter()
                .find(|(tv, _)| *tv >= value)
                .map(|(_, cb)| cb.to_owned())
        });

        if let Some(cb) = cb_to_wait {
            return Ok(Some(Box::new(MetalCommandBufferBlockingWait { cb })));
        }

        let waiter = self
            .state
            .contexts
            .get(&ctx)
            .context("Invalid context handle")?
            .lock()
            .unwrap()
            .timeline_waiter
            .clone();
        Ok(Some(Box::new(MetalWaiterBlockingWait {
            waiter,
            value,
            device_lost: std::sync::Arc::clone(&self.state.device_lost),
        })))
    }

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        use std::sync::atomic::Ordering;
        let device = self.context_device(ctx);
        let _dz = crate::tracy_zone!("mtl.wait_until.deletion_queue");
        let retired = context::device_retired(&self.state, device);
        if let Some(sc_arc) = self.state.contexts.get(&ctx) {
            let mut sc = sc_arc.lock().unwrap();
            drain_completed_cbs(&mut sc);
            sc.deletion_queue.process_up_to(value);
        }
        if let Some(ld) = self.state.devices.get(&device) {
            ld.process_deletion_queue_up_to(value.min(retired));
        }
        {
            let _pz = crate::tracy_zone!("mtl.wait_until.drain_pending_slots");
            drain_all_pending_slots(&mut self.state);
        }
        if self.state.device_lost.load(Ordering::Relaxed) {
            anyhow::bail!("Metal device lost");
        }
        Ok(())
    }
}

impl crate::backend::GpuBackendPresentSplit for MetalBackend {
    fn take_present_gpu_work(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn crate::backend::PresentGpuWork>> {
        surface::prepare_present_work(&mut self.state, frame, submit_tv)
    }

    fn finish_present(
        &mut self,
        finish: crate::backend::PresentFinishState,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        surface::finish_present(&mut self.state, finish, submit_tv)
    }
}

impl GpuBackend for MetalBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Metal
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        device::enumerate(&self.state.adapters)
    }

    fn adapter_capabilities(&self, adapter_id: u32) -> crate::device::DeviceCapabilities {
        device::adapter_capabilities(adapter_id)
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        device::create(&mut self.state, adapter_id)
    }

    fn destroy_device(&mut self, device: DeviceHandle) {
        let ctxs: Vec<_> = self
            .state
            .contexts
            .iter()
            .filter(|(_, sc_arc)| sc_arc.lock().unwrap().device == device)
            .map(|(id, _)| *id)
            .collect();
        for ctx in ctxs {
            context::destroy(&mut self.state, ctx);
        }
        device::destroy(&mut self.state, device);
    }

    fn device_wait_idle(&mut self, device: DeviceHandle) -> Result<()> {
        wait_device_idle(&self.state, device)
    }

    fn create_context(&mut self, device: DeviceHandle) -> Result<ContextHandle> {
        context::create(&mut self.state, device)
    }

    fn destroy_context(&mut self, ctx: ContextHandle) {
        context::destroy(&mut self.state, ctx);
    }

    fn clone_context_timeline_reader(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn crate::backend::ContextTimelineReader>> {
        Some(std::sync::Arc::new(MetalContextTimelineReader {
            sc: std::sync::Arc::clone(self.state.contexts.get(&ctx)?),
        }))
    }

    fn clone_device_timeline_reader(
        &self,
        device: DeviceHandle,
    ) -> Option<std::sync::Arc<dyn crate::backend::DeviceTimelineReader>> {
        Some(std::sync::Arc::new(MetalDeviceTimelineReader {
            ld: std::sync::Arc::clone(self.state.devices.get(&device)?),
        }))
    }

    fn clone_context_deletion_flush(
        &self,
        ctx: ContextHandle,
        context_readers: std::sync::Arc<
            std::sync::Mutex<
                std::collections::HashMap<ContextHandle, std::sync::Arc<dyn crate::backend::ContextTimelineReader>>,
            >,
        >,
    ) -> Option<std::sync::Arc<dyn crate::backend::ContextDeferredDeletionFlush>> {
        let sc = std::sync::Arc::clone(self.state.contexts.get(&ctx)?);
        let device_handle = {
            let sc_guard = sc.lock().unwrap();
            sc_guard.device
        };
        Some(std::sync::Arc::new(MetalContextDeferredDeletionFlush {
            sc,
            ld: std::sync::Arc::clone(self.state.devices.get(&device_handle)?),
            timeline: std::sync::Arc::clone(context_readers.lock().unwrap().get(&ctx)?),
        }))
    }

    fn clone_context_reclamation_scope(
        &self,
        ctx: ContextHandle,
    ) -> std::sync::Arc<dyn crate::backend::ContextReclamationScope> {
        if let Some(sc) = self.state.contexts.get(&ctx) {
            return std::sync::Arc::new(MetalContextReclamationScope {
                sc: std::sync::Arc::clone(sc),
            });
        }
        std::sync::Arc::new(crate::backend::NoOpReclamationScope)
    }

    fn context_device(&self, ctx: ContextHandle) -> DeviceHandle {
        context::context_device(&self.state, ctx)
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        device::is_valid(&self.state, device)
    }

    fn is_device_lost(&self, _device: DeviceHandle) -> bool {
        self.state.device_lost.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn create_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
    ) -> Result<BufferHandle> {
        buffer::create(&mut self.state, device, size, access, element_stride, flags)
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) {
        buffer::destroy(&mut self.state, buffer);
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        buffer::write(&self.state, buffer, offset, data)
    }

    fn buffer_size(&self, buffer: BufferHandle) -> u64 {
        buffer::size(&self.state, buffer)
    }

    fn buffer_capacity(&self, buffer: BufferHandle) -> u64 {
        buffer::buffer_capacity(&self.state, buffer)
    }

    fn create_buffer_with_capacity(
        &mut self,
        device: DeviceHandle,
        initial_size: u64,
        capacity: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        buffer::create_with_capacity(
            &mut self.state,
            device,
            initial_size,
            capacity,
            access,
            element_stride,
            flags,
        )
    }

    fn set_buffer_logical_size(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        new_logical_size: u64,
    ) -> Result<()> {
        buffer::set_logical_size(&mut self.state, device, buffer, new_logical_size)
    }

    fn hint_buffer_unused_above(&mut self, buffer: BufferHandle, offset: u64) {
        buffer::hint_unused_above(&mut self.state, buffer, offset);
    }

    fn device_capabilities(&self, _device: DeviceHandle) -> crate::device::DeviceCapabilities {
        device::adapter_capabilities(0)
    }

    fn buffer_bindless_index(&self, buffer: BufferHandle) -> Option<u32> {
        buffer::bindless_index(&self.state, buffer)
    }

    fn buffer_bindless_srv_index(&self, buffer: BufferHandle) -> Option<u32> {
        // Metal uses the same argument buffer slot for both StructuredBuffer and RWStructuredBuffer
        buffer::bindless_index(&self.state, buffer)
    }

    fn create_buffer_view(
        &mut self,
        parent: BufferHandle,
        offset: u64,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        buffer::create_view(&mut self.state, parent, offset, size, element_stride)
    }

    fn resize_buffer(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        new_size: u64,
        preserve_contents: bool,
    ) -> Result<()> {
        buffer::resize(&mut self.state, device, buffer, new_size, preserve_contents)
    }

    fn read_buffer_to_cpu(&mut self, device: DeviceHandle, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        buffer::read_to_cpu(&self.state, device, buffer, output)
    }

    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle> {
        buffer::alloc_readback_buffer(&mut self.state, device, size)
    }

    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        buffer::read_readback_buffer(&self.state, buffer, output)
    }

    fn free_readback_buffer(&mut self, buffer: BufferHandle) {
        buffer::destroy(&mut self.state, buffer);
    }

    fn query_texture_copy_footprint(
        &self,
        _device: DeviceHandle,
        width: u32,
        height: u32,
        format: crate::types::TextureFormat,
    ) -> Result<crate::backend::TextureCopyFootprint> {
        Ok(buffer::query_texture_copy_footprint(width, height, format))
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
        buffer::alloc_texture_readback_staging(&mut self.state, device, layout)
    }

    fn read_texture_readback_staging(
        &self,
        buffer: BufferHandle,
        layout: crate::backend::TextureCopyFootprint,
        output: &mut [u8],
    ) -> Result<()> {
        buffer::read_texture_readback_staging(&self.state, buffer, layout, output)
    }

    fn clear_buffer(&mut self, device: DeviceHandle, buffer: BufferHandle, offset: u64, size: u64) -> Result<()> {
        buffer::clear(&self.state, device, buffer, offset, size)
    }

    fn create_shader_with_paths(
        &mut self,
        device: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
    ) -> Result<ShaderHandle> {
        self.create_shader_with_checks(device, slang_source, search_paths, defines, optimization_level, vec![])
    }

    fn create_shader_with_checks(
        &mut self,
        device: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
        layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
    ) -> Result<ShaderHandle> {
        shader::create(
            &self.state.devices,
            &mut self.state.shaders,
            &mut self.state.next_shader_handle,
            crate::backend::shared::ShaderDesc::new(device, slang_source, search_paths, defines, optimization_level)
                .with_layout_checks(layout_checks),
        )
    }

    fn destroy_shader(&mut self, shader: ShaderHandle) {
        shader::destroy(&self.state.devices, &mut self.state.shaders, shader);
    }

    fn create_pipeline(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        let raster = crate::backend::shared::PipelineDesc::new(vertex_layout, topology, target_format);
        let desc = crate::backend::shared::GraphicsPipelineCreateDesc {
            device_handle: device,
            vertex_shader,
            fragment_shader,
            raster: &raster,
        };
        pipeline::create_with_depth(&mut self.state, &desc)
    }

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
        let raster = crate::backend::shared::PipelineDesc::new(vertex_layout, topology, target_format)
            .with_depth_stencil(depth_stencil);
        let desc = crate::backend::shared::GraphicsPipelineCreateDesc {
            device_handle: device,
            vertex_shader,
            fragment_shader,
            raster: &raster,
        };
        pipeline::create_with_depth(&mut self.state, &desc)
    }

    fn destroy_pipeline(&mut self, pipeline: PipelineHandle) {
        pipeline::destroy(&mut self.state, pipeline);
    }

    fn create_render_target(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<RenderTargetHandle> {
        render_target::create(&mut self.state, device, width, height, format)
    }

    fn create_render_target_with_depth(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        render_target::create_with_depth(&mut self.state, device, width, height, color_format, depth_format)
    }

    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        render_target::destroy(&mut self.state, target);
    }

    fn render_to_target(
        &mut self,
        device: DeviceHandle,
        target: RenderTargetHandle,
        commands: &[RenderCommand],
    ) -> Result<()> {
        render_target::render_to(&mut self.state, device, target, commands)
    }

    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()> {
        render_target::read_to_cpu(&self.state, target, output)
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
        texture::create(&mut self.state, device, width, height, format, access, flags)
    }

    fn write_texture(&mut self, texture: TextureHandle, data: &[u8], width: u32, height: u32) -> Result<()> {
        texture::write(&mut self.state, texture, data, width, height)
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
        texture::write_region(&mut self.state, texture, x, y, width, height, data)
    }

    fn destroy_texture(&mut self, texture: TextureHandle) {
        texture::destroy(&mut self.state, texture);
    }

    fn read_texture_to_cpu(&mut self, texture: TextureHandle, output: &mut [u8]) -> Result<()> {
        texture::read_to_cpu(&self.state, texture, output)
    }

    fn texture_bindless_index(&self, texture: TextureHandle) -> Option<u32> {
        texture::bindless_index(&self.state, texture)
    }

    fn texture_bindless_sampled_index(&self, texture: TextureHandle) -> Option<u32> {
        texture::bindless_sampled_index(&self.state, texture)
    }

    fn create_sampler(&mut self, device: DeviceHandle, desc: &SamplerDesc) -> Result<SamplerHandle> {
        sampler::create(&mut self.state, device, desc)
    }

    fn destroy_sampler(&mut self, sampler: SamplerHandle) {
        sampler::destroy(&mut self.state, sampler);
    }

    fn sampler_bindless_index(&self, sampler: SamplerHandle) -> Option<u32> {
        sampler::bindless_index(&self.state, sampler)
    }

    fn create_surface(
        &mut self,
        device: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        display: &dyn raw_window_handle::HasDisplayHandle,
        depth_format: Option<DepthFormat>,
    ) -> Result<SurfaceHandle> {
        surface::create(&mut self.state, device, window, display, depth_format)
    }

    fn destroy_surface(&mut self, surface: SurfaceHandle) {
        surface::destroy(&mut self.state, surface);
    }

    fn begin_frame(&mut self, surface: SurfaceHandle, ctx: ContextHandle) -> Result<(FrameToken, TextureHandle)> {
        let (image, present_slot) = surface::acquire(&mut self.state, surface, ctx)?;
        let tex =
            surface::frame_texture(&self.state, surface).context("begin_frame: surface frame texture unavailable")?;
        Ok((
            FrameToken {
                surface,
                image,
                context: ctx,
                // Metal bindless slot and returned image index both use current_frame.
                frame_slot: image as u32,
                present_slot,
            },
            tex,
        ))
    }

    fn cancel_frame(&mut self, frame: FrameToken) -> Result<()> {
        surface::cancel_frame(&mut self.state, frame)
    }

    fn take_surface_acquire_work(
        &mut self,
        surface: SurfaceHandle,
        ctx: ContextHandle,
    ) -> Result<Option<Box<dyn crate::backend::SurfaceAcquireWork>>> {
        Ok(Some(surface::take_surface_acquire_work(&mut self.state, surface, ctx)?))
    }

    fn finish_surface_acquire(
        &mut self,
        surface: SurfaceHandle,
        ctx: ContextHandle,
        drawable: crate::backend::SurfaceAcquireDrawable,
    ) -> Result<(FrameToken, TextureHandle)> {
        let (image, present_slot) =
            surface::finish_surface_acquire_from_drawable(&mut self.state, surface, ctx, drawable)?;
        let tex =
            surface::frame_texture(&self.state, surface).context("begin_frame: surface frame texture unavailable")?;
        Ok((
            FrameToken {
                surface,
                image,
                context: ctx,
                frame_slot: image as u32,
                present_slot,
            },
            tex,
        ))
    }

    fn clone_context_poll_reader(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn crate::backend::ContextPollReader>> {
        let sc = std::sync::Arc::clone(self.state.contexts.get(&ctx)?);
        let (signal_queue, pending_swapchain_returns) = {
            let sc_guard = sc.lock().unwrap();
            (
                std::sync::Arc::clone(&sc_guard.signal_queue),
                std::sync::Arc::clone(&sc_guard.pending_swapchain_returns),
            )
        };
        Some(std::sync::Arc::new(MetalContextPollReader {
            signal_queue,
            pending_swapchain_returns,
        }))
    }

    fn record_render(&mut self, frame: &FrameToken, commands: &[RenderCommand]) -> Result<()> {
        surface::render(
            &mut self.state,
            frame.surface,
            frame.image,
            frame.present_slot,
            commands,
        )
    }

    fn surface_resize(&mut self, surface: SurfaceHandle, width: u32, height: u32) -> Result<()> {
        surface::resize(&mut self.state, surface, width, height)
    }

    fn surface_size(&self, surface: SurfaceHandle) -> (u32, u32) {
        surface::size(&self.state, surface)
    }

    fn surface_format(&self, surface: SurfaceHandle) -> TextureFormat {
        surface::format(&self.state, surface)
    }

    fn surface_set_present_mode(&mut self, surface: SurfaceHandle, mode: crate::types::PresentMode) -> Result<()> {
        surface::set_present_mode(&mut self.state, surface, mode)
    }

    fn surface_present_mode(&self, surface: SurfaceHandle) -> crate::types::PresentMode {
        surface::present_mode(&self.state, surface)
    }

    fn create_compute_pipeline(
        &mut self,
        device: DeviceHandle,
        compute_shader: ShaderHandle,
    ) -> Result<ComputePipelineHandle> {
        compute::create(&mut self.state, device, compute_shader)
    }

    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue {
        self.state
            .contexts
            .get(&ctx)
            .map(|sc_arc| context::context_gpu_progress(&sc_arc.lock().unwrap()))
            .unwrap_or(0)
    }

    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue {
        context::device_retired(&self.state, device)
    }

    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> anyhow::Result<()> {
        let ld = self
            .state
            .devices
            .get(&device)
            .ok_or_else(|| anyhow::anyhow!("Invalid device handle"))?;
        ld.submission_worker.flush()?;
        let horizon = ld.timeline_scheduled_max.load(std::sync::atomic::Ordering::Acquire);
        ld.submission_worker
            .wait_submitted_if_scheduled(value, horizon)?;
        let timeout = std::time::Duration::from_secs(60);
        if context::wait_until_device_seq_at_least(&self.state, device, value, timeout) {
            Ok(())
        } else {
            anyhow::bail!("device_wait_until: timed out after 60 s waiting for timeline value {value}")
        }
    }

    fn poll_signals(
        &mut self,
        ctx: ContextHandle,
        _progress: crate::timeline::TimelineValue,
    ) -> Vec<crate::signal::Signal> {
        let sc_arc = match self.state.contexts.get(&ctx) {
            Some(sc) => sc.clone(),
            None => return Vec::new(),
        };
        let sc = sc_arc.lock().unwrap();
        let returns: Vec<types::PendingSwapchainReturn> =
            std::mem::take(&mut *sc.pending_swapchain_returns.lock().unwrap());
        drop(sc);
        apply_pending_swapchain_returns(&returns);
        let sc2 = sc_arc.lock().unwrap();
        crate::signal::drain_all_signals(&sc2.signal_queue)
    }

    fn peek_oldest_in_flight(&self, ctx: ContextHandle) -> Option<crate::timeline::TimelineValue> {
        let sc_arc = self.state.contexts.get(&ctx)?;
        let sc = sc_arc.lock().unwrap();
        let progress = context::context_gpu_progress(&sc);
        if progress < sc.last_submitted_seq {
            Some(progress.saturating_add(1))
        } else {
            None
        }
    }

    fn pending_acquire_count(&self, surface: SurfaceHandle) -> u32 {
        self.state
            .surfaces
            .get(&surface)
            .map(|s| s.pending_acquire_count.load(std::sync::atomic::Ordering::Acquire))
            .unwrap_or(0)
    }

    fn wait_until_timeout(
        &mut self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
        timeout_ms: u32,
    ) -> Result<bool> {
        use std::sync::atomic::Ordering;
        let device = self.context_device(ctx);

        if self.state.device_lost.load(Ordering::Relaxed) {
            anyhow::bail!("Metal device lost");
        }

        let waiter = self
            .state
            .contexts
            .get(&ctx)
            .context("Invalid context handle")?
            .lock()
            .unwrap()
            .timeline_waiter
            .clone();

        let timeout = std::time::Duration::from_millis(u64::from(timeout_ms));
        if !waiter.wait_until(value, timeout) {
            if self.state.device_lost.load(Ordering::Relaxed) {
                anyhow::bail!("Metal device lost");
            }
            return Ok(false);
        }

        let retired = context::device_retired(&self.state, device);
        if let Some(sc_arc) = self.state.contexts.get(&ctx) {
            sc_arc.lock().unwrap().deletion_queue.process_up_to(value);
        }
        if let Some(ld) = self.state.devices.get(&device) {
            ld.process_deletion_queue_up_to(value.min(retired));
        }
        drain_all_pending_slots(&mut self.state);
        Ok(true)
    }

    fn submit_standalone(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        compute::submit(&mut self.state, ctx, commands, sync)
    }

    fn submit_graph(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        compute::submit_graph(&mut self.state, ctx, commands, None, sync)
    }

    fn submit_graph_and_retain(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        compute::submit_graph_and_retain(&mut self.state, ctx, commands, key, sync)
    }

    fn try_resubmit_retained(
        &mut self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<crate::timeline::TimelineValue>> {
        compute::try_resubmit_retained(&mut self.state, ctx, key, sync)
    }

    fn retains_present_partitions(&self) -> bool {
        false
    }

    fn evict_retained(&mut self, ctx: ContextHandle, key: u64) {
        compute::evict_retained(&mut self.state, ctx, key);
    }

    fn record_gpu_work(&mut self, frame: &FrameToken, commands: &[GpuCommand]) -> Result<()> {
        let surf = self
            .state
            .surfaces
            .get_mut(&frame.surface)
            .context("Invalid surface handle")?;
        surf.frame_pending_gpu_commands.extend_from_slice(commands);
        Ok(())
    }

    fn submit_frame(&mut self, frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        surface::submit_frame(&mut self.state, frame)
    }

    fn present_frame(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        surface::present_frame(&mut self.state, frame, submit_tv)
    }

    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        compute::destroy(&mut self.state, pipeline);
    }

    fn available_bindless_slots(&self, device: DeviceHandle, category: crate::types::ResourceCategory) -> u32 {
        self.state
            .devices
            .get(&device)
            .map(|ld| {
                ld.descriptors
                    .lock()
                    .unwrap()
                    .resource_registry
                    .available_slots(category)
            })
            .unwrap_or(0)
    }

    fn max_bindless_slots_per_category(&self, _device: DeviceHandle, _category: crate::types::ResourceCategory) -> u32 {
        types::MAX_RESOURCES_PER_CATEGORY
    }

    fn flush_deferred_deletions(&mut self, ctx: ContextHandle) {
        let device = self.context_device(ctx);
        let retired = context::device_retired(&self.state, device);
        if let Some(sc_arc) = self.state.contexts.get(&ctx) {
            let mut sc = sc_arc.lock().unwrap();
            let ctx_signaled = sc.timeline_event.as_ref().signaled_value();
            sc.deletion_queue.process_up_to(ctx_signaled);
        }
        if let Some(ld) = self.state.devices.get(&device) {
            ld.process_deletion_queue_up_to(retired);
        }
    }

    fn set_reclamation_context(&mut self, ctx: ContextHandle, epoch: Option<crate::timeline::TimelineValue>) {
        if let Some(sc_arc) = self.state.contexts.get(&ctx) {
            sc_arc.lock().unwrap().reclamation_context = epoch.map(|epoch| (std::thread::current().id(), epoch));
        }
    }

    fn reset_buffer_heaps(&mut self, device: DeviceHandle) {
        // Safety: dropping the old primary heap and clearing overflow heaps
        // is only sound once every command buffer that allocated buffers from
        // them has finished. Wait on all outstanding fences (bounded; see
        // wait_all_in_flight) and drain deferred-release slots before
        // mutating the allocator. If the wait fails we log and skip the
        // reset — leaving memory slightly over-committed is far cheaper than
        // yanking a heap out from under a still-running dispatch.
        if let Err(e) = wait_all_in_flight(&self.state) {
            tracing::warn!("reset_buffer_heaps skipped: could not confirm GPU idle ({e})");
            return;
        }
        drain_all_pending_slots(&mut self.state);
        if let Some(logical_device) = self.state.devices.get(&device) {
            logical_device.heap_allocator.lock().unwrap().reset_for_frame();
        }
    }

    fn ensure_buffer_heap_capacity(&mut self, device: DeviceHandle, min_capacity: u64) {
        if let Some(logical_device) = self.state.devices.get(&device) {
            logical_device
                .heap_allocator
                .lock()
                .unwrap()
                .ensure_primary_capacity(min_capacity);
        }
    }

    fn compact_overflow_heaps(&mut self, device: DeviceHandle) {
        if let Some(logical_device) = self.state.devices.get(&device) {
            logical_device.heap_allocator.lock().unwrap().compact_overflow();
            logical_device.texture_heap.lock().unwrap().compact_overflow();
        }
    }

    fn release_idle_shader_compiler(&mut self) {
        self.state.slang_compiler = None;
        tracing::info!("Released Metal Slang compiler session (freed host-side compiler memory)");
    }

    fn deferred_deletion_pending_count(&self, ctx: ContextHandle) -> usize {
        let device = self.context_device(ctx);
        let ctx_count = self
            .state
            .contexts
            .get(&ctx)
            .map(|sc_arc| sc_arc.lock().unwrap().deletion_queue.pending_len())
            .unwrap_or(0);
        let device_count = self
            .state
            .devices
            .get(&device)
            .map(|d| d.deletion_queue.lock().unwrap().pending_len())
            .unwrap_or(0);
        ctx_count + device_count
    }

    fn buffer_heap_stats(&self, device: DeviceHandle) -> Option<super::BufferHeapStats> {
        self.state.devices.get(&device).map(|ld| {
            let ha = ld.heap_allocator.lock().unwrap();
            super::BufferHeapStats {
                buffer_count: ha.buffer_count(),
                overflow_count: ha.overflow_count(),
                high_water_bytes: ha.high_water_mark(),
                primary_heap_bytes: ha.primary_size(),
            }
        })
    }

    fn texture_heap_stats(&self, device: DeviceHandle) -> Option<super::TextureHeapStats> {
        self.state.devices.get(&device).map(|ld| {
            let th = ld.texture_heap.lock().unwrap();
            super::TextureHeapStats {
                texture_count: th.texture_count(),
                overflow_count: th.overflow_count(),
            }
        })
    }

    fn in_flight_command_buffer_count(&self, ctx: ContextHandle) -> usize {
        self.state
            .contexts
            .get(&ctx)
            .map(|sc_arc| sc_arc.lock().unwrap().in_flight_command_buffers.len())
            .unwrap_or(0)
    }
}

impl crate::backend::GpuBackendSubmitSession for MetalBackend {
    fn clone_context_submit_session(
        &self,
        _ctx: ContextHandle,
        backend: std::sync::Arc<std::sync::Mutex<Box<dyn crate::backend::GpuBackend>>>,
    ) -> std::sync::Arc<dyn crate::backend::ContextSubmitSession> {
        crate::backend::LockedSubmitSession::with_backend_type(backend, BackendType::Metal)
    }
}

struct MetalCommandBufferBlockingWait {
    cb: mtl::CommandBuffer,
}

impl crate::backend::TimelineBlockingWait for MetalCommandBufferBlockingWait {
    fn block(self: Box<Self>) -> Result<()> {
        let _wz = crate::tracy_zone!("mtl.wait_until.waitUntilCompleted");
        self.cb.wait_until_completed();
        Ok(())
    }

    fn block_timeout(self: Box<Self>, timeout_ms: u32) -> Result<bool> {
        use mtl::MTLCommandBufferStatus;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(u64::from(timeout_ms));
        loop {
            match self.cb.status() {
                MTLCommandBufferStatus::Completed => return Ok(true),
                MTLCommandBufferStatus::Error => anyhow::bail!("Metal command buffer failed"),
                _ if std::time::Instant::now() >= deadline => return Ok(false),
                _ => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
    }
}

struct MetalContextTimelineReader {
    sc: types::SharedMetalSubmissionContext,
}

fn apply_pending_swapchain_returns(returns: &[types::PendingSwapchainReturn]) {
    use std::sync::atomic::Ordering;
    for r in returns {
        r.pending_acquire_count.fetch_sub(1, Ordering::AcqRel);
    }
}

struct MetalContextPollReader {
    signal_queue: std::sync::Arc<crate::signal::SignalQueue>,
    pending_swapchain_returns: std::sync::Arc<std::sync::Mutex<Vec<types::PendingSwapchainReturn>>>,
}

impl crate::backend::ContextPollReader for MetalContextPollReader {
    fn poll_signals(&self, _progress: crate::timeline::TimelineValue) -> Vec<crate::signal::Signal> {
        let returns = std::mem::take(&mut *self.pending_swapchain_returns.lock().unwrap());
        apply_pending_swapchain_returns(&returns);
        crate::signal::drain_all_signals(&self.signal_queue)
    }
}

impl crate::backend::ContextTimelineReader for MetalContextTimelineReader {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue {
        let sc = self.sc.lock().unwrap();
        context::context_gpu_progress(&sc)
    }

    fn peek_oldest_in_flight(&self) -> Option<crate::timeline::TimelineValue> {
        let sc = self.sc.lock().unwrap();
        let progress = context::context_gpu_progress(&sc);
        if progress < sc.last_submitted_seq {
            Some(progress.saturating_add(1))
        } else {
            None
        }
    }
}

struct MetalWaiterBlockingWait {
    waiter: types::TimelineWaiter,
    value: crate::timeline::TimelineValue,
    device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl crate::backend::TimelineBlockingWait for MetalWaiterBlockingWait {
    fn block(self: Box<Self>) -> Result<()> {
        use std::sync::atomic::Ordering;
        let timeout = std::time::Duration::from_secs(300);
        let reached = {
            let _wz = crate::tracy_zone!("mtl.wait_until.condvar_fallback");
            self.waiter.wait_until(self.value, timeout)
        };
        if !reached {
            if self.device_lost.load(Ordering::Relaxed) {
                anyhow::bail!("Metal device lost");
            }
            anyhow::bail!("wait_until exceeded 300s");
        }
        Ok(())
    }

    fn block_timeout(self: Box<Self>, timeout_ms: u32) -> Result<bool> {
        use std::sync::atomic::Ordering;
        let timeout = std::time::Duration::from_millis(u64::from(timeout_ms));
        let reached = {
            let _wz = crate::tracy_zone!("mtl.wait_until.condvar_fallback");
            self.waiter.wait_until(self.value, timeout)
        };
        if !reached {
            if self.device_lost.load(Ordering::Relaxed) {
                anyhow::bail!("Metal device lost");
            }
            return Ok(false);
        }
        Ok(true)
    }
}

struct MetalDeviceTimelineReader {
    ld: types::SharedLogicalDevice,
}

impl crate::backend::DeviceTimelineReader for MetalDeviceTimelineReader {
    fn device_horizon(&self) -> crate::timeline::TimelineValue {
        use std::sync::atomic::Ordering;
        self.ld.retired_floor.load(Ordering::Relaxed)
    }
}

struct MetalContextDeferredDeletionFlush {
    sc: types::SharedMetalSubmissionContext,
    ld: types::SharedLogicalDevice,
    timeline: std::sync::Arc<dyn crate::backend::ContextTimelineReader>,
}

impl crate::backend::ContextDeferredDeletionFlush for MetalContextDeferredDeletionFlush {
    fn flush(&self, device_retired: crate::timeline::TimelineValue) {
        let ctx_signaled = self.timeline.gpu_progress();
        if let Ok(mut sc) = self.sc.lock() {
            sc.deletion_queue.process_up_to(ctx_signaled);
        }
        self.ld.process_deletion_queue_up_to(device_retired);
    }
}

struct MetalContextReclamationScope {
    sc: types::SharedMetalSubmissionContext,
}

impl crate::backend::ContextReclamationScope for MetalContextReclamationScope {
    fn set_epoch(&self, epoch: Option<crate::timeline::TimelineValue>) {
        if let Ok(mut sc) = self.sc.lock() {
            sc.reclamation_context = epoch.map(|epoch| (std::thread::current().id(), epoch));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_backend_creation() {
        let backend = MetalBackend::new();
        assert!(backend.is_ok(), "Failed to create Metal backend: {:?}", backend.err());
        let backend = backend.unwrap();
        assert_eq!(backend.backend_type(), BackendType::Metal);
    }

    #[test]
    fn test_metal_adapters() {
        let backend = MetalBackend::new().unwrap();
        let adapters = backend.enumerate_adapters();
        assert!(!adapters.is_empty(), "No Metal adapters found");
        for adapter in &adapters {
            println!("Adapter: {} ({})", adapter.name, adapter.vendor);
        }
    }

    #[test]
    fn test_metal_device_creation() {
        let mut backend = MetalBackend::new().unwrap();
        let device = backend.create_device(0);
        assert!(device.is_ok(), "Failed to create Metal device: {:?}", device.err());
        let device = device.unwrap();
        assert!(backend.is_device_valid(device));
        backend.destroy_device(device);
        assert!(!backend.is_device_valid(device));
    }

    /// Regression test for issue #162 (secondary concern): the sampler argument
    /// encoder's stride must match the 8-byte-per-slot assumption baked into
    /// `ARGUMENT_BUFFER_SIZE` and the sampler category layout.
    ///
    /// Before the fix, `sampler.rs` hardcoded `GPU_RESOURCE_ID_BYTES = 8` as the
    /// per-slot stride and wrote sampler GPU resource IDs via an unsafe pointer
    /// cast.  If Metal ever returned a different `encoded_length()` for a sampler
    /// argument encoder (e.g. on a future GPU family), the sampler slots would
    /// silently diverge from the buffer/texture slots, corrupting all bindless
    /// sampler reads.
    ///
    /// The fix:
    ///   1. A `sampler_encoder` (`MTLDataType::Sampler`) is created at device
    ///      construction and stored in `LogicalDevice`.
    ///   2. `sampler.rs` derives the stride from `sampler_encoder.encoded_length()`
    ///      and uses `set_argument_buffer` + `set_sampler_state` to write — the
    ///      same pattern as buffer and texture encoding.
    ///   3. Device creation asserts `encoded_length() == 8` and panics loudly if
    ///      the assumption is ever violated.
    ///
    /// This test verifies that a freshly-constructed device reports stride 8 for
    /// the sampler encoder, and that the stride exposed by `create_argument_encoders`
    /// matches the per-slot size implied by `ARGUMENT_BUFFER_SIZE`.
    #[test]
    fn test_sampler_encoder_stride() {
        use super::device::create_argument_encoders;
        use super::types::ARGUMENT_BUFFER_SIZE;
        use ::metal::Device as MTLDevice;

        let device = MTLDevice::system_default().expect("No Metal device available");
        let (buf_enc, tex_enc, si_enc, smp_enc) = create_argument_encoders(&device);

        // All four encoders must report the same 8-byte stride.
        assert_eq!(
            smp_enc.encoded_length(),
            8,
            "sampler encoder stride is {}, expected 8",
            smp_enc.encoded_length()
        );
        assert_eq!(
            buf_enc.encoded_length(),
            smp_enc.encoded_length(),
            "buffer and sampler encoder strides differ"
        );
        assert_eq!(
            tex_enc.encoded_length(),
            smp_enc.encoded_length(),
            "texture and sampler encoder strides differ"
        );
        assert_eq!(
            si_enc.encoded_length(),
            smp_enc.encoded_length(),
            "storage-image and sampler encoder strides differ"
        );

        // The stride must evenly divide ARGUMENT_BUFFER_SIZE so that all 5
        // resource categories fit without overflow.
        assert_eq!(
            ARGUMENT_BUFFER_SIZE % smp_enc.encoded_length(),
            0,
            "ARGUMENT_BUFFER_SIZE={ARGUMENT_BUFFER_SIZE} is not a multiple of \
             sampler stride={}",
            smp_enc.encoded_length()
        );
    }

    #[test]
    fn test_frame_table_reserves_storage_slots_zero_and_one() {
        let mut backend = MetalBackend::new().unwrap();
        let device = backend.create_device(0).unwrap();
        let buffer = backend
            .create_buffer(
                device,
                64,
                BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
            )
            .unwrap();
        assert_eq!(
            backend.buffer_bindless_index(buffer),
            Some(crate::frame_table::FRAME_TABLE_USER_SLOT_BASE),
            "first user scattered buffer must start at slot 2 (selector+table reserved)"
        );
        backend.destroy_buffer(buffer);
        backend.destroy_device(device);
    }

    #[test]
    fn test_metal_buffer_operations() {
        let mut backend = MetalBackend::new().unwrap();
        let device = backend.create_device(0).unwrap();

        let buffer = backend
            .create_buffer(
                device,
                256,
                BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
            )
            .unwrap();

        assert_eq!(backend.buffer_size(buffer), 256);

        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        backend.write_buffer(buffer, 0, &data).unwrap();

        backend.destroy_buffer(buffer);
        backend.destroy_device(device);
    }
}
