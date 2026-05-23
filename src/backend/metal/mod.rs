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

mod buffer;
mod compute;
mod device;
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
use anyhow::{Context, Result};
use types::MetalState;

/// Returns `true` when each device's GPU timeline has caught up to all scheduled work.
pub(in crate::backend::metal) fn gpu_is_idle(state: &MetalState) -> bool {
    state.devices.values().all(|ld| {
        ld.timeline_scheduled_max == 0
            || ld.timeline_event.as_ref().signaled_value() >= ld.timeline_scheduled_max
    })
}

/// Move every entry in each device's pending-slot list to its free list.
///
/// Called after [`GpuBackend::wait_until`] has confirmed GPU completion so slots
/// parked pending while work was in flight can be recycled.
pub(in crate::backend::metal) fn drain_all_pending_slots(state: &mut MetalState) {
    for device in state.devices.values_mut() {
        device.resource_registry.drain_pending_slots();
    }
}

/// Block until scheduled timeline values have been signaled on every device, or timeout.
pub(in crate::backend::metal) fn wait_all_in_flight(state: &MetalState) -> Result<()> {
    use std::sync::atomic::Ordering;
    if state.device_lost.load(Ordering::Relaxed) {
        anyhow::bail!("GPU device is lost; refusing to wait for in-flight work");
    }
    let timeout = std::time::Duration::from_millis(5000);
    for ld in state.devices.values() {
        let target = ld.timeline_scheduled_max;
        if target == 0 {
            continue;
        }
        if !ld.timeline_waiter.wait_until(target, timeout) {
            state.device_lost.store(true, Ordering::Relaxed);
            anyhow::bail!(
                "GPU wait_all_in_flight timed out after {}ms",
                timeout.as_millis()
            );
        }
    }
    Ok(())
}

/// Metal backend for macOS.
pub struct MetalBackend {
    state: MetalState,
}

impl MetalBackend {
    /// Create a new Metal backend.
    pub fn new() -> Result<Self> {
        let _span = goldy_span!("backend.metal.init").entered();
        tracing::info!("Initializing Metal backend");

        // Runtime Metal shader validation reads `MTL_SHADER_VALIDATION` before the first
        // device is created when GPU API validation is on (`GOLDY_VALIDATION=1`, `api`, `all`, …).
        if crate::backend::goldy_validation_enabled()
            && std::env::var_os("MTL_SHADER_VALIDATION").is_none()
        {
            std::env::set_var("MTL_SHADER_VALIDATION", "1");
            tracing::info!("Set MTL_SHADER_VALIDATION=1 (GOLDY_VALIDATION); was unset");
        }

        let slang_compiler =
            crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?;

        goldy_event!("backend.metal.init", success = true);

        Ok(Self {
            state: MetalState {
                device_lost: std::sync::atomic::AtomicBool::new(false),
                devices: std::collections::HashMap::new(),
                next_device_handle: 1,
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

impl GpuBackend for MetalBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Metal
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        device::enumerate()
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        device::create(&mut self.state, adapter_id)
    }

    fn destroy_device(&mut self, device: DeviceHandle) {
        device::destroy(&mut self.state, device);
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        device::is_valid(&self.state, device)
    }

    fn is_device_lost(&self, _device: DeviceHandle) -> bool {
        self.state
            .device_lost
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn create_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        access: DataAccess,
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
        access: DataAccess,
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

    fn device_capabilities(&self, device: DeviceHandle) -> crate::device::DeviceCapabilities {
        let _ = device;
        crate::device::DeviceCapabilities {
            has_zero_copy_storage_readback: true,
            buffer_resize_cost: crate::types::BufferResizeCost::Constant,
            buffer_page_size: 16 * 1024,
            buffer_decommit_supported: true,
            ..crate::device::DeviceCapabilities::default()
        }
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

    fn read_buffer_to_cpu(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        output: &mut [u8],
    ) -> Result<()> {
        buffer::read_to_cpu(&self.state, device, buffer, output)
    }

    fn clear_buffer(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<()> {
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
        self.create_shader_with_checks(
            device,
            slang_source,
            search_paths,
            defines,
            optimization_level,
            vec![],
        )
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
            crate::backend::shared::ShaderDesc::new(
                device,
                slang_source,
                search_paths,
                defines,
                optimization_level,
            )
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
        let raster =
            crate::backend::shared::PipelineDesc::new(vertex_layout, topology, target_format);
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
        let raster =
            crate::backend::shared::PipelineDesc::new(vertex_layout, topology, target_format)
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
        render_target::create_with_depth(
            &mut self.state,
            device,
            width,
            height,
            color_format,
            depth_format,
        )
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
        access: SpatialAccess,
        flags: TextureFlags,
    ) -> Result<TextureHandle> {
        texture::create(
            &mut self.state,
            device,
            width,
            height,
            format,
            access,
            flags,
        )
    }

    fn write_texture(
        &mut self,
        texture: TextureHandle,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        texture::write(&self.state, texture, data, width, height)
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
        texture::write_region(&self.state, texture, x, y, width, height, data)
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

    fn create_sampler(
        &mut self,
        device: DeviceHandle,
        desc: &SamplerDesc,
    ) -> Result<SamplerHandle> {
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

    fn begin_frame(&mut self, surface: SurfaceHandle) -> Result<(FrameToken, TextureHandle)> {
        let image = surface::acquire(&mut self.state, surface)?;
        let tex = surface::frame_texture(&self.state, surface)
            .context("begin_frame: surface frame texture unavailable")?;
        Ok((FrameToken { surface, image }, tex))
    }

    fn record_render(&mut self, frame: &FrameToken, commands: &[RenderCommand]) -> Result<()> {
        surface::render(&mut self.state, frame.surface, frame.image, commands)
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

    fn surface_set_present_mode(
        &mut self,
        surface: SurfaceHandle,
        mode: crate::types::PresentMode,
    ) -> Result<()> {
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

    fn gpu_progress(&self, device: DeviceHandle) -> crate::timeline::TimelineValue {
        self.state
            .devices
            .get(&device)
            .map(|ld| ld.timeline_event.as_ref().signaled_value())
            .unwrap_or(0)
    }

    fn wait_until(
        &mut self,
        device: DeviceHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<()> {
        use std::sync::atomic::Ordering;

        if self.state.device_lost.load(Ordering::Relaxed) {
            anyhow::bail!("Metal device lost");
        }

        let waiter = self
            .state
            .devices
            .get(&device)
            .context("Invalid device handle")?
            .timeline_waiter
            .clone();

        let timeout = std::time::Duration::from_secs(300);
        if !waiter.wait_until(value, timeout) {
            if self.state.device_lost.load(Ordering::Relaxed) {
                anyhow::bail!("Metal device lost");
            }
            anyhow::bail!("wait_until exceeded 300s");
        }

        if let Some(ld) = self.state.devices.get_mut(&device) {
            ld.process_deletion_queue_up_to_signaled();
        }
        drain_all_pending_slots(&mut self.state);
        Ok(())
    }

    fn wait_until_timeout(
        &mut self,
        device: DeviceHandle,
        value: crate::timeline::TimelineValue,
        timeout_ms: u32,
    ) -> Result<bool> {
        use std::sync::atomic::Ordering;

        if self.state.device_lost.load(Ordering::Relaxed) {
            anyhow::bail!("Metal device lost");
        }

        let waiter = self
            .state
            .devices
            .get(&device)
            .context("Invalid device handle")?
            .timeline_waiter
            .clone();

        let timeout = std::time::Duration::from_millis(u64::from(timeout_ms));
        if !waiter.wait_until(value, timeout) {
            if self.state.device_lost.load(Ordering::Relaxed) {
                anyhow::bail!("Metal device lost");
            }
            return Ok(false);
        }

        if let Some(ld) = self.state.devices.get_mut(&device) {
            ld.process_deletion_queue_up_to_signaled();
        }
        drain_all_pending_slots(&mut self.state);
        Ok(true)
    }

    fn submit_standalone(
        &mut self,
        device: DeviceHandle,
        commands: &[GpuCommand],
    ) -> Result<crate::timeline::TimelineValue> {
        compute::submit(&mut self.state, device, commands)
    }

    fn submit_graph(
        &mut self,
        device: DeviceHandle,
        commands: &[GraphCommand],
    ) -> Result<crate::timeline::TimelineValue> {
        compute::submit_graph(&mut self.state, device, commands)
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

    fn end_frame(&mut self, frame: FrameToken) -> Result<crate::timeline::TimelineValue> {
        surface::end_frame(&mut self.state, frame)
    }

    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        compute::destroy(&mut self.state, pipeline);
    }

    fn available_bindless_slots(
        &self,
        device: DeviceHandle,
        category: crate::types::BindlessCategory,
    ) -> u32 {
        self.state
            .devices
            .get(&device)
            .map(|ld| ld.resource_registry.available_slots(category))
            .unwrap_or(0)
    }

    fn max_bindless_slots_per_category(
        &self,
        _device: DeviceHandle,
        _category: crate::types::BindlessCategory,
    ) -> u32 {
        types::MAX_RESOURCES_PER_CATEGORY
    }

    fn flush_deferred_deletions(&mut self, device: DeviceHandle) {
        if let Some(ld) = self.state.devices.get_mut(&device) {
            ld.process_deletion_queue_up_to_signaled();
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
        if let Some(logical_device) = self.state.devices.get_mut(&device) {
            logical_device.heap_allocator.reset_for_frame();
        }
    }

    fn ensure_buffer_heap_capacity(&mut self, device: DeviceHandle, min_capacity: u64) {
        if let Some(logical_device) = self.state.devices.get_mut(&device) {
            logical_device
                .heap_allocator
                .ensure_primary_capacity(min_capacity);
        }
    }

    fn compact_overflow_heaps(&mut self, device: DeviceHandle) {
        if let Some(logical_device) = self.state.devices.get_mut(&device) {
            logical_device.heap_allocator.compact_overflow();
            logical_device.texture_heap.compact_overflow();
        }
    }

    fn release_idle_shader_compiler(&mut self) {
        self.state.slang_compiler = None;
        tracing::info!("Released Metal Slang compiler session (freed host-side compiler memory)");
    }

    fn deferred_deletion_pending_count(&self, device: DeviceHandle) -> usize {
        self.state
            .devices
            .get(&device)
            .map(|d| d.deletion_queue.pending_len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_backend_creation() {
        let backend = MetalBackend::new();
        assert!(
            backend.is_ok(),
            "Failed to create Metal backend: {:?}",
            backend.err()
        );
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
        assert!(
            device.is_ok(),
            "Failed to create Metal device: {:?}",
            device.err()
        );
        let device = device.unwrap();
        assert!(backend.is_device_valid(device));
        backend.destroy_device(device);
        assert!(!backend.is_device_valid(device));
    }

    #[test]
    fn test_metal_buffer_operations() {
        let mut backend = MetalBackend::new().unwrap();
        let device = backend.create_device(0).unwrap();

        let buffer = backend
            .create_buffer(
                device,
                256,
                DataAccess::Scattered,
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
