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

#![allow(deprecated)]

mod buffer;
mod compute;
mod device;
mod pipeline;
mod render_commands;
mod render_target;
mod sampler;
mod shader;
mod surface;
mod texture;
mod types;
mod utils;

use super::*;
use crate::{goldy_event, goldy_span};
use ::metal as mtl;
use anyhow::{Context, Result};
use types::MetalState;

/// Returns `true` when every command buffer we've submitted has reached
/// `MTLCommandBufferStatus::Completed`. Used by the destroy paths to decide
/// whether a just-released bindless slot can be recycled immediately (GPU
/// idle) or must park on the registry's pending list until the next
/// `wait_fence()` confirms completion.
///
/// ## Why this exists
///
/// Metal argument buffers are CPU-writable GPU memory: each descriptor is a
/// device pointer the shader dereferences at dispatch time. If the CPU
/// overwrites slot N's descriptor while any in-flight command buffer still
/// has a pending dispatch that will read slot N, the GPU reads the *new*
/// descriptor and ends up pointing at the wrong buffer. Observationally this
/// presents as:
/// - Random glitches in parts of the scene that encode late (e.g. the
///   stats/HUD overlay drawn after the main scene).
/// - `MTLCommandBufferError::Internal` when the shader dereferences a pointer
///   that isn't in the command buffer's residency set.
///
/// A conservative approximation — "if there's nothing in the fence pool, the
/// GPU is idle" — isn't quite enough because `submit_graph` (non-blocking)
/// leaves old fences in the pool until someone waits on them. Checking
/// `status() == Completed` on each entry gives us an accurate snapshot.
pub(in crate::backend::metal) fn gpu_is_idle(state: &MetalState) -> bool {
    let pool = state.compute_fence_pool.lock().unwrap();
    pool.values().all(|entry| {
        // Prefer the handler-published status to avoid a Metal call on the
        // fast (already-done) path; fall back to the live `status()` read if
        // the handler has not yet been dispatched.
        if let Some(status) = *entry.signal.done.lock().unwrap() {
            status == mtl::MTLCommandBufferStatus::Completed
        } else {
            entry.buffer.status() == mtl::MTLCommandBufferStatus::Completed
        }
    })
}

/// Move every entry in each device's pending-slot list to its free list.
///
/// Called by the compute module after a successful `wait_fence()`: at that
/// point we've established that all previously-submitted GPU work has
/// completed, so any slot that was parked pending (because it was released
/// while the GPU was still busy) is now safe to recycle.
pub(in crate::backend::metal) fn drain_all_pending_slots(state: &mut MetalState) {
    for device in state.devices.values_mut() {
        device.resource_registry.drain_pending_slots();
    }
}

/// Block until every command buffer in `compute_fence_pool` is `Completed`
/// (or `Error`), then remove the drained entries so the pool doesn't keep
/// growing across frames. Uses the same bounded-polling pattern as
/// [`compute::wait_fence`] so a wedged GPU is reported as a timeout rather
/// than hanging the caller forever.
///
/// Returns `Ok(())` if all in-flight work finished; bails if any command
/// buffer ended in `Error` status (so an upstream failure isn't silently
/// swallowed by a reset) or if the timeout budget is exhausted.
pub(in crate::backend::metal) fn wait_all_in_flight(state: &MetalState) -> Result<()> {
    use std::sync::atomic::Ordering;
    if state.device_lost.load(Ordering::Relaxed) {
        anyhow::bail!("GPU device is lost; refusing to wait for in-flight work");
    }
    // Snapshot (token, signal) pairs without holding the pool lock across a
    // blocking wait — the completion handler also locks the pool briefly
    // to drop itself, and we must not deadlock against it.
    let entries: Vec<(super::FenceToken, std::sync::Arc<types::FenceSignal>)> = {
        let pool = state.compute_fence_pool.lock().unwrap();
        pool.iter().map(|(t, e)| (*t, e.signal.clone())).collect()
    };
    if entries.is_empty() {
        return Ok(());
    }
    let timeout = std::time::Duration::from_millis(5000);
    let start = std::time::Instant::now();
    for (token, signal) in entries {
        let remaining = timeout.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            state.device_lost.store(true, Ordering::Relaxed);
            anyhow::bail!(
                "GPU wait_all_in_flight timed out after {}ms",
                timeout.as_millis()
            );
        }
        let status = {
            let mut guard = signal.done.lock().unwrap();
            loop {
                if let Some(status) = *guard {
                    break status;
                }
                let now_remaining = timeout.saturating_sub(start.elapsed());
                if now_remaining.is_zero() {
                    break mtl::MTLCommandBufferStatus::NotEnqueued;
                }
                let (g, result) = signal.cv.wait_timeout(guard, now_remaining).unwrap();
                guard = g;
                if result.timed_out() && guard.is_none() {
                    break mtl::MTLCommandBufferStatus::NotEnqueued;
                }
            }
        };
        match status {
            mtl::MTLCommandBufferStatus::Completed => {
                state.compute_fence_pool.lock().unwrap().remove(&token);
            }
            mtl::MTLCommandBufferStatus::Error => {
                state.compute_fence_pool.lock().unwrap().remove(&token);
                anyhow::bail!("Metal command buffer errored while waiting for idle");
            }
            _ => {
                state.device_lost.store(true, Ordering::Relaxed);
                anyhow::bail!(
                    "GPU wait_all_in_flight timed out after {}ms (status={:?})",
                    timeout.as_millis(),
                    status
                );
            }
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

        let slang_compiler =
            crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?;

        goldy_event!("backend.metal.init", success = true);

        Ok(Self {
            state: MetalState {
                compute_fence_pool: std::sync::Mutex::new(std::collections::HashMap::new()),
                next_compute_fence_token: std::sync::atomic::AtomicU64::new(1),
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
                slang_compiler,
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

#[allow(clippy::manual_find)]
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

    fn read_buffer_to_cpu(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        output: &mut [u8],
    ) -> Result<()> {
        buffer::read_to_cpu(&self.state, device, buffer, output)
    }

    fn read_buffer_coherent(
        &self,
        buffer: BufferHandle,
        offset: u64,
        output: &mut [u8],
    ) -> Result<()> {
        buffer::read_coherent(&self.state.buffers, buffer, offset, output)
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
            device,
            slang_source,
            search_paths,
            defines,
            optimization_level,
            layout_checks,
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
        pipeline::create_with_depth(
            &mut self.state,
            device,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
            None,
        )
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
        pipeline::create_with_depth(
            &mut self.state,
            device,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
            depth_stencil,
        )
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

    fn surface_acquire(&mut self, surface: SurfaceHandle) -> Result<SwapchainImageHandle> {
        surface::acquire(&mut self.state, surface)
    }

    fn surface_frame_texture(&self, surface: SurfaceHandle) -> Option<TextureHandle> {
        surface::frame_texture(&self.state, surface)
    }

    fn surface_render(
        &mut self,
        surface: SurfaceHandle,
        image: SwapchainImageHandle,
        commands: &[RenderCommand],
    ) -> Result<()> {
        surface::render(&mut self.state, surface, image, commands)
    }

    fn surface_present(
        &mut self,
        surface: SurfaceHandle,
        image: SwapchainImageHandle,
    ) -> Result<()> {
        surface::present(&mut self.state, surface, image)
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

    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        compute::destroy(&mut self.state, pipeline);
    }

    fn submit_compute(
        &mut self,
        device: DeviceHandle,
        commands: &[ComputeCommand],
    ) -> Result<super::FenceToken> {
        compute::submit(&mut self.state, device, commands)
    }

    fn is_fence_complete(&self, device: DeviceHandle, token: super::FenceToken) -> bool {
        compute::is_fence_complete(&self.state, device, token)
    }

    fn wait_fence(&mut self, device: DeviceHandle, token: super::FenceToken) -> Result<()> {
        compute::wait_fence(&self.state, device, token)?;
        // Successful wait establishes GPU idleness for everything submitted
        // up to and including `token`. Any bindless slots parked pending while
        // those command buffers were in-flight are now safe to recycle.
        drain_all_pending_slots(&mut self.state);
        Ok(())
    }

    fn wait_fence_timeout(
        &mut self,
        device: DeviceHandle,
        token: super::FenceToken,
        timeout_ms: u32,
    ) -> Result<bool> {
        let signaled = compute::wait_fence_timeout(&self.state, device, token, timeout_ms)?;
        if signaled {
            drain_all_pending_slots(&mut self.state);
        }
        Ok(signaled)
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
