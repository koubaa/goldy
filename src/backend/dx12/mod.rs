//! DirectX 12 backend implementation.
//!
//! Targets D3D12 Feature Level 12.0+ on Windows.
//! Uses Slang for shader compilation (Slang -> DXIL directly with SM 6.6).
//!
//! ## WARP (software D3D12)
//!
//! Set **`GOLDY_DX12_FORCE_WARP=1`** to run on the DX12 WARP software rasterizer.
//! This registers WARP with DXGI and redirects [`Instance::create_device`](crate::Instance::create_device)
//! to it, even when hardware GPUs are present. Use on headless CI (no GPU) or locally to
//! reproduce WARP-specific rendering bugs.
//!
//! After the first WARP device is created, Goldy logs one stderr line showing which
//! `d3d10warp.dll` was loaded — useful to confirm a side-loaded NuGet build is active.
//!
//! See `docs/src/architecture/backends.md` (DX12 / WARP section) for NuGet side-loading
//! instructions.
//!
//! ## Module Structure
//!
//! - `types`: Internal state structs for devices, buffers, shaders, etc.
//! - `utils`: Format conversion and helpers

mod barriers;
mod buffer;
mod compute;
mod diagnostic;
pub(crate) use diagnostic::log_warp_module_path_once;
mod device;
mod pipeline;
mod pso_cache;
mod render_commands;
mod render_target;
mod sampler;
mod shader;
mod staging;
mod surface;
mod texture;
mod transient;
mod types;
mod utils;

use types::{Dx12State, DxgiAdapterInfo, LogicalDevice};

use super::*;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once, OnceLock};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::{D3D12GetDebugInterface, ID3D12Debug, ID3D12Debug1};
use windows::Win32::Graphics::Dxgi::*;

/// Adapter ID for the WARP device from [`IDXGIFactory4::EnumWarpAdapter`].
/// Used when `GOLDY_DX12_FORCE_WARP=1`.
pub const WARP_ADAPTER_ID: u32 = u32::MAX;

/// Whether `GOLDY_DX12_FORCE_WARP=1` is set.
///
/// Registers WARP with DXGI and redirects [`Instance::create_device`](crate::Instance::create_device)
/// to the WARP adapter regardless of what hardware GPUs are present.
pub(crate) fn env_force_warp() -> bool {
    std::env::var("GOLDY_DX12_FORCE_WARP").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn env_allow_warp() -> bool {
    env_force_warp()
}

static DEBUG_LAYER_INIT: Once = Once::new();

/// Shared DX12 backend singleton.
///
/// The D3D12 debug layer tracks all objects process-wide and is not thread-safe
/// under concurrent operations from multiple backend instances. Sharing a single
/// backend lets the existing `Arc<Mutex<...>>` in `Instance`/`Device` naturally
/// serialize all D3D12 calls, preventing access violations in parallel tests.
static SHARED_DX12: OnceLock<Arc<Mutex<Box<dyn super::GpuBackend>>>> = OnceLock::new();

/// True when the D3D12 debug layer will be enabled (debug build or GOLDY_DX12_DEBUG=1).
/// Used to decide between singleton (debug) vs per-instance (release) backend.
fn is_debug_mode() -> bool {
    let no_debug = std::env::var("GOLDY_DX12_NO_DEBUG").is_ok_and(|v| v == "1" || v == "true");
    !no_debug
        && (cfg!(debug_assertions)
            || std::env::var("GOLDY_DX12_DEBUG").is_ok_and(|v| v == "1" || v == "true"))
}

/// Get or create the shared DX12 backend.
pub fn shared_backend() -> anyhow::Result<Arc<Mutex<Box<dyn super::GpuBackend>>>> {
    if is_debug_mode() {
        Ok(SHARED_DX12
            .get_or_init(|| {
                let backend = Dx12Backend::new().expect("Failed to create DX12 backend");
                Arc::new(Mutex::new(Box::new(backend) as Box<dyn super::GpuBackend>))
            })
            .clone())
    } else {
        let backend = Dx12Backend::new()?;
        Ok(Arc::new(Mutex::new(
            Box::new(backend) as Box<dyn super::GpuBackend>
        )))
    }
}

/// DirectX 12 backend.
pub struct Dx12Backend {
    state: Dx12State,
}

impl Dx12Backend {
    /// Create a new DX12 backend.
    pub fn new() -> Result<Self> {
        tracing::info!("Initializing DX12 backend");

        // Enable debug layer in debug builds or when GOLDY_DX12_DEBUG=1.
        // Set GOLDY_DX12_NO_DEBUG=1 to force-disable (avoids debug-layer crashes
        // in parallel test threads where multiple D3D12 devices coexist).
        DEBUG_LAYER_INIT.call_once(|| {
            if is_debug_mode() {
                let mut debug_interface: Option<ID3D12Debug> = None;
                if unsafe { D3D12GetDebugInterface(&mut debug_interface) }.is_ok() {
                    if let Some(d) = debug_interface {
                        unsafe { d.EnableDebugLayer() };
                        tracing::info!("D3D12 debug layer enabled");

                        // GPU-Based Validation: catches UAV/SRV descriptor mismatches,
                        // resource state errors, and out-of-bounds access on the GPU timeline.
                        // Very slow — enable with GOLDY_DX12_GBV=1.
                        let enable_gbv = std::env::var("GOLDY_DX12_GBV")
                            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
                        if enable_gbv {
                            if let Ok(debug1) = d.cast::<ID3D12Debug1>() {
                                unsafe { debug1.SetEnableGPUBasedValidation(true) };
                                tracing::info!("D3D12 GPU-Based Validation (GBV) enabled");
                            } else {
                                tracing::warn!(
                                    "ID3D12Debug1 not available — GPU-Based Validation unavailable"
                                );
                            }
                        }
                    }
                }
            }
        });

        // Create DXGI factory
        let factory_flags = if is_debug_mode() {
            DXGI_CREATE_FACTORY_DEBUG
        } else {
            DXGI_CREATE_FACTORY_FLAGS(0)
        };

        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(factory_flags) }
            .context("Failed to create DXGI factory")?;

        // Check tearing support (IDXGIFactory5::CheckFeatureSupport)
        let allow_tearing = factory
            .cast::<IDXGIFactory5>()
            .ok()
            .and_then(|f5| {
                let mut allow: i32 = 0;
                let hr = unsafe {
                    f5.CheckFeatureSupport(
                        DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                        &mut allow as *mut _ as *mut _,
                        std::mem::size_of::<i32>() as u32,
                    )
                };
                hr.ok().map(|()| allow != 0)
            })
            .unwrap_or(false);
        tracing::info!("DXGI tearing support: {allow_tearing}");

        // Enumerate adapters
        let mut adapters = Vec::new();
        let mut adapter_index = 0u32;

        loop {
            let adapter_result: Result<IDXGIAdapter1, _> =
                unsafe { factory.EnumAdapters1(adapter_index) };
            match adapter_result {
                Ok(adapter) => {
                    let desc = match unsafe { adapter.GetDesc1() } {
                        Ok(d) => d,
                        Err(_) => continue,
                    };

                    // Skip software adapters unless explicitly requested
                    let flags = DXGI_ADAPTER_FLAG(desc.Flags as i32);
                    if !flags.contains(DXGI_ADAPTER_FLAG_SOFTWARE) {
                        let name = String::from_utf16_lossy(&desc.Description)
                            .trim_end_matches('\0')
                            .to_string();
                        tracing::info!("  [{}] {}", adapter_index, name);

                        adapters.push(DxgiAdapterInfo {
                            adapter,
                            desc,
                            adapter_id: adapter_index,
                        });
                    }
                    adapter_index += 1;
                }
                Err(_) => break,
            }
        }

        tracing::info!("Found {} hardware DXGI adapters", adapters.len());

        if env_allow_warp() {
            let warp_result: windows::core::Result<IDXGIAdapter> =
                unsafe { factory.EnumWarpAdapter() };
            match warp_result {
                Ok(warp) => match warp.cast::<IDXGIAdapter1>() {
                    Ok(adapter) => match unsafe { adapter.GetDesc1() } {
                        Ok(desc) => {
                            let name = String::from_utf16_lossy(&desc.Description)
                                .trim_end_matches('\0')
                                .to_string();
                            tracing::info!("  [{}] {} (WARP)", WARP_ADAPTER_ID, name);
                            adapters.push(DxgiAdapterInfo {
                                adapter,
                                desc,
                                adapter_id: WARP_ADAPTER_ID,
                            });
                        }
                        Err(e) => tracing::warn!("WARP GetDesc1 failed: {:?}", e),
                    },
                    Err(e) => tracing::warn!("WARP IDXGIAdapter cast failed: {:?}", e),
                },
                Err(e) => tracing::warn!("EnumWarpAdapter failed: {:?}", e),
            }
        }

        tracing::info!(
            "Total {} DX12 adapters (including WARP if enabled)",
            adapters.len()
        );

        // Create Slang compiler
        let slang_compiler =
            crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?;

        let state = Dx12State {
            factory,
            allow_tearing,
            adapters,
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
            next_rtv_offset: 0,
            free_rtv_offsets: Vec::new(),
            next_dsv_offset: 0,
            free_dsv_offsets: Vec::new(),
            slang_compiler,
            staging_belts: HashMap::new(),
            transient_heaps: HashMap::new(),
            next_transient_heap_handle: 1,
        };

        Ok(Self { state })
    }

    /// Wait for the GPU to finish all work on a device.
    fn wait_for_gpu(&self, device: &LogicalDevice) -> Result<()> {
        let fence_value = device.fence_value;
        unsafe { device.command_queue.Signal(&device.fence, fence_value) }
            .context("Failed to signal fence")?;
        utils::wait_for_fence(&device.fence, fence_value)
    }
}

impl Dx12Backend {
    fn destroy_device_inner(&mut self, device_handle: DeviceHandle) {
        transient::destroy_all_for_device(&mut self.state, device_handle);
        if let Some(mut logical_device) = self.state.devices.remove(&device_handle) {
            let _ = self.wait_for_gpu(&logical_device);
            logical_device
                .deletion_queue
                .flush_all(&mut logical_device.resource_registry);

            if logical_device.pso_disk_cache_dirty {
                if let Some(cache_root) = dirs::cache_dir() {
                    let path = cache_root
                        .join("goldy")
                        .join(format!("dx12_pso_{}.bin", logical_device.adapter_id));
                    if let Err(e) = pso_cache::save_maps(
                        &path,
                        &logical_device.graphics_pso_blobs,
                        &logical_device.compute_pso_blobs,
                    ) {
                        tracing::warn!(
                            error = ?e,
                            path = ?path,
                            "failed to save DX12 PSO disk cache"
                        );
                    }
                }
            }

            if let Some(mut belt) = self.state.staging_belts.remove(&device_handle) {
                unsafe {
                    belt.destroy_all();
                }
            }

            let buffer_handles: Vec<_> = self
                .state
                .buffers
                .iter()
                .filter(|(_, b)| b.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in buffer_handles {
                self.state.buffers.remove(&handle);
            }

            let shader_handles: Vec<_> = self
                .state
                .shaders
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in shader_handles {
                self.state.shaders.remove(&handle);
            }

            let pipeline_handles: Vec<_> = self
                .state
                .pipelines
                .iter()
                .filter(|(_, p)| p.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in pipeline_handles {
                self.state.pipelines.remove(&handle);
            }

            let target_handles: Vec<_> = self
                .state
                .render_targets
                .iter()
                .filter(|(_, t)| t.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in target_handles {
                self.state.render_targets.remove(&handle);
            }

            let surface_handles: Vec<_> = self
                .state
                .surfaces
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in surface_handles {
                self.state.surfaces.remove(&handle);
            }

            let texture_handles: Vec<_> = self
                .state
                .textures
                .iter()
                .filter(|(_, t)| t.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in texture_handles {
                self.state.textures.remove(&handle);
            }

            let sampler_handles: Vec<_> = self
                .state
                .samplers
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in sampler_handles {
                self.state.samplers.remove(&handle);
            }

            tracing::info!("Destroyed DX12 device {}", device_handle);
        }
    }
}

impl Drop for Dx12Backend {
    fn drop(&mut self) {
        tracing::info!("Shutting down DX12 backend");

        let device_handles: Vec<_> = self.state.devices.keys().copied().collect();
        for handle in device_handles {
            self.destroy_device_inner(handle);
        }
    }
}

impl GpuBackend for Dx12Backend {
    fn backend_type(&self) -> BackendType {
        BackendType::Dx12
    }

    fn enumerate_adapters(&self) -> Vec<super::AdapterInfo> {
        device::enumerate(&self.state.adapters)
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        device::create(&mut self.state, adapter_id)
    }

    fn destroy_device(&mut self, device_handle: DeviceHandle) {
        self.destroy_device_inner(device_handle);
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.state.devices.contains_key(&device)
    }

    fn create_buffer(
        &mut self,
        device_handle: DeviceHandle,
        size: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
    ) -> Result<BufferHandle> {
        buffer::create(
            &mut self.state,
            device_handle,
            size,
            access,
            element_stride,
            flags,
        )
    }

    fn destroy_buffer(&mut self, buffer_handle: BufferHandle) {
        buffer::destroy(&mut self.state, buffer_handle);
    }

    fn write_buffer(
        &mut self,
        buffer_handle: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<()> {
        buffer::write(&mut self.state, buffer_handle, offset, data)
    }

    fn buffer_size(&self, buffer_handle: BufferHandle) -> u64 {
        buffer::size(&self.state, buffer_handle)
    }

    fn buffer_bindless_index(&self, buffer_handle: BufferHandle) -> Option<u32> {
        buffer::bindless_index(&self.state, buffer_handle)
    }

    fn buffer_bindless_srv_index(&self, buffer_handle: BufferHandle) -> Option<u32> {
        buffer::bindless_srv_index(&self.state, buffer_handle)
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
        buffer::read_to_cpu(&mut self.state, device, buffer, output)
    }

    fn device_capabilities(&self, device: DeviceHandle) -> crate::device::DeviceCapabilities {
        let _ = device;
        crate::device::DeviceCapabilities {
            has_zero_copy_storage_readback: false,
            ..Default::default()
        }
    }

    fn clear_buffer(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        buffer::clear(&mut self.state, device, buffer, offset, size)
    }

    fn create_shader_with_paths(
        &mut self,
        device_handle: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
    ) -> Result<ShaderHandle> {
        self.create_shader_with_checks(
            device_handle,
            slang_source,
            search_paths,
            defines,
            optimization_level,
            vec![],
        )
    }

    fn create_shader_with_checks(
        &mut self,
        device_handle: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
        layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
    ) -> Result<ShaderHandle> {
        shader::create_with_checks(
            &mut self.state,
            device_handle,
            slang_source,
            search_paths,
            defines,
            optimization_level,
            layout_checks,
        )
    }

    fn destroy_shader(&mut self, shader_handle: ShaderHandle) {
        shader::destroy(&mut self.state, shader_handle);
    }

    fn create_pipeline(
        &mut self,
        device_handle: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        pipeline::create(
            &mut self.state,
            device_handle,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
        )
    }
    fn destroy_pipeline(&mut self, pipeline_handle: PipelineHandle) {
        pipeline::destroy(&mut self.state, pipeline_handle);
    }

    fn create_render_target(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<RenderTargetHandle> {
        render_target::create(&mut self.state, device_handle, width, height, format)
    }
    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        render_target::destroy(&mut self.state, target);
    }

    fn render_to_target(
        &mut self,
        device_handle: DeviceHandle,
        target: RenderTargetHandle,
        commands: &[RenderCommand],
    ) -> Result<()> {
        render_target::render(&mut self.state, device_handle, target, commands)
    }
    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()> {
        render_target::read_to_cpu(&mut self.state, target, output)
    }
    fn create_surface(
        &mut self,
        device_handle: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        display: &dyn raw_window_handle::HasDisplayHandle,
        depth_format: Option<crate::types::DepthFormat>,
    ) -> Result<SurfaceHandle> {
        surface::create(
            &mut self.state,
            device_handle,
            window,
            display,
            depth_format,
        )
    }
    fn destroy_surface(&mut self, surface_handle: SurfaceHandle) {
        surface::destroy(&mut self.state, surface_handle);
    }

    fn begin_frame(
        &mut self,
        surface_handle: SurfaceHandle,
    ) -> Result<(FrameToken, TextureHandle)> {
        let image = surface::acquire(&mut self.state, surface_handle)?;
        let tex = surface::frame_texture(&self.state, surface_handle)
            .context("begin_frame: surface frame texture unavailable")?;
        Ok((
            FrameToken {
                surface: surface_handle,
                image,
            },
            tex,
        ))
    }

    fn record_render(&mut self, frame: &FrameToken, commands: &[RenderCommand]) -> Result<()> {
        surface::render(&mut self.state, frame.surface, frame.image, commands)
    }

    fn surface_resize(
        &mut self,
        surface_handle: SurfaceHandle,
        width: u32,
        height: u32,
    ) -> Result<()> {
        surface::resize(&mut self.state, surface_handle, width, height)
    }
    fn surface_size(&self, surface_handle: SurfaceHandle) -> (u32, u32) {
        surface::size(&self.state, surface_handle)
    }

    fn surface_format(&self, surface_handle: SurfaceHandle) -> TextureFormat {
        surface::format(&self.state, surface_handle)
    }

    fn surface_set_present_mode(
        &mut self,
        surface_handle: SurfaceHandle,
        mode: crate::types::PresentMode,
    ) -> Result<()> {
        surface::set_present_mode(&mut self.state, surface_handle, mode)
    }

    fn surface_present_mode(&self, surface_handle: SurfaceHandle) -> crate::types::PresentMode {
        surface::get_present_mode(&self.state, surface_handle)
    }

    fn gpu_progress(&self, device_handle: DeviceHandle) -> crate::timeline::TimelineValue {
        self.state
            .devices
            .get(&device_handle)
            .map(|d| unsafe { d.fence.GetCompletedValue() })
            .unwrap_or(0)
    }

    fn wait_until(
        &mut self,
        device_handle: DeviceHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<()> {
        let fence = self
            .state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?
            .fence
            .clone();
        utils::wait_for_fence(&fence, value)?;
        if let Some(dev) = self.state.devices.get_mut(&device_handle) {
            dev.deletion_queue
                .process(&dev.fence, &mut dev.resource_registry);
        }
        Ok(())
    }

    fn wait_until_timeout(
        &mut self,
        device_handle: DeviceHandle,
        value: crate::timeline::TimelineValue,
        timeout_ms: u32,
    ) -> Result<bool> {
        let fence = self
            .state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?
            .fence
            .clone();
        let ok = utils::wait_for_fence_timeout(&fence, value, timeout_ms)?;
        if ok {
            if let Some(dev) = self.state.devices.get_mut(&device_handle) {
                dev.deletion_queue
                    .process(&dev.fence, &mut dev.resource_registry);
            }
        }
        Ok(ok)
    }

    fn submit_standalone(
        &mut self,
        device_handle: DeviceHandle,
        commands: &[GpuCommand],
    ) -> Result<crate::timeline::TimelineValue> {
        compute::submit(&mut self.state, device_handle, commands)
    }

    fn record_gpu_work(&mut self, frame: &FrameToken, commands: &[GpuCommand]) -> Result<()> {
        surface::record_gpu_work(&mut self.state, frame.surface, commands)
    }

    fn end_frame(&mut self, frame: FrameToken) -> Result<crate::timeline::TimelineValue> {
        surface::end_frame(&mut self.state, frame)
    }

    fn create_pipeline_with_depth(
        &mut self,
        device_handle: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        depth_stencil: Option<&crate::types::DepthStencilState>,
    ) -> Result<PipelineHandle> {
        pipeline::create_with_depth(
            &mut self.state,
            device_handle,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
            depth_stencil,
        )
    }

    fn create_render_target_with_depth(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<crate::types::DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        render_target::create_with_depth(
            &mut self.state,
            device_handle,
            width,
            height,
            color_format,
            depth_format,
        )
    }
    fn create_texture(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: SpatialAccess,
        flags: TextureFlags,
    ) -> Result<TextureHandle> {
        texture::create(
            &mut self.state,
            device_handle,
            width,
            height,
            format,
            access,
            flags,
        )
    }

    fn write_texture(
        &mut self,
        texture_handle: TextureHandle,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        texture::write(&mut self.state, texture_handle, data, width, height)
    }

    fn write_texture_region(
        &mut self,
        texture_handle: TextureHandle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<()> {
        texture::write_region(&mut self.state, texture_handle, x, y, width, height, data)
    }

    fn destroy_texture(&mut self, texture_handle: TextureHandle) {
        texture::destroy(&mut self.state, texture_handle);
    }

    fn read_texture_to_cpu(
        &mut self,
        texture_handle: TextureHandle,
        output: &mut [u8],
    ) -> Result<()> {
        texture::read_to_cpu(&mut self.state, texture_handle, output)
    }

    fn texture_bindless_index(&self, texture_handle: TextureHandle) -> Option<u32> {
        texture::bindless_index(&self.state, texture_handle)
    }

    fn create_sampler(
        &mut self,
        device_handle: DeviceHandle,
        desc: &crate::types::SamplerDesc,
    ) -> Result<SamplerHandle> {
        sampler::create(&mut self.state, device_handle, desc)
    }

    fn destroy_sampler(&mut self, sampler_handle: SamplerHandle) {
        sampler::destroy(&mut self.state, sampler_handle);
    }

    fn sampler_bindless_index(&self, sampler_handle: SamplerHandle) -> Option<u32> {
        sampler::bindless_index(&self.state, sampler_handle)
    }

    fn create_compute_pipeline(
        &mut self,
        device_handle: DeviceHandle,
        compute_shader: ShaderHandle,
    ) -> Result<ComputePipelineHandle> {
        compute::create(&mut self.state, device_handle, compute_shader)
    }

    fn destroy_compute_pipeline(&mut self, pipeline_handle: ComputePipelineHandle) {
        compute::destroy(&mut self.state, pipeline_handle);
    }

    fn reset_buffer_heaps(&mut self, device_handle: DeviceHandle) {
        if let Some(belt) = self.state.staging_belts.get_mut(&device_handle) {
            belt.trim();
        }
    }

    fn transient_texture_heap_footprint(
        &self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: crate::types::TextureFormat,
        access: crate::types::SpatialAccess,
        flags: crate::types::TextureFlags,
    ) -> Result<(u64, u64)> {
        transient::transient_texture_heap_footprint(
            &self.state,
            device,
            width,
            height,
            format,
            access,
            flags,
        )
    }

    fn create_transient_heap(
        &mut self,
        device: DeviceHandle,
        size: u64,
    ) -> Result<Option<TransientHeapHandle>> {
        transient::create_transient_heap(&mut self.state, device, size)
    }

    fn place_buffer_in_transient_heap(
        &mut self,
        device: DeviceHandle,
        heap: TransientHeapHandle,
        offset: u64,
        size: u64,
    ) -> Result<BufferHandle> {
        transient::place_buffer_in_transient_heap(&mut self.state, device, heap, offset, size)
    }

    fn place_texture_in_transient_heap(
        &mut self,
        device: DeviceHandle,
        heap: TransientHeapHandle,
        offset: u64,
        width: u32,
        height: u32,
        format: crate::types::TextureFormat,
        access: crate::types::SpatialAccess,
        flags: crate::types::TextureFlags,
    ) -> Result<TextureHandle> {
        transient::place_texture_in_transient_heap(
            &mut self.state,
            device,
            heap,
            offset,
            width,
            height,
            format,
            access,
            flags,
        )
    }

    fn destroy_transient_heap(
        &mut self,
        device: DeviceHandle,
        heap: TransientHeapHandle,
    ) -> Result<()> {
        transient::destroy_transient_heap(&mut self.state, device, heap)
    }

    fn transient_heap_alignment_hints(&self, device: DeviceHandle) -> TransientHeapAlignments {
        transient::transient_heap_alignment_hints(&self.state, device)
    }

    fn available_bindless_slots(
        &self,
        device_handle: DeviceHandle,
        category: crate::types::BindlessCategory,
    ) -> u32 {
        self.state
            .devices
            .get(&device_handle)
            .map(|ld| ld.resource_registry.available_slots(category))
            .unwrap_or(0)
    }

    fn max_bindless_slots_per_category(
        &self,
        _device_handle: DeviceHandle,
        category: crate::types::BindlessCategory,
    ) -> u32 {
        types::ResourceRegistry::max_slots(category)
    }

    fn flush_deferred_deletions(&mut self, device_handle: DeviceHandle) {
        if let Some(ld) = self.state.devices.get_mut(&device_handle) {
            ld.deletion_queue
                .process(&ld.fence, &mut ld.resource_registry);
        }
    }

    fn deferred_deletion_pending_count(&self, device_handle: DeviceHandle) -> usize {
        self.state
            .devices
            .get(&device_handle)
            .map(|d| d.deletion_queue.pending_len())
            .unwrap_or(0)
    }
}
