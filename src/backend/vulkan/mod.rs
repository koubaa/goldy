//! Vulkan backend implementation.
//!
//! Targets Vulkan 1.4+ with dynamic rendering.
//! Supports surface presentation on Windows (VK_KHR_win32_surface) and Linux (VK_KHR_wayland_surface).
//!
//! ## Module Structure
//!
//! - `types`: Internal state structs for devices, buffers, shaders, etc.
//! - `utils`: Format conversion and memory type helpers

// Allow isize casts needed for FFI with raw-window-handle and ash
#![allow(clippy::unnecessary_cast)]

mod buffer;
mod compute;
mod device;
mod pipeline;
mod render_commands;
mod render_target;
mod sampler;
mod shader;
mod sparse;
mod staging;
mod surface;
mod texture;
mod types;
mod utils;

use types::*;

use super::*;
use anyhow::{Context, Result};
use ash::{khr, vk};
use std::collections::HashMap;
use std::ffi::{c_char, CStr};

/// Extract push-constant slot categories for a render pipeline from shader
/// reflection. Fragment shader data takes precedence; vertex is a fallback.
fn render_reflection_data(
    shaders: &HashMap<ShaderHandle, ShaderState>,
    vertex_shader: ShaderHandle,
    fragment_shader: ShaderHandle,
) -> (
    Vec<Option<crate::types::BindlessCategory>>,
    Vec<Option<u32>>,
) {
    let preferred = shaders
        .get(&fragment_shader)
        .and_then(|s| s.reflection.as_ref())
        .filter(|r| !r.push_constant_categories.is_empty())
        .or_else(|| {
            shaders
                .get(&vertex_shader)
                .and_then(|s| s.reflection.as_ref())
        });
    match preferred {
        Some(r) => (
            r.push_constant_categories.clone(),
            r.binding_element_strides.clone(),
        ),
        None => (Vec::new(), Vec::new()),
    }
}

/// Khronos instance validation when GPU API validation is requested (`GOLDY_VALIDATION=1`,
/// `api`, `all`, … — see `validation_env`), or when the loader forces
/// `VK_LAYER_KHRONOS_validation` via `VK_INSTANCE_LAYERS`.
fn vulkan_instance_validation_enabled() -> bool {
    if super::goldy_validation_enabled() {
        return true;
    }
    std::env::var("VK_INSTANCE_LAYERS")
        .map(|layers| layers.contains("VK_LAYER_KHRONOS_validation"))
        .unwrap_or(false)
}

/// Vulkan backend.
pub struct VulkanBackend {
    state: VulkanState,
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        tracing::info!(
            devices = self.state.devices.len(),
            surfaces = self.state.surfaces.len(),
            compute_fences = self.state.compute_fence_pool.len(),
            "VulkanBackend drop"
        );

        // Explicitly destroy the Vulkan instance before ash::Entry drops and unloads
        // the DLL. On device-lost, vkDestroyDevice may leave driver-internal background
        // state (TDR recovery threads) alive; vkDestroyInstance signals them to stop
        // before the DLL code is unmapped. Without this call the loader entry drop
        // (FreeLibrary) races with those threads and causes STATUS_HEAP_CORRUPTION.
        //
        // Only safe when all child objects (devices, surfaces) have been destroyed first.
        if self.state.devices.is_empty() && self.state.surfaces.is_empty() {
            unsafe {
                self.state.instance.destroy_instance(None);
            }
            tracing::info!("vkDestroyInstance complete");
        } else {
            tracing::warn!(
                devices = self.state.devices.len(),
                surfaces = self.state.surfaces.len(),
                "skipped vkDestroyInstance with child devices/surfaces still live (cleanup order bug?)"
            );
        }
    }
}

impl VulkanBackend {
    /// Create a new Vulkan backend.
    pub fn new() -> Result<Self> {
        tracing::info!("Initializing Vulkan backend");

        // Load Vulkan library
        let entry = unsafe { ash::Entry::load() }.context("Failed to load Vulkan library")?;

        // Check instance version (note: this is the loader version, not driver version)
        let instance_version = unsafe { entry.try_enumerate_instance_version() }
            .context("Failed to enumerate instance version")?
            .unwrap_or(vk::API_VERSION_1_0);

        let major = vk::api_version_major(instance_version);
        let minor = vk::api_version_minor(instance_version);
        tracing::info!("Vulkan loader version: {}.{}", major, minor);

        // Note: We request 1.4 from the instance, but the loader may be older.
        // The actual version check happens per-device when we enumerate physical devices.
        // Drivers can support 1.4 even if the loader is 1.3.

        // Create instance with Vulkan 1.4 and surface extensions
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"goldy")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"goldy")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 4, 0));

        // Surface extensions for windowed presentation
        let mut extensions: Vec<*const c_char> = vec![khr::surface::NAME.as_ptr()];

        #[cfg(target_os = "windows")]
        extensions.push(khr::win32_surface::NAME.as_ptr());

        #[cfg(target_os = "linux")]
        extensions.push(khr::wayland_surface::NAME.as_ptr());

        // Enable Khronos validation + VK_EXT_debug_utils when requested (see DEBUGGING.md).
        let enable_validation = vulkan_instance_validation_enabled();
        let validation_layers: Vec<*const c_char> = if enable_validation {
            tracing::info!("Vulkan validation layers ENABLED");
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
        } else {
            vec![]
        };

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&validation_layers);

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .context("Failed to create Vulkan instance")?;

        if enable_validation {
            tracing::info!("Vulkan instance created with validation layers");
        }

        // Enumerate physical devices
        let physical_devices_raw = unsafe { instance.enumerate_physical_devices() }
            .context("Failed to enumerate physical devices")?;

        // Only keep devices that report Vulkan 1.4+
        let mut adapter_id = 0u32;
        let mut rejected: Vec<String> = Vec::new();
        let physical_devices: Vec<PhysicalDeviceInfo> = physical_devices_raw
            .into_iter()
            .filter_map(|handle| {
                let properties = unsafe { instance.get_physical_device_properties(handle) };
                let major = vk::api_version_major(properties.api_version);
                let minor = vk::api_version_minor(properties.api_version);
                let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };

                if major > 1 || (major == 1 && minor >= 4) {
                    let id = adapter_id;
                    adapter_id += 1;
                    tracing::info!(
                        "  [{}] {} ({:?}) - Vulkan {}.{}",
                        id,
                        name.to_string_lossy(),
                        properties.device_type,
                        major,
                        minor
                    );
                    Some(PhysicalDeviceInfo {
                        handle,
                        properties,
                        adapter_id: id,
                    })
                } else {
                    rejected.push(format!("{}: {}.{}", name.to_string_lossy(), major, minor));
                    None
                }
            })
            .collect();

        if !rejected.is_empty() {
            tracing::info!("Skipped sub-1.4 devices: [{}]", rejected.join(", "));
        }

        tracing::info!(
            "Found {} Vulkan 1.4+ physical devices",
            physical_devices.len()
        );

        if physical_devices.is_empty() {
            anyhow::bail!(
                "Goldy requires Vulkan 1.4+, but no compatible devices found. Rejected: [{}]",
                rejected.join(", ")
            );
        }

        // Create per-backend Slang compiler (avoids global state issues)
        let slang_compiler =
            crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?;

        let state = VulkanState {
            entry,
            instance,
            physical_devices,
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
            slang_compiler,
            compute_fence_pool: HashMap::new(),
            compute_texture_staging_pool: HashMap::new(),
            staging_belts: HashMap::new(),
            timeline_cmd_buffers: HashMap::new(),
            device_lost: std::sync::atomic::AtomicBool::new(false),
        };

        Ok(Self { state })
    }

    /// Compile a shader for a specific stage on demand.
    fn ensure_shader_stage_compiled(
        &mut self,
        shader_handle: ShaderHandle,
        stage: crate::slang::SlangStage,
    ) -> Result<vk::ShaderModule> {
        shader::ensure_stage_compiled(
            &self.state.slang_compiler,
            &self.state.devices,
            &mut self.state.shaders,
            shader_handle,
            stage,
        )
    }
}

// GpuBackend trait implementation - thin wrapper delegating to domain modules
#[allow(clippy::manual_find)]
impl GpuBackend for VulkanBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Vulkan
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        device::enumerate(&self.state.physical_devices)
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        device::create(&mut self.state, adapter_id)
    }

    fn destroy_device(&mut self, device_handle: DeviceHandle) {
        device::destroy(&mut self.state, device_handle);
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
        device_handle: DeviceHandle,
        size: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
    ) -> Result<BufferHandle> {
        buffer::create(
            &mut self.state.devices,
            &mut self.state.buffers,
            &mut self.state.next_buffer_handle,
            &self.state.instance,
            device_handle,
            size,
            size,
            access,
            element_stride,
            flags,
        )
    }

    fn destroy_buffer(&mut self, buffer_handle: BufferHandle) {
        buffer::destroy(
            &mut self.state.devices,
            &mut self.state.buffers,
            buffer_handle,
        );
    }

    fn write_buffer(
        &mut self,
        buffer_handle: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<()> {
        buffer::write(
            &self.state.instance,
            &self.state.devices,
            &mut self.state.buffers,
            buffer_handle,
            offset,
            data,
        )
    }

    fn buffer_size(&self, buffer_handle: BufferHandle) -> u64 {
        buffer::size(&self.state.buffers, buffer_handle)
    }

    fn buffer_capacity(&self, buffer_handle: BufferHandle) -> u64 {
        buffer::capacity(&self.state.buffers, buffer_handle)
    }

    fn create_buffer_with_capacity(
        &mut self,
        device_handle: DeviceHandle,
        initial_size: u64,
        capacity: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        let cap = capacity.max(initial_size);
        let use_sparse = self.state.devices.get(&device_handle).is_some_and(|d| {
            d.supports_sparse_buffer
                && cap > initial_size
                && access == DataAccess::Scattered
                && !flags.contains(crate::types::BufferFlags::CPU_READABLE)
        });
        if use_sparse {
            let handle = buffer::create_sparse_with_capacity(
                &self.state.instance,
                &mut self.state.devices,
                &mut self.state.buffers,
                &mut self.state.next_buffer_handle,
                device_handle,
                initial_size,
                cap,
                element_stride,
                flags,
            )?;
            let cap_out = buffer::capacity(&self.state.buffers, handle);
            return Ok((handle, cap_out));
        }
        let handle = buffer::create(
            &mut self.state.devices,
            &mut self.state.buffers,
            &mut self.state.next_buffer_handle,
            &self.state.instance,
            device_handle,
            initial_size,
            cap,
            access,
            element_stride,
            flags,
        )?;
        Ok((handle, cap))
    }

    fn set_buffer_logical_size(
        &mut self,
        device_handle: DeviceHandle,
        buffer_handle: BufferHandle,
        new_logical_size: u64,
    ) -> Result<()> {
        buffer::set_logical_size(
            &mut self.state.devices,
            &mut self.state.buffers,
            device_handle,
            buffer_handle,
            new_logical_size,
        )
    }

    fn hint_buffer_unused_above(&mut self, buffer_handle: BufferHandle, offset: u64) {
        buffer::hint_unused_above(
            &mut self.state.devices,
            &mut self.state.buffers,
            buffer_handle,
            offset,
        );
    }

    fn device_capabilities(
        &self,
        device_handle: DeviceHandle,
    ) -> crate::device::DeviceCapabilities {
        let mut caps = crate::device::DeviceCapabilities::default();
        if self
            .state
            .devices
            .get(&device_handle)
            .is_some_and(|d| d.supports_sparse_buffer)
        {
            caps.buffer_resize_cost = crate::types::BufferResizeCost::PageBind;
            caps.buffer_page_size = 64 * 1024;
            caps.buffer_decommit_supported = true;
        }
        caps
    }

    fn buffer_bindless_index(&self, buffer_handle: BufferHandle) -> Option<u32> {
        buffer::bindless_index(&self.state.buffers, buffer_handle)
    }

    fn buffer_bindless_srv_index(&self, buffer_handle: BufferHandle) -> Option<u32> {
        // Vulkan uses the same storage buffer descriptor for both StructuredBuffer and RWStructuredBuffer
        buffer::bindless_index(&self.state.buffers, buffer_handle)
    }

    fn create_buffer_view(
        &mut self,
        parent: BufferHandle,
        offset: u64,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        buffer::create_view(
            &mut self.state.devices,
            &mut self.state.buffers,
            &mut self.state.next_buffer_handle,
            parent,
            offset,
            size,
            element_stride,
        )
    }

    fn resize_buffer(
        &mut self,
        device_handle: DeviceHandle,
        buffer_handle: BufferHandle,
        new_size: u64,
        preserve_contents: bool,
    ) -> Result<()> {
        buffer::resize(
            &self.state.instance,
            &mut self.state.devices,
            &mut self.state.buffers,
            device_handle,
            buffer_handle,
            new_size,
            preserve_contents,
        )
    }

    fn read_buffer_to_cpu(
        &mut self,
        device_handle: DeviceHandle,
        buffer_handle: BufferHandle,
        output: &mut [u8],
    ) -> Result<()> {
        buffer::read_to_cpu(
            &self.state.instance,
            &self.state.devices,
            &mut self.state.buffers,
            device_handle,
            buffer_handle,
            output,
        )
    }

    fn clear_buffer(
        &mut self,
        device_handle: DeviceHandle,
        buffer_handle: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        buffer::clear(
            &mut self.state.devices,
            &self.state.buffers,
            device_handle,
            buffer_handle,
            offset,
            size,
        )
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
        shader::create(
            &self.state.devices,
            &mut self.state.shaders,
            &mut self.state.next_shader_handle,
            crate::backend::shared::ShaderDesc::new(
                device_handle,
                slang_source,
                search_paths,
                defines,
                optimization_level,
            )
            .with_layout_checks(layout_checks),
        )
    }

    fn destroy_shader(&mut self, shader_handle: ShaderHandle) {
        shader::destroy(&self.state.devices, &mut self.state.shaders, shader_handle);
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
        // Compile shaders on-demand
        let vs_module =
            self.ensure_shader_stage_compiled(vertex_shader, crate::slang::SlangStage::Vertex)?;
        let fs_module =
            self.ensure_shader_stage_compiled(fragment_shader, crate::slang::SlangStage::Fragment)?;

        let (cats, strides) =
            render_reflection_data(&self.state.shaders, vertex_shader, fragment_shader);
        let shader_debug_name = format!("shader(vs=#{vertex_shader}, fs=#{fragment_shader})");

        let raster =
            crate::backend::shared::PipelineDesc::new(vertex_layout, topology, target_format);
        let handle = pipeline::create(pipeline::VulkanGraphicsPipelineCreateBundle {
            devices: &self.state.devices,
            pipelines: &mut self.state.pipelines,
            next_pipeline_handle: &mut self.state.next_pipeline_handle,
            device_handle,
            vs_module,
            fs_module,
            raster: &raster,
            shader_debug_name,
        })?;

        if let Some(ps) = self.state.pipelines.get_mut(&handle) {
            ps.push_constant_categories = cats;
            ps.binding_element_strides = strides;
        }
        Ok(handle)
    }

    fn destroy_pipeline(&mut self, pipeline_handle: PipelineHandle) {
        pipeline::destroy(
            &self.state.devices,
            &mut self.state.pipelines,
            pipeline_handle,
        );
    }

    fn create_render_target(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<RenderTargetHandle> {
        render_target::create(
            &self.state.instance,
            &self.state.devices,
            &mut self.state.render_targets,
            &mut self.state.next_render_target_handle,
            device_handle,
            width,
            height,
            format,
        )
    }

    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        render_target::destroy(&self.state.devices, &mut self.state.render_targets, target);
    }

    fn render_to_target(
        &mut self,
        device_handle: DeviceHandle,
        target: RenderTargetHandle,
        commands: &[RenderCommand],
    ) -> Result<()> {
        render_target::render_to(
            &self.state.devices,
            &mut self.state.render_targets,
            device_handle,
            target,
            commands,
            |cmd, cmds, logical_device, current_pipeline| {
                render_commands::record(
                    cmd,
                    cmds,
                    logical_device,
                    &self.state.pipelines,
                    &self.state.buffers,
                    current_pipeline,
                )
            },
        )
    }

    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()> {
        render_target::read_to_cpu(
            &self.state.instance,
            &self.state.devices,
            &mut self.state.render_targets,
            target,
            output,
        )
    }

    fn create_surface(
        &mut self,
        device_handle: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        display: &dyn raw_window_handle::HasDisplayHandle,
        depth_format: Option<crate::types::DepthFormat>,
    ) -> Result<SurfaceHandle> {
        surface::create(
            &self.state.entry,
            &self.state.instance,
            &self.state.devices,
            &mut self.state.surfaces,
            &mut self.state.next_surface_handle,
            device_handle,
            window,
            display,
            depth_format,
        )
    }

    fn destroy_surface(&mut self, surface_handle: SurfaceHandle) {
        surface::destroy(
            &self.state.entry,
            &self.state.instance,
            &mut self.state.devices,
            &mut self.state.surfaces,
            &mut self.state.textures,
            surface_handle,
        );
    }

    fn begin_frame(
        &mut self,
        surface_handle: SurfaceHandle,
    ) -> Result<(FrameToken, TextureHandle)> {
        let image = surface::acquire(&mut self.state, surface_handle)?;
        let tex = surface::frame_texture(&self.state.surfaces, surface_handle)
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
        surface::render(
            &mut self.state.devices,
            &mut self.state.surfaces,
            frame.surface,
            frame.image,
            commands,
            |cmd, cmds, logical_device, current_pipeline| {
                render_commands::record(
                    cmd,
                    cmds,
                    logical_device,
                    &self.state.pipelines,
                    &self.state.buffers,
                    current_pipeline,
                )
            },
        )
    }

    fn surface_resize(
        &mut self,
        surface_handle: SurfaceHandle,
        width: u32,
        height: u32,
    ) -> Result<()> {
        surface::resize(
            &self.state.entry,
            &self.state.instance,
            &self.state.devices,
            &mut self.state.surfaces,
            surface_handle,
            width,
            height,
        )
    }

    fn surface_size(&self, surface_handle: SurfaceHandle) -> (u32, u32) {
        surface::size(&self.state.surfaces, surface_handle)
    }

    fn surface_format(&self, surface_handle: SurfaceHandle) -> TextureFormat {
        surface::format(&self.state.surfaces, surface_handle)
    }

    fn surface_set_present_mode(
        &mut self,
        surface_handle: SurfaceHandle,
        mode: crate::types::PresentMode,
    ) -> Result<()> {
        surface::set_present_mode(
            &self.state.entry,
            &self.state.instance,
            &self.state.devices,
            &mut self.state.surfaces,
            surface_handle,
            mode,
        )
    }

    fn surface_present_mode(&self, surface_handle: SurfaceHandle) -> crate::types::PresentMode {
        surface::get_present_mode(&self.state.surfaces, surface_handle)
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
        // Compile shaders on-demand
        let vs_module =
            self.ensure_shader_stage_compiled(vertex_shader, crate::slang::SlangStage::Vertex)?;
        let fs_module =
            self.ensure_shader_stage_compiled(fragment_shader, crate::slang::SlangStage::Fragment)?;

        let shader_debug_name = format!("shader(vs=#{vertex_shader}, fs=#{fragment_shader})");

        let (cats, strides) =
            render_reflection_data(&self.state.shaders, vertex_shader, fragment_shader);

        let raster =
            crate::backend::shared::PipelineDesc::new(vertex_layout, topology, target_format)
                .with_depth_stencil(depth_stencil);
        let handle = pipeline::create_with_depth(pipeline::VulkanGraphicsPipelineCreateBundle {
            devices: &self.state.devices,
            pipelines: &mut self.state.pipelines,
            next_pipeline_handle: &mut self.state.next_pipeline_handle,
            device_handle,
            vs_module,
            fs_module,
            raster: &raster,
            shader_debug_name,
        })?;

        if let Some(ps) = self.state.pipelines.get_mut(&handle) {
            ps.push_constant_categories = cats;
            ps.binding_element_strides = strides;
        }
        Ok(handle)
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
            &self.state.instance,
            &self.state.devices,
            &mut self.state.render_targets,
            &mut self.state.next_render_target_handle,
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
            &self.state.instance,
            &mut self.state.devices,
            &mut self.state.textures,
            &mut self.state.next_texture_handle,
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
        texture::write(
            &self.state.instance,
            &self.state.devices,
            &mut self.state.textures,
            texture_handle,
            data,
            width,
            height,
        )
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
        texture::write_region(
            &self.state.instance,
            &self.state.devices,
            &mut self.state.textures,
            texture_handle,
            x,
            y,
            width,
            height,
            data,
        )
    }

    fn destroy_texture(&mut self, texture_handle: TextureHandle) {
        texture::destroy(
            &mut self.state.devices,
            &mut self.state.textures,
            texture_handle,
        );
    }

    fn read_texture_to_cpu(
        &mut self,
        texture_handle: TextureHandle,
        output: &mut [u8],
    ) -> Result<()> {
        texture::read_to_cpu(
            &self.state.instance,
            &self.state.devices,
            &mut self.state.textures,
            texture_handle,
            output,
        )
    }

    fn texture_bindless_index(&self, texture_handle: TextureHandle) -> Option<u32> {
        texture::bindless_index(&self.state.textures, texture_handle)
    }

    fn create_sampler(
        &mut self,
        device_handle: DeviceHandle,
        desc: &crate::types::SamplerDesc,
    ) -> Result<SamplerHandle> {
        sampler::create(
            &mut self.state.devices,
            &mut self.state.samplers,
            &mut self.state.next_sampler_handle,
            device_handle,
            desc,
        )
    }

    fn destroy_sampler(&mut self, sampler_handle: SamplerHandle) {
        sampler::destroy(
            &mut self.state.devices,
            &mut self.state.samplers,
            sampler_handle,
        );
    }

    fn sampler_bindless_index(&self, sampler_handle: SamplerHandle) -> Option<u32> {
        sampler::bindless_index(&self.state.samplers, sampler_handle)
    }

    fn create_compute_pipeline(
        &mut self,
        device_handle: DeviceHandle,
        compute_shader: ShaderHandle,
    ) -> Result<ComputePipelineHandle> {
        // Compile shader on-demand
        let cs_module =
            self.ensure_shader_stage_compiled(compute_shader, crate::slang::SlangStage::Compute)?;

        let (cats, strides) = self
            .state
            .shaders
            .get(&compute_shader)
            .and_then(|s| s.reflection.as_ref())
            .map(|r| {
                (
                    r.push_constant_categories.clone(),
                    r.binding_element_strides.clone(),
                )
            })
            .unwrap_or_default();

        let shader_debug_name = format!("compute_shader#{compute_shader}");

        let handle = compute::create(
            &self.state.devices,
            &mut self.state.compute_pipelines,
            &mut self.state.next_compute_pipeline_handle,
            device_handle,
            cs_module,
            shader_debug_name,
        )?;

        if let Some(ps) = self.state.compute_pipelines.get_mut(&handle) {
            ps.push_constant_categories = cats;
            ps.binding_element_strides = strides;
        }
        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline_handle: ComputePipelineHandle) {
        compute::destroy(
            &self.state.devices,
            &mut self.state.compute_pipelines,
            pipeline_handle,
        );
    }

    fn gpu_progress(&self, device_handle: DeviceHandle) -> crate::timeline::TimelineValue {
        let Some(ld) = self.state.devices.get(&device_handle) else {
            return 0;
        };
        unsafe { ld.device.get_semaphore_counter_value(ld.timeline_semaphore) }.unwrap_or(0)
    }

    fn wait_until(
        &mut self,
        device_handle: DeviceHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<()> {
        let sem = self
            .state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?
            .timeline_semaphore;
        let dev = &self.state.devices.get(&device_handle).unwrap().device;
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&sem))
            .values(std::slice::from_ref(&value));
        if let Err(e) = unsafe { dev.wait_semaphores(&wait, u64::MAX) } {
            if e == vk::Result::ERROR_DEVICE_LOST {
                self.state
                    .device_lost
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            return Err(anyhow::anyhow!("wait_semaphores: {:?}", e));
        }
        compute::reap_timeline_cmd_buffers_up_to(&mut self.state, device_handle, value);
        if let Some(ld) = self.state.devices.get_mut(&device_handle) {
            let drained = ld.deletion_queue.drain_up_to(value);
            for r in drained {
                types::destroy_pending_deletion(ld, r);
            }
        }
        Ok(())
    }

    fn wait_until_timeout(
        &mut self,
        device_handle: DeviceHandle,
        value: crate::timeline::TimelineValue,
        timeout_ms: u32,
    ) -> Result<bool> {
        let sem = self
            .state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?
            .timeline_semaphore;
        let dev = &self.state.devices.get(&device_handle).unwrap().device;
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&sem))
            .values(std::slice::from_ref(&value));
        let timeout_ns = (timeout_ms as u64).saturating_mul(1_000_000);
        match unsafe { dev.wait_semaphores(&wait, timeout_ns) } {
            Ok(()) => {
                compute::reap_timeline_cmd_buffers_up_to(&mut self.state, device_handle, value);
                if let Some(ld) = self.state.devices.get_mut(&device_handle) {
                    let drained = ld.deletion_queue.drain_up_to(value);
                    for r in drained {
                        types::destroy_pending_deletion(ld, r);
                    }
                }
                Ok(true)
            }
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => {
                if e == vk::Result::ERROR_DEVICE_LOST {
                    self.state
                        .device_lost
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
                Err(anyhow::anyhow!("wait_semaphores: {:?}", e))
            }
        }
    }

    fn submit_standalone(
        &mut self,
        device_handle: DeviceHandle,
        commands: &[GpuCommand],
    ) -> Result<crate::timeline::TimelineValue> {
        compute::submit(&mut self.state, device_handle, commands, None)
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

    fn reset_buffer_heaps(&mut self, device_handle: DeviceHandle) {
        if let (Some(logical_device), Some(belt)) = (
            self.state.devices.get(&device_handle),
            self.state.staging_belts.get_mut(&device_handle),
        ) {
            unsafe { belt.trim(logical_device) };
        }
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
        _category: crate::types::BindlessCategory,
    ) -> u32 {
        types::MAX_BINDLESS_RESOURCES
    }

    fn flush_deferred_deletions(&mut self, device_handle: DeviceHandle) {
        if let Some(ld) = self.state.devices.get_mut(&device_handle) {
            ld.process_deletion_queue_up_to_gpu_progress();
        }
    }

    fn deferred_deletion_pending_count(&self, device_handle: DeviceHandle) -> usize {
        self.state
            .devices
            .get(&device_handle)
            .map(|d| d.deletion_queue.len())
            .unwrap_or(0)
    }
}
