//! Vulkan backend implementation.
//!
//! Targets Vulkan 1.3+ with dynamic rendering.
//! Supports surface presentation on Windows (VK_KHR_win32_surface) and Linux (VK_KHR_wayland_surface).
//!
//! ## Module Structure
//!
//! - `types`: Internal state structs for devices, buffers, shaders, etc.
//! - `utils`: Format conversion and memory type helpers

mod types;
mod utils;

use types::*;
use utils::{format_to_vk, vertex_format_to_vk, topology_to_vk};

use super::*;
use crate::types::Color;
use anyhow::{Context, Result};
use ash::{vk, khr};
use std::collections::HashMap;
use std::ffi::CStr;

#[cfg(target_os = "windows")]
use raw_window_handle::RawWindowHandle;

#[cfg(target_os = "linux")]
use raw_window_handle::{RawWindowHandle, RawDisplayHandle};

/// Vulkan backend.
pub struct VulkanBackend {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_devices: Vec<PhysicalDeviceInfo>,
    devices: HashMap<DeviceHandle, LogicalDevice>,
    next_device_handle: DeviceHandle,
    buffers: HashMap<BufferHandle, BufferState>,
    next_buffer_handle: BufferHandle,
    shaders: HashMap<ShaderHandle, ShaderState>,
    next_shader_handle: ShaderHandle,
    pipelines: HashMap<PipelineHandle, PipelineState>,
    next_pipeline_handle: PipelineHandle,
    bind_group_layouts: HashMap<BindGroupLayoutHandle, BindGroupLayoutState>,
    next_bind_group_layout_handle: BindGroupLayoutHandle,
    bind_groups: HashMap<BindGroupHandle, BindGroupState>,
    next_bind_group_handle: BindGroupHandle,
    render_targets: HashMap<RenderTargetHandle, RenderTargetState>,
    next_render_target_handle: RenderTargetHandle,
    surfaces: HashMap<SurfaceHandle, SurfaceState>,
    next_surface_handle: SurfaceHandle,
}

impl VulkanBackend {
    /// Create a new Vulkan backend.
    pub fn new() -> Result<Self> {
        tracing::info!("Initializing Vulkan backend");

        // Load Vulkan library
        let entry = unsafe { ash::Entry::load() }.context("Failed to load Vulkan library")?;

        // Check instance version
        let instance_version = unsafe { entry.try_enumerate_instance_version() }
            .context("Failed to enumerate instance version")?
            .unwrap_or(vk::API_VERSION_1_0);

        let major = vk::api_version_major(instance_version);
        let minor = vk::api_version_minor(instance_version);
        tracing::info!("Vulkan instance version: {}.{}", major, minor);

        if major < 1 || (major == 1 && minor < 3) {
            anyhow::bail!("RAG requires Vulkan 1.3+, found {}.{}", major, minor);
        }

        // Create instance with Vulkan 1.3 and surface extensions
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"rag")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"rag")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        // Surface extensions for windowed presentation
        let mut extensions: Vec<*const i8> = vec![
            khr::surface::NAME.as_ptr(),
        ];
        
        #[cfg(target_os = "windows")]
        extensions.push(khr::win32_surface::NAME.as_ptr());
        
        #[cfg(target_os = "linux")]
        extensions.push(khr::wayland_surface::NAME.as_ptr());

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions);

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .context("Failed to create Vulkan instance")?;

        // Enumerate physical devices
        let physical_devices_raw = unsafe { instance.enumerate_physical_devices() }
            .context("Failed to enumerate physical devices")?;

        let physical_devices: Vec<PhysicalDeviceInfo> = physical_devices_raw
            .into_iter()
            .enumerate()
            .map(|(idx, handle)| {
                let properties = unsafe { instance.get_physical_device_properties(handle) };
                PhysicalDeviceInfo {
                    handle,
                    properties,
                    adapter_id: idx as u32,
                }
            })
            .collect();

        tracing::info!("Found {} Vulkan physical devices", physical_devices.len());
        for dev in &physical_devices {
            let name = unsafe { CStr::from_ptr(dev.properties.device_name.as_ptr()) };
            tracing::info!(
                "  [{}] {} ({:?})",
                dev.adapter_id,
                name.to_string_lossy(),
                dev.properties.device_type
            );
        }

        Ok(Self {
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
            bind_group_layouts: HashMap::new(),
            next_bind_group_layout_handle: 1,
            bind_groups: HashMap::new(),
            next_bind_group_handle: 1,
            render_targets: HashMap::new(),
            next_render_target_handle: 1,
            surfaces: HashMap::new(),
            next_surface_handle: 1,
        })
    }

    /// Find a suitable memory type for allocation.
    fn find_memory_type(
        &self,
        physical_device: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        utils::find_memory_type(&self.instance, physical_device, type_filter, properties)
    }

    /// Create a platform-specific Vulkan surface.
    fn create_platform_surface(
        &self,
        window: &dyn raw_window_handle::HasWindowHandle,
        _display: &dyn raw_window_handle::HasDisplayHandle,
    ) -> Result<vk::SurfaceKHR> {
        let window_handle = window.window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {:?}", e))?;

        #[cfg(target_os = "windows")]
        {
            match window_handle.as_raw() {
                RawWindowHandle::Win32(h) => {
                    let create_info = vk::Win32SurfaceCreateInfoKHR::default()
                        .hwnd(h.hwnd.get() as isize)
                        .hinstance(
                            h.hinstance
                                .map(|i| i.get() as isize)
                                .unwrap_or(0)
                        );

                    let win32_surface = khr::win32_surface::Instance::new(&self.entry, &self.instance);
                    unsafe { win32_surface.create_win32_surface(&create_info, None) }
                        .context("Failed to create Win32 surface")
                }
                _ => anyhow::bail!("Expected Win32 window handle on Windows"),
            }
        }

        #[cfg(target_os = "linux")]
        {
            let display_handle = _display.display_handle()
                .map_err(|e| anyhow::anyhow!("Failed to get display handle: {:?}", e))?;

            match (window_handle.as_raw(), display_handle.as_raw()) {
                (RawWindowHandle::Wayland(w), RawDisplayHandle::Wayland(d)) => {
                    let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                        .display(d.display.as_ptr())
                        .surface(w.surface.as_ptr());

                    let wayland_surface = khr::wayland_surface::Instance::new(&self.entry, &self.instance);
                    unsafe { wayland_surface.create_wayland_surface(&create_info, None) }
                        .context("Failed to create Wayland surface")
                }
                _ => anyhow::bail!("Expected Wayland window/display handles on Linux (X11 not supported)"),
            }
        }

        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            anyhow::bail!("Surface creation not supported on this platform - use Metal backend on macOS")
        }
    }
    
    /// Compile a shader for a specific stage on demand.
    fn ensure_shader_stage_compiled(
        &mut self,
        shader_handle: ShaderHandle,
        stage: crate::slang::SlangStage,
    ) -> Result<vk::ShaderModule> {
        let shader = self.shaders.get_mut(&shader_handle)
            .context("Invalid shader handle")?;
        
        // Check if already compiled for this stage
        let cached_module = match stage {
            crate::slang::SlangStage::Vertex => shader.vertex_module,
            crate::slang::SlangStage::Fragment => shader.fragment_module,
            _ => None,
        };
        
        if let Some(module) = cached_module {
            return Ok(module);
        }
        
        // Get the entry point name based on stage
        let entry_point_name = match stage {
            crate::slang::SlangStage::Vertex => "vs_main",
            crate::slang::SlangStage::Fragment => "fs_main",
            _ => anyhow::bail!("Unsupported shader stage"),
        };
        
        // Compile with Slang
        let compiler = crate::slang::global_compiler()
            .context("Failed to get Slang compiler")?;
        
        let compiled = compiler.compile_entry_point(
            &shader.slang_source,
            crate::slang::ShaderTarget::Spirv,
            Some((entry_point_name, stage)),
        ).with_context(|| format!("Failed to compile {} shader", entry_point_name))?;
        
        let spirv = compiled.as_spirv()
            .context("Invalid SPIR-V output")?;
        
        // Get device
        let logical_device = self.devices.get(&shader.device_handle)
            .context("Shader's device no longer valid")?;
        
        // Create Vulkan shader module
        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let module = unsafe { logical_device.device.create_shader_module(&create_info, None) }
            .context("Failed to create Vulkan shader module")?;
        
        tracing::debug!("Compiled {} ({} SPIR-V words)", entry_point_name, spirv.len());
        
        // Cache the module - need to re-get shader as mutable
        let shader = self.shaders.get_mut(&shader_handle).unwrap();
        match stage {
            crate::slang::SlangStage::Vertex => shader.vertex_module = Some(module),
            crate::slang::SlangStage::Fragment => shader.fragment_module = Some(module),
            _ => {}
        }
        
        Ok(module)
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        tracing::info!("Shutting down Vulkan backend");

        // Destroy all devices (which will clean up their resources)
        let device_handles: Vec<_> = self.devices.keys().copied().collect();
        for handle in device_handles {
            self.destroy_device(handle);
        }

        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

// Include the GpuBackend implementation from the old file
// This is kept inline for now but could be further modularized
include!("impl_gpu_backend.rs");

