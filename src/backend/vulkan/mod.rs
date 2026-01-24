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

mod types;
mod utils;

use types::*;
use utils::{format_to_vk, index_format_to_vk, topology_to_vk, vertex_format_to_vk};

use super::*;
use crate::types::Color;
use anyhow::{Context, Result};
use ash::{khr, vk};
use std::collections::HashMap;
use std::ffi::CStr;

#[cfg(target_os = "windows")]
use raw_window_handle::RawWindowHandle;

#[cfg(target_os = "linux")]
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

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
    compute_pipelines: HashMap<ComputePipelineHandle, ComputePipelineState>,
    next_compute_pipeline_handle: ComputePipelineHandle,
    render_targets: HashMap<RenderTargetHandle, RenderTargetState>,
    next_render_target_handle: RenderTargetHandle,
    surfaces: HashMap<SurfaceHandle, SurfaceState>,
    next_surface_handle: SurfaceHandle,
    textures: HashMap<TextureHandle, TextureState>,
    next_texture_handle: TextureHandle,
    samplers: HashMap<SamplerHandle, SamplerState>,
    next_sampler_handle: SamplerHandle,
    /// Per-backend Slang compiler instance (avoids global state issues in tests)
    slang_compiler: crate::slang::SlangCompiler,
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
        let mut extensions: Vec<*const i8> = vec![khr::surface::NAME.as_ptr()];

        #[cfg(target_os = "windows")]
        extensions.push(khr::win32_surface::NAME.as_ptr());

        #[cfg(target_os = "linux")]
        extensions.push(khr::wayland_surface::NAME.as_ptr());

        // Enable validation layers if RAG_VALIDATION=1
        let enable_validation = std::env::var("RAG_VALIDATION")
            .map(|v| v == "1")
            .unwrap_or(false);
        let validation_layers: Vec<*const i8> = if enable_validation {
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
            let dev_major = vk::api_version_major(dev.properties.api_version);
            let dev_minor = vk::api_version_minor(dev.properties.api_version);
            tracing::info!(
                "  [{}] {} ({:?}) - Vulkan {}.{}",
                dev.adapter_id,
                name.to_string_lossy(),
                dev.properties.device_type,
                dev_major,
                dev_minor
            );
        }

        // Check that at least one device supports Vulkan 1.4+
        let has_vulkan_14 = physical_devices.iter().any(|dev| {
            let major = vk::api_version_major(dev.properties.api_version);
            let minor = vk::api_version_minor(dev.properties.api_version);
            major > 1 || (major == 1 && minor >= 4)
        });

        if !has_vulkan_14 {
            let versions: Vec<String> = physical_devices
                .iter()
                .map(|dev| {
                    let name = unsafe { CStr::from_ptr(dev.properties.device_name.as_ptr()) };
                    let major = vk::api_version_major(dev.properties.api_version);
                    let minor = vk::api_version_minor(dev.properties.api_version);
                    format!("{}: {}.{}", name.to_string_lossy(), major, minor)
                })
                .collect();
            anyhow::bail!(
                "Goldy requires Vulkan 1.4+, but no compatible devices found. Available: [{}]",
                versions.join(", ")
            );
        }

        // Create per-backend Slang compiler (avoids global state issues)
        let slang_compiler =
            crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?;

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
        #[cfg(target_os = "windows")]
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {:?}", e))?;

        #[cfg(target_os = "linux")]
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {:?}", e))?;

        // Silence unused warning on platforms where surface creation isn't supported
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        let _ = window;

        #[cfg(target_os = "windows")]
        {
            match window_handle.as_raw() {
                RawWindowHandle::Win32(h) => {
                    let create_info = vk::Win32SurfaceCreateInfoKHR::default()
                        .hwnd(h.hwnd.get() as isize)
                        .hinstance(h.hinstance.map(|i| i.get() as isize).unwrap_or(0));

                    let win32_surface =
                        khr::win32_surface::Instance::new(&self.entry, &self.instance);
                    unsafe { win32_surface.create_win32_surface(&create_info, None) }
                        .context("Failed to create Win32 surface")
                }
                _ => anyhow::bail!("Expected Win32 window handle on Windows"),
            }
        }

        #[cfg(target_os = "linux")]
        {
            let display_handle = _display
                .display_handle()
                .map_err(|e| anyhow::anyhow!("Failed to get display handle: {:?}", e))?;

            match (window_handle.as_raw(), display_handle.as_raw()) {
                (RawWindowHandle::Wayland(w), RawDisplayHandle::Wayland(d)) => {
                    let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                        .display(d.display.as_ptr())
                        .surface(w.surface.as_ptr());

                    let wayland_surface =
                        khr::wayland_surface::Instance::new(&self.entry, &self.instance);
                    unsafe { wayland_surface.create_wayland_surface(&create_info, None) }
                        .context("Failed to create Wayland surface")
                }
                _ => anyhow::bail!(
                    "Expected Wayland window/display handles on Linux (X11 not supported)"
                ),
            }
        }

        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            anyhow::bail!(
                "Surface creation not supported on this platform - use Metal backend on macOS"
            )
        }
    }

    /// Compile a shader for a specific stage on demand.
    fn ensure_shader_stage_compiled(
        &mut self,
        shader_handle: ShaderHandle,
        stage: crate::slang::SlangStage,
    ) -> Result<vk::ShaderModule> {
        let shader = self
            .shaders
            .get_mut(&shader_handle)
            .context("Invalid shader handle")?;

        // Check if already compiled for this stage
        let cached_module = match stage {
            crate::slang::SlangStage::Vertex => shader.vertex_module,
            crate::slang::SlangStage::Fragment => shader.fragment_module,
            crate::slang::SlangStage::Compute => shader.compute_module,
            _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
        };

        if let Some(module) = cached_module {
            return Ok(module);
        }

        // Get the entry point name based on stage
        let entry_point_name = match stage {
            crate::slang::SlangStage::Vertex => "vs_main",
            crate::slang::SlangStage::Fragment => "fs_main",
            crate::slang::SlangStage::Compute => "cs_main",
            _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
        };

        // Clone source and search paths to avoid borrow issues
        let slang_source = shader.slang_source.clone();
        let search_paths: Vec<&str> = shader.search_paths.iter().map(|s| s.as_str()).collect();
        let device_handle = shader.device_handle;

        // Compile shader with reflection data for resource binding
        let result = self
            .slang_compiler
            .compile_bindless_with_reflection(
                &slang_source,
                crate::slang::ShaderTarget::Spirv,
                &[(entry_point_name, stage)],
                &search_paths,
            )
            .with_context(|| format!("Failed to compile {} shader", entry_point_name))?;

        let spirv_data = result
            .shader
            .as_spirv()
            .context("Invalid SPIR-V output")?
            .to_vec();
        let reflection = Some(result.reflection);

        // Get device
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Shader's device no longer valid")?;

        // Create Vulkan shader module
        // Convert Vec<u8> to &[u32] for SPIR-V
        let spirv_u32: &[u32] = bytemuck::cast_slice(&spirv_data);
        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32);
        let module = unsafe {
            logical_device
                .device
                .create_shader_module(&create_info, None)
        }
        .context("Failed to create Vulkan shader module")?;

        tracing::debug!(
            "Compiled {} ({} SPIR-V words)",
            entry_point_name,
            spirv_u32.len()
        );

        // Dump SPIR-V for debugging when GOLDY_DUMP_SHADERS is set
        if let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") {
            use std::io::Write;
            let path =
                std::path::Path::new(&dump_dir).join(format!("{}_vulkan.spv", entry_point_name));
            if let Ok(mut file) = std::fs::File::create(&path) {
                let spirv_bytes: &[u8] = bytemuck::cast_slice(spirv_u32);
                let _ = file.write_all(spirv_bytes);
                tracing::info!("Dumped SPIR-V bytecode to {}", path.display());
            }
        }

        // Cache the module and reflection data
        let shader = self.shaders.get_mut(&shader_handle).unwrap();
        match stage {
            crate::slang::SlangStage::Vertex => shader.vertex_module = Some(module),
            crate::slang::SlangStage::Fragment => shader.fragment_module = Some(module),
            crate::slang::SlangStage::Compute => shader.compute_module = Some(module),
            _ => {} // Already validated above, shouldn't reach here
        }

        // Store reflection data (merge with existing if any)
        if let Some(ref new_reflection) = reflection {
            if let Some(ref mut existing) = shader.reflection {
                // Merge parameter blocks
                for pb in &new_reflection.parameter_blocks {
                    if !existing.parameter_blocks.iter().any(|p| p.name == pb.name) {
                        existing.parameter_blocks.push(pb.clone());
                    }
                }
            } else {
                shader.reflection = reflection;
            }
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
