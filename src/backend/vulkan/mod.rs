//! Vulkan backend implementation.
//!
//! Targets Vulkan 1.4+ with dynamic rendering.
//! Supports surface presentation on Windows (`VK_KHR_win32_surface`), Linux
//! (`VK_KHR_wayland_surface`), and Android (`VK_KHR_android_surface`).
//!
//! ## Module Structure
//!
//! - `types`: Internal state structs for devices, buffers, shaders, etc.
//! - `utils`: Format conversion and memory type helpers

// Allow isize casts needed for FFI with raw-window-handle and ash
#![allow(clippy::unnecessary_cast)]

mod api_log;
mod buffer;
mod compute;
mod context;
mod debug_utils;
mod device;
mod frame_table;
mod pending_submit;
mod pipeline;
mod present_split;
mod render_commands;
mod render_target;
mod sampler;
mod shader;
mod sparse;
mod staging;
mod submit_session;
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
use std::sync::{Arc, Mutex, RwLock};

/// Process-global lock serialising `vkCreateInstance` and `vkDestroyInstance`.
///
/// The Vulkan spec marks both calls as implicitly externally synchronized at the
/// loader level (ICD enumeration, dispatch-table construction, layer init). No
/// internal lock protects that global state, so concurrent calls from different
/// test threads produce undefined behaviour — visible as a SIGSEGV on lavapipe
/// and as heap corruption on other software renderers.  Hardware drivers often
/// have incidental internal serialisation that masks the race, but the UB exists
/// regardless.  Holding this lock only for the duration of instance
/// creation/destruction adds negligible overhead in production (instances are
/// long-lived) while making tests safe under the default parallel test runner.
static VK_INSTANCE_LOCK: Mutex<()> = Mutex::new(());

/// Extract push-constant slot categories for a render pipeline from shader
/// reflection. Fragment shader data takes precedence; vertex is a fallback.
fn render_reflection_data(
    shaders: &HashMap<ShaderHandle, ShaderState>, // caller passes entries snapshot
    vertex_shader: ShaderHandle,
    fragment_shader: ShaderHandle,
) -> (Vec<Option<crate::types::ResourceCategory>>, Vec<Option<u32>>) {
    let preferred = shaders
        .get(&fragment_shader)
        .and_then(|s| s.reflection.as_ref())
        .filter(|r| !r.push_constant_categories.is_empty())
        .or_else(|| shaders.get(&vertex_shader).and_then(|s| s.reflection.as_ref()));
    match preferred {
        Some(r) => (r.push_constant_categories.clone(), r.binding_element_strides.clone()),
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
pub(crate) struct VulkanBackend {
    state: VulkanState,
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        tracing::info!(
            devices = self.state.devices.len(),
            surfaces = self.state.surfaces.len(),
            compute_fences = self.state.compute_fence_pool.lock().unwrap().len(),
            "VulkanBackend drop"
        );

        debug_utils::destroy_messenger(self.state.debug_utils.as_ref(), self.state.debug_messenger);
        self.state.debug_messenger = vk::DebugUtilsMessengerEXT::null();
        self.state.debug_utils = None;

        // Explicitly destroy the Vulkan instance before ash::Entry drops and unloads
        // the DLL. On device-lost, vkDestroyDevice may leave driver-internal background
        // state (TDR recovery threads) alive; vkDestroyInstance signals them to stop
        // before the DLL code is unmapped. Without this call the loader entry drop
        // (FreeLibrary) races with those threads and causes STATUS_HEAP_CORRUPTION.
        //
        // Only safe when all child objects (devices, surfaces) have been destroyed first.
        if self.state.devices.is_empty() && self.state.surfaces.is_empty() {
            let _guard = VK_INSTANCE_LOCK.lock().unwrap();
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

        if !std::thread::panicking() {
            if let Err(e) = debug_utils::fail_if_validation_fatal(self.state.validation_sink.as_ref()) {
                panic!("{e:#}");
            }
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

        #[cfg(target_os = "android")]
        extensions.push(khr::android_surface::NAME.as_ptr());

        // Enable Khronos validation + VK_EXT_debug_utils when requested (see DEBUGGING.md).
        let enable_validation = vulkan_instance_validation_enabled();
        let mut enabled_layers: Vec<*const c_char> = Vec::new();
        if enable_validation {
            tracing::info!("Vulkan validation layers ENABLED");
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            enabled_layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        }

        // GOLDY_API_LOG → VK_LAYER_LUNARG_api_dump (JSON file); see api_log.rs.
        if let Some(layer) = api_log::configure_and_layer(&entry) {
            enabled_layers.push(layer);
        }

        let validation_sink = enable_validation.then(|| {
            Arc::new(debug_utils::ValidationSink::new(
                crate::validation_env::validation_fatal_enabled(),
            ))
        });
        let mut debug_ci = validation_sink
            .as_ref()
            .map(|sink| debug_utils::messenger_create_info(debug_utils::sink_user_data(sink)));

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&enabled_layers);
        if let Some(debug_ci) = debug_ci.as_mut() {
            create_info = create_info.push_next(debug_ci);
        }

        let instance = {
            let _guard = VK_INSTANCE_LOCK.lock().unwrap();
            unsafe { entry.create_instance(&create_info, None) }.context("Failed to create Vulkan instance")?
        };

        let (debug_utils_loader, debug_messenger) = if let Some(debug_ci) = debug_ci.as_ref() {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            match unsafe { loader.create_debug_utils_messenger(debug_ci, None) } {
                Ok(messenger) => {
                    tracing::info!("Vulkan instance created with validation layers + debug messenger");
                    (Some(loader), messenger)
                }
                Err(e) => {
                    let _guard = VK_INSTANCE_LOCK.lock().unwrap();
                    unsafe {
                        instance.destroy_instance(None);
                    }
                    return Err(e).context("vkCreateDebugUtilsMessengerEXT");
                }
            }
        } else {
            (None, vk::DebugUtilsMessengerEXT::null())
        };

        // Enumerate physical devices
        let physical_devices_raw =
            unsafe { instance.enumerate_physical_devices() }.context("Failed to enumerate physical devices")?;

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
                    let pdev_features = unsafe { instance.get_physical_device_features(handle) };
                    let supports_sparse =
                        pdev_features.sparse_binding != 0 && pdev_features.sparse_residency_buffer != 0;
                    let rt_mesh = device::query_rt_mesh_features(&instance, handle);
                    tracing::info!(
                        "  [{}] {} ({:?}) - Vulkan {}.{}; ray_query={} rt_pipe={} mesh={} task={}",
                        id,
                        name.to_string_lossy(),
                        properties.device_type,
                        major,
                        minor,
                        rt_mesh.ray_query,
                        rt_mesh.ray_tracing_pipelines,
                        rt_mesh.mesh_shaders,
                        rt_mesh.amplification_shaders
                    );
                    Some(PhysicalDeviceInfo {
                        handle,
                        properties,
                        adapter_id: id,
                        supports_sparse_buffer: supports_sparse,
                        vk_timestamp_compute_and_graphics: properties.limits.timestamp_compute_and_graphics != 0,
                        vk_timestamp_period_ns: properties.limits.timestamp_period,
                        ray_query: rt_mesh.ray_query,
                        ray_tracing_pipelines: rt_mesh.ray_tracing_pipelines,
                        mesh_shaders: rt_mesh.mesh_shaders,
                        amplification_shaders: rt_mesh.amplification_shaders,
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

        tracing::info!("Found {} Vulkan 1.4+ physical devices", physical_devices.len());

        if physical_devices.is_empty() {
            anyhow::bail!(
                "Goldy requires Vulkan 1.4+, but no compatible devices found. Rejected: [{}]",
                rejected.join(", ")
            );
        }

        // Create per-backend Slang compiler (avoids global state issues)
        let slang_compiler = crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?;

        let state = VulkanState {
            entry,
            instance,
            physical_devices,
            devices: HashMap::new(),
            next_device_handle: 1,
            contexts: Arc::new(RwLock::new(HashMap::new())),
            next_context_id: 1,
            device_owner_handles: HashMap::new(),
            buffers: Arc::new(RwLock::new(BufferTable::new())),
            shaders: Arc::new(RwLock::new(ShaderTable::new())),
            pipelines: Arc::new(RwLock::new(PipelineTable::new())),
            compute_pipelines: Arc::new(RwLock::new(ComputePipelineTable::new())),
            render_targets: Arc::new(RwLock::new(RenderTargetTable::new())),
            surfaces: HashMap::new(),
            next_surface_handle: 1,
            textures: Arc::new(RwLock::new(TextureTable::new())),
            samplers: Arc::new(RwLock::new(SamplerTable::new())),
            slang_compiler,
            compute_fence_pool: Arc::new(Mutex::new(HashMap::new())),
            device_lost: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            enable_validation,
            debug_utils: debug_utils_loader,
            debug_messenger,
            validation_sink,
        };

        Ok(Self { state })
    }

    fn with_validation<T>(&self, result: Result<T>) -> Result<T> {
        debug_utils::combine_validation(self.state.validation_sink.as_ref(), result)
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
            &self.state.shaders,
            shader_handle,
            stage,
        )
    }
}

// GpuBackend trait implementation - thin wrapper delegating to domain modules
#[allow(clippy::manual_find)]
impl crate::backend::GpuBackendTimelineWait for VulkanBackend {
    fn take_timeline_submission_epoch_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<crate::backend::submission_worker::SubmissionEpochWait>> {
        if self.gpu_progress(ctx) >= value {
            return Ok(None);
        }
        let device_handle = self.context_device(ctx);
        let Some(ld) = self.state.devices.get(&device_handle) else {
            return Ok(None);
        };
        let horizon = crate::backend::submission_worker::submission_horizon(&ld.timeline_next);
        if value == 0 || value > horizon {
            return Ok(None);
        }
        Ok(Some(crate::backend::submission_worker::SubmissionEpochWait::new(
            std::sync::Arc::clone(&ld.submission_worker),
            value,
            horizon,
        )))
    }

    fn take_timeline_blocking_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn crate::backend::TimelineBlockingWait>>> {
        if self.gpu_progress(ctx) >= value {
            return Ok(None);
        }
        let device_handle = self.context_device(ctx);
        let sem = self
            .state
            .contexts
            .read()
            .unwrap()
            .get(&ctx)
            .context("Invalid context handle")?
            .lock()
            .unwrap()
            .timeline_semaphore;
        let device = self.state.devices.get(&device_handle).unwrap().device.clone();
        Ok(Some(Box::new(VulkanTimelineBlockingWait {
            device,
            semaphore: sem,
            value,
            device_lost: std::sync::Arc::clone(&self.state.device_lost),
        })))
    }

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        let device_handle = self.context_device(ctx);
        if let Some(ld) = self.state.devices.get(&device_handle) {
            ld.submission_worker.flush()?;
        }
        {
            let _reap = crate::tracy_zone!("vk.wait_until.reap_timeline_cmd_buffers");
            compute::reap_timeline_cmd_buffers_up_to(&self.state, ctx, value);
        }
        if let Some(ld) = self.state.devices.get(&device_handle) {
            if let Some(sc_arc) = self.state.contexts.read().unwrap().get(&ctx) {
                pending_submit::vulkan_drain_context_deletion_up_to(
                    ld,
                    &self.state.contexts,
                    device_handle,
                    sc_arc,
                    value,
                );
                pending_submit::vulkan_drain_pending_gpu_profiles_up_to(ld, &mut sc_arc.lock().unwrap(), value);
            }
            {
                let _destroy = crate::tracy_zone!("vk.wait_until.deletion_queue.destroy");
                let completed_values =
                    types::snapshot_context_completed_values(&ld.device, &self.state.contexts, device_handle);
                let device_batch = ld.drain_deletion_queue_ready(&completed_values);
                let descriptors_arc = std::sync::Arc::clone(&ld.descriptors);
                let mut registry = descriptors_arc.lock().unwrap();
                for r in device_batch {
                    types::destroy_pending_deletion(ld, &mut registry, r);
                }
                registry.drain_ready_slot_reclamations(&completed_values);
            }
        }
        self.with_validation(Ok(()))
    }
}

impl crate::backend::GpuBackendPresentSplit for VulkanBackend {
    fn take_present_gpu_work(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn crate::backend::PresentGpuWork>> {
        present_split::prepare_present_work(&self.state, frame, submit_tv)
    }

    fn finish_present(
        &mut self,
        finish: crate::backend::PresentFinishState,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        present_split::finish_present(&mut self.state, finish)
    }
}

#[allow(clippy::manual_find)]
impl GpuBackend for VulkanBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Vulkan
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        device::enumerate(&self.state.physical_devices)
    }

    fn adapter_capabilities(&self, adapter_id: u32) -> crate::device::DeviceCapabilities {
        device::adapter_capabilities(&self.state.physical_devices, adapter_id)
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        let r = device::create(&mut self.state, adapter_id);
        debug_utils::combine_validation(self.state.validation_sink.as_ref(), r)
    }

    fn destroy_device(&mut self, device_handle: DeviceHandle) {
        let ctxs: Vec<ContextHandle> = self
            .state
            .contexts
            .read()
            .unwrap()
            .iter()
            .filter(|(_, sc)| {
                let sc = sc.lock().unwrap();
                sc.device == device_handle && !sc.is_device_owner
            })
            .map(|(k, _)| *k)
            .collect();
        for ctx in ctxs {
            crate::backend::destroy_context_mut(self, ctx);
        }
        device::destroy(&mut self.state, device_handle);
    }

    fn device_wait_idle(&mut self, device_handle: DeviceHandle) -> Result<()> {
        let ld = self
            .state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        ld.synchronized_device_wait_idle()
            .map_err(|e| anyhow::anyhow!("device_wait_idle: {:?}", e))?;
        self.with_validation(Ok(()))
    }

    fn create_context(&mut self, device: DeviceHandle) -> Result<ContextHandle> {
        let r = context::create(&mut self.state, device);
        debug_utils::combine_validation(self.state.validation_sink.as_ref(), r)
    }

    fn detach_context_for_destroy(
        &mut self,
        ctx: ContextHandle,
    ) -> Option<Box<dyn crate::backend::ContextDestroyHandle>> {
        context::detach_for_destroy(&self.state, ctx)
            .map(|work| Box::new(work) as Box<dyn crate::backend::ContextDestroyHandle>)
    }

    fn clone_context_deletion_flush(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn crate::backend::ContextDeferredDeletionFlush>> {
        let sc = std::sync::Arc::clone(self.state.contexts.read().unwrap().get(&ctx)?);
        let device_handle = {
            let sc_guard = sc.lock().unwrap();
            sc_guard.device
        };
        let ld = std::sync::Arc::clone(self.state.devices.get(&device_handle)?);
        Some(std::sync::Arc::new(VulkanContextDeferredDeletionFlush {
            ctx,
            sc,
            ld,
            contexts: std::sync::Arc::clone(&self.state.contexts),
            device_handle,
        }))
    }

    fn clone_context_gpu_progress(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn crate::backend::ContextGpuProgress>> {
        let sc = std::sync::Arc::clone(self.state.contexts.read().unwrap().get(&ctx)?);
        let device_handle = {
            let sc_guard = sc.lock().unwrap();
            sc_guard.device
        };
        let ld = std::sync::Arc::clone(self.state.devices.get(&device_handle)?);
        Some(std::sync::Arc::new(VulkanContextGpuProgress { sc, ld }))
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
        device_handle: DeviceHandle,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
    ) -> Result<BufferHandle> {
        buffer::create(
            &self.state.devices,
            &self.state.buffers,
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
        buffer::destroy(&self.state, buffer_handle);
    }

    fn write_buffer(&mut self, buffer_handle: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        buffer::write(
            &self.state.instance,
            &self.state.devices,
            &self.state.buffers,
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
        access: BufferKind,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        let cap = capacity.max(initial_size);
        let use_sparse = self.state.devices.get(&device_handle).is_some_and(|d| {
            d.supports_sparse_buffer
                && cap > initial_size
                && access == BufferKind::Scattered
                && !flags.contains(crate::types::BufferFlags::CPU_READABLE)
        });
        if use_sparse {
            let handle = buffer::create_sparse_with_capacity(
                &self.state.instance,
                &self.state.devices,
                &self.state.buffers,
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
            &self.state.devices,
            &self.state.buffers,
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
            &self.state,
            &self.state.devices,
            &self.state.buffers,
            device_handle,
            buffer_handle,
            new_logical_size,
        )
    }

    fn hint_buffer_unused_above(&mut self, buffer_handle: BufferHandle, offset: u64) {
        buffer::hint_unused_above(&self.state.devices, &self.state.buffers, buffer_handle, offset);
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
            &self.state.devices,
            &self.state.buffers,
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
            &self.state,
            &self.state.instance,
            &self.state.devices,
            &self.state.buffers,
            device_handle,
            buffer_handle,
            new_size,
            preserve_contents,
        )
    }

    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle> {
        buffer::alloc_readback_buffer(
            &self.state.instance,
            &self.state.devices,
            &self.state.buffers,
            device,
            size,
        )
    }

    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        buffer::read_readback_buffer(&self.state.buffers, buffer, output)
    }

    fn free_readback_buffer(&mut self, buffer: BufferHandle) {
        buffer::destroy(&self.state, buffer);
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
        self.state
            .textures
            .read()
            .unwrap()
            .entries
            .get(&texture)
            .map(|t| t.image_layout().as_raw() as u64)
            .unwrap_or(0)
    }

    fn alloc_texture_readback_staging(
        &mut self,
        device: DeviceHandle,
        layout: crate::backend::TextureCopyFootprint,
    ) -> Result<BufferHandle> {
        buffer::alloc_texture_readback_staging(
            &self.state.instance,
            &self.state.devices,
            &self.state.buffers,
            device,
            layout,
        )
    }

    fn read_texture_readback_staging(
        &self,
        buffer: BufferHandle,
        layout: crate::backend::TextureCopyFootprint,
        output: &mut [u8],
    ) -> Result<()> {
        buffer::read_texture_readback_staging(&self.state.buffers, buffer, layout, output)
    }

    fn clear_buffer(
        &mut self,
        device_handle: DeviceHandle,
        buffer_handle: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        buffer::clear(
            &self.state.devices,
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
            &self.state.shaders,
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
        shader::destroy(&self.state.devices, &self.state.shaders, shader_handle);
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
        let vs_module = self.ensure_shader_stage_compiled(vertex_shader, crate::slang::SlangStage::Vertex)?;
        let fs_module = self.ensure_shader_stage_compiled(fragment_shader, crate::slang::SlangStage::Fragment)?;

        let (cats, strides) = render_reflection_data(
            &self.state.shaders.read().unwrap().entries,
            vertex_shader,
            fragment_shader,
        );
        let shader_debug_name = format!("shader(vs=#{vertex_shader}, fs=#{fragment_shader})");

        let raster = crate::backend::shared::PipelineDesc::new(vertex_layout, topology, target_format);
        let handle = pipeline::create(pipeline::VulkanGraphicsPipelineCreateBundle {
            devices: &self.state.devices,
            pipelines: &self.state.pipelines,
            device_handle,
            vs_module,
            fs_module,
            raster: &raster,
            shader_debug_name,
        })?;

        if let Some(ps) = self.state.pipelines.write().unwrap().entries.get_mut(&handle) {
            ps.push_constant_categories = cats;
            ps.binding_element_strides = strides;
        }
        Ok(handle)
    }

    fn destroy_pipeline(&mut self, pipeline_handle: PipelineHandle) {
        pipeline::destroy(&self.state.devices, &self.state.pipelines, pipeline_handle);
    }

    fn render_to_target(
        &mut self,
        device_handle: DeviceHandle,
        target: RenderTargetHandle,
        color_load: crate::types::TargetLoad,
        commands: &[RenderCommand],
    ) -> Result<()> {
        let instance = self.state.instance.clone();
        let frame_table = frame_table::ensure_legacy_frame_table(&mut self.state, &instance, device_handle)?;
        let render_resources = render_target::RenderToResources {
            devices: &self.state.devices,
            frame_table: &frame_table,
            buffers: &self.state.buffers,
            pipelines: &self.state.pipelines,
        };
        render_target::render_to(
            render_resources,
            &self.state.render_targets,
            device_handle,
            target,
            color_load,
            commands,
            |cmd, cmds, logical_device, current_pipeline| {
                let pipelines_read = self.state.pipelines.read().unwrap();
                let buffers_read = self.state.buffers.read().unwrap();
                render_commands::record(
                    cmd,
                    cmds,
                    logical_device,
                    &pipelines_read.entries,
                    &buffers_read.entries,
                    current_pipeline,
                    (frame_table.selector_slot, frame_table.table_slot),
                )
            },
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
            &self.state.textures,
            &mut self.state.next_surface_handle,
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
        ctx: ContextHandle,
    ) -> Result<(FrameToken, TextureHandle)> {
        let (image, present_slot) = surface::acquire(&mut self.state, surface_handle, ctx)?;
        let tex = surface::frame_texture(&self.state.surfaces, surface_handle)
            .context("begin_frame: surface frame texture unavailable")?;
        Ok((
            FrameToken {
                surface: surface_handle,
                image,
                context: ctx,
                frame_slot: present_slot,
                present_slot,
            },
            tex,
        ))
    }

    fn surface_resize(&mut self, surface_handle: SurfaceHandle, width: u32, height: u32) -> Result<()> {
        surface::resize(
            &self.state.entry,
            &self.state.instance,
            &self.state.devices,
            &mut self.state.surfaces,
            &self.state.textures,
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
        surface::set_present_mode(&mut self.state, surface_handle, mode)
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
        let vs_module = self.ensure_shader_stage_compiled(vertex_shader, crate::slang::SlangStage::Vertex)?;
        let fs_module = self.ensure_shader_stage_compiled(fragment_shader, crate::slang::SlangStage::Fragment)?;

        let shader_debug_name = format!("shader(vs=#{vertex_shader}, fs=#{fragment_shader})");

        let (cats, strides) = render_reflection_data(
            &self.state.shaders.read().unwrap().entries,
            vertex_shader,
            fragment_shader,
        );

        let raster = crate::backend::shared::PipelineDesc::new(vertex_layout, topology, target_format)
            .with_depth_stencil(depth_stencil);
        let handle = pipeline::create_with_depth(pipeline::VulkanGraphicsPipelineCreateBundle {
            devices: &self.state.devices,
            pipelines: &self.state.pipelines,
            device_handle,
            vs_module,
            fs_module,
            raster: &raster,
            shader_debug_name,
        })?;

        if let Some(ps) = self.state.pipelines.write().unwrap().entries.get_mut(&handle) {
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
            &self.state.render_targets,
            device_handle,
            width,
            height,
            color_format,
            depth_format,
        )
    }

    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        render_target::destroy(&self.state.devices, &self.state.render_targets, target);
    }

    fn create_texture(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<TextureHandle> {
        texture::create(
            &self.state.instance,
            &self.state.devices,
            &self.state.textures,
            device_handle,
            width,
            height,
            format,
            access,
            flags,
        )
    }

    fn write_texture(&mut self, texture_handle: TextureHandle, data: &[u8], width: u32, height: u32) -> Result<()> {
        texture::write(
            &self.state.instance,
            &self.state.devices,
            &self.state.textures,
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
            &self.state.textures,
            texture_handle,
            x,
            y,
            width,
            height,
            data,
        )
    }

    fn destroy_texture(&mut self, texture_handle: TextureHandle) {
        texture::destroy(&self.state, &self.state.devices, &self.state.textures, texture_handle);
    }

    fn set_texture_debug_name(&mut self, handle: TextureHandle, name: &str) {
        if !self.state.enable_validation {
            return;
        }
        texture::set_debug_name(
            &self.state.instance,
            &self.state.devices,
            &self.state.textures,
            handle,
            name,
        );
    }

    fn texture_bindless_index(&self, texture_handle: TextureHandle) -> Option<u32> {
        texture::bindless_index(&self.state.textures, texture_handle)
    }

    fn texture_bindless_sampled_index(&self, texture_handle: TextureHandle) -> Option<u32> {
        texture::bindless_sampled_index(&self.state.textures, texture_handle)
    }

    fn create_sampler(
        &mut self,
        device_handle: DeviceHandle,
        desc: &crate::types::SamplerDesc,
    ) -> Result<SamplerHandle> {
        sampler::create(&self.state.devices, &self.state.samplers, device_handle, desc)
    }

    fn destroy_sampler(&mut self, sampler_handle: SamplerHandle) {
        sampler::destroy(&self.state.devices, &self.state.samplers, sampler_handle);
    }

    fn sampler_bindless_index(&self, sampler_handle: SamplerHandle) -> Option<u32> {
        sampler::bindless_index(&self.state.samplers, sampler_handle)
    }

    fn create_compute_pipeline(
        &mut self,
        device_handle: DeviceHandle,
        compute_shader: ShaderHandle,
        debug_name: Option<&str>,
    ) -> Result<ComputePipelineHandle> {
        // Compile shader on-demand
        let cs_module = {
            let _st = crate::shader_timing::scope("vk.ensure_stage_compiled", debug_name.unwrap_or(""));
            self.ensure_shader_stage_compiled(compute_shader, crate::slang::SlangStage::Compute)?
        };

        let (cats, strides) = {
            let shaders = self.state.shaders.read().unwrap();
            shaders
                .entries
                .get(&compute_shader)
                .and_then(|s| s.reflection.as_ref())
                .map(|r| (r.push_constant_categories.clone(), r.binding_element_strides.clone()))
                .unwrap_or_default()
        };

        let shader_debug_name = debug_name
            .map(str::to_owned)
            .unwrap_or_else(|| format!("compute_shader#{compute_shader}"));

        let handle = compute::create(
            &self.state.devices,
            &self.state.compute_pipelines,
            device_handle,
            cs_module,
            shader_debug_name,
        )?;

        if let Some(ps) = self.state.compute_pipelines.write().unwrap().entries.get_mut(&handle) {
            ps.push_constant_categories = cats;
            ps.binding_element_strides = strides;
        }
        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline_handle: ComputePipelineHandle) {
        compute::destroy(&self.state.devices, &self.state.compute_pipelines, pipeline_handle);
    }

    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue {
        let _tz = crate::tracy_zone!("vk.gpu_progress");
        let contexts = self.state.contexts.read().unwrap();
        let Some(sc_arc) = contexts.get(&ctx).cloned() else {
            return 0;
        };
        let sc = sc_arc.lock().unwrap();
        let Some(ld) = self.state.devices.get(&sc.device) else {
            return 0;
        };
        unsafe { ld.device.get_semaphore_counter_value(sc.timeline_semaphore) }.unwrap_or(0)
    }

    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue {
        context::device_retired(&self.state, device)
    }

    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> anyhow::Result<()> {
        if let Some(ld) = self.state.devices.get(&device) {
            ld.submission_worker.flush()?;
            let horizon = crate::backend::submission_worker::submission_horizon(&ld.timeline_next);
            if value <= horizon {
                ld.submission_worker.wait_submitted(value)?;
            }
        }
        context::wait_until_device_seq_at_least(&self.state, device, value);
        self.with_validation(Ok(()))
    }

    fn poll_signals(
        &mut self,
        ctx: ContextHandle,
        progress: crate::timeline::TimelineValue,
    ) -> Vec<crate::signal::QueuedSignal> {
        let device_handle = self.context_device(ctx);
        let signal_queue = self
            .state
            .contexts
            .read()
            .unwrap()
            .get(&ctx)
            .map(|sc| std::sync::Arc::clone(&sc.lock().unwrap().signal_queue));
        let Some(signal_queue) = signal_queue else {
            return Vec::new();
        };
        for surface in self.state.surfaces.values_mut() {
            if surface.device_handle != device_handle {
                continue;
            }
            surface.pending_swapchain_returns.retain(|&(idx, tv)| {
                if progress >= tv {
                    signal_queue.push(crate::signal::Signal::SwapchainReturned { image_index: idx });
                    surface.pending_acquire_count = surface.pending_acquire_count.saturating_sub(1);
                    false
                } else {
                    true
                }
            });
        }
        crate::signal::drain_all_queued_signals(&signal_queue)
    }

    fn submit_standalone(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        self.with_validation(compute::submit(&self.state, ctx, commands, sync))
    }

    fn submit_graph(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        self.with_validation(compute::submit_graph(&self.state, ctx, commands, sync))
    }

    fn submit_graph_and_retain(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        self.with_validation(compute::submit_graph_and_retain(&self.state, ctx, commands, key, sync))
    }

    fn try_resubmit_retained(
        &mut self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<crate::timeline::TimelineValue>> {
        self.with_validation(compute::try_resubmit_retained(&self.state, ctx, key, sync))
    }

    fn evict_retained(&mut self, ctx: ContextHandle, key: u64) {
        compute::evict_retained(&self.state, ctx, key);
    }

    fn submit_frame(&mut self, frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        let r = surface::submit_frame(&mut self.state, frame);
        debug_utils::combine_validation(self.state.validation_sink.as_ref(), r)
    }

    fn reset_buffer_heaps(&mut self, device_handle: DeviceHandle) {
        let Some(logical_device) = self.state.devices.get(&device_handle) else {
            return;
        };
        for sc_arc in self.state.contexts.read().unwrap().values() {
            let mut sc = sc_arc.lock().unwrap();
            if sc.device == device_handle {
                unsafe { sc.staging_belt.trim(logical_device) };
            }
        }
    }

    fn available_bindless_slots(&self, device_handle: DeviceHandle, category: crate::types::ResourceCategory) -> u32 {
        self.state
            .devices
            .get(&device_handle)
            .map(|ld| {
                ld.descriptors
                    .lock()
                    .unwrap()
                    .resource_registry
                    .available_slots(category)
            })
            .unwrap_or(0)
    }

    fn max_bindless_slots_per_category(
        &self,
        _device_handle: DeviceHandle,
        _category: crate::types::ResourceCategory,
    ) -> u32 {
        types::MAX_BINDLESS_RESOURCES
    }

    fn max_submission_contexts(&self, device_handle: DeviceHandle) -> u32 {
        self.state
            .devices
            .get(&device_handle)
            .map(|ld| ld.compute_queues.len() as u32)
            .unwrap_or(0)
    }

    fn deferred_deletion_pending_count(&self, ctx: ContextHandle) -> usize {
        self.state
            .contexts
            .read()
            .unwrap()
            .get(&ctx)
            .map(|sc| sc.lock().unwrap().deletion_queue.len())
            .unwrap_or(0)
    }

    fn device_deferred_deletion_pending_count(&self, device: DeviceHandle) -> usize {
        self.state
            .devices
            .get(&device)
            .map(|ld| ld.deletion_queue.lock().unwrap().pending_len())
            .unwrap_or(0)
    }
}

struct VulkanTimelineBlockingWait {
    device: ash::Device,
    semaphore: vk::Semaphore,
    value: crate::timeline::TimelineValue,
    device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl crate::backend::TimelineBlockingWait for VulkanTimelineBlockingWait {
    fn block(self: Box<Self>) -> Result<()> {
        let _wait = crate::tracy_zone!("vk.wait_until.wait_semaphores");
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&self.semaphore))
            .values(std::slice::from_ref(&self.value));
        if let Err(e) = unsafe { self.device.wait_semaphores(&wait, u64::MAX) } {
            if e == vk::Result::ERROR_DEVICE_LOST {
                self.device_lost.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            anyhow::bail!("wait_semaphores: {:?}", e);
        }
        Ok(())
    }

    fn block_timeout(self: Box<Self>, timeout_ms: u32) -> Result<bool> {
        let _wait = crate::tracy_zone!("vk.wait_until.wait_semaphores");
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&self.semaphore))
            .values(std::slice::from_ref(&self.value));
        let timeout_ns = (timeout_ms as u64).saturating_mul(1_000_000);
        match unsafe { self.device.wait_semaphores(&wait, timeout_ns) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => {
                if e == vk::Result::ERROR_DEVICE_LOST {
                    self.device_lost.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                Err(anyhow::anyhow!("wait_semaphores: {:?}", e))
            }
        }
    }
}

struct VulkanContextGpuProgress {
    sc: types::SharedSubmissionContext,
    ld: types::SharedLogicalDevice,
}

impl crate::backend::ContextGpuProgress for VulkanContextGpuProgress {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue {
        let sc = self.sc.lock().unwrap();
        unsafe { self.ld.device.get_semaphore_counter_value(sc.timeline_semaphore) }.unwrap_or(0)
    }
}

struct VulkanContextDeferredDeletionFlush {
    ctx: ContextHandle,
    sc: types::SharedSubmissionContext,
    ld: types::SharedLogicalDevice,
    contexts: types::SharedContextMap,
    device_handle: DeviceHandle,
}

impl crate::backend::ContextDeferredDeletionFlush for VulkanContextDeferredDeletionFlush {
    fn flush(&self) {
        let completed_values =
            types::snapshot_context_completed_values(&self.ld.device, &self.contexts, self.device_handle);
        let completed = completed_values.get(&self.ctx).copied().unwrap_or(0);
        let ctx_batch: Vec<_> = self.sc.lock().unwrap().deletion_queue.drain_up_to(completed);
        let descriptors_arc = std::sync::Arc::clone(&self.ld.descriptors);
        {
            let mut registry = descriptors_arc.lock().unwrap();
            for r in ctx_batch {
                types::destroy_pending_deletion(&self.ld, &mut registry, r);
            }
            registry.drain_ready_slot_reclamations(&completed_values);
        }
        self.ld.process_deletion_queue_up_to(&completed_values);
    }
}

impl crate::backend::GpuBackendSubmitSession for VulkanBackend {
    fn clone_context_submit_session(
        &self,
        ctx: ContextHandle,
        _backend: std::sync::Arc<std::sync::Mutex<Box<dyn crate::backend::GpuBackend>>>,
    ) -> std::sync::Arc<dyn crate::backend::ContextSubmitSession> {
        submit_session::VulkanSubmitSession::clone_from_state(&self.state, ctx)
            .unwrap_or_else(|e| panic!("clone_context_submit_session({ctx}): {e:#}"))
    }
}

#[cfg(test)]
mod validation_fatal_tests {
    use super::*;
    use ash::vk;

    /// True when this process would select Vulkan as the Goldy backend.
    ///
    /// Non-Vulkan CI jobs (Metal, DX12, WebGPU) compile `feature = "vulkan"` but
    /// must not spawn a Khronos subprocess.
    fn selected_backend_is_vulkan() -> bool {
        if let Ok(s) = std::env::var("GOLDY_BACKEND") {
            return matches!(s.to_ascii_lowercase().as_str(), "vulkan" | "vk");
        }
        // Match `create_default_backend()`: macOS→Metal, Windows→DX12, else Vulkan if compiled.
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        {
            false
        }
        #[cfg(all(
            feature = "dx12",
            target_os = "windows",
            not(all(feature = "metal", any(target_os = "macos", target_os = "ios")))
        ))]
        {
            false
        }
        #[cfg(not(any(
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            all(feature = "dx12", target_os = "windows")
        )))]
        {
            cfg!(feature = "vulkan")
        }
    }

    #[test]
    fn vk_validation_fatal_zero_size_buffer() {
        if std::env::var("GOLDY_SUBPROC").is_err() {
            if !selected_backend_is_vulkan() {
                return;
            }
            let test_name = std::thread::current()
                .name()
                .expect("cargo test thread name")
                .to_string();
            let exe = std::env::current_exe().expect("current_exe");
            let output = std::process::Command::new(exe)
                .args([&test_name, "--exact", "--nocapture"])
                .env("GOLDY_SUBPROC", "1")
                .env("GOLDY_VALIDATION", "api")
                .env("GOLDY_VALIDATION_FATAL", "1")
                .env("GOLDY_BACKEND", "vk")
                .env_remove("VK_LAYER_PATH")
                .output()
                .expect("spawn validation-fatal subprocess");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{stdout}\n{stderr}");
            if combined.contains("GOLDY_SKIP_VK_VALIDATION_FATAL") {
                return;
            }
            assert!(
                output.status.success(),
                "subprocess failed (exit {:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status.code()
            );
            assert!(
                combined.contains("GOLDY_VALIDATION_FATAL") || combined.contains("VUID"),
                "expected validation fatal / VUID text\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            return;
        }

        let mut backend = match VulkanBackend::new() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("GOLDY_SKIP_VK_VALIDATION_FATAL: {e:#}");
                return;
            }
        };
        assert!(
            backend.state.enable_validation,
            "subprocess should have enabled the debug messenger"
        );
        let handle = device::create(&mut backend.state, 0).expect("create logical device");
        let ash_dev = backend.state.devices.get(&handle).unwrap().device.clone();
        let info = vk::BufferCreateInfo::default()
            .size(0)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let created = unsafe { ash_dev.create_buffer(&info, None) };
        eprintln!("vkCreateBuffer(size=0) => {created:?}");
        if let Ok(buf) = created {
            unsafe {
                ash_dev.destroy_buffer(buf, None);
            }
        }
        device::destroy(&mut backend.state, handle);
        let err = backend
            .with_validation(Ok(()))
            .expect_err("zero-size VkBuffer should record a fatal validation ERROR");
        let msg = format!("{err:#}");
        println!("{msg}");
        assert!(msg.contains("GOLDY_VALIDATION_FATAL") || msg.contains("VUID"), "{msg}");
    }
}
