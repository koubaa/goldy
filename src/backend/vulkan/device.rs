//! Device management logic.

use super::types::{self, PhysicalDeviceInfo};
use super::{DeviceHandle, VulkanState};
use crate::backend::{AdapterInfo, BackendType, DeviceType};
use anyhow::{Context, Result};
use ash::vk;
use ash::{ext, khr};
use std::ffi::CStr;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

/// Enumerate available physical devices/adapters.
pub(super) fn enumerate(physical_devices: &[PhysicalDeviceInfo]) -> Vec<AdapterInfo> {
    physical_devices
        .iter()
        .map(|dev| {
            let name = unsafe { CStr::from_ptr(dev.properties.device_name.as_ptr()) };
            let device_type = match dev.properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::DiscreteGpu,
                vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::IntegratedGpu,
                vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
                _ => DeviceType::Other,
            };

            let vendor = match dev.properties.vendor_id {
                0x1002 | 0x1022 => "AMD",
                0x10DE => "NVIDIA",
                0x8086 => "Intel",
                0x13B5 => "ARM",
                0x5143 => "Qualcomm",
                0x106B => "Apple",
                _ => "Unknown",
            };

            AdapterInfo {
                id: dev.adapter_id,
                name: name.to_string_lossy().into_owned(),
                vendor: vendor.to_string(),
                backend: BackendType::Vulkan,
                device_type,
            }
        })
        .collect()
}

/// Build the public capability snapshot for a physical adapter.
pub(super) fn adapter_capabilities(
    physical_devices: &[PhysicalDeviceInfo],
    adapter_id: u32,
) -> crate::device::DeviceCapabilities {
    let mut caps = crate::device::DeviceCapabilities::default();
    if physical_devices
        .iter()
        .find(|d| d.adapter_id == adapter_id)
        .is_some_and(|d| d.supports_sparse_buffer)
    {
        caps.buffer_resize_cost = crate::types::BufferResizeCost::PageBind;
        caps.buffer_page_size = 64 * 1024; // 64 KiB — universal sparse granularity; asserted in device::create
        caps.buffer_decommit_supported = true;
    }
    caps
}

/// Create a logical device from a physical device adapter ID.
#[allow(clippy::too_many_lines)]
pub(super) fn create(state: &mut VulkanState, adapter_id: u32) -> Result<DeviceHandle> {
    let physical_device = state
        .physical_devices
        .iter()
        .find(|d| d.adapter_id == adapter_id)
        .context("Invalid adapter ID")?;

    let physical_device_handle = physical_device.handle;

    // Find a graphics queue family
    let queue_families = unsafe {
        state
            .instance
            .get_physical_device_queue_family_properties(physical_device_handle)
    };

    let queue_family_index = queue_families
        .iter()
        .enumerate()
        .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|(idx, _)| idx as u32)
        .context("No graphics queue family found")?;

    let pdev_features = unsafe { state.instance.get_physical_device_features(physical_device_handle) };
    let supports_sparse = physical_device.supports_sparse_buffer;
    debug_assert_eq!(
        supports_sparse,
        pdev_features.sparse_binding != 0 && pdev_features.sparse_residency_buffer != 0,
        "PhysicalDeviceInfo sparse flag out of sync with live query"
    );

    let available_device_exts: std::collections::HashSet<String> = unsafe {
        state
            .instance
            .enumerate_device_extension_properties(physical_device_handle)
    }
    .unwrap_or_default()
    .into_iter()
    .map(|ext| {
        unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    })
    .collect();

    const KHR_COMPUTE_DERIVATIVES: &CStr = c"VK_KHR_compute_shader_derivatives";
    const NV_COMPUTE_DERIVATIVES: &CStr = c"VK_NV_compute_shader_derivatives";
    let has_khr_compute_derivatives = available_device_exts.contains("VK_KHR_compute_shader_derivatives");
    let has_nv_compute_derivatives = available_device_exts.contains("VK_NV_compute_shader_derivatives");
    let supports_compute_derivative_quads = if has_khr_compute_derivatives || has_nv_compute_derivatives {
        let mut supported_compute_derivatives = vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default();
        let mut supported_features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut supported_compute_derivatives);
        unsafe {
            state
                .instance
                .get_physical_device_features2(physical_device_handle, &mut supported_features2);
        }
        supported_compute_derivatives.compute_derivative_group_quads != vk::FALSE
    } else {
        false
    };

    let sparse_queue_family_index = if supports_sparse {
        queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING))
            .map(|(idx, _)| idx as u32)
            .context("sparse features enabled but no queue family reports SPARSE_BINDING")?
    } else {
        queue_family_index
    };

    // Verify this physical device supports Vulkan 1.4
    let dev_api = physical_device.properties.api_version;
    let dev_major = vk::api_version_major(dev_api);
    let dev_minor = vk::api_version_minor(dev_api);
    if dev_major < 1 || (dev_major == 1 && dev_minor < 4) {
        let name = unsafe { CStr::from_ptr(physical_device.properties.device_name.as_ptr()) };
        anyhow::bail!(
            "Adapter {} reports Vulkan {}.{}, but Goldy requires 1.4+",
            name.to_string_lossy(),
            dev_major,
            dev_minor
        );
    }

    // Vulkan 1.2 features: descriptor indexing for the global bindless set.
    // dynamicRendering and synchronization2 are mandatory in 1.3+ (guaranteed by 1.4).
    // Descriptor indexing sub-features are still optional; we request them here.
    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
        // Device timeline semaphore (`VkSemaphoreType::TIMELINE`) for `gpu_progress` / deferred destroy.
        .timeline_semaphore(true)
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_sampled_image_update_after_bind(true)
        .descriptor_binding_storage_buffer_update_after_bind(true)
        .descriptor_binding_storage_image_update_after_bind(true)
        .descriptor_binding_uniform_buffer_update_after_bind(true)
        .runtime_descriptor_array(true)
        .shader_storage_buffer_array_non_uniform_indexing(true)
        .shader_sampled_image_array_non_uniform_indexing(true)
        .shader_uniform_buffer_array_non_uniform_indexing(true)
        // Required for SPIR-V Float16 capability used by ekrano shaders.
        .shader_float16(true)
        // Required for SPIR-V Int8 capability (enabled alongside float16 in Vulkan 1.2).
        .shader_int8(true);

    // Vulkan 1.1 features: shaderDrawParameters is needed for SV_InstanceID
    // (SPIR-V DrawParameters capability) in vertex shaders.
    let mut vulkan_11_features = vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);

    // Vulkan 1.3 features: mandatory in 1.4, but must still be enabled.
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);

    // Pipeline robustness (core in Vulkan 1.4): enables per-pipeline OOB safety for bindless.
    let mut pipeline_robustness_features =
        vk::PhysicalDevicePipelineRobustnessFeaturesEXT::default().pipeline_robustness(true);

    // Texture sampling in compute shaders requires Slang to emit SPV_KHR_compute_shader_derivatives.
    let mut compute_derivatives_features =
        vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default().compute_derivative_group_quads(true);

    let core_features = vk::PhysicalDeviceFeatures {
        vertex_pipeline_stores_and_atomics: vk::TRUE,
        fragment_stores_and_atomics: vk::TRUE,
        shader_int16: vk::TRUE,
        shader_int64: vk::TRUE,
        sparse_binding: if supports_sparse { vk::TRUE } else { vk::FALSE },
        sparse_residency_buffer: if supports_sparse { vk::TRUE } else { vk::FALSE },
        ..Default::default()
    };

    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(core_features)
        .push_next(&mut vulkan_11_features)
        .push_next(&mut vulkan_13_features)
        .push_next(&mut vulkan_12_features)
        .push_next(&mut pipeline_robustness_features);

    if supports_compute_derivative_quads {
        features2 = features2.push_next(&mut compute_derivatives_features);
    }

    // Create logical device with swapchain extension
    let mut queue_family_set = std::collections::BTreeSet::new();
    queue_family_set.insert(queue_family_index);
    if supports_sparse {
        queue_family_set.insert(sparse_queue_family_index);
    }

    let queue_priorities = [1.0f32];
    let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = queue_family_set
        .iter()
        .map(|&family| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&queue_priorities)
        })
        .collect();
    // Extensions: swapchain for presentation, plus 1.4-promoted extensions whose KHR
    // variants must still be requested, plus optional compute-derivatives support.
    let mut device_extensions = vec![
        khr::swapchain::NAME.as_ptr(),
        khr::map_memory2::NAME.as_ptr(),
        ext::pipeline_robustness::NAME.as_ptr(),
    ];
    if supports_compute_derivative_quads && has_khr_compute_derivatives {
        device_extensions.push(KHR_COMPUTE_DERIVATIVES.as_ptr());
    }
    if supports_compute_derivative_quads && has_nv_compute_derivatives {
        device_extensions.push(NV_COMPUTE_DERIVATIVES.as_ptr());
    }

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions)
        .push_next(&mut features2);

    let device = unsafe {
        state
            .instance
            .create_device(physical_device_handle, &device_create_info, None)
    }
    .context("Failed to create logical device")?;

    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let sparse_binding_queue = if supports_sparse {
        unsafe { device.get_device_queue(sparse_queue_family_index, 0) }
    } else {
        vk::Queue::default()
    };

    let (sparse_buffer_block_size, sparse_memory_type_index, sparse_page_pool) = if supports_sparse {
        let bs = super::sparse::query_sparse_buffer_block_size(&device).context("query_sparse_buffer_block_size")?;
        let (mt_idx, _) = super::sparse::sparse_storage_memory_type(&state.instance, physical_device_handle, &device)
            .context("sparse_storage_memory_type")?;
        (bs, mt_idx, Some(super::sparse::SparsePagePool::new(bs, mt_idx)))
    } else {
        (0u64, 0u32, None)
    };
    if supports_sparse {
        debug_assert_eq!(
            sparse_buffer_block_size,
            64 * 1024,
            "sparse_buffer_block_size deviates from the 64 KiB assumed by DeviceCapabilities::buffer_page_size"
        );
    }

    // Load Vulkan 1.4 core APIs via KHR extension loaders (ash 0.38 predates 1.4 headers).
    // On a 1.4 device these functions are core — the KHR entry points are aliases.
    let map_memory2_loader = ash::khr::map_memory2::Device::new(&state.instance, &device);

    // Create command pool
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    let command_pool =
        unsafe { device.create_command_pool(&pool_info, None) }.context("Failed to create command pool")?;

    // Create descriptor infrastructure for resource binding
    let (bindless_descriptor_pool, bindless_descriptor_set_layout, bindless_descriptor_set, bindless_pipeline_layout) = {
        // Create descriptor set layout with update-after-bind flag
        // Bindings organized by ACCESS PATTERN (see types.rs::bindless_bindings)
        let binding_flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        ];

        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let bindings = [
            // Binding 0: Scattered buffer access (read/write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(types::bindless_bindings::STORAGE_BUFFERS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(types::MAX_BINDLESS_RESOURCES)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // Binding 1: Broadcast buffer access (read-only uniforms)
            vk::DescriptorSetLayoutBinding::default()
                .binding(types::bindless_bindings::UNIFORM_BUFFERS)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(types::MAX_BINDLESS_RESOURCES)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // Binding 2: Filtered image reads
            vk::DescriptorSetLayoutBinding::default()
                .binding(types::bindless_bindings::SAMPLED_IMAGES)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(types::MAX_BINDLESS_RESOURCES)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // Binding 3: Unfiltered image access (read/write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(types::bindless_bindings::STORAGE_IMAGES)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(types::MAX_BINDLESS_RESOURCES)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // Binding 4: Filter configuration
            vk::DescriptorSetLayoutBinding::default()
                .binding(types::bindless_bindings::SAMPLERS)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(types::MAX_BINDLESS_RESOURCES)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut binding_flags_info);

        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
            .context("Failed to create bindless descriptor set layout")?;

        // Create descriptor pool with update-after-bind flag
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: types::MAX_BINDLESS_RESOURCES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: types::MAX_BINDLESS_RESOURCES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: types::MAX_BINDLESS_RESOURCES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: types::MAX_BINDLESS_RESOURCES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: types::MAX_BINDLESS_RESOURCES,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .context("Failed to create bindless descriptor pool")?;

        // Allocate the global descriptor set
        let set_layouts = [descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .context("Failed to allocate bindless descriptor set")?;

        let descriptor_set = descriptor_sets[0];

        // Create a pipeline layout that includes the bindless set and resource slots
        let layouts = [descriptor_set_layout];

        // Vulkan push constant range for the packed 128-byte PushLayout
        let slot_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::ALL,
            offset: 0,
            size: types::TOTAL_PUSH_BYTES as u32,
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&slot_range));

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
            .context("Failed to create bindless pipeline layout")?;

        tracing::info!(
            "Pipeline layout includes {} bytes of push constants for resource slot indices",
            slot_range.size
        );

        tracing::info!("Created descriptor infrastructure: pool, layout, set, pipeline layout");

        (
            Some(descriptor_pool),
            Some(descriptor_set_layout),
            Some(descriptor_set),
            Some(pipeline_layout),
        )
    };

    let initial_pipeline_cache_bytes = dirs::cache_dir()
        .map(|d| d.join("goldy").join(format!("pipeline_cache_{adapter_id}.bin")))
        .and_then(|path| std::fs::read(path).ok())
        .unwrap_or_default();
    let pipeline_cache_ci = vk::PipelineCacheCreateInfo::default().initial_data(&initial_pipeline_cache_bytes);
    let pipeline_cache = unsafe { device.create_pipeline_cache(&pipeline_cache_ci, None) }
        .context("Failed to create VkPipelineCache")?;

    let handle = state.next_device_handle;
    state.next_device_handle += 1;

    state.devices.insert(
        handle,
        Arc::new(types::LogicalDevice {
            device,
            physical_device: physical_device_handle,
            adapter_id,
            queue,
            queue_family: queue_family_index,
            sparse_binding_queue,
            command_pool,
            supports_sparse_buffer: supports_sparse,
            sparse_buffer_block_size,
            sparse_memory_type_index,
            sparse_page_pool: Mutex::new(sparse_page_pool),
            map_memory2: map_memory2_loader,
            bindless_descriptor_pool,
            bindless_descriptor_set_layout,
            bindless_descriptor_set,
            bindless_pipeline_layout,
            descriptors: Arc::new(Mutex::new(types::DescriptorRegistry::new())),
            deletion_queue: Mutex::new(types::DeletionQueue::new()),
            timeline_next: Arc::new(AtomicU64::new(1)),
            retired_floor: AtomicU64::new(0),
            queue_lock: Arc::new(Mutex::new(())),
            pipeline_cache,
            vk_timestamp_compute_and_graphics: physical_device.vk_timestamp_compute_and_graphics,
            vk_timestamp_period_ns: physical_device.vk_timestamp_period_ns,
            submission_worker: Arc::new(crate::backend::submission_worker::SubmissionWorker::new(
                crate::backend::submission_worker::SUBMISSION_QUEUE_CAPACITY,
            )),
        }),
    );

    tracing::info!("Created Vulkan device {} for adapter {}", handle, adapter_id);
    if let Some(ld) = state.devices.get(&handle) {
        super::frame_table::reserve_device_bindless_slots(ld);
    }
    Ok(handle)
}

/// Destroy a logical device and all resources associated with it.
#[allow(clippy::too_many_lines)]
pub(super) fn destroy(state: &mut VulkanState, device_handle: DeviceHandle) {
    tracing::info!(
        %device_handle,
        global_devices = state.devices.len(),
        buffers = state.buffers.read().unwrap().entries.len(),
        shaders = state.shaders.read().unwrap().entries.len(),
        graphics_pipelines = state.pipelines.read().unwrap().entries.len(),
        compute_pipelines = state.compute_pipelines.read().unwrap().entries.len(),
        render_targets = state.render_targets.read().unwrap().entries.len(),
        textures = state.textures.read().unwrap().entries.len(),
        samplers = state.samplers.read().unwrap().entries.len(),
        "destroying Vulkan device"
    );
    if let Some(logical_device) = state.devices.remove(&device_handle) {
        let wait_result = logical_device.synchronized_device_wait_idle();

        unsafe {
            // When the device is lost, individual Vulkan destroy calls are unsafe
            // (driver bookkeeping is already corrupt). Per spec, vkDestroyDevice is
            // always valid and implicitly reclaims all child objects, so skip
            // individual cleanup and jump straight to it.
            if matches!(wait_result, Err(vk::Result::ERROR_DEVICE_LOST)) {
                let pending = logical_device.deletion_queue.lock().unwrap().len();
                tracing::warn!(
                    %device_handle,
                    pending_deferred = pending,
                    "lost Vulkan device — skipping per-object destroy, calling vkDestroyDevice only (driver may be in an invalid state)"
                );
                // Drop map entries without calling Vulkan (handles become invalid).
                state
                    .buffers
                    .write()
                    .unwrap()
                    .entries
                    .retain(|_, b| b.device_handle != device_handle);
                state
                    .shaders
                    .write()
                    .unwrap()
                    .entries
                    .retain(|_, s| s.device_handle != device_handle);
                state
                    .pipelines
                    .write()
                    .unwrap()
                    .entries
                    .retain(|_, p| p.device_handle != device_handle);
                state
                    .compute_pipelines
                    .write()
                    .unwrap()
                    .entries
                    .retain(|_, p| p.device_handle != device_handle);
                state
                    .render_targets
                    .write()
                    .unwrap()
                    .entries
                    .retain(|_, t| t.device_handle != device_handle);
                state
                    .textures
                    .write()
                    .unwrap()
                    .entries
                    .retain(|_, t| t.device_handle != device_handle);
                state
                    .samplers
                    .write()
                    .unwrap()
                    .entries
                    .retain(|_, s| s.device_handle != device_handle);
                state
                    .compute_fence_pool
                    .lock()
                    .unwrap()
                    .retain(|_, (dh, _, _)| *dh != device_handle);
                // Per-context staging belt and texture pool: device is being lost so just
                // drop entries without Vulkan destroy calls (handles are invalid after
                // device loss). Drop the entire context entries for this device.
                state
                    .contexts
                    .write()
                    .unwrap()
                    .retain(|_, sc| sc.lock().unwrap().device != device_handle);
                logical_device.device.destroy_device(None);
                return;
            }

            // Flush any pending deferred deletions via the new &self helper.
            logical_device.flush_deletion_queue();

            // Destroy per-context staging belt and texture pool for this device.
            // Command pools/semaphores in the context are intentionally NOT destroyed
            // here — they are child objects of the device and reclaimed by vkDestroyDevice.
            let ctx_keys: Vec<_> = state
                .contexts
                .read()
                .unwrap()
                .iter()
                .filter(|(_, sc)| sc.lock().unwrap().device == device_handle)
                .map(|(k, _)| *k)
                .collect();
            for key in ctx_keys {
                if let Some(sc_arc) = state.contexts.write().unwrap().remove(&key) {
                    let mut sc = sc_arc.lock().unwrap();
                    sc.staging_belt.destroy_all(&logical_device);
                    sc.texture_staging_pool.destroy_all(&logical_device);
                }
            }

            // Destroy buffers owned by this device
            let buffer_handles: Vec<_> = state
                .buffers
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, b)| b.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();

            for handle in buffer_handles {
                if let Some(buffer) = state.buffers.write().unwrap().entries.remove(&handle) {
                    if !buffer.is_view {
                        logical_device.device.destroy_buffer(buffer.buffer, None);
                        logical_device.device.free_memory(buffer.memory, None);
                        if let Some(staging) = buffer.staging_buffer {
                            logical_device.device.destroy_buffer(staging, None);
                        }
                        if let Some(staging_mem) = buffer.staging_memory {
                            logical_device.device.free_memory(staging_mem, None);
                        }
                    }
                }
            }

            // Destroy shaders owned by this device
            let shader_handles: Vec<_> = state
                .shaders
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in shader_handles {
                if let Some(shader) = state.shaders.write().unwrap().entries.remove(&handle) {
                    if let Some(module) = shader.vertex_module {
                        logical_device.device.destroy_shader_module(module, None);
                    }
                    if let Some(module) = shader.fragment_module {
                        logical_device.device.destroy_shader_module(module, None);
                    }
                    if let Some(module) = shader.compute_module {
                        logical_device.device.destroy_shader_module(module, None);
                    }
                }
            }

            // Destroy graphics pipelines owned by this device
            let pipeline_handles: Vec<_> = state
                .pipelines
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, p)| p.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in pipeline_handles {
                if let Some(pipeline) = state.pipelines.write().unwrap().entries.remove(&handle) {
                    if pipeline.pipeline != vk::Pipeline::null() {
                        logical_device.device.destroy_pipeline(pipeline.pipeline, None);
                    }
                    // Only destroy layout if we own it (not the global bindless layout)
                    if pipeline.owns_layout && pipeline.layout != vk::PipelineLayout::null() {
                        logical_device.device.destroy_pipeline_layout(pipeline.layout, None);
                    }
                }
            }

            // Destroy compute pipelines owned by this device
            let compute_pipeline_handles: Vec<_> = state
                .compute_pipelines
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, p)| p.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in compute_pipeline_handles {
                if let Some(pipeline) = state.compute_pipelines.write().unwrap().entries.remove(&handle) {
                    if pipeline.pipeline != vk::Pipeline::null() {
                        logical_device.device.destroy_pipeline(pipeline.pipeline, None);
                    }
                    // Only destroy layout if we own it (not the global bindless layout)
                    if pipeline.owns_layout && pipeline.layout != vk::PipelineLayout::null() {
                        logical_device.device.destroy_pipeline_layout(pipeline.layout, None);
                    }
                }
            }

            // Serialize pipeline cache after all pipelines referencing it are destroyed.
            let pipeline_cache_disk_path = dirs::cache_dir().map(|d| {
                d.join("goldy")
                    .join(format!("pipeline_cache_{}.bin", logical_device.adapter_id))
            });
            if logical_device.pipeline_cache != vk::PipelineCache::null() {
                match logical_device
                    .device
                    .get_pipeline_cache_data(logical_device.pipeline_cache)
                {
                    Ok(data) => {
                        if let Some(path) = pipeline_cache_disk_path.as_ref() {
                            if let Some(parent) = path.parent() {
                                let _ = std::fs::create_dir_all(parent);
                            }
                            if let Err(e) = std::fs::write(path, data) {
                                tracing::warn!(?e, path = ?path, "failed to write VkPipelineCache");
                            }
                        }
                    }
                    Err(e) => tracing::warn!(?e, "failed vkGetPipelineCacheData"),
                }
                logical_device
                    .device
                    .destroy_pipeline_cache(logical_device.pipeline_cache, None);
            }

            // Destroy render targets owned by this device
            let target_handles: Vec<_> = state
                .render_targets
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, t)| t.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in target_handles {
                if let Some(target) = state.render_targets.write().unwrap().entries.remove(&handle) {
                    logical_device.device.destroy_image_view(target.image_view, None);
                    logical_device.device.destroy_image(target.image, None);
                    logical_device.device.free_memory(target.image_memory, None);
                    // Clean up depth buffer if present
                    if let Some(depth_view) = target.depth_view {
                        logical_device.device.destroy_image_view(depth_view, None);
                    }
                    if let Some(depth_image) = target.depth_image {
                        logical_device.device.destroy_image(depth_image, None);
                    }
                    if let Some(depth_memory) = target.depth_memory {
                        logical_device.device.free_memory(depth_memory, None);
                    }
                    if let Some(staging_buffer) = target.staging_buffer {
                        logical_device.device.destroy_buffer(staging_buffer, None);
                    }
                    if let Some(staging_memory) = target.staging_memory {
                        logical_device.device.free_memory(staging_memory, None);
                    }
                }
            }

            // Destroy surfaces owned by this device before the generic texture loop.
            // A secondary `Device` clone (e.g. GoldyRenderer's tracked allocator device)
            // may drop before `Surface`, so this path must use the full `surface::destroy`
            // implementation (work_done semaphores, scratch-slot memory, command buffers).
            let surface_handles: Vec<_> = state
                .surfaces
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in surface_handles {
                super::surface::destroy_with_logical_device(
                    &state.entry,
                    &state.instance,
                    &logical_device,
                    &state.devices,
                    &mut state.surfaces,
                    &state.textures,
                    handle,
                    true,
                );
            }

            // Destroy textures owned by this device
            let texture_handles: Vec<_> = state
                .textures
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, t)| t.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in texture_handles {
                if let Some(texture) = state.textures.write().unwrap().entries.remove(&handle) {
                    logical_device.device.destroy_image_view(texture.view, None);
                    logical_device.device.destroy_image(texture.image, None);
                    logical_device.device.free_memory(texture.memory, None);
                    if let Some(staging_buffer) = texture.staging_buffer {
                        logical_device.device.destroy_buffer(staging_buffer, None);
                    }
                    if let Some(staging_memory) = texture.staging_memory {
                        logical_device.device.free_memory(staging_memory, None);
                    }
                }
            }

            // Destroy samplers owned by this device
            let sampler_handles: Vec<_> = state
                .samplers
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in sampler_handles {
                if let Some(sampler) = state.samplers.write().unwrap().entries.remove(&handle) {
                    logical_device.device.destroy_sampler(sampler.sampler, None);
                }
            }

            // Tests (and async dispatch patterns) may leave signaled-but-uncollected
            // fences in the pool; they must be destroyed before vkDestroyDevice.
            let fence_tokens: Vec<u64> = state
                .compute_fence_pool
                .lock()
                .unwrap()
                .iter()
                .filter(|(_, (dh, _, _))| *dh == device_handle)
                .map(|(tok, _)| *tok)
                .collect();
            let mut fence_pool = state.compute_fence_pool.lock().unwrap();
            for tok in fence_tokens {
                if let Some((_, fence, cmd_buf)) = fence_pool.remove(&tok) {
                    if let Some(cb) = cmd_buf {
                        logical_device
                            .device
                            .free_command_buffers(logical_device.command_pool, &[cb]);
                    }
                    logical_device.device.destroy_fence(fence, None);
                }
                // Texture staging for fence pool tokens is now handled via the
                // TextureStagingPool; it will be destroyed below.
            }

            if let Some(pipeline_layout) = logical_device.bindless_pipeline_layout {
                logical_device.device.destroy_pipeline_layout(pipeline_layout, None);
            }
            if let Some(pool) = logical_device.bindless_descriptor_pool {
                logical_device.device.destroy_descriptor_pool(pool, None);
            }
            if let Some(layout) = logical_device.bindless_descriptor_set_layout {
                logical_device.device.destroy_descriptor_set_layout(layout, None);
            }

            logical_device
                .device
                .destroy_command_pool(logical_device.command_pool, None);

            // Free all VkDeviceMemory chunks held by the sparse page pool.
            // All sparse buffers have already been unbound and destroyed above,
            // so the memories are no longer bound to any VkBuffer sparse region.
            if let Some(pool) = logical_device.sparse_page_pool.lock().unwrap().take() {
                pool.destroy(&logical_device.device);
            }

            logical_device.device.destroy_device(None);
        }
        tracing::info!(%device_handle, "destroyed Vulkan device");
    }
}

/// Check if a device handle is valid.
pub(super) fn is_valid(state: &VulkanState, device_handle: DeviceHandle) -> bool {
    state.devices.contains_key(&device_handle)
}
