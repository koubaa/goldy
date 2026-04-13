//! Device management logic.

use super::types::{self, PhysicalDeviceInfo};
use super::{DeviceHandle, VulkanState};
use crate::backend::{AdapterInfo, BackendType, DeviceType};
use anyhow::{Context, Result};
use ash::khr;
use ash::vk;
use std::ffi::CStr;

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
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_sampled_image_update_after_bind(true)
        .descriptor_binding_storage_buffer_update_after_bind(true)
        .descriptor_binding_uniform_buffer_update_after_bind(true)
        .runtime_descriptor_array(true)
        .shader_storage_buffer_array_non_uniform_indexing(true)
        .shader_sampled_image_array_non_uniform_indexing(true)
        .shader_uniform_buffer_array_non_uniform_indexing(true);

    // Vulkan 1.3 features: mandatory in 1.4, but must still be enabled.
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);

    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut vulkan_13_features)
        .push_next(&mut vulkan_12_features);

    // Create logical device with swapchain extension
    let queue_priorities = [1.0f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    // Enable swapchain extension for surface presentation
    let device_extensions = [khr::swapchain::NAME.as_ptr()];

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info))
        .enabled_extension_names(&device_extensions)
        .push_next(&mut features2);

    let device = unsafe {
        state
            .instance
            .create_device(physical_device_handle, &device_create_info, None)
    }
    .context("Failed to create logical device")?;

    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    // Create command pool
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
        .context("Failed to create command pool")?;

    // Create descriptor infrastructure for resource binding
    let (
        bindless_descriptor_pool,
        bindless_descriptor_set_layout,
        bindless_descriptor_set,
        bindless_pipeline_layout,
    ) = {
        // Create descriptor set layout with update-after-bind flag
        // Bindings organized by ACCESS PATTERN (see types.rs::bindless_bindings)
        let binding_flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
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

        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None) }
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

        // Create a pipeline layout that includes the bindless set and push constants
        let layouts = [descriptor_set_layout];

        // Push constant range for resource indices (16 x u32 = 64 bytes)
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::ALL,
            offset: 0,
            size: (types::MAX_PUSH_CONSTANT_INDICES * std::mem::size_of::<u32>()) as u32,
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
            .context("Failed to create bindless pipeline layout")?;

        tracing::info!(
            "Pipeline layout includes {} bytes of push constants for resource indices",
            push_constant_range.size
        );

        tracing::info!("Created descriptor infrastructure: pool, layout, set, pipeline layout");

        (
            Some(descriptor_pool),
            Some(descriptor_set_layout),
            Some(descriptor_set),
            Some(pipeline_layout),
        )
    };

    let handle = state.next_device_handle;
    state.next_device_handle += 1;

    state.devices.insert(
        handle,
        types::LogicalDevice {
            device,
            physical_device: physical_device_handle,
            adapter_id,
            queue,
            queue_family: queue_family_index,
            command_pool,
            bindless_descriptor_pool,
            bindless_descriptor_set_layout,
            bindless_descriptor_set,
            bindless_pipeline_layout,
            resource_registry: types::ResourceRegistry::new(),
            deletion_queue: types::DeletionQueue::new(),
        },
    );

    tracing::info!(
        "Created Vulkan device {} for adapter {}",
        handle,
        adapter_id
    );
    Ok(handle)
}

/// Destroy a logical device and all resources associated with it.
#[allow(clippy::too_many_lines)]
pub(super) fn destroy(state: &mut VulkanState, device_handle: DeviceHandle) {
    if let Some(mut logical_device) = state.devices.remove(&device_handle) {
        unsafe {
            logical_device.device.device_wait_idle().ok();

            // Flush any pending deferred deletions
            logical_device
                .deletion_queue
                .flush_all(&logical_device.device);

            // Destroy buffers owned by this device
            let buffer_handles: Vec<_> = state
                .buffers
                .iter()
                .filter(|(_, b)| b.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();

            for handle in buffer_handles {
                if let Some(buffer) = state.buffers.remove(&handle) {
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
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in shader_handles {
                if let Some(shader) = state.shaders.remove(&handle) {
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
                .iter()
                .filter(|(_, p)| p.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in pipeline_handles {
                if let Some(pipeline) = state.pipelines.remove(&handle) {
                    if pipeline.pipeline != vk::Pipeline::null() {
                        logical_device
                            .device
                            .destroy_pipeline(pipeline.pipeline, None);
                    }
                    // Only destroy layout if we own it (not the global bindless layout)
                    if pipeline.owns_layout && pipeline.layout != vk::PipelineLayout::null() {
                        logical_device
                            .device
                            .destroy_pipeline_layout(pipeline.layout, None);
                    }
                }
            }

            // Destroy compute pipelines owned by this device
            let compute_pipeline_handles: Vec<_> = state
                .compute_pipelines
                .iter()
                .filter(|(_, p)| p.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in compute_pipeline_handles {
                if let Some(pipeline) = state.compute_pipelines.remove(&handle) {
                    if pipeline.pipeline != vk::Pipeline::null() {
                        logical_device
                            .device
                            .destroy_pipeline(pipeline.pipeline, None);
                    }
                    // Only destroy layout if we own it (not the global bindless layout)
                    if pipeline.owns_layout && pipeline.layout != vk::PipelineLayout::null() {
                        logical_device
                            .device
                            .destroy_pipeline_layout(pipeline.layout, None);
                    }
                }
            }

            // Destroy render targets owned by this device
            let target_handles: Vec<_> = state
                .render_targets
                .iter()
                .filter(|(_, t)| t.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in target_handles {
                if let Some(target) = state.render_targets.remove(&handle) {
                    logical_device
                        .device
                        .destroy_image_view(target.image_view, None);
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

            // Destroy textures owned by this device
            let texture_handles: Vec<_> = state
                .textures
                .iter()
                .filter(|(_, t)| t.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in texture_handles {
                if let Some(texture) = state.textures.remove(&handle) {
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
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in sampler_handles {
                if let Some(sampler) = state.samplers.remove(&handle) {
                    logical_device.device.destroy_sampler(sampler.sampler, None);
                }
            }

            // Destroy bindless infrastructure
            if let Some(pipeline_layout) = logical_device.bindless_pipeline_layout {
                logical_device
                    .device
                    .destroy_pipeline_layout(pipeline_layout, None);
            }
            if let Some(pool) = logical_device.bindless_descriptor_pool {
                logical_device.device.destroy_descriptor_pool(pool, None);
            }
            if let Some(layout) = logical_device.bindless_descriptor_set_layout {
                logical_device
                    .device
                    .destroy_descriptor_set_layout(layout, None);
            }

            logical_device
                .device
                .destroy_command_pool(logical_device.command_pool, None);
            logical_device.device.destroy_device(None);
        }
        tracing::info!("Destroyed Vulkan device {}", device_handle);
    }
}

/// Check if a device handle is valid.
pub(super) fn is_valid(state: &VulkanState, device_handle: DeviceHandle) -> bool {
    state.devices.contains_key(&device_handle)
}
