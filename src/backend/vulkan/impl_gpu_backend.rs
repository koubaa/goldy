// Allow manual find loops for Vulkan memory type selection (common pattern)
#[allow(clippy::manual_find)]
impl GpuBackend for VulkanBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Vulkan
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        self.physical_devices
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

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        let physical_device = self
            .physical_devices
            .iter()
            .find(|d| d.adapter_id == adapter_id)
            .context("Invalid adapter ID")?;

        let physical_device_handle = physical_device.handle;

        // Find a graphics queue family
        let queue_families = unsafe {
            self.instance
                .get_physical_device_queue_family_properties(physical_device_handle)
        };

        let queue_family_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(idx, _)| idx as u32)
            .context("No graphics queue family found")?;

        // Enable Vulkan 1.4 features (dynamic rendering)
        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        // Enable Vulkan 1.2 descriptor indexing features for bindless rendering
        let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_uniform_buffer_update_after_bind(true)
            .runtime_descriptor_array(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_uniform_buffer_array_non_uniform_indexing(true);

        // Query supported features first
        let mut supported_descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        let mut supported_features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut supported_descriptor_indexing);
        unsafe {
            self.instance
                .get_physical_device_features2(physical_device_handle, &mut supported_features2);
        }

        // Check if bindless is supported (all required features must be available)
        let bindless_supported = supported_descriptor_indexing.descriptor_binding_partially_bound != 0
            && supported_descriptor_indexing.descriptor_binding_sampled_image_update_after_bind != 0
            && supported_descriptor_indexing.runtime_descriptor_array != 0
            && supported_descriptor_indexing.shader_storage_buffer_array_non_uniform_indexing != 0;

        if bindless_supported {
            tracing::info!("Vulkan descriptor indexing supported - enabling bindless");
        } else {
            tracing::info!("Vulkan descriptor indexing not fully supported - using traditional binding");
        }

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_13_features)
            .push_next(&mut descriptor_indexing_features);

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
            self.instance
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

        // Create bindless descriptor infrastructure if supported
        let (
            bindless_descriptor_pool,
            bindless_descriptor_set_layout,
            bindless_descriptor_set,
            bindless_pipeline_layout,
        ) = if bindless_supported {
            // Create descriptor set layout with update-after-bind flag
            let binding_flags = [
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
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                    .binding_flags(&binding_flags);

            let bindings = [
                // Storage buffers (binding 0)
                vk::DescriptorSetLayoutBinding::default()
                    .binding(types::bindless_bindings::STORAGE_BUFFERS)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(types::MAX_BINDLESS_RESOURCES)
                    .stage_flags(vk::ShaderStageFlags::ALL),
                // Uniform buffers (binding 1)
                vk::DescriptorSetLayoutBinding::default()
                    .binding(types::bindless_bindings::UNIFORM_BUFFERS)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(types::MAX_BINDLESS_RESOURCES)
                    .stage_flags(vk::ShaderStageFlags::ALL),
                // Sampled images (binding 2)
                vk::DescriptorSetLayoutBinding::default()
                    .binding(types::bindless_bindings::SAMPLED_IMAGES)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(types::MAX_BINDLESS_RESOURCES)
                    .stage_flags(vk::ShaderStageFlags::ALL),
                // Samplers (binding 3)
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

            let pipeline_layout =
                unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
                    .context("Failed to create bindless pipeline layout")?;
            
            tracing::info!(
                "Bindless pipeline layout includes {} bytes of push constants for resource indices",
                push_constant_range.size
            );

            tracing::info!(
                "Created bindless descriptor infrastructure: pool, layout, set, pipeline layout"
            );

            (
                Some(descriptor_pool),
                Some(descriptor_set_layout),
                Some(descriptor_set),
                Some(pipeline_layout),
            )
        } else {
            (None, None, None, None)
        };

        let handle = self.next_device_handle;
        self.next_device_handle += 1;

        self.devices.insert(
            handle,
            LogicalDevice {
                device,
                physical_device: physical_device_handle,
                adapter_id,
                queue,
                queue_family: queue_family_index,
                command_pool,
                bindless_enabled: bindless_supported,
                bindless_descriptor_pool,
                bindless_descriptor_set_layout,
                bindless_descriptor_set,
                bindless_pipeline_layout,
                resource_registry: types::ResourceRegistry::new(),
                deletion_queue: types::DeletionQueue::new(),
            },
        );

        tracing::info!(
            "Created Vulkan device {} for adapter {} [bindless={}]",
            handle,
            adapter_id,
            bindless_supported
        );
        Ok(handle)
    }

    fn destroy_device(&mut self, device_handle: DeviceHandle) {
        if let Some(mut logical_device) = self.devices.remove(&device_handle) {
            unsafe {
                logical_device.device.device_wait_idle().ok();

                // Flush any pending deferred deletions
                logical_device.deletion_queue.flush_all(&logical_device.device);

                // Destroy buffers owned by this device
                let buffer_handles: Vec<_> = self.buffers
                    .iter()
                    .filter(|(_, b)| b.device_handle == device_handle)
                    .map(|(h, _)| *h)
                    .collect();
                for handle in buffer_handles {
                    if let Some(buffer) = self.buffers.remove(&handle) {
                        logical_device.device.destroy_buffer(buffer.buffer, None);
                        logical_device.device.free_memory(buffer.memory, None);
                    }
                }

                // Destroy shaders owned by this device
                let shader_handles: Vec<_> = self.shaders
                    .iter()
                    .filter(|(_, s)| s.device_handle == device_handle)
                    .map(|(h, _)| *h)
                    .collect();
                for handle in shader_handles {
                    if let Some(shader) = self.shaders.remove(&handle) {
                        if let Some(module) = shader.vertex_module {
                            logical_device.device.destroy_shader_module(module, None);
                        }
                        if let Some(module) = shader.fragment_module {
                            logical_device.device.destroy_shader_module(module, None);
                        }
                    }
                }

                // Destroy graphics pipelines owned by this device
                let pipeline_handles: Vec<_> = self.pipelines
                    .iter()
                    .filter(|(_, p)| p.device_handle == device_handle)
                    .map(|(h, _)| *h)
                    .collect();
                for handle in pipeline_handles {
                    if let Some(pipeline) = self.pipelines.remove(&handle) {
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
                let compute_pipeline_handles: Vec<_> = self.compute_pipelines
                    .iter()
                    .filter(|(_, p)| p.device_handle == device_handle)
                    .map(|(h, _)| *h)
                    .collect();
                for handle in compute_pipeline_handles {
                    if let Some(pipeline) = self.compute_pipelines.remove(&handle) {
                        if pipeline.pipeline != vk::Pipeline::null() {
                            logical_device.device.destroy_pipeline(pipeline.pipeline, None);
                        }
                        // Only destroy layout if we own it (not the global bindless layout)
                        if pipeline.owns_layout && pipeline.layout != vk::PipelineLayout::null() {
                            logical_device.device.destroy_pipeline_layout(pipeline.layout, None);
                        }
                    }
                }

                // Destroy render targets owned by this device
                let target_handles: Vec<_> = self.render_targets
                    .iter()
                    .filter(|(_, t)| t.device_handle == device_handle)
                    .map(|(h, _)| *h)
                    .collect();
                for handle in target_handles {
                    if let Some(target) = self.render_targets.remove(&handle) {
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

                // Destroy textures owned by this device
                let texture_handles: Vec<_> = self.textures
                    .iter()
                    .filter(|(_, t)| t.device_handle == device_handle)
                    .map(|(h, _)| *h)
                    .collect();
                for handle in texture_handles {
                    if let Some(texture) = self.textures.remove(&handle) {
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
                let sampler_handles: Vec<_> = self.samplers
                    .iter()
                    .filter(|(_, s)| s.device_handle == device_handle)
                    .map(|(h, _)| *h)
                    .collect();
                for handle in sampler_handles {
                    if let Some(sampler) = self.samplers.remove(&handle) {
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

                logical_device.device.destroy_command_pool(logical_device.command_pool, None);
                logical_device.device.destroy_device(None);
            }
            tracing::info!("Destroyed Vulkan device {}", device_handle);
        }
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn create_buffer(&mut self, device_handle: DeviceHandle, size: u64, usage: BufferUsage, _element_stride: Option<u32>) -> Result<BufferHandle> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let mut vk_usage = vk::BufferUsageFlags::empty();
        if usage.contains(BufferUsage::VERTEX) {
            vk_usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if usage.contains(BufferUsage::INDEX) {
            vk_usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if usage.contains(BufferUsage::UNIFORM) {
            vk_usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if usage.contains(BufferUsage::STORAGE) {
            vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if usage.contains(BufferUsage::COPY_SRC) {
            vk_usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if usage.contains(BufferUsage::COPY_DST) {
            vk_usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }

        let is_storage = usage.contains(BufferUsage::STORAGE);
        let is_uniform = usage.contains(BufferUsage::UNIFORM);
        let should_register_bindless = is_storage || is_uniform; // Only register UNIFORM/STORAGE buffers
        let bindless_enabled = logical_device.bindless_enabled;
        let bindless_descriptor_set = logical_device.bindless_descriptor_set;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { logical_device.device.create_buffer(&buffer_info, None) }
            .context("Failed to create buffer")?;

        let mem_requirements = unsafe { logical_device.device.get_buffer_memory_requirements(buffer) };

        let memory_type = self
            .find_memory_type(
                logical_device.physical_device,
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .context("Failed to find suitable memory type")?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);

        let memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
            .context("Failed to allocate buffer memory")?;

        unsafe { logical_device.device.bind_buffer_memory(buffer, memory, 0) }
            .context("Failed to bind buffer memory")?;

        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;

        // Register buffer in bindless descriptor set if enabled AND buffer is UNIFORM or STORAGE
        // (VERTEX/INDEX buffers should not be in the uniform/storage descriptor arrays)
        let bindless_index = if bindless_enabled && should_register_bindless {
            let logical_device = self.devices.get_mut(&device_handle).unwrap();
            let index = logical_device.resource_registry.register_buffer(handle, is_storage);

            // Update the global descriptor set with this buffer
            if let Some(descriptor_set) = bindless_descriptor_set {
                let buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(0)
                    .range(size);

                let binding = if is_storage {
                    types::bindless_bindings::STORAGE_BUFFERS
                } else {
                    types::bindless_bindings::UNIFORM_BUFFERS
                };

                let descriptor_type = if is_storage {
                    vk::DescriptorType::STORAGE_BUFFER
                } else {
                    vk::DescriptorType::UNIFORM_BUFFER
                };

                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .dst_array_element(index)
                    .descriptor_type(descriptor_type)
                    .buffer_info(std::slice::from_ref(&buffer_info));

                unsafe {
                    logical_device
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[]);
                }

                tracing::trace!(
                    "Registered buffer {} at bindless index {} (storage={})",
                    handle,
                    index,
                    is_storage
                );
            }

            Some(index)
        } else {
            None
        };

        self.buffers.insert(
            handle,
            BufferState {
                device_handle,
                buffer,
                memory,
                size,
                bindless_index,
                is_storage,
            },
        );

        Ok(handle)
    }

    fn destroy_buffer(&mut self, buffer_handle: BufferHandle) {
        if let Some(buffer) = self.buffers.remove(&buffer_handle) {
            if let Some(device) = self.devices.get_mut(&buffer.device_handle) {
                // Unregister from bindless registry
                device.resource_registry.unregister_buffer(buffer_handle);

                // Queue for deferred deletion - the buffer may still be in use by in-flight commands
                device.deletion_queue.queue(types::PendingDeletion::Buffer {
                    buffer: buffer.buffer,
                    memory: buffer.memory,
                });
            }
        }
    }

    fn write_buffer(&mut self, buffer_handle: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buffer = self
            .buffers
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;

        let device = self
            .devices
            .get(&buffer.device_handle)
            .context("Buffer's device is invalid")?;

        if offset + data.len() as u64 > buffer.size {
            anyhow::bail!("Write would exceed buffer bounds");
        }

        unsafe {
            let ptr = device
                .device
                .map_memory(buffer.memory, offset, data.len() as u64, vk::MemoryMapFlags::empty())
                .context("Failed to map buffer memory")?;

            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());

            device.device.unmap_memory(buffer.memory);
        }

        Ok(())
    }

    fn buffer_size(&self, buffer_handle: BufferHandle) -> u64 {
        self.buffers.get(&buffer_handle).map(|b| b.size).unwrap_or(0)
    }

    fn buffer_bindless_index(&self, buffer_handle: BufferHandle) -> Option<u32> {
        self.buffers.get(&buffer_handle).and_then(|b| b.bindless_index)
    }

    fn create_shader(&mut self, device_handle: DeviceHandle, slang_source: &str) -> Result<ShaderHandle> {
        self.create_shader_with_paths(device_handle, slang_source, &[])
    }

    fn create_shader_with_paths(&mut self, device_handle: DeviceHandle, slang_source: &str, search_paths: &[&str]) -> Result<ShaderHandle> {
        // Just validate the device exists - actual compilation happens at pipeline creation
        let _ = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let handle = self.next_shader_handle;
        self.next_shader_handle += 1;

        self.shaders.insert(
            handle,
            ShaderState {
                device_handle,
                slang_source: slang_source.to_string(),
                search_paths: search_paths.iter().map(|s| s.to_string()).collect(),
                vertex_module: None,
                fragment_module: None,
                compute_module: None,
                reflection: None,
            },
        );

        tracing::debug!("Created shader handle {} (compilation deferred)", handle);
        Ok(handle)
    }

    fn destroy_shader(&mut self, shader_handle: ShaderHandle) {
        if let Some(shader) = self.shaders.remove(&shader_handle) {
            if let Some(device) = self.devices.get(&shader.device_handle) {
                unsafe {
                    if let Some(module) = shader.vertex_module {
                        device.device.destroy_shader_module(module, None);
                    }
                    if let Some(module) = shader.fragment_module {
                        device.device.destroy_shader_module(module, None);
                    }
                }
            }
        }
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

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Shader stages - Slang outputs "main" as the entry point name in SPIR-V
        let vs_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs_module)
            .name(c"main");

        let fs_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fs_module)
            .name(c"main");

        let shader_stages = [vs_stage, fs_stage];

        // Vertex input
        let binding_desc = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(vertex_layout.stride)
            .input_rate(vk::VertexInputRate::VERTEX);

        let attribute_descs: Vec<_> = vertex_layout
            .attributes
            .iter()
            .map(|attr| {
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(attr.location)
                    .format(vertex_format_to_vk(attr.format))
                    .offset(attr.offset)
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_desc))
            .vertex_attribute_descriptions(&attribute_descs);

        // Input assembly
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(topology_to_vk(topology))
            .primitive_restart_enable(false);

        // Viewport/scissor (dynamic)
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Rasterization
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        // Multisampling
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Color blending
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        // Dynamic state
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        // Pipeline layout - includes bindless descriptor set if enabled
        let layout = if logical_device.bindless_enabled {
            // Bindless mode: include the bindless descriptor set layout and push constants
            let bindless_set_layout = logical_device.bindless_descriptor_set_layout
                .context("Bindless enabled but no descriptor set layout")?;
            let layouts = [bindless_set_layout];
            
            // Push constant range for resource indices (16 x u32 = 64 bytes)
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::ALL,
                offset: 0,
                size: (types::MAX_PUSH_CONSTANT_INDICES * std::mem::size_of::<u32>()) as u32,
            };
            
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(std::slice::from_ref(&push_constant_range));
            
            unsafe { logical_device.device.create_pipeline_layout(&layout_info, None) }
                .context("Failed to create bindless pipeline layout")?
        } else {
            // Traditional mode: empty layout
            let layout_info = vk::PipelineLayoutCreateInfo::default();
            unsafe { logical_device.device.create_pipeline_layout(&layout_info, None) }
                .context("Failed to create pipeline layout")?
        };

        // Dynamic rendering info (Vulkan 1.4)
        let color_format = format_to_vk(target_format);
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(std::slice::from_ref(&color_format));

        // Create pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipelines = unsafe {
            logical_device.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        }
        .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e.1))?;

        let handle = self.next_pipeline_handle;
        self.next_pipeline_handle += 1;

        self.pipelines.insert(
            handle,
            PipelineState {
                device_handle,
                pipeline: pipelines[0],
                layout,
                owns_layout: true, // Simple create_pipeline always owns its layout
                parameter_block_layouts: Vec::new(),
            },
        );

        tracing::debug!("Created render pipeline {}", handle);
        Ok(handle)
    }

    fn destroy_pipeline(&mut self, pipeline_handle: PipelineHandle) {
        if let Some(pipeline) = self.pipelines.remove(&pipeline_handle) {
            if let Some(device) = self.devices.get(&pipeline.device_handle) {
                unsafe {
                    if pipeline.pipeline != vk::Pipeline::null() {
                        device.device.destroy_pipeline(pipeline.pipeline, None);
                    }
                    // Only destroy layout if we own it (not the global bindless layout)
                    if pipeline.owns_layout && pipeline.layout != vk::PipelineLayout::null() {
                        device.device.destroy_pipeline_layout(pipeline.layout, None);
                    }
                }
            }
        }
    }


    fn create_render_target(&mut self, device_handle: DeviceHandle, width: u32, height: u32, format: TextureFormat) -> Result<RenderTargetHandle> {
        // Get physical device for memory type lookup
        let physical_device = {
            let logical_device = self
                .devices
                .get(&device_handle)
                .context("Invalid device handle")?;
            logical_device.physical_device
        };

        let mem_props = unsafe { self.instance.get_physical_device_memory_properties(physical_device) };
        
        let find_mem_type = |type_filter: u32, properties: vk::MemoryPropertyFlags| -> Option<u32> {
            for i in 0..mem_props.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && (mem_props.memory_types[i as usize].property_flags & properties) == properties
                {
                    return Some(i);
                }
            }
            None
        };

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Create render target image (GPU only - no staging yet)
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format_to_vk(format))
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { logical_device.device.create_image(&image_info, None) }
            .context("Failed to create render target image")?;

        let mem_reqs = unsafe { logical_device.device.get_image_memory_requirements(image) };
        let memory_type = find_mem_type(mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .context("Failed to find memory type for render target")?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type);

        let image_memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
            .context("Failed to allocate render target memory")?;

        unsafe { logical_device.device.bind_image_memory(image, image_memory, 0) }
            .context("Failed to bind render target memory")?;

        // Create image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format_to_vk(format))
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let image_view = unsafe { logical_device.device.create_image_view(&view_info, None) }
            .context("Failed to create render target view")?;

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(logical_device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
            .context("Failed to allocate command buffer")?;

        let handle = self.next_render_target_handle;
        self.next_render_target_handle += 1;

        self.render_targets.insert(handle, RenderTargetState {
            device_handle,
            width,
            height,
            format,
            image,
            image_memory,
            image_view,
            depth_format: None,
            depth_image: None,
            depth_memory: None,
            depth_view: None,
            staging_buffer: None,
            staging_memory: None,
            command_buffer: command_buffers[0],
            has_rendered: false,
        });

        tracing::debug!("Created render target {}x{} (handle={})", width, height, handle);
        Ok(handle)
    }

    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        if let Some(state) = self.render_targets.remove(&target) {
            if let Some(logical_device) = self.devices.get(&state.device_handle) {
                unsafe {
                    let _ = logical_device.device.device_wait_idle();
                    logical_device.device.destroy_image_view(state.image_view, None);
                    logical_device.device.destroy_image(state.image, None);
                    logical_device.device.free_memory(state.image_memory, None);
                    if let Some(staging_buffer) = state.staging_buffer {
                        logical_device.device.destroy_buffer(staging_buffer, None);
                    }
                    if let Some(staging_memory) = state.staging_memory {
                        logical_device.device.free_memory(staging_memory, None);
                    }
                }
            }
        }
    }

    fn render_to_target(&mut self, device_handle: DeviceHandle, target: RenderTargetHandle, commands: &[RenderCommand]) -> Result<()> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let render_target = self
            .render_targets
            .get(&target)
            .context("Invalid render target handle")?;

        if render_target.device_handle != device_handle {
            anyhow::bail!("Render target belongs to a different device");
        }

        let cmd = render_target.command_buffer;
        let width = render_target.width;
        let height = render_target.height;
        let image = render_target.image;
        let image_view = render_target.image_view;
        let depth_view = render_target.depth_view;
        let depth_format = render_target.depth_format;

        // Find the first Clear command to use as the initial clear color
        let clear_color = commands
            .iter()
            .find_map(|c| match c {
                RenderCommand::Clear(color) => Some(*color),
                _ => None,
            })
            .unwrap_or(Color::BLACK);
        
        // Find the first ClearDepth command
        let clear_depth = commands
            .iter()
            .find_map(|c| match c {
                RenderCommand::ClearDepth(depth) => Some(*depth),
                _ => None,
            })
            .unwrap_or(1.0);

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
            .context("Failed to begin command buffer")?;

        // Transition image to color attachment
        let color_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        // Prepare image barriers - color always, depth if present
        let mut barriers = vec![color_barrier];
        
        // Add depth barrier if depth buffer exists
        if let (Some(depth_img), Some(df)) = (render_target.depth_image, depth_format) {
            let depth_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .image(depth_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: utils::depth_aspect_mask(df),
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            barriers.push(depth_barrier);
        }

        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(&barriers);

        unsafe { logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info) };

        // Begin dynamic rendering
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [clear_color.r, clear_color.g, clear_color.b, clear_color.a],
                },
            });

        // Create depth attachment if present
        let depth_attachment = depth_view.map(|dv| {
            vk::RenderingAttachmentInfo::default()
                .image_view(dv)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: clear_depth,
                        stencil: 0,
                    },
                })
        });

        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));
        
        // Add depth attachment if present
        if let Some(ref depth_att) = depth_attachment {
            rendering_info = rendering_info.depth_attachment(depth_att);
        }

        unsafe { logical_device.device.cmd_begin_rendering(cmd, &rendering_info) };

        // Set viewport and scissor
        // Use negative height to flip Y axis - makes Vulkan coordinate system match DX12
        // This requires VK_KHR_maintenance1 (core in Vulkan 1.1+)
        let viewport = vk::Viewport {
            x: 0.0,
            y: height as f32,          // Start from bottom
            width: width as f32,
            height: -(height as f32),  // Negative height flips Y
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe { logical_device.device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport)) };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        unsafe { logical_device.device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor)) };

        // Track current pipeline for bind group binding
        let mut current_pipeline: Option<PipelineHandle> = None;

        // Execute render commands (render_to_target)
        for command in commands {
            match command {
                RenderCommand::Clear(_) => {
                    // Already handled via load op
                }
                RenderCommand::ClearDepth(_) => {
                    // TODO: Implement depth clear when depth buffer is supported
                }
                RenderCommand::SetPipeline(pipeline_handle) => {
                    current_pipeline = Some(*pipeline_handle);
                    if let Some(pipeline) = self.pipelines.get(pipeline_handle) {
                        unsafe {
                            logical_device.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline.pipeline,
                            );

                            // Bind the global bindless descriptor set if enabled
                            // Use the PIPELINE's layout (not the global bindless_pipeline_layout)
                            // because the pipeline has a hybrid layout with both bindless + user sets
                            if logical_device.bindless_enabled {
                                if let Some(bindless_set) = logical_device.bindless_descriptor_set {
                                    logical_device.device.cmd_bind_descriptor_sets(
                                        cmd,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        pipeline.layout,  // Use pipeline's own layout
                                        0,
                                        std::slice::from_ref(&bindless_set),
                                        &[],
                                    );
                                }
                            }
                        }
                    }
                }
                RenderCommand::SetVertexBuffer { slot, buffer, offset } => {
                    if let Some(buf_state) = self.buffers.get(buffer) {
                        unsafe {
                            logical_device.device.cmd_bind_vertex_buffers(
                                cmd,
                                *slot,
                                std::slice::from_ref(&buf_state.buffer),
                                std::slice::from_ref(offset),
                            );
                        }
                    }
                }
                RenderCommand::SetBindGroup { index, bind_group } => {
                    if let Some(bg_state) = self.bind_groups.get(bind_group) {
                        // Use the current pipeline's layout for binding
                        let pipeline_layout = current_pipeline
                            .and_then(|p| self.pipelines.get(&p))
                            .map(|ps| ps.layout);
                        
                        if let Some(layout) = pipeline_layout {
                            if logical_device.bindless_enabled {
                                // Bindless mode: push resource indices and bind at index+1
                                // (set 0 is global bindless, user sets start at 1)
                                let mut indices = types::BindlessIndices::default();
                                for (i, (_, resource_ref)) in bg_state.entries.iter().enumerate() {
                                    if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                                    indices.indices[i] = match resource_ref {
                                        BindGroupResourceRef::Buffer(h) => {
                                            self.buffers.get(h).and_then(|b| b.bindless_index).unwrap_or(0)
                                        }
                                        BindGroupResourceRef::Texture(h) => {
                                            self.textures.get(h).and_then(|t| t.bindless_index).unwrap_or(0)
                                        }
                                        BindGroupResourceRef::Sampler(h) => {
                                            self.samplers.get(h).and_then(|s| s.bindless_index).unwrap_or(0)
                                        }
                                    };
                                }
                                
                                unsafe {
                                    logical_device.device.cmd_push_constants(
                                        cmd, layout, vk::ShaderStageFlags::ALL, 0,
                                        bytemuck::bytes_of(&indices),
                                    );
                                    logical_device.device.cmd_bind_descriptor_sets(
                                        cmd, vk::PipelineBindPoint::GRAPHICS, layout,
                                        *index + 1, // Offset by 1 for hybrid layout
                                        std::slice::from_ref(&bg_state.descriptor_set), &[],
                                    );
                                }
                            } else {
                                // Traditional mode: bind at requested index
                                unsafe {
                                    logical_device.device.cmd_bind_descriptor_sets(
                                        cmd, vk::PipelineBindPoint::GRAPHICS, layout,
                                        *index, std::slice::from_ref(&bg_state.descriptor_set), &[],
                                    );
                                }
                            }
                        }
                    }
                }
                RenderCommand::SetPushConstants { buffers } => {
                    // Fully bindless mode: push buffer indices directly (no bind groups needed)
                    if logical_device.bindless_enabled {
                        if let Some(pipeline) = current_pipeline.and_then(|p| self.pipelines.get(&p)) {
                            let mut indices = types::BindlessIndices::default();
                            for (i, buffer_handle) in buffers.iter().enumerate() {
                                if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                                indices.indices[i] = self.buffers.get(buffer_handle)
                                    .and_then(|b| b.bindless_index)
                                    .unwrap_or(0);
                            }
                            unsafe {
                                logical_device.device.cmd_push_constants(
                                    cmd, pipeline.layout, vk::ShaderStageFlags::ALL, 0,
                                    bytemuck::bytes_of(&indices),
                                );
                            }
                        }
                    }
                }
                RenderCommand::SetPushConstantsRaw { indices: raw_indices } => {
                    // Fully bindless mode: push raw indices directly (for textures/samplers)
                    if logical_device.bindless_enabled {
                        if let Some(pipeline) = current_pipeline.and_then(|p| self.pipelines.get(&p)) {
                            let mut indices = types::BindlessIndices::default();
                            for (i, &idx) in raw_indices.iter().enumerate() {
                                if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                                indices.indices[i] = idx;
                            }
                            unsafe {
                                logical_device.device.cmd_push_constants(
                                    cmd, pipeline.layout, vk::ShaderStageFlags::ALL, 0,
                                    bytemuck::bytes_of(&indices),
                                );
                            }
                        }
                    }
                }
                RenderCommand::SetIndexBuffer { buffer, offset, format } => {
                    if let Some(buf_state) = self.buffers.get(buffer) {
                        unsafe {
                            logical_device.device.cmd_bind_index_buffer(
                                cmd,
                                buf_state.buffer,
                                *offset,
                                index_format_to_vk(*format),
                            );
                        }
                    }
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    unsafe {
                        logical_device.device.cmd_draw(
                            cmd,
                            *vertex_count,
                            *instance_count,
                            *first_vertex,
                            *first_instance,
                        );
                    }
                }
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    unsafe {
                        logical_device.device.cmd_draw_indexed(
                            cmd,
                            *index_count,
                            *instance_count,
                            *first_index,
                            *base_vertex,
                            *first_instance,
                        );
                    }
                }
            }
        }

        // End dynamic rendering
        unsafe { logical_device.device.cmd_end_rendering(cmd) };

        // Transition image to transfer src (ready for potential readback)
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe { logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info) };

        // End command buffer
        unsafe { logical_device.device.end_command_buffer(cmd) }
            .context("Failed to end command buffer")?;

        // Submit
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cmd));

        unsafe {
            logical_device.device.queue_submit(
                logical_device.queue,
                std::slice::from_ref(&submit_info),
                vk::Fence::null(),
            )
        }
        .context("Failed to submit command buffer")?;

        // Wait for completion
        unsafe { logical_device.device.queue_wait_idle(logical_device.queue) }
            .context("Failed to wait for queue")?;

        // Mark as rendered
        if let Some(rt) = self.render_targets.get_mut(&target) {
            rt.has_rendered = true;
        }

        Ok(())
    }

    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()> {
        // Get render target info and device
        let (device_handle, width, height, format, image, physical_device) = {
            let render_target = self
                .render_targets
                .get(&target)
                .context("Invalid render target handle")?;

            if !render_target.has_rendered {
                anyhow::bail!("Cannot read from render target that hasn't been rendered to");
            }

            let logical_device = self
                .devices
                .get(&render_target.device_handle)
                .context("Invalid device handle")?;

            (
                render_target.device_handle,
                render_target.width,
                render_target.height,
                render_target.format,
                render_target.image,
                logical_device.physical_device,
            )
        };

        let expected_size = (width * height * format.bytes_per_pixel()) as usize;
        if output.len() < expected_size {
            anyhow::bail!(
                "Output buffer too small: {} < {}",
                output.len(),
                expected_size
            );
        }

        // Ensure staging buffer exists (lazy creation)
        let needs_staging = {
            let render_target = self.render_targets.get(&target).unwrap();
            render_target.staging_buffer.is_none()
        };

        if needs_staging {
            // Create staging buffer
            let mem_props = unsafe { self.instance.get_physical_device_memory_properties(physical_device) };
            
            let find_mem_type = |type_filter: u32, properties: vk::MemoryPropertyFlags| -> Option<u32> {
                for i in 0..mem_props.memory_type_count {
                    if (type_filter & (1 << i)) != 0
                        && (mem_props.memory_types[i as usize].property_flags & properties) == properties
                    {
                        return Some(i);
                    }
                }
                None
            };

            let logical_device = self.devices.get(&device_handle).unwrap();
            let buffer_size = expected_size as u64;

            let staging_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let staging_buffer = unsafe { logical_device.device.create_buffer(&staging_info, None) }
                .context("Failed to create staging buffer")?;

            let staging_reqs = unsafe { logical_device.device.get_buffer_memory_requirements(staging_buffer) };
            let staging_memory_type = find_mem_type(
                staging_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .context("Failed to find memory type for staging buffer")?;

            let staging_alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(staging_reqs.size)
                .memory_type_index(staging_memory_type);

            let staging_memory = unsafe { logical_device.device.allocate_memory(&staging_alloc, None) }
                .context("Failed to allocate staging buffer memory")?;

            unsafe { logical_device.device.bind_buffer_memory(staging_buffer, staging_memory, 0) }
                .context("Failed to bind staging buffer memory")?;

            let render_target = self.render_targets.get_mut(&target).unwrap();
            render_target.staging_buffer = Some(staging_buffer);
            render_target.staging_memory = Some(staging_memory);

            tracing::debug!("Created staging buffer for render target {}", target);
        }

        // Now copy and read
        let render_target = self.render_targets.get(&target).unwrap();
        let staging_buffer = render_target.staging_buffer.unwrap();
        let staging_memory = render_target.staging_memory.unwrap();
        let cmd = render_target.command_buffer;

        let logical_device = self.devices.get(&device_handle).unwrap();

        // Reset and record copy command
        unsafe { logical_device.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }
            .context("Failed to reset command buffer")?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
            .context("Failed to begin command buffer")?;

        // Copy image to staging buffer
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        unsafe {
            logical_device.device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging_buffer,
                std::slice::from_ref(&region),
            );
        }

        unsafe { logical_device.device.end_command_buffer(cmd) }
            .context("Failed to end command buffer")?;

        // Submit
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cmd));

        unsafe {
            logical_device.device.queue_submit(
                logical_device.queue,
                std::slice::from_ref(&submit_info),
                vk::Fence::null(),
            )
        }
        .context("Failed to submit command buffer")?;

        unsafe { logical_device.device.queue_wait_idle(logical_device.queue) }
            .context("Failed to wait for queue")?;

        // Read from staging buffer
        unsafe {
            let ptr = logical_device
                .device
                .map_memory(
                    staging_memory,
                    0,
                    expected_size as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .context("Failed to map staging buffer")?;

            std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), expected_size);

            logical_device.device.unmap_memory(staging_memory);
        }

        Ok(())
    }

    fn create_bind_group_layout(&mut self, device_handle: DeviceHandle, entries: &[BindGroupLayoutEntry]) -> Result<BindGroupLayoutHandle> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Build binding types map and layout bindings
        let mut binding_types = std::collections::HashMap::new();
        let bindings: Vec<_> = entries
            .iter()
            .map(|e| {
                let stage_flags = if e.visibility.0 & ShaderStages::VERTEX.0 != 0 && e.visibility.0 & ShaderStages::FRAGMENT.0 != 0 {
                    vk::ShaderStageFlags::ALL_GRAPHICS
                } else if e.visibility.0 & ShaderStages::VERTEX.0 != 0 {
                    vk::ShaderStageFlags::VERTEX
                } else if e.visibility.0 & ShaderStages::COMPUTE.0 != 0 {
                    vk::ShaderStageFlags::COMPUTE
                } else {
                    vk::ShaderStageFlags::FRAGMENT
                };

                let descriptor_type = match &e.ty {
                    BindingType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                    BindingType::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
                    BindingType::Texture => vk::DescriptorType::SAMPLED_IMAGE,
                    BindingType::Sampler => vk::DescriptorType::SAMPLER,
                    BindingType::StorageTexture => vk::DescriptorType::STORAGE_IMAGE,
                };

                // Store the descriptor type for use in create_bind_group
                binding_types.insert(e.binding, descriptor_type);

                vk::DescriptorSetLayoutBinding::default()
                    .binding(e.binding)
                    .descriptor_type(descriptor_type)
                    .descriptor_count(1)
                    .stage_flags(stage_flags)
            })
            .collect();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);

        let layout = unsafe { logical_device.device.create_descriptor_set_layout(&layout_info, None) }
            .context("Failed to create descriptor set layout")?;

        let handle = self.next_bind_group_layout_handle;
        self.next_bind_group_layout_handle += 1;

        self.bind_group_layouts.insert(handle, BindGroupLayoutState {
            device_handle,
            layout,
            binding_types,
        });

        Ok(handle)
    }

    fn create_bind_group(&mut self, device_handle: DeviceHandle, layout_handle: BindGroupLayoutHandle, entries: &[BindGroupEntry]) -> Result<BindGroupHandle> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let layout_state = self
            .bind_group_layouts
            .get(&layout_handle)
            .context("Invalid bind group layout handle")?;

        // Clone what we need before the borrow ends
        let layout = layout_state.layout;
        let binding_types = layout_state.binding_types.clone();

        // Create a descriptor pool for this bind group
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(entries.len() as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(entries.len() as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(entries.len() as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(entries.len() as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(entries.len() as u32),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let pool = unsafe { logical_device.device.create_descriptor_pool(&pool_info, None) }
            .context("Failed to create descriptor pool")?;

        // Allocate descriptor set
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&layout));

        let descriptor_sets = unsafe { logical_device.device.allocate_descriptor_sets(&alloc_info) }
            .context("Failed to allocate descriptor set")?;

        let descriptor_set = descriptor_sets[0];

        // Write buffer descriptors
        let buffer_infos: Vec<_> = entries
            .iter()
            .filter_map(|e| match &e.resource {
                BindingResource::Buffer { buffer, offset, size } => {
                    self.buffers.get(buffer).map(|b| (e.binding, b.buffer, *offset, *size))
                }
                _ => None,
            })
            .collect();

        let vk_buffer_infos: Vec<_> = buffer_infos
            .iter()
            .map(|(_, buf, offset, size)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(*buf)
                    .offset(*offset)
                    .range(*size)
            })
            .collect();

        let writes: Vec<_> = buffer_infos
            .iter()
            .enumerate()
            .map(|(idx, (binding, _, _, _))| {
                // Look up the correct descriptor type from the layout
                let descriptor_type = binding_types
                    .get(binding)
                    .copied()
                    .unwrap_or(vk::DescriptorType::UNIFORM_BUFFER);
                
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding)
                    .dst_array_element(0)
                    .descriptor_type(descriptor_type)
                    .buffer_info(std::slice::from_ref(&vk_buffer_infos[idx]))
            })
            .collect();

        // TODO: Handle texture and sampler bindings
        // For now, only buffer bindings are supported

        unsafe { logical_device.device.update_descriptor_sets(&writes, &[]) };

        // Collect entries for bindless index lookup
        let bind_entries: Vec<(u32, BindGroupResourceRef)> = entries
            .iter()
            .map(|e| {
                let resource_ref = match &e.resource {
                    BindingResource::Buffer { buffer, .. } => BindGroupResourceRef::Buffer(*buffer),
                    BindingResource::Texture(tex) => BindGroupResourceRef::Texture(*tex),
                    BindingResource::Sampler(samp) => BindGroupResourceRef::Sampler(*samp),
                };
                (e.binding, resource_ref)
            })
            .collect();

        let handle = self.next_bind_group_handle;
        self.next_bind_group_handle += 1;

        self.bind_groups.insert(handle, BindGroupState {
            device_handle,
            descriptor_set,
            pool,
            entries: bind_entries,
        });

        Ok(handle)
    }

    fn destroy_bind_group(&mut self, bind_group_handle: BindGroupHandle) {
        if let Some(bg) = self.bind_groups.remove(&bind_group_handle) {
            if let Some(device) = self.devices.get(&bg.device_handle) {
                unsafe {
                    device.device.destroy_descriptor_pool(bg.pool, None);
                }
            }
        }
    }

    fn create_pipeline_with_layout(
        &mut self,
        device_handle: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        bind_group_layouts: &[BindGroupLayoutHandle],
    ) -> Result<PipelineHandle> {
        // Compile shaders on-demand
        let vs_module = self.ensure_shader_stage_compiled(vertex_shader, crate::slang::SlangStage::Vertex)?;
        let fs_module = self.ensure_shader_stage_compiled(fragment_shader, crate::slang::SlangStage::Fragment)?;

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Determine if we use bindless layout or traditional layouts
        let (use_bindless_layout, _bindless_pipeline_layout) = if logical_device.bindless_enabled {
            (true, logical_device.bindless_pipeline_layout)
        } else {
            (false, None)
        };

        // Collect descriptor set layouts (only used in traditional mode)
        let vk_layouts: Vec<_> = bind_group_layouts
            .iter()
            .filter_map(|h| self.bind_group_layouts.get(h).map(|s| s.layout))
            .collect();

        // Shader stages - Slang outputs "main" as the entry point name in SPIR-V
        let vs_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs_module)
            .name(c"main");

        let fs_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fs_module)
            .name(c"main");

        let shader_stages = [vs_stage, fs_stage];

        // Vertex input
        let binding_desc = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(vertex_layout.stride)
            .input_rate(vk::VertexInputRate::VERTEX);

        let attribute_descs: Vec<_> = vertex_layout
            .attributes
            .iter()
            .map(|attr| {
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(attr.location)
                    .format(vertex_format_to_vk(attr.format))
                    .offset(attr.offset)
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_desc))
            .vertex_attribute_descriptions(&attribute_descs);

        // Input assembly
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(topology_to_vk(topology))
            .primitive_restart_enable(false);

        // Viewport/scissor (dynamic)
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Rasterization
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        // Multisampling
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Color blending
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        // Dynamic state
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        // Pipeline layout - use bindless hybrid layout or traditional
        let (layout, owns_layout) = if use_bindless_layout {
            // Bindless mode: create hybrid layout with:
            // - Set 0: global bindless descriptor set
            // - Sets 1+: user-provided bind group layouts
            let bindless_set_layout = logical_device.bindless_descriptor_set_layout
                .context("Bindless enabled but no bindless descriptor set layout")?;
            
            // Combine bindless set layout with user layouts
            let mut all_layouts = vec![bindless_set_layout];
            all_layouts.extend(vk_layouts.iter().copied());
            
            // Push constant range for resource indices (16 x u32 = 64 bytes)
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::ALL,
                offset: 0,
                size: (types::MAX_PUSH_CONSTANT_INDICES * std::mem::size_of::<u32>()) as u32,
            };
            
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&all_layouts)
                .push_constant_ranges(std::slice::from_ref(&push_constant_range));
            
            let layout = unsafe { logical_device.device.create_pipeline_layout(&layout_info, None) }
                .context("Failed to create hybrid bindless pipeline layout")?;
            (layout, true) // Own this layout
        } else {
            // Traditional mode: use user-provided layouts at set 0
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&vk_layouts);
            let layout = unsafe { logical_device.device.create_pipeline_layout(&layout_info, None) }
                .context("Failed to create pipeline layout")?;
            (layout, true) // Own this layout
        };

        // Dynamic rendering info (Vulkan 1.4)
        let color_format = format_to_vk(target_format);
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(std::slice::from_ref(&color_format));

        // Create pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipelines = unsafe {
            logical_device.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        }
        .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e.1))?;

        let handle = self.next_pipeline_handle;
        self.next_pipeline_handle += 1;

        self.pipelines.insert(
            handle,
            PipelineState {
                device_handle,
                pipeline: pipelines[0],
                layout,
                owns_layout,
                parameter_block_layouts: Vec::new(),
            },
        );

        tracing::debug!("Created render pipeline with layout {} (bindless={})", handle, !owns_layout);
        Ok(handle)
    }

    // Surface API implementation
    fn create_surface(
        &mut self,
        device_handle: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        display: &dyn raw_window_handle::HasDisplayHandle,
    ) -> Result<SurfaceHandle> {
        let logical_device = self.devices.get(&device_handle)
            .context("Invalid device handle")?;
        let physical_device = logical_device.physical_device;

        // Create platform-specific surface
        let surface = self.create_platform_surface(window, display)?;

        // Get surface capabilities
        let surface_loader = khr::surface::Instance::new(&self.entry, &self.instance);
        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
        }.context("Failed to get surface capabilities")?;

        // Choose surface format (prefer BGRA8 for better compatibility)
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)
        }.context("Failed to get surface formats")?;

        let format = formats.iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB || f.format == vk::Format::B8G8R8A8_UNORM)
            .or_else(|| formats.first())
            .context("No suitable surface format")?;

        // Choose present mode (FIFO = vsync)
        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
        }.context("Failed to get present modes")?;

        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX // Triple buffering if available
        } else {
            vk::PresentModeKHR::FIFO // Vsync (always available)
        };

        // Determine extent
        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: capabilities.min_image_extent.width.max(800).min(capabilities.max_image_extent.width),
                height: capabilities.min_image_extent.height.max(600).min(capabilities.max_image_extent.height),
            }
        };

        // Create swapchain
        let image_count = (capabilities.min_image_count + 1).min(
            if capabilities.max_image_count > 0 { capabilities.max_image_count } else { u32::MAX }
        );

        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain_loader = khr::swapchain::Device::new(&self.instance, &logical_device.device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None) }
            .context("Failed to create swapchain")?;

        // Get swapchain images
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
            .context("Failed to get swapchain images")?;

        // Create image views
        let swapchain_image_views: Vec<vk::ImageView> = swapchain_images.iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { logical_device.device.create_image_view(&view_info, None) }
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to create swapchain image views")?;

        // Create per-frame synchronization resources
        let mut frame_sync = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            // Create semaphores
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            let image_available_semaphore = unsafe { logical_device.device.create_semaphore(&semaphore_info, None) }
                .context("Failed to create image available semaphore")?;
            let render_finished_semaphore = unsafe { logical_device.device.create_semaphore(&semaphore_info, None) }
                .context("Failed to create render finished semaphore")?;
            
            // Create fence (signaled so first wait succeeds)
            let fence_info = vk::FenceCreateInfo::default()
                .flags(vk::FenceCreateFlags::SIGNALED);
            let in_flight_fence = unsafe { logical_device.device.create_fence(&fence_info, None) }
                .context("Failed to create in-flight fence")?;
            
            // Allocate command buffer
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(logical_device.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
                .context("Failed to allocate command buffer")?;
            
            frame_sync.push(FrameSync {
                command_buffer: command_buffers[0],
                image_available_semaphore,
                render_finished_semaphore,
                in_flight_fence,
            });
        }

        let handle = self.next_surface_handle;
        self.next_surface_handle += 1;

        self.surfaces.insert(handle, SurfaceState {
            device_handle,
            surface,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            width: extent.width,
            height: extent.height,
            format: format.format,
            current_frame: 0,
            current_image_index: None,
            frame_sync,
        });

        tracing::info!("Created surface {}x{} with {} images", extent.width, extent.height, image_count);
        Ok(handle)
    }

    fn destroy_surface(&mut self, surface_handle: SurfaceHandle) {
        if let Some(surface_state) = self.surfaces.remove(&surface_handle) {
            if let Some(logical_device) = self.devices.get(&surface_state.device_handle) {
                unsafe {
                    let _ = logical_device.device.device_wait_idle();

                    // Destroy per-frame sync resources
                    for frame in surface_state.frame_sync {
                        logical_device.device.destroy_semaphore(frame.image_available_semaphore, None);
                        logical_device.device.destroy_semaphore(frame.render_finished_semaphore, None);
                        logical_device.device.destroy_fence(frame.in_flight_fence, None);
                    }

                    for view in surface_state.swapchain_image_views {
                        logical_device.device.destroy_image_view(view, None);
                    }

                    let swapchain_loader = khr::swapchain::Device::new(&self.instance, &logical_device.device);
                    swapchain_loader.destroy_swapchain(surface_state.swapchain, None);

                    let surface_loader = khr::surface::Instance::new(&self.entry, &self.instance);
                    surface_loader.destroy_surface(surface_state.surface, None);
                }
            }
        }
    }

    fn surface_acquire(&mut self, surface_handle: SurfaceHandle) -> Result<SwapchainImageHandle> {
        // Get surface state and current frame index
        let (device_handle, _current_frame, swapchain, in_flight_fence, image_available_semaphore) = {
            let surface_state = self.surfaces.get(&surface_handle)
                .context("Invalid surface handle")?;
            let frame = &surface_state.frame_sync[surface_state.current_frame];
            (
                surface_state.device_handle,
                surface_state.current_frame,
                surface_state.swapchain,
                frame.in_flight_fence,
                frame.image_available_semaphore,
            )
        };

        let logical_device = self.devices.get(&device_handle)
            .context("Surface's device is invalid")?;

        // Wait for the previous frame using this slot to finish
        unsafe {
            logical_device.device.wait_for_fences(
                &[in_flight_fence],
                true,
                u64::MAX,
            )
        }.context("Failed to wait for frame fence")?;

        // Process deferred deletions - resources from frames that have now completed
        // Since we just waited for the fence, frame (current_deletion_frame - MAX_FRAMES_IN_FLIGHT) has completed
        {
            let logical_device = self.devices.get_mut(&device_handle)
                .context("Surface's device is invalid")?;
            let current_frame = logical_device.deletion_queue.current_frame;
            if current_frame >= types::MAX_FRAMES_IN_FLIGHT as u64 {
                let completed_frame = current_frame - types::MAX_FRAMES_IN_FLIGHT as u64;
                logical_device.deletion_queue.process_deletions(&logical_device.device, completed_frame);
            }
        }

        let logical_device = self.devices.get(&device_handle)
            .context("Surface's device is invalid")?;

        // Reset fence for this frame
        unsafe {
            logical_device.device.reset_fences(&[in_flight_fence])
        }.context("Failed to reset frame fence")?;

        // Acquire next swapchain image
        let swapchain_loader = khr::swapchain::Device::new(&self.instance, &logical_device.device);

        let acquire_result = unsafe {
            swapchain_loader.acquire_next_image(
                swapchain,
                u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )
        };

        match acquire_result {
            Ok((image_index, suboptimal)) => {
                if suboptimal {
                    tracing::debug!("Swapchain suboptimal - consider resizing");
                }
                // Update surface state
                let surface_state = self.surfaces.get_mut(&surface_handle).unwrap();
                surface_state.current_image_index = Some(image_index);
                Ok(image_index as SwapchainImageHandle)
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // Swapchain is out of date - caller should resize and retry
                tracing::info!("Swapchain out of date - resize required");
                anyhow::bail!("Surface out of date - call resize() and retry")
            }
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                tracing::error!("Surface lost");
                anyhow::bail!("Surface lost - recreate surface")
            }
            Err(e) => {
                anyhow::bail!("Failed to acquire swapchain image: {:?}", e)
            }
        }
    }

    fn surface_render(&mut self, surface_handle: SurfaceHandle, _image: SwapchainImageHandle, commands: &[RenderCommand]) -> Result<()> {
        let surface_state = self.surfaces.get(&surface_handle)
            .context("Invalid surface handle")?;

        let image_index = surface_state.current_image_index
            .context("No image acquired - call surface_acquire first")?;

        let logical_device = self.devices.get(&surface_state.device_handle)
            .context("Surface's device is invalid")?;

        let current_frame = surface_state.current_frame;
        let frame = &surface_state.frame_sync[current_frame];
        let cmd = frame.command_buffer;
        let width = surface_state.width;
        let height = surface_state.height;
        let image = surface_state.swapchain_images[image_index as usize];
        let image_view = surface_state.swapchain_image_views[image_index as usize];

        // Find clear color
        let clear_color = commands.iter()
            .find_map(|c| match c {
                RenderCommand::Clear(color) => Some(*color),
                _ => None,
            })
            .unwrap_or(Color::BLACK);

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
            .context("Failed to begin command buffer")?;

        // Transition image to color attachment
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe { logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info) };

        // Begin dynamic rendering
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [clear_color.r, clear_color.g, clear_color.b, clear_color.a],
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));

        unsafe { logical_device.device.cmd_begin_rendering(cmd, &rendering_info) };

        // Set viewport and scissor
        // Use negative height to flip Y axis - makes Vulkan coordinate system match DX12
        // This requires VK_KHR_maintenance1 (core in Vulkan 1.1+)
        let viewport = vk::Viewport {
            x: 0.0,
            y: height as f32,          // Start from bottom
            width: width as f32,
            height: -(height as f32),  // Negative height flips Y
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe { logical_device.device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport)) };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        unsafe { logical_device.device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor)) };

        // Track current pipeline for bind group binding
        let mut current_pipeline: Option<PipelineHandle> = None;

        // Execute render commands
        for command in commands {
            match command {
                RenderCommand::Clear(_) => { /* Already handled */ }
                RenderCommand::ClearDepth(_) => { /* TODO: Implement depth clear */ }
                RenderCommand::SetPipeline(pipeline_handle) => {
                    current_pipeline = Some(*pipeline_handle);
                    if let Some(pipeline) = self.pipelines.get(pipeline_handle) {
                        unsafe {
                            logical_device.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline.pipeline,
                            );

                            // Bind the global bindless descriptor set if enabled
                            // Use the PIPELINE's layout (not the global bindless_pipeline_layout)
                            // because the pipeline has a hybrid layout with both bindless + user sets
                            if logical_device.bindless_enabled {
                                if let Some(bindless_set) = logical_device.bindless_descriptor_set {
                                    logical_device.device.cmd_bind_descriptor_sets(
                                        cmd,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        pipeline.layout,  // Use pipeline's own layout
                                        0,
                                        std::slice::from_ref(&bindless_set),
                                        &[],
                                    );
                                }
                            }
                        }
                    }
                }
                RenderCommand::SetVertexBuffer { slot, buffer, offset } => {
                    if let Some(buf_state) = self.buffers.get(buffer) {
                        unsafe {
                            logical_device.device.cmd_bind_vertex_buffers(
                                cmd,
                                *slot,
                                std::slice::from_ref(&buf_state.buffer),
                                std::slice::from_ref(offset),
                            );
                        }
                    }
                }
                RenderCommand::SetBindGroup { index, bind_group } => {
                    if let Some(bg_state) = self.bind_groups.get(bind_group) {
                        // Use the current pipeline's layout for binding
                        let pipeline_layout = current_pipeline
                            .and_then(|p| self.pipelines.get(&p))
                            .map(|ps| ps.layout);
                        
                        if let Some(layout) = pipeline_layout {
                            if logical_device.bindless_enabled {
                                // Bindless mode: push resource indices and bind at index+1
                                // (set 0 is global bindless, user sets start at 1)
                                let mut indices = types::BindlessIndices::default();
                                for (i, (_, resource_ref)) in bg_state.entries.iter().enumerate() {
                                    if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                                    indices.indices[i] = match resource_ref {
                                        BindGroupResourceRef::Buffer(h) => {
                                            self.buffers.get(h).and_then(|b| b.bindless_index).unwrap_or(0)
                                        }
                                        BindGroupResourceRef::Texture(h) => {
                                            self.textures.get(h).and_then(|t| t.bindless_index).unwrap_or(0)
                                        }
                                        BindGroupResourceRef::Sampler(h) => {
                                            self.samplers.get(h).and_then(|s| s.bindless_index).unwrap_or(0)
                                        }
                                    };
                                }
                                
                                unsafe {
                                    logical_device.device.cmd_push_constants(
                                        cmd, layout, vk::ShaderStageFlags::ALL, 0,
                                        bytemuck::bytes_of(&indices),
                                    );
                                    logical_device.device.cmd_bind_descriptor_sets(
                                        cmd, vk::PipelineBindPoint::GRAPHICS, layout,
                                        *index + 1, // Offset by 1 for hybrid layout
                                        std::slice::from_ref(&bg_state.descriptor_set), &[],
                                    );
                                }
                            } else {
                                // Traditional mode: bind at requested index
                                unsafe {
                                    logical_device.device.cmd_bind_descriptor_sets(
                                        cmd, vk::PipelineBindPoint::GRAPHICS, layout,
                                        *index, std::slice::from_ref(&bg_state.descriptor_set), &[],
                                    );
                                }
                            }
                        }
                    }
                }
                RenderCommand::SetPushConstants { buffers } => {
                    // Fully bindless mode: push buffer indices directly (no bind groups needed)
                    if logical_device.bindless_enabled {
                        if let Some(pipeline) = current_pipeline.and_then(|p| self.pipelines.get(&p)) {
                            let mut indices = types::BindlessIndices::default();
                            for (i, buffer_handle) in buffers.iter().enumerate() {
                                if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                                indices.indices[i] = self.buffers.get(buffer_handle)
                                    .and_then(|b| b.bindless_index)
                                    .unwrap_or(0);
                            }
                            unsafe {
                                logical_device.device.cmd_push_constants(
                                    cmd, pipeline.layout, vk::ShaderStageFlags::ALL, 0,
                                    bytemuck::bytes_of(&indices),
                                );
                            }
                        }
                    }
                }
                RenderCommand::SetPushConstantsRaw { indices: raw_indices } => {
                    // Fully bindless mode: push raw indices directly (for textures/samplers)
                    if logical_device.bindless_enabled {
                        if let Some(pipeline) = current_pipeline.and_then(|p| self.pipelines.get(&p)) {
                            let mut indices = types::BindlessIndices::default();
                            for (i, &idx) in raw_indices.iter().enumerate() {
                                if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                                indices.indices[i] = idx;
                            }
                            unsafe {
                                logical_device.device.cmd_push_constants(
                                    cmd, pipeline.layout, vk::ShaderStageFlags::ALL, 0,
                                    bytemuck::bytes_of(&indices),
                                );
                            }
                        }
                    }
                }
                RenderCommand::SetIndexBuffer { buffer, offset, format } => {
                    if let Some(buf_state) = self.buffers.get(buffer) {
                        unsafe {
                            logical_device.device.cmd_bind_index_buffer(
                                cmd,
                                buf_state.buffer,
                                *offset,
                                index_format_to_vk(*format),
                            );
                        }
                    }
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    unsafe {
                        logical_device.device.cmd_draw(
                            cmd,
                            *vertex_count,
                            *instance_count,
                            *first_vertex,
                            *first_instance,
                        );
                    }
                }
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    unsafe {
                        logical_device.device.cmd_draw_indexed(
                            cmd,
                            *index_count,
                            *instance_count,
                            *first_index,
                            *base_vertex,
                            *first_instance,
                        );
                    }
                }
            }
        }

        // End dynamic rendering
        unsafe { logical_device.device.cmd_end_rendering(cmd) };

        // Transition image for presentation
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe { logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info) };

        // End command buffer
        unsafe { logical_device.device.end_command_buffer(cmd) }
            .context("Failed to end command buffer")?;

        // Get per-frame sync primitives
        let frame = &surface_state.frame_sync[current_frame];
        let wait_semaphores = [frame.image_available_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [frame.render_finished_semaphore];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd))
            .signal_semaphores(&signal_semaphores);

        // Submit with fence for frame tracking
        unsafe {
            logical_device.device.queue_submit(
                logical_device.queue,
                std::slice::from_ref(&submit_info),
                frame.in_flight_fence,
            )
        }.context("Failed to submit command buffer")?;

        Ok(())
    }

    fn surface_present(&mut self, surface_handle: SurfaceHandle, _image: SwapchainImageHandle) -> Result<()> {
        let surface_state = self.surfaces.get_mut(&surface_handle)
            .context("Invalid surface handle")?;

        let image_index = surface_state.current_image_index
            .context("No image to present - call surface_render first")?;

        let current_frame = surface_state.current_frame;
        let frame = &surface_state.frame_sync[current_frame];

        let logical_device = self.devices.get(&surface_state.device_handle)
            .context("Surface's device is invalid")?;

        let swapchain_loader = khr::swapchain::Device::new(&self.instance, &logical_device.device);

        let swapchains = [surface_state.swapchain];
        let image_indices = [image_index];
        let wait_semaphores = [frame.render_finished_semaphore];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let result = unsafe { swapchain_loader.queue_present(logical_device.queue, &present_info) };

        // Clear the current image and advance frame counter
        let device_handle = surface_state.device_handle;
        let surface_state = self.surfaces.get_mut(&surface_handle).unwrap();
        surface_state.current_image_index = None;
        surface_state.current_frame = (surface_state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        // Advance the deletion queue's frame counter
        if let Some(device) = self.devices.get_mut(&device_handle) {
            device.deletion_queue.advance_frame();
        }

        // Handle suboptimal or out of date
        match result {
            Ok(_) => Ok(()),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                // TODO: Signal that resize is needed
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!("Failed to present: {:?}", e)),
        }
    }

    fn surface_resize(&mut self, surface_handle: SurfaceHandle, width: u32, height: u32) -> Result<()> {
        // Get surface info we need
        let (device_handle, surface, old_swapchain, format) = {
            let surface_state = self.surfaces.get(&surface_handle)
                .context("Invalid surface handle")?;
            (
                surface_state.device_handle,
                surface_state.surface,
                surface_state.swapchain,
                surface_state.format,
            )
        };

        let logical_device = self.devices.get(&device_handle)
            .context("Surface's device is invalid")?;
        let physical_device = logical_device.physical_device;

        // Wait for all in-flight frames to complete before resizing
        unsafe { logical_device.device.device_wait_idle() }?;

        // Destroy old image views
        if let Some(surface_state) = self.surfaces.get(&surface_handle) {
            for view in &surface_state.swapchain_image_views {
                unsafe { logical_device.device.destroy_image_view(*view, None) };
            }
        }

        // Get new capabilities
        let surface_loader = khr::surface::Instance::new(&self.entry, &self.instance);
        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
        }.context("Failed to get surface capabilities")?;

        let extent = vk::Extent2D {
            width: width.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
            height: height.clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
        };

        let image_count = (capabilities.min_image_count + 1).min(
            if capabilities.max_image_count > 0 { capabilities.max_image_count } else { u32::MAX }
        );

        // Create new swapchain (reusing old one for efficiency)
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain_loader = khr::swapchain::Device::new(&self.instance, &logical_device.device);
        let new_swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None) }
            .context("Failed to recreate swapchain")?;

        // Destroy old swapchain
        unsafe { swapchain_loader.destroy_swapchain(old_swapchain, None) };

        // Get new images and create views
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(new_swapchain) }
            .context("Failed to get swapchain images")?;

        let swapchain_image_views: Vec<vk::ImageView> = swapchain_images.iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { logical_device.device.create_image_view(&view_info, None) }
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to create swapchain image views")?;

        // Update surface state - reset frame counter since we waited for idle
        if let Some(surface_state) = self.surfaces.get_mut(&surface_handle) {
            surface_state.swapchain = new_swapchain;
            surface_state.swapchain_images = swapchain_images;
            surface_state.swapchain_image_views = swapchain_image_views;
            surface_state.width = extent.width;
            surface_state.height = extent.height;
            surface_state.current_frame = 0;
            surface_state.current_image_index = None;
        }

        tracing::debug!("Resized surface to {}x{}", extent.width, extent.height);
        Ok(())
    }

    fn surface_size(&self, surface_handle: SurfaceHandle) -> (u32, u32) {
        self.surfaces.get(&surface_handle)
            .map(|s| (s.width, s.height))
            .unwrap_or((0, 0))
    }

    fn surface_format(&self, surface_handle: SurfaceHandle) -> TextureFormat {
        self.surfaces.get(&surface_handle)
            .and_then(|s| utils::vk_to_format(s.format))
            .unwrap_or(TextureFormat::Bgra8UnormSrgb) // Safe fallback
    }

    fn create_pipeline_with_depth(
        &mut self,
        device_handle: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        bind_group_layouts: &[BindGroupLayoutHandle],
        depth_stencil: Option<&crate::types::DepthStencilState>,
    ) -> Result<PipelineHandle> {
        // Compile shaders on-demand
        let vs_module = self.ensure_shader_stage_compiled(vertex_shader, crate::slang::SlangStage::Vertex)?;
        let fs_module = self.ensure_shader_stage_compiled(fragment_shader, crate::slang::SlangStage::Fragment)?;

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Determine if we use bindless layout or traditional layouts
        let (use_bindless_layout, _bindless_pipeline_layout) = if logical_device.bindless_enabled {
            (true, logical_device.bindless_pipeline_layout)
        } else {
            (false, None)
        };

        // Collect descriptor set layouts (only used in traditional mode)
        let vk_layouts: Vec<_> = bind_group_layouts
            .iter()
            .filter_map(|h| self.bind_group_layouts.get(h).map(|s| s.layout))
            .collect();

        // Shader stages
        let vs_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs_module)
            .name(c"main");

        let fs_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fs_module)
            .name(c"main");

        let shader_stages = [vs_stage, fs_stage];

        // Vertex input
        let binding_desc = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(vertex_layout.stride)
            .input_rate(vk::VertexInputRate::VERTEX);

        let attribute_descs: Vec<_> = vertex_layout
            .attributes
            .iter()
            .map(|attr| {
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(attr.location)
                    .format(vertex_format_to_vk(attr.format))
                    .offset(attr.offset)
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_desc))
            .vertex_attribute_descriptions(&attribute_descs);

        // Input assembly
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(topology_to_vk(topology))
            .primitive_restart_enable(false);

        // Viewport/scissor (dynamic)
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Rasterization
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        // Multisampling
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Depth stencil state
        let depth_stencil_state = if let Some(ds) = depth_stencil {
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(ds.depth_write_enabled || ds.depth_compare != crate::types::CompareFunction::Always)
                .depth_write_enable(ds.depth_write_enabled)
                .depth_compare_op(utils::compare_to_vk(ds.depth_compare))
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
        } else {
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(false)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::ALWAYS)
        };

        // Color blending
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        // Dynamic state
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        // Pipeline layout - use bindless hybrid layout or traditional
        let (layout, owns_layout) = if use_bindless_layout {
            // Bindless mode: create hybrid layout with:
            // - Set 0: global bindless descriptor set
            // - Sets 1+: user-provided bind group layouts
            let bindless_set_layout = logical_device.bindless_descriptor_set_layout
                .context("Bindless enabled but no bindless descriptor set layout")?;
            
            // Combine bindless set layout with user layouts
            let mut all_layouts = vec![bindless_set_layout];
            all_layouts.extend(vk_layouts.iter().copied());
            
            // Push constant range for resource indices (16 x u32 = 64 bytes)
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::ALL,
                offset: 0,
                size: (types::MAX_PUSH_CONSTANT_INDICES * std::mem::size_of::<u32>()) as u32,
            };
            
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&all_layouts)
                .push_constant_ranges(std::slice::from_ref(&push_constant_range));
            
            let layout = unsafe { logical_device.device.create_pipeline_layout(&layout_info, None) }
                .context("Failed to create hybrid bindless pipeline layout")?;
            (layout, true) // Own this layout
        } else {
            // Traditional mode: use user-provided layouts at set 0
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&vk_layouts);
            let layout = unsafe { logical_device.device.create_pipeline_layout(&layout_info, None) }
                .context("Failed to create pipeline layout")?;
            (layout, true) // Own this layout
        };

        // Dynamic rendering info (Vulkan 1.4)
        let color_format = format_to_vk(target_format);
        let depth_format_vk = depth_stencil
            .map(|ds| utils::depth_format_to_vk(ds.format))
            .unwrap_or(vk::Format::UNDEFINED);
        
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(std::slice::from_ref(&color_format))
            .depth_attachment_format(depth_format_vk);

        // Create pipeline with depth stencil state
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipelines = unsafe {
            logical_device.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        }
        .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e.1))?;

        let handle = self.next_pipeline_handle;
        self.next_pipeline_handle += 1;

        self.pipelines.insert(
            handle,
            PipelineState {
                device_handle,
                pipeline: pipelines[0],
                layout,
                owns_layout,
                parameter_block_layouts: Vec::new(),
            },
        );

        tracing::debug!("Created pipeline with depth stencil (handle={}, bindless={})", handle, !owns_layout);
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
        // Get physical device for memory type lookup
        let physical_device = {
            let logical_device = self
                .devices
                .get(&device_handle)
                .context("Invalid device handle")?;
            logical_device.physical_device
        };

        let mem_props = unsafe { self.instance.get_physical_device_memory_properties(physical_device) };
        
        let find_mem_type = |type_filter: u32, properties: vk::MemoryPropertyFlags| -> Option<u32> {
            for i in 0..mem_props.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && (mem_props.memory_types[i as usize].property_flags & properties) == properties
                {
                    return Some(i);
                }
            }
            None
        };

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Create color render target image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format_to_vk(color_format))
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { logical_device.device.create_image(&image_info, None) }
            .context("Failed to create render target image")?;

        let mem_reqs = unsafe { logical_device.device.get_image_memory_requirements(image) };
        let memory_type = find_mem_type(mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .context("Failed to find memory type for render target")?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type);

        let image_memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
            .context("Failed to allocate render target memory")?;

        unsafe { logical_device.device.bind_image_memory(image, image_memory, 0) }
            .context("Failed to bind render target memory")?;

        // Create color image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format_to_vk(color_format))
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let image_view = unsafe { logical_device.device.create_image_view(&view_info, None) }
            .context("Failed to create render target view")?;

        // Create depth buffer if requested
        let (depth_image, depth_memory, depth_view) = if let Some(df) = depth_format {
            let vk_depth_format = utils::depth_format_to_vk(df);
            
            let depth_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk_depth_format)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let d_image = unsafe { logical_device.device.create_image(&depth_info, None) }
                .context("Failed to create depth buffer image")?;

            let d_mem_reqs = unsafe { logical_device.device.get_image_memory_requirements(d_image) };
            let d_memory_type = find_mem_type(d_mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .context("Failed to find memory type for depth buffer")?;

            let d_alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(d_mem_reqs.size)
                .memory_type_index(d_memory_type);

            let d_memory = unsafe { logical_device.device.allocate_memory(&d_alloc_info, None) }
                .context("Failed to allocate depth buffer memory")?;

            unsafe { logical_device.device.bind_image_memory(d_image, d_memory, 0) }
                .context("Failed to bind depth buffer memory")?;

            let d_view_info = vk::ImageViewCreateInfo::default()
                .image(d_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk_depth_format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: utils::depth_aspect_mask(df),
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let d_view = unsafe { logical_device.device.create_image_view(&d_view_info, None) }
                .context("Failed to create depth buffer view")?;

            (Some(d_image), Some(d_memory), Some(d_view))
        } else {
            (None, None, None)
        };

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(logical_device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
            .context("Failed to allocate command buffer")?;

        let handle = self.next_render_target_handle;
        self.next_render_target_handle += 1;

        self.render_targets.insert(handle, RenderTargetState {
            device_handle,
            width,
            height,
            format: color_format,
            image,
            image_memory,
            image_view,
            depth_format,
            depth_image,
            depth_memory,
            depth_view,
            staging_buffer: None,
            staging_memory: None,
            command_buffer: command_buffers[0],
            has_rendered: false,
        });

        tracing::debug!("Created render target {}x{} with depth={:?} (handle={})", width, height, depth_format.is_some(), handle);
        Ok(handle)
    }

    fn create_texture(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: crate::types::TextureUsage,
    ) -> Result<TextureHandle> {
        // Get physical device for memory type lookup
        let physical_device = {
            let logical_device = self.devices.get(&device_handle)
                .context("Invalid device handle")?;
            logical_device.physical_device
        };

        let mem_props = unsafe { self.instance.get_physical_device_memory_properties(physical_device) };
        
        let find_mem_type = |type_filter: u32, properties: vk::MemoryPropertyFlags| -> Option<u32> {
            for i in 0..mem_props.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && (mem_props.memory_types[i as usize].property_flags & properties) == properties
                {
                    return Some(i);
                }
            }
            None
        };

        let logical_device = self.devices.get(&device_handle)
            .context("Invalid device handle")?;

        // Convert usage flags
        let mut vk_usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        if usage.contains(crate::types::TextureUsage::RENDER_TARGET) {
            vk_usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if usage.contains(crate::types::TextureUsage::COPY_SRC) {
            vk_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if usage.contains(crate::types::TextureUsage::STORAGE) {
            vk_usage |= vk::ImageUsageFlags::STORAGE;
        }

        // Create texture image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format_to_vk(format))
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { logical_device.device.create_image(&image_info, None) }
            .context("Failed to create texture image")?;

        let mem_reqs = unsafe { logical_device.device.get_image_memory_requirements(image) };
        let memory_type = find_mem_type(mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .context("Failed to find memory type for texture")?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type);

        let memory = unsafe { logical_device.device.allocate_memory(&alloc_info, None) }
            .context("Failed to allocate texture memory")?;

        unsafe { logical_device.device.bind_image_memory(image, memory, 0) }
            .context("Failed to bind texture memory")?;

        // Create image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format_to_vk(format))
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { logical_device.device.create_image_view(&view_info, None) }
            .context("Failed to create texture view")?;

        let bindless_enabled = logical_device.bindless_enabled;
        let bindless_descriptor_set = logical_device.bindless_descriptor_set;

        let handle = self.next_texture_handle;
        self.next_texture_handle += 1;

        // Register texture in bindless descriptor set if enabled
        let bindless_index = if bindless_enabled {
            let logical_device = self.devices.get_mut(&device_handle).unwrap();
            let index = logical_device.resource_registry.register_texture(handle);

            // Update the global descriptor set with this texture
            if let Some(descriptor_set) = bindless_descriptor_set {
                let image_info = vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(types::bindless_bindings::SAMPLED_IMAGES)
                    .dst_array_element(index)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(&image_info));

                unsafe {
                    logical_device
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[]);
                }

                tracing::trace!(
                    "Registered texture {} at bindless index {}",
                    handle,
                    index
                );
            }

            Some(index)
        } else {
            None
        };

        self.textures.insert(handle, TextureState {
            device_handle,
            width,
            height,
            format,
            image,
            memory,
            view,
            staging_buffer: None,
            staging_memory: None,
            bindless_index,
        });

        tracing::debug!("Created texture {}x{} (handle={})", width, height, handle);
        Ok(handle)
    }

    fn write_texture(&mut self, texture_handle: TextureHandle, data: &[u8], width: u32, height: u32) -> Result<()> {
        let texture = self.textures.get(&texture_handle)
            .context("Invalid texture handle")?;
        
        let device_handle = texture.device_handle;
        let image = texture.image;
        let tex_width = texture.width;
        let tex_height = texture.height;
        
        // Validate dimensions
        if width != tex_width || height != tex_height {
            anyhow::bail!("Texture dimensions mismatch: expected {}x{}, got {}x{}", tex_width, tex_height, width, height);
        }

        // Get physical device for memory type lookup
        let physical_device = {
            let logical_device = self.devices.get(&device_handle)
                .context("Invalid device handle")?;
            logical_device.physical_device
        };

        let mem_props = unsafe { self.instance.get_physical_device_memory_properties(physical_device) };
        
        let find_mem_type = |type_filter: u32, properties: vk::MemoryPropertyFlags| -> Option<u32> {
            for i in 0..mem_props.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && (mem_props.memory_types[i as usize].property_flags & properties) == properties
                {
                    return Some(i);
                }
            }
            None
        };

        let logical_device = self.devices.get(&device_handle)
            .context("Invalid device handle")?;

        // Create staging buffer
        let buffer_size = data.len() as u64;
        let staging_buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe { logical_device.device.create_buffer(&staging_buffer_info, None) }
            .context("Failed to create staging buffer")?;

        let staging_mem_reqs = unsafe { logical_device.device.get_buffer_memory_requirements(staging_buffer) };
        let staging_memory_type = find_mem_type(
            staging_mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        ).context("Failed to find memory type for staging buffer")?;

        let staging_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_mem_reqs.size)
            .memory_type_index(staging_memory_type);

        let staging_memory = unsafe { logical_device.device.allocate_memory(&staging_alloc_info, None) }
            .context("Failed to allocate staging memory")?;

        unsafe { logical_device.device.bind_buffer_memory(staging_buffer, staging_memory, 0) }
            .context("Failed to bind staging memory")?;

        // Copy data to staging buffer
        unsafe {
            let ptr = logical_device.device.map_memory(staging_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .context("Failed to map staging memory")?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            logical_device.device.unmap_memory(staging_memory);
        }

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(logical_device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
            .context("Failed to allocate command buffer")?;
        let cmd_buffer = cmd_buffers[0];

        // Record commands
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            logical_device.device.begin_command_buffer(cmd_buffer, &begin_info)
                .context("Failed to begin command buffer")?;

            // Transition image to transfer dst
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let dep_info = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&barrier));
            logical_device.device.cmd_pipeline_barrier2(cmd_buffer, &dep_info);

            // Copy buffer to image
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D { width, height, depth: 1 });

            logical_device.device.cmd_copy_buffer_to_image(
                cmd_buffer,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Transition image to shader read
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let dep_info = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&barrier));
            logical_device.device.cmd_pipeline_barrier2(cmd_buffer, &dep_info);

            logical_device.device.end_command_buffer(cmd_buffer)
                .context("Failed to end command buffer")?;

            // Submit and wait
            let cmd_buffers = [cmd_buffer];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&cmd_buffers);

            logical_device.device.queue_submit(logical_device.queue, &[submit_info], vk::Fence::null())
                .context("Failed to submit command buffer")?;
            logical_device.device.queue_wait_idle(logical_device.queue)
                .context("Failed to wait for queue")?;

            // Cleanup
            logical_device.device.free_command_buffers(logical_device.command_pool, &[cmd_buffer]);
            logical_device.device.destroy_buffer(staging_buffer, None);
            logical_device.device.free_memory(staging_memory, None);
        }

        tracing::debug!("Wrote {}x{} texture data ({} bytes)", width, height, data.len());
        Ok(())
    }

    fn destroy_texture(&mut self, texture_handle: TextureHandle) {
        if let Some(texture) = self.textures.remove(&texture_handle) {
            if let Some(logical_device) = self.devices.get_mut(&texture.device_handle) {
                // Unregister from bindless registry
                logical_device.resource_registry.unregister_texture(texture_handle);

                unsafe {
                    logical_device.device.device_wait_idle().ok();
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
        }
    }

    fn texture_bindless_index(&self, texture_handle: TextureHandle) -> Option<u32> {
        self.textures.get(&texture_handle).and_then(|t| t.bindless_index)
    }

    fn create_sampler(&mut self, device_handle: DeviceHandle, desc: &crate::types::SamplerDesc) -> Result<SamplerHandle> {
        let logical_device = self.devices.get(&device_handle)
            .context("Invalid device handle")?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(utils::filter_to_vk(desc.mag_filter))
            .min_filter(utils::filter_to_vk(desc.min_filter))
            .mipmap_mode(utils::mipmap_mode_to_vk(desc.mipmap_filter))
            .address_mode_u(utils::address_mode_to_vk(desc.address_mode_u))
            .address_mode_v(utils::address_mode_to_vk(desc.address_mode_v))
            .address_mode_w(utils::address_mode_to_vk(desc.address_mode_w))
            .mip_lod_bias(0.0)
            .anisotropy_enable(desc.max_anisotropy > 1.0)
            .max_anisotropy(desc.max_anisotropy)
            .compare_enable(desc.compare.is_some())
            .compare_op(desc.compare.map(utils::compare_to_vk).unwrap_or(vk::CompareOp::ALWAYS))
            .min_lod(desc.lod_min_clamp)
            .max_lod(desc.lod_max_clamp)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .unnormalized_coordinates(false);

        let sampler = unsafe { logical_device.device.create_sampler(&sampler_info, None) }
            .context("Failed to create sampler")?;

        let bindless_enabled = logical_device.bindless_enabled;
        let bindless_descriptor_set = logical_device.bindless_descriptor_set;

        let handle = self.next_sampler_handle;
        self.next_sampler_handle += 1;

        // Register sampler in bindless descriptor set if enabled
        let bindless_index = if bindless_enabled {
            let logical_device = self.devices.get_mut(&device_handle).unwrap();
            let index = logical_device.resource_registry.register_sampler(handle);

            // Update the global descriptor set with this sampler
            if let Some(descriptor_set) = bindless_descriptor_set {
                let sampler_info = vk::DescriptorImageInfo::default().sampler(sampler);

                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(types::bindless_bindings::SAMPLERS)
                    .dst_array_element(index)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(std::slice::from_ref(&sampler_info));

                unsafe {
                    logical_device
                        .device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[]);
                }

                tracing::trace!("Registered sampler {} at bindless index {}", handle, index);
            }

            Some(index)
        } else {
            None
        };

        self.samplers.insert(handle, SamplerState {
            device_handle,
            sampler,
            bindless_index,
        });

        tracing::debug!("Created sampler (handle={})", handle);
        Ok(handle)
    }

    fn destroy_sampler(&mut self, sampler_handle: SamplerHandle) {
        if let Some(sampler) = self.samplers.remove(&sampler_handle) {
            if let Some(logical_device) = self.devices.get_mut(&sampler.device_handle) {
                // Unregister from bindless registry
                logical_device.resource_registry.unregister_sampler(sampler_handle);

                unsafe {
                    logical_device.device.device_wait_idle().ok();
                    logical_device.device.destroy_sampler(sampler.sampler, None);
                }
            }
        }
    }

    fn sampler_bindless_index(&self, sampler_handle: SamplerHandle) -> Option<u32> {
        self.samplers.get(&sampler_handle).and_then(|s| s.bindless_index)
    }

    fn create_compute_pipeline(
        &mut self,
        device_handle: DeviceHandle,
        compute_shader: ShaderHandle,
        bind_group_layouts: &[BindGroupLayoutHandle],
    ) -> Result<ComputePipelineHandle> {
        // Compile shader on-demand
        let cs_module = self.ensure_shader_stage_compiled(compute_shader, crate::slang::SlangStage::Compute)?;

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Use bindless pipeline layout if bindless is enabled, otherwise create from user layouts
        let (pipeline_layout, owns_layout) = if logical_device.bindless_enabled {
            // In bindless mode, use the global bindless pipeline layout
            let layout = logical_device.bindless_pipeline_layout
                .context("Bindless enabled but no bindless pipeline layout available")?;
            (layout, false) // Don't own - don't destroy when pipeline is destroyed
        } else {
            // Traditional mode: create pipeline layout from user-provided bind group layouts
            let vk_layouts: Vec<_> = bind_group_layouts
                .iter()
                .filter_map(|h| self.bind_group_layouts.get(h).map(|s| s.layout))
                .collect();

            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&vk_layouts);

            let layout = unsafe { logical_device.device.create_pipeline_layout(&layout_info, None) }
                .context("Failed to create compute pipeline layout")?;
            (layout, true) // Own this layout - destroy when pipeline is destroyed
        };

        // Compute shader stage
        let cs_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(cs_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(cs_stage)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            logical_device.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, e)| anyhow::anyhow!("Failed to create compute pipeline: {:?}", e))?;

        let handle = self.next_compute_pipeline_handle;
        self.next_compute_pipeline_handle += 1;

        self.compute_pipelines.insert(handle, ComputePipelineState {
            device_handle,
            pipeline: pipelines[0],
            layout: pipeline_layout,
            owns_layout,
            parameter_block_layouts: Vec::new(),
        });

        tracing::debug!("Created compute pipeline (handle={}, bindless={})", handle, !owns_layout);
        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline_handle: ComputePipelineHandle) {
        if let Some(pipeline) = self.compute_pipelines.remove(&pipeline_handle) {
            if let Some(logical_device) = self.devices.get(&pipeline.device_handle) {
                unsafe {
                    logical_device.device.device_wait_idle().ok();
                    logical_device.device.destroy_pipeline(pipeline.pipeline, None);
                    // Only destroy layout if we own it (not the global bindless layout)
                    if pipeline.owns_layout {
                        logical_device.device.destroy_pipeline_layout(pipeline.layout, None);
                    }
                }
            }
        }
    }

    fn dispatch_compute(&mut self, device_handle: DeviceHandle, commands: &[ComputeCommand]) -> Result<()> {
        let logical_device = self.devices.get(&device_handle)
            .context("Invalid device handle")?;

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(logical_device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
            .context("Failed to allocate command buffer")?;
        let cmd = cmd_buffers[0];

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
            .context("Failed to begin command buffer")?;

        // Track current pipeline for bind group binding
        let mut current_pipeline: Option<ComputePipelineHandle> = None;
        let mut current_pipeline_layout: Option<vk::PipelineLayout> = None;

        // Process commands
        for command in commands {
            match command {
                ComputeCommand::SetPipeline(handle) => {
                    if let Some(pipeline_state) = self.compute_pipelines.get(handle) {
                        unsafe {
                            logical_device.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::COMPUTE,
                                pipeline_state.pipeline,
                            );

                            // Bind the global bindless descriptor set if enabled
                            if logical_device.bindless_enabled {
                                if let (Some(bindless_set), Some(bindless_layout)) = (
                                    logical_device.bindless_descriptor_set,
                                    logical_device.bindless_pipeline_layout,
                                ) {
                                    logical_device.device.cmd_bind_descriptor_sets(
                                        cmd,
                                        vk::PipelineBindPoint::COMPUTE,
                                        bindless_layout,
                                        0,
                                        std::slice::from_ref(&bindless_set),
                                        &[],
                                    );
                                }
                            }
                        }
                        current_pipeline = Some(*handle);
                        current_pipeline_layout = Some(pipeline_state.layout);
                    }
                }
                ComputeCommand::SetBindGroup { index, bind_group } => {
                    if let Some(bg) = self.bind_groups.get(bind_group) {
                        if logical_device.bindless_enabled {
                            // Bindless mode: push resource indices via push constants
                            if let Some(bindless_layout) = logical_device.bindless_pipeline_layout {
                                let mut indices = types::BindlessIndices::default();
                                
                                // Collect bindless indices from bound resources
                                for (i, (_, resource_ref)) in bg.entries.iter().enumerate() {
                                    if i >= types::MAX_PUSH_CONSTANT_INDICES {
                                        break;
                                    }
                                    indices.indices[i] = match resource_ref {
                                        BindGroupResourceRef::Buffer(h) => {
                                            self.buffers.get(h)
                                                .and_then(|b| b.bindless_index)
                                                .unwrap_or(0)
                                        }
                                        BindGroupResourceRef::Texture(h) => {
                                            self.textures.get(h)
                                                .and_then(|t| t.bindless_index)
                                                .unwrap_or(0)
                                        }
                                        BindGroupResourceRef::Sampler(h) => {
                                            self.samplers.get(h)
                                                .and_then(|s| s.bindless_index)
                                                .unwrap_or(0)
                                        }
                                    };
                                }
                                
                                unsafe {
                                    logical_device.device.cmd_push_constants(
                                        cmd,
                                        bindless_layout,
                                        vk::ShaderStageFlags::COMPUTE,
                                        0,
                                        bytemuck::bytes_of(&indices),
                                    );
                                }
                            }
                        } else if let Some(layout) = current_pipeline_layout {
                            // Traditional mode: bind descriptor sets
                            unsafe {
                                logical_device.device.cmd_bind_descriptor_sets(
                                    cmd,
                                    vk::PipelineBindPoint::COMPUTE,
                                    layout,
                                    *index,
                                    &[bg.descriptor_set],
                                    &[],
                                );
                            }
                        }
                    }
                }
                ComputeCommand::SetPushConstants { buffers } => {
                    // Fully bindless mode: push buffer indices directly (no bind groups needed)
                    if logical_device.bindless_enabled {
                        if let Some(pipeline) = current_pipeline.and_then(|p| self.compute_pipelines.get(&p)) {
                            let mut indices = types::BindlessIndices::default();
                            for (i, buffer_handle) in buffers.iter().enumerate() {
                                if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                                indices.indices[i] = self.buffers.get(buffer_handle)
                                    .and_then(|b| b.bindless_index)
                                    .unwrap_or(0);
                            }
                            unsafe {
                                logical_device.device.cmd_push_constants(
                                    cmd,
                                    pipeline.layout,
                                    vk::ShaderStageFlags::COMPUTE,
                                    0,
                                    bytemuck::bytes_of(&indices),
                                );
                            }
                        }
                    }
                }
                ComputeCommand::Dispatch { workgroups_x, workgroups_y, workgroups_z } => {
                    unsafe {
                        logical_device.device.cmd_dispatch(cmd, *workgroups_x, *workgroups_y, *workgroups_z);
                    }
                }
            }
        }

        // End command buffer
        unsafe { logical_device.device.end_command_buffer(cmd) }
            .context("Failed to end command buffer")?;

        // Submit and wait
        let cmd_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmd_buffers);

        unsafe {
            logical_device.device.queue_submit(logical_device.queue, &[submit_info], vk::Fence::null())
                .context("Failed to submit command buffer")?;
            logical_device.device.queue_wait_idle(logical_device.queue)
                .context("Failed to wait for queue")?;
        }

        // Cleanup
        unsafe {
            logical_device.device.free_command_buffers(logical_device.command_pool, &cmd_buffers);
        }

        Ok(())
    }
}

