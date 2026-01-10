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

        // Enable Vulkan 1.3 features (dynamic rendering)
        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_13_features);

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
            },
        );

        tracing::info!("Created Vulkan device {} for adapter {}", handle, adapter_id);
        Ok(handle)
    }

    fn destroy_device(&mut self, device_handle: DeviceHandle) {
        if let Some(logical_device) = self.devices.remove(&device_handle) {
            unsafe {
                logical_device.device.device_wait_idle().ok();

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

                // Destroy pipelines owned by this device
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
                        if pipeline.layout != vk::PipelineLayout::null() {
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
                        if let Some(staging_buffer) = target.staging_buffer {
                            logical_device.device.destroy_buffer(staging_buffer, None);
                        }
                        if let Some(staging_memory) = target.staging_memory {
                            logical_device.device.free_memory(staging_memory, None);
                        }
                    }
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

    fn create_buffer(&mut self, device_handle: DeviceHandle, size: u64, usage: BufferUsage) -> Result<BufferHandle> {
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

        self.buffers.insert(
            handle,
            BufferState {
                device_handle,
                buffer,
                memory,
                size,
            },
        );

        Ok(handle)
    }

    fn destroy_buffer(&mut self, buffer_handle: BufferHandle) {
        if let Some(buffer) = self.buffers.remove(&buffer_handle) {
            if let Some(device) = self.devices.get(&buffer.device_handle) {
                unsafe {
                    device.device.destroy_buffer(buffer.buffer, None);
                    device.device.free_memory(buffer.memory, None);
                }
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

    fn create_shader(&mut self, device_handle: DeviceHandle, slang_source: &str) -> Result<ShaderHandle> {
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
                vertex_module: None,
                fragment_module: None,
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

        // Pipeline layout (empty for now)
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        let layout = unsafe { logical_device.device.create_pipeline_layout(&layout_info, None) }
            .context("Failed to create pipeline layout")?;

        // Dynamic rendering info (Vulkan 1.3)
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
                    if pipeline.layout != vk::PipelineLayout::null() {
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

        // Find the first Clear command to use as the initial clear color
        let clear_color = commands
            .iter()
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
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe { logical_device.device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport)) };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        unsafe { logical_device.device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor)) };

        // Execute render commands
        for command in commands {
            match command {
                RenderCommand::Clear(_) => {
                    // Already handled via load op
                }
                RenderCommand::SetPipeline(pipeline_handle) => {
                    if let Some(pipeline) = self.pipelines.get(pipeline_handle) {
                        unsafe {
                            logical_device.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline.pipeline,
                            );
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
                        // Find the pipeline layout for this bind group
                        // For now, we need to iterate pipelines to find one with a layout
                        // This is a simplification - in production we'd track the current pipeline
                        if let Some(pipeline) = self.pipelines.values().next() {
                            unsafe {
                                logical_device.device.cmd_bind_descriptor_sets(
                                    cmd,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    pipeline.layout,
                                    *index,
                                    std::slice::from_ref(&bg_state.descriptor_set),
                                    &[],
                                );
                            }
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

        let bindings: Vec<_> = entries
            .iter()
            .map(|e| {
                let stage_flags = if e.visibility.0 & ShaderStages::VERTEX.0 != 0 && e.visibility.0 & ShaderStages::FRAGMENT.0 != 0 {
                    vk::ShaderStageFlags::ALL_GRAPHICS
                } else if e.visibility.0 & ShaderStages::VERTEX.0 != 0 {
                    vk::ShaderStageFlags::VERTEX
                } else {
                    vk::ShaderStageFlags::FRAGMENT
                };

                let descriptor_type = match &e.ty {
                    BindingType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                    BindingType::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
                };

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

        // Create a descriptor pool for this bind group
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(entries.len() as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
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
            .set_layouts(std::slice::from_ref(&layout_state.layout));

        let descriptor_sets = unsafe { logical_device.device.allocate_descriptor_sets(&alloc_info) }
            .context("Failed to allocate descriptor set")?;

        let descriptor_set = descriptor_sets[0];

        // Write descriptors
        let buffer_infos: Vec<_> = entries
            .iter()
            .filter_map(|e| match &e.resource {
                BindingResource::Buffer { buffer, offset, size } => {
                    self.buffers.get(buffer).map(|b| (e.binding, b.buffer, *offset, *size))
                }
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
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&vk_buffer_infos[idx]))
            })
            .collect();

        unsafe { logical_device.device.update_descriptor_sets(&writes, &[]) };

        let handle = self.next_bind_group_handle;
        self.next_bind_group_handle += 1;

        self.bind_groups.insert(handle, BindGroupState {
            device_handle,
            descriptor_set,
            pool,
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

        // Collect descriptor set layouts
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

        // Pipeline layout with descriptor set layouts
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&vk_layouts);
        let layout = unsafe { logical_device.device.create_pipeline_layout(&layout_info, None) }
            .context("Failed to create pipeline layout")?;

        // Dynamic rendering info (Vulkan 1.3)
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
            },
        );

        tracing::debug!("Created render pipeline with layout {}", handle);
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
        let (device_handle, current_frame, swapchain, in_flight_fence, image_available_semaphore) = {
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
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe { logical_device.device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport)) };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        unsafe { logical_device.device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor)) };

        // Execute render commands
        for command in commands {
            match command {
                RenderCommand::Clear(_) => { /* Already handled */ }
                RenderCommand::SetPipeline(pipeline_handle) => {
                    if let Some(pipeline) = self.pipelines.get(pipeline_handle) {
                        unsafe {
                            logical_device.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline.pipeline,
                            );
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
                        if let Some(pipeline) = self.pipelines.values().next() {
                            unsafe {
                                logical_device.device.cmd_bind_descriptor_sets(
                                    cmd,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    pipeline.layout,
                                    *index,
                                    std::slice::from_ref(&bg_state.descriptor_set),
                                    &[],
                                );
                            }
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
        let surface_state = self.surfaces.get_mut(&surface_handle).unwrap();
        surface_state.current_image_index = None;
        surface_state.current_frame = (surface_state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

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
}

