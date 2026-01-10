//! Vulkan backend implementation.
//!
//! Targets Vulkan 1.3+ with dynamic rendering.

use super::*;
use crate::types::Color;
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::ffi::CStr;

/// Vulkan backend.
pub struct VulkanBackend {
    _entry: ash::Entry,
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
}

struct PhysicalDeviceInfo {
    handle: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    adapter_id: u32,
}

struct LogicalDevice {
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    adapter_id: u32,
    queue: vk::Queue,
    queue_family: u32,
    command_pool: vk::CommandPool,
    // Frame rendering state
    frame_state: Option<FrameState>,
}

struct FrameState {
    width: u32,
    height: u32,
    format: TextureFormat,
    // Render target
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    // Staging buffer for readback
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    // Command buffer
    command_buffer: vk::CommandBuffer,
}

struct BufferState {
    device_handle: DeviceHandle,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

struct ShaderState {
    device_handle: DeviceHandle,
    slang_source: String,
    // Cached compiled modules for each stage
    vertex_module: Option<vk::ShaderModule>,
    fragment_module: Option<vk::ShaderModule>,
}

struct PipelineState {
    device_handle: DeviceHandle,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

struct BindGroupLayoutState {
    device_handle: DeviceHandle,
    layout: vk::DescriptorSetLayout,
}

struct BindGroupState {
    device_handle: DeviceHandle,
    descriptor_set: vk::DescriptorSet,
    pool: vk::DescriptorPool,
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

        // Create instance with Vulkan 1.3
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"rag")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"rag")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

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
            _entry: entry,
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
        })
    }

    fn find_memory_type(
        &self,
        physical_device: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let mem_props = unsafe { self.instance.get_physical_device_memory_properties(physical_device) };

        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (mem_props.memory_types[i as usize].property_flags & properties) == properties
            {
                return Some(i);
            }
        }
        None
    }

    fn format_to_vk(format: TextureFormat) -> vk::Format {
        match format {
            TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
            TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
            TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        }
    }

    fn vertex_format_to_vk(format: VertexFormat) -> vk::Format {
        match format {
            VertexFormat::Float32 => vk::Format::R32_SFLOAT,
            VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
            VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
            VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
            VertexFormat::Uint32 => vk::Format::R32_UINT,
            VertexFormat::Sint32 => vk::Format::R32_SINT,
            VertexFormat::Uint8x4 => vk::Format::R8G8B8A8_UINT,
            VertexFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
        }
    }

    fn topology_to_vk(topology: PrimitiveTopology) -> vk::PrimitiveTopology {
        match topology {
            PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
            PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
            PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
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

        // Create logical device
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
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
                frame_state: None,
            },
        );

        tracing::info!("Created Vulkan device {} for adapter {}", handle, adapter_id);
        Ok(handle)
    }

    fn destroy_device(&mut self, device_handle: DeviceHandle) {
        if let Some(logical_device) = self.devices.remove(&device_handle) {
            unsafe {
                logical_device.device.device_wait_idle().ok();

                // Destroy frame state
                if let Some(frame) = logical_device.frame_state {
                    logical_device.device.destroy_image_view(frame.image_view, None);
                    logical_device.device.destroy_image(frame.image, None);
                    logical_device.device.free_memory(frame.image_memory, None);
                    logical_device.device.destroy_buffer(frame.staging_buffer, None);
                    logical_device.device.free_memory(frame.staging_memory, None);
                }

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
                    .format(Self::vertex_format_to_vk(attr.format))
                    .offset(attr.offset)
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_desc))
            .vertex_attribute_descriptions(&attribute_descs);

        // Input assembly
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(Self::topology_to_vk(topology))
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
        let color_format = Self::format_to_vk(target_format);
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

    fn begin_frame(&mut self, device_handle: DeviceHandle, width: u32, height: u32, format: TextureFormat) -> Result<()> {
        // First, check if we need to recreate and get physical device if needed
        let (needs_recreate, physical_device) = {
            let logical_device = self
                .devices
                .get(&device_handle)
                .context("Invalid device handle")?;

            let needs_recreate = match &logical_device.frame_state {
                Some(state) => state.width != width || state.height != height || state.format != format,
                None => true,
            };

            (needs_recreate, logical_device.physical_device)
        };

        if needs_recreate {
            // Find memory types before mutably borrowing
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
                .get_mut(&device_handle)
                .context("Invalid device handle")?;

            // Destroy old frame state
            if let Some(old_state) = logical_device.frame_state.take() {
                unsafe {
                    logical_device.device.device_wait_idle()?;
                    logical_device.device.destroy_image_view(old_state.image_view, None);
                    logical_device.device.destroy_image(old_state.image, None);
                    logical_device.device.free_memory(old_state.image_memory, None);
                    logical_device.device.destroy_buffer(old_state.staging_buffer, None);
                    logical_device.device.free_memory(old_state.staging_memory, None);
                }
            }

            // Create render target image
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(Self::format_to_vk(format))
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
                .format(Self::format_to_vk(format))
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let image_view = unsafe { logical_device.device.create_image_view(&view_info, None) }
                .context("Failed to create render target view")?;

            // Create staging buffer for readback
            let buffer_size = (width * height * format.bytes_per_pixel()) as u64;
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

            // Allocate command buffer
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(logical_device.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
                .context("Failed to allocate command buffer")?;

            logical_device.frame_state = Some(FrameState {
                width,
                height,
                format,
                image,
                image_memory,
                image_view,
                staging_buffer,
                staging_memory,
                command_buffer: command_buffers[0],
            });

            tracing::debug!("Created frame resources {}x{}", width, height);
        }

        Ok(())
    }

    fn execute_commands(&mut self, device_handle: DeviceHandle, commands: &[RenderCommand]) -> Result<()> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let frame = logical_device
            .frame_state
            .as_ref()
            .context("begin_frame not called")?;

        let cmd = frame.command_buffer;

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
            .image(frame.image)
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
            .image_view(frame.image_view)
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
                extent: vk::Extent2D {
                    width: frame.width,
                    height: frame.height,
                },
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));

        unsafe { logical_device.device.cmd_begin_rendering(cmd, &rendering_info) };

        // Set viewport and scissor
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: frame.width as f32,
            height: frame.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe { logical_device.device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport)) };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: frame.width,
                height: frame.height,
            },
        };
        unsafe { logical_device.device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor)) };

        // Execute commands
        for command in commands {
            match command {
                RenderCommand::Clear(color) => {
                    // Clear is handled by the load_op, but we can do mid-pass clears if needed
                    let _ = color; // Currently handled by initial clear
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
                    if let Some(buf) = self.buffers.get(buffer) {
                        unsafe {
                            logical_device.device.cmd_bind_vertex_buffers(
                                cmd,
                                *slot,
                                std::slice::from_ref(&buf.buffer),
                                std::slice::from_ref(offset),
                            );
                        }
                    }
                }
                RenderCommand::SetBindGroup { index, bind_group } => {
                    if let (Some(bg), Some(pipeline)) = (
                        self.bind_groups.get(bind_group),
                        self.pipelines.values().next(), // Get current pipeline layout
                    ) {
                        unsafe {
                            logical_device.device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline.layout,
                                *index,
                                std::slice::from_ref(&bg.descriptor_set),
                                &[],
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
            }
        }

        // End dynamic rendering
        unsafe { logical_device.device.cmd_end_rendering(cmd) };

        // Transition image for transfer
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(frame.image)
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
                width: frame.width,
                height: frame.height,
                depth: 1,
            });

        unsafe {
            logical_device.device.cmd_copy_image_to_buffer(
                cmd,
                frame.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                frame.staging_buffer,
                std::slice::from_ref(&region),
            );
        }

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

        Ok(())
    }

    fn end_frame(&mut self, device_handle: DeviceHandle, output: &mut [u8]) -> Result<()> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let frame = logical_device
            .frame_state
            .as_ref()
            .context("begin_frame not called")?;

        let expected_size = (frame.width * frame.height * frame.format.bytes_per_pixel()) as usize;
        if output.len() < expected_size {
            anyhow::bail!(
                "Output buffer too small: {} < {}",
                output.len(),
                expected_size
            );
        }

        // Read back from staging buffer
        unsafe {
            let ptr = logical_device
                .device
                .map_memory(
                    frame.staging_memory,
                    0,
                    expected_size as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .context("Failed to map staging buffer")?;

            std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), expected_size);

            logical_device.device.unmap_memory(frame.staging_memory);
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
                    .format(Self::vertex_format_to_vk(attr.format))
                    .offset(attr.offset)
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_desc))
            .vertex_attribute_descriptions(&attribute_descs);

        // Input assembly
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(Self::topology_to_vk(topology))
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
        let color_format = Self::format_to_vk(target_format);
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
}

