//! Graphics pipeline management logic.

use super::types::{self, PipelineState, SharedPipelineTable};
use super::utils::{compare_to_vk, depth_format_to_vk, format_to_vk, topology_to_vk, vertex_format_to_vk};
use super::{DeviceHandle, PipelineHandle};
use crate::types::CompareFunction;
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

/// Shader modules plus raster state for Vulkan graphics pipeline creation.
pub(super) struct VulkanGraphicsPipelineCreateBundle<'a> {
    pub devices: &'a HashMap<DeviceHandle, types::SharedLogicalDevice>,
    pub pipelines: &'a SharedPipelineTable,

    pub device_handle: DeviceHandle,
    pub vs_module: vk::ShaderModule,
    pub fs_module: vk::ShaderModule,
    pub raster: &'a crate::backend::shared::PipelineDesc<'a>,
    pub shader_debug_name: String,
}

/// Create a graphics pipeline without depth testing.
pub(super) fn create(bundle: VulkanGraphicsPipelineCreateBundle<'_>) -> Result<PipelineHandle> {
    let VulkanGraphicsPipelineCreateBundle {
        devices,
        pipelines,
        device_handle,
        vs_module,
        fs_module,
        raster: raster_desc,
        shader_debug_name,
    } = bundle;

    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

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

    // Vertex input — only declare binding 0 when there are actual attributes
    let binding_descs: Vec<vk::VertexInputBindingDescription> = if raster_desc.vertex_layout.attributes.is_empty() {
        Vec::new()
    } else {
        vec![vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(raster_desc.vertex_layout.stride)
            .input_rate(vk::VertexInputRate::VERTEX)]
    };

    let attribute_descs: Vec<_> = raster_desc
        .vertex_layout
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
        .vertex_binding_descriptions(&binding_descs)
        .vertex_attribute_descriptions(&attribute_descs);

    // Input assembly
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(topology_to_vk(raster_desc.topology))
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
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    // Reuse the device's shared bindless pipeline layout
    let layout = logical_device
        .bindless_pipeline_layout
        .context("Bindless pipeline layout required")?;
    let owns_layout = false;

    // Dynamic rendering (core since Vulkan 1.3, mandatory in 1.4)
    let color_format = format_to_vk(raster_desc.target_format);
    let mut rendering_info =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(std::slice::from_ref(&color_format));

    // Pipeline robustness (core in Vulkan 1.4): OOB descriptor access returns zero.
    // vertex_inputs must be covered too; without it the spec requires every vertex
    // attribute fetch to be strictly in-bounds, which the validation layer enforces
    // via VUID-vkCmdDraw-None-02721.
    let mut robustness = vk::PipelineRobustnessCreateInfoEXT::default()
        .storage_buffers(vk::PipelineRobustnessBufferBehaviorEXT::ROBUST_BUFFER_ACCESS_2)
        .uniform_buffers(vk::PipelineRobustnessBufferBehaviorEXT::ROBUST_BUFFER_ACCESS_2)
        .vertex_inputs(vk::PipelineRobustnessBufferBehaviorEXT::ROBUST_BUFFER_ACCESS_2)
        .images(vk::PipelineRobustnessImageBehaviorEXT::ROBUST_IMAGE_ACCESS_2);

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
        .push_next(&mut rendering_info)
        .push_next(&mut robustness);

    let vk_pipelines = unsafe {
        logical_device.device.create_graphics_pipelines(
            logical_device.pipeline_cache,
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e.1))?;

    let handle = pipelines.write().unwrap().alloc_handle();

    pipelines.write().unwrap().entries.insert(
        handle,
        PipelineState {
            device_handle,
            pipeline: vk_pipelines[0],
            layout,
            owns_layout,
            parameter_block_layouts: Vec::new(),
            push_constant_categories: Vec::new(),
            binding_element_strides: Vec::new(),
            shader_debug_name,
        },
    );

    tracing::debug!("Created render pipeline {}", handle);
    Ok(handle)
}

/// Create a graphics pipeline with depth testing support.
pub(super) fn create_with_depth(bundle: VulkanGraphicsPipelineCreateBundle<'_>) -> Result<PipelineHandle> {
    let VulkanGraphicsPipelineCreateBundle {
        devices,
        pipelines,
        device_handle,
        vs_module,
        fs_module,
        raster: raster_desc,
        shader_debug_name,
    } = bundle;

    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

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

    // Vertex input — only declare binding 0 when there are actual attributes
    let binding_descs: Vec<vk::VertexInputBindingDescription> = if raster_desc.vertex_layout.attributes.is_empty() {
        Vec::new()
    } else {
        vec![vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(raster_desc.vertex_layout.stride)
            .input_rate(vk::VertexInputRate::VERTEX)]
    };

    let attribute_descs: Vec<_> = raster_desc
        .vertex_layout
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
        .vertex_binding_descriptions(&binding_descs)
        .vertex_attribute_descriptions(&attribute_descs);

    // Input assembly
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(topology_to_vk(raster_desc.topology))
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
    let depth_stencil_state = if let Some(ds) = raster_desc.depth_stencil {
        vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(ds.depth_write_enabled || ds.depth_compare != CompareFunction::Always)
            .depth_write_enable(ds.depth_write_enabled)
            .depth_compare_op(compare_to_vk(ds.depth_compare))
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
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    // Pipeline layout - always use bindless with resource slots
    let bindless_set_layout = logical_device
        .bindless_descriptor_set_layout
        .context("Bindless descriptor set layout required")?;

    let all_layouts = vec![bindless_set_layout];

    // Vulkan push constant range for the packed 128-byte PushLayout
    let slot_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::ALL,
        offset: 0,
        size: types::TOTAL_PUSH_BYTES as u32,
    };

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&all_layouts)
        .push_constant_ranges(std::slice::from_ref(&slot_range));

    let layout = unsafe { logical_device.device.create_pipeline_layout(&layout_info, None) }
        .context("Failed to create bindless pipeline layout")?;
    let owns_layout = true;

    // Dynamic rendering (core since Vulkan 1.3, mandatory in 1.4)
    let color_format = format_to_vk(raster_desc.target_format);
    let depth_format_vk = raster_desc
        .depth_stencil
        .map(|ds| depth_format_to_vk(ds.format))
        .unwrap_or(vk::Format::UNDEFINED);

    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(std::slice::from_ref(&color_format))
        .depth_attachment_format(depth_format_vk);

    // Pipeline robustness (core in Vulkan 1.4): OOB descriptor access returns zero.
    // vertex_inputs must be covered too; without it the spec requires every vertex
    // attribute fetch to be strictly in-bounds, which the validation layer enforces
    // via VUID-vkCmdDraw-None-02721.
    let mut robustness = vk::PipelineRobustnessCreateInfoEXT::default()
        .storage_buffers(vk::PipelineRobustnessBufferBehaviorEXT::ROBUST_BUFFER_ACCESS_2)
        .uniform_buffers(vk::PipelineRobustnessBufferBehaviorEXT::ROBUST_BUFFER_ACCESS_2)
        .vertex_inputs(vk::PipelineRobustnessBufferBehaviorEXT::ROBUST_BUFFER_ACCESS_2)
        .images(vk::PipelineRobustnessImageBehaviorEXT::ROBUST_IMAGE_ACCESS_2);

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
        .push_next(&mut rendering_info)
        .push_next(&mut robustness);

    let vk_pipelines = unsafe {
        logical_device.device.create_graphics_pipelines(
            logical_device.pipeline_cache,
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e.1))?;

    let handle = pipelines.write().unwrap().alloc_handle();

    pipelines.write().unwrap().entries.insert(
        handle,
        PipelineState {
            device_handle,
            pipeline: vk_pipelines[0],
            layout,
            owns_layout,
            parameter_block_layouts: Vec::new(),
            push_constant_categories: Vec::new(),
            binding_element_strides: Vec::new(),
            shader_debug_name,
        },
    );

    tracing::debug!("Created pipeline with depth stencil (handle={})", handle);
    Ok(handle)
}

/// Destroy a graphics pipeline and clean up GPU resources.
pub(super) fn destroy(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    pipelines: &SharedPipelineTable,
    pipeline_handle: PipelineHandle,
) {
    if let Some(pipeline) = pipelines.write().unwrap().entries.remove(&pipeline_handle) {
        if let Some(device) = devices.get(&pipeline.device_handle) {
            unsafe {
                // Wait for all in-flight work to finish before freeing a pipeline
                // that may still be referenced by an in-flight command buffer.
                let _ = device.synchronized_device_wait_idle();
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
