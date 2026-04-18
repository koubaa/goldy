//! Graphics pipeline management logic.

use super::super::{DeviceHandle, PipelineHandle, ShaderHandle};
use super::types::{MetalState, PipelineState};
use super::utils::{
    compare_to_mtl, depth_format_to_mtl, format_to_mtl, topology_to_mtl, vertex_format_to_mtl,
};
use crate::slang::SlangStage;
use crate::types::PrimitiveTopology;
use crate::types::{DepthStencilState, TextureFormat, VertexBufferLayout};
use ::metal as mtl;
use anyhow::{Context, Result};

/// Create a graphics pipeline (with optional depth stencil).
#[allow(clippy::too_many_arguments)]
pub(super) fn create_with_depth(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    vertex_shader: ShaderHandle,
    fragment_shader: ShaderHandle,
    vertex_layout: &VertexBufferLayout,
    topology: PrimitiveTopology,
    target_format: TextureFormat,
    depth_stencil: Option<&DepthStencilState>,
) -> Result<PipelineHandle> {
    super::shader::ensure_stage_compiled(
        &state.slang_compiler,
        &state.devices,
        &mut state.shaders,
        vertex_shader,
        SlangStage::Vertex,
    )?;
    super::shader::ensure_stage_compiled(
        &state.slang_compiler,
        &state.devices,
        &mut state.shaders,
        fragment_shader,
        SlangStage::Fragment,
    )?;

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let vs_shader = state
        .shaders
        .get(&vertex_shader)
        .context("Invalid vertex shader")?;
    let fs_shader = state
        .shaders
        .get(&fragment_shader)
        .context("Invalid fragment shader")?;

    let vs_library = vs_shader.vertex_library.as_ref().unwrap();
    let fs_library = fs_shader.fragment_library.as_ref().unwrap();

    let vs_function = vs_library
        .get_function("vs_main", None)
        .map_err(|e| anyhow::anyhow!("Failed to get vertex function: {}", e))?;

    let fs_function = fs_library
        .get_function("fs_main", None)
        .map_err(|e| anyhow::anyhow!("Failed to get fragment function: {}", e))?;

    let descriptor = mtl::RenderPipelineDescriptor::new();
    descriptor.set_vertex_function(Some(&vs_function));
    descriptor.set_fragment_function(Some(&fs_function));

    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(format_to_mtl(target_format));

    if !vertex_layout.attributes.is_empty() {
        let vertex_descriptor = mtl::VertexDescriptor::new();
        let layout = vertex_descriptor
            .layouts()
            .object_at(super::types::VERTEX_BUFFER_START_SLOT)
            .unwrap();
        layout.set_stride(vertex_layout.stride as u64);
        layout.set_step_function(mtl::MTLVertexStepFunction::PerVertex);

        for attr in &vertex_layout.attributes {
            let attr_desc = vertex_descriptor
                .attributes()
                .object_at(attr.location as u64)
                .unwrap();
            attr_desc.set_format(vertex_format_to_mtl(attr.format));
            attr_desc.set_offset(attr.offset as u64);
            attr_desc.set_buffer_index(super::types::VERTEX_BUFFER_START_SLOT);
        }

        descriptor.set_vertex_descriptor(Some(vertex_descriptor));
    }

    let depth_stencil_state = if let Some(ds) = depth_stencil {
        descriptor.set_depth_attachment_pixel_format(depth_format_to_mtl(ds.format));

        let ds_descriptor = mtl::DepthStencilDescriptor::new();
        ds_descriptor.set_depth_compare_function(compare_to_mtl(ds.depth_compare));
        ds_descriptor.set_depth_write_enabled(ds.depth_write_enabled);

        Some(
            logical_device
                .device
                .new_depth_stencil_state(&ds_descriptor),
        )
    } else {
        None
    };

    let pipeline = logical_device
        .device
        .new_render_pipeline_state(&descriptor)
        .map_err(|e| anyhow::anyhow!("Failed to create render pipeline: {}", e))?;

    // Fragment shader is the primary source of push-constant access patterns
    // (vertex shaders rarely use goldy_dyn_*); fall back to vertex if the
    // fragment shader has no reflection.
    let push_constant_categories = fs_shader
        .reflection
        .as_ref()
        .map(|r| r.push_constant_categories.clone())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            vs_shader
                .reflection
                .as_ref()
                .map(|r| r.push_constant_categories.clone())
        })
        .unwrap_or_default();

    let handle = state.next_pipeline_handle;
    state.next_pipeline_handle += 1;

    state.pipelines.insert(
        handle,
        PipelineState {
            device_handle,
            pipeline,
            depth_stencil: depth_stencil_state,
            primitive_type: topology_to_mtl(topology),
            push_constant_categories,
            shader_debug_name: "fs_main/vs_main".to_string(),
        },
    );

    tracing::debug!(
        "Created render pipeline {} with topology {:?}",
        handle,
        topology
    );
    Ok(handle)
}

/// Destroy a graphics pipeline.
pub(super) fn destroy(state: &mut MetalState, pipeline_handle: PipelineHandle) {
    state.pipelines.remove(&pipeline_handle);
}
