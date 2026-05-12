//! Shared render command recording logic.
//!
//! Used by both render_target and surface to avoid code duplication.

use super::super::shared;
use super::super::{BufferHandle, PipelineHandle, RenderCommand};
use super::types::{PipelineState, PushLayout, RESOURCE_SLOT_BUFFER};
use super::utils::index_format_to_mtl;
use crate::types::IndexFormat;
use ::metal as mtl;
use anyhow::Result;
use mtl::MTLPrimitiveType;
use std::collections::HashMap;

/// Record render commands into a Metal render command encoder.
pub(super) fn record(
    encoder: &mtl::RenderCommandEncoderRef,
    commands: &[RenderCommand],
    pipelines: &HashMap<PipelineHandle, PipelineState>,
    buffers: &HashMap<BufferHandle, super::types::BufferState>,
) -> Result<()> {
    let mut current_index_buffer: Option<(BufferHandle, u64, IndexFormat)> = None;
    let mut current_primitive_type = MTLPrimitiveType::Triangle;
    let mut current_pipeline_handle: Option<PipelineHandle> = None;

    for cmd in commands {
        match cmd {
            RenderCommand::Clear(_) | RenderCommand::ClearDepth(_) => {}
            RenderCommand::SetPipeline(pipeline_handle) => {
                current_pipeline_handle = Some(*pipeline_handle);
                if let Some(pipeline) = pipelines.get(pipeline_handle) {
                    encoder.set_render_pipeline_state(&pipeline.pipeline);
                    current_primitive_type = pipeline.primitive_type;
                    if let Some(ds) = &pipeline.depth_stencil {
                        encoder.set_depth_stencil_state(ds);
                    }
                }
            }
            RenderCommand::SetVertexBuffer {
                slot,
                buffer,
                offset,
            } => {
                if let Some(buf) = buffers.get(buffer) {
                    let metal_slot = (*slot as u64) + super::types::VERTEX_BUFFER_START_SLOT;
                    encoder.set_vertex_buffer(metal_slot, Some(&buf.buffer), *offset);
                } else {
                    tracing::error!(
                        "SetVertexBuffer: buffer handle {buffer} not found; vertex binding will be missing"
                    );
                }
            }
            RenderCommand::SetIndexBuffer {
                buffer,
                offset,
                format,
            } => {
                current_index_buffer = Some((*buffer, *offset, *format));
            }
            RenderCommand::BindResources {
                buffers: buf_handles,
            } => {
                if crate::slang::layout_validation_enabled() {
                    if let Some(pipeline) = current_pipeline_handle.and_then(|h| pipelines.get(&h))
                    {
                        if !pipeline.binding_element_strides.is_empty() {
                            let actual: Vec<Option<u32>> = buf_handles
                                .iter()
                                .map(|h| buffers.get(h).and_then(|b| b.element_stride))
                                .collect();
                            crate::backend::validate_binding_strides(
                                &actual,
                                &pipeline.binding_element_strides,
                                &pipeline.shader_debug_name,
                            )?;
                        }
                    }
                }
                let mut layout = PushLayout::default();
                shared::fill_bindless(
                    &mut layout,
                    buf_handles
                        .iter()
                        .map(|h| buffers.get(h).map(|b| b.arg_buffer_index).unwrap_or(0)),
                );
                let layout_bytes = layout.as_bytes();
                encoder.set_vertex_bytes(
                    RESOURCE_SLOT_BUFFER,
                    layout_bytes.len() as u64,
                    layout_bytes.as_ptr() as *const _,
                );
                encoder.set_fragment_bytes(
                    RESOURCE_SLOT_BUFFER,
                    layout_bytes.len() as u64,
                    layout_bytes.as_ptr() as *const _,
                );
            }
            RenderCommand::BindResourcesRaw {
                indices: raw_indices,
                user: raw_user,
            } => {
                let mut layout = PushLayout::default();
                shared::fill_raw(&mut layout, raw_indices, raw_user);
                let layout_bytes = layout.as_bytes();
                encoder.set_vertex_bytes(
                    RESOURCE_SLOT_BUFFER,
                    layout_bytes.len() as u64,
                    layout_bytes.as_ptr() as *const _,
                );
                encoder.set_fragment_bytes(
                    RESOURCE_SLOT_BUFFER,
                    layout_bytes.len() as u64,
                    layout_bytes.as_ptr() as *const _,
                );
            }
            RenderCommand::BindResourcesTyped {
                handles: typed_handles,
            } => {
                if let Some(pipeline) = current_pipeline_handle.and_then(|h| pipelines.get(&h)) {
                    crate::backend::validate_typed_push_constants(
                        typed_handles,
                        &pipeline.push_constant_categories,
                        &pipeline.shader_debug_name,
                    )?;
                }
                let mut layout = PushLayout::default();
                shared::fill_typed(&mut layout, typed_handles.iter().copied());
                let layout_bytes = layout.as_bytes();
                encoder.set_vertex_bytes(
                    RESOURCE_SLOT_BUFFER,
                    layout_bytes.len() as u64,
                    layout_bytes.as_ptr() as *const _,
                );
                encoder.set_fragment_bytes(
                    RESOURCE_SLOT_BUFFER,
                    layout_bytes.len() as u64,
                    layout_bytes.as_ptr() as *const _,
                );
            }
            RenderCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => {
                if *first_instance != 0 {
                    tracing::warn!("Metal backend: first_instance != 0 not supported");
                }
                encoder.draw_primitives_instanced(
                    current_primitive_type,
                    *first_vertex as u64,
                    *vertex_count as u64,
                    *instance_count as u64,
                );
            }
            RenderCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => {
                if *first_instance != 0 || *base_vertex != 0 {
                    tracing::warn!("Metal backend: first_instance/base_vertex != 0 not supported");
                }
                if let Some((buffer_handle, offset, format)) = current_index_buffer {
                    if let Some(buf) = buffers.get(&buffer_handle) {
                        let index_type = index_format_to_mtl(format);
                        let index_offset = offset + (*first_index as u64 * format.size() as u64);
                        encoder.draw_indexed_primitives_instanced(
                            current_primitive_type,
                            *index_count as u64,
                            index_type,
                            &buf.buffer,
                            index_offset,
                            *instance_count as u64,
                        );
                    } else {
                        tracing::error!(
                            "DrawIndexed: index buffer handle {buffer_handle} not found; draw call skipped"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// Create a render pass descriptor for the given texture.
/// Returns a reference from the autorelease pool (valid until pool drains).
pub(super) fn create_render_pass<'a>(
    texture: &mtl::TextureRef,
    depth_texture: Option<&mtl::TextureRef>,
    clear_color: Option<crate::types::Color>,
    clear_depth: Option<f32>,
) -> &'a mtl::RenderPassDescriptorRef {
    let descriptor = mtl::RenderPassDescriptor::new();

    let color_attachment = descriptor
        .color_attachments()
        .object_at(0)
        .expect("Metal render pass descriptor must have at least one color attachment");
    color_attachment.set_texture(Some(texture));

    if let Some(color) = clear_color {
        color_attachment.set_load_action(mtl::MTLLoadAction::Clear);
        color_attachment.set_clear_color(mtl::MTLClearColor::new(
            color.r as f64,
            color.g as f64,
            color.b as f64,
            color.a as f64,
        ));
    } else {
        color_attachment.set_load_action(mtl::MTLLoadAction::Load);
    }
    color_attachment.set_store_action(mtl::MTLStoreAction::Store);

    if let Some(depth) = depth_texture {
        let depth_attachment = descriptor.depth_attachment().expect(
            "Metal render pass descriptor must have a depth attachment when depth texture is set",
        );
        depth_attachment.set_texture(Some(depth));
        if let Some(depth_value) = clear_depth {
            depth_attachment.set_load_action(mtl::MTLLoadAction::Clear);
            depth_attachment.set_clear_depth(depth_value as f64);
        } else {
            depth_attachment.set_load_action(mtl::MTLLoadAction::Load);
        }
        depth_attachment.set_store_action(mtl::MTLStoreAction::Store);
    }

    descriptor
}
