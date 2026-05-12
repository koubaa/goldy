//! Shared render command recording logic.
//!
//! This module contains `record_render_commands` which is used by both
//! `render_to_target` and `surface_render` to avoid code duplication.

use super::super::shared;
use super::types::{self, PushLayout};
use super::utils::index_format_to_vk;
use super::{BufferHandle, PipelineHandle, RenderCommand};
use ash::vk;

/// Record render commands into a command buffer.
/// This is shared between render_to_target and surface_render to avoid duplication.
pub(super) fn record(
    cmd: vk::CommandBuffer,
    commands: &[RenderCommand],
    logical_device: &types::LogicalDevice,
    pipelines: &std::collections::HashMap<PipelineHandle, types::PipelineState>,
    buffers: &std::collections::HashMap<BufferHandle, types::BufferState>,
    current_pipeline: &mut Option<PipelineHandle>,
) -> anyhow::Result<()> {
    for command in commands {
        match command {
            RenderCommand::Clear(_) => {
                // Already handled via load op
            }
            RenderCommand::ClearDepth(_) => {
                // TODO: Implement depth clear when depth buffer is supported
            }
            RenderCommand::SetPipeline(pipeline_handle) => {
                *current_pipeline = Some(*pipeline_handle);
                if let Some(pipeline) = pipelines.get(pipeline_handle) {
                    unsafe {
                        logical_device.device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.pipeline,
                        );

                        // Bind the global bindless descriptor set.
                        // Use the pipeline's layout (not bindless_pipeline_layout alone)
                        // when layouts combine bindless + user sets.
                        if let Some(bindless_set) = logical_device.bindless_descriptor_set {
                            logical_device.device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline.layout,
                                0,
                                std::slice::from_ref(&bindless_set),
                                &[],
                            );
                        }
                    }
                }
            }
            RenderCommand::SetVertexBuffer {
                slot,
                buffer,
                offset,
            } => {
                if let Some(buf_state) = buffers.get(buffer) {
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
            RenderCommand::BindResources {
                buffers: buf_handles,
            } => {
                if let Some(pipeline) = current_pipeline.and_then(|p| pipelines.get(&p)) {
                    if crate::slang::layout_validation_enabled()
                        && !pipeline.binding_element_strides.is_empty()
                    {
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
                    let mut layout = PushLayout::default();
                    shared::fill_bindless(
                        &mut layout,
                        buf_handles
                            .iter()
                            .map(|h| buffers.get(h).and_then(|b| b.bindless_index).unwrap_or(0)),
                    );
                    unsafe {
                        logical_device.device.cmd_push_constants(
                            cmd,
                            pipeline.layout,
                            vk::ShaderStageFlags::ALL,
                            0,
                            layout.as_bytes(),
                        );
                    }
                }
            }
            RenderCommand::BindResourcesRaw {
                indices: raw_indices,
                user: raw_user,
            } => {
                if let Some(pipeline) = current_pipeline.and_then(|p| pipelines.get(&p)) {
                    let mut layout = PushLayout::default();
                    shared::fill_raw(&mut layout, raw_indices, raw_user);
                    unsafe {
                        logical_device.device.cmd_push_constants(
                            cmd,
                            pipeline.layout,
                            vk::ShaderStageFlags::ALL,
                            0,
                            layout.as_bytes(),
                        );
                    }
                }
            }
            RenderCommand::BindResourcesTyped {
                handles: typed_handles,
            } => {
                if let Some(pipeline) = current_pipeline.and_then(|p| pipelines.get(&p)) {
                    crate::backend::validate_typed_push_constants(
                        typed_handles,
                        &pipeline.push_constant_categories,
                        &pipeline.shader_debug_name,
                    )?;
                    let mut layout = PushLayout::default();
                    shared::fill_typed(&mut layout, typed_handles.iter().copied());
                    unsafe {
                        logical_device.device.cmd_push_constants(
                            cmd,
                            pipeline.layout,
                            vk::ShaderStageFlags::ALL,
                            0,
                            layout.as_bytes(),
                        );
                    }
                }
            }
            RenderCommand::SetIndexBuffer {
                buffer,
                offset,
                format,
            } => {
                if let Some(buf_state) = buffers.get(buffer) {
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
            } => unsafe {
                logical_device.device.cmd_draw(
                    cmd,
                    *vertex_count,
                    *instance_count,
                    *first_vertex,
                    *first_instance,
                );
            },
            RenderCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => unsafe {
                logical_device.device.cmd_draw_indexed(
                    cmd,
                    *index_count,
                    *instance_count,
                    *first_index,
                    *base_vertex,
                    *first_instance,
                );
            },
        }
    }
    Ok(())
}
