//! Shared render command recording logic.
//!
//! Used by both render_target and surface to avoid code duplication.

use super::super::shared;
use super::super::{BufferHandle, DeviceHandle, PipelineHandle, RenderCommand};
use super::types::{LogicalDevice, PipelineState, PushLayout, RESOURCE_SLOT_BUFFER};
use super::utils::index_format_to_mtl;
use crate::types::IndexFormat;
use ::metal as mtl;
use anyhow::Result;
use mtl::MTLPrimitiveType;
use std::collections::HashMap;

/// metal-rs 0.33 omits object/mesh bits on [`MTLRenderStages`] (macOS 13 / iOS 16).
const RENDER_STAGE_OBJECT: u64 = 1 << 3;
const RENDER_STAGE_MESH: u64 = 1 << 4;

pub(super) fn commands_use_mesh(commands: &[RenderCommand]) -> bool {
    commands.iter().any(|c| matches!(c, RenderCommand::DispatchMesh { .. }))
}

pub(super) fn render_stages_for_pass(is_mesh: bool) -> mtl::MTLRenderStages {
    let mut bits = mtl::MTLRenderStages::Vertex.bits() | mtl::MTLRenderStages::Fragment.bits();
    if is_mesh {
        bits |= RENDER_STAGE_OBJECT | RENDER_STAGE_MESH;
    }
    mtl::MTLRenderStages::from_bits_truncate(bits)
}

fn bind_goldy_argument_buffer(
    encoder: &mtl::RenderCommandEncoderRef,
    argument_buffer: &mtl::BufferRef,
    is_mesh: bool,
) {
    encoder.set_fragment_buffer(0, Some(argument_buffer), 0);
    if is_mesh {
        encoder.set_mesh_buffer(0, Some(argument_buffer), 0);
        encoder.set_object_buffer(0, Some(argument_buffer), 0);
    } else {
        encoder.set_vertex_buffer(0, Some(argument_buffer), 0);
    }
}

fn bind_push_bytes(encoder: &mtl::RenderCommandEncoderRef, layout_bytes: &[u8], is_mesh: bool) {
    encoder.set_fragment_bytes(
        RESOURCE_SLOT_BUFFER,
        layout_bytes.len() as u64,
        layout_bytes.as_ptr() as *const _,
    );
    if is_mesh {
        encoder.set_mesh_bytes(
            RESOURCE_SLOT_BUFFER,
            layout_bytes.len() as u64,
            layout_bytes.as_ptr() as *const _,
        );
        encoder.set_object_bytes(
            RESOURCE_SLOT_BUFFER,
            layout_bytes.len() as u64,
            layout_bytes.as_ptr() as *const _,
        );
    } else {
        encoder.set_vertex_bytes(
            RESOURCE_SLOT_BUFFER,
            layout_bytes.len() as u64,
            layout_bytes.as_ptr() as *const _,
        );
    }
}

/// Heaps, bindless argument buffer, and buffer residency for a render encoder.
pub(super) fn declare_pass_resources(
    encoder: &mtl::RenderCommandEncoderRef,
    logical_device: &LogicalDevice,
    buffers: &HashMap<BufferHandle, super::types::BufferState>,
    device_handle: DeviceHandle,
    is_mesh: bool,
) {
    let render_stages = render_stages_for_pass(is_mesh);
    logical_device
        .heap_allocator
        .lock()
        .unwrap()
        .use_heaps_for_render(encoder, render_stages);
    logical_device
        .texture_heap
        .lock()
        .unwrap()
        .use_heaps_for_render(encoder, render_stages);
    for buf_state in buffers.values() {
        if buf_state.device_handle == device_handle {
            encoder.use_resource_at(
                &buf_state.buffer,
                mtl::MTLResourceUsage::Read | mtl::MTLResourceUsage::Write,
                render_stages,
            );
        }
    }
    {
        let ft = logical_device.frame_table.lock().unwrap();
        encoder.use_resource_at(ft.table_buffer(), mtl::MTLResourceUsage::Read, render_stages);
    }
    bind_goldy_argument_buffer(encoder, &logical_device.argument_buffer, is_mesh);
}

/// Record render commands into a Metal render command encoder.
pub(super) fn record(
    encoder: &mtl::RenderCommandEncoderRef,
    commands: &[RenderCommand],
    pipelines: &HashMap<PipelineHandle, PipelineState>,
    buffers: &HashMap<BufferHandle, super::types::BufferState>,
    prologue_row: Option<u32>,
) -> Result<()> {
    let mut current_index_buffer: Option<(BufferHandle, u64, IndexFormat)> = None;
    let mut current_primitive_type = MTLPrimitiveType::Triangle;
    let mut current_is_mesh = false;
    let mut current_object_tg = mtl::MTLSize {
        width: 0,
        height: 0,
        depth: 0,
    };
    let mut current_mesh_tg = mtl::MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    for cmd in commands {
        match cmd {
            RenderCommand::ClearDepth(_) => {}
            RenderCommand::SetPipeline(pipeline_handle) => {
                if let Some(pipeline) = pipelines.get(pipeline_handle) {
                    encoder.set_render_pipeline_state(&pipeline.pipeline);
                    current_primitive_type = pipeline.primitive_type;
                    current_is_mesh = pipeline.is_mesh;
                    current_object_tg = pipeline.object_threadgroup;
                    current_mesh_tg = pipeline.mesh_threadgroup;
                    if let Some(ds) = &pipeline.depth_stencil {
                        encoder.set_depth_stencil_state(ds);
                    }
                }
            }
            RenderCommand::SetVertexBuffer { slot, buffer, offset } => {
                if let Some(buf) = buffers.get(buffer) {
                    let metal_slot = (*slot as u64) + super::types::VERTEX_BUFFER_START_SLOT;
                    encoder.set_vertex_buffer(metal_slot, Some(&buf.buffer), *offset);
                } else {
                    tracing::error!(
                        "SetVertexBuffer: buffer handle {buffer} not found; vertex binding will be missing"
                    );
                }
            }
            RenderCommand::SetIndexBuffer { buffer, offset, format } => {
                current_index_buffer = Some((*buffer, *offset, *format));
            }
            RenderCommand::BindResources { .. } => {
                anyhow::bail!(
                    "RenderCommand::BindResources must be lowered before Metal record; \
                     use frame_table::prepare_render_commands or lower_render_pass_commands"
                );
            }
            RenderCommand::BindResourcesRaw {
                indices: raw_indices,
                user: raw_user,
                frame_table_base,
            } => {
                if !raw_indices.is_empty() {
                    anyhow::bail!(
                        "BindResourcesRaw with indices in frame-table path: \
                         use frame_table::prepare_render_commands or lower_render_pass_commands"
                    );
                }
                let absolute_base =
                    prologue_row.unwrap_or(0) * crate::frame_table::FRAME_TABLE_ROW_STRIDE + frame_table_base;
                let mut layout = PushLayout::default();
                shared::fill_frame_table_dispatch(&mut layout, absolute_base, raw_user);
                // Metal's frame table is device-level at fixed arg slots.
                shared::set_frame_table_slots(
                    &mut layout,
                    crate::frame_table::FRAME_TABLE_SELECTOR_SLOT,
                    crate::frame_table::FRAME_TABLE_DEVICE_SLOT,
                );
                bind_push_bytes(encoder, layout.as_bytes(), current_is_mesh);
            }
            RenderCommand::BindResourcesTyped { .. } => {
                anyhow::bail!(
                    "BindResourcesTyped in frame-table path: \
                     use frame_table::lower_render_pass_commands or prepare_render_commands"
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
            RenderCommand::DispatchMesh { x, y, z } => {
                anyhow::ensure!(
                    current_is_mesh,
                    "DispatchMesh requires a mesh pipeline (set_mesh_pipeline)"
                );
                encoder.draw_mesh_threadgroups(
                    mtl::MTLSize {
                        width: *x as u64,
                        height: *y as u64,
                        depth: *z as u64,
                    },
                    current_object_tg,
                    current_mesh_tg,
                );
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
    color_load: crate::types::TargetLoad,
    clear_depth: Option<f32>,
) -> &'a mtl::RenderPassDescriptorRef {
    let descriptor = mtl::RenderPassDescriptor::new();

    let color_attachment = descriptor
        .color_attachments()
        .object_at(0)
        .expect("Metal render pass descriptor must have at least one color attachment");
    color_attachment.set_texture(Some(texture));

    match color_load {
        crate::types::TargetLoad::Clear(color) => {
            color_attachment.set_load_action(mtl::MTLLoadAction::Clear);
            color_attachment.set_clear_color(mtl::MTLClearColor::new(
                color.r as f64,
                color.g as f64,
                color.b as f64,
                color.a as f64,
            ));
        }
        crate::types::TargetLoad::Load => {
            color_attachment.set_load_action(mtl::MTLLoadAction::Load);
        }
        crate::types::TargetLoad::Discard => {
            color_attachment.set_load_action(mtl::MTLLoadAction::DontCare);
        }
    }
    color_attachment.set_store_action(mtl::MTLStoreAction::Store);

    if let Some(depth) = depth_texture {
        let depth_attachment = descriptor
            .depth_attachment()
            .expect("Metal render pass descriptor must have a depth attachment when depth texture is set");
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
