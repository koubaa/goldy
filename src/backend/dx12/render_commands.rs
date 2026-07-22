//! Shared render command recording logic.
//!
//! This module contains `record` which is used by both
//! `render_to_target` and `surface_render` to avoid code duplication.

use super::super::shared;
use super::submit_session::Dx12RecordState;
use super::types;
use super::utils::{index_format_to_dxgi, topology_to_d3d12};
use super::{DeviceHandle, RenderCommand};
use windows::Win32::Graphics::Direct3D12::*;

/// Record render commands into a command list.
/// This is shared between render_to_target and surface_render to avoid duplication.
pub(super) fn record(
    cmd: &ID3D12GraphicsCommandList7,
    commands: &[RenderCommand],
    device_handle: DeviceHandle,
    record: &Dx12RecordState<'_>,
) -> anyhow::Result<()> {
    record_with_tables(cmd, commands, device_handle, record)
}

fn record_with_tables(
    cmd: &ID3D12GraphicsCommandList7,
    commands: &[RenderCommand],
    device_handle: DeviceHandle,
    record: &Dx12RecordState<'_>,
) -> anyhow::Result<()> {
    // COM: same pointer as ID3D12GraphicsCommandList for method calls.
    let cmd: &ID3D12GraphicsCommandList = unsafe { std::mem::transmute(cmd) };
    let mut current_vertex_stride = 24u32; // Default stride
    let mut current_pipeline_handle: Option<super::PipelineHandle> = None;
    for command in commands {
        match command {
            RenderCommand::ClearDepth(_) => {
                // Depth clear is applied at pass begin.
            }
            RenderCommand::SetPipeline(pipeline_handle) => {
                let pipelines_read = record.pipelines.read().unwrap();
                if let Some(pipeline) = pipelines_read.entries.get(pipeline_handle) {
                    current_vertex_stride = pipeline.vertex_stride;
                    current_pipeline_handle = Some(*pipeline_handle);
                    unsafe {
                        cmd.SetGraphicsRootSignature(&pipeline.root_signature);
                        cmd.SetPipelineState(&pipeline.pipeline_state);
                        cmd.IASetPrimitiveTopology(topology_to_d3d12(pipeline.topology));
                    }
                }
            }
            RenderCommand::SetVertexBuffer { slot, buffer, offset } => {
                let buffers_read = record.buffers.read().unwrap();
                if let Some(buf_state) = buffers_read.entries.get(buffer) {
                    let view = D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: unsafe { buf_state.resource.GetGPUVirtualAddress() } + offset,
                        SizeInBytes: (buf_state.size - offset) as u32,
                        StrideInBytes: current_vertex_stride,
                    };
                    unsafe { cmd.IASetVertexBuffers(*slot, Some(&[view])) };
                }
            }
            RenderCommand::SetIndexBuffer { buffer, offset, format } => {
                let buffers_read = record.buffers.read().unwrap();
                if let Some(buf_state) = buffers_read.entries.get(buffer) {
                    let view = D3D12_INDEX_BUFFER_VIEW {
                        BufferLocation: unsafe { buf_state.resource.GetGPUVirtualAddress() } + offset,
                        SizeInBytes: (buf_state.size - offset) as u32,
                        Format: index_format_to_dxgi(*format),
                    };
                    unsafe { cmd.IASetIndexBuffer(Some(&view)) };
                }
            }
            RenderCommand::BindResources { .. } => {
                anyhow::bail!(
                    "RenderCommand::BindResources must be lowered before DX12 record; \
                     use frame_table::prepare_render_commands or lower_render_pass_commands"
                );
            }
            RenderCommand::BindResourcesRaw {
                indices: raw_indices,
                user: raw_user,
                frame_table_base,
            } => {
                let pipelines_read = record.pipelines.read().unwrap();
                if let Some(h) = current_pipeline_handle {
                    if let Some(pipeline) = pipelines_read.entries.get(&h) {
                        crate::backend::with_layout_validation(|| {
                            crate::backend::validate_bindless_slot_kinds(
                                raw_indices,
                                &pipeline.push_constant_slot_kinds,
                                |idx| {
                                    super::buffer::bindless_slot_kind_for_index(
                                        &record.buffers.read().unwrap().entries,
                                        device_handle,
                                        idx,
                                    )
                                },
                                &pipeline.shader_debug_name,
                            )
                        })?;
                    }
                }
                let mut layout = types::PushLayout::default();
                shared::fill_frame_table_dispatch(&mut layout, *frame_table_base, raw_user);
                shared::set_frame_table_slots(
                    &mut layout,
                    record.frame_table.selector_slot,
                    record.frame_table.table_slot,
                );
                unsafe {
                    cmd.SetGraphicsRoot32BitConstants(
                        0,
                        (types::TOTAL_PUSH_BYTES / 4) as u32,
                        &layout as *const _ as *const _,
                        0,
                    );
                }
            }
            RenderCommand::BindResourcesTyped { handles: typed_handles } => {
                let pipelines_read = record.pipelines.read().unwrap();
                if let Some(h) = current_pipeline_handle {
                    if let Some(pipeline) = pipelines_read.entries.get(&h) {
                        crate::backend::validate_typed_push_constants(
                            typed_handles,
                            &pipeline.push_constant_categories,
                            &pipeline.shader_debug_name,
                        )?;
                        let indices: Vec<u32> = typed_handles.iter().map(|h| h.index()).collect();
                        crate::backend::with_layout_validation(|| {
                            crate::backend::validate_bindless_slot_kinds(
                                &indices,
                                &pipeline.push_constant_slot_kinds,
                                |idx| {
                                    super::buffer::bindless_slot_kind_for_index(
                                        &record.buffers.read().unwrap().entries,
                                        device_handle,
                                        idx,
                                    )
                                },
                                &pipeline.shader_debug_name,
                            )
                        })?;
                    }
                }
                anyhow::bail!(
                    "RenderCommand::BindResourcesTyped must be lowered before DX12 record; \
                     use frame_table::lower_render_pass_commands or prepare_render_commands"
                );
            }
            RenderCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => unsafe {
                // Topology is now set in SetPipeline, not hardcoded here
                cmd.DrawInstanced(*vertex_count, *instance_count, *first_vertex, *first_instance);
            },
            RenderCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => unsafe {
                // Topology is now set in SetPipeline, not hardcoded here
                cmd.DrawIndexedInstanced(
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
