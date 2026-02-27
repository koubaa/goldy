//! Shared render command recording logic.
//!
//! This module contains `record` which is used by both
//! `render_to_target` and `surface_render` to avoid code duplication.

use super::types::{self, Dx12State};
use super::utils::{index_format_to_dxgi, topology_to_d3d12};
use super::{DeviceHandle, RenderCommand};
use windows::Win32::Graphics::Direct3D12::*;

/// Record render commands into a command list.
/// This is shared between render_to_target and surface_render to avoid duplication.
pub(super) fn record(
    cmd: &ID3D12GraphicsCommandList,
    commands: &[RenderCommand],
    device_handle: DeviceHandle,
    state: &Dx12State,
) {
    let mut current_vertex_stride = 24u32; // Default stride
    for command in commands {
        match command {
            RenderCommand::Clear(_) => {
                // Already handled by caller
            }
            RenderCommand::ClearDepth(_) => {
                // TODO: Implement depth clear
            }
            RenderCommand::SetPipeline(pipeline_handle) => {
                if let Some(pipeline) = state.pipelines.get(pipeline_handle) {
                    current_vertex_stride = pipeline.vertex_stride;
                    unsafe {
                        cmd.SetGraphicsRootSignature(&pipeline.root_signature);
                        cmd.SetPipelineState(&pipeline.pipeline_state);
                        cmd.IASetPrimitiveTopology(topology_to_d3d12(pipeline.topology));
                    }
                }
            }
            RenderCommand::SetVertexBuffer {
                slot,
                buffer,
                offset,
            } => {
                if let Some(buf_state) = state.buffers.get(buffer) {
                    let view = D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: unsafe { buf_state.resource.GetGPUVirtualAddress() }
                            + offset,
                        SizeInBytes: (buf_state.size - offset) as u32,
                        StrideInBytes: current_vertex_stride,
                    };
                    unsafe { cmd.IASetVertexBuffers(*slot, Some(&[view])) };
                }
            }
            RenderCommand::SetIndexBuffer {
                buffer,
                offset,
                format,
            } => {
                if let Some(buf_state) = state.buffers.get(buffer) {
                    let view = D3D12_INDEX_BUFFER_VIEW {
                        BufferLocation: unsafe { buf_state.resource.GetGPUVirtualAddress() }
                            + offset,
                        SizeInBytes: (buf_state.size - offset) as u32,
                        Format: index_format_to_dxgi(*format),
                    };
                    unsafe { cmd.IASetIndexBuffer(Some(&view)) };
                }
            }
            RenderCommand::SetPushConstants { buffers } => {
                // Fully bindless mode: push buffer indices directly via root constants
                let bindless_enabled = state
                    .devices
                    .get(&device_handle)
                    .map(|d| d.bindless_enabled)
                    .unwrap_or(false);

                if bindless_enabled {
                    let mut indices = types::BindlessIndices::default();
                    for (i, buffer_handle) in buffers.iter().enumerate() {
                        if i >= types::MAX_ROOT_CONSTANT_INDICES {
                            break;
                        }
                        if let Some(buf_state) = state.buffers.get(buffer_handle) {
                            // Shaders using goldy_dyn_scattered() return RWStructuredBuffer which needs UAV.
                            // Always use UAV offset (bindless_offset) for storage buffers.
                            // For uniform buffers (Broadcast), use bindless_offset directly (CBV).
                            let offset = buf_state.bindless_offset.unwrap_or(0);
                            indices.indices[i] = offset;
                        }
                    }
                    unsafe {
                        cmd.SetGraphicsRoot32BitConstants(
                            0, // Root parameter index for constants
                            types::MAX_ROOT_CONSTANT_INDICES as u32,
                            indices.indices.as_ptr() as *const _,
                            0,
                        );
                    }
                }
            }
            RenderCommand::SetPushConstantsRaw {
                indices: raw_indices,
            } => {
                // Fully bindless mode: push raw indices directly (for textures/samplers)
                let bindless_enabled = state
                    .devices
                    .get(&device_handle)
                    .map(|d| d.bindless_enabled)
                    .unwrap_or(false);

                if bindless_enabled {
                    let mut indices = types::BindlessIndices::default();
                    for (i, &idx) in raw_indices.iter().enumerate() {
                        if i >= types::MAX_ROOT_CONSTANT_INDICES {
                            break;
                        }
                        indices.indices[i] = idx;
                    }
                    unsafe {
                        cmd.SetGraphicsRoot32BitConstants(
                            0,
                            types::MAX_ROOT_CONSTANT_INDICES as u32,
                            indices.indices.as_ptr() as *const _,
                            0,
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
                // Topology is now set in SetPipeline, not hardcoded here
                cmd.DrawInstanced(
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
}
