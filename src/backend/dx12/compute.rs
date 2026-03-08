//! Compute pipeline and dispatch logic.

use super::shader;
use super::types::{self, ComputePipelineState, Dx12State};
use super::{ComputePipelineHandle, DeviceHandle, ShaderHandle};
use crate::backend::ComputeCommand;
use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// Create a compute pipeline.
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    compute_shader: ShaderHandle,
) -> Result<ComputePipelineHandle> {
    // Compile shader on-demand
    let cs_bytecode =
        shader::ensure_stage_compiled(state, compute_shader, crate::slang::SlangStage::Compute)?;

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Use the shared bindless root signature from the device
    let root_signature = logical_device
        .bindless_root_signature
        .as_ref()
        .context("Bindless root signature not available")?
        .clone();

    tracing::debug!("Using shared bindless root signature for compute pipeline");

    // Create compute PSO
    let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: cs_bytecode.as_ptr() as *const _,
            BytecodeLength: cs_bytecode.len(),
        },
        NodeMask: 0,
        CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
        Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
    };

    let pipeline_state: ID3D12PipelineState =
        unsafe { logical_device.device.CreateComputePipelineState(&pso_desc) }
            .context("Failed to create compute pipeline state")?;

    let handle = state.next_compute_pipeline_handle;
    state.next_compute_pipeline_handle += 1;

    state.compute_pipelines.insert(
        handle,
        ComputePipelineState {
            device_handle,
            pipeline_state,
            root_signature,
            parameter_block_layouts: Vec::new(),
        },
    );

    tracing::debug!("Created compute pipeline {}", handle);
    Ok(handle)
}

/// Destroy a compute pipeline.
pub(super) fn destroy(state: &mut Dx12State, pipeline_handle: ComputePipelineHandle) {
    state.compute_pipelines.remove(&pipeline_handle);
}

/// Dispatch compute commands.
pub(super) fn dispatch(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    commands: &[ComputeCommand],
) -> Result<()> {
    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    // Reset command allocator
    unsafe { logical_device.command_allocator.Reset() }
        .context("Failed to reset command allocator")?;

    // Create command list
    let command_list: ID3D12GraphicsCommandList = unsafe {
        logical_device.device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &logical_device.command_allocator,
            None,
        )
    }
    .context("Failed to create command list")?;

    // Bind descriptor heaps for bindless rendering (must be done before any dispatch calls)
    // Re-borrow logical_device to get heaps
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    if logical_device.bindless_enabled {
        unsafe {
            command_list.SetDescriptorHeaps(&[
                Some(logical_device.cbv_srv_uav_heap.clone()),
                Some(logical_device.sampler_heap.clone()),
            ]);
        }
    }

    // Track current pipeline (reserved for future use)
    let mut _current_pipeline_handle: Option<ComputePipelineHandle> = None;

    // Process commands
    for command in commands {
        match command {
            ComputeCommand::SetPipeline(handle) => {
                if let Some(pipeline_state) = state.compute_pipelines.get(handle) {
                    unsafe {
                        command_list.SetComputeRootSignature(&pipeline_state.root_signature);
                        command_list.SetPipelineState(&pipeline_state.pipeline_state);
                    }
                    _current_pipeline_handle = Some(*handle);
                }
            }
            ComputeCommand::SetPushConstants { buffers } => {
                // Fully bindless mode: push buffer indices directly (no bind groups needed)
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
                            // Compute shaders use goldy_dyn_scattered() which returns RWStructuredBuffer.
                            // RWStructuredBuffer requires UAV descriptors, not SRV.
                            // Always use bindless_offset (UAV) for storage buffers in compute shaders.
                            // For uniform buffers (Broadcast), use bindless_offset directly (CBV).
                            let offset = buf_state.bindless_offset.unwrap_or(0);
                            indices.indices[i] = offset;
                            tracing::trace!(
                                "Compute push constant [{}]: buffer {} -> UAV offset {}",
                                i,
                                buffer_handle,
                                offset
                            );
                        }
                    }

                    tracing::trace!(
                        "Setting compute root constants: {:?}",
                        &indices.indices[..buffers.len().min(types::MAX_ROOT_CONSTANT_INDICES)]
                    );

                    unsafe {
                        command_list.SetComputeRoot32BitConstants(
                            0, // Root parameter index
                            types::MAX_ROOT_CONSTANT_INDICES as u32,
                            indices.indices.as_ptr() as *const std::ffi::c_void,
                            0,
                        );
                    }
                }
            }
            ComputeCommand::SetPushConstantsRaw {
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
                        command_list.SetComputeRoot32BitConstants(
                            0,
                            types::MAX_ROOT_CONSTANT_INDICES as u32,
                            indices.indices.as_ptr() as *const std::ffi::c_void,
                            0,
                        );
                    }
                }
            }
            ComputeCommand::Dispatch {
                workgroups_x,
                workgroups_y,
                workgroups_z,
            } => {
                // UAV barrier so previous dispatch's writes (e.g. path_tiling -> ptcl) are visible
                let uav_barrier = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(None),
                        }),
                    },
                };
                unsafe { command_list.ResourceBarrier(&[uav_barrier]) };
                unsafe {
                    command_list.Dispatch(*workgroups_x, *workgroups_y, *workgroups_z);
                }
            }
            ComputeCommand::DispatchIndirect { buffer, offset } => {
                let buf_state = state
                    .buffers
                    .get(buffer)
                    .context("DispatchIndirect: invalid buffer handle")?;
                let signature = logical_device
                    .compute_dispatch_indirect_signature
                    .as_ref()
                    .context("DispatchIndirect: compute indirect signature not available")?;

                // Transition: UAV → INDIRECT_ARGUMENT (setup shader wrote args as UAV)
                let to_indirect = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(
                            D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: unsafe {
                                    std::mem::transmute_copy(&buf_state.resource)
                                },
                                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                StateAfter: D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                            },
                        ),
                    },
                };
                unsafe { command_list.ResourceBarrier(&[to_indirect]) };

                unsafe {
                    command_list.ExecuteIndirect(
                        signature,
                        1,
                        &buf_state.resource,
                        *offset,
                        None,
                        0,
                    );
                }

                // Transition back: INDIRECT_ARGUMENT → UAV (buffer may be reused)
                let to_uav = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(
                            D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: unsafe {
                                    std::mem::transmute_copy(&buf_state.resource)
                                },
                                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                StateBefore: D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                                StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            },
                        ),
                    },
                };
                unsafe { command_list.ResourceBarrier(&[to_uav]) };
            }
            ComputeCommand::ClearBuffer {
                buffer,
                offset,
                size,
            } => {
                let buf_state = state
                    .buffers
                    .get(buffer)
                    .context("ClearBuffer: invalid buffer handle")?;
                let clear_size = if *size == 0 {
                    buf_state.size.saturating_sub(*offset)
                } else {
                    *size
                };
                if clear_size > 0 {
                    if let Some(upload_buf) = &buf_state.upload_buffer {
                        // Storage buffer (DEFAULT heap): zero the upload buffer region, then copy
                        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
                        let no_read = D3D12_RANGE { Begin: 0, End: 0 };
                        unsafe { upload_buf.Map(0, Some(&no_read), Some(&mut mapped)) }
                            .context("ClearBuffer: failed to map upload buffer")?;
                        unsafe {
                            std::ptr::write_bytes(
                                (mapped as *mut u8).add(*offset as usize),
                                0,
                                clear_size as usize,
                            );
                        }
                        let written = D3D12_RANGE {
                            Begin: *offset as usize,
                            End: (*offset + clear_size) as usize,
                        };
                        unsafe { upload_buf.Unmap(0, Some(&written)) };

                        // Transition to COPY_DEST, copy, transition back to UAV
                        let to_copy = D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                Transition: std::mem::ManuallyDrop::new(
                                    D3D12_RESOURCE_TRANSITION_BARRIER {
                                        pResource: unsafe {
                                            std::mem::transmute_copy(&buf_state.resource)
                                        },
                                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                        StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
                                    },
                                ),
                            },
                        };
                        unsafe { command_list.ResourceBarrier(&[to_copy]) };
                        unsafe {
                            command_list.CopyBufferRegion(
                                &buf_state.resource,
                                *offset,
                                upload_buf,
                                *offset,
                                clear_size,
                            );
                        }
                        let to_uav = D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                Transition: std::mem::ManuallyDrop::new(
                                    D3D12_RESOURCE_TRANSITION_BARRIER {
                                        pResource: unsafe {
                                            std::mem::transmute_copy(&buf_state.resource)
                                        },
                                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                        StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                                        StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                    },
                                ),
                            },
                        };
                        unsafe { command_list.ResourceBarrier(&[to_uav]) };
                    } else {
                        // UPLOAD heap buffer: CPU-accessible, just memset
                        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
                        let no_read = D3D12_RANGE { Begin: 0, End: 0 };
                        unsafe { buf_state.resource.Map(0, Some(&no_read), Some(&mut mapped)) }
                            .context("ClearBuffer: failed to map buffer")?;
                        unsafe {
                            std::ptr::write_bytes(
                                (mapped as *mut u8).add(*offset as usize),
                                0,
                                clear_size as usize,
                            );
                        }
                        let written = D3D12_RANGE {
                            Begin: *offset as usize,
                            End: (*offset + clear_size) as usize,
                        };
                        unsafe { buf_state.resource.Unmap(0, Some(&written)) };
                    }
                }
            }
        }
    }

    // Add UAV barrier to ensure compute writes are visible to subsequent operations
    // This is critical for ping-pong buffers where compute writes are read by render
    let uav_barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None), // NULL means barrier on all UAVs
            }),
        },
    };
    unsafe { command_list.ResourceBarrier(&[uav_barrier]) };

    // Close and execute
    unsafe { command_list.Close() }.context("Failed to close command list")?;

    let cmd_list: ID3D12CommandList = command_list.cast().context("Failed to cast command list")?;

    let logical_device = state.devices.get(&device_handle).unwrap();
    unsafe {
        logical_device
            .command_queue
            .ExecuteCommandLists(&[Some(cmd_list)]);
    }

    // Wait for completion
    let fence_value = logical_device.fence_value;
    unsafe {
        logical_device
            .command_queue
            .Signal(&logical_device.fence, fence_value)
    }
    .context("Failed to signal fence")?;
    super::utils::wait_for_fence(&logical_device.fence, fence_value)?;

    // Increment fence value for next operation
    if let Some(dev) = state.devices.get_mut(&device_handle) {
        dev.fence_value += 1;
    }

    Ok(())
}
