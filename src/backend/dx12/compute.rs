//! Compute pipeline and dispatch logic.

use super::barriers;
use super::shader;
use super::types::{self, ComputeAllocatorSlot, ComputePipelineState, Dx12State};
use super::{ComputePipelineHandle, DeviceHandle, ShaderHandle};
use crate::backend::{ComputeCommand, FenceToken};
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

/// Submit compute commands without blocking. Returns a fence token for polling/waiting.
pub(super) fn submit(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    commands: &[ComputeCommand],
) -> Result<FenceToken> {
    let (allocator, fence_value, slot_idx) = {
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        // Find a free slot (allocator whose work has completed) or create a new one
        let fence = &logical_device.fence;
        let completed = unsafe { fence.GetCompletedValue() };
        let pool = &mut logical_device.compute_allocator_pool;
        let slot_idx = pool.iter().position(|s| completed >= s.fence_value);
        let (allocator, slot_idx) = if let Some(idx) = slot_idx {
            let slot = &mut pool[idx];
            unsafe { slot.allocator.Reset() }.context("Failed to reset command allocator")?;
            (slot.allocator.clone(), idx)
        } else {
            let new_allocator: ID3D12CommandAllocator = unsafe {
                logical_device
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
            .context("Failed to create command allocator")?;
            pool.push(ComputeAllocatorSlot {
                allocator: new_allocator.clone(),
                fence_value: 0,
            });
            unsafe { new_allocator.Reset() }.context("Failed to reset new command allocator")?;
            (new_allocator, pool.len() - 1)
        };
        let token = logical_device.fence_value;
        logical_device.fence_value += 1;
        (allocator, token, slot_idx)
    };

    // Create command list (allocator is owned, no borrow of state)
    let command_list: ID3D12GraphicsCommandList = {
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        unsafe {
            logical_device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )
        }
        .context("Failed to create command list")?
    };
    let command_list7: ID3D12GraphicsCommandList7 = command_list
        .cast()
        .context("ID3D12GraphicsCommandList7 required")?;

    // Bind descriptor heaps for bindless rendering
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    unsafe {
        command_list.SetDescriptorHeaps(&[
            Some(logical_device.cbv_srv_uav_heap.clone()),
            Some(logical_device.sampler_heap.clone()),
        ]);
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
                // Push buffer indices directly (no bind groups)
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
            ComputeCommand::SetPushConstantsRaw {
                indices: raw_indices,
            } => {
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
            ComputeCommand::Dispatch {
                workgroups_x,
                workgroups_y,
                workgroups_z,
            } => {
                // UAV memory barrier so previous dispatch's writes are visible
                let g = D3D12_GLOBAL_BARRIER {
                    SyncBefore: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    AccessAfter: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                };
                unsafe { barriers::barrier_globals(&command_list7, &[g]) };
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

                let g = D3D12_GLOBAL_BARRIER {
                    SyncBefore: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    AccessAfter: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                };
                unsafe { barriers::barrier_globals(&command_list7, &[g]) };

                let to_indirect = barriers::buffer_barrier_full(
                    &buf_state.resource,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
                );
                unsafe { barriers::barrier_buffers(&command_list7, &[to_indirect]) };

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

                let to_uav = barriers::buffer_barrier_full(
                    &buf_state.resource,
                    D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                );
                unsafe { barriers::barrier_buffers(&command_list7, &[to_uav]) };
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
                    if buf_state.is_storage {
                        // Storage buffer (DEFAULT heap): GPU-side UAV clear (no staging needed)
                        super::buffer::uav_clear(
                            logical_device,
                            buf_state,
                            &command_list,
                            *offset,
                            clear_size,
                        )?;
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

    // Global barrier so compute UAV writes are visible to subsequent graphics / CPU
    let tail = D3D12_GLOBAL_BARRIER {
        SyncBefore: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
        SyncAfter: D3D12_BARRIER_SYNC_ALL,
        AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
        AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
    };
    unsafe { barriers::barrier_globals(&command_list7, &[tail]) };

    // Close and execute
    unsafe { command_list.Close() }.context("Failed to close command list")?;

    let cmd_list: ID3D12CommandList = command_list.cast().context("Failed to cast command list")?;

    let logical_device = state.devices.get(&device_handle).unwrap();
    unsafe {
        logical_device
            .command_queue
            .ExecuteCommandLists(&[Some(cmd_list)]);
    }

    // Signal fence with the token we reserved
    unsafe {
        logical_device
            .command_queue
            .Signal(&logical_device.fence, fence_value)
    }
    .context("Failed to signal fence")?;

    // Update the slot's fence_value so we know when it can be reused
    if let Some(dev) = state.devices.get_mut(&device_handle) {
        if let Some(slot) = dev.compute_allocator_pool.get_mut(slot_idx) {
            slot.fence_value = fence_value;
        }
    }

    Ok(fence_value)
}

/// Check if the fence for the given token has signaled.
pub(super) fn is_fence_complete(
    state: &Dx12State,
    device_handle: DeviceHandle,
    token: FenceToken,
) -> bool {
    let logical_device = match state.devices.get(&device_handle) {
        Some(dev) => dev,
        None => return false,
    };
    (unsafe { logical_device.fence.GetCompletedValue() }) >= token
}

/// Block until the fence signals.
pub(super) fn wait_fence(
    state: &Dx12State,
    device_handle: DeviceHandle,
    token: FenceToken,
) -> Result<()> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    super::utils::wait_for_fence(&logical_device.fence, token)?;

    // Detect TDR: device removal signals all fences with UINT64_MAX
    let completed = unsafe { logical_device.fence.GetCompletedValue() };
    if completed == u64::MAX {
        let reason = unsafe { logical_device.device.GetDeviceRemovedReason() };
        anyhow::bail!("GPU device removed (TDR) after fence wait: {:?}", reason);
    }
    Ok(())
}

/// Wait with timeout. Returns Ok(true) if signaled, Ok(false) if timeout elapsed.
pub(super) fn wait_fence_timeout(
    state: &Dx12State,
    device_handle: DeviceHandle,
    token: FenceToken,
    timeout_ms: u32,
) -> Result<bool> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    super::utils::wait_for_fence_timeout(&logical_device.fence, token, timeout_ms)
}
