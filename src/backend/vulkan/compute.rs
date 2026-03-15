//! Compute pipeline and dispatch logic.

use super::types::{self, BindlessIndices, ComputePipelineState, LogicalDevice};
use super::{ComputePipelineHandle, DeviceHandle};
use crate::backend::{ComputeCommand, FenceToken};
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

/// Create a compute pipeline.
pub(super) fn create(
    devices: &HashMap<DeviceHandle, LogicalDevice>,
    compute_pipelines: &mut HashMap<ComputePipelineHandle, ComputePipelineState>,
    next_compute_pipeline_handle: &mut ComputePipelineHandle,
    device_handle: DeviceHandle,
    cs_module: vk::ShaderModule,
) -> Result<ComputePipelineHandle> {
    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Always use bindless pipeline layout
    let pipeline_layout = logical_device
        .bindless_pipeline_layout
        .context("Bindless pipeline layout required")?;
    let owns_layout = false; // Don't own - global bindless layout

    // Compute shader stage
    let cs_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(cs_module)
        .name(c"main");

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(cs_stage)
        .layout(pipeline_layout);

    let pipelines = unsafe {
        logical_device.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[pipeline_info],
            None,
        )
    }
    .map_err(|(_, e)| anyhow::anyhow!("Failed to create compute pipeline: {:?}", e))?;

    let handle = *next_compute_pipeline_handle;
    *next_compute_pipeline_handle += 1;

    compute_pipelines.insert(
        handle,
        ComputePipelineState {
            device_handle,
            pipeline: pipelines[0],
            layout: pipeline_layout,
            owns_layout,
            parameter_block_layouts: Vec::new(),
        },
    );

    tracing::debug!(
        "Created compute pipeline (handle={}, bindless={})",
        handle,
        !owns_layout
    );
    Ok(handle)
}

/// Destroy a compute pipeline.
pub(super) fn destroy(
    devices: &HashMap<DeviceHandle, LogicalDevice>,
    compute_pipelines: &mut HashMap<ComputePipelineHandle, ComputePipelineState>,
    pipeline_handle: ComputePipelineHandle,
) {
    if let Some(pipeline) = compute_pipelines.remove(&pipeline_handle) {
        if let Some(logical_device) = devices.get(&pipeline.device_handle) {
            unsafe {
                logical_device.device.device_wait_idle().ok();
                logical_device
                    .device
                    .destroy_pipeline(pipeline.pipeline, None);
                // Only destroy layout if we own it (not the global bindless layout)
                if pipeline.owns_layout {
                    logical_device
                        .device
                        .destroy_pipeline_layout(pipeline.layout, None);
                }
            }
        }
    }
}

/// Submit compute commands without blocking. Returns a fence token for polling/waiting.
pub(super) fn submit(
    state: &mut super::types::VulkanState,
    device_handle: DeviceHandle,
    commands: &[ComputeCommand],
) -> Result<FenceToken> {
    let devices = &state.devices;
    let compute_pipelines = &state.compute_pipelines;
    let buffers = &state.buffers;

    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Create fence for this submission
    let fence_create_info = vk::FenceCreateInfo::default();
    let fence = unsafe { logical_device.device.create_fence(&fence_create_info, None) }
        .context("Failed to create fence")?;

    // Allocate command buffer
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(logical_device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer")?;
    let cmd = cmd_buffers[0];

    // Begin command buffer
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) }
        .context("Failed to begin command buffer")?;

    // Track current pipeline for push constants
    let mut current_pipeline: Option<ComputePipelineHandle> = None;

    // Process commands (same logic as dispatch)
    for command in commands {
        match command {
            ComputeCommand::SetPipeline(handle) => {
                if let Some(pipeline_state) = compute_pipelines.get(handle) {
                    unsafe {
                        logical_device.device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline_state.pipeline,
                        );

                        if logical_device.bindless_enabled {
                            if let (Some(bindless_set), Some(bindless_layout)) = (
                                logical_device.bindless_descriptor_set,
                                logical_device.bindless_pipeline_layout,
                            ) {
                                logical_device.device.cmd_bind_descriptor_sets(
                                    cmd,
                                    vk::PipelineBindPoint::COMPUTE,
                                    bindless_layout,
                                    0,
                                    std::slice::from_ref(&bindless_set),
                                    &[],
                                );
                            }
                        }
                    }
                    current_pipeline = Some(*handle);
                }
            }
            ComputeCommand::SetPushConstants {
                buffers: buffer_handles,
            } => {
                if let Some(pipeline) = current_pipeline.and_then(|p| compute_pipelines.get(&p)) {
                    let mut indices = BindlessIndices::default();
                    for (i, buffer_handle) in buffer_handles.iter().enumerate() {
                        if i >= types::MAX_PUSH_CONSTANT_INDICES {
                            break;
                        }
                        indices.indices[i] = buffers
                            .get(buffer_handle)
                            .and_then(|b| b.bindless_index)
                            .unwrap_or(0);
                    }
                    unsafe {
                        logical_device.device.cmd_push_constants(
                            cmd,
                            pipeline.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            bytemuck::bytes_of(&indices),
                        );
                    }
                }
            }
            ComputeCommand::SetPushConstantsRaw {
                indices: raw_indices,
            } => {
                if let Some(pipeline) = current_pipeline.and_then(|p| compute_pipelines.get(&p)) {
                    let mut indices = BindlessIndices::default();
                    for (i, &idx) in raw_indices.iter().enumerate() {
                        if i >= types::MAX_PUSH_CONSTANT_INDICES {
                            break;
                        }
                        indices.indices[i] = idx;
                    }
                    unsafe {
                        logical_device.device.cmd_push_constants(
                            cmd,
                            pipeline.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            bytemuck::bytes_of(&indices),
                        );
                    }
                }
            }
            ComputeCommand::Dispatch {
                workgroups_x,
                workgroups_y,
                workgroups_z,
            } => unsafe {
                let mem_barrier = vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(
                        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    );
                let dep_info = vk::DependencyInfo::default()
                    .memory_barriers(std::slice::from_ref(&mem_barrier));
                logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);

                logical_device.device.cmd_dispatch(
                    cmd,
                    *workgroups_x,
                    *workgroups_y,
                    *workgroups_z,
                );
            },
            ComputeCommand::DispatchIndirect { buffer, offset } => {
                let buf_state = buffers
                    .get(buffer)
                    .context("DispatchIndirect: invalid buffer handle")?;
                unsafe {
                    let mem_barrier = vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::COMPUTE_SHADER
                                | vk::PipelineStageFlags2::DRAW_INDIRECT,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2::SHADER_READ
                                | vk::AccessFlags2::SHADER_WRITE
                                | vk::AccessFlags2::INDIRECT_COMMAND_READ,
                        );
                    let dep_info = vk::DependencyInfo::default()
                        .memory_barriers(std::slice::from_ref(&mem_barrier));
                    logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);

                    logical_device
                        .device
                        .cmd_dispatch_indirect(cmd, buf_state.buffer, *offset);
                }
            }
            ComputeCommand::ClearBuffer {
                buffer,
                offset,
                size,
            } => {
                let buf_state = buffers
                    .get(buffer)
                    .context("ClearBuffer: invalid buffer handle")?;
                let clear_size = if *size == 0 {
                    buf_state.size.saturating_sub(*offset)
                } else {
                    *size
                };
                if clear_size > 0 {
                    unsafe {
                        logical_device.device.cmd_fill_buffer(
                            cmd,
                            buf_state.buffer,
                            *offset,
                            clear_size,
                            0,
                        );

                        let mem_barrier = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(
                                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                            );
                        let dep_info = vk::DependencyInfo::default()
                            .memory_barriers(std::slice::from_ref(&mem_barrier));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                    }
                }
            }
        }
    }

    // End command buffer
    unsafe { logical_device.device.end_command_buffer(cmd) }
        .context("Failed to end command buffer")?;

    // Submit with fence (non-blocking)
    let cmd_buffers = [cmd];
    let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);

    unsafe {
        logical_device
            .device
            .queue_submit(logical_device.queue, &[submit_info], fence)
            .context("Failed to submit command buffer")?;
    }

    // Cleanup command buffer (fence tracks completion)
    unsafe {
        logical_device
            .device
            .free_command_buffers(logical_device.command_pool, &cmd_buffers);
    }

    // Store fence and return token
    let token = state.next_compute_fence_token;
    state.next_compute_fence_token += 1;
    state
        .compute_fence_pool
        .insert(token, (device_handle, fence));

    Ok(token)
}

/// Check if the fence for the given token has signaled.
pub(super) fn is_fence_complete(
    state: &super::types::VulkanState,
    _device_handle: DeviceHandle,
    token: FenceToken,
) -> bool {
    let Some((device_handle, fence)) = state.compute_fence_pool.get(&token) else {
        return true; // Unknown token, treat as complete
    };
    let Some(logical_device) = state.devices.get(device_handle) else {
        return true;
    };
    unsafe { logical_device.device.get_fence_status(*fence) }.unwrap_or_default()
}

/// Block until the fence signals.
pub(super) fn wait_fence(
    state: &super::types::VulkanState,
    _device_handle: DeviceHandle,
    token: FenceToken,
) -> Result<()> {
    let (stored_device, fence) = state
        .compute_fence_pool
        .get(&token)
        .context("Invalid fence token")?;
    let logical_device = state
        .devices
        .get(stored_device)
        .context("Device for fence no longer exists")?;

    unsafe {
        logical_device
            .device
            .wait_for_fences(&[*fence], true, u64::MAX)
            .context("Failed to wait for fence")?;
    }
    Ok(())
}

/// Wait with timeout. Returns Ok(true) if signaled, Ok(false) if timeout elapsed.
pub(super) fn wait_fence_timeout(
    state: &super::types::VulkanState,
    _device: DeviceHandle,
    token: FenceToken,
    timeout_ms: u32,
) -> Result<bool> {
    let (stored_device, fence) = state
        .compute_fence_pool
        .get(&token)
        .context("Invalid fence token")?;
    let logical_device = state
        .devices
        .get(stored_device)
        .context("Device for fence no longer exists")?;

    // vkWaitForFences uses nanoseconds
    let timeout_ns = u64::from(timeout_ms) * 1_000_000;

    let result = unsafe {
        logical_device
            .device
            .wait_for_fences(&[*fence], true, timeout_ns)
    };

    match result {
        Ok(()) => Ok(true),
        Err(vk::Result::TIMEOUT) => Ok(false),
        Err(e) => Err(anyhow::anyhow!("Failed to wait for fence: {:?}", e)),
    }
}
