//! Compute pipeline and dispatch logic.

use super::types::{self, BindlessIndices, BufferState, ComputePipelineState, LogicalDevice};
use super::{BufferHandle, ComputePipelineHandle, DeviceHandle};
use crate::backend::ComputeCommand;
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

/// Dispatch compute commands.
pub(super) fn dispatch(
    devices: &HashMap<DeviceHandle, LogicalDevice>,
    compute_pipelines: &HashMap<ComputePipelineHandle, ComputePipelineState>,
    buffers: &HashMap<BufferHandle, BufferState>,
    device_handle: DeviceHandle,
    commands: &[ComputeCommand],
) -> Result<()> {
    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

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

    // Process commands
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

                        // Bind the global bindless descriptor set if enabled
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
            ComputeCommand::SetPushConstants { buffers: buffer_handles } => {
                // Fully bindless mode: push buffer indices directly
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
            ComputeCommand::Dispatch {
                workgroups_x,
                workgroups_y,
                workgroups_z,
            } => unsafe {
                logical_device
                    .device
                    .cmd_dispatch(cmd, *workgroups_x, *workgroups_y, *workgroups_z);
            },
        }
    }

    // End command buffer
    unsafe { logical_device.device.end_command_buffer(cmd) }
        .context("Failed to end command buffer")?;

    // Submit and wait
    let cmd_buffers = [cmd];
    let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);

    unsafe {
        logical_device
            .device
            .queue_submit(logical_device.queue, &[submit_info], vk::Fence::null())
            .context("Failed to submit command buffer")?;
        logical_device
            .device
            .queue_wait_idle(logical_device.queue)
            .context("Failed to wait for queue")?;
    }

    // Cleanup
    unsafe {
        logical_device
            .device
            .free_command_buffers(logical_device.command_pool, &cmd_buffers);
    }

    Ok(())
}
