//! Compute pipeline and dispatch logic.

use super::staging;
use super::types::{self, ComputePipelineState, LogicalDevice, PushLayout};
use super::{ComputePipelineHandle, DeviceHandle};
use crate::backend::{FenceToken, GpuCommand};
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

/// Reap fences that have already signaled from the compute fence pool.
///
/// Without this, every `submit_graph` that doesn't have a paired `wait_fence`
/// (the common ekrano pattern: only the *last* `GpuFuture` in a recording is
/// waited on; the intermediate ones are silently dropped) leaks a `VkFence` +
/// `VkCommandBuffer` into the pool. Over a few thousand frames the pool grows
/// large enough that the driver's command-pool / fence-pool internal arenas
/// fall over, the device is lost (`ERROR_DEVICE_LOST` on the next
/// `queue_submit`), and the unbounded HashMap teardown corrupts the heap on
/// shutdown.
fn reap_signaled_fences(state: &mut super::types::VulkanState) {
    let signaled: Vec<u64> = state
        .compute_fence_pool
        .iter()
        .filter_map(|(token, (device_handle, fence, _))| {
            let logical_device = state.devices.get(device_handle)?;
            let signaled =
                unsafe { logical_device.device.get_fence_status(*fence) }.unwrap_or(false);
            if signaled {
                Some(*token)
            } else {
                None
            }
        })
        .collect();

    for token in signaled {
        if let Some((device_handle, fence, cmd_buf)) = state.compute_fence_pool.remove(&token) {
            if let Some(logical_device) = state.devices.get(&device_handle) {
                unsafe {
                    if let Some(cb) = cmd_buf {
                        logical_device
                            .device
                            .free_command_buffers(logical_device.command_pool, &[cb]);
                    }
                    logical_device.device.destroy_fence(fence, None);
                }
            }
            if let Some(staging) = state.compute_texture_staging_pool.remove(&token) {
                if let Some(logical_device) = state.devices.get(&device_handle) {
                    super::texture::destroy_texture_staging_list(logical_device, staging);
                }
            }
        }
    }
}

/// Create a compute pipeline.
pub(super) fn create(
    devices: &HashMap<DeviceHandle, LogicalDevice>,
    compute_pipelines: &mut HashMap<ComputePipelineHandle, ComputePipelineState>,
    next_compute_pipeline_handle: &mut ComputePipelineHandle,
    device_handle: DeviceHandle,
    cs_module: vk::ShaderModule,
    shader_debug_name: String,
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

    // Pipeline robustness (core in Vulkan 1.4): OOB descriptor access returns zero
    let mut robustness = vk::PipelineRobustnessCreateInfoEXT::default()
        .storage_buffers(vk::PipelineRobustnessBufferBehaviorEXT::ROBUST_BUFFER_ACCESS_2)
        .uniform_buffers(vk::PipelineRobustnessBufferBehaviorEXT::ROBUST_BUFFER_ACCESS_2)
        .images(vk::PipelineRobustnessImageBehaviorEXT::ROBUST_IMAGE_ACCESS_2);

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(cs_stage)
        .layout(pipeline_layout)
        .push_next(&mut robustness);

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
            shader_debug_name,
        },
    );

    tracing::debug!("Created compute pipeline (handle={})", handle);
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
    commands: &[GpuCommand],
) -> Result<FenceToken> {
    // Reap any previously-submitted fences that have already signaled. Keeps
    // the pool bounded when callers (ekrano) don't wait on every GpuFuture.
    // Belt before reap: need live VkFence handles to poll completion.
    for belt in state.staging_belts.values_mut() {
        belt.reclaim(
            &state.compute_fence_pool,
            &state.devices,
            &state.deferred_pending_tokens,
        )?;
    }
    reap_signaled_fences(state);

    // Belt slices for DEVICE_LOCAL WriteBuffer copies (same iteration order as command loop).
    let mut belt_slices: Vec<(vk::Buffer, u64)> = Vec::new();

    // Pre-pass: stage CPU data for WriteBuffer commands (needs mutable buffer
    // access before we borrow state immutably for the command loop).
    for command in commands {
        if let GpuCommand::WriteBuffer {
            buffer: buf_handle,
            offset,
            data,
        } = command
        {
            let buf = state
                .buffers
                .get(buf_handle)
                .context("WriteBuffer: invalid buffer handle")?;
            if let Some(base) = buf.host_mapped {
                let p = base as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        p.add(*offset as usize),
                        data.len(),
                    );
                }
            } else if !buf.is_storage {
                let dev = state
                    .devices
                    .get(&buf.device_handle)
                    .context("WriteBuffer: device invalid")?;
                unsafe {
                    let ptr = dev
                        .map_memory2(buf.memory, *offset, data.len() as u64)
                        .context("WriteBuffer: map failed")?;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
                    dev.unmap_memory2(buf.memory)
                        .context("WriteBuffer: unmap failed")?;
                }
            } else {
                let buf_device = buf.device_handle;
                let dev = state
                    .devices
                    .get(&buf_device)
                    .context("WriteBuffer: device invalid")?;
                let belt_entry = state.staging_belts.entry(buf_device).or_insert_with(|| {
                    staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE)
                });
                let (stg_buf, stg_off) = belt_entry.write(&state.instance, dev, data.as_slice())?;
                belt_slices.push((stg_buf, stg_off));
            }
        }
    }

    let devices = &state.devices;
    let compute_pipelines = &state.compute_pipelines;
    let buffers = &state.buffers;

    let logical_device = devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // The coarse pass pattern uses blocking `dispatch_compute` per command, then
    // `submit_recording` calls `submit` on a fresh encoder (often empty). Submitting an
    // empty primary command buffer + fence breaks on some Vulkan drivers (device lost /
    // wait failure on the trailing fence). A pre-signaled fence matches "no GPU work"
    // without touching the queue.
    if commands.is_empty() {
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { logical_device.device.create_fence(&fence_create_info, None) }
            .context("Failed to create signaled fence for empty submit")?;
        let token = state.next_compute_fence_token;
        state.next_compute_fence_token += 1;
        state
            .compute_fence_pool
            .insert(token, (device_handle, fence, None));
        return Ok(token);
    }

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

    if let Err(e) = unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) } {
        unsafe {
            logical_device
                .device
                .free_command_buffers(logical_device.command_pool, &[cmd]);
        }
        return Err(anyhow::anyhow!("Failed to begin command buffer: {:?}", e));
    }

    let mut texture_upload_scratch: Vec<super::texture::ComputeTextureScratch> = Vec::new();
    for command in commands {
        match command {
            GpuCommand::WriteTexture {
                texture,
                data,
                width,
                height,
            } => {
                texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                    &state.instance,
                    &state.devices,
                    &state.textures,
                    *texture,
                    data.as_slice(),
                    0,
                    0,
                    *width,
                    *height,
                )?);
            }
            GpuCommand::WriteTextureRegion {
                texture,
                x,
                y,
                width,
                height,
                data,
            } => {
                texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                    &state.instance,
                    &state.devices,
                    &state.textures,
                    *texture,
                    data.as_slice(),
                    *x,
                    *y,
                    *width,
                    *height,
                )?);
            }
            _ => {}
        }
    }

    // Cross-submission memory barrier: ensure writes from prior queue
    // visible to this submission's operations.  Vulkan guarantees execution
    // ordering for same-queue submissions but NOT memory visibility.
    unsafe {
        let acquire = vk::MemoryBarrier2::default()
            .src_stage_mask(
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            )
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            )
            .dst_access_mask(
                vk::AccessFlags2::TRANSFER_READ
                    | vk::AccessFlags2::TRANSFER_WRITE
                    | vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::SHADER_WRITE,
            );
        let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&acquire));
        logical_device.device.cmd_pipeline_barrier2(cmd, &dep);
    }

    // Track current pipeline for resource slot binding
    let mut current_pipeline: Option<ComputePipelineHandle> = None;
    let mut belt_idx = 0usize;
    let mut texture_upload_idx = 0usize;

    // Process commands (same logic as dispatch)
    for command in commands {
        match command {
            GpuCommand::SetPipeline(handle) => {
                if let Some(pipeline_state) = compute_pipelines.get(handle) {
                    unsafe {
                        logical_device.device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline_state.pipeline,
                        );

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
                    current_pipeline = Some(*handle);
                }
            }
            GpuCommand::BindResources {
                buffers: buffer_handles,
            } => {
                if let Some(pipeline) = current_pipeline.and_then(|p| compute_pipelines.get(&p)) {
                    let mut layout = PushLayout::default();
                    for (i, buffer_handle) in buffer_handles.iter().enumerate() {
                        if i >= types::MAX_BINDLESS_SLOTS {
                            break;
                        }
                        layout.bindless[i] = buffers
                            .get(buffer_handle)
                            .and_then(|b| b.bindless_index)
                            .unwrap_or(0) as u16;
                    }
                    unsafe {
                        logical_device.device.cmd_push_constants(
                            cmd,
                            pipeline.layout,
                            vk::ShaderStageFlags::ALL,
                            0,
                            bytemuck::bytes_of(&layout),
                        );
                    }
                }
            }
            GpuCommand::BindResourcesRaw {
                indices: raw_indices,
                user: raw_user,
            } => {
                if let Some(pipeline) = current_pipeline.and_then(|p| compute_pipelines.get(&p)) {
                    let mut layout = PushLayout::default();
                    for (i, &idx) in raw_indices.iter().enumerate() {
                        if i >= types::MAX_BINDLESS_SLOTS {
                            break;
                        }
                        layout.bindless[i] = idx as u16;
                    }
                    for (i, &val) in raw_user.iter().enumerate() {
                        if i >= types::MAX_USER_SLOTS {
                            break;
                        }
                        layout.user[i] = val;
                    }
                    unsafe {
                        logical_device.device.cmd_push_constants(
                            cmd,
                            pipeline.layout,
                            vk::ShaderStageFlags::ALL,
                            0,
                            bytemuck::bytes_of(&layout),
                        );
                    }
                }
            }
            GpuCommand::BindResourcesTyped {
                handles: typed_handles,
            } => {
                if let Some(pipeline) = current_pipeline.and_then(|p| compute_pipelines.get(&p)) {
                    let mut layout = PushLayout::default();
                    for (i, handle) in typed_handles.iter().enumerate() {
                        if i >= types::MAX_BINDLESS_SLOTS {
                            break;
                        }
                        layout.bindless[i] = handle.index() as u16;
                    }
                    unsafe {
                        logical_device.device.cmd_push_constants(
                            cmd,
                            pipeline.layout,
                            vk::ShaderStageFlags::ALL,
                            0,
                            bytemuck::bytes_of(&layout),
                        );
                    }
                }
            }
            GpuCommand::Dispatch {
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
            GpuCommand::DispatchIndirect { buffer, offset } => {
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
            GpuCommand::Barrier => {
                // No-op: Vulkan already emits pipeline barriers before each dispatch.
            }
            GpuCommand::ResourceBarrier { .. } => {
                // Falls back to global barrier behavior. Vulkan already inserts a
                // compute→compute pipeline barrier before each dispatch, so this
                // is a no-op. Per-resource VkBufferMemoryBarrier is a future optimization.
            }
            GpuCommand::ClearBuffer {
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
            GpuCommand::WriteBuffer {
                buffer: buf_handle,
                offset,
                data,
            } => {
                let buf_state = buffers
                    .get(buf_handle)
                    .context("WriteBuffer: invalid buffer handle")?;
                // HOST_VISIBLE / CPU_READABLE paths were handled in the pre-pass;
                // DEVICE_LOCAL storage uses the staging belt (see pre-pass).
                if buf_state.is_storage && buf_state.host_mapped.is_none() {
                    let (stg, stg_off) = belt_slices
                        .get(belt_idx)
                        .context("WriteBuffer: belt slice missing (internal error)")?;
                    belt_idx += 1;
                    let region = vk::BufferCopy {
                        src_offset: *stg_off,
                        dst_offset: *offset,
                        size: data.len() as u64,
                    };
                    unsafe {
                        logical_device.device.cmd_copy_buffer(
                            cmd,
                            *stg,
                            buf_state.buffer,
                            std::slice::from_ref(&region),
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
            GpuCommand::WriteTexture { .. } | GpuCommand::WriteTextureRegion { .. } => {
                let scratch = texture_upload_scratch
                    .get(texture_upload_idx)
                    .context("WriteTexture: scratch missing (internal)")?;
                texture_upload_idx += 1;
                super::texture::record_compute_texture_upload(
                    &state.devices,
                    &mut state.textures,
                    cmd,
                    scratch,
                )?;
            }
        }
    }

    debug_assert_eq!(
        texture_upload_idx,
        texture_upload_scratch.len(),
        "WriteTexture commands mismatch texture scratch pre-pass"
    );

    // Release barrier: flush compute/transfer writes so they are available to
    // subsequent queue submissions (e.g. the present-barrier layout transition
    // in the surface present path).  Same-queue ordering guarantees execution
    // order but NOT memory visibility across submits; this barrier closes the
    // gap by making all writes from this command buffer available before the
    // submit completes.
    unsafe {
        let release = vk::MemoryBarrier2::default()
            .src_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
            )
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&release));
        logical_device.device.cmd_pipeline_barrier2(cmd, &dep);
    }

    // End command buffer
    if let Err(e) = unsafe { logical_device.device.end_command_buffer(cmd) } {
        unsafe {
            logical_device
                .device
                .free_command_buffers(logical_device.command_pool, &[cmd]);
        }
        return Err(anyhow::anyhow!("Failed to end command buffer: {:?}", e));
    }

    let token = state.next_compute_fence_token;

    // If a surface has an acquired image, defer this submit so present() can
    // batch the compute CB + present-barrier CB in a single vkQueueSubmit.
    // This eliminates the cross-submit memory visibility gap that caused the
    // visual glitch on some drivers.
    let has_active_surface = state
        .surfaces
        .values()
        .any(|s| s.current_image_index.is_some());
    if has_active_surface {
        state.next_compute_fence_token += 1;
        state.deferred_pending_tokens.insert(token, None);
        state.deferred_compute.push(super::types::DeferredCompute {
            cmd,
            token,
        });

        if !texture_upload_scratch.is_empty() {
            let pooled: Vec<(vk::Buffer, vk::DeviceMemory)> = texture_upload_scratch
                .into_iter()
                .map(|s| (s.buffer, s.memory))
                .collect();
            state.compute_texture_staging_pool.insert(token, pooled);
        }

        state
            .staging_belts
            .entry(device_handle)
            .or_insert_with(|| staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE))
            .finish(token);

        return Ok(token);
    }

    // No active surface — submit immediately with a per-compute fence.
    let fence_create_info = vk::FenceCreateInfo::default();
    let fence =
        unsafe { logical_device.device.create_fence(&fence_create_info, None) }.map_err(|e| {
            unsafe {
                logical_device
                    .device
                    .free_command_buffers(logical_device.command_pool, &[cmd]);
            }
            anyhow::anyhow!("Failed to create fence: {:?}", e)
        })?;

    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

    if let Err(e) = unsafe {
        logical_device
            .device
            .queue_submit(logical_device.queue, &[submit_info], fence)
    } {
        unsafe {
            logical_device.device.destroy_fence(fence, None);
            logical_device
                .device
                .free_command_buffers(logical_device.command_pool, &[cmd]);
        }
        return Err(anyhow::anyhow!("Failed to submit command buffer: {:?}", e));
    }

    state.next_compute_fence_token += 1;
    state
        .compute_fence_pool
        .insert(token, (device_handle, fence, Some(cmd)));

    if !texture_upload_scratch.is_empty() {
        let pooled: Vec<(vk::Buffer, vk::DeviceMemory)> = texture_upload_scratch
            .into_iter()
            .map(|s| (s.buffer, s.memory))
            .collect();
        state.compute_texture_staging_pool.insert(token, pooled);
    }

    state
        .staging_belts
        .entry(device_handle)
        .or_insert_with(|| staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE))
        .finish(token);

    debug_assert_eq!(
        belt_idx,
        belt_slices.len(),
        "WriteBuffer DEVICE_LOCAL count must match belt pre-pass"
    );

    Ok(token)
}

/// Check if the fence for the given token has signaled.
pub(super) fn is_fence_complete(
    state: &super::types::VulkanState,
    _device_handle: DeviceHandle,
    token: FenceToken,
) -> bool {
    if let Some(maybe_fence) = state.deferred_pending_tokens.get(&token) {
        return match maybe_fence {
            None => false,
            Some(fence) => {
                let dh = _device_handle;
                state
                    .devices
                    .get(&dh)
                    .map(|ld| unsafe { ld.device.get_fence_status(*fence) }.unwrap_or(false))
                    .unwrap_or(true)
            }
        };
    }
    let Some((device_handle, fence, _)) = state.compute_fence_pool.get(&token) else {
        return true;
    };
    let Some(logical_device) = state.devices.get(device_handle) else {
        return true;
    };
    unsafe { logical_device.device.get_fence_status(*fence) }.unwrap_or_default()
}

/// Block until the fence signals, then destroy the fence, free the command buffer,
/// and drop the pool entry.
///
/// Tokens that were deferred (submitted as part of the single present submit)
/// are NOT in `compute_fence_pool`; their CBs are freed by `acquire()`.
/// This function still cleans up any associated staging resources.
pub(super) fn wait_fence(
    state: &mut super::types::VulkanState,
    _device_handle: DeviceHandle,
    token: FenceToken,
) -> Result<()> {
    let pool_entry = state
        .compute_fence_pool
        .get(&token)
        .map(|(d, f, _)| (*d, *f));

    if let Some((stored_device, fence)) = pool_entry {
        // Token was submitted with its own fence — wait and clean up.
        let wait_result = unsafe {
            let logical_device = state
                .devices
                .get(&stored_device)
                .context("Device for fence no longer exists")?;
            logical_device
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
        };

        let device_lost = matches!(wait_result, Err(vk::Result::ERROR_DEVICE_LOST));
        if !device_lost {
            if let Some((device_handle, fence, cmd_buf)) = state.compute_fence_pool.remove(&token) {
                let logical_device = state.devices.get(&device_handle).unwrap();
                unsafe {
                    if let Some(cb) = cmd_buf {
                        logical_device
                            .device
                            .free_command_buffers(logical_device.command_pool, &[cb]);
                    }
                    logical_device.device.destroy_fence(fence, None);
                }
                if let Some(staging) = state.compute_texture_staging_pool.remove(&token) {
                    super::texture::destroy_texture_staging_list(logical_device, staging);
                }
            }
        } else {
            tracing::warn!(
                %token,
                "skipping free_command_buffers/destroy_fence on lost device"
            );
        }

        if let Err(e) = wait_result {
            return Err(anyhow::anyhow!(
                "Failed to wait for Vulkan compute fence (token={}); VkResult={:?}; often VK_ERROR_DEVICE_LOST after GPU reset or resource exhaustion",
                token,
                e
            ));
        }
    } else if let Some(Some(fence)) = state.deferred_pending_tokens.get(&token).copied() {
        // Deferred token — CB is freed in acquire(). Wait on the frame fence
        // if present() has already submitted the batch.
        let logical_device = state
            .devices
            .get(&_device_handle)
            .context("Device for deferred fence no longer exists")?;
        let wait_result = unsafe {
            logical_device
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
        };
        if let Err(e) = &wait_result {
            tracing::warn!(%token, result = ?e, "wait_fence on deferred token's frame fence failed");
        }
        wait_result.context("Failed to wait for deferred token's frame fence")?;
    } else if let Some(None) = state.deferred_pending_tokens.get(&token).copied() {
        // Deferred token, but the batch is not on the GPU yet — staging cleanup
        // stays deferred to acquire() along with the CB.
    } else {
        // Unknown token (already cleaned up). Nothing to do.
    }
    Ok(())
}

/// Wait with timeout. On success or non-timeout error, removes and destroys the fence.
pub(super) fn wait_fence_timeout(
    state: &mut super::types::VulkanState,
    _device: DeviceHandle,
    token: FenceToken,
    timeout_ms: u32,
) -> Result<bool> {
    let (stored_device, fence) = state
        .compute_fence_pool
        .get(&token)
        .map(|(d, f, _)| (*d, *f))
        .context("Invalid fence token")?;
    let logical_device = state
        .devices
        .get(&stored_device)
        .context("Device for fence no longer exists")?;

    // vkWaitForFences uses nanoseconds
    let timeout_ns = u64::from(timeout_ms) * 1_000_000;

    let result = unsafe {
        logical_device
            .device
            .wait_for_fences(&[fence], true, timeout_ns)
    };

    match result {
        Ok(()) => {
            if let Some((_dh, f, cb)) = state.compute_fence_pool.remove(&token) {
                unsafe {
                    if let Some(c) = cb {
                        logical_device
                            .device
                            .free_command_buffers(logical_device.command_pool, &[c]);
                    }
                    logical_device.device.destroy_fence(f, None);
                }
                if let Some(staging) = state.compute_texture_staging_pool.remove(&token) {
                    super::texture::destroy_texture_staging_list(logical_device, staging);
                }
            }
            Ok(true)
        }
        Err(vk::Result::TIMEOUT) => Ok(false),
        Err(e) => {
            // Same device-lost cleanup hazard as wait_fence: skip Vulkan
            // destroy calls if the device is lost (vkDestroyDevice will
            // implicitly reclaim them).
            let device_lost = matches!(e, vk::Result::ERROR_DEVICE_LOST);
            if let Some((_dh, f, cb)) = state.compute_fence_pool.remove(&token) {
                if !device_lost {
                    unsafe {
                        if let Some(c) = cb {
                            logical_device
                                .device
                                .free_command_buffers(logical_device.command_pool, &[c]);
                        }
                        logical_device.device.destroy_fence(f, None);
                    }
                    if let Some(staging) = state.compute_texture_staging_pool.remove(&token) {
                        super::texture::destroy_texture_staging_list(logical_device, staging);
                    }
                } else {
                    tracing::warn!(
                        %token,
                        "skipping free_command_buffers/destroy_fence on lost device in wait_fence_timeout (avoids driver heap corruption on some Windows drivers)"
                    );
                }
            }
            Err(anyhow::anyhow!(
                "Failed to wait for Vulkan compute fence (token={}): {:?}",
                token,
                e
            ))
        }
    }
}
