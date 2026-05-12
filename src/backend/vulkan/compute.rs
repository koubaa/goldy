//! Compute pipeline and dispatch logic.

use super::super::shared;
use super::staging;
use super::types::{ComputePipelineState, LogicalDevice, PushLayout};
use super::{ComputePipelineHandle, DeviceHandle, RenderTargetHandle, SurfaceHandle};
use crate::backend::{GpuCommand, GraphCommand};
use crate::timeline::TimelineValue;
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

/// Reap fences that have already signaled from the compute fence pool.
///
/// Without this, every `submit_graph` that doesn't have a paired `wait_fence`
/// (the common ekrano pattern: only the *last* timeline value in a recording is
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
            if let Some(staging) = state
                .compute_texture_staging_pool
                .remove(&(device_handle, token))
            {
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
            logical_device.pipeline_cache,
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
            push_constant_categories: Vec::new(),
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
    defer_to_present_for_surface: Option<SurfaceHandle>,
) -> Result<TimelineValue> {
    // Reap any previously-submitted fences that have already signaled. Keeps
    // the pool bounded when callers (ekrano) don't wait on every intermediate submit.
    // Belt before reap: need live VkFence handles to poll completion.
    //
    // For timeline-keyed staging chunks (standalone-submit path) we also need the
    // current device timeline counter so `reclaim` knows which chunks are safe to
    // recycle without reaching into `compute_fence_pool`.
    let completed_timeline = state
        .devices
        .get(&device_handle)
        .map(|ld| unsafe {
            ld.device
                .get_semaphore_counter_value(ld.timeline_semaphore)
                .unwrap_or(0)
        })
        .unwrap_or(0);

    for belt in state.staging_belts.values_mut() {
        belt.reclaim(
            &state.compute_fence_pool,
            &state.devices,
            completed_timeline,
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
                let (stg_buf, stg_off) = belt_entry.write(&state.instance, dev, data)?;
                belt_slices.push((stg_buf, stg_off));
            }
        }
    }

    if commands.is_empty() {
        let signal_value = {
            let ld = state
                .devices
                .get(&device_handle)
                .context("Invalid device handle")?;
            ld.timeline_next
        };
        let ld = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(ld.timeline_semaphore)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let submit_info2 =
            vk::SubmitInfo2::default().signal_semaphore_infos(std::slice::from_ref(&signal_info));
        let r = unsafe {
            ld.device.queue_submit2(
                ld.queue,
                std::slice::from_ref(&submit_info2),
                vk::Fence::null(),
            )
        };
        r.context("Failed queue_submit2 for empty compute submit")?;
        if let Some(ld) = state.devices.get_mut(&device_handle) {
            ld.timeline_next = signal_value.saturating_add(1);
            ld.process_deletion_queue_up_to_gpu_progress();
        }
        return Ok(signal_value);
    }

    let compute_pipelines = &state.compute_pipelines;
    let buffers = &state.buffers;

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
                    data,
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
                    data,
                    *x,
                    *y,
                    *width,
                    *height,
                )?);
            }
            _ => {}
        }
    }

    let (cmd, belt_idx, _texture_upload_idx) = {
        let logical_device = state
            .devices
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
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        if let Err(e) = unsafe { logical_device.device.begin_command_buffer(cmd, &begin_info) } {
            unsafe {
                logical_device
                    .device
                    .free_command_buffers(logical_device.command_pool, &[cmd]);
            }
            return Err(anyhow::anyhow!("Failed to begin command buffer: {:?}", e));
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
                    if let Some(pipeline) = current_pipeline.and_then(|p| compute_pipelines.get(&p))
                    {
                        let mut layout = PushLayout::default();
                        shared::fill_bindless(
                            &mut layout,
                            buffer_handles.iter().map(|h| {
                                buffers.get(h).and_then(|b| b.bindless_index).unwrap_or(0)
                            }),
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
                GpuCommand::BindResourcesRaw {
                    indices: raw_indices,
                    user: raw_user,
                } => {
                    if let Some(pipeline) = current_pipeline.and_then(|p| compute_pipelines.get(&p))
                    {
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
                GpuCommand::BindResourcesTyped {
                    handles: typed_handles,
                } => {
                    if let Some(pipeline) = current_pipeline.and_then(|p| compute_pipelines.get(&p))
                    {
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

        (cmd, belt_idx, texture_upload_idx)
    };

    if let Some(sid) = defer_to_present_for_surface {
        let surf = state
            .surfaces
            .get_mut(&sid)
            .context("defer compute: invalid surface handle")?;
        if surf.current_image_index.is_none() {
            anyhow::bail!("defer compute: surface has no acquired image");
        }
        let cf = surf.current_frame;
        surf.frame_sync[cf].deferred_compute_cbs.push(cmd);
        if !texture_upload_scratch.is_empty() {
            let pooled: Vec<(vk::Buffer, vk::DeviceMemory)> = texture_upload_scratch
                .into_iter()
                .map(|s| (s.buffer, s.memory))
                .collect();
            surf.frame_sync[cf]
                .pending_compute_texture_staging
                .extend(pooled);
        }
        // Staging belt `finish` and compute_texture_staging_pool insert happen in
        // `surface::present` once the timeline signal value for this batch is known.
        return Ok(0);
    }

    // Standalone submit: signal device timeline semaphore (Vulkan 1.2+).
    let signal_value = {
        let ld = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        ld.timeline_next
    };

    let submit_device_core = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
    let signal_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(submit_device_core.timeline_semaphore)
        .value(signal_value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let submit_info2 = vk::SubmitInfo2::default()
        .command_buffer_infos(std::slice::from_ref(&cmd_info))
        .signal_semaphore_infos(std::slice::from_ref(&signal_info));

    if let Err(e) = unsafe {
        submit_device_core.device.queue_submit2(
            submit_device_core.queue,
            std::slice::from_ref(&submit_info2),
            vk::Fence::null(),
        )
    } {
        unsafe {
            submit_device_core
                .device
                .free_command_buffers(submit_device_core.command_pool, &[cmd]);
        }
        return Err(anyhow::anyhow!(
            "Failed to queue_submit2 command buffer: {:?}",
            e
        ));
    }

    if let Some(ld) = state.devices.get_mut(&device_handle) {
        ld.timeline_next = signal_value.saturating_add(1);
    }

    state
        .timeline_cmd_buffers
        .entry(signal_value)
        .or_default()
        .push((device_handle, cmd));

    if !texture_upload_scratch.is_empty() {
        let pooled: Vec<(vk::Buffer, vk::DeviceMemory)> = texture_upload_scratch
            .into_iter()
            .map(|s| (s.buffer, s.memory))
            .collect();
        state
            .compute_texture_staging_pool
            .insert((device_handle, signal_value), pooled);
    }

    state
        .staging_belts
        .entry(device_handle)
        .or_insert_with(|| staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE))
        .finish(signal_value);

    debug_assert_eq!(
        belt_idx,
        belt_slices.len(),
        "WriteBuffer DEVICE_LOCAL count must match belt pre-pass"
    );

    if let Some(ld) = state.devices.get_mut(&device_handle) {
        ld.process_deletion_queue_up_to_gpu_progress();
    }

    Ok(signal_value)
}

/// Submit mixed compute + render graph commands in a single command buffer.
///
/// Eliminates CPU waits between compute and render segments by recording
/// everything into one `VkCommandBuffer` and performing a single
/// `queue_submit2` with a timeline semaphore signal at the end.
pub(super) fn submit_graph(
    state: &mut super::types::VulkanState,
    device_handle: DeviceHandle,
    commands: &[GraphCommand],
) -> Result<TimelineValue> {
    // --- Same housekeeping as `submit` ---
    let completed_timeline = state
        .devices
        .get(&device_handle)
        .map(|ld| unsafe {
            ld.device
                .get_semaphore_counter_value(ld.timeline_semaphore)
                .unwrap_or(0)
        })
        .unwrap_or(0);

    for belt in state.staging_belts.values_mut() {
        belt.reclaim(
            &state.compute_fence_pool,
            &state.devices,
            completed_timeline,
        )?;
    }
    reap_signaled_fences(state);

    // --- Pre-pass: stage CPU data for WriteBuffer in compute segments ---
    let mut belt_slices: Vec<(vk::Buffer, u64)> = Vec::new();
    let mut texture_upload_scratch: Vec<super::texture::ComputeTextureScratch> = Vec::new();

    for graph_cmd in commands {
        if let GraphCommand::Compute(gpu_cmd) = graph_cmd {
            match gpu_cmd {
                GpuCommand::WriteBuffer {
                    buffer: buf_handle,
                    offset,
                    data,
                } => {
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
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr(),
                                ptr as *mut u8,
                                data.len(),
                            );
                            dev.unmap_memory2(buf.memory)
                                .context("WriteBuffer: unmap failed")?;
                        }
                    } else {
                        let buf_device = buf.device_handle;
                        let dev = state
                            .devices
                            .get(&buf_device)
                            .context("WriteBuffer: device invalid")?;
                        let belt_entry =
                            state.staging_belts.entry(buf_device).or_insert_with(|| {
                                staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE)
                            });
                        let (stg_buf, stg_off) = belt_entry.write(&state.instance, dev, data)?;
                        belt_slices.push((stg_buf, stg_off));
                    }
                }
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
                        data,
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
                        data,
                        *x,
                        *y,
                        *width,
                        *height,
                    )?);
                }
                _ => {}
            }
        }
    }

    // --- Allocate single command buffer ---
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(logical_device.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe { logical_device.device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer")?;
    let cmd = cmd_buffers[0];

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

    // Cross-submission acquire barrier
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

    // --- Walk GraphCommands (inline to satisfy the borrow checker) ---
    let compute_pipelines = &state.compute_pipelines;
    let buffers = &state.buffers;
    let mut current_compute_pipeline: Option<ComputePipelineHandle> = None;
    let mut belt_idx = 0usize;
    let mut texture_upload_idx = 0usize;
    let mut rendered_targets: Vec<RenderTargetHandle> = Vec::new();

    for graph_cmd in commands {
        match graph_cmd {
            GraphCommand::Compute(gpu_cmd) => match gpu_cmd {
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
                        current_compute_pipeline = Some(*handle);
                    }
                }
                GpuCommand::BindResources {
                    buffers: buffer_handles,
                } => {
                    if let Some(pipeline) =
                        current_compute_pipeline.and_then(|p| compute_pipelines.get(&p))
                    {
                        let mut layout = PushLayout::default();
                        shared::fill_bindless(
                            &mut layout,
                            buffer_handles.iter().map(|h| {
                                buffers.get(h).and_then(|b| b.bindless_index).unwrap_or(0)
                            }),
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
                GpuCommand::BindResourcesRaw {
                    indices: raw_indices,
                    user: raw_user,
                } => {
                    if let Some(pipeline) =
                        current_compute_pipeline.and_then(|p| compute_pipelines.get(&p))
                    {
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
                GpuCommand::BindResourcesTyped {
                    handles: typed_handles,
                } => {
                    if let Some(pipeline) =
                        current_compute_pipeline.and_then(|p| compute_pipelines.get(&p))
                    {
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
                GpuCommand::Barrier => {}
                GpuCommand::ResourceBarrier { .. } => {}
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
            },
            GraphCommand::Render {
                target,
                commands: render_cmds,
            } => {
                // Flush compute writes before the render pass
                unsafe {
                    let barrier = vk::MemoryBarrier2::default()
                        .src_stage_mask(
                            vk::PipelineStageFlags2::COMPUTE_SHADER
                                | vk::PipelineStageFlags2::TRANSFER,
                        )
                        .src_access_mask(
                            vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
                        )
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                                | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::VERTEX_SHADER
                                | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                                | vk::AccessFlags2::SHADER_READ
                                | vk::AccessFlags2::MEMORY_READ,
                        );
                    let dep = vk::DependencyInfo::default()
                        .memory_barriers(std::slice::from_ref(&barrier));
                    logical_device.device.cmd_pipeline_barrier2(cmd, &dep);
                }

                let pipelines = &state.pipelines;
                let rt_buffers = &state.buffers;
                super::render_target::record_render_pass_to_buffer(
                    &state.devices,
                    &state.render_targets,
                    device_handle,
                    *target,
                    render_cmds,
                    cmd,
                    |cb, cmds, ld, cur_pipe| {
                        super::render_commands::record(
                            cb, cmds, ld, pipelines, rt_buffers, cur_pipe,
                        )
                    },
                )?;

                rendered_targets.push(*target);

                // Make render writes visible to subsequent compute
                unsafe {
                    let barrier = vk::MemoryBarrier2::default()
                        .src_stage_mask(
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        )
                        .src_access_mask(
                            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::COMPUTE_SHADER
                                | vk::PipelineStageFlags2::TRANSFER,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2::SHADER_READ
                                | vk::AccessFlags2::SHADER_WRITE
                                | vk::AccessFlags2::TRANSFER_READ
                                | vk::AccessFlags2::TRANSFER_WRITE,
                        );
                    let dep = vk::DependencyInfo::default()
                        .memory_barriers(std::slice::from_ref(&barrier));
                    logical_device.device.cmd_pipeline_barrier2(cmd, &dep);
                }
            }
        }
    }

    // --- Release barrier ---
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

    // --- End + submit ---
    if let Err(e) = unsafe { logical_device.device.end_command_buffer(cmd) } {
        unsafe {
            logical_device
                .device
                .free_command_buffers(logical_device.command_pool, &[cmd]);
        }
        return Err(anyhow::anyhow!("Failed to end command buffer: {:?}", e));
    }

    let signal_value = {
        let ld = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        ld.timeline_next
    };

    let submit_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
    let signal_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(submit_device.timeline_semaphore)
        .value(signal_value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let submit_info2 = vk::SubmitInfo2::default()
        .command_buffer_infos(std::slice::from_ref(&cmd_info))
        .signal_semaphore_infos(std::slice::from_ref(&signal_info));

    if let Err(e) = unsafe {
        submit_device.device.queue_submit2(
            submit_device.queue,
            std::slice::from_ref(&submit_info2),
            vk::Fence::null(),
        )
    } {
        unsafe {
            submit_device
                .device
                .free_command_buffers(submit_device.command_pool, &[cmd]);
        }
        return Err(anyhow::anyhow!(
            "Failed to queue_submit2 command buffer: {:?}",
            e
        ));
    }

    if let Some(ld) = state.devices.get_mut(&device_handle) {
        ld.timeline_next = signal_value.saturating_add(1);
    }

    state
        .timeline_cmd_buffers
        .entry(signal_value)
        .or_default()
        .push((device_handle, cmd));

    if !texture_upload_scratch.is_empty() {
        let pooled: Vec<(vk::Buffer, vk::DeviceMemory)> = texture_upload_scratch
            .into_iter()
            .map(|s| (s.buffer, s.memory))
            .collect();
        state
            .compute_texture_staging_pool
            .insert((device_handle, signal_value), pooled);
    }

    state
        .staging_belts
        .entry(device_handle)
        .or_insert_with(|| staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE))
        .finish(signal_value);

    // Mark rendered targets
    for t in rendered_targets {
        if let Some(rt) = state.render_targets.get_mut(&t) {
            rt.has_rendered = true;
        }
    }

    if let Some(ld) = state.devices.get_mut(&device_handle) {
        ld.process_deletion_queue_up_to_gpu_progress();
    }

    Ok(signal_value)
}

pub(super) fn reap_timeline_cmd_buffers_up_to(
    state: &mut super::types::VulkanState,
    device_handle: DeviceHandle,
    max_completed_value: u64,
) {
    let keys: Vec<u64> = state
        .timeline_cmd_buffers
        .keys()
        .copied()
        .filter(|k| *k <= max_completed_value)
        .collect();
    for k in keys {
        if let Some(entries) = state.timeline_cmd_buffers.remove(&k) {
            let mut reinsert: Vec<(DeviceHandle, vk::CommandBuffer)> = Vec::new();
            for (dh, cb) in entries {
                if dh == device_handle {
                    if let Some(ld) = state.devices.get(&dh) {
                        unsafe {
                            ld.device.free_command_buffers(ld.command_pool, &[cb]);
                        }
                    }
                } else {
                    reinsert.push((dh, cb));
                }
            }
            if !reinsert.is_empty() {
                state.timeline_cmd_buffers.insert(k, reinsert);
            }
        }
        if let Some(staging) = state
            .compute_texture_staging_pool
            .remove(&(device_handle, k))
        {
            if let Some(ld) = state.devices.get(&device_handle) {
                super::texture::destroy_texture_staging_list(ld, staging);
            }
        }
    }
}
