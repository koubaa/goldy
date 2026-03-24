//! Compute pipeline and dispatch logic.

use super::super::{ComputeCommand, ComputePipelineHandle, DeviceHandle, FenceToken, ShaderHandle};
use super::shader::parse_numthreads;
use super::types::PUSH_CONSTANTS_SLOT;
use super::types::{BindlessIndices, ComputePipelineState, MetalState, MAX_PUSH_CONSTANT_INDICES};
use crate::slang::SlangStage;
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{MTLCommandBufferStatus, MTLSize};
use std::sync::atomic::Ordering;

/// Create a compute pipeline.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    compute_shader: ShaderHandle,
) -> Result<ComputePipelineHandle> {
    super::shader::ensure_stage_compiled(
        &state.slang_compiler,
        &state.devices,
        &mut state.shaders,
        compute_shader,
        SlangStage::Compute,
    )?;

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let shader = state
        .shaders
        .get(&compute_shader)
        .context("Invalid compute shader")?;

    let workgroup_size = parse_numthreads(&shader.slang_source).unwrap_or([64, 1, 1]);

    let library = shader.compute_library.as_ref().unwrap();

    let function = library
        .get_function("cs_main", None)
        .map_err(|e| anyhow::anyhow!("Failed to get compute function: {}", e))?;

    let pipeline = logical_device
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("Failed to create compute pipeline: {}", e))?;

    let handle = state.next_compute_pipeline_handle;
    state.next_compute_pipeline_handle += 1;

    state.compute_pipelines.insert(
        handle,
        ComputePipelineState {
            device_handle,
            pipeline,
            workgroup_size,
        },
    );

    tracing::debug!(
        "Created compute pipeline (handle={}, workgroup_size={:?})",
        handle,
        workgroup_size
    );
    Ok(handle)
}

/// Destroy a compute pipeline.
pub(super) fn destroy(state: &mut MetalState, pipeline_handle: ComputePipelineHandle) {
    state.compute_pipelines.remove(&pipeline_handle);
}

/// Begin a fresh compute encoder on the command buffer with heap and argument buffer bindings.
///
/// Calls `use_resource` on each individual buffer so Metal's hazard tracking
/// can detect cross-encoder dependencies (e.g. compute→blit→compute). Without
/// this, Metal may overlap encoders that share buffers accessed via argument
/// buffers, since the indirect access is opaque to automatic hazard tracking.
fn begin_compute_encoder<'a>(
    command_buffer: &'a mtl::CommandBufferRef,
    state: &MetalState,
    logical_device: &super::types::LogicalDevice,
    device_handle: DeviceHandle,
) -> &'a mtl::ComputeCommandEncoderRef {
    let encoder = command_buffer.new_compute_command_encoder();
    logical_device.heap_allocator.use_heaps_for_compute(encoder);
    if logical_device.heap_texture_count > 0 {
        encoder.use_heap(&logical_device.texture_heap);
    }
    for buf_state in state.buffers.values() {
        if buf_state.device_handle == device_handle {
            encoder.use_resource(
                &buf_state.buffer,
                mtl::MTLResourceUsage::Read | mtl::MTLResourceUsage::Write,
            );
        }
    }
    encoder.set_buffer(0, Some(&logical_device.argument_buffer), 0);
    encoder
}

/// Record compute commands to a command buffer (shared by submit and dispatch).
fn record_commands_to_buffer(
    state: &MetalState,
    command_buffer: &mtl::CommandBufferRef,
    logical_device: &super::types::LogicalDevice,
    device_handle: DeviceHandle,
    commands: &[ComputeCommand],
) -> Result<()> {
    let mut encoder: Option<&mtl::ComputeCommandEncoderRef> = None;
    let mut current_pipeline: Option<&ComputePipelineState> = None;

    macro_rules! end_compute {
        () => {
            if let Some(enc) = encoder.take() {
                enc.end_encoding();
            }
        };
    }

    macro_rules! ensure_compute {
        () => {
            if encoder.is_none() {
                let enc =
                    begin_compute_encoder(command_buffer, state, logical_device, device_handle);
                if let Some(pipeline) = current_pipeline {
                    enc.set_compute_pipeline_state(&pipeline.pipeline);
                }
                encoder = Some(enc);
            }
        };
    }

    for cmd in commands {
        match cmd {
            ComputeCommand::SetPipeline(handle) => {
                ensure_compute!();
                if let Some(pipeline) = state.compute_pipelines.get(handle) {
                    encoder
                        .unwrap()
                        .set_compute_pipeline_state(&pipeline.pipeline);
                    current_pipeline = Some(pipeline);
                }
            }
            ComputeCommand::SetPushConstants { buffers } => {
                ensure_compute!();
                let mut indices = BindlessIndices::default();
                for (i, buffer_handle) in buffers.iter().enumerate() {
                    if i >= MAX_PUSH_CONSTANT_INDICES {
                        break;
                    }
                    if let Some(buf) = state.buffers.get(buffer_handle) {
                        indices.indices[i] = buf.arg_buffer_index;
                    }
                }
                let indices_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        &indices as *const _ as *const u8,
                        std::mem::size_of::<BindlessIndices>(),
                    )
                };
                encoder.unwrap().set_bytes(
                    PUSH_CONSTANTS_SLOT,
                    indices_bytes.len() as u64,
                    indices_bytes.as_ptr() as *const _,
                );
            }
            ComputeCommand::SetPushConstantsRaw {
                indices: raw_indices,
            } => {
                ensure_compute!();
                let mut indices = BindlessIndices::default();
                for (i, &idx) in raw_indices.iter().enumerate() {
                    if i >= MAX_PUSH_CONSTANT_INDICES {
                        break;
                    }
                    indices.indices[i] = idx;
                }
                let indices_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        &indices as *const _ as *const u8,
                        std::mem::size_of::<BindlessIndices>(),
                    )
                };
                encoder.unwrap().set_bytes(
                    PUSH_CONSTANTS_SLOT,
                    indices_bytes.len() as u64,
                    indices_bytes.as_ptr() as *const _,
                );
            }
            ComputeCommand::Dispatch {
                workgroups_x,
                workgroups_y,
                workgroups_z,
            } => {
                ensure_compute!();
                if let Some(pipeline) = current_pipeline {
                    let threads_per_group = MTLSize {
                        width: pipeline.workgroup_size[0] as u64,
                        height: pipeline.workgroup_size[1] as u64,
                        depth: pipeline.workgroup_size[2] as u64,
                    };
                    let threadgroups = MTLSize {
                        width: *workgroups_x as u64,
                        height: *workgroups_y as u64,
                        depth: *workgroups_z as u64,
                    };
                    encoder
                        .unwrap()
                        .dispatch_thread_groups(threadgroups, threads_per_group);
                }
            }
            ComputeCommand::DispatchIndirect { buffer, offset } => {
                ensure_compute!();
                let buf_state = match state.buffers.get(buffer) {
                    Some(b) => b,
                    None => {
                        end_compute!();
                        anyhow::bail!("DispatchIndirect: invalid buffer handle");
                    }
                };
                let pipeline = match current_pipeline {
                    Some(p) => p,
                    None => {
                        end_compute!();
                        anyhow::bail!("DispatchIndirect: no pipeline bound");
                    }
                };
                let threads_per_group = MTLSize {
                    width: pipeline.workgroup_size[0] as u64,
                    height: pipeline.workgroup_size[1] as u64,
                    depth: pipeline.workgroup_size[2] as u64,
                };
                encoder.unwrap().dispatch_thread_groups_indirect(
                    &buf_state.buffer,
                    *offset,
                    threads_per_group,
                );
            }
            ComputeCommand::ClearBuffer {
                buffer,
                offset,
                size,
            } => {
                let buf_state = match state.buffers.get(buffer) {
                    Some(b) => b,
                    None => {
                        end_compute!();
                        anyhow::bail!("ClearBuffer: invalid buffer handle");
                    }
                };
                let clear_size = if *size == 0 {
                    buf_state.size.saturating_sub(*offset)
                } else {
                    *size
                };
                if clear_size > 0 {
                    end_compute!();
                    let blit = command_buffer.new_blit_command_encoder();
                    let range = mtl::NSRange::new(*offset, clear_size);
                    blit.fill_buffer(&buf_state.buffer, range, 0);
                    blit.end_encoding();
                }
            }
        }
    }

    end_compute!();
    Ok(())
}

/// Submit compute commands without blocking. Returns a fence token for polling/waiting.
pub(super) fn submit(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    commands: &[ComputeCommand],
) -> Result<FenceToken> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let command_buffer_ref = logical_device.command_queue.new_command_buffer();
    record_commands_to_buffer(
        state,
        command_buffer_ref,
        logical_device,
        device_handle,
        commands,
    )?;

    let owned_buffer = command_buffer_ref.to_owned();
    let token = state
        .next_compute_fence_token
        .fetch_add(1, Ordering::SeqCst);
    state
        .compute_fence_pool
        .lock()
        .unwrap()
        .insert(token, owned_buffer);

    command_buffer_ref.commit();

    Ok(token)
}

/// Check if the fence for the given token has signaled (work complete).
pub(super) fn is_fence_complete(
    state: &MetalState,
    _device: DeviceHandle,
    token: FenceToken,
) -> bool {
    let pool = state.compute_fence_pool.lock().unwrap();
    if let Some(buf) = pool.get(&token) {
        buf.status() == MTLCommandBufferStatus::Completed
    } else {
        true // Already removed (waited on), consider complete
    }
}

/// Block until the fence signals.
pub(super) fn wait_fence(
    state: &MetalState,
    _device: DeviceHandle,
    token: FenceToken,
) -> Result<()> {
    let mut pool = state.compute_fence_pool.lock().unwrap();
    if let Some(buf) = pool.remove(&token) {
        buf.wait_until_completed();
    }
    Ok(())
}

/// Wait with timeout. Returns Ok(true) if signaled, Ok(false) if timeout elapsed.
pub(super) fn wait_fence_timeout(
    state: &MetalState,
    _device: DeviceHandle,
    token: FenceToken,
    timeout_ms: u32,
) -> Result<bool> {
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_millis(timeout_ms as u64);

    loop {
        {
            let pool = state.compute_fence_pool.lock().unwrap();
            if let Some(buf) = pool.get(&token) {
                if buf.status() == MTLCommandBufferStatus::Completed {
                    drop(pool);
                    let mut p = state.compute_fence_pool.lock().unwrap();
                    p.remove(&token);
                    return Ok(true);
                }
            } else {
                return Ok(true); // Already removed
            }
        }
        if start.elapsed() >= timeout {
            return Ok(false);
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
