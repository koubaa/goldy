//! Compute pipeline and dispatch logic.

use super::super::{ComputeCommand, ComputePipelineHandle, DeviceHandle, ShaderHandle};
use super::shader::parse_numthreads;
use super::types::PUSH_CONSTANTS_SLOT;
use super::types::{BindlessIndices, ComputePipelineState, MetalState, MAX_PUSH_CONSTANT_INDICES};
use crate::slang::SlangStage;
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::MTLSize;

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
fn begin_compute_encoder<'a>(
    command_buffer: &'a mtl::CommandBufferRef,
    logical_device: &super::types::LogicalDevice,
) -> &'a mtl::ComputeCommandEncoderRef {
    let encoder = command_buffer.new_compute_command_encoder();
    if logical_device.heap_buffer_count > 0 {
        encoder.use_heap(&logical_device.buffer_heap);
    }
    if logical_device.heap_texture_count > 0 {
        encoder.use_heap(&logical_device.texture_heap);
    }
    encoder.set_buffer(0, Some(&logical_device.argument_buffer), 0);
    encoder
}

/// Dispatch compute commands.
///
/// ClearBuffer commands are executed via a blit encoder (`fill_buffer`) so they
/// are ordered correctly with respect to surrounding compute dispatches.
pub(super) fn dispatch(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    commands: &[ComputeCommand],
) -> Result<()> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let command_buffer = logical_device.command_queue.new_command_buffer();

    // Defer creating the first compute encoder until the first non-ClearBuffer
    // command so we never emit an empty encoder before a leading blit.
    let mut encoder: Option<&mtl::ComputeCommandEncoderRef> = None;
    let mut current_pipeline: Option<&ComputePipelineState> = None;

    /// End the active compute encoder, if any.
    macro_rules! end_compute {
        () => {
            if let Some(enc) = encoder.take() {
                enc.end_encoding();
            }
        };
    }

    /// Ensure a compute encoder is active, (re-)creating one if needed and
    /// rebinding the pipeline state that was set before the last blit.
    macro_rules! ensure_compute {
        () => {
            if encoder.is_none() {
                let enc = begin_compute_encoder(command_buffer, logical_device);
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
                    // End compute encoder, issue GPU-ordered fill via blit encoder,
                    // then lazily resume compute on the next non-clear command.
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
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}
