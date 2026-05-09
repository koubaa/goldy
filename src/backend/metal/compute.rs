//! Compute pipeline and dispatch logic.

use super::super::{ComputePipelineHandle, DeviceHandle, GpuCommand, ShaderHandle};
use super::shader::parse_numthreads;
use super::types::RESOURCE_SLOT_BUFFER;
use super::types::{
    ComputePipelineState, MetalState, PushLayout, MAX_BINDLESS_SLOTS, MAX_USER_SLOTS,
    TOTAL_PUSH_BYTES,
};
use crate::slang::SlangStage;
use crate::timeline::TimelineValue;
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{MTLCommandBufferStatus, MTLOrigin, MTLSize};
use objc::{msg_send, sel, sel_impl};
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
            push_constant_categories: Vec::new(),
            shader_debug_name: String::new(),
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
    logical_device.texture_heap.use_heaps_for_compute(encoder);
    for buf_state in state.buffers.values() {
        if buf_state.device_handle == device_handle {
            encoder.use_resource(
                &buf_state.buffer,
                mtl::MTLResourceUsage::Read | mtl::MTLResourceUsage::Write,
            );
        }
    }
    // Declare textures resident for indirect access through the argument buffer.
    //
    // Heap-allocated textures are already covered by `use_heaps_for_compute`,
    // but swapchain drawables (CAMetalLayer-owned `MTLTexture`s registered
    // transiently in `state.textures`) are NOT in any Goldy-owned heap, so
    // Metal Tier-2 bindless will read them as unresident unless we explicitly
    // call `use_resource` on them before dispatch. Calling `use_resource` on
    // already-heap-resident textures is safe and idempotent.
    for tex_state in state.textures.values() {
        if tex_state.device_handle == device_handle {
            let usage = if tex_state.is_storage_image {
                mtl::MTLResourceUsage::Read | mtl::MTLResourceUsage::Write
            } else {
                mtl::MTLResourceUsage::Read
            };
            encoder.use_resource(&tex_state.texture, usage);
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
    commands: &[GpuCommand],
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
            GpuCommand::SetPipeline(handle) => {
                ensure_compute!();
                if let Some(pipeline) = state.compute_pipelines.get(handle) {
                    encoder
                        .unwrap()
                        .set_compute_pipeline_state(&pipeline.pipeline);
                    current_pipeline = Some(pipeline);
                }
            }
            GpuCommand::BindResources { buffers } => {
                ensure_compute!();
                let mut layout = PushLayout::default();
                for (i, buffer_handle) in buffers.iter().enumerate() {
                    if i >= MAX_BINDLESS_SLOTS {
                        break;
                    }
                    if let Some(buf) = state.buffers.get(buffer_handle) {
                        layout.bindless[i] = buf.arg_buffer_index as u16;
                    }
                }
                let layout_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(&layout as *const _ as *const u8, TOTAL_PUSH_BYTES)
                };
                encoder.unwrap().set_bytes(
                    RESOURCE_SLOT_BUFFER,
                    layout_bytes.len() as u64,
                    layout_bytes.as_ptr() as *const _,
                );
            }
            GpuCommand::BindResourcesRaw {
                indices: raw_indices,
                user: raw_user,
            } => {
                ensure_compute!();
                let mut layout = PushLayout::default();
                for (i, &idx) in raw_indices.iter().enumerate() {
                    if i >= MAX_BINDLESS_SLOTS {
                        break;
                    }
                    layout.bindless[i] = idx as u16;
                }
                for (i, &val) in raw_user.iter().enumerate() {
                    if i >= MAX_USER_SLOTS {
                        break;
                    }
                    layout.user[i] = val;
                }
                let layout_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(&layout as *const _ as *const u8, TOTAL_PUSH_BYTES)
                };
                encoder.unwrap().set_bytes(
                    RESOURCE_SLOT_BUFFER,
                    layout_bytes.len() as u64,
                    layout_bytes.as_ptr() as *const _,
                );
            }
            GpuCommand::BindResourcesTyped { handles } => {
                ensure_compute!();
                if let Some(pipeline) = current_pipeline {
                    crate::backend::validate_typed_push_constants(
                        handles,
                        &pipeline.push_constant_categories,
                        &pipeline.shader_debug_name,
                    )?;
                }
                let mut layout = PushLayout::default();
                for (i, handle) in handles.iter().enumerate() {
                    if i >= MAX_BINDLESS_SLOTS {
                        break;
                    }
                    layout.bindless[i] = handle.index() as u16;
                }
                let layout_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(&layout as *const _ as *const u8, TOTAL_PUSH_BYTES)
                };
                encoder.unwrap().set_bytes(
                    RESOURCE_SLOT_BUFFER,
                    layout_bytes.len() as u64,
                    layout_bytes.as_ptr() as *const _,
                );
            }
            GpuCommand::Dispatch {
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
            GpuCommand::DispatchIndirect { buffer, offset } => {
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
            GpuCommand::Barrier => {
                if let Some(enc) = encoder {
                    // memoryBarrierWithScope: ensures all prior writes within
                    // this encoder are visible before subsequent dispatches.
                    // MTLBarrierScope: Buffers = 1, Textures = 2
                    let scope: mtl::NSUInteger = 1 | 2;
                    let () = unsafe { msg_send![enc, memoryBarrierWithScope: scope] };
                }
            }
            GpuCommand::ResourceBarrier {
                buffers: buf_handles,
                textures: tex_handles,
            } => {
                if let Some(enc) = encoder {
                    let mut resources: Vec<&mtl::ResourceRef> = Vec::new();
                    for handle in buf_handles {
                        if let Some(buf_state) = state.buffers.get(handle) {
                            let buf_ref: &mtl::BufferRef = &buf_state.buffer;
                            resources.push(unsafe {
                                std::mem::transmute::<&mtl::BufferRef, &mtl::ResourceRef>(buf_ref)
                            });
                        }
                    }
                    for handle in tex_handles {
                        if let Some(tex_state) = state.textures.get(handle) {
                            let tex_ref: &mtl::TextureRef = &tex_state.texture;
                            resources.push(unsafe {
                                std::mem::transmute::<&mtl::TextureRef, &mtl::ResourceRef>(tex_ref)
                            });
                        }
                    }
                    if !resources.is_empty() {
                        let count: mtl::NSUInteger = resources.len() as mtl::NSUInteger;
                        let ptr = resources.as_ptr();
                        let () =
                            unsafe { msg_send![enc, memoryBarrierWithResources: ptr count: count] };
                    }
                }
            }
            GpuCommand::ClearBuffer {
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
            GpuCommand::WriteBuffer {
                buffer: buf_handle,
                offset,
                data,
            } => {
                let buf_state = match state.buffers.get(buf_handle) {
                    Some(b) => b,
                    None => {
                        end_compute!();
                        anyhow::bail!("WriteBuffer: invalid buffer handle");
                    }
                };
                if data.is_empty() {
                    continue;
                }
                if *offset + data.len() as u64 > buf_state.size {
                    end_compute!();
                    anyhow::bail!("WriteBuffer: write exceeds buffer bounds");
                }
                // Same rationale as `buffer::write`: a CPU memcpy into `contents()` is not
                // ordered with other command buffers on the queue. Two back-to-back submits that
                // write the same buffer would record the second memcpy while the first submit's
                // GPU work is still reading that memory (shared storage), corrupting results.
                end_compute!();
                let staging = logical_device.device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    mtl::MTLResourceOptions::StorageModeShared,
                );
                let blit = command_buffer.new_blit_command_encoder();
                blit.copy_from_buffer(&staging, 0, &buf_state.buffer, *offset, data.len() as u64);
                blit.end_encoding();
            }
            GpuCommand::WriteTexture {
                texture: tex_handle,
                data,
                width,
                height,
            } => {
                let tex_state = match state.textures.get(tex_handle) {
                    Some(t) => t,
                    None => {
                        end_compute!();
                        anyhow::bail!("WriteTexture: invalid texture handle");
                    }
                };
                if *width != tex_state.width || *height != tex_state.height {
                    end_compute!();
                    anyhow::bail!("WriteTexture: dimension mismatch");
                }
                let bpp = tex_state.format.bytes_per_pixel();
                let expected = (*width as usize) * (*height as usize) * (bpp as usize);
                if data.len() != expected {
                    end_compute!();
                    anyhow::bail!(
                        "WriteTexture: expected {} bytes for {}x{}, got {}",
                        expected,
                        width,
                        height,
                        data.len()
                    );
                }
                if expected == 0 {
                    continue;
                }
                end_compute!();
                let staging = logical_device.device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    mtl::MTLResourceOptions::StorageModeShared,
                );
                let bytes_per_row = (*width as u64) * (bpp as u64);
                let blit = command_buffer.new_blit_command_encoder();
                blit.copy_from_buffer_to_texture(
                    &staging,
                    0,
                    bytes_per_row,
                    0,
                    MTLSize {
                        width: *width as u64,
                        height: *height as u64,
                        depth: 1,
                    },
                    &tex_state.texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    mtl::MTLBlitOption::empty(),
                );
                blit.end_encoding();
            }
            GpuCommand::WriteTextureRegion {
                texture: tex_handle,
                x,
                y,
                width,
                height,
                data,
            } => {
                let tex_state = match state.textures.get(tex_handle) {
                    Some(t) => t,
                    None => {
                        end_compute!();
                        anyhow::bail!("WriteTextureRegion: invalid texture handle");
                    }
                };
                if *x + *width > tex_state.width || *y + *height > tex_state.height {
                    end_compute!();
                    anyhow::bail!("WriteTextureRegion: region out of bounds");
                }
                let bpp = tex_state.format.bytes_per_pixel();
                let expected = (*width as usize) * (*height as usize) * (bpp as usize);
                if data.len() != expected {
                    end_compute!();
                    anyhow::bail!(
                        "WriteTextureRegion: expected {} bytes, got {}",
                        expected,
                        data.len()
                    );
                }
                if expected == 0 {
                    continue;
                }
                end_compute!();
                let staging = logical_device.device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    mtl::MTLResourceOptions::StorageModeShared,
                );
                let bytes_per_row = (*width as u64) * (bpp as u64);
                let blit = command_buffer.new_blit_command_encoder();
                blit.copy_from_buffer_to_texture(
                    &staging,
                    0,
                    bytes_per_row,
                    0,
                    MTLSize {
                        width: *width as u64,
                        height: *height as u64,
                        depth: 1,
                    },
                    &tex_state.texture,
                    0,
                    0,
                    MTLOrigin {
                        x: *x as u64,
                        y: *y as u64,
                        z: 0,
                    },
                    mtl::MTLBlitOption::empty(),
                );
                blit.end_encoding();
            }
        }
    }

    end_compute!();
    Ok(())
}

/// Submit compute commands without blocking. Returns the timeline value signaled when the work completes.
pub(super) fn submit(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    commands: &[GpuCommand],
) -> Result<TimelineValue> {
    if state.device_lost.load(Ordering::Relaxed) {
        anyhow::bail!("GPU device is lost (earlier wait timed out); refusing to submit new work");
    }

    let owned_command_buffer = {
        let ld = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        ld.command_queue.new_command_buffer().to_owned()
    };
    let command_buffer_ref = owned_command_buffer.as_ref();

    {
        let ld = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        record_commands_to_buffer(state, command_buffer_ref, ld, device_handle, commands)?;
    }

    let signal_value = {
        let ld = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;
        let v = ld.timeline_next;
        ld.timeline_next += 1;
        ld.timeline_scheduled_max = ld.timeline_scheduled_max.max(v);
        v
    };

    let waiter = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?
        .timeline_waiter
        .clone();

    let handler = block::ConcreteBlock::new(move |cb: &mtl::CommandBufferRef| {
        let status = cb.status();
        if status != MTLCommandBufferStatus::Completed {
            let description = read_command_buffer_error_description(cb);
            tracing::error!(
                "GPU command buffer (timeline={}) finished with status={:?}: {}",
                signal_value,
                status,
                description
            );
        }
        waiter.signal(signal_value);
    })
    .copy();
    command_buffer_ref.add_completed_handler(&handler);

    let ld = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    command_buffer_ref.encode_signal_event(ld.timeline_event.as_ref(), signal_value);

    command_buffer_ref.commit();

    if let Some(ld) = state.devices.get_mut(&device_handle) {
        ld.process_deletion_queue_up_to_signaled();
    }

    Ok(signal_value)
}

/// Extract `localizedDescription` from an `MTLCommandBuffer`'s `error` property.
/// Returns `"<none>"` when the buffer has no attached error, or a best-effort
/// diagnostic string when it does.
fn read_command_buffer_error_description(buf: &mtl::CommandBufferRef) -> String {
    use objc::runtime::Object;
    unsafe {
        let err: *mut Object = msg_send![buf, error];
        if err.is_null() {
            return "<none>".into();
        }
        let nsstr: *mut Object = msg_send![err, localizedDescription];
        if nsstr.is_null() {
            return "<error with no description>".into();
        }
        let utf8: *const std::os::raw::c_char = msg_send![nsstr, UTF8String];
        if utf8.is_null() {
            return "<error with null UTF8>".into();
        }
        std::ffi::CStr::from_ptr(utf8)
            .to_string_lossy()
            .into_owned()
    }
}
