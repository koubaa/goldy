//! Compute pipeline and dispatch logic.

use super::super::shared;
use super::super::{ComputePipelineHandle, DeviceHandle, GpuCommand, ShaderHandle};
use super::types::RESOURCE_SLOT_BUFFER;
use super::types::{ComputePipelineState, MetalState, PushLayout};
use crate::slang::parse_numthreads;
use crate::slang::SlangStage;

/// Fallback workgroup size used when a compute shader's `[numthreads]` annotation
/// cannot be parsed. Matches the Metal/Slang default used elsewhere in the codebase.
const DEFAULT_WORKGROUP: [u32; 3] = [64, 1, 1];
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
    super::shader::ensure_stage_compiled(state, compute_shader, SlangStage::Compute)?;

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let shader = state
        .shaders
        .get(&compute_shader)
        .context("Invalid compute shader")?;

    let workgroup_size = parse_numthreads(&shader.slang_source).unwrap_or_else(|| {
        tracing::warn!(
            "Could not parse [numthreads] annotation for compute shader {}; \
             using default workgroup {:?}",
            compute_shader,
            DEFAULT_WORKGROUP
        );
        DEFAULT_WORKGROUP
    });

    let library = shader
        .compute_library
        .as_ref()
        .expect("compute library must be compiled before pipeline creation");

    let function = library
        .get_function("cs_main", None)
        .map_err(|e| anyhow::anyhow!("Failed to get compute function: {}", e))?;

    let pipeline = logical_device
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("Failed to create compute pipeline: {}", e))?;

    let handle = state.next_compute_pipeline_handle;
    state.next_compute_pipeline_handle += 1;

    let (cats, strides) = state
        .shaders
        .get(&compute_shader)
        .and_then(|s| s.reflection.as_ref())
        .map(|r| {
            (
                r.push_constant_categories.clone(),
                r.binding_element_strides.clone(),
            )
        })
        .unwrap_or_default();

    let shader_debug_name = format!("compute_shader#{compute_shader}");

    state.compute_pipelines.insert(
        handle,
        ComputePipelineState {
            device_handle,
            pipeline,
            workgroup_size,
            push_constant_categories: cats,
            binding_element_strides: strides,
            shader_debug_name,
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
/// Calls `use_resource` on every buffer and texture so Metal's hazard tracking
/// can detect cross-encoder dependencies (e.g. compute→blit→compute). `use_heap`
/// alone provides residency but NOT hazard tracking — without per-resource
/// declarations, Metal GPU Validation rejects dispatches that touch heap-resident
/// resources via argument buffers.
pub(super) fn begin_compute_encoder<'a>(
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

/// Guard that ensures Metal command encoders are ended on all exit paths
/// (including early `?` returns). Metal asserts at dealloc time if an encoder
/// was never ended, so this prevents crashes on error paths.
pub(super) struct EncoderGuard<'a> {
    pub(super) compute: Option<&'a mtl::ComputeCommandEncoderRef>,
    pub(super) blit: Option<&'a mtl::BlitCommandEncoderRef>,
}

impl Drop for EncoderGuard<'_> {
    fn drop(&mut self) {
        if let Some(enc) = self.blit.take() {
            enc.end_encoding();
        }
        if let Some(enc) = self.compute.take() {
            enc.end_encoding();
        }
    }
}

/// Record compute commands to a command buffer (shared by submit and dispatch).
///
/// Uses a **single-pass** structure that lazily transitions between blit and
/// compute encoders as needed. Consecutive blit commands (clears, uploads)
/// share one blit encoder; consecutive compute commands share one compute
/// encoder. When a command of the other type is encountered, the active
/// encoder is ended before the new one begins. Metal guarantees sequential
/// execution of encoders within a command buffer, so this preserves the
/// caller's intended ordering (e.g. dispatch → clear → dispatch).
pub(super) fn record_commands_to_buffer(
    state: &MetalState,
    command_buffer: &mtl::CommandBufferRef,
    logical_device: &super::types::LogicalDevice,
    device_handle: DeviceHandle,
    commands: &[GpuCommand],
) -> Result<()> {
    let mut guard = EncoderGuard {
        compute: None,
        blit: None,
    };
    let mut current_pipeline: Option<&ComputePipelineState> = None;

    macro_rules! end_compute {
        () => {
            if let Some(enc) = guard.compute.take() {
                enc.end_encoding();
            }
        };
    }

    macro_rules! end_blit {
        () => {
            if let Some(enc) = guard.blit.take() {
                enc.end_encoding();
            }
        };
    }

    macro_rules! ensure_compute {
        () => {
            end_blit!();
            if guard.compute.is_none() {
                let enc =
                    begin_compute_encoder(command_buffer, state, logical_device, device_handle);
                if let Some(pipeline) = current_pipeline {
                    enc.set_compute_pipeline_state(&pipeline.pipeline);
                }
                guard.compute = Some(enc);
            }
        };
    }

    macro_rules! ensure_blit {
        () => {
            end_compute!();
            // End any active blit encoder before opening a new one. Metal does not
            // guarantee ordering between commands within the same blit encoder (e.g.
            // fill_buffer and copy_from_buffer targeting the same buffer may execute
            // in any order). Ending and reopening creates a new encoder boundary which
            // Metal serializes, so each ClearBuffer/WriteBuffer command executes in
            // strictly program order relative to every other blit command.
            end_blit!();
            guard.blit = Some(command_buffer.new_blit_command_encoder());
        };
    }

    for cmd in commands {
        match cmd {
            GpuCommand::ClearBuffer {
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
                    ensure_blit!();
                    let range = mtl::NSRange::new(*offset, clear_size);
                    guard.blit.unwrap().fill_buffer(&buf_state.buffer, range, 0);
                }
            }
            GpuCommand::WriteBuffer {
                buffer: buf_handle,
                offset,
                data,
            } => {
                let buf_state = state
                    .buffers
                    .get(buf_handle)
                    .context("WriteBuffer: invalid buffer handle")?;
                if data.is_empty() {
                    continue;
                }
                anyhow::ensure!(
                    *offset + data.len() as u64 <= buf_state.size,
                    "WriteBuffer: write exceeds buffer bounds"
                );
                // Direct CPU memcpy is safe only when no previously-committed GPU
                // work is still in flight: if the GPU has already signaled past the
                // last committed timeline, all reads from this buffer are complete.
                // Otherwise we fall through to the staged blit path to avoid a race.
                const SMALL_WRITE_THRESHOLD: usize = 4096;
                let gpu_idle = logical_device
                    .last_committed_timeline
                    .map(|last| logical_device.timeline_event.as_ref().signaled_value() >= last)
                    .unwrap_or(true);
                if gpu_idle
                    && !buf_state
                        .flags
                        .contains(crate::types::BufferFlags::GPU_ONLY)
                    && data.len() <= SMALL_WRITE_THRESHOLD
                {
                    let ptr = buf_state.buffer.contents();
                    if !ptr.is_null() {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr(),
                                (ptr as *mut u8).add(*offset as usize),
                                data.len(),
                            );
                        }
                        continue;
                    }
                }

                ensure_blit!();
                let staging = logical_device.device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    mtl::MTLResourceOptions::StorageModeShared,
                );
                guard.blit.unwrap().copy_from_buffer(
                    &staging,
                    0,
                    &buf_state.buffer,
                    *offset,
                    data.len() as u64,
                );
            }
            GpuCommand::WriteTexture {
                texture: tex_handle,
                data,
                width,
                height,
            } => {
                let tex_state = state
                    .textures
                    .get(tex_handle)
                    .context("WriteTexture: invalid texture handle")?;
                anyhow::ensure!(
                    *width == tex_state.width && *height == tex_state.height,
                    "WriteTexture: dimension mismatch"
                );
                let bpp = tex_state.format.bytes_per_pixel();
                let expected = (*width as usize) * (*height as usize) * (bpp as usize);
                anyhow::ensure!(
                    data.len() == expected,
                    "WriteTexture: expected {} bytes for {}x{}, got {}",
                    expected,
                    width,
                    height,
                    data.len()
                );
                if expected == 0 {
                    continue;
                }
                ensure_blit!();
                let staging = logical_device.device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    mtl::MTLResourceOptions::StorageModeShared,
                );
                let bytes_per_row = (*width as u64) * (bpp as u64);
                guard.blit.unwrap().copy_from_buffer_to_texture(
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
            }
            GpuCommand::WriteTextureRegion {
                texture: tex_handle,
                x,
                y,
                width,
                height,
                data,
            } => {
                let tex_state = state
                    .textures
                    .get(tex_handle)
                    .context("WriteTextureRegion: invalid texture handle")?;
                anyhow::ensure!(
                    *x + *width <= tex_state.width && *y + *height <= tex_state.height,
                    "WriteTextureRegion: region out of bounds"
                );
                let bpp = tex_state.format.bytes_per_pixel();
                let expected = (*width as usize) * (*height as usize) * (bpp as usize);
                anyhow::ensure!(
                    data.len() == expected,
                    "WriteTextureRegion: expected {} bytes, got {}",
                    expected,
                    data.len()
                );
                if expected == 0 {
                    continue;
                }
                ensure_blit!();
                let staging = logical_device.device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    mtl::MTLResourceOptions::StorageModeShared,
                );
                let bytes_per_row = (*width as u64) * (bpp as u64);
                guard.blit.unwrap().copy_from_buffer_to_texture(
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
            }
            GpuCommand::SetPipeline(handle) => {
                ensure_compute!();
                if let Some(pipeline) = state.compute_pipelines.get(handle) {
                    guard
                        .compute
                        .expect("encoder must be set after ensure_compute!()")
                        .set_compute_pipeline_state(&pipeline.pipeline);
                    current_pipeline = Some(pipeline);
                }
            }
            GpuCommand::BindResources { buffers } => {
                ensure_compute!();
                if crate::slang::layout_validation_enabled() {
                    if let Some(pipeline) = current_pipeline {
                        if !pipeline.binding_element_strides.is_empty() {
                            let actual: Vec<Option<u32>> = buffers
                                .iter()
                                .map(|h| state.buffers.get(h).and_then(|b| b.element_stride))
                                .collect();
                            crate::backend::validate_binding_strides(
                                &actual,
                                &pipeline.binding_element_strides,
                                &pipeline.shader_debug_name,
                            )?;
                        }
                    }
                }
                let mut layout = PushLayout::default();
                shared::fill_bindless(
                    &mut layout,
                    buffers.iter().map(|h| {
                        state
                            .buffers
                            .get(h)
                            .map(|b| b.arg_buffer_index)
                            .unwrap_or(0)
                    }),
                );
                let layout_bytes = layout.as_bytes();
                guard
                    .compute
                    .expect("encoder must be set after ensure_compute!()")
                    .set_bytes(
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
                shared::fill_raw(&mut layout, raw_indices, raw_user);
                let layout_bytes = layout.as_bytes();
                guard
                    .compute
                    .expect("encoder must be set after ensure_compute!()")
                    .set_bytes(
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
                shared::fill_typed(&mut layout, handles.iter().copied());
                let layout_bytes = layout.as_bytes();
                guard
                    .compute
                    .expect("encoder must be set after ensure_compute!()")
                    .set_bytes(
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
                    guard
                        .compute
                        .expect("encoder must be set after ensure_compute!()")
                        .dispatch_thread_groups(threadgroups, threads_per_group);
                }
            }
            GpuCommand::DispatchIndirect { buffer, offset } => {
                ensure_compute!();
                let buf_state = state
                    .buffers
                    .get(buffer)
                    .context("DispatchIndirect: invalid buffer handle")?;
                let pipeline = current_pipeline.context("DispatchIndirect: no pipeline bound")?;
                let threads_per_group = MTLSize {
                    width: pipeline.workgroup_size[0] as u64,
                    height: pipeline.workgroup_size[1] as u64,
                    depth: pipeline.workgroup_size[2] as u64,
                };
                guard
                    .compute
                    .expect("encoder must be set after ensure_compute!()")
                    .dispatch_thread_groups_indirect(&buf_state.buffer, *offset, threads_per_group);
            }
            GpuCommand::Barrier => {
                if let Some(enc) = guard.compute {
                    const MTL_BARRIER_SCOPE_BUFFERS_AND_TEXTURES: mtl::NSUInteger = 1 | 2;
                    let () = unsafe {
                        msg_send![enc, memoryBarrierWithScope: MTL_BARRIER_SCOPE_BUFFERS_AND_TEXTURES]
                    };
                }
            }
            GpuCommand::ResourceBarrier {
                buffers: buf_handles,
                textures: tex_handles,
            } => {
                if let Some(enc) = guard.compute {
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
        }
    }

    // Explicit cleanup (guard's Drop also handles early-return paths).
    end_blit!();
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
        ld.last_committed_timeline = Some(signal_value);
        ld.process_deletion_queue_up_to_signaled();
    }

    Ok(signal_value)
}

/// Submit a mixed compute + render graph in a single command buffer.
///
/// Unlike the default `submit_graph` which does CPU waits between compute
/// batches and render passes, this records everything into one `MTLCommandBuffer`
/// by switching between compute, blit, and render encoders. Metal guarantees
/// sequential execution of encoders within a command buffer, so GPU ordering
/// is preserved without any CPU-side synchronization.
pub(super) fn submit_graph(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    commands: &[super::super::GraphCommand],
) -> Result<TimelineValue> {
    use super::super::GraphCommand;

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

    // Walk GraphCommands, collecting contiguous compute batches and recording
    // render passes inline. Encoder transitions within a single command buffer
    // provide implicit full pipeline barriers on Metal.
    {
        let ld = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let mut compute_batch: Vec<GpuCommand> = Vec::new();

        for cmd in commands {
            match cmd {
                GraphCommand::Compute(c) => {
                    compute_batch.push(c.clone());
                }
                GraphCommand::Render {
                    target,
                    commands: render_cmds,
                } => {
                    // Flush any pending compute work first.
                    if !compute_batch.is_empty() {
                        record_commands_to_buffer(
                            state,
                            command_buffer_ref,
                            ld,
                            device_handle,
                            &compute_batch,
                        )?;
                        compute_batch.clear();
                    }

                    // Record the render pass into the same command buffer.
                    record_render_pass_to_buffer(
                        state,
                        command_buffer_ref,
                        ld,
                        device_handle,
                        *target,
                        render_cmds,
                    )?;
                }
            }
        }

        // Flush trailing compute work.
        if !compute_batch.is_empty() {
            record_commands_to_buffer(
                state,
                command_buffer_ref,
                ld,
                device_handle,
                &compute_batch,
            )?;
        }
    }

    // Signal timeline and commit — same pattern as `submit`.
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
                "GPU command buffer (graph, timeline={}) finished with status={:?}: {}",
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
        ld.last_committed_timeline = Some(signal_value);
        ld.process_deletion_queue_up_to_signaled();
    }

    // Mark render targets as rendered.
    for cmd in commands {
        if let GraphCommand::Render { target, .. } = cmd {
            if let Some(rt) = state.render_targets.get_mut(target) {
                rt.has_rendered = true;
            }
        }
    }

    Ok(signal_value)
}

/// Record an offscreen render pass into an existing command buffer (no commit/wait).
fn record_render_pass_to_buffer(
    state: &MetalState,
    command_buffer: &mtl::CommandBufferRef,
    logical_device: &super::types::LogicalDevice,
    device_handle: DeviceHandle,
    target: super::super::RenderTargetHandle,
    commands: &[super::super::RenderCommand],
) -> Result<()> {
    let render_target = state
        .render_targets
        .get(&target)
        .context("Invalid render target")?;

    let mut clear_color = None;
    let mut clear_depth = None;
    for cmd in commands {
        match cmd {
            super::super::RenderCommand::Clear(color) => clear_color = Some(*color),
            super::super::RenderCommand::ClearDepth(depth) => clear_depth = Some(*depth),
            _ => {}
        }
    }

    let render_pass = super::render_commands::create_render_pass(
        &render_target.texture,
        render_target.depth_texture.as_deref(),
        clear_color,
        clear_depth,
    );

    let encoder = command_buffer.new_render_command_encoder(render_pass);

    let render_stages = mtl::MTLRenderStages::Vertex | mtl::MTLRenderStages::Fragment;
    logical_device
        .heap_allocator
        .use_heaps_for_render(encoder, render_stages);
    logical_device
        .texture_heap
        .use_heaps_for_render(encoder, render_stages);
    for buf_state in state.buffers.values() {
        if buf_state.device_handle == device_handle {
            encoder.use_resource_at(
                &buf_state.buffer,
                mtl::MTLResourceUsage::Read | mtl::MTLResourceUsage::Write,
                render_stages,
            );
        }
    }

    encoder.set_vertex_buffer(0, Some(&logical_device.argument_buffer), 0);
    encoder.set_fragment_buffer(0, Some(&logical_device.argument_buffer), 0);

    encoder.set_viewport(mtl::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: render_target.width as f64,
        height: render_target.height as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    encoder.set_scissor_rect(mtl::MTLScissorRect {
        x: 0,
        y: 0,
        width: render_target.width as u64,
        height: render_target.height as u64,
    });

    super::render_commands::record(encoder, commands, &state.pipelines, &state.buffers)?;

    encoder.end_encoding();
    Ok(())
}

/// Extract `localizedDescription` from an `MTLCommandBuffer`'s `error` property.
/// Returns `"<none>"` when the buffer has no attached error, or a best-effort
/// diagnostic string when it does.
pub(super) fn read_command_buffer_error_description(buf: &mtl::CommandBufferRef) -> String {
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
