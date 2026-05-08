//! Compute pipeline and dispatch logic.

use super::super::{ComputePipelineHandle, DeviceHandle, FenceToken, GpuCommand, ShaderHandle};
use super::shader::parse_numthreads;
use super::types::RESOURCE_SLOT_BUFFER;
use super::types::{
    ComputePipelineState, FenceEntry, FenceSignal, MetalState, PushLayout, MAX_BINDLESS_SLOTS,
    MAX_USER_SLOTS, TOTAL_PUSH_BYTES,
};
use crate::slang::SlangStage;
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{MTLCommandBufferStatus, MTLOrigin, MTLSize};
use objc::{msg_send, sel, sel_impl};
use std::sync::atomic::Ordering;
use std::sync::Arc;

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

/// Submit compute commands without blocking. Returns a fence token for polling/waiting.
pub(super) fn submit(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    commands: &[GpuCommand],
) -> Result<FenceToken> {
    if state.device_lost.load(Ordering::Relaxed) {
        anyhow::bail!("GPU device is lost (earlier wait timed out); refusing to submit new work");
    }

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

    // Completion handler responsibilities:
    //  1. Publish the terminal status into the `FenceSignal` so any thread
    //     waiting in `wait_fence` / `wait_fence_timeout` / `wait_all_in_flight`
    //     wakes up immediately instead of ticking a 1 ms poll clock. This is
    //     the pipeline win: a frame's GPU tail latency + 0 µs instead of
    //     tail + avg(0.5 ms) polling jitter.
    //  2. Log any non-`Completed` status so silent GPU faults on work that
    //     nobody waits on (e.g. `submit_graph` whose `GpuFuture` was dropped
    //     because a later `flush_graph` superseded it) still surface in the
    //     tracing stream rather than only manifesting as a hang on the next
    //     `wait_fence` call.
    let handler_token = token;
    let signal = Arc::new(FenceSignal::new());
    let signal_for_handler = signal.clone();
    let handler = block::ConcreteBlock::new(move |cb: &mtl::CommandBufferRef| {
        let status = cb.status();
        if status != MTLCommandBufferStatus::Completed {
            let description = read_command_buffer_error_description(cb);
            tracing::error!(
                "GPU command buffer (fence={}) finished with status={:?}: {}",
                handler_token,
                status,
                description
            );
        }
        let mut done = signal_for_handler.done.lock().unwrap();
        *done = Some(status);
        drop(done);
        signal_for_handler.cv.notify_all();
    })
    .copy();
    command_buffer_ref.add_completed_handler(&handler);

    state.compute_fence_pool.lock().unwrap().insert(
        token,
        FenceEntry {
            buffer: owned_buffer,
            signal,
        },
    );

    command_buffer_ref.commit();

    Ok(token)
}

/// Check if the fence for the given token has signaled (work complete).
pub(super) fn is_fence_complete(
    state: &MetalState,
    _device: DeviceHandle,
    token: FenceToken,
) -> bool {
    if state.device_lost.load(Ordering::Relaxed) {
        return true;
    }
    let pool = state.compute_fence_pool.lock().unwrap();
    if let Some(entry) = pool.get(&token) {
        // Prefer the completion-handler-published status so callers don't
        // need to reach into Metal. Fall back to the live `status()` read if
        // the handler has not fired yet.
        if let Some(status) = *entry.signal.done.lock().unwrap() {
            status == MTLCommandBufferStatus::Completed
        } else {
            entry.buffer.status() == MTLCommandBufferStatus::Completed
        }
    } else {
        true // Already removed (waited on), consider complete
    }
}

/// Block until the fence signals — with a hard timeout — then check for GPU errors.
///
/// We deliberately do **not** call `MTLCommandBuffer::waitUntilCompleted` here:
/// on Apple Silicon the GPU can wedge in a compute shader without the kernel
/// watchdog firing (observed as a 20s+ "Unresponsive" hang with
/// `GPU Restart Count` unchanged). In that state `waitUntilCompleted` blocks
/// forever and the whole app has to be force-quit.
///
/// Instead we block on a [`Condvar`] that the command buffer's completion
/// handler signals from the Metal dispatch queue. A hard timeout of
/// `GOLDY_GPU_WAIT_TIMEOUT_MS` (default 5000 ms) bounds the wait so a wedged
/// GPU — where the completion handler never fires — is still recoverable.
/// The 0.5 ms average jitter from the previous 1 ms poll loop is gone: wait
/// completion is now granular to the handler dispatch (microseconds).
pub(super) fn wait_fence(
    state: &MetalState,
    _device: DeviceHandle,
    token: FenceToken,
) -> Result<()> {
    if state.device_lost.load(Ordering::Relaxed) {
        anyhow::bail!("GPU device is lost (earlier wait timed out); refusing to wait on fence");
    }

    let entry = {
        let mut pool = state.compute_fence_pool.lock().unwrap();
        match pool.remove(&token) {
            Some(e) => e,
            None => return Ok(()),
        }
    };

    let timeout = std::time::Duration::from_millis(gpu_wait_timeout_ms());
    let warn_threshold = std::time::Duration::from_millis(500);
    let status = wait_signal(&entry.signal, warn_threshold, timeout, |elapsed, status| {
        tracing::warn!(
            "GPU wait_fence exceeding {}ms (status={:?}); still waiting up to {}ms",
            elapsed.as_millis(),
            status,
            timeout.as_millis()
        );
    });

    match status {
        Some(MTLCommandBufferStatus::Completed) => Ok(()),
        Some(MTLCommandBufferStatus::Error) => {
            let description = read_command_buffer_error_description(&entry.buffer);
            tracing::error!("Metal command buffer finished with error: {}", description);
            anyhow::bail!("Metal compute command buffer error: {}", description);
        }
        Some(other) => {
            // Completion handler fired with an unexpected non-terminal status.
            // Treat as a wedge — the handler contract says it only fires on
            // Completed or Error.
            state.device_lost.store(true, Ordering::Relaxed);
            anyhow::bail!(
                "Metal completion handler reported non-terminal status={:?}",
                other
            );
        }
        None => {
            // Timeout reached without the completion handler firing.
            state.device_lost.store(true, Ordering::Relaxed);
            let status = entry.buffer.status();
            tracing::error!(
                "GPU wait_fence timed out after {}ms without the completion \
                 handler firing (status={:?}). The GPU is likely wedged in a \
                 compute shader. Marking device as lost; all subsequent fence \
                 waits will fail fast so the app can exit cleanly instead of \
                 deadlocking.",
                timeout.as_millis(),
                status
            );
            // Letting `entry` go out of scope calls `release`, but the Metal
            // runtime retains the command buffer while the GPU is still
            // "running" it. Attempting another wait here would reproduce the
            // hang, so we bail.
            anyhow::bail!(
                "GPU wait_fence timed out after {}ms (status={:?}); GPU appears wedged",
                timeout.as_millis(),
                status
            );
        }
    }
}

/// Block on `signal.cv` until the completion handler publishes a terminal
/// status or `hard_timeout` elapses. Returns the published status, or `None`
/// on timeout.
///
/// If the wait exceeds `warn_after`, `on_warn` is invoked exactly once and
/// the wait continues with the remaining budget. This preserves the legacy
/// "GPU wait_fence exceeding 500ms" heartbeat without burning CPU on it.
fn wait_signal(
    signal: &FenceSignal,
    warn_after: std::time::Duration,
    hard_timeout: std::time::Duration,
    mut on_warn: impl FnMut(std::time::Duration, Option<MTLCommandBufferStatus>),
) -> Option<MTLCommandBufferStatus> {
    let start = std::time::Instant::now();
    let mut warned = false;
    let mut guard = signal.done.lock().unwrap();
    loop {
        if let Some(status) = *guard {
            return Some(status);
        }
        let elapsed = start.elapsed();
        let remaining = hard_timeout.saturating_sub(elapsed);
        if remaining.is_zero() {
            return None;
        }
        let wait_slice = if !warned {
            warn_after.saturating_sub(elapsed).min(remaining)
        } else {
            remaining
        };
        // `wait_slice` of zero means the warn threshold is reached and we've
        // already fallen through once — loop again to log and continue.
        if wait_slice.is_zero() && !warned {
            warned = true;
            on_warn(elapsed, *guard);
            continue;
        }
        let (g, result) = signal.cv.wait_timeout(guard, wait_slice).unwrap();
        guard = g;
        if !warned && result.timed_out() && start.elapsed() >= warn_after {
            warned = true;
            on_warn(start.elapsed(), *guard);
        }
    }
}

/// Read the timeout budget (in milliseconds) for a single `wait_fence` call.
///
/// Configurable via `GOLDY_GPU_WAIT_TIMEOUT_MS`. Defaults to 5000ms, which
/// is generous for a single compute frame (Apple's own GPU hang watchdog
/// fires around 2s) but short enough that a true wedge is recoverable.
fn gpu_wait_timeout_ms() -> u64 {
    std::env::var("GOLDY_GPU_WAIT_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000)
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

/// Wait with timeout. Returns Ok(true) if signaled, Ok(false) if timeout elapsed.
pub(super) fn wait_fence_timeout(
    state: &MetalState,
    _device: DeviceHandle,
    token: FenceToken,
    timeout_ms: u32,
) -> Result<bool> {
    if state.device_lost.load(Ordering::Relaxed) {
        anyhow::bail!("GPU device is lost (earlier wait timed out)");
    }
    let timeout = std::time::Duration::from_millis(timeout_ms as u64);
    // Snapshot the signal without removing the entry so a subsequent
    // wait_fence call can still observe the command buffer when the timeout
    // path returns `Ok(false)`.
    let signal = {
        let pool = state.compute_fence_pool.lock().unwrap();
        match pool.get(&token) {
            Some(entry) => entry.signal.clone(),
            None => return Ok(true), // Already waited on / removed.
        }
    };
    let status = wait_signal(&signal, timeout, timeout, |_, _| {});
    match status {
        Some(MTLCommandBufferStatus::Completed) | Some(MTLCommandBufferStatus::Error) => {
            state.compute_fence_pool.lock().unwrap().remove(&token);
            Ok(true)
        }
        Some(_) => Ok(true), // Non-terminal but handler fired; treat as signaled.
        None => Ok(false),
    }
}
