//! Compute pipeline and dispatch logic.

use super::super::shared;
use super::super::{ComputePipelineHandle, ContextHandle, DeviceHandle, GpuCommand, ShaderHandle};
use super::staging::TextureStagingEntry;
use super::types::RESOURCE_SLOT_BUFFER;
use super::types::{ComputePipelineState, MetalState, PushLayout};
use crate::slang::parse_numthreads;
use crate::slang::SlangStage;
use crate::tracy_zone;

/// Fallback workgroup size used when a compute shader's `[numthreads]` annotation
/// cannot be parsed. Matches the Metal/Slang default used elsewhere in the codebase.
const DEFAULT_WORKGROUP: [u32; 3] = [64, 1, 1];
use crate::timeline::TimelineValue;
use crate::types::{BufferKind, ResourceCategory};

fn buffer_stride_for_arg_index(state: &MetalState, index: u32, cat: ResourceCategory) -> Option<u32> {
    let expected_kind = match cat {
        ResourceCategory::Scattered => BufferKind::Scattered,
        ResourceCategory::Broadcast => BufferKind::Broadcast,
        _ => return None,
    };
    state
        .buffers
        .values()
        .find(|b| b.arg_buffer_index == index && b.access == expected_kind)
        .and_then(|b| b.element_stride)
}
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{MTLCommandBufferStatus, MTLOrigin, MTLSize};
use objc::{msg_send, sel, sel_impl};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Submit counter for `goldy::diag::mem` throttling.
static MEM_DIAG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Cadence for `goldy::diag::mem` snapshots. Reads `GOLDY_MEM_CADENCE` once, defaults to 60.
fn mem_diag_cadence() -> u64 {
    static CADENCE: OnceLock<u64> = OnceLock::new();
    *CADENCE.get_or_init(|| {
        std::env::var("GOLDY_MEM_CADENCE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60)
    })
}

fn maybe_log_mem_diag(ld: &super::types::LogicalDevice) {
    if tracing::enabled!(target: "goldy::diag::mem", tracing::Level::INFO) {
        let n = MEM_DIAG_COUNTER.fetch_add(1, Ordering::Relaxed);
        if n.is_multiple_of(mem_diag_cadence()) {
            let mib = ld.device.current_allocated_size() / (1024 * 1024);
            let ha = ld.heap_allocator.lock().unwrap();
            let heap_primary_mib = ha.primary_size() / (1024 * 1024);
            let heap_overflow = ha.overflow_count();
            let heap_hwm_mib = ha.high_water_mark() / (1024 * 1024);
            tracing::info!(
                target: "goldy::diag::mem",
                metal_current_allocated_mib = mib,
                heap_primary_mib,
                heap_overflow,
                heap_hwm_mib,
                "metal-alloc"
            );
        }
    }
}

/// Pre-scan a `GpuCommand` slice and return a compact submit summary for logging.
///
/// Collects the unique sequence of pipeline names (in order of first appearance)
/// and counts total dispatch calls. Used by `submit` / `submit_graph` when
/// `goldy::diag::submit` is enabled in `RUST_LOG`.
fn summarise_commands<'a>(commands: impl Iterator<Item = &'a super::super::GpuCommand>) -> (usize, Vec<&'static str>) {
    let mut dispatch_count = 0usize;
    let mut pipeline_names: Vec<&'static str> = Vec::new();
    let mut pending_label: Option<&'static str> = None;
    for cmd in commands {
        match cmd {
            super::super::GpuCommand::SetPipeline(_) => {
                // label is attached to the following Dispatch; reset pending
                pending_label = None;
            }
            super::super::GpuCommand::Dispatch { label, .. }
            | super::super::GpuCommand::DispatchIndirect { label, .. }
            | super::super::GpuCommand::DispatchBatch { label, .. } => {
                dispatch_count += match cmd {
                    super::super::GpuCommand::DispatchBatch { count, .. } => *count as usize,
                    _ => 1,
                };
                if let Some(name) = label.or(pending_label) {
                    if !pipeline_names.contains(&name) {
                        pipeline_names.push(name);
                    }
                }
                pending_label = None;
            }
            _ => {}
        }
    }
    (dispatch_count, pipeline_names)
}

/// Create a compute pipeline.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    compute_shader: ShaderHandle,
) -> Result<ComputePipelineHandle> {
    super::shader::ensure_stage_compiled(state, compute_shader, SlangStage::Compute)?;

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    let shader = state.shaders.get(&compute_shader).context("Invalid compute shader")?;

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
        .map(|r| (r.push_constant_categories.clone(), r.binding_element_strides.clone()))
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
/// Calls `useResources:count:usage:` (batch form) on all device-owned buffers and textures
/// so Metal's hazard tracking can detect cross-encoder dependencies (e.g. compute→blit→compute).
/// `use_heap` alone provides residency but NOT hazard tracking — without per-resource
/// declarations, Metal GPU Validation rejects dispatches that touch heap-resident
/// resources via argument buffers.
///
/// Using the batched form reduces Objective-C msg_send overhead from O(N) per encoder open
/// to O(1) regardless of how many resources the device owns.
pub(super) fn begin_compute_encoder<'a>(
    command_buffer: &'a mtl::CommandBufferRef,
    state: &MetalState,
    logical_device: &super::types::LogicalDevice,
    device_handle: DeviceHandle,
) -> &'a mtl::ComputeCommandEncoderRef {
    let encoder = command_buffer.new_compute_command_encoder();
    logical_device
        .heap_allocator
        .lock()
        .unwrap()
        .use_heaps_for_compute(encoder);
    logical_device
        .texture_heap
        .lock()
        .unwrap()
        .use_heaps_for_compute(encoder);

    // Collect resource refs by usage tier then call use_resources once per tier,
    // replacing one ObjC msg_send per resource with at most three total.
    // Safety: BufferRef/TextureRef are subclasses of Resource in the Metal ObjC
    // hierarchy, so transmuting the reference type is sound (same pointer, same layout).
    let mut rw_refs: Vec<&mtl::ResourceRef> = Vec::new();
    let mut ro_refs: Vec<&mtl::ResourceRef> = Vec::new();
    for buf_state in state.buffers.values() {
        if buf_state.device_handle == device_handle {
            let buf_ref: &mtl::BufferRef = &buf_state.buffer;
            rw_refs.push(unsafe { std::mem::transmute::<&mtl::BufferRef, &mtl::ResourceRef>(buf_ref) });
        }
    }
    for tex_state in state.textures.values() {
        if tex_state.device_handle == device_handle {
            let tex_ref: &mtl::TextureRef = &tex_state.texture;
            let res_ref = unsafe { std::mem::transmute::<&mtl::TextureRef, &mtl::ResourceRef>(tex_ref) };
            if tex_state.is_storage_image {
                rw_refs.push(res_ref);
            } else {
                ro_refs.push(res_ref);
            }
        }
    }
    if !rw_refs.is_empty() {
        encoder.use_resources(&rw_refs, mtl::MTLResourceUsage::Read | mtl::MTLResourceUsage::Write);
    }
    if !ro_refs.is_empty() {
        encoder.use_resources(&ro_refs, mtl::MTLResourceUsage::Read);
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
///
/// `belt_slices` and `texture_scratches` are pre-staged data produced by
/// [`stage_uploads`].  `belt_idx` and `tex_idx` are advanced in-place so
/// callers that invoke this function multiple times (e.g. `submit_graph`)
/// can share a single staging pre-pass across all compute batches.
///
/// `gpu_idle` must equal `last_committed_timeline.map(|l| signaled >= l).unwrap_or(true)`
/// as computed by the caller before the pre-pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn record_commands_to_buffer(
    state: &MetalState,
    command_buffer: &mtl::CommandBufferRef,
    logical_device: &super::types::LogicalDevice,
    device_handle: DeviceHandle,
    commands: &[GpuCommand],
    belt_slices: &[(mtl::Buffer, u64)],
    texture_scratches: &[TextureStagingEntry],
    belt_idx: &mut usize,
    tex_idx: &mut usize,
    gpu_idle: bool,
) -> Result<()> {
    let mut guard = EncoderGuard {
        compute: None,
        blit: None,
    };
    let mut current_pipeline: Option<&ComputePipelineState> = None;

    // Set to true once any GPU command (blit or compute) has been recorded
    // into the current command buffer. The WriteBuffer CPU memcpy fast path
    // must be skipped when this is true, because prior recorded commands
    // (e.g. ClearBuffer fill_buffer) haven't executed yet and would overwrite
    // the memcpy result when the command buffer is later committed.
    let mut has_recorded_gpu_work = false;

    // Tracks which buffer/texture handles have been touched in the current blit
    // encoder session. Metal does not guarantee ordering between two blit ops on
    // the same resource within one encoder (e.g. fill_buffer → copy_from_buffer on
    // the same buffer). We end+reopen the encoder only when the incoming command
    // targets a handle already touched in this session; distinct-handle blit ops
    // share the encoder and avoid the per-command encoder-open overhead.
    let mut blit_touched_bufs: Vec<super::BufferHandle> = Vec::new();
    let mut blit_touched_texs: Vec<super::TextureHandle> = Vec::new();

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
            blit_touched_bufs.clear();
            blit_touched_texs.clear();
            if guard.compute.is_none() {
                let enc = begin_compute_encoder(command_buffer, state, logical_device, device_handle);
                if let Some(pipeline) = current_pipeline {
                    enc.set_compute_pipeline_state(&pipeline.pipeline);
                }
                guard.compute = Some(enc);
            }
            has_recorded_gpu_work = true;
        };
    }

    /// Open a new blit encoder, clearing both touched-handle sets.
    macro_rules! open_blit {
        () => {
            end_compute!();
            end_blit!();
            guard.blit = Some(command_buffer.new_blit_command_encoder());
            blit_touched_bufs.clear();
            blit_touched_texs.clear();
            has_recorded_gpu_work = true;
        };
    }

    /// Ensure a blit encoder is open, splitting only if `$buf` (a BufferHandle)
    /// has already been written in the current encoder session.
    macro_rules! ensure_blit_buf {
        ($handle:expr) => {
            if guard.blit.is_none() || blit_touched_bufs.contains(&$handle) {
                open_blit!();
            }
            blit_touched_bufs.push($handle);
        };
    }

    /// Ensure a blit encoder is open, splitting only if `$tex` (a TextureHandle)
    /// has already been written in the current encoder session.
    macro_rules! ensure_blit_tex {
        ($handle:expr) => {
            if guard.blit.is_none() || blit_touched_texs.contains(&$handle) {
                open_blit!();
            }
            blit_touched_texs.push($handle);
        };
    }

    for cmd in commands {
        match cmd {
            GpuCommand::FrameTableStaging { .. } => {}
            GpuCommand::ClearBuffer { buffer, offset, size } => {
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
                    ensure_blit_buf!(*buffer);
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
                // Direct CPU memcpy is safe only when (a) no previously-committed GPU
                // work is still in flight AND (b) no GPU commands have been recorded
                // into the current command buffer. Condition (b) is critical: a prior
                // ClearBuffer fill_buffer is recorded but hasn't executed; a CPU memcpy
                // here would be overwritten when the command buffer commits.
                const SMALL_WRITE_THRESHOLD: usize = 4096;
                if gpu_idle
                    && !has_recorded_gpu_work
                    && !buf_state.flags.contains(crate::types::BufferFlags::GPU_ONLY)
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

                // Slow path: consume the pre-staged belt slice for this write.
                ensure_blit_buf!(*buf_handle);
                let (stg_buf, stg_off) = belt_slices
                    .get(*belt_idx)
                    .context("WriteBuffer: belt_slices index out of range (pre-pass mismatch)")?;
                *belt_idx += 1;
                guard
                    .blit
                    .unwrap()
                    .copy_from_buffer(stg_buf, *stg_off, &buf_state.buffer, *offset, data.len() as u64);
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
                // Consume the pre-staged texture entry for this upload.
                ensure_blit_tex!(*tex_handle);
                let scratch = texture_scratches
                    .get(*tex_idx)
                    .context("WriteTexture: texture_scratches index out of range")?;
                *tex_idx += 1;
                let bytes_per_row = (*width as u64) * (bpp as u64);
                guard.blit.unwrap().copy_from_buffer_to_texture(
                    &scratch.buffer,
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
                // Consume the pre-staged texture entry for this upload.
                ensure_blit_tex!(*tex_handle);
                let scratch = texture_scratches
                    .get(*tex_idx)
                    .context("WriteTextureRegion: texture_scratches index out of range")?;
                *tex_idx += 1;
                let bytes_per_row = (*width as u64) * (bpp as u64);
                guard.blit.unwrap().copy_from_buffer_to_texture(
                    &scratch.buffer,
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
            GpuCommand::BindResourcesRaw {
                indices: raw_indices,
                user: raw_user,
                frame_table_base: _,
            } => {
                ensure_compute!();
                if let Some(pipeline) = current_pipeline {
                    crate::backend::validate_raw_binding_strides(
                        raw_indices,
                        &pipeline.push_constant_categories,
                        &pipeline.binding_element_strides,
                        |idx, cat| buffer_stride_for_arg_index(state, idx, cat),
                        &pipeline.shader_debug_name,
                    )?;
                }
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
                label: _,
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
            GpuCommand::DispatchBatch {
                label: _,
                arg_data,
                count,
            } => {
                ensure_compute!();
                if let Some(pipeline) = current_pipeline {
                    let push_size = std::mem::size_of::<PushLayout>();
                    let stride = shared::DISPATCH_BATCH_STRIDE;
                    let entry_count = *count as usize;
                    let needed = entry_count
                        .checked_mul(stride)
                        .context("DispatchBatch: stride overflow")?;
                    anyhow::ensure!(
                        arg_data.len() >= needed,
                        "DispatchBatch: arg_data len {} < {} entries × stride {}",
                        arg_data.len(),
                        entry_count,
                        stride,
                    );
                    let threads_per_group = MTLSize {
                        width: pipeline.workgroup_size[0] as u64,
                        height: pipeline.workgroup_size[1] as u64,
                        depth: pipeline.workgroup_size[2] as u64,
                    };
                    let enc = guard.compute.expect("encoder must be set after ensure_compute!()");
                    for i in 0..entry_count {
                        let base = i * stride;
                        let layout_slice = &arg_data[base..base + push_size];
                        enc.set_bytes(
                            RESOURCE_SLOT_BUFFER,
                            layout_slice.len() as u64,
                            layout_slice.as_ptr() as *const _,
                        );
                        let wg_off = base + push_size;
                        let wg_x = u32::from_ne_bytes(arg_data[wg_off..wg_off + 4].try_into()?);
                        let wg_y = u32::from_ne_bytes(arg_data[wg_off + 4..wg_off + 8].try_into()?);
                        let wg_z = u32::from_ne_bytes(arg_data[wg_off + 8..wg_off + 12].try_into()?);
                        let threadgroups = MTLSize {
                            width: wg_x as u64,
                            height: wg_y as u64,
                            depth: wg_z as u64,
                        };
                        enc.dispatch_thread_groups(threadgroups, threads_per_group);
                    }
                }
            }
            GpuCommand::DispatchIndirect {
                label: _,
                buffer,
                offset,
            } => {
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
            GpuCommand::CopyTexture { src, dst } => {
                ensure_blit_tex!(*src);
                ensure_blit_tex!(*dst);
                let src_state = state.textures.get(src).context("CopyTexture: src texture not found")?;
                let dst_state = state.textures.get(dst).context("CopyTexture: dst texture not found")?;
                let w = src_state.width as u64;
                let h = src_state.height as u64;
                guard.blit.unwrap().copy_from_texture(
                    &src_state.texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize {
                        width: w,
                        height: h,
                        depth: 1,
                    },
                    &dst_state.texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            GpuCommand::CopyRenderTarget { src, dst } => {
                ensure_blit_tex!(*dst);
                let src_state = state
                    .render_targets
                    .get(src)
                    .context("CopyRenderTarget: src render target not found")?;
                let dst_state = state
                    .textures
                    .get(dst)
                    .context("CopyRenderTarget: dst texture not found")?;
                let w = src_state.width as u64;
                let h = src_state.height as u64;
                guard.blit.unwrap().copy_from_texture(
                    &src_state.texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize {
                        width: w,
                        height: h,
                        depth: 1,
                    },
                    &dst_state.texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            GpuCommand::Barrier => {
                if let Some(enc) = guard.compute {
                    const MTL_BARRIER_SCOPE_BUFFERS_AND_TEXTURES: mtl::NSUInteger = 1 | 2;
                    let () = unsafe { msg_send![enc, memoryBarrierWithScope: MTL_BARRIER_SCOPE_BUFFERS_AND_TEXTURES] };
                }
            }
            GpuCommand::ResourceBarrier {
                buffers: buf_entries,
                textures: tex_entries,
                ..
            } => {
                if let Some(enc) = guard.compute {
                    let mut resources: Vec<&mtl::ResourceRef> = Vec::new();
                    for (handle, _) in buf_entries {
                        if let Some(buf_state) = state.buffers.get(handle) {
                            let buf_ref: &mtl::BufferRef = &buf_state.buffer;
                            resources
                                .push(unsafe { std::mem::transmute::<&mtl::BufferRef, &mtl::ResourceRef>(buf_ref) });
                        }
                    }
                    for (handle, _) in tex_entries {
                        if let Some(tex_state) = state.textures.get(handle) {
                            let tex_ref: &mtl::TextureRef = &tex_state.texture;
                            resources
                                .push(unsafe { std::mem::transmute::<&mtl::TextureRef, &mtl::ResourceRef>(tex_ref) });
                        }
                    }
                    if !resources.is_empty() {
                        let count: mtl::NSUInteger = resources.len() as mtl::NSUInteger;
                        let ptr = resources.as_ptr();
                        let () = unsafe { msg_send![enc, memoryBarrierWithResources: ptr count: count] };
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

// ── Staging pre-pass ─────────────────────────────────────────────────────────

type StagedBufferUpload = (mtl::Buffer, u64);
type StagedUploads = (Vec<StagedBufferUpload>, Vec<TextureStagingEntry>, bool);

/// Reclaim completed staging resources and pre-stage all upload commands.
///
/// Returns `(belt_slices, texture_scratches, gpu_idle)`.
///
/// `belt_slices[i]` corresponds to the i-th non-fast-path `WriteBuffer` command
/// in `commands` (in source order).  `texture_scratches[i]` corresponds to the
/// i-th `WriteTexture` or `WriteTextureRegion` command in `commands`.
///
/// `gpu_idle` is `true` when no previously-committed GPU work is still in flight
/// (i.e. the GPU timeline has caught up to `last_committed_timeline`).  It is
/// forwarded to `record_commands_to_buffer` so the fast-path check there uses the
/// same value computed here — keeping the pre-pass and command loop in sync.
fn stage_uploads(
    state: &mut MetalState,
    ctx: ContextHandle,
    device_handle: super::super::DeviceHandle,
    commands: &[GpuCommand],
) -> Result<StagedUploads> {
    let has_upload = commands.iter().any(|c| {
        matches!(
            c,
            GpuCommand::WriteBuffer { .. } | GpuCommand::WriteTexture { .. } | GpuCommand::WriteTextureRegion { .. }
        )
    });

    let gpu_idle = state
        .contexts
        .get(&ctx)
        .map(|sc_arc| {
            let sc = sc_arc.lock().unwrap();
            sc.last_committed_timeline
                .map(|last| sc.timeline_event.as_ref().signaled_value() >= last)
                .unwrap_or(true)
        })
        .unwrap_or(true);

    if !has_upload {
        return Ok((Vec::new(), Vec::new(), gpu_idle));
    }

    // Reclaim: return completed in-flight resources to the free lists.
    {
        if let Some(sc_arc) = state.contexts.get(&ctx) {
            let mut sc = sc_arc.lock().unwrap();
            let completed = sc.timeline_event.as_ref().signaled_value();
            sc.staging_belt.reclaim(completed);
            sc.texture_staging_pool.reclaim(completed);
        }
    }

    // Pre-pass: stage data for every command that will need the slow path.
    //
    // We shadow `would_have_gpu_work` to mirror the `has_recorded_gpu_work`
    // flag in `record_commands_to_buffer` so the fast-path eligibility check
    // here is identical to the one in the command loop.
    let mut belt_slices: Vec<(mtl::Buffer, u64)> = Vec::new();
    let mut texture_scratches: Vec<TextureStagingEntry> = Vec::new();
    let mut would_have_gpu_work = false;

    // Cache the device pointer once to avoid repeated HashMap lookups.
    let device_mtl: mtl::Device = state
        .devices
        .get(&device_handle)
        .context("stage_uploads: invalid device handle")?
        .device
        .clone();

    const SMALL_WRITE_THRESHOLD: usize = 4096;

    for cmd in commands {
        match cmd {
            GpuCommand::WriteBuffer {
                buffer: buf_handle,
                data,
                ..
            } => {
                if data.is_empty() {
                    continue;
                }
                // Extract only what we need so the borrow of state.buffers ends
                // before we mutably borrow state.devices below.
                let (buf_flags, contents_null) = state
                    .buffers
                    .get(buf_handle)
                    .map(|b| (b.flags, b.buffer.contents().is_null()))
                    .unwrap_or((crate::types::BufferFlags::empty(), true));

                let fast_path = gpu_idle
                    && !would_have_gpu_work
                    && !buf_flags.contains(crate::types::BufferFlags::GPU_ONLY)
                    && data.len() <= SMALL_WRITE_THRESHOLD
                    && !contents_null;

                if fast_path {
                    // Fast path will do a direct CPU memcpy; no staging needed.
                    // The fast path does NOT open an encoder, so would_have_gpu_work stays as-is.
                } else {
                    let sc_arc = state
                        .contexts
                        .get(&ctx)
                        .context("stage_uploads: invalid context handle")?;
                    let (buf, off) = sc_arc.lock().unwrap().staging_belt.write(&device_mtl, data)?;
                    belt_slices.push((buf, off));
                    // Slow-path WriteBuffer opens a blit encoder.
                    would_have_gpu_work = true;
                }
            }
            GpuCommand::WriteTexture { data, .. } | GpuCommand::WriteTextureRegion { data, .. } => {
                if data.is_empty() {
                    continue;
                }
                let sc_arc = state
                    .contexts
                    .get(&ctx)
                    .context("stage_uploads: invalid context handle")?;
                let entry = sc_arc
                    .lock()
                    .unwrap()
                    .texture_staging_pool
                    .acquire(&device_mtl, data.len() as u64)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), entry.mapped_ptr(), data.len());
                }
                texture_scratches.push(entry);
                would_have_gpu_work = true;
            }
            // Commands that open an encoder set would_have_gpu_work.
            GpuCommand::ClearBuffer { .. }
            | GpuCommand::CopyTexture { .. }
            | GpuCommand::CopyRenderTarget { .. }
            | GpuCommand::SetPipeline(_)
            | GpuCommand::BindResourcesRaw { .. }
            | GpuCommand::BindResourcesTyped { .. }
            | GpuCommand::Dispatch { .. }
            | GpuCommand::DispatchBatch { .. }
            | GpuCommand::DispatchIndirect { .. } => {
                would_have_gpu_work = true;
            }
            GpuCommand::Barrier | GpuCommand::ResourceBarrier { .. } => {}
        }
    }

    Ok((belt_slices, texture_scratches, gpu_idle))
}

/// Submit compute commands without blocking. Returns the timeline value signaled when the work completes.
pub(super) fn submit(state: &mut MetalState, ctx: ContextHandle, commands: &[GpuCommand]) -> Result<TimelineValue> {
    let _tz = tracy_zone!("mtl.submit");
    if state.device_lost.load(Ordering::Relaxed) {
        anyhow::bail!("GPU device is lost (earlier wait timed out); refusing to submit new work");
    }

    if tracing::enabled!(target: "goldy::diag::submit", tracing::Level::INFO) {
        let (dispatch_count, pipeline_names) = summarise_commands(commands.iter());
        tracing::info!(
            target: "goldy::diag::submit",
            dispatch_count,
            ?pipeline_names,
            "gpu.submit kind=compute"
        );
    }

    let device_handle = super::context::context_device(state, ctx);

    let owned_command_buffer = {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        ld.command_queue.new_command_buffer().to_owned()
    };
    let command_buffer_ref = owned_command_buffer.as_ref();

    // Reclaim and pre-stage all uploads before recording.
    let (belt_slices, texture_scratches, gpu_idle) = stage_uploads(state, ctx, device_handle, commands)?;

    {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        let mut belt_idx = 0usize;
        let mut tex_idx = 0usize;
        record_commands_to_buffer(
            state,
            command_buffer_ref,
            ld,
            device_handle,
            commands,
            &belt_slices,
            &texture_scratches,
            &mut belt_idx,
            &mut tex_idx,
            gpu_idle,
        )?;
    }

    let signal_value = {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        let v = ld.timeline_next.fetch_add(1, Ordering::Relaxed);
        ld.timeline_scheduled_max.fetch_max(v, Ordering::Relaxed);
        v
    };

    let waiter = state
        .contexts
        .get(&ctx)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
        .timeline_waiter
        .clone();

    let compute_commit_instant = std::time::Instant::now();
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
        let cpu_lifetime = compute_commit_instant.elapsed();
        let (gpu_start, gpu_end): (f64, f64) = unsafe {
            (
                objc::msg_send![cb, GPUStartTime],
                objc::msg_send![cb, GPUEndTime],
            )
        };
        let gpu_ms = (gpu_end - gpu_start) * 1000.0;
        tracing::debug!(
            "[mtl.cb_done] kind=compute signal_value={signal_value} commit_to_complete={cpu_lifetime:?} gpu_exec={gpu_ms:.3}ms"
        );
        if crate::gpu_profiler::gpu_profile_enabled() {
            crate::gpu_profiler::log_cb_timing("metal", signal_value, gpu_ms);
        }
        waiter.signal(signal_value);
    })
    .copy();
    command_buffer_ref.add_completed_handler(&handler);

    let timeline_event = state
        .contexts
        .get(&ctx)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
        .timeline_event
        .clone();
    command_buffer_ref.encode_signal_event(timeline_event.as_ref(), signal_value);

    // Clone queue_lock before commit so we hold only the per-device queue lock
    // during the actual enqueue — not all of MetalState.
    let queue_lock = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?
        .queue_lock
        .clone();

    tracing::debug!(
        "[mtl.cb_commit] kind=compute signal_value={signal_value} queue=command_queue commands={n}",
        n = commands.len()
    );
    {
        let _queue_guard = queue_lock.lock().unwrap();
        command_buffer_ref.commit();
    }

    // Post-submit: tag in-flight staging resources with the timeline signal value.
    if let Some(sc_arc) = state.contexts.get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        sc.staging_belt.finish(signal_value);
        sc.texture_staging_pool.release(signal_value, texture_scratches);
        sc.last_committed_timeline = Some(signal_value);
        sc.in_flight_command_buffers
            .push_back((signal_value, owned_command_buffer));
        sc.last_submitted_seq = signal_value;
    }
    // Drain per-context deletion queue on the context's own clock (hot path),
    // then the device-level queue as the async GC safety net (see issue #190).
    if let Some(sc_arc) = state.contexts.get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        let ctx_signaled = sc.timeline_event.as_ref().signaled_value();
        sc.deletion_queue.process_up_to(ctx_signaled);
    }
    {
        let retired = super::context::device_retired(state, device_handle);
        if let Some(ld) = state.devices.get(&device_handle) {
            ld.process_deletion_queue_up_to(retired);
            maybe_log_mem_diag(ld);
        }
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
    ctx: ContextHandle,
    commands: &[super::super::GraphCommand],
) -> Result<TimelineValue> {
    let _tz = tracy_zone!("mtl.submit_graph");
    use super::super::GraphCommand;

    if state.device_lost.load(Ordering::Relaxed) {
        anyhow::bail!("GPU device is lost (earlier wait timed out); refusing to submit new work");
    }

    if tracing::enabled!(target: "goldy::diag::submit", tracing::Level::INFO) {
        let gpu_cmds = commands.iter().filter_map(|c| {
            if let GraphCommand::Compute(gc) = c {
                Some(gc)
            } else {
                None
            }
        });
        let (dispatch_count, pipeline_names) = summarise_commands(gpu_cmds);
        let render_passes = commands
            .iter()
            .filter(|c| matches!(c, GraphCommand::Render { .. }))
            .count();
        tracing::info!(
            target: "goldy::diag::submit",
            dispatch_count,
            render_passes,
            ?pipeline_names,
            "gpu.submit kind=graph"
        );
    }

    let device_handle = super::context::context_device(state, ctx);

    let owned_command_buffer = {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        ld.command_queue.new_command_buffer().to_owned()
    };
    let command_buffer_ref = owned_command_buffer.as_ref();

    // Pre-pass: collect all compute commands across the entire graph into a flat
    // list, run the staging pre-pass once, then replay the graph using shared
    // belt/tex indices that advance across compute batches.
    let all_compute_cmds: Vec<GpuCommand> = commands
        .iter()
        .filter_map(|c| {
            if let GraphCommand::Compute(gpu_cmd) = c {
                Some(gpu_cmd.clone())
            } else {
                None
            }
        })
        .collect();

    let (belt_slices, texture_scratches, gpu_idle) = stage_uploads(state, ctx, device_handle, &all_compute_cmds)?;

    // Walk GraphCommands, collecting contiguous compute batches and recording
    // render passes inline. Encoder transitions within a single command buffer
    // provide implicit full pipeline barriers on Metal.
    //
    // belt_idx and tex_idx advance across all compute-batch calls to
    // record_commands_to_buffer so they consume the single pre-pass result.
    {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;

        let mut compute_batch: Vec<GpuCommand> = Vec::new();
        let mut belt_idx = 0usize;
        let mut tex_idx = 0usize;

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
                            &belt_slices,
                            &texture_scratches,
                            &mut belt_idx,
                            &mut tex_idx,
                            gpu_idle,
                        )?;
                        compute_batch.clear();
                    }

                    // Record the render pass into the same command buffer.
                    record_render_pass_to_buffer(state, command_buffer_ref, ld, device_handle, *target, render_cmds)?;
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
                &belt_slices,
                &texture_scratches,
                &mut belt_idx,
                &mut tex_idx,
                gpu_idle,
            )?;
        }
    }

    // Signal timeline and commit — same pattern as `submit`.
    let signal_value = {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        let v = ld.timeline_next.fetch_add(1, Ordering::Relaxed);
        ld.timeline_scheduled_max.fetch_max(v, Ordering::Relaxed);
        v
    };

    let waiter = state
        .contexts
        .get(&ctx)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
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
        // Capture per-command-buffer GPU timestamps. `GPUStartTime` / `GPUEndTime`
        // are only valid after completion, which is guaranteed here.
        if crate::gpu_profiler::gpu_profile_enabled() {
            let gpu_start: f64 = unsafe { objc::msg_send![cb, GPUStartTime] };
            let gpu_end: f64 = unsafe { objc::msg_send![cb, GPUEndTime] };
            let ms = (gpu_end - gpu_start) * 1000.0;
            crate::gpu_profiler::log_cb_timing("metal", signal_value, ms);
        }
        waiter.signal(signal_value);
    })
    .copy();
    command_buffer_ref.add_completed_handler(&handler);

    let timeline_event = state
        .contexts
        .get(&ctx)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
        .timeline_event
        .clone();
    command_buffer_ref.encode_signal_event(timeline_event.as_ref(), signal_value);

    // Clone queue_lock before commit so we hold only the per-device queue lock
    // during the actual enqueue — not all of MetalState.
    let queue_lock = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?
        .queue_lock
        .clone();
    {
        let _queue_guard = queue_lock.lock().unwrap();
        command_buffer_ref.commit();
    }

    // Post-submit: tag in-flight staging resources with the timeline signal value.
    if let Some(sc_arc) = state.contexts.get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        sc.staging_belt.finish(signal_value);
        sc.texture_staging_pool.release(signal_value, texture_scratches);
        sc.last_committed_timeline = Some(signal_value);
        sc.in_flight_command_buffers
            .push_back((signal_value, owned_command_buffer));
        sc.last_submitted_seq = signal_value;
    }
    // Drain per-context deletion queue on the context's own clock (hot path),
    // then the device-level queue as the async GC safety net (see issue #190).
    if let Some(sc_arc) = state.contexts.get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        let ctx_signaled = sc.timeline_event.as_ref().signaled_value();
        sc.deletion_queue.process_up_to(ctx_signaled);
    }
    {
        let retired = super::context::device_retired(state, device_handle);
        if let Some(ld) = state.devices.get(&device_handle) {
            ld.process_deletion_queue_up_to(retired);
            maybe_log_mem_diag(ld);
        }
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
    let render_target = state.render_targets.get(&target).context("Invalid render target")?;

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
        .lock()
        .unwrap()
        .use_heaps_for_render(encoder, render_stages);
    logical_device
        .texture_heap
        .lock()
        .unwrap()
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
        std::ffi::CStr::from_ptr(utf8).to_string_lossy().into_owned()
    }
}
