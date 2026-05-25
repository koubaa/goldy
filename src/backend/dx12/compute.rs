//! Compute pipeline and dispatch logic.

use super::super::shared;
use super::barriers;
use super::pso_cache;
use super::shader;
use super::staging;
use super::types::{self, ComputeAllocatorSlot, ComputePipelineState, Dx12State};
use super::{ComputePipelineHandle, DeviceHandle, RenderTargetHandle, ShaderHandle};
use crate::backend::{GpuCommand, GraphCommand};
use crate::timeline::TimelineValue;
use crate::tracy_zone;
use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use crate::task_graph::{NodeAccessUnion, SlotUsageSet, UsageKindFlags};

/// Convert the Koubaa-level producer/consumer kind set to a DX12 sync scope.
///
/// Returns `D3D12_BARRIER_SYNC_ALL` if no kinds are recorded (empty set) so
/// that the barrier is conservatively correct even without IR information.
fn slot_usage_to_dx12_sync(usage: &SlotUsageSet) -> D3D12_BARRIER_SYNC {
    if usage.kinds.is_empty() {
        return D3D12_BARRIER_SYNC_ALL;
    }
    let mut sync = D3D12_BARRIER_SYNC(0);
    if usage.kinds.contains(UsageKindFlags::COMPUTE) {
        sync.0 |= D3D12_BARRIER_SYNC_COMPUTE_SHADING.0;
    }
    if usage.kinds.contains(UsageKindFlags::TRANSFER) {
        sync.0 |= D3D12_BARRIER_SYNC_COPY.0;
    }
    if usage.kinds.contains(UsageKindFlags::RENDER) {
        sync.0 |= D3D12_BARRIER_SYNC_RENDER_TARGET.0 | D3D12_BARRIER_SYNC_DEPTH_STENCIL.0;
    }
    sync
}

/// Convert the Koubaa-level producer/consumer access set to a DX12 access mask.
///
/// Returns `D3D12_BARRIER_ACCESS_COMMON` if no kinds are recorded.
///
/// For compute: always includes both UAV and SRV because the `SlotUsageSet`
/// cannot distinguish whether the shader binds a buffer via UAV or SRV
/// descriptor.  A mismatch (e.g. `AccessAfter = SRV` when the shader reads
/// via UAV) causes the driver to use the wrong cache coherence protocol,
/// resulting in implicit full stalls on hardware.
fn slot_usage_to_dx12_access(usage: &SlotUsageSet) -> D3D12_BARRIER_ACCESS {
    if usage.kinds.is_empty() {
        return D3D12_BARRIER_ACCESS_COMMON;
    }
    let mut access = D3D12_BARRIER_ACCESS(0);
    if usage.kinds.contains(UsageKindFlags::COMPUTE) {
        access.0 |=
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0;
    }
    if usage.kinds.contains(UsageKindFlags::TRANSFER) {
        if usage.access == NodeAccessUnion::Write {
            access.0 |= D3D12_BARRIER_ACCESS_COPY_DEST.0;
        } else {
            access.0 |= D3D12_BARRIER_ACCESS_COPY_SOURCE.0;
        }
    }
    if usage.kinds.contains(UsageKindFlags::RENDER) {
        access.0 |=
            D3D12_BARRIER_ACCESS_RENDER_TARGET.0 | D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE.0;
    }
    if access.0 == 0 {
        D3D12_BARRIER_ACCESS_COMMON
    } else {
        access
    }
}

fn texture_barrier_state_for_layout(
    layout: D3D12_BARRIER_LAYOUT,
) -> (
    D3D12_BARRIER_SYNC,
    D3D12_BARRIER_ACCESS,
    D3D12_BARRIER_LAYOUT,
) {
    if layout == D3D12_BARRIER_LAYOUT_COPY_SOURCE {
        (
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_ACCESS_COPY_SOURCE,
            layout,
        )
    } else if layout == D3D12_BARRIER_LAYOUT_COPY_DEST {
        (
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_ACCESS_COPY_DEST,
            layout,
        )
    } else if layout == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS
        || layout == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS
    {
        (
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS,
        )
    } else if layout == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE
        || layout == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE
    {
        (
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
            D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE,
        )
    } else {
        (D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON, layout)
    }
}

fn texture_barrier_state_for_usage(
    usage: &SlotUsageSet,
    is_storage: bool,
) -> (
    D3D12_BARRIER_SYNC,
    D3D12_BARRIER_ACCESS,
    D3D12_BARRIER_LAYOUT,
) {
    if usage.kinds.contains(UsageKindFlags::TRANSFER) {
        if usage.access == NodeAccessUnion::Write {
            (
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_LAYOUT_COPY_DEST,
            )
        } else {
            (
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_LAYOUT_COPY_SOURCE,
            )
        }
    } else if usage.kinds.contains(UsageKindFlags::COMPUTE) && usage.access.writes() && is_storage {
        (
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS,
        )
    } else if usage.kinds.contains(UsageKindFlags::COMPUTE) {
        (
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
            D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE,
        )
    } else if usage.kinds.contains(UsageKindFlags::RENDER) {
        (
            D3D12_BARRIER_SYNC_RENDER_TARGET,
            D3D12_BARRIER_ACCESS_RENDER_TARGET,
            D3D12_BARRIER_LAYOUT_RENDER_TARGET,
        )
    } else {
        (
            D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_ACCESS_COMMON,
            D3D12_BARRIER_LAYOUT_COMMON,
        )
    }
}

#[derive(Debug)]
struct Dx12GpuProfileResources {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    query_count: u32,
    dispatch_labels: Vec<Option<&'static str>>,
}

fn dx12_collect_dispatch_labels(commands: &[GpuCommand]) -> (usize, Vec<Option<&'static str>>) {
    let mut labels = Vec::new();
    for c in commands {
        match c {
            GpuCommand::Dispatch { label, .. }
            | GpuCommand::DispatchIndirect { label, .. }
            | GpuCommand::DispatchBatch { label, .. } => {
                labels.push(*label);
            }
            _ => {}
        }
    }
    let n = labels.len();
    (n, labels)
}

fn dx12_collect_dispatch_labels_graph(
    commands: &[GraphCommand],
) -> (usize, Vec<Option<&'static str>>) {
    let mut labels = Vec::new();
    for gc in commands {
        if let GraphCommand::Compute(
            GpuCommand::Dispatch { label, .. }
            | GpuCommand::DispatchIndirect { label, .. }
            | GpuCommand::DispatchBatch { label, .. },
        ) = gc
        {
            labels.push(*label);
        }
    }
    let n = labels.len();
    (n, labels)
}

fn dx12_try_create_gpu_profile(
    device: &ID3D12Device10,
    dispatch_count: usize,
    dispatch_labels: Vec<Option<&'static str>>,
) -> Result<Option<Dx12GpuProfileResources>> {
    if !crate::gpu_profiler::gpu_profile_enabled() {
        return Ok(None);
    }
    debug_assert_eq!(dispatch_labels.len(), dispatch_count);
    let query_count = 2u32.saturating_add((dispatch_count as u32).saturating_mul(2));

    let heap_desc = D3D12_QUERY_HEAP_DESC {
        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
        Count: query_count,
        NodeMask: 0,
    };
    let mut heap_opt: Option<ID3D12QueryHeap> = None;
    unsafe { device.CreateQueryHeap(&heap_desc, &mut heap_opt) }
        .context("CreateQueryHeap for GOLDY_GPU_PROFILE")?;
    let heap = heap_opt.context("CreateQueryHeap returned null")?;

    let data_bytes = (query_count as u64).saturating_mul(8);
    let aligned_width = data_bytes.max(256).next_multiple_of(256);

    let buffer_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: aligned_width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let mut readback_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut readback_opt,
        )
    }
    .context("CreateCommittedResource readback for GOLDY_GPU_PROFILE")?;
    let readback = readback_opt.context("readback resource null")?;

    Ok(Some(Dx12GpuProfileResources {
        heap,
        readback,
        query_count,
        dispatch_labels,
    }))
}

fn dx12_decode_duration_ns(start: u64, end: u64, freq: u64) -> u64 {
    if freq == 0 {
        return 0;
    }
    let delta = end.wrapping_sub(start);
    ((delta as f64 / freq as f64) * 1e9) as u64
}

fn dx12_finish_gpu_profile(
    logical_device: &types::LogicalDevice,
    fence_value: u64,
    profile: Dx12GpuProfileResources,
) -> Result<()> {
    use crate::gpu_profiler::{self, DispatchGpuNs};
    super::utils::wait_for_fence(&logical_device.fence, fence_value)?;

    let freq = unsafe { logical_device.command_queue.GetTimestampFrequency() }
        .context("GetTimestampFrequency")?;

    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
    let no_read = D3D12_RANGE { Begin: 0, End: 0 };
    unsafe { profile.readback.Map(0, Some(&no_read), Some(&mut mapped)) }
        .context("Map gpu profile readback")?;

    let vals: Vec<u64> = unsafe {
        std::slice::from_raw_parts(mapped as *const u64, profile.query_count as usize).to_vec()
    };

    unsafe {
        profile.readback.Unmap(0, None);
    }

    let cb_ns = dx12_decode_duration_ns(vals[0], vals[1], freq);
    gpu_profiler::log_cb_timing("dx12", fence_value, cb_ns as f64 / 1_000_000.0);

    let n = profile.dispatch_labels.len();
    if n > 0 {
        let mut dispatches = Vec::with_capacity(n);
        for i in 0..n {
            let si = 2 + 2 * i;
            let ns = dx12_decode_duration_ns(vals[si], vals[si + 1], freq);
            let label = profile.dispatch_labels[i].unwrap_or("dispatch");
            dispatches.push(DispatchGpuNs { label, gpu_ns: ns });
        }
        gpu_profiler::log_dispatch_timings("dx12", fence_value, &dispatches);
    }

    Ok(())
}

/// Drain any pending debug-layer messages for this device into a single
/// human-readable string. Returns `None` when the device has no
/// `ID3D12InfoQueue` (debug layer disabled) or no messages are queued.
///
/// Useful in error paths like a `Close()` failure, where the actual cause
/// is written to the info queue before the HRESULT bubbles up. Without
/// this drain the debug-layer text goes to `OutputDebugString` and is
/// invisible to anyone not attached with a debugger.
fn drain_info_queue(device: &ID3D12Device10) -> Option<String> {
    let info_queue: ID3D12InfoQueue = device.cast().ok()?;
    let count = unsafe { info_queue.GetNumStoredMessages() };
    if count == 0 {
        return None;
    }
    let mut out = String::new();
    for i in 0..count {
        let mut len: usize = 0;
        unsafe {
            if info_queue.GetMessage(i, None, &mut len).is_err() {
                continue;
            }
        }
        let mut buf = vec![0u8; len];
        let msg_ptr = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
        unsafe {
            if info_queue.GetMessage(i, Some(msg_ptr), &mut len).is_err() {
                continue;
            }
            let msg = &*msg_ptr;
            let desc = std::slice::from_raw_parts(
                msg.pDescription,
                msg.DescriptionByteLength.saturating_sub(1),
            );
            let text = std::str::from_utf8(desc).unwrap_or("<non-utf8 description>");
            let severity = match msg.Severity {
                D3D12_MESSAGE_SEVERITY_CORRUPTION => "CORRUPTION",
                D3D12_MESSAGE_SEVERITY_ERROR => "ERROR",
                D3D12_MESSAGE_SEVERITY_WARNING => "WARNING",
                D3D12_MESSAGE_SEVERITY_INFO => "INFO",
                D3D12_MESSAGE_SEVERITY_MESSAGE => "MSG",
                _ => "?",
            };
            out.push_str(&format!(
                "  [D3D12 {}] id={} {}\n",
                severity, msg.ID.0, text
            ));
        }
    }
    unsafe { info_queue.ClearStoredMessages() };
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Create a compute pipeline.
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    compute_shader: ShaderHandle,
) -> Result<ComputePipelineHandle> {
    // Compile shader on-demand
    let cs_bytecode =
        shader::ensure_stage_compiled(state, compute_shader, crate::slang::SlangStage::Compute)?;

    let shader_debug_name = format!("compute_shader#{compute_shader}");

    let key = pso_cache::compute_pso_key(&cs_bytecode);

    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    // Use the shared bindless root signature from the device
    let root_signature = logical_device
        .bindless_root_signature
        .as_ref()
        .context("Bindless root signature not available")?
        .clone();

    tracing::debug!("Using shared bindless root signature for compute pipeline");

    let disk_blob = logical_device.compute_pso_blobs.get(&key);
    let mut try_drop_stale_cached_blob = disk_blob.is_some();
    let cached_pso = disk_blob
        .map(|b| pso_cache::d3d12_cached_pso(b.as_slice()))
        .unwrap_or_default();

    // Create compute PSO
    let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: cs_bytecode.as_ptr() as *const _,
            BytecodeLength: cs_bytecode.len(),
        },
        NodeMask: 0,
        CachedPSO: cached_pso,
        Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
    };

    let pipeline_state: ID3D12PipelineState = {
        let _tz = crate::tracy_zone!("goldy.dx12.CreateComputePipelineState");
        loop {
            match unsafe { logical_device.device.CreateComputePipelineState(&pso_desc) } {
                Ok(p) => break p,
                Err(e) if try_drop_stale_cached_blob => {
                    tracing::warn!(
                        device = device_handle,
                        error = ?e,
                        "discarding stale DX12 compute PSO blob; rebuilding without cache entry"
                    );
                    logical_device.compute_pso_blobs.remove(&key);
                    logical_device.pso_disk_cache_dirty = true;
                    pso_desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
                    try_drop_stale_cached_blob = false;
                }
                Err(e) => anyhow::bail!("Failed to create compute pipeline state: {:?}", e),
            }
        }
    };

    let blob = unsafe {
        pipeline_state
            .GetCachedBlob()
            .context("GetCachedBlob (compute PSO)")?
    };
    let new_blob = unsafe { pso_cache::id3dblob_to_vec(&blob) };

    match logical_device.compute_pso_blobs.get(&key) {
        Some(prev) if *prev == new_blob => {}
        _ => {
            logical_device.compute_pso_blobs.insert(key, new_blob);
            logical_device.pso_disk_cache_dirty = true;
        }
    }

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

    state.compute_pipelines.insert(
        handle,
        ComputePipelineState {
            device_handle,
            pipeline_state,
            root_signature,
            parameter_block_layouts: Vec::new(),
            push_constant_categories: cats,
            binding_element_strides: strides,
            shader_debug_name,
        },
    );

    tracing::debug!("Created compute pipeline {}", handle);
    Ok(handle)
}

/// Destroy a compute pipeline.
pub(super) fn destroy(state: &mut Dx12State, pipeline_handle: ComputePipelineHandle) {
    state.compute_pipelines.remove(&pipeline_handle);
}

// ---------------------------------------------------------------------------
// Shared submit helpers
// ---------------------------------------------------------------------------

/// Acquire (or create) a compute allocator slot, reserving the next fence token.
///
/// Returns `(command_list, fence_value, slot_idx)`.  The slot is taken from the
/// pool when its fence has already signalled; otherwise a fresh one is created.
fn acquire_allocator_slot(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
) -> Result<(ID3D12GraphicsCommandList, u64, usize)> {
    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    let fence = &logical_device.fence;
    let completed = unsafe { fence.GetCompletedValue() };
    let pool = &mut logical_device.compute_allocator_pool;
    // Skip retained slots — their allocator must not be reset until evict_retained is called.
    let slot_idx = pool
        .iter()
        .position(|s| completed >= s.fence_value && !s.retained);
    let (cmd_list, slot_idx) = if let Some(idx) = slot_idx {
        let slot = &mut pool[idx];
        unsafe { slot.allocator.Reset() }.context("Failed to reset command allocator")?;
        let list = if let Some(ref existing) = slot.command_list {
            unsafe { existing.Reset(&slot.allocator, None) }
                .context("Failed to reset command list")?;
            existing.clone()
        } else {
            let new_list: ID3D12GraphicsCommandList = unsafe {
                logical_device.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &slot.allocator,
                    None,
                )
            }
            .context("Failed to create command list")?;
            slot.command_list = Some(new_list.clone());
            new_list
        };
        (list, idx)
    } else {
        let new_allocator: ID3D12CommandAllocator = unsafe {
            logical_device
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .context("Failed to create command allocator")?;
        let new_list: ID3D12GraphicsCommandList = unsafe {
            logical_device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &new_allocator,
                None,
            )
        }
        .context("Failed to create command list")?;
        pool.push(ComputeAllocatorSlot {
            allocator: new_allocator,
            fence_value: 0,
            command_list: Some(new_list.clone()),
            retained: false,
        });
        (new_list, pool.len() - 1)
    };
    let token = logical_device.fence_value;
    logical_device.fence_value += 1;
    Ok((cmd_list, token, slot_idx))
}

/// Mutable state threaded through the per-command recording loop.
struct CmdCtx<'a> {
    command_list: &'a ID3D12GraphicsCommandList,
    command_list7: &'a ID3D12GraphicsCommandList7,
    use_global_buffer_barriers: bool,
    belt_slices: &'a [(ID3D12Resource, u64)],
    belt_idx: usize,
    staged_texture_uploads: &'a [super::texture::StagedTextureUpload],
    texture_upload_idx: usize,
    gpu_profile: &'a mut Option<Dx12GpuProfileResources>,
    dispatch_idx: u32,
    current_compute_pipeline: Option<ComputePipelineHandle>,
}

/// Emit one `GpuCommand` onto the open command list in `ctx`.
#[allow(clippy::too_many_lines)]
fn record_gpu_command(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    ctx: &mut CmdCtx<'_>,
    cmd: &GpuCommand,
) -> Result<()> {
    let cl = ctx.command_list;
    let cl7 = ctx.command_list7;
    match cmd {
        GpuCommand::SetPipeline(handle) => {
            let _tz = tracy_zone!("dx12.set_pipeline");
            ctx.current_compute_pipeline = Some(*handle);
            if let Some(pipeline_state) = state.compute_pipelines.get(handle) {
                unsafe {
                    cl.SetComputeRootSignature(&pipeline_state.root_signature);
                    cl.SetPipelineState(&pipeline_state.pipeline_state);
                }
            }
        }
        GpuCommand::BindResources { buffers } => {
            if crate::slang::layout_validation_enabled() {
                if let Some(pipeline) = ctx
                    .current_compute_pipeline
                    .and_then(|h| state.compute_pipelines.get(&h))
                {
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
            let mut layout = types::PushLayout::default();
            shared::fill_bindless(
                &mut layout,
                buffers.iter().map(|h| {
                    let offset = state
                        .buffers
                        .get(h)
                        .and_then(|b| b.bindless_offset)
                        .unwrap_or(0);
                    tracing::trace!(
                        "Compute BindResources: buffer {} -> UAV offset {}",
                        h,
                        offset
                    );
                    offset
                }),
            );
            tracing::trace!(
                "Setting compute root constants (bindless): {:?}",
                &layout.bindless[..buffers.len().min(types::MAX_BINDLESS_SLOTS)]
            );
            unsafe {
                cl.SetComputeRoot32BitConstants(
                    0,
                    (types::TOTAL_PUSH_BYTES / 4) as u32,
                    &layout as *const _ as *const std::ffi::c_void,
                    0,
                );
            }
        }
        GpuCommand::BindResourcesRaw {
            indices: raw_indices,
            user: raw_user,
        } => {
            let mut layout = types::PushLayout::default();
            shared::fill_raw(&mut layout, raw_indices, raw_user);
            unsafe {
                cl.SetComputeRoot32BitConstants(
                    0,
                    (types::TOTAL_PUSH_BYTES / 4) as u32,
                    &layout as *const _ as *const std::ffi::c_void,
                    0,
                );
            }
        }
        GpuCommand::BindResourcesTyped {
            handles: typed_handles,
        } => {
            if let Some(pipeline) = ctx
                .current_compute_pipeline
                .and_then(|h| state.compute_pipelines.get(&h))
            {
                crate::backend::validate_typed_push_constants(
                    typed_handles,
                    &pipeline.push_constant_categories,
                    &pipeline.shader_debug_name,
                )?;
            }
            let mut layout = types::PushLayout::default();
            shared::fill_typed(&mut layout, typed_handles.iter().copied());
            unsafe {
                cl.SetComputeRoot32BitConstants(
                    0,
                    (types::TOTAL_PUSH_BYTES / 4) as u32,
                    &layout as *const _ as *const std::ffi::c_void,
                    0,
                );
            }
        }
        GpuCommand::Dispatch {
            label: _,
            workgroups_x,
            workgroups_y,
            workgroups_z,
        } => {
            let _tz = tracy_zone!("dx12.dispatch");
            if let Some(ref prof) = ctx.gpu_profile {
                let base = 2u32 + ctx.dispatch_idx * 2;
                unsafe { cl.EndQuery(&prof.heap, D3D12_QUERY_TYPE_TIMESTAMP, base) };
            }
            unsafe { cl.Dispatch(*workgroups_x, *workgroups_y, *workgroups_z) };
            if let Some(ref prof) = ctx.gpu_profile {
                let base = 2u32 + ctx.dispatch_idx * 2;
                unsafe { cl.EndQuery(&prof.heap, D3D12_QUERY_TYPE_TIMESTAMP, base + 1) };
            }
            ctx.dispatch_idx += 1;
        }
        GpuCommand::DispatchIndirect {
            buffer,
            offset,
            label: _,
        } => {
            let _tz = tracy_zone!("dx12.dispatch_indirect");
            let logical_device = state
                .devices
                .get(&device_handle)
                .context("DispatchIndirect: invalid device")?;
            let buf_state = state
                .buffers
                .get(buffer)
                .context("DispatchIndirect: invalid buffer handle")?;
            let signature = logical_device
                .compute_dispatch_indirect_signature
                .as_ref()
                .context("DispatchIndirect: compute indirect signature not available")?;

            if ctx.use_global_buffer_barriers {
                let g = D3D12_GLOBAL_BARRIER {
                    SyncBefore: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    SyncAfter: D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                    AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    AccessAfter: D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
                };
                unsafe { barriers::barrier_globals(cl7, &[g]) };
            } else {
                let mut to_indirect = [barriers::buffer_barrier_full(
                    &buf_state.resource,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
                )];
                unsafe { barriers::barrier_buffers(cl7, &to_indirect) };
                unsafe { barriers::drop_buffer_barriers(&mut to_indirect) };
            }

            if let Some(ref prof) = ctx.gpu_profile {
                let base = 2u32 + ctx.dispatch_idx * 2;
                unsafe { cl.EndQuery(&prof.heap, D3D12_QUERY_TYPE_TIMESTAMP, base) };
            }
            unsafe {
                cl.ExecuteIndirect(signature, 1, &buf_state.resource, *offset, None, 0);
            }
            if let Some(ref prof) = ctx.gpu_profile {
                let base = 2u32 + ctx.dispatch_idx * 2;
                unsafe { cl.EndQuery(&prof.heap, D3D12_QUERY_TYPE_TIMESTAMP, base + 1) };
            }
            ctx.dispatch_idx += 1;

            if ctx.use_global_buffer_barriers {
                let g = D3D12_GLOBAL_BARRIER {
                    SyncBefore: D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                    SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    AccessBefore: D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
                    AccessAfter: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                };
                unsafe { barriers::barrier_globals(cl7, &[g]) };
            } else {
                let mut to_uav = [barriers::buffer_barrier_full(
                    &buf_state.resource,
                    D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                )];
                unsafe { barriers::barrier_buffers(cl7, &to_uav) };
                unsafe { barriers::drop_buffer_barriers(&mut to_uav) };
            }
        }
        GpuCommand::DispatchBatch {
            label: _,
            arg_data,
            count,
        } => {
            let _tz = tracy_zone!("dx12.dispatch_batch");
            let logical_device = state
                .devices
                .get(&device_handle)
                .context("DispatchBatch: invalid device")?;

            if let Some(batch_sig) = logical_device.compute_batch_dispatch_signature.clone() {
                let buf_size = arg_data.len() as u64;
                let arg_buf_desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Alignment: 0,
                    Width: buf_size,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_UNKNOWN,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_NONE,
                };
                let upload_heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_UPLOAD,
                    ..Default::default()
                };
                let mut arg_resource: Option<ID3D12Resource> = None;
                unsafe {
                    logical_device.device.CreateCommittedResource(
                        &upload_heap,
                        D3D12_HEAP_FLAG_NONE,
                        &arg_buf_desc,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                        None,
                        &mut arg_resource,
                    )
                }
                .context("DispatchBatch: failed to create arg buffer")?;
                let arg_resource = arg_resource.context("DispatchBatch: arg_resource is None")?;

                let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
                let no_read = D3D12_RANGE { Begin: 0, End: 0 };
                unsafe {
                    arg_resource
                        .Map(0, Some(&no_read), Some(&mut mapped))
                        .context("DispatchBatch: failed to map arg buffer")?;
                    std::ptr::copy_nonoverlapping(
                        arg_data.as_ptr(),
                        mapped as *mut u8,
                        arg_data.len(),
                    );
                    let written = D3D12_RANGE {
                        Begin: 0,
                        End: arg_data.len(),
                    };
                    arg_resource.Unmap(0, Some(&written));
                }
                unsafe {
                    cl.ExecuteIndirect(&batch_sig, *count, &arg_resource, 0, None, 0);
                }

                let fence_val = logical_device.fence_value.saturating_sub(1);
                let dev = state
                    .devices
                    .get_mut(&device_handle)
                    .context("DispatchBatch: device gone")?;
                dev.deletion_queue.queue(
                    fence_val,
                    super::types::PendingDeletion::StandaloneResource(arg_resource),
                );
            } else {
                use crate::backend::shared::{PushLayout, DISPATCH_BATCH_STRIDE};
                let stride = DISPATCH_BATCH_STRIDE;
                let push_size = std::mem::size_of::<PushLayout>();
                for i in 0..*count as usize {
                    let base = i * stride;
                    let layout_bytes = &arg_data[base..base + push_size];
                    let wg_off = base + push_size;
                    let wg_x = u32::from_ne_bytes(arg_data[wg_off..wg_off + 4].try_into().unwrap());
                    let wg_y =
                        u32::from_ne_bytes(arg_data[wg_off + 4..wg_off + 8].try_into().unwrap());
                    let wg_z =
                        u32::from_ne_bytes(arg_data[wg_off + 8..wg_off + 12].try_into().unwrap());
                    unsafe {
                        cl.SetComputeRoot32BitConstants(
                            0,
                            (push_size / 4) as u32,
                            layout_bytes.as_ptr() as *const _,
                            0,
                        );
                        cl.Dispatch(wg_x, wg_y, wg_z);
                    }
                }
            }
        }
        GpuCommand::Barrier => {
            // Legacy non-graph barrier: no access semantics available, use the
            // conservative global sync.  This path is audited by `remove-legacy-barrier`.
            let _tz = tracy_zone!("dx12.barrier");
            let g = D3D12_GLOBAL_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_ALL,
                SyncAfter: D3D12_BARRIER_SYNC_ALL,
                AccessBefore: D3D12_BARRIER_ACCESS_COMMON,
                AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
            };
            unsafe { barriers::barrier_globals(cl7, &[g]) };
        }
        GpuCommand::ResourceBarrier {
            buffers: buf_handles,
            textures: tex_handles,
            src_usage,
            dst_usage,
        } => {
            let _tz = tracy_zone!("dx12.resource_barrier");
            let sync_before = slot_usage_to_dx12_sync(src_usage);
            let access_before = slot_usage_to_dx12_access(src_usage);
            let sync_after = slot_usage_to_dx12_sync(dst_usage);
            let access_after = slot_usage_to_dx12_access(dst_usage);

            // WARP silently removes the device when D3D12_BARRIER_TYPE_BUFFER
            // enhanced barriers are used, causing ExecuteCommandLists to fail
            // and the subsequent Signal() to AV. Fall back to a global barrier
            // which is correct (just less precise).
            if ctx.use_global_buffer_barriers || buf_handles.is_empty() {
                if !buf_handles.is_empty() {
                    let g = D3D12_GLOBAL_BARRIER {
                        SyncBefore: sync_before,
                        SyncAfter: sync_after,
                        AccessBefore: access_before,
                        AccessAfter: access_after,
                    };
                    unsafe { barriers::barrier_globals(cl7, &[g]) };
                }
            } else {
                let mut buf_barriers: Vec<D3D12_BUFFER_BARRIER> = buf_handles
                    .iter()
                    .filter_map(|h| state.buffers.get(h))
                    .map(|bs| {
                        barriers::buffer_barrier_full(
                            &bs.resource,
                            sync_before,
                            sync_after,
                            access_before,
                            access_after,
                        )
                    })
                    .collect();
                unsafe { barriers::barrier_buffers(cl7, &buf_barriers) };
                unsafe { barriers::drop_buffer_barriers(&mut buf_barriers) };
            }

            let mut tex_barriers: Vec<D3D12_TEXTURE_BARRIER> = tex_handles
                .iter()
                .filter_map(|h| state.textures.get(h))
                .map(|ts| {
                    let (tex_sync_after, tex_access_after, tex_layout_after) =
                        texture_barrier_state_for_usage(dst_usage, ts.is_storage);
                    let (tex_sync_before, tex_access_before, tex_layout_before) =
                        texture_barrier_state_for_layout(ts.last_layout);
                    barriers::texture_barrier_full(
                        &ts.resource,
                        tex_sync_before,
                        tex_sync_after,
                        tex_access_before,
                        tex_access_after,
                        tex_layout_before,
                        tex_layout_after,
                    )
                })
                .collect();
            unsafe { barriers::barrier_textures(cl7, &tex_barriers) };
            unsafe { barriers::drop_texture_barriers(&mut tex_barriers) };
            for h in tex_handles {
                if let Some(ts) = state.textures.get_mut(h) {
                    let (_, _, tex_layout_after) =
                        texture_barrier_state_for_usage(dst_usage, ts.is_storage);
                    ts.last_layout = tex_layout_after;
                }
            }
        }
        GpuCommand::ClearBuffer {
            buffer,
            offset,
            size,
        } => {
            let _tz = tracy_zone!("dx12.clear_buffer");
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
                if buf_state.is_storage {
                    let logical_device = state
                        .devices
                        .get(&device_handle)
                        .context("ClearBuffer: invalid device")?;
                    let zero = logical_device.zero_buffer.clone();
                    let buf_resource = buf_state.resource.clone();
                    if ctx.use_global_buffer_barriers {
                        let pre = D3D12_GLOBAL_BARRIER {
                            SyncBefore: D3D12_BARRIER_SYNC_ALL,
                            SyncAfter: D3D12_BARRIER_SYNC_COPY,
                            AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                            AccessAfter: D3D12_BARRIER_ACCESS_COPY_DEST,
                        };
                        unsafe { barriers::barrier_globals(cl7, &[pre]) };
                    } else {
                        let mut b_to_copy = [barriers::buffer_barrier_full(
                            &buf_resource,
                            D3D12_BARRIER_SYNC_ALL,
                            D3D12_BARRIER_SYNC_COPY,
                            D3D12_BARRIER_ACCESS_COMMON,
                            D3D12_BARRIER_ACCESS_COPY_DEST,
                        )];
                        unsafe {
                            barriers::barrier_buffers(cl7, &b_to_copy);
                            barriers::drop_buffer_barriers(&mut b_to_copy);
                        }
                    }
                    let mut cleared = 0u64;
                    while cleared < clear_size {
                        let this_chunk =
                            (clear_size - cleared).min(super::buffer::ZERO_BUFFER_SIZE);
                        unsafe {
                            cl.CopyBufferRegion(
                                &buf_resource,
                                *offset + cleared,
                                &zero,
                                0,
                                this_chunk,
                            );
                        }
                        cleared += this_chunk;
                    }
                } else {
                    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
                    let no_read = D3D12_RANGE { Begin: 0, End: 0 };
                    unsafe { buf_state.resource.Map(0, Some(&no_read), Some(&mut mapped)) }
                        .context("ClearBuffer: failed to map buffer")?;
                    unsafe {
                        std::ptr::write_bytes(
                            (mapped as *mut u8).add(*offset as usize),
                            0,
                            clear_size as usize,
                        );
                    }
                    let written = D3D12_RANGE {
                        Begin: *offset as usize,
                        End: (*offset + clear_size) as usize,
                    };
                    unsafe { buf_state.resource.Unmap(0, Some(&written)) };
                }
            }
        }
        GpuCommand::WriteBuffer {
            buffer: buf_handle,
            offset,
            data,
        } => {
            let _tz = tracy_zone!("dx12.write_buffer");
            let buf_state = state
                .buffers
                .get(buf_handle)
                .context("WriteBuffer: invalid buffer handle")?;
            if !buf_state.is_storage {
                let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
                let no_read = D3D12_RANGE { Begin: 0, End: 0 };
                unsafe { buf_state.resource.Map(0, Some(&no_read), Some(&mut mapped)) }
                    .context("WriteBuffer: map failed")?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        (mapped as *mut u8).add(*offset as usize),
                        data.len(),
                    );
                }
                let written_range = D3D12_RANGE {
                    Begin: *offset as usize,
                    End: (*offset as usize) + data.len(),
                };
                unsafe { buf_state.resource.Unmap(0, Some(&written_range)) };
            } else {
                let belt_entry = ctx
                    .belt_slices
                    .get(ctx.belt_idx)
                    .context("WriteBuffer: belt slice missing (internal)")?;
                ctx.belt_idx += 1;
                let upload_src = belt_entry.0.clone();
                let upload_off = belt_entry.1;
                let buf_state = state.buffers.get(buf_handle).unwrap();

                if ctx.use_global_buffer_barriers {
                    let pre = D3D12_GLOBAL_BARRIER {
                        SyncBefore: D3D12_BARRIER_SYNC_ALL,
                        SyncAfter: D3D12_BARRIER_SYNC_COPY,
                        AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        AccessAfter: D3D12_BARRIER_ACCESS_COPY_DEST,
                    };
                    unsafe {
                        barriers::barrier_globals(cl7, &[pre]);
                        cl.CopyBufferRegion(
                            &buf_state.resource,
                            *offset,
                            &upload_src,
                            upload_off,
                            data.len() as u64,
                        );
                    }
                } else {
                    let mut b_to_copy = [barriers::buffer_barrier_full(
                        &buf_state.resource,
                        D3D12_BARRIER_SYNC_ALL,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_COPY_DEST,
                    )];
                    unsafe {
                        barriers::barrier_buffers(cl7, &b_to_copy);
                        barriers::drop_buffer_barriers(&mut b_to_copy);
                        cl.CopyBufferRegion(
                            &buf_state.resource,
                            *offset,
                            &upload_src,
                            upload_off,
                            data.len() as u64,
                        );
                    }
                }
            }
        }
        GpuCommand::WriteTexture { .. } | GpuCommand::WriteTextureRegion { .. } => {
            let _tz = tracy_zone!("dx12.write_texture");
            let upload = ctx
                .staged_texture_uploads
                .get(ctx.texture_upload_idx)
                .context("WriteTexture: staged upload missing (internal)")?;
            ctx.texture_upload_idx += 1;
            super::texture::record_staged_texture_upload(cl, cl7, &mut state.textures, upload)?;
        }
        GpuCommand::CopyTexture { src, dst } => {
            let _tz = tracy_zone!("dx12.copy_texture");
            let (src_res, src_layout, src_is_storage) = {
                let ts = state
                    .textures
                    .get(src)
                    .context("CopyTexture: src texture not found")?;
                (ts.resource.clone(), ts.last_layout, ts.is_storage)
            };
            let (dst_res, dst_layout, dst_is_storage) = {
                let ts = state
                    .textures
                    .get(dst)
                    .context("CopyTexture: dst texture not found")?;
                (ts.resource.clone(), ts.last_layout, ts.is_storage)
            };

            let (src_sync_before, src_access_before, src_layout_before) =
                texture_barrier_state_for_layout(src_layout);
            let (dst_sync_before, dst_access_before, dst_layout_before) =
                texture_barrier_state_for_layout(dst_layout);

            let mut pre_barriers = vec![
                barriers::texture_barrier_full(
                    &src_res,
                    src_sync_before,
                    D3D12_BARRIER_SYNC_COPY,
                    src_access_before,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    src_layout_before,
                    D3D12_BARRIER_LAYOUT_COPY_SOURCE,
                ),
                barriers::texture_barrier_full(
                    &dst_res,
                    dst_sync_before,
                    D3D12_BARRIER_SYNC_COPY,
                    dst_access_before,
                    D3D12_BARRIER_ACCESS_COPY_DEST,
                    dst_layout_before,
                    D3D12_BARRIER_LAYOUT_COPY_DEST,
                ),
            ];
            unsafe { barriers::barrier_textures(cl7, &pre_barriers) };
            unsafe { barriers::drop_texture_barriers(&mut pre_barriers) };

            unsafe { cl.CopyResource(&dst_res, &src_res) };

            let src_post_state = if src_is_storage {
                (
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS,
                )
            } else {
                (
                    D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE,
                )
            };
            let dst_post_state = if dst_is_storage {
                (
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS,
                )
            } else {
                (
                    D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE,
                )
            };
            let mut post_barriers = vec![
                barriers::texture_barrier_full(
                    &src_res,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    src_post_state.0,
                    D3D12_BARRIER_LAYOUT_COPY_SOURCE,
                    src_post_state.1,
                ),
                barriers::texture_barrier_full(
                    &dst_res,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_COPY_DEST,
                    dst_post_state.0,
                    D3D12_BARRIER_LAYOUT_COPY_DEST,
                    dst_post_state.1,
                ),
            ];
            unsafe { barriers::barrier_textures(cl7, &post_barriers) };
            unsafe { barriers::drop_texture_barriers(&mut post_barriers) };

            if let Some(ts) = state.textures.get_mut(src) {
                ts.last_layout = src_post_state.1;
            }
            if let Some(ts) = state.textures.get_mut(dst) {
                ts.last_layout = dst_post_state.1;
            }
        }
    }
    Ok(())
}

struct SubmitFinish {
    device_handle: DeviceHandle,
    fence_value: u64,
    slot_idx: usize,
    retain_key: Option<u64>,
}

struct StagingFinish {
    texture_uploads: Vec<super::texture::StagedTextureUpload>,
    belt_slices_len: usize,
    belt_idx: usize,
}

/// Close the command list, execute, signal, update the pool slot, and finish staging.
///
/// `retain_key`: when `Some(k)`, stores the closed command list in
/// `LogicalDevice::retained_graph` for zero-cost re-execution via
/// [`try_resubmit_retained`].
fn execute_signal_and_finish(
    state: &mut Dx12State,
    command_list: &ID3D12GraphicsCommandList,
    gpu_profile: Option<Dx12GpuProfileResources>,
    submit: SubmitFinish,
    staging_finish: StagingFinish,
) -> Result<TimelineValue> {
    let SubmitFinish {
        device_handle,
        fence_value,
        slot_idx,
        retain_key,
    } = submit;
    let StagingFinish {
        texture_uploads: staged_texture_uploads,
        belt_slices_len,
        belt_idx,
    } = staging_finish;

    debug_assert_eq!(
        belt_idx, belt_slices_len,
        "WriteBuffer storage count mismatch vs belt prepass"
    );

    if let Err(e) = unsafe { command_list.Close() } {
        let diag = state
            .devices
            .get(&device_handle)
            .and_then(|dev| drain_info_queue(&dev.device))
            .unwrap_or_else(|| {
                "  (no debug-layer messages; enable GOLDY_DX12_DEBUG=1)\n".to_string()
            });
        return Err(anyhow::anyhow!(
            "Failed to close command list: {e}\nDebug layer messages:\n{diag}"
        ));
    }

    let cmd_list: ID3D12CommandList = command_list.cast().context("Failed to cast command list")?;

    let logical_device = state.devices.get(&device_handle).unwrap();
    {
        let _tz = tracy_zone!("dx12.execute_and_signal");
        unsafe {
            logical_device
                .command_queue
                .ExecuteCommandLists(&[Some(cmd_list)])
        };
        unsafe {
            logical_device
                .command_queue
                .Signal(&logical_device.fence, fence_value)
        }
        .context("Failed to signal fence")?;
    }

    if let Some(prof) = gpu_profile {
        if let Err(e) = dx12_finish_gpu_profile(logical_device, fence_value, prof) {
            tracing::warn!("GOLDY_GPU_PROFILE: DX12 readback failed: {e}");
        }
    }

    if let Some(dev) = state.devices.get_mut(&device_handle) {
        if let Some(slot) = dev.compute_allocator_pool.get_mut(slot_idx) {
            slot.fence_value = fence_value;
        }
        if let Some(key) = retain_key {
            if let Some(old) = dev.retained_graph.take() {
                if let Some(old_slot) = dev.compute_allocator_pool.get_mut(old.slot_idx) {
                    old_slot.retained = false;
                }
            }
            if let Some(cl) = dev
                .compute_allocator_pool
                .get(slot_idx)
                .and_then(|s| s.command_list.clone())
            {
                dev.compute_allocator_pool[slot_idx].retained = true;
                dev.retained_graph = Some(types::RetainedGraph {
                    fingerprint: key,
                    command_list: cl,
                    slot_idx,
                });
            }
        }
        dev.process_deletion_queue();
    }

    state
        .staging_belts
        .entry(device_handle)
        .or_insert_with(|| staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE))
        .finish(fence_value);

    if !staged_texture_uploads.is_empty() {
        let entries = staged_texture_uploads
            .into_iter()
            .map(|u| u.staging_entry)
            .collect::<Vec<_>>();
        state
            .texture_staging_pools
            .entry(device_handle)
            .or_insert_with(super::staging::TextureStagingPool::new)
            .release(fence_value, entries);
    }

    Ok(fence_value)
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Submit compute commands without blocking. Returns a fence token for polling/waiting.
pub(super) fn submit(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    commands: &[GpuCommand],
) -> Result<TimelineValue> {
    let _tz = tracy_zone!("dx12.submit");
    let (command_list, fence_value, slot_idx) = {
        let _tz_acq = tracy_zone!("dx12.submit.acquire_allocator");
        acquire_allocator_slot(state, device_handle)?
    };

    let (fence_clone, use_global_buffer_barriers) = {
        let dev = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        (dev.fence.clone(), dev.adapter_id == super::WARP_ADAPTER_ID)
    };

    let has_upload = commands.iter().any(|c| {
        matches!(
            c,
            GpuCommand::WriteBuffer { .. }
                | GpuCommand::WriteTexture { .. }
                | GpuCommand::WriteTextureRegion { .. }
        )
    });
    if has_upload {
        let _tz_reclaim = tracy_zone!("dx12.submit.staging_reclaim");
        state
            .staging_belts
            .entry(device_handle)
            .or_insert_with(|| staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE))
            .reclaim(&fence_clone)?;
        if let Some(pool) = state.texture_staging_pools.get_mut(&device_handle) {
            let completed = unsafe { fence_clone.GetCompletedValue() };
            pool.reclaim(completed);
        }
    }

    let command_list7: ID3D12GraphicsCommandList7 = command_list
        .cast()
        .context("ID3D12GraphicsCommandList7 required")?;

    let mut belt_slices: Vec<(ID3D12Resource, u64)> = Vec::new();
    let mut staged_texture_uploads: Vec<super::texture::StagedTextureUpload> = Vec::new();
    if has_upload {
        let mut pool = state
            .texture_staging_pools
            .remove(&device_handle)
            .unwrap_or_else(super::staging::TextureStagingPool::new);

        let _tz_prepass = tracy_zone!("dx12.submit.upload_prepass");
        for command in commands {
            match command {
                GpuCommand::WriteBuffer {
                    buffer: buf_handle,
                    data,
                    ..
                } => {
                    let buf = state
                        .buffers
                        .get(buf_handle)
                        .context("WriteBuffer pre-pass: invalid handle")?;
                    if buf.is_storage {
                        let buf_dev = buf.device_handle;
                        let ld = state
                            .devices
                            .get(&buf_dev)
                            .context("WriteBuffer pre-pass: device missing")?;
                        let belt = state.staging_belts.entry(buf_dev).or_insert_with(|| {
                            staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE)
                        });
                        let (res, off) = belt.write(ld, data)?;
                        belt_slices.push((res, off));
                    }
                }
                GpuCommand::WriteTexture {
                    texture,
                    data,
                    width,
                    height,
                } => {
                    staged_texture_uploads.push(super::texture::stage_texture_upload_full(
                        &state.devices,
                        &state.textures,
                        &mut pool,
                        *texture,
                        data,
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
                    staged_texture_uploads.push(super::texture::stage_texture_upload_region(
                        &state.devices,
                        &state.textures,
                        &mut pool,
                        super::texture::TextureUploadRegion {
                            texture_handle: *texture,
                            x: *x,
                            y: *y,
                            width: *width,
                            height: *height,
                            data,
                        },
                    )?);
                }
                _ => {}
            }
        }

        state.texture_staging_pools.insert(device_handle, pool);
    }

    let mut dx_gpu_profile = {
        let _tz_gp = tracy_zone!("dx12.submit.gpu_profile_setup");
        let logical_device_ref = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        let (dispatch_count, dispatch_labels) = dx12_collect_dispatch_labels(commands);
        let prof = match dx12_try_create_gpu_profile(
            &logical_device_ref.device,
            dispatch_count,
            dispatch_labels,
        ) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("GOLDY_GPU_PROFILE: DX12 timestamp heap creation failed: {e}");
                None
            }
        };
        unsafe {
            command_list.SetDescriptorHeaps(&[
                Some(logical_device_ref.cbv_srv_uav_heap.clone()),
                Some(logical_device_ref.sampler_heap.clone()),
            ]);
        }
        if let Some(ref p) = prof {
            unsafe { command_list.EndQuery(&p.heap, D3D12_QUERY_TYPE_TIMESTAMP, 0) };
        }
        prof
    };

    let belt_idx_final = {
        let _tz_cmds = tracy_zone!("dx12.submit.record_commands");
        let mut ctx = CmdCtx {
            command_list: &command_list,
            command_list7: &command_list7,
            use_global_buffer_barriers,
            belt_slices: &belt_slices,
            belt_idx: 0,
            staged_texture_uploads: &staged_texture_uploads,
            texture_upload_idx: 0,
            gpu_profile: &mut dx_gpu_profile,
            dispatch_idx: 0,
            current_compute_pipeline: None,
        };
        for cmd in commands {
            record_gpu_command(state, device_handle, &mut ctx, cmd)?;
        }
        debug_assert_eq!(
            ctx.texture_upload_idx,
            staged_texture_uploads.len(),
            "WriteTexture command count mismatch vs staging pre-pass"
        );
        ctx.belt_idx
    };

    // Tail barrier: make UAV and copy writes visible to subsequent operations.
    let tail = D3D12_GLOBAL_BARRIER {
        SyncBefore: D3D12_BARRIER_SYNC(
            D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0,
        ),
        SyncAfter: D3D12_BARRIER_SYNC_ALL,
        AccessBefore: D3D12_BARRIER_ACCESS(
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_COPY_DEST.0,
        ),
        AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
    };
    unsafe { barriers::barrier_globals(&command_list7, &[tail]) };

    if let Some(ref prof) = dx_gpu_profile {
        unsafe {
            command_list.EndQuery(&prof.heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            command_list.ResolveQueryData(
                &prof.heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                0,
                prof.query_count,
                &prof.readback,
                0,
            );
        }
    }

    execute_signal_and_finish(
        state,
        &command_list,
        dx_gpu_profile.take(),
        SubmitFinish {
            device_handle,
            fence_value,
            slot_idx,
            retain_key: None,
        },
        StagingFinish {
            texture_uploads: staged_texture_uploads,
            belt_slices_len: belt_slices.len(),
            belt_idx: belt_idx_final,
        },
    )
}

/// Submit mixed compute + render graph commands in a single command list.
///
/// Eliminates CPU waits between compute and render segments by recording
/// everything into one `ID3D12GraphicsCommandList7` and performing a single
/// `ExecuteCommandLists` + `Signal(fence)` at the end.
///
/// When `retain_key` is `Some(k)`, the closed command list is stored in
/// `LogicalDevice::retained_graph` keyed by `k` for future zero-cost re-execution
/// via [`try_resubmit_retained`].  Any previously retained graph is evicted first.
pub(super) fn submit_graph(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    commands: &[GraphCommand],
    retain_key: Option<u64>,
) -> Result<TimelineValue> {
    let _tz = tracy_zone!("dx12.submit_graph");
    let (command_list, fence_value, slot_idx) = acquire_allocator_slot(state, device_handle)?;

    let (fence_clone, use_global_buffer_barriers) = {
        let dev = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        (dev.fence.clone(), dev.adapter_id == super::WARP_ADAPTER_ID)
    };

    let has_upload = commands.iter().any(|c| {
        matches!(
            c,
            GraphCommand::Compute(
                GpuCommand::WriteBuffer { .. }
                    | GpuCommand::WriteTexture { .. }
                    | GpuCommand::WriteTextureRegion { .. }
            )
        )
    });
    if has_upload {
        state
            .staging_belts
            .entry(device_handle)
            .or_insert_with(|| staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE))
            .reclaim(&fence_clone)?;
        if let Some(pool) = state.texture_staging_pools.get_mut(&device_handle) {
            let completed = unsafe { fence_clone.GetCompletedValue() };
            pool.reclaim(completed);
        }
    }

    let command_list7: ID3D12GraphicsCommandList7 = command_list
        .cast()
        .context("ID3D12GraphicsCommandList7 required")?;

    let mut belt_slices: Vec<(ID3D12Resource, u64)> = Vec::new();
    let mut staged_texture_uploads: Vec<super::texture::StagedTextureUpload> = Vec::new();
    if has_upload {
        let mut pool = state
            .texture_staging_pools
            .remove(&device_handle)
            .unwrap_or_else(super::staging::TextureStagingPool::new);

        let _tz_prepass = tracy_zone!("dx12.submit_graph.upload_prepass");
        for graph_cmd in commands {
            if let GraphCommand::Compute(gpu_cmd) = graph_cmd {
                match gpu_cmd {
                    GpuCommand::WriteBuffer {
                        buffer: buf_handle,
                        data,
                        ..
                    } => {
                        let buf = state
                            .buffers
                            .get(buf_handle)
                            .context("WriteBuffer pre-pass: invalid handle")?;
                        if buf.is_storage {
                            let buf_dev = buf.device_handle;
                            let ld = state
                                .devices
                                .get(&buf_dev)
                                .context("WriteBuffer pre-pass: device missing")?;
                            let belt = state.staging_belts.entry(buf_dev).or_insert_with(|| {
                                staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE)
                            });
                            let (res, off) = belt.write(ld, data)?;
                            belt_slices.push((res, off));
                        }
                    }
                    GpuCommand::WriteTexture {
                        texture,
                        data,
                        width,
                        height,
                    } => {
                        staged_texture_uploads.push(super::texture::stage_texture_upload_full(
                            &state.devices,
                            &state.textures,
                            &mut pool,
                            *texture,
                            data,
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
                        staged_texture_uploads.push(super::texture::stage_texture_upload_region(
                            &state.devices,
                            &state.textures,
                            &mut pool,
                            super::texture::TextureUploadRegion {
                                texture_handle: *texture,
                                x: *x,
                                y: *y,
                                width: *width,
                                height: *height,
                                data,
                            },
                        )?);
                    }
                    _ => {}
                }
            }
        }

        state.texture_staging_pools.insert(device_handle, pool);
    }

    let mut dx_gpu_profile = {
        let _tz_gp = tracy_zone!("dx12.submit_graph.gpu_profile_setup");
        let logical_device_ref = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        let (dispatch_count, dispatch_labels) = dx12_collect_dispatch_labels_graph(commands);
        let prof = match dx12_try_create_gpu_profile(
            &logical_device_ref.device,
            dispatch_count,
            dispatch_labels,
        ) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("GOLDY_GPU_PROFILE: DX12 timestamp heap creation failed: {e}");
                None
            }
        };
        unsafe {
            command_list.SetDescriptorHeaps(&[
                Some(logical_device_ref.cbv_srv_uav_heap.clone()),
                Some(logical_device_ref.sampler_heap.clone()),
            ]);
        }
        if let Some(ref p) = prof {
            unsafe { command_list.EndQuery(&p.heap, D3D12_QUERY_TYPE_TIMESTAMP, 0) };
        }
        prof
    };

    let mut rendered_targets: Vec<RenderTargetHandle> = Vec::new();
    let belt_idx_final;
    {
        let _tz_cmds = tracy_zone!("dx12.submit_graph.record_commands");
        let mut ctx = CmdCtx {
            command_list: &command_list,
            command_list7: &command_list7,
            use_global_buffer_barriers,
            belt_slices: &belt_slices,
            belt_idx: 0,
            staged_texture_uploads: &staged_texture_uploads,
            texture_upload_idx: 0,
            gpu_profile: &mut dx_gpu_profile,
            dispatch_idx: 0,
            current_compute_pipeline: None,
        };

        for graph_cmd in commands {
            match graph_cmd {
                GraphCommand::Compute(gpu_cmd) => {
                    record_gpu_command(state, device_handle, &mut ctx, gpu_cmd)?;
                }
                GraphCommand::Render {
                    target,
                    commands: render_cmds,
                } => {
                    let _tz = tracy_zone!("dx12.render_pass");
                    let compute_to_render = D3D12_GLOBAL_BARRIER {
                        SyncBefore: D3D12_BARRIER_SYNC(
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0,
                        ),
                        SyncAfter: D3D12_BARRIER_SYNC(
                            D3D12_BARRIER_SYNC_RENDER_TARGET.0
                                | D3D12_BARRIER_SYNC_DEPTH_STENCIL.0
                                | D3D12_BARRIER_SYNC_VERTEX_SHADING.0
                                | D3D12_BARRIER_SYNC_PIXEL_SHADING.0,
                        ),
                        AccessBefore: D3D12_BARRIER_ACCESS(
                            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0
                                | D3D12_BARRIER_ACCESS_COPY_DEST.0,
                        ),
                        AccessAfter: D3D12_BARRIER_ACCESS(
                            D3D12_BARRIER_ACCESS_RENDER_TARGET.0
                                | D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE.0
                                | D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0,
                        ),
                    };
                    unsafe { barriers::barrier_globals(ctx.command_list7, &[compute_to_render]) };

                    super::render_target::record_render_pass_to_list(
                        state,
                        device_handle,
                        *target,
                        render_cmds,
                        &command_list7,
                    )?;
                    rendered_targets.push(*target);

                    let render_to_compute = D3D12_GLOBAL_BARRIER {
                        SyncBefore: D3D12_BARRIER_SYNC(
                            D3D12_BARRIER_SYNC_RENDER_TARGET.0 | D3D12_BARRIER_SYNC_DEPTH_STENCIL.0,
                        ),
                        SyncAfter: D3D12_BARRIER_SYNC(
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0,
                        ),
                        AccessBefore: D3D12_BARRIER_ACCESS(
                            D3D12_BARRIER_ACCESS_RENDER_TARGET.0
                                | D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE.0,
                        ),
                        AccessAfter: D3D12_BARRIER_ACCESS(
                            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0
                                | D3D12_BARRIER_ACCESS_COPY_SOURCE.0
                                | D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0,
                        ),
                    };
                    unsafe { barriers::barrier_globals(ctx.command_list7, &[render_to_compute]) };
                }
            }
        }

        debug_assert_eq!(
            ctx.texture_upload_idx,
            staged_texture_uploads.len(),
            "WriteTexture command count mismatch vs staging pre-pass"
        );
        belt_idx_final = ctx.belt_idx;
    } // drop ctx → release borrows on command_list / command_list7

    let tail = D3D12_GLOBAL_BARRIER {
        SyncBefore: D3D12_BARRIER_SYNC(
            D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0,
        ),
        SyncAfter: D3D12_BARRIER_SYNC_ALL,
        AccessBefore: D3D12_BARRIER_ACCESS(
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_COPY_DEST.0,
        ),
        AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
    };
    unsafe { barriers::barrier_globals(&command_list7, &[tail]) };

    if let Some(ref prof) = dx_gpu_profile {
        unsafe {
            command_list.EndQuery(&prof.heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            command_list.ResolveQueryData(
                &prof.heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                0,
                prof.query_count,
                &prof.readback,
                0,
            );
        }
    }

    let result = execute_signal_and_finish(
        state,
        &command_list,
        dx_gpu_profile.take(),
        SubmitFinish {
            device_handle,
            fence_value,
            slot_idx,
            retain_key,
        },
        StagingFinish {
            texture_uploads: staged_texture_uploads,
            belt_slices_len: belt_slices.len(),
            belt_idx: belt_idx_final,
        },
    )?;

    for t in rendered_targets {
        if let Some(rt) = state.render_targets.get_mut(&t) {
            rt.has_rendered = true;
        }
    }

    Ok(result)
}

/// Re-execute a previously retained command list without re-recording.
///
/// Calls `ExecuteCommandLists` on the closed list stored by a prior
/// `submit_graph(..., Some(key))` call, then signals the device fence.
/// Returns `Ok(Some(tv))` on success, `Ok(None)` if no retained list matches `key`.
pub(super) fn try_resubmit_retained(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    key: u64,
) -> Result<Option<TimelineValue>> {
    let retained = {
        let dev = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        match dev.retained_graph.as_ref() {
            Some(r) if r.fingerprint == key => Some((r.command_list.clone(), r.slot_idx)),
            _ => None,
        }
    };

    let Some((command_list, slot_idx)) = retained else {
        return Ok(None);
    };

    let fence_value = {
        let dev = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;
        let token = dev.fence_value;
        dev.fence_value += 1;
        token
    };

    let cmd_list: ID3D12CommandList = command_list
        .cast()
        .context("Failed to cast retained command list")?;

    let dev = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    {
        let _tz = tracy_zone!("dx12.resubmit_retained");
        unsafe {
            dev.command_queue.ExecuteCommandLists(&[Some(cmd_list)]);
        }
        unsafe { dev.command_queue.Signal(&dev.fence, fence_value) }
            .context("Failed to signal fence after retained resubmit")?;
    }

    if let Some(dev) = state.devices.get_mut(&device_handle) {
        // Update the retained slot's fence_value so the GPU timeline stays accurate.
        if let Some(slot) = dev.compute_allocator_pool.get_mut(slot_idx) {
            slot.fence_value = fence_value;
        }
        dev.process_deletion_queue();
    }

    Ok(Some(fence_value))
}

/// Drop the retained command list for `key`, marking its pool slot as reusable.
pub(super) fn evict_retained(state: &mut Dx12State, device_handle: DeviceHandle) {
    if let Some(dev) = state.devices.get_mut(&device_handle) {
        if let Some(old) = dev.retained_graph.take() {
            if let Some(slot) = dev.compute_allocator_pool.get_mut(old.slot_idx) {
                slot.retained = false;
            }
        }
    }
}

/// Check if the fence for the given token has signaled.
#[allow(
    dead_code,
    reason = "retained for deprecated fence-based paths; timeline uses fence internally"
)]
pub(super) fn is_fence_complete(
    state: &Dx12State,
    device_handle: DeviceHandle,
    token: TimelineValue,
) -> bool {
    let logical_device = match state.devices.get(&device_handle) {
        Some(dev) => dev,
        None => return false,
    };
    (unsafe { logical_device.fence.GetCompletedValue() }) >= token
}

/// Block until the fence signals.
#[allow(dead_code, reason = "retained for deprecated fence-based paths")]
pub(super) fn wait_fence(
    state: &Dx12State,
    device_handle: DeviceHandle,
    token: TimelineValue,
) -> Result<()> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    super::utils::wait_for_fence(&logical_device.fence, token)?;

    // Detect TDR: device removal signals all fences with UINT64_MAX
    let completed = unsafe { logical_device.fence.GetCompletedValue() };
    if completed == u64::MAX {
        let reason = unsafe { logical_device.device.GetDeviceRemovedReason() };
        anyhow::bail!("GPU device removed (TDR) after fence wait: {:?}", reason);
    }
    Ok(())
}

/// Wait with timeout. Returns Ok(true) if signaled, Ok(false) if timeout elapsed.
#[allow(dead_code, reason = "retained for deprecated fence-based paths")]
pub(super) fn wait_fence_timeout(
    state: &Dx12State,
    device_handle: DeviceHandle,
    token: TimelineValue,
    timeout_ms: u32,
) -> Result<bool> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    super::utils::wait_for_fence_timeout(&logical_device.fence, token, timeout_ms)
}
