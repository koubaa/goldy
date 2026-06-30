//! Compute pipeline and dispatch logic.

use super::super::shared;
use super::super::shared::{PushLayout, DISPATCH_BATCH_STRIDE};
use super::barriers;
use super::pso_cache;
use super::shader;
use super::submit_session::{record_state_from_backend, Dx12SubmitScope};
use super::types::{self, ComputeAllocatorSlot, ComputePipelineState, DeferredSlot, Dx12State};
use super::{ComputePipelineHandle, ContextHandle, DeviceHandle, RenderTargetHandle, ShaderHandle};
use crate::backend::submission_worker::allocate_timeline_value;
use crate::backend::{GpuCommand, GraphCommand, RenderCommand, SubmitSync};
use crate::timeline::TimelineValue;
use crate::tracy_zone;
use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use crate::task_graph::{NodeAccessUnion, SlotUsageSet, UsageKindFlags};
use crate::types::ResourceCategory;

/// WARP's buffer state tracker corrupts with precise enhanced/global barriers that
/// name UAV/SRV access (see `buffer.rs` upload path). Use ALL/ALL/COMMON/COMMON.
fn warp_full_global_barrier() -> D3D12_GLOBAL_BARRIER {
    D3D12_GLOBAL_BARRIER {
        SyncBefore: D3D12_BARRIER_SYNC_ALL,
        SyncAfter: D3D12_BARRIER_SYNC_ALL,
        AccessBefore: D3D12_BARRIER_ACCESS_COMMON,
        AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
    }
}

fn buffer_stride_for_bindless_index(
    buffers: &std::collections::HashMap<super::BufferHandle, types::BufferState>,
    device_handle: DeviceHandle,
    index: u32,
    cat: ResourceCategory,
) -> Option<u32> {
    // DX12 uses a process-wide backend singleton; bindless indices are per-device heap
    // offsets, so parallel tests must not resolve strides from another device's buffers.
    for b in buffers.values() {
        if b.device_handle != device_handle {
            continue;
        }
        match cat {
            ResourceCategory::Scattered
                if b.is_storage && (b.bindless_offset == Some(index) || b.bindless_srv_offset == Some(index)) =>
            {
                return b.element_stride;
            }
            ResourceCategory::Broadcast if !b.is_storage && b.bindless_offset == Some(index) => {
                return b.element_stride;
            }
            _ => {}
        }
    }
    None
}

/// Collect bindless heap indices referenced by a flat GPU command stream.
fn collect_bindless_slots_from_gpu_commands(
    commands: &[GpuCommand],
    _buffers: &std::collections::HashMap<super::BufferHandle, types::BufferState>,
) -> Vec<DeferredSlot> {
    let mut slots = Vec::new();
    for cmd in commands {
        match cmd {
            GpuCommand::BindResourcesRaw { indices, .. } => {
                slots.extend(indices.iter().copied().map(DeferredSlot::CbvSrvUav));
            }
            GpuCommand::BindResourcesTyped { handles } => {
                slots.extend(handles.iter().map(|h| DeferredSlot::CbvSrvUav(h.index())));
            }
            GpuCommand::DispatchBatch { arg_data, count, .. } => {
                let layout_size = std::mem::size_of::<PushLayout>();
                for i in 0..*count as usize {
                    let base = i * DISPATCH_BATCH_STRIDE;
                    if base + layout_size <= arg_data.len() {
                        let layout: &PushLayout = bytemuck::from_bytes(&arg_data[base..base + layout_size]);
                        // Skip zero entries: PushLayout::bindless is a fixed [u16; N]
                        // array default-initialised to 0. Positions the caller did not
                        // fill remain 0 and do not correspond to an actual binding.
                        // Tracking them would create spurious slot_last_seen[0] entries
                        // on every batch submit, causing CbvSrvUav(0) reclamation to
                        // wait for unrelated contexts. BindResourcesRaw/Typed are not
                        // filtered because those vecs contain only the slots the caller
                        // explicitly provided.
                        for &idx in &layout.bindless {
                            if idx != 0 {
                                slots.push(DeferredSlot::CbvSrvUav(idx as u32));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    slots
}

/// Collect bindless heap indices from a mixed compute/render graph submission.
fn collect_bindless_slots_from_graph_commands(
    commands: &[GraphCommand],
    buffers: &std::collections::HashMap<super::BufferHandle, types::BufferState>,
) -> Vec<DeferredSlot> {
    let mut slots = Vec::new();
    for gc in commands {
        match gc {
            GraphCommand::Compute(cmd) => {
                slots.extend(collect_bindless_slots_from_gpu_commands(
                    std::slice::from_ref(cmd),
                    buffers,
                ));
            }
            GraphCommand::Render {
                commands: render_cmds, ..
            } => {
                for rc in render_cmds {
                    match rc {
                        RenderCommand::BindResources { buffers: buf_handles } => {
                            for h in buf_handles {
                                if let Some(offset) = buffers.get(h).and_then(|b| b.bindless_offset) {
                                    slots.push(DeferredSlot::CbvSrvUav(offset));
                                }
                            }
                        }
                        RenderCommand::BindResourcesRaw { indices, .. } => {
                            slots.extend(indices.iter().copied().map(DeferredSlot::CbvSrvUav));
                        }
                        RenderCommand::BindResourcesTyped { handles } => {
                            slots.extend(handles.iter().map(|h| DeferredSlot::CbvSrvUav(h.index())));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    slots
}

/// Convert the Koubaa-level producer/consumer kind set to a DX12 sync scope.
///
/// Returns `D3D12_BARRIER_SYNC_ALL` if no kinds are recorded (empty set) so
/// that the barrier is conservatively correct even without IR information.
///
/// When `for_buffer` is true, RENDER maps to `VERTEX_SHADING | PIXEL_SHADING`
/// (shader stages that can read a buffer in a render pass) rather than the
/// texture-only `RENDER_TARGET | DEPTH_STENCIL` stages.
/// Lower Koubaa slot usage to a DX12 sync-stage mask for a **buffer** barrier.
///
/// `is_storage` is whether the buffer was created with
/// `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`. Non-storage (upload) buffers
/// are never written by the GPU, so transfer/compute *writes* against them are
/// nonsensical; we still clamp the access mask in
/// [`slot_usage_to_dx12_access_for_buffer`] and keep the sync stages here valid.
fn slot_usage_to_dx12_sync(usage: &SlotUsageSet, is_storage: bool) -> D3D12_BARRIER_SYNC {
    if usage.kinds.is_empty() {
        return D3D12_BARRIER_SYNC_ALL;
    }
    let mut sync = D3D12_BARRIER_SYNC(0);
    if usage.kinds.contains(UsageKindFlags::COMPUTE) {
        sync.0 |= D3D12_BARRIER_SYNC_COMPUTE_SHADING.0;
    }
    if usage.kinds.contains(UsageKindFlags::TRANSFER) {
        // A non-storage upload buffer can only ever be a copy *source*, but the
        // sync stage (COPY) is identical either way.
        let _ = is_storage;
        sync.0 |= D3D12_BARRIER_SYNC_COPY.0;
    }
    if usage.kinds.contains(UsageKindFlags::RENDER) {
        // Buffer reads inside a render pass happen in vertex/pixel shader stages.
        sync.0 |= D3D12_BARRIER_SYNC_VERTEX_SHADING.0 | D3D12_BARRIER_SYNC_PIXEL_SHADING.0;
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
/// Lower Koubaa slot usage to DX12 access flags for a **buffer** barrier.
///
/// `is_storage` reflects whether the resource was created with
/// `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`. The D3D12 debug layer rejects
/// (ID 1332) any buffer barrier that names `UNORDERED_ACCESS` on a resource
/// without that flag, and similarly `COPY_DEST` is invalid for an upload-heap
/// buffer (which is read-only from the GPU's perspective). We therefore clamp
/// the access mask for non-storage buffers to read-only accesses (SRV / copy
/// source), which is the only thing the GPU can legally do to them.
fn slot_usage_to_dx12_access_for_buffer(usage: &SlotUsageSet, is_storage: bool) -> D3D12_BARRIER_ACCESS {
    if usage.kinds.is_empty() {
        return D3D12_BARRIER_ACCESS_COMMON;
    }
    let mut access = D3D12_BARRIER_ACCESS(0);
    if usage.kinds.contains(UsageKindFlags::COMPUTE) {
        if is_storage && usage.access == NodeAccessUnion::Write {
            access.0 |= D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0;
        } else {
            // Read-only compute binding, or a non-storage buffer that can only
            // ever be bound as an SRV.
            access.0 |= D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0;
        }
    }
    if usage.kinds.contains(UsageKindFlags::TRANSFER) {
        if usage.access == NodeAccessUnion::Write && is_storage {
            access.0 |= D3D12_BARRIER_ACCESS_COPY_DEST.0;
        } else {
            // Non-storage upload buffers can only ever be a copy source.
            access.0 |= D3D12_BARRIER_ACCESS_COPY_SOURCE.0;
        }
    }
    if usage.kinds.contains(UsageKindFlags::RENDER) {
        // Buffer read by vertex/pixel shader inside a render pass → SHADER_RESOURCE.
        access.0 |= D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0;
    }
    if access.0 == 0 {
        D3D12_BARRIER_ACCESS_COMMON
    } else {
        access
    }
}

fn texture_barrier_state_for_layout(
    layout: D3D12_BARRIER_LAYOUT,
) -> (D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS, D3D12_BARRIER_LAYOUT) {
    if layout == D3D12_BARRIER_LAYOUT_COPY_SOURCE {
        (D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, layout)
    } else if layout == D3D12_BARRIER_LAYOUT_COPY_DEST {
        (D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST, layout)
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
) -> (D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS, D3D12_BARRIER_LAYOUT) {
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
pub(super) struct Dx12GpuProfileResources {
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

fn dx12_collect_dispatch_labels_graph(commands: &[GraphCommand]) -> (usize, Vec<Option<&'static str>>) {
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
    unsafe { device.CreateQueryHeap(&heap_desc, &mut heap_opt) }.context("CreateQueryHeap for GOLDY_GPU_PROFILE")?;
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
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
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

pub(super) fn dx12_readback_gpu_profile(
    command_queue: &ID3D12CommandQueue,
    fence_value: u64,
    profile: Dx12GpuProfileResources,
) -> Result<()> {
    use crate::gpu_profiler::{self, DispatchGpuNs};

    let freq = unsafe { command_queue.GetTimestampFrequency() }.context("GetTimestampFrequency")?;

    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
    let no_read = D3D12_RANGE { Begin: 0, End: 0 };
    unsafe { profile.readback.Map(0, Some(&no_read), Some(&mut mapped)) }.context("Map gpu profile readback")?;

    let vals: Vec<u64> =
        unsafe { std::slice::from_raw_parts(mapped as *const u64, profile.query_count as usize).to_vec() };

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
            let desc = std::slice::from_raw_parts(msg.pDescription, msg.DescriptionByteLength.saturating_sub(1));
            let text = std::str::from_utf8(desc).unwrap_or("<non-utf8 description>");
            let severity = match msg.Severity {
                D3D12_MESSAGE_SEVERITY_CORRUPTION => "CORRUPTION",
                D3D12_MESSAGE_SEVERITY_ERROR => "ERROR",
                D3D12_MESSAGE_SEVERITY_WARNING => "WARNING",
                D3D12_MESSAGE_SEVERITY_INFO => "INFO",
                D3D12_MESSAGE_SEVERITY_MESSAGE => "MSG",
                _ => "?",
            };
            out.push_str(&format!("  [D3D12 {}] id={} {}\n", severity, msg.ID.0, text));
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
    let cs_bytecode = shader::ensure_stage_compiled(state, compute_shader, crate::slang::SlangStage::Compute)?;

    let shader_debug_name = format!("compute_shader#{compute_shader}");

    let key = pso_cache::compute_pso_key(&cs_bytecode);

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    // Use the shared bindless root signature from the device
    let root_signature = logical_device
        .bindless_root_signature
        .as_ref()
        .context("Bindless root signature not available")?
        .clone();

    tracing::debug!("Using shared bindless root signature for compute pipeline");

    let pso_cache_arc = std::sync::Arc::clone(&logical_device.pso_cache);
    let disk_blob_bytes: Option<Vec<u8>> = pso_cache_arc.read().unwrap().compute_blobs.get(&key).cloned();
    let mut try_drop_stale_cached_blob = disk_blob_bytes.is_some();
    let cached_pso = disk_blob_bytes
        .as_ref()
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
                    let mut cache = pso_cache_arc.write().unwrap();
                    cache.compute_blobs.remove(&key);
                    cache.dirty = true;
                    drop(cache);
                    pso_desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
                    try_drop_stale_cached_blob = false;
                }
                Err(e) => anyhow::bail!("Failed to create compute pipeline state: {:?}", e),
            }
        }
    };

    let blob = unsafe { pipeline_state.GetCachedBlob().context("GetCachedBlob (compute PSO)")? };
    let new_blob = unsafe { pso_cache::id3dblob_to_vec(&blob) };

    {
        let mut cache = pso_cache_arc.write().unwrap();
        match cache.compute_blobs.get(&key) {
            Some(prev) if *prev == new_blob => {}
            _ => {
                cache.compute_blobs.insert(key, new_blob);
                cache.dirty = true;
            }
        }
    }

    let handle = state.compute_pipelines.write().unwrap().alloc_handle();

    let (cats, slot_kinds, strides) = state
        .shaders
        .read()
        .unwrap()
        .entries
        .get(&compute_shader)
        .and_then(|s| s.reflection.as_ref())
        .map(|r| {
            (
                r.push_constant_categories.clone(),
                r.push_constant_slot_kinds.clone(),
                r.binding_element_strides.clone(),
            )
        })
        .unwrap_or_default();

    state.compute_pipelines.write().unwrap().entries.insert(
        handle,
        ComputePipelineState {
            device_handle,
            pipeline_state,
            root_signature,
            parameter_block_layouts: Vec::new(),
            push_constant_categories: cats,
            push_constant_slot_kinds: slot_kinds,
            binding_element_strides: strides,
            shader_debug_name,
        },
    );

    tracing::debug!("Created compute pipeline {}", handle);
    Ok(handle)
}

/// Destroy a compute pipeline.
pub(super) fn destroy(state: &mut Dx12State, pipeline_handle: ComputePipelineHandle) {
    state
        .compute_pipelines
        .write()
        .unwrap()
        .entries
        .remove(&pipeline_handle);
}

// ---------------------------------------------------------------------------
// Shared submit helpers
// ---------------------------------------------------------------------------

/// Latest device-global seq retired on the scope's device.
fn device_retired_for_scope(scope: &Dx12SubmitScope<'_>) -> u64 {
    let device = scope.device_handle;
    let floor = scope.ld().retired_floor.load(std::sync::atomic::Ordering::Relaxed);
    let fences = scope.context_fences.read().unwrap();
    let max_ctx = fences
        .values()
        .filter(|(dev, _)| *dev == device)
        .map(|(_, fence)| unsafe { fence.GetCompletedValue() })
        .max()
        .unwrap_or(0);
    drop(fences);
    let device_sync = unsafe { scope.ld().fence.GetCompletedValue() };
    floor.max(max_ctx).max(device_sync)
}

pub(super) fn scope_from_state(state: &Dx12State, ctx: ContextHandle) -> Result<Dx12SubmitScope<'_>> {
    let sc = std::sync::Arc::clone(
        state
            .contexts
            .read()
            .unwrap()
            .get(&ctx)
            .with_context(|| format!("Invalid context handle {ctx}"))?,
    );
    let device_handle = sc.lock().unwrap().device;
    let record = record_state_from_backend(state, device_handle)?;
    let use_global_buffer_barriers = record.ld.adapter_id == super::WARP_ADAPTER_ID;
    Ok(Dx12SubmitScope {
        ctx,
        device_handle,
        sc,
        record,
        context_fences: &state.context_fences,
        use_global_buffer_barriers,
    })
}

/// Acquire (or create) a compute allocator slot.
///
/// Returns `(command_list, slot_idx)`.  The slot is taken from the
/// pool when its fence has already signalled; otherwise a fresh one is created.
fn acquire_allocator_slot(scope: &Dx12SubmitScope<'_>) -> Result<(ID3D12GraphicsCommandList, usize)> {
    let _device_handle = scope.device_handle;
    let logical_device = scope.ld();

    let ctx_fence = scope
        .context_fences
        .read()
        .unwrap()
        .get(&scope.ctx)
        .context("Invalid context handle")?
        .1
        .clone();
    let completed = unsafe { ctx_fence.GetCompletedValue() };
    let mut sc = scope.sc.lock().unwrap();
    let pool = &mut sc.compute_allocator_pool;
    // Skip retained slots — their allocator must not be reset until evict_retained is called.
    let slot_idx = pool.iter().position(|s| completed >= s.fence_value && !s.retained);
    let (cmd_list, slot_idx) = if let Some(idx) = slot_idx {
        let slot = &mut pool[idx];
        unsafe { slot.allocator.Reset() }.context("Failed to reset command allocator")?;
        let list = if let Some(ref existing) = slot.command_list {
            unsafe { existing.Reset(&slot.allocator, None) }.context("Failed to reset command list")?;
            existing.clone()
        } else {
            let new_list: ID3D12GraphicsCommandList = unsafe {
                logical_device
                    .device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &slot.allocator, None)
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
            logical_device
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &new_allocator, None)
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
    Ok((cmd_list, slot_idx))
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
    pending_deletions: Vec<super::types::PendingDeletion>,
}

/// Emit one `GpuCommand` onto the open command list in `ctx`.
///
/// Some arms take `textures.write()` to update [`TextureState::last_layout`]. That is
/// safe without per-context layout tracking: parcel exclusivity prevents two contexts
/// from recording against the same texture concurrently (see the field doc on
/// `last_layout`).
#[allow(clippy::too_many_lines)]
fn record_gpu_command(
    scope: &Dx12SubmitScope<'_>,
    device_handle: DeviceHandle,
    _ctx_handle: super::ContextHandle,
    ctx: &mut CmdCtx<'_>,
    cmd: &GpuCommand,
) -> Result<()> {
    let cl = ctx.command_list;
    let cl7 = ctx.command_list7;
    match cmd {
        GpuCommand::FrameTableStaging { data } => {
            super::frame_table::record_prologue(
                scope.contexts(),
                scope.frame_table(),
                &scope.buffers().read().unwrap().entries,
                device_handle,
                cl7,
                data,
            )?;
        }
        GpuCommand::SetPipeline(handle) => {
            let _tz = tracy_zone!("dx12.set_pipeline");

            ctx.current_compute_pipeline = Some(*handle);
            // This optimization was attempted but led to a slight performance regression
            // It seems to be related to latency hiding - changing the root signatures
            // caused the driver to warm up the GPU while the barrier drained.
            // This may be driver-specific, but kernel fusion (a future goldy optimization)
            // will render this kind of optimization moot - so we can do the conservative
            // thing for now.

            /*let pipeline_changed = ctx.current_compute_pipeline != Some(*handle);
            if pipeline_changed {
                {
                    let compute_pipelines_read = scope.compute_pipelines().read().unwrap();
                    if let Some(pipeline_state) = compute_pipelines_read.entries.get(handle) {
                    unsafe {
                        cl.SetComputeRootSignature(&pipeline_state.root_signature);
                        cl.SetPipelineState(&pipeline_state.pipeline_state);
                    }
                }
            }*/
            {
                let compute_pipelines_read = scope.compute_pipelines().read().unwrap();
                if let Some(pipeline_state) = compute_pipelines_read.entries.get(handle) {
                    unsafe {
                        cl.SetComputeRootSignature(&pipeline_state.root_signature);
                        cl.SetPipelineState(&pipeline_state.pipeline_state);
                    }
                }
            }
        }
        GpuCommand::BindResourcesRaw {
            indices: raw_indices,
            user: raw_user,
            frame_table_base,
        } => {
            let pipelines_read = scope.compute_pipelines().read().unwrap();
            if let Some(h) = ctx.current_compute_pipeline {
                if let Some(pipeline) = pipelines_read.entries.get(&h) {
                    crate::backend::with_layout_validation(|| {
                        crate::backend::validate_raw_binding_strides(
                            raw_indices,
                            &pipeline.push_constant_categories,
                            &pipeline.binding_element_strides,
                            |idx, cat| {
                                buffer_stride_for_bindless_index(
                                    &scope.buffers().read().unwrap().entries,
                                    device_handle,
                                    idx,
                                    cat,
                                )
                            },
                            &pipeline.shader_debug_name,
                        )?;
                        crate::backend::validate_bindless_slot_kinds(
                            raw_indices,
                            &pipeline.push_constant_slot_kinds,
                            |idx| {
                                super::buffer::bindless_slot_kind_for_index(
                                    &scope.buffers().read().unwrap().entries,
                                    device_handle,
                                    idx,
                                )
                            },
                            &pipeline.shader_debug_name,
                        )
                    })?;
                }
            }
            let mut layout = types::PushLayout::default();
            shared::fill_frame_table_dispatch(&mut layout, *frame_table_base, raw_user);
            unsafe {
                cl.SetComputeRoot32BitConstants(
                    0,
                    (types::TOTAL_PUSH_BYTES / 4) as u32,
                    &layout as *const _ as *const std::ffi::c_void,
                    0,
                );
            }
        }
        GpuCommand::BindResourcesTyped { handles: typed_handles } => {
            let pipelines_read = scope.compute_pipelines().read().unwrap();
            if let Some(h) = ctx.current_compute_pipeline {
                if let Some(pipeline) = pipelines_read.entries.get(&h) {
                    crate::backend::validate_typed_push_constants(
                        typed_handles,
                        &pipeline.push_constant_categories,
                        &pipeline.shader_debug_name,
                    )?;
                    let indices: Vec<u32> = typed_handles.iter().map(|h| h.index()).collect();
                    crate::backend::with_layout_validation(|| {
                        crate::backend::validate_bindless_slot_kinds(
                            &indices,
                            &pipeline.push_constant_slot_kinds,
                            |idx| {
                                super::buffer::bindless_slot_kind_for_index(
                                    &scope.buffers().read().unwrap().entries,
                                    device_handle,
                                    idx,
                                )
                            },
                            &pipeline.shader_debug_name,
                        )
                    })?;
                }
            }
            anyhow::bail!(
                "GpuCommand::BindResourcesTyped must be lowered to BindResourcesRaw before DX12 record; \
                 call frame_table::lower_gpu_commands first"
            );
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
            let logical_device = scope
                .devices()
                .get(&device_handle)
                .context("DispatchIndirect: invalid device")?;
            let buffers_read = scope.buffers().read().unwrap();
            let buf_state = buffers_read
                .entries
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
            let logical_device = scope
                .devices()
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
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
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
                    std::ptr::copy_nonoverlapping(arg_data.as_ptr(), mapped as *mut u8, arg_data.len());
                    let written = D3D12_RANGE {
                        Begin: 0,
                        End: arg_data.len(),
                    };
                    arg_resource.Unmap(0, Some(&written));
                }
                unsafe {
                    cl.ExecuteIndirect(&batch_sig, *count, &arg_resource, 0, None, 0);
                }

                ctx.pending_deletions
                    .push(super::types::PendingDeletion::StandaloneResource(arg_resource));
            } else {
                use crate::backend::shared::{PushLayout, DISPATCH_BATCH_STRIDE};
                let stride = DISPATCH_BATCH_STRIDE;
                let push_size = std::mem::size_of::<PushLayout>();
                for i in 0..*count as usize {
                    let base = i * stride;
                    let layout_bytes = &arg_data[base..base + push_size];
                    let wg_off = base + push_size;
                    let wg_x = u32::from_ne_bytes(arg_data[wg_off..wg_off + 4].try_into().unwrap());
                    let wg_y = u32::from_ne_bytes(arg_data[wg_off + 4..wg_off + 8].try_into().unwrap());
                    let wg_z = u32::from_ne_bytes(arg_data[wg_off + 8..wg_off + 12].try_into().unwrap());
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
            buffers: buf_entries,
            textures: tex_entries,
        } => {
            let _tz = tracy_zone!("dx12.resource_barrier");

            // WARP silently removes the device when D3D12_BARRIER_TYPE_BUFFER
            // enhanced barriers are used, causing ExecuteCommandLists to fail
            // and the subsequent Signal() to AV. Fall back to a global barrier
            // which is correct (just less precise).
            let mut buf_barriers: Vec<D3D12_BUFFER_BARRIER> = Vec::new();
            if ctx.use_global_buffer_barriers {
                for (h, usage) in buf_entries {
                    // Clamp access against the resource's real capabilities: a
                    // non-storage (upload) buffer can never carry UAV/COPY_DEST.
                    let buffers_read = scope.buffers().read().unwrap();
                    let is_storage = buffers_read.entries.get(h).map(|bs| bs.is_storage).unwrap_or(false);
                    let access_before = slot_usage_to_dx12_access_for_buffer(&usage.src, is_storage);
                    let mut access_after = slot_usage_to_dx12_access_for_buffer(&usage.dst, is_storage);
                    // WARP validation (ID 1331): global barriers with AccessBefore=COMMON
                    // must use AccessAfter=COMMON. Empty producer usage sets COMMON; clamp
                    // rather than emitting UAV/SRV which fails Close() on the debug layer.
                    if access_before == D3D12_BARRIER_ACCESS_COMMON {
                        access_after = D3D12_BARRIER_ACCESS_COMMON;
                    }
                    let g = if is_storage {
                        warp_full_global_barrier()
                    } else {
                        D3D12_GLOBAL_BARRIER {
                            SyncBefore: slot_usage_to_dx12_sync(&usage.src, is_storage),
                            SyncAfter: slot_usage_to_dx12_sync(&usage.dst, is_storage),
                            AccessBefore: access_before,
                            AccessAfter: access_after,
                        }
                    };
                    unsafe { barriers::barrier_globals(cl7, &[g]) };
                }
            } else {
                buf_barriers = buf_entries
                    .iter()
                    .filter_map(|(h, usage)| {
                        scope.buffers().read().unwrap().entries.get(h).map(|bs| {
                            barriers::buffer_barrier_full(
                                &bs.resource,
                                slot_usage_to_dx12_sync(&usage.src, bs.is_storage),
                                slot_usage_to_dx12_sync(&usage.dst, bs.is_storage),
                                slot_usage_to_dx12_access_for_buffer(&usage.src, bs.is_storage),
                                slot_usage_to_dx12_access_for_buffer(&usage.dst, bs.is_storage),
                            )
                        })
                    })
                    .collect();
            }

            let mut tex_barriers: Vec<D3D12_TEXTURE_BARRIER> = tex_entries
                .iter()
                .filter_map(|(h, usage)| {
                    scope.textures().read().unwrap().entries.get(h).map(|ts| {
                        let (tex_sync_after, tex_access_after, tex_layout_after) =
                            texture_barrier_state_for_usage(&usage.dst, ts.is_storage);
                        let (tex_sync_before, tex_access_before, tex_layout_before) =
                            texture_barrier_state_for_layout(ts.last_layout);
                        (
                            barriers::texture_barrier_full(
                                &ts.resource,
                                tex_sync_before,
                                tex_sync_after,
                                tex_access_before,
                                tex_access_after,
                                tex_layout_before,
                                tex_layout_after,
                            ),
                            tex_layout_after,
                        )
                    })
                })
                .map(|(b, _)| b)
                .collect();

            if !ctx.use_global_buffer_barriers {
                unsafe { barriers::barrier_groups(cl7, &buf_barriers, &tex_barriers) };
                unsafe { barriers::drop_buffer_barriers(&mut buf_barriers) };
            } else if !tex_barriers.is_empty() {
                unsafe { barriers::barrier_textures(cl7, &tex_barriers) };
            }
            unsafe { barriers::drop_texture_barriers(&mut tex_barriers) };

            for (h, usage) in tex_entries {
                {
                    let mut textures_write = scope.textures().write().unwrap();
                    if let Some(ts) = textures_write.entries.get_mut(h) {
                        let (_, _, tex_layout_after) = texture_barrier_state_for_usage(&usage.dst, ts.is_storage);
                        ts.last_layout = tex_layout_after;
                    }
                }
            }
        }
        GpuCommand::ClearBuffer { buffer, offset, size } => {
            let _tz = tracy_zone!("dx12.clear_buffer");
            let buffers_read = scope.buffers().read().unwrap();
            let buf_state = buffers_read
                .entries
                .get(buffer)
                .context("ClearBuffer: invalid buffer handle")?;
            let clear_size = if *size == 0 {
                buf_state.size.saturating_sub(*offset)
            } else {
                *size
            };
            if clear_size > 0 {
                if buf_state.is_storage {
                    let logical_device = scope
                        .devices()
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
                        let this_chunk = (clear_size - cleared).min(super::buffer::ZERO_BUFFER_SIZE);
                        unsafe {
                            cl.CopyBufferRegion(&buf_resource, *offset + cleared, &zero, 0, this_chunk);
                        }
                        cleared += this_chunk;
                    }
                } else {
                    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
                    let no_read = D3D12_RANGE { Begin: 0, End: 0 };
                    unsafe { buf_state.resource.Map(0, Some(&no_read), Some(&mut mapped)) }
                        .context("ClearBuffer: failed to map buffer")?;
                    unsafe {
                        std::ptr::write_bytes((mapped as *mut u8).add(*offset as usize), 0, clear_size as usize);
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
            let buffers_read = scope.buffers().read().unwrap();
            let buf_state = buffers_read
                .entries
                .get(buf_handle)
                .context("WriteBuffer: invalid buffer handle")?;
            if !buf_state.is_storage {
                let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
                let no_read = D3D12_RANGE { Begin: 0, End: 0 };
                unsafe { buf_state.resource.Map(0, Some(&no_read), Some(&mut mapped)) }
                    .context("WriteBuffer: map failed")?;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), (mapped as *mut u8).add(*offset as usize), data.len());
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
                let buffers_read = scope.buffers().read().unwrap();
                let buf_state = buffers_read.entries.get(buf_handle).unwrap();

                if ctx.use_global_buffer_barriers {
                    let pre = D3D12_GLOBAL_BARRIER {
                        SyncBefore: D3D12_BARRIER_SYNC_ALL,
                        SyncAfter: D3D12_BARRIER_SYNC_COPY,
                        AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        AccessAfter: D3D12_BARRIER_ACCESS_COPY_DEST,
                    };
                    unsafe {
                        barriers::barrier_globals(cl7, &[pre]);
                        cl.CopyBufferRegion(&buf_state.resource, *offset, &upload_src, upload_off, data.len() as u64);
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
                        cl.CopyBufferRegion(&buf_state.resource, *offset, &upload_src, upload_off, data.len() as u64);
                    }
                }
            }
        }
        GpuCommand::WriteTexture { .. }
        | GpuCommand::WriteTextureRegion { .. }
        | GpuCommand::CopyBufferToTexture { .. } => {
            let _tz = tracy_zone!("dx12.write_texture");
            let upload = ctx
                .staged_texture_uploads
                .get(ctx.texture_upload_idx)
                .context("WriteTexture: staged upload missing (internal)")?;
            ctx.texture_upload_idx += 1;
            super::texture::record_staged_texture_upload(
                cl,
                cl7,
                &mut scope.textures().write().unwrap().entries,
                upload,
            )?;
        }
        GpuCommand::CopyTexture { src, dst } => {
            let _tz = tracy_zone!("dx12.copy_texture");
            let (src_res, src_layout, src_is_storage) = {
                let textures_read = scope.textures().read().unwrap();
                let ts = textures_read
                    .entries
                    .get(src)
                    .context("CopyTexture: src texture not found")?;
                (ts.resource.clone(), ts.last_layout, ts.is_storage)
            };
            let (dst_res, dst_layout, dst_is_storage) = {
                let textures_read = scope.textures().read().unwrap();
                let ts = textures_read
                    .entries
                    .get(dst)
                    .context("CopyTexture: dst texture not found")?;
                (ts.resource.clone(), ts.last_layout, ts.is_storage)
            };

            let (src_sync_before, src_access_before, src_layout_before) = texture_barrier_state_for_layout(src_layout);
            let (dst_sync_before, dst_access_before, dst_layout_before) = texture_barrier_state_for_layout(dst_layout);

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

            {
                let mut textures_write = scope.textures().write().unwrap();
                if let Some(ts) = textures_write.entries.get_mut(src) {
                    ts.last_layout = src_post_state.1;
                }
                if let Some(ts) = textures_write.entries.get_mut(dst) {
                    ts.last_layout = dst_post_state.1;
                }
            }
        }
        GpuCommand::CopyBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        } => {
            let _tz = tracy_zone!("dx12.copy_buffer");
            let (src_resource, dst_resource, src_off, dst_off, src_is_upload) = {
                let buffers_read = scope.buffers().read().unwrap();
                let src_buf = buffers_read.entries.get(src).context("CopyBuffer: invalid src")?;
                let dst_buf = buffers_read.entries.get(dst).context("CopyBuffer: invalid dst")?;
                if src_offset.saturating_add(*size) > src_buf.size || dst_offset.saturating_add(*size) > dst_buf.size {
                    anyhow::bail!("CopyBuffer: size exceeds buffer bounds");
                }
                let src_is_upload = src_buf.flags.contains(crate::types::BufferFlags::CPU_WRITABLE);
                let src_resource = if src_is_upload {
                    src_buf
                        .upload_buffer
                        .clone()
                        .context("CopyBuffer: CPU_WRITABLE src missing upload buffer")?
                } else {
                    src_buf.resource.clone()
                };
                (
                    src_resource,
                    dst_buf.resource.clone(),
                    *src_offset,
                    *dst_offset,
                    src_is_upload,
                )
            };
            let src_access_before = if src_is_upload {
                D3D12_BARRIER_ACCESS_COPY_SOURCE
            } else {
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS
            };
            if ctx.use_global_buffer_barriers {
                let pre = D3D12_GLOBAL_BARRIER {
                    SyncBefore: D3D12_BARRIER_SYNC_ALL,
                    SyncAfter: D3D12_BARRIER_SYNC_COPY,
                    AccessBefore: src_access_before,
                    AccessAfter: D3D12_BARRIER_ACCESS_COPY_SOURCE,
                };
                unsafe { barriers::barrier_globals(cl7, &[pre]) };
            } else {
                let mut b_to_copy = [barriers::buffer_barrier_full(
                    &src_resource,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_SYNC_COPY,
                    src_access_before,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                )];
                unsafe {
                    barriers::barrier_buffers(cl7, &b_to_copy);
                    barriers::drop_buffer_barriers(&mut b_to_copy);
                }
            }
            unsafe { cl.CopyBufferRegion(&dst_resource, dst_off, &src_resource, src_off, *size) };
        }
        GpuCommand::CopyTextureToReadback { src, dst, layout } => {
            let _tz = tracy_zone!("dx12.copy_texture_to_readback");
            super::texture::record_copy_texture_to_readback(
                cl,
                cl7,
                &mut scope.textures().write().unwrap().entries,
                &scope.buffers().read().unwrap().entries,
                *src,
                *dst,
                *layout,
            )?;
        }
        GpuCommand::CopyRenderTarget { src, dst } => {
            let _tz = tracy_zone!("dx12.copy_render_target");
            let src_res = {
                let render_targets_read = scope.render_targets().read().unwrap();
                let rt = render_targets_read
                    .entries
                    .get(src)
                    .context("CopyRenderTarget: src render target not found")?;
                rt.texture.clone()
            };
            let (dst_res, dst_layout, dst_is_storage) = {
                let textures_read = scope.textures().read().unwrap();
                let ts = textures_read
                    .entries
                    .get(dst)
                    .context("CopyRenderTarget: dst texture not found")?;
                (ts.resource.clone(), ts.last_layout, ts.is_storage)
            };

            let (dst_sync_before, dst_access_before, dst_layout_before) = texture_barrier_state_for_layout(dst_layout);

            let mut pre_barriers = vec![
                barriers::texture_barrier_full(
                    &src_res,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_LAYOUT_COPY_SOURCE,
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
                    D3D12_BARRIER_SYNC_RENDER_TARGET,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_ACCESS_RENDER_TARGET,
                    D3D12_BARRIER_LAYOUT_COPY_SOURCE,
                    D3D12_BARRIER_LAYOUT_RENDER_TARGET,
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

            {
                let mut textures_write = scope.textures().write().unwrap();
                if let Some(ts) = textures_write.entries.get_mut(dst) {
                    ts.last_layout = dst_post_state.1;
                }
            }
        }
    }
    Ok(())
}

struct SubmitFinish {
    ctx: ContextHandle,
    device_handle: DeviceHandle,
    slot_idx: usize,
    retain_key: Option<u64>,
    used_slots: Vec<DeferredSlot>,
    frame_table_staging: Option<std::sync::Arc<[u32]>>,
    pending_deletions: Vec<super::types::PendingDeletion>,
}

struct StagingFinish {
    texture_uploads: Vec<super::texture::StagedTextureUpload>,
    belt_slices_len: usize,
    belt_idx: usize,
}

/// Close the command list, execute, signal, update the pool slot, and finish staging.
///
/// `retain_key`: when `Some(k)`, stores the closed command list in
/// `Dx12SubmissionContext::retained_graphs` for zero-cost re-execution via
/// [`try_resubmit_retained`].
///
/// # Abandoned optimizations
///
/// Two approaches were attempted here to reduce per-frame CPU recording cost and were
/// reverted due to the reasons noted:
///
/// 1. **CBV binding table + fingerprint-based CL reuse**: replaced 128-byte root constants
///    with a persistently-mapped UPLOAD buffer (CBV at root param 1) and slot indices
///    (1-DWORD root param 0).  The binding table allowed resubmitting the same closed
///    command list across frames when `compute_retention_fingerprint` was stable.
///    Reverted: the required `wait_for_fence` after every `ExecuteCommandLists` to prevent
///    CPU/GPU races on the shared binding-table buffer cut throughput from ~2500 FPS to
///    ~1200 FPS — a net regression for the common case. Note: this may be obsolete, as
///    it was observed in an earlier design for binding tables.
///
/// 2. **CBV binding table + bind groups**: same binding-table layout as above, with
///    per-pipeline bind groups (descriptor-table caching) to amortise heap binding cost.
///    Reverted: DX12 bundles do not support `Dispatch`, and a descriptor-table approach
///    without bundles did not provide a clean enough win to justify the complexity.
fn execute_signal_and_finish(
    scope: &Dx12SubmitScope<'_>,
    command_list: &ID3D12GraphicsCommandList,
    gpu_profile: Option<Dx12GpuProfileResources>,
    submit: SubmitFinish,
    staging_finish: StagingFinish,
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    let SubmitFinish {
        ctx,
        device_handle,
        slot_idx,
        retain_key,
        used_slots,
        frame_table_staging,
        pending_deletions,
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
        let diag = scope
            .devices()
            .get(&device_handle)
            .and_then(|dev| drain_info_queue(&dev.device))
            .unwrap_or_else(|| "  (no debug-layer messages; enable GOLDY_DX12_DEBUG=1)\n".to_string());
        return Err(anyhow::anyhow!(
            "Failed to close command list: {e}\nDebug layer messages:\n{diag}"
        ));
    }

    let cmd_list: ID3D12CommandList = command_list.cast().context("Failed to cast command list")?;

    let logical_device = scope.ld();
    let fence_value = allocate_timeline_value(&logical_device.timeline_next);

    logical_device
        .descriptors
        .lock()
        .unwrap()
        .record_slot_usage(ctx, fence_value, used_slots.iter().copied());

    if !pending_deletions.is_empty() {
        let mut sc = scope.sc.lock().unwrap();
        for resource in pending_deletions {
            sc.deletion_queue.queue(fence_value, resource);
        }
    }

    {
        let mut sc = scope.sc.lock().unwrap();
        if let Some(slot) = sc.compute_allocator_pool.get_mut(slot_idx) {
            slot.fence_value = fence_value;
        }
        if let Some(key) = retain_key {
            if let Some(old) = sc.retained_graphs.remove(&key) {
                if let Some(row) = old.frame_table_row {
                    super::frame_table::unpin_row(scope.frame_table(), row);
                }
                if let Some(old_slot) = sc.compute_allocator_pool.get_mut(old.slot_idx) {
                    old_slot.retained = false;
                }
            }
            let frame_table_row = frame_table_staging
                .as_ref()
                .and_then(|_| super::frame_table::last_prologue_row(scope.frame_table()));
            if let Some(row) = frame_table_row {
                super::frame_table::pin_row(scope.frame_table(), row)?;
            }
            if let Some(cl) = sc
                .compute_allocator_pool
                .get(slot_idx)
                .and_then(|s| s.command_list.clone())
            {
                sc.compute_allocator_pool[slot_idx].retained = true;
                sc.retained_graphs.insert(
                    key,
                    types::RetainedGraph {
                        command_list: cl,
                        slot_idx,
                        used_slots,
                        frame_table_staging,
                        frame_table_row,
                    },
                );
            }
        }
        sc.last_submitted_seq = fence_value;
    }

    let ctx_fence = scope
        .context_fences
        .read()
        .unwrap()
        .get(&ctx)
        .context("Invalid context handle")?
        .1
        .clone();

    {
        let _tz = crate::tracy_zone!("goldy.submit.dx12.deletion_drain");
        let ctx_completed = unsafe { ctx_fence.GetCompletedValue() };
        let mut sc_guard = scope.sc.lock().unwrap();
        super::context::drain_context_deletion_queue_up_to(logical_device, &mut sc_guard, ctx_completed);
        super::context::drain_pending_gpu_profiles_up_to(logical_device, &mut sc_guard, ctx_completed);
    }

    let staged_texture_entries = staged_texture_uploads
        .into_iter()
        .filter_map(|u| {
            if let super::texture::TextureUploadSource::Pooled(entry) = u.source {
                Some(entry)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    super::pending_submit::enqueue_compute_submit(
        logical_device,
        scope.context_fences,
        ctx_fence,
        vec![Some(cmd_list)],
        sync,
        fence_value,
    )?;

    if let Some(prof) = gpu_profile {
        scope.sc.lock().unwrap().pending_gpu_profiles.push((fence_value, prof));
    }

    {
        let _tz = crate::tracy_zone!("goldy.submit.dx12.staging_finish");
        let mut sc = scope.sc.lock().unwrap();
        sc.staging_belt.finish(fence_value);
        if !staged_texture_entries.is_empty() {
            sc.texture_staging_pool.release(fence_value, staged_texture_entries);
        }
    }

    Ok(fence_value)
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Submit compute commands without blocking. Returns a fence token for polling/waiting.
pub(super) fn submit_with_scope(
    scope: &Dx12SubmitScope<'_>,
    ctx: ContextHandle,
    commands: &[GpuCommand],
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    let mut commands = commands.to_vec();
    crate::frame_table::lower_gpu_commands(&mut commands);
    let frame_table_staging = super::frame_table::extract_staging_from_commands(&commands);
    let device_handle = scope.device_handle;
    let _tz = tracy_zone!("dx12.submit");
    let (command_list, slot_idx) = {
        let _tz_acq = tracy_zone!("dx12.submit.acquire_allocator");
        acquire_allocator_slot(scope)?
    };

    let ctx_fence_clone = scope
        .context_fences
        .read()
        .unwrap()
        .get(&ctx)
        .context("Invalid context handle")?
        .1
        .clone();
    let use_global_buffer_barriers = scope.use_global_buffer_barriers;

    let has_upload = commands.iter().any(|c| {
        matches!(
            c,
            GpuCommand::WriteBuffer { .. }
                | GpuCommand::WriteTexture { .. }
                | GpuCommand::WriteTextureRegion { .. }
                | GpuCommand::CopyBufferToTexture { .. }
        )
    });
    if has_upload {
        let _tz_reclaim = tracy_zone!("dx12.submit.staging_reclaim");
        let completed = device_retired_for_scope(scope);
        let mut sc = scope.sc.lock().unwrap();
        sc.staging_belt.reclaim(&ctx_fence_clone)?;
        sc.texture_staging_pool.reclaim(completed);
    }

    let command_list7: ID3D12GraphicsCommandList7 =
        command_list.cast().context("ID3D12GraphicsCommandList7 required")?;

    let mut belt_slices: Vec<(ID3D12Resource, u64)> = Vec::new();
    let mut staged_texture_uploads: Vec<super::texture::StagedTextureUpload> = Vec::new();
    if has_upload {
        let mut pool = {
            let mut sc = scope.sc.lock().unwrap();
            std::mem::replace(&mut sc.texture_staging_pool, super::staging::TextureStagingPool::new())
        };

        let _tz_prepass = tracy_zone!("dx12.submit.upload_prepass");
        for command in &commands {
            match command {
                GpuCommand::WriteBuffer {
                    buffer: buf_handle,
                    data,
                    ..
                } => {
                    let buffers_read = scope.buffers().read().unwrap();
                    let buf = buffers_read
                        .entries
                        .get(buf_handle)
                        .context("WriteBuffer pre-pass: invalid handle")?;
                    if buf.is_storage {
                        let buf_dev = buf.device_handle;
                        let ld = scope
                            .devices()
                            .get(&buf_dev)
                            .context("WriteBuffer pre-pass: device missing")?;
                        let mut sc = scope.sc.lock().unwrap();
                        let (res, off) = sc.staging_belt.write(ld, data)?;
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
                        scope.devices(),
                        &scope.textures().read().unwrap().entries,
                        &mut pool,
                        *texture,
                        data.as_ref(),
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
                        scope.devices(),
                        &scope.textures().read().unwrap().entries,
                        &mut pool,
                        super::texture::TextureUploadRegion {
                            texture_handle: *texture,
                            x: *x,
                            y: *y,
                            width: *width,
                            height: *height,
                            data: data.as_ref(),
                        },
                    )?);
                }
                GpuCommand::CopyBufferToTexture {
                    src,
                    src_offset,
                    src_row_pitch,
                    dst,
                    x,
                    y,
                    width,
                    height,
                } => {
                    staged_texture_uploads.push(super::texture::stage_copy_buffer_to_texture_upload(
                        scope.devices(),
                        &scope.textures().read().unwrap().entries,
                        &scope.buffers().read().unwrap().entries,
                        &mut pool,
                        *src,
                        *src_offset,
                        *src_row_pitch,
                        *dst,
                        *x,
                        *y,
                        *width,
                        *height,
                    )?);
                }
                _ => {}
            }
        }

        scope.sc.lock().unwrap().texture_staging_pool = pool;
    }

    let mut dx_gpu_profile = {
        let _tz_gp = tracy_zone!("dx12.submit.gpu_profile_setup");
        let logical_device_ref = scope.ld();
        let (dispatch_count, dispatch_labels) = dx12_collect_dispatch_labels(&commands);
        let prof = match dx12_try_create_gpu_profile(&logical_device_ref.device, dispatch_count, dispatch_labels) {
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

    let (belt_idx_final, pending_deletions) = {
        let _tz_cmds = tracy_zone!("dx12.submit.record_commands");
        let mut cmd_ctx = CmdCtx {
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
            pending_deletions: Vec::new(),
        };
        for cmd in &commands {
            record_gpu_command(scope, device_handle, ctx, &mut cmd_ctx, cmd)?;
        }
        debug_assert_eq!(
            cmd_ctx.texture_upload_idx,
            staged_texture_uploads.len(),
            "WriteTexture command count mismatch vs staging pre-pass"
        );
        (cmd_ctx.belt_idx, cmd_ctx.pending_deletions)
    };

    // Tail barrier: make UAV and copy writes visible to subsequent operations.
    let tail = D3D12_GLOBAL_BARRIER {
        SyncBefore: D3D12_BARRIER_SYNC(D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0),
        SyncAfter: D3D12_BARRIER_SYNC_ALL,
        AccessBefore: D3D12_BARRIER_ACCESS(D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_COPY_DEST.0),
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

    let used_slots = collect_bindless_slots_from_gpu_commands(&commands, &scope.buffers().read().unwrap().entries);
    let tv = execute_signal_and_finish(
        scope,
        &command_list,
        dx_gpu_profile.take(),
        SubmitFinish {
            ctx,
            device_handle,
            slot_idx,
            retain_key: None,
            used_slots,
            frame_table_staging,
            pending_deletions,
        },
        StagingFinish {
            texture_uploads: staged_texture_uploads,
            belt_slices_len: belt_slices.len(),
            belt_idx: belt_idx_final,
        },
        sync,
    )?;
    Ok(tv)
}

pub(super) fn submit(
    state: &mut Dx12State,
    ctx: ContextHandle,
    commands: &[GpuCommand],
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    submit_with_scope(&scope_from_state(state, ctx)?, ctx, commands, sync)
}

/// Submit mixed compute + render graph commands in a single command list.
///
/// Eliminates CPU waits between compute and render segments by recording
/// everything into one `ID3D12GraphicsCommandList7` and performing a single
/// `ExecuteCommandLists` + `Signal(fence)` at the end.
///
/// When `retain_key` is `Some(k)`, the closed command list is stored in
/// `Dx12SubmissionContext::retained_graphs` keyed by `k` for future zero-cost re-execution
/// via [`try_resubmit_retained`].  Any previously retained graph for the same key is evicted first.
pub(super) fn submit_graph_with_scope(
    scope: &Dx12SubmitScope<'_>,
    ctx: ContextHandle,
    commands: &[GraphCommand],
    retain_key: Option<u64>,
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    let frame_table_staging = super::frame_table::extract_staging_from_graph(commands);
    let device_handle = scope.device_handle;
    let _tz = tracy_zone!("dx12.submit_graph");
    let (command_list, slot_idx) = acquire_allocator_slot(scope)?;

    let ctx_fence_clone = scope
        .context_fences
        .read()
        .unwrap()
        .get(&ctx)
        .context("Invalid context handle")?
        .1
        .clone();
    let use_global_buffer_barriers = scope.use_global_buffer_barriers;

    let has_upload = commands.iter().any(|c| {
        matches!(
            c,
            GraphCommand::Compute(
                GpuCommand::WriteBuffer { .. }
                    | GpuCommand::WriteTexture { .. }
                    | GpuCommand::WriteTextureRegion { .. }
                    | GpuCommand::CopyBufferToTexture { .. }
            )
        )
    });
    if has_upload {
        let completed = device_retired_for_scope(scope);
        let mut sc = scope.sc.lock().unwrap();
        sc.staging_belt.reclaim(&ctx_fence_clone)?;
        sc.texture_staging_pool.reclaim(completed);
    }

    let command_list7: ID3D12GraphicsCommandList7 =
        command_list.cast().context("ID3D12GraphicsCommandList7 required")?;

    let mut belt_slices: Vec<(ID3D12Resource, u64)> = Vec::new();
    let mut staged_texture_uploads: Vec<super::texture::StagedTextureUpload> = Vec::new();
    if has_upload {
        let mut pool = {
            let mut sc = scope.sc.lock().unwrap();
            std::mem::replace(&mut sc.texture_staging_pool, super::staging::TextureStagingPool::new())
        };

        let _tz_prepass = tracy_zone!("dx12.submit_graph.upload_prepass");
        for graph_cmd in commands {
            if let GraphCommand::Compute(gpu_cmd) = graph_cmd {
                match gpu_cmd {
                    GpuCommand::WriteBuffer {
                        buffer: buf_handle,
                        data,
                        ..
                    } => {
                        let buffers_read = scope.buffers().read().unwrap();
                        let buf = buffers_read
                            .entries
                            .get(buf_handle)
                            .context("WriteBuffer pre-pass: invalid handle")?;
                        if buf.is_storage {
                            let buf_dev = buf.device_handle;
                            let ld = scope
                                .devices()
                                .get(&buf_dev)
                                .context("WriteBuffer pre-pass: device missing")?;
                            let mut sc = scope.sc.lock().unwrap();
                            let (res, off) = sc.staging_belt.write(ld, data)?;
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
                            scope.devices(),
                            &scope.textures().read().unwrap().entries,
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
                            scope.devices(),
                            &scope.textures().read().unwrap().entries,
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
                    GpuCommand::CopyBufferToTexture {
                        src,
                        src_offset,
                        src_row_pitch,
                        dst,
                        x,
                        y,
                        width,
                        height,
                    } => {
                        staged_texture_uploads.push(super::texture::stage_copy_buffer_to_texture_upload(
                            scope.devices(),
                            &scope.textures().read().unwrap().entries,
                            &scope.buffers().read().unwrap().entries,
                            &mut pool,
                            *src,
                            *src_offset,
                            *src_row_pitch,
                            *dst,
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

        scope.sc.lock().unwrap().texture_staging_pool = pool;
    }

    let mut dx_gpu_profile = {
        let _tz_gp = tracy_zone!("dx12.submit_graph.gpu_profile_setup");
        let logical_device_ref = scope.ld();
        let (dispatch_count, dispatch_labels) = dx12_collect_dispatch_labels_graph(commands);
        let prof = match dx12_try_create_gpu_profile(&logical_device_ref.device, dispatch_count, dispatch_labels) {
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
    let (belt_idx_final, pending_deletions);
    {
        let _tz_cmds = tracy_zone!("dx12.submit_graph.record_commands");
        let mut cmd_ctx = CmdCtx {
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
            pending_deletions: Vec::new(),
        };

        let mut frame_table_prologue_in_cb = false;
        for graph_cmd in commands {
            match graph_cmd {
                GraphCommand::Compute(gpu_cmd) => {
                    if matches!(gpu_cmd, GpuCommand::FrameTableStaging { .. }) {
                        frame_table_prologue_in_cb = true;
                    }
                    record_gpu_command(scope, device_handle, ctx, &mut cmd_ctx, gpu_cmd)?;
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
                            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_COPY_DEST.0,
                        ),
                        AccessAfter: D3D12_BARRIER_ACCESS(
                            D3D12_BARRIER_ACCESS_RENDER_TARGET.0
                                | D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE.0
                                | D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0,
                        ),
                    };
                    let pre_render_barrier = if use_global_buffer_barriers {
                        warp_full_global_barrier()
                    } else {
                        compute_to_render
                    };
                    unsafe { barriers::barrier_globals(cmd_ctx.command_list7, &[pre_render_barrier]) };

                    let touched = super::render_target::record_render_pass_to_list_with_record(
                        &scope.record,
                        device_handle,
                        *target,
                        render_cmds,
                        &command_list7,
                        frame_table_prologue_in_cb,
                    )?;
                    frame_table_prologue_in_cb |= touched;
                    {
                        let render_targets_read = scope.render_targets().read().unwrap();
                        if let Some(rt) = render_targets_read.entries.get(target) {
                            if let Some(ref depth_res) = rt.depth_texture {
                                let depth_after = barriers::texture_barrier_full(
                                    depth_res,
                                    D3D12_BARRIER_SYNC_DEPTH_STENCIL,
                                    D3D12_BARRIER_SYNC_ALL,
                                    D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,
                                    D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ,
                                    D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
                                    D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ,
                                );
                                unsafe {
                                    barriers::barrier_textures(cmd_ctx.command_list7, &[depth_after]);
                                }
                            }
                        }
                    }
                    rendered_targets.push(*target);
                }
            }
        }

        debug_assert_eq!(
            cmd_ctx.texture_upload_idx,
            staged_texture_uploads.len(),
            "WriteTexture command count mismatch vs staging pre-pass"
        );
        belt_idx_final = cmd_ctx.belt_idx;
        pending_deletions = cmd_ctx.pending_deletions;
    }

    let tail = D3D12_GLOBAL_BARRIER {
        SyncBefore: D3D12_BARRIER_SYNC(D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0),
        SyncAfter: D3D12_BARRIER_SYNC_ALL,
        AccessBefore: D3D12_BARRIER_ACCESS(D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_COPY_DEST.0),
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

    let used_slots = collect_bindless_slots_from_graph_commands(commands, &scope.buffers().read().unwrap().entries);
    let result = execute_signal_and_finish(
        scope,
        &command_list,
        dx_gpu_profile.take(),
        SubmitFinish {
            ctx,
            device_handle,
            slot_idx,
            retain_key,
            used_slots,
            frame_table_staging,
            pending_deletions,
        },
        StagingFinish {
            texture_uploads: staged_texture_uploads,
            belt_slices_len: belt_slices.len(),
            belt_idx: belt_idx_final,
        },
        sync,
    )?;

    for t in rendered_targets {
        let mut render_targets_write = scope.render_targets().write().unwrap();
        if let Some(rt) = render_targets_write.entries.get_mut(&t) {
            rt.has_rendered = true;
        }
    }

    Ok(result)
}

pub(super) fn submit_graph(
    state: &mut Dx12State,
    ctx: ContextHandle,
    commands: &[GraphCommand],
    retain_key: Option<u64>,
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    submit_graph_with_scope(&scope_from_state(state, ctx)?, ctx, commands, retain_key, sync)
}

/// Re-execute a previously retained command list without re-recording.
///
/// Calls `ExecuteCommandLists` on the closed list stored by a prior
/// `submit_graph(..., Some(key))` call, then signals the device fence.
/// Returns `Ok(Some(tv))` on success, `Ok(None)` if no retained list matches `key`.
///
/// No CPU wait is required: the retained slot's allocator is not reset while in flight
/// (`acquire_allocator_slot` skips retained slots), and re-executing a closed list is legal.
pub(super) fn try_resubmit_retained_with_scope(
    scope: &Dx12SubmitScope<'_>,
    ctx: ContextHandle,
    key: u64,
    sync: Option<&SubmitSync>,
) -> Result<Option<TimelineValue>> {
    let _tz = tracy_zone!("dx12.resubmit_retained");
    let _device_handle = scope.device_handle;
    let retained = {
        let _tz_lookup = tracy_zone!("dx12.resubmit_retained.lookup");
        let sc = scope.sc.lock().unwrap();
        sc.retained_graphs.get(&key).map(|r| {
            (
                r.command_list.clone(),
                r.slot_idx,
                r.used_slots.clone(),
                r.frame_table_staging.clone(),
            )
        })
    };

    let Some((command_list, slot_idx, used_slots, _frame_table_staging)) = retained else {
        return Ok(None);
    };

    let cmd_list: ID3D12CommandList = command_list.cast().context("Failed to cast retained command list")?;

    let ctx_fence = {
        let _tz_fence = tracy_zone!("dx12.resubmit_retained.ctx_fence");
        scope
            .context_fences
            .read()
            .unwrap()
            .get(&ctx)
            .context("Invalid context handle")?
            .1
            .clone()
    };
    let logical_device = scope.ld();
    let fence_value = allocate_timeline_value(&logical_device.timeline_next);

    logical_device
        .descriptors
        .lock()
        .unwrap()
        .record_slot_usage(ctx, fence_value, used_slots.iter().copied());

    {
        let mut sc = scope.sc.lock().unwrap();
        if let Some(slot) = sc.compute_allocator_pool.get_mut(slot_idx) {
            slot.fence_value = fence_value;
        }
        sc.last_submitted_seq = fence_value;
    }

    {
        let _tz = crate::tracy_zone!("goldy.submit.dx12.deletion_drain");
        let ctx_completed = unsafe { ctx_fence.GetCompletedValue() };
        super::context::drain_context_deletion_queue_up_to(
            logical_device,
            &mut scope.sc.lock().unwrap(),
            ctx_completed,
        );
    }

    super::pending_submit::enqueue_retained_resubmit(
        logical_device,
        scope.context_fences,
        ctx_fence,
        vec![Some(cmd_list)],
        sync,
        fence_value,
    )?;

    Ok(Some(fence_value))
}

pub(super) fn try_resubmit_retained(
    state: &mut Dx12State,
    ctx: ContextHandle,
    key: u64,
    sync: Option<&SubmitSync>,
) -> Result<Option<TimelineValue>> {
    try_resubmit_retained_with_scope(&scope_from_state(state, ctx)?, ctx, key, sync)
}

fn evict_retained_on_context(
    contexts: &types::SharedContextMap,
    frame_table: &super::frame_table::FrameTableDevice,
    ctx: ContextHandle,
    key: u64,
) {
    if let Some(sc_arc) = contexts.read().unwrap().get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        if let Some(old) = sc.retained_graphs.remove(&key) {
            if let Some(row) = old.frame_table_row {
                super::frame_table::unpin_row(frame_table, row);
            }
            if let Some(slot) = sc.compute_allocator_pool.get_mut(old.slot_idx) {
                slot.retained = false;
            }
        }
    }
}

/// Evict every retained graph on contexts for `device_handle` that pins `row`.
///
/// Called when the frame-table ring wraps and a new prologue needs to overwrite staging
/// bytes for a row still held by a retained command list from partitioned retention.
pub(super) fn evict_retained_pinning_row(
    contexts: &types::SharedContextMap,
    frame_table: &super::frame_table::FrameTableDevice,
    device_handle: DeviceHandle,
    row: u32,
) {
    // Collect all (ctx, key) pairs that pin `row` under a single short-lived read guard,
    // then drop the guard before calling evict_retained_on_context.
    //
    // evict_retained_on_context takes its own contexts.read() internally.  Windows
    // SRWLOCK has write-priority: if any thread queues a write (e.g. context::destroy)
    // between the outer read acquisition and the inner one, the inner read blocks while
    // the outer read prevents the writer from completing — deadlock.  Releasing the
    // guard before calling evict avoids the re-entrant read.
    let evict_list: Vec<(ContextHandle, Vec<u64>)> = {
        let contexts_read = contexts.read().unwrap();
        contexts_read
            .iter()
            .filter(|(_, sc_arc)| sc_arc.lock().unwrap().device == device_handle)
            .filter_map(|(ctx_h, sc_arc)| {
                let keys: Vec<u64> = sc_arc
                    .lock()
                    .unwrap()
                    .retained_graphs
                    .iter()
                    .filter(|(_, g)| g.frame_table_row == Some(row))
                    .map(|(k, _)| *k)
                    .collect();
                if keys.is_empty() {
                    None
                } else {
                    Some((*ctx_h, keys))
                }
            })
            .collect()
    }; // contexts_read dropped here — no read guard held during eviction

    for (ctx, keys) in evict_list {
        for key in keys {
            evict_retained_on_context(contexts, frame_table, ctx, key);
        }
    }
}

pub(super) fn evict_retained_with_scope(scope: &Dx12SubmitScope<'_>, ctx: ContextHandle, key: u64) {
    evict_retained_on_context(scope.contexts(), scope.frame_table(), ctx, key);
}

/// Drop the retained command list for `key`, marking its pool slot as reusable.
pub(super) fn evict_retained(state: &Dx12State, ctx: ContextHandle, key: u64) {
    if let Ok(scope) = scope_from_state(state, ctx) {
        evict_retained_with_scope(&scope, ctx, key);
    }
}

/// Check if the fence for the given token has signaled.
#[allow(
    dead_code,
    reason = "retained for deprecated fence-based paths; timeline uses fence internally"
)]
pub(super) fn is_fence_complete(state: &Dx12State, device_handle: DeviceHandle, token: TimelineValue) -> bool {
    let logical_device = match state.devices.get(&device_handle) {
        Some(dev) => dev,
        None => return false,
    };
    (unsafe { logical_device.fence.GetCompletedValue() }) >= token
}

/// Block until the fence signals.
#[allow(dead_code, reason = "retained for deprecated fence-based paths")]
pub(super) fn wait_fence(state: &Dx12State, device_handle: DeviceHandle, token: TimelineValue) -> Result<()> {
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
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
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
    super::utils::wait_for_fence_timeout(&logical_device.fence, token, timeout_ms)
}

#[cfg(test)]
mod barrier_lowering_tests {
    use super::*;
    use crate::task_graph::NodeAccess;

    fn slot(access: NodeAccess, kinds: UsageKindFlags) -> SlotUsageSet {
        let mut s = SlotUsageSet::default();
        s.merge(access, kinds);
        s
    }

    fn has(mask: D3D12_BARRIER_ACCESS, bit: D3D12_BARRIER_ACCESS) -> bool {
        mask.0 & bit.0 != 0
    }

    // --- The regression that crashed goldy-doom (ID 1332) -------------------

    /// A non-storage (upload/uniform) buffer must NEVER carry UAV or COPY_DEST
    /// access bits — the D3D12 debug layer rejects (ID 1332) such barriers and
    /// `Close()` fails. This is the exact case that only Doom exercised.
    #[test]
    fn non_storage_buffer_never_emits_uav_or_copy_dest() {
        // Compute write recorded against a non-storage buffer (e.g. a uniform
        // buffer conservatively stamped as written by parcel `record_any`).
        let compute_write = slot(NodeAccess::Write, UsageKindFlags::COMPUTE);
        let access = slot_usage_to_dx12_access_for_buffer(&compute_write, /*is_storage=*/ false);
        assert!(
            !has(access, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS),
            "non-storage buffer must not get UNORDERED_ACCESS"
        );

        // Transfer write against a non-storage buffer must not be COPY_DEST.
        let transfer_write = slot(NodeAccess::Write, UsageKindFlags::TRANSFER);
        let access = slot_usage_to_dx12_access_for_buffer(&transfer_write, false);
        assert!(
            !has(access, D3D12_BARRIER_ACCESS_COPY_DEST),
            "non-storage buffer must not get COPY_DEST"
        );

        // The combined producer set that doom actually generated.
        let mut combined = SlotUsageSet::default();
        combined.merge(NodeAccess::Write, UsageKindFlags::COMPUTE | UsageKindFlags::TRANSFER);
        let access = slot_usage_to_dx12_access_for_buffer(&combined, false);
        assert!(!has(access, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS));
        assert!(!has(access, D3D12_BARRIER_ACCESS_COPY_DEST));
        assert!(has(access, D3D12_BARRIER_ACCESS_SHADER_RESOURCE));
        assert!(has(access, D3D12_BARRIER_ACCESS_COPY_SOURCE));
    }

    /// A storage buffer with an actual compute write still gets UAV — clamping
    /// must not regress the legitimate case the tests already covered.
    #[test]
    fn storage_buffer_compute_write_keeps_uav() {
        let compute_write = slot(NodeAccess::Write, UsageKindFlags::COMPUTE);
        let access = slot_usage_to_dx12_access_for_buffer(&compute_write, /*is_storage=*/ true);
        assert!(has(access, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS));
        assert!(has(access, D3D12_BARRIER_ACCESS_SHADER_RESOURCE));
    }

    /// A read-only compute binding never needs UAV, even on a storage buffer.
    #[test]
    fn storage_buffer_compute_read_is_srv_only() {
        let compute_read = slot(NodeAccess::Read, UsageKindFlags::COMPUTE);
        let access = slot_usage_to_dx12_access_for_buffer(&compute_read, true);
        assert!(has(access, D3D12_BARRIER_ACCESS_SHADER_RESOURCE));
        assert!(!has(access, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS));
    }

    /// Storage transfer write -> COPY_DEST; storage transfer read -> COPY_SOURCE.
    #[test]
    fn storage_buffer_transfer_direction() {
        let w = slot(NodeAccess::Write, UsageKindFlags::TRANSFER);
        assert!(has(
            slot_usage_to_dx12_access_for_buffer(&w, true),
            D3D12_BARRIER_ACCESS_COPY_DEST
        ));
        let r = slot(NodeAccess::Read, UsageKindFlags::TRANSFER);
        assert!(has(
            slot_usage_to_dx12_access_for_buffer(&r, true),
            D3D12_BARRIER_ACCESS_COPY_SOURCE
        ));
    }

    /// Render-pass buffer read maps to SHADER_RESOURCE (vertex/pixel), never an
    /// attachment access (which is illegal on a buffer barrier — ID 1332).
    #[test]
    fn render_buffer_read_is_shader_resource() {
        let render_read = slot(NodeAccess::Read, UsageKindFlags::RENDER);
        let access = slot_usage_to_dx12_access_for_buffer(&render_read, false);
        assert!(has(access, D3D12_BARRIER_ACCESS_SHADER_RESOURCE));
        assert!(!has(access, D3D12_BARRIER_ACCESS_RENDER_TARGET));
        assert!(!has(access, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE));
    }

    /// Empty usage -> COMMON (caller treats this specially for WARP global barriers).
    #[test]
    fn empty_usage_is_common() {
        let empty = SlotUsageSet::default();
        assert_eq!(
            slot_usage_to_dx12_access_for_buffer(&empty, false).0,
            D3D12_BARRIER_ACCESS_COMMON.0
        );
        assert_eq!(slot_usage_to_dx12_sync(&empty, false).0, D3D12_BARRIER_SYNC_ALL.0);
    }

    // --- Sync-stage / access pairing invariant (ID 1331 / barriers.rs assert) ---

    /// Every non-empty render buffer usage must produce a non-zero sync stage so
    /// it never pairs SYNC_NONE with a non-NO_ACCESS access (the panic that hit
    /// scheme_game_of_life_update_100).
    #[test]
    fn render_buffer_sync_is_non_zero() {
        let render_read = slot(NodeAccess::Read, UsageKindFlags::RENDER);
        let sync = slot_usage_to_dx12_sync(&render_read, false);
        assert_ne!(sync.0, 0, "render buffer usage must have a valid sync stage");
        assert!(sync.0 & D3D12_BARRIER_SYNC_VERTEX_SHADING.0 != 0);
        assert!(sync.0 & D3D12_BARRIER_SYNC_PIXEL_SHADING.0 != 0);
    }

    /// Any non-empty usage that lowers to a non-COMMON access must also lower to
    /// a non-zero sync stage (mirrors the D3D12 SyncNone/AccessNoAccess pairing rule).
    #[test]
    fn nonempty_usage_has_paired_sync_and_access() {
        let cases = [
            slot(NodeAccess::Write, UsageKindFlags::COMPUTE),
            slot(NodeAccess::Read, UsageKindFlags::COMPUTE),
            slot(NodeAccess::Write, UsageKindFlags::TRANSFER),
            slot(NodeAccess::Read, UsageKindFlags::TRANSFER),
            slot(NodeAccess::Read, UsageKindFlags::RENDER),
        ];
        for (storage, set) in cases.iter().flat_map(|s| [(true, s), (false, s)]) {
            let access = slot_usage_to_dx12_access_for_buffer(set, storage);
            let sync = slot_usage_to_dx12_sync(set, storage);
            if access.0 != D3D12_BARRIER_ACCESS_COMMON.0 {
                assert_ne!(
                    sync.0, 0,
                    "non-COMMON access {:#x} (storage={storage}) must have a sync stage",
                    access.0
                );
            }
        }
    }
}
