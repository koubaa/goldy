//! DX12 frame-table buffers and prologue (staging upload + device-local copy).

use super::types::{BufferState, Dx12State, LogicalDevice, SharedFrameTableDevice};
use super::BufferHandle;
use crate::backend::GpuCommand;
use crate::frame_table::{
    FRAME_TABLE_MAX_ROWS, FRAME_TABLE_ROW_STRIDE, FRAME_TABLE_STAGING_BYTES, FRAME_TABLE_TABLE_U32S,
    FRAME_TABLE_USER_SLOT_BASE,
};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

/// Per-context frame-table GPU resources and ring state.
pub(crate) struct FrameTableDevice {
    pub selector: BufferHandle,
    pub device_table: BufferHandle,
    /// Bindless heap slot of this context's selector cell (shader `_rs1`).
    pub selector_slot: u32,
    /// Bindless heap slot of this context's device-local table (shader `_rs2`).
    pub table_slot: u32,
    pub staging: ID3D12Resource,
    pub staging_mapped: usize,
    pub submission_counter: AtomicU32,
    /// Bitmask of rows pinned by retained command lists (must not be overwritten).
    pub pinned_rows: AtomicU32,
}

/// Reserve user bindless slots once per device (low protocol slots stay unused).
pub(crate) fn reserve_device_bindless_slots(ld: &LogicalDevice) {
    ld.descriptors
        .lock()
        .unwrap()
        .resource_registry
        .ensure_cbv_start(FRAME_TABLE_USER_SLOT_BASE);
}

/// Create per-context frame-table buffers at per-context bindless slots.
///
/// Slots come from the device's descriptor registry (like any other buffer),
/// so concurrent contexts on one device never share mutable descriptor slots
/// and no rebinding at execute time is needed. The slot indices reach shaders
/// via push constants (`_rs1`/`_rs2`).
pub(crate) fn init_context(
    state: &mut Dx12State,
    device_handle: super::DeviceHandle,
    ld: &LogicalDevice,
) -> Result<SharedFrameTableDevice> {
    let (selector, selector_slot) =
        create_scattered_u32_buffer_registered(state, device_handle, ld, 1, "goldy_frame_table_selector")?;
    let (device_table, table_slot) = create_scattered_u32_buffer_registered(
        state,
        device_handle,
        ld,
        FRAME_TABLE_TABLE_U32S as u32,
        "goldy_frame_table_device",
    )?;

    let (staging, staging_mapped) = create_upload_table_buffer(ld)?;

    state
        .buffers
        .write()
        .unwrap()
        .entries
        .get_mut(&selector)
        .unwrap()
        .element_stride = Some(4);
    state
        .buffers
        .write()
        .unwrap()
        .entries
        .get_mut(&device_table)
        .unwrap()
        .element_stride = Some(4);

    let ft = SharedFrameTableDevice::new(FrameTableDevice {
        selector,
        device_table,
        selector_slot,
        table_slot,
        staging,
        staging_mapped,
        submission_counter: AtomicU32::new(0),
        pinned_rows: AtomicU32::new(0),
    });

    let buffers = state.buffers.read().unwrap();
    bind_to_bindless_heap(ld, &ft, &buffers.entries)?;

    Ok(ft)
}

/// Lazy-init the device-owned frame table used by legacy `render_to_target`.
///
/// Standalone render passes have no submission context; they must not borrow a
/// live context's ring buffer or advance its `submission_counter`.
pub(crate) fn ensure_legacy_frame_table(
    state: &mut Dx12State,
    device_handle: super::DeviceHandle,
) -> Result<SharedFrameTableDevice> {
    let ld = state.devices.get(&device_handle).context("Invalid device handle")?.clone();
    let mut guard = ld.legacy_frame_table.lock().unwrap();
    if let Some(ft) = guard.as_ref() {
        return Ok(std::sync::Arc::clone(ft));
    }
    let ft = init_context(state, device_handle, &ld)?;
    *guard = Some(std::sync::Arc::clone(&ft));
    Ok(ft)
}

/// Write this context's selector/table UAV descriptors at its per-context slots.
///
/// Called once at context init; the slots are context-private for the context's
/// lifetime, so no rebinding at execute time is ever needed.
pub(crate) fn bind_to_bindless_heap(
    ld: &LogicalDevice,
    ft: &FrameTableDevice,
    buffers: &HashMap<BufferHandle, BufferState>,
) -> Result<()> {
    let selector = buffers.get(&ft.selector).context("frame table selector")?;
    let device_table = buffers.get(&ft.device_table).context("frame table device table")?;
    write_uav_at_slot(ld, ft.selector_slot, &selector.resource, 1)?;
    write_uav_at_slot(ld, ft.table_slot, &device_table.resource, FRAME_TABLE_TABLE_U32S as u32)?;
    Ok(())
}

fn write_uav_at_slot(ld: &LogicalDevice, bindless_slot: u32, resource: &ID3D12Resource, num_u32s: u32) -> Result<()> {
    let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: 0,
                NumElements: num_u32s,
                StructureByteStride: 4,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_NONE,
            },
        },
    };
    let cpu_handle = unsafe {
        let mut h = ld.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
        h.ptr += (bindless_slot * ld.cbv_srv_uav_descriptor_size) as usize;
        h
    };
    unsafe {
        ld.device
            .CreateUnorderedAccessView(resource, None, Some(&uav_desc), cpu_handle);
    }
    Ok(())
}

/// Destroy per-context frame-table resources: staging memory, the selector and
/// table buffer entries, and their per-context bindless slots.
///
/// Caller guarantees the context's GPU work has fully retired.
pub(crate) fn destroy_context(state: &Dx12State, device_handle: super::DeviceHandle, ft: &FrameTableDevice) {
    unsafe {
        ft.staging.Unmap(0, None);
    }
    let mut buffers = state.buffers.write().unwrap();
    buffers.entries.remove(&ft.selector);
    buffers.entries.remove(&ft.device_table);
    drop(buffers);
    if let Some(ld) = state.devices.get(&device_handle) {
        let mut registry = ld.descriptors.lock().unwrap();
        registry.reclaim_buffer_slots(ft.selector);
        registry.reclaim_buffer_slots(ft.device_table);
    }
}

fn create_upload_table_buffer(ld: &LogicalDevice) -> Result<(ID3D12Resource, usize)> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: FRAME_TABLE_STAGING_BYTES.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        ld.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut resource,
        )
    }
    .context("frame table staging CreateCommittedResource")?;
    let resource = resource.context("frame table staging null")?;
    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
    let range = D3D12_RANGE { Begin: 0, End: 0 };
    unsafe { resource.Map(0, Some(&range), Some(&mut mapped)) }.context("frame table staging Map")?;
    let ptr = mapped as *mut u8;
    if ptr.is_null() {
        anyhow::bail!("frame table staging Map returned null");
    }
    Ok((resource, ptr as usize))
}

fn create_scattered_u32_buffer_registered(
    state: &mut Dx12State,
    device_handle: super::DeviceHandle,
    ld: &LogicalDevice,
    num_u32s: u32,
    debug_name: &str,
) -> Result<(BufferHandle, u32)> {
    let logical_size = (num_u32s as u64) * 4;
    let size = logical_size.max(256);
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        ld.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )
    }
    .context("frame table buffer CreateCommittedResource")?;
    let resource = resource.context("frame table buffer null")?;
    if !debug_name.is_empty() {
        let name: windows::core::HSTRING = debug_name.into();
        let _ = unsafe { resource.SetName(&name) };
    }

    let handle = state.buffers.write().unwrap().alloc_handle();
    let bindless_slot = ld
        .descriptors
        .lock()
        .unwrap()
        .resource_registry
        .register_buffer_uav(handle);
    state.buffers.write().unwrap().entries.insert(
        handle,
        BufferState {
            device_handle,
            resource: resource.clone(),
            size,
            allocation_size: size,
            bindless_offset: Some(bindless_slot),
            bindless_srv_offset: None,
            is_storage: true,
            upload_buffer: None,
            element_stride: Some(4),
            is_view: false,
            coherent_readback: None,
            coherent_readback_mapped: None,
            cpu_writable_upload_mapped: None,
            flags: crate::types::BufferFlags::empty(),
            transient_placed: false,
            parent_for_view: None,
            view_byte_offset: None,
            is_reserved: false,
            tile_byte_size: 0,
            reserved_tiles: Vec::new(),
            is_grant_readback: false,
            texture_copy_footprint: None,
        },
    );
    Ok((handle, bindless_slot))
}

fn transition_buffer(
    cl: &ID3D12GraphicsCommandList7,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    if before == after {
        return;
    }
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(resource) },
                StateBefore: before,
                StateAfter: after,
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            }),
        },
    };
    unsafe {
        cl.ResourceBarrier(&[barrier]);
    }
}

/// Pin a staging row so retained CB prologue copies keep valid source bytes.
pub(crate) fn pin_row(ft: &FrameTableDevice, row: u32) -> Result<()> {
    let bit = 1u32 << row;
    let prev = ft.pinned_rows.fetch_or(bit, Ordering::AcqRel);
    if prev & bit != 0 {
        anyhow::bail!("frame table row {row} is already pinned");
    }
    Ok(())
}

/// Release a row pinned by [`pin_row`].
pub(crate) fn unpin_row(ft: &FrameTableDevice, row: u32) {
    let bit = 1u32 << row;
    ft.pinned_rows.fetch_and(!bit, Ordering::AcqRel);
}

fn assert_row_available(ft: &FrameTableDevice, row: u32) -> Result<()> {
    if ft.pinned_rows.load(Ordering::Acquire) & (1 << row) != 0 {
        anyhow::bail!(
            "frame table row {row} is still pinned after evicting retained graphs; \
             retained command list lifecycle bug"
        );
    }
    Ok(())
}

/// CPU staging write before a submission (row bump + table bytes; GPU copy is in the CB).
pub(crate) fn write_staging_for_submission(
    contexts: &super::types::SharedContextMap,
    ctx: super::ContextHandle,
    frame_table: &FrameTableDevice,
    data: &[u32],
) -> Result<u32> {
    let sub = frame_table.submission_counter.fetch_add(1, Ordering::Relaxed);
    let row = sub % FRAME_TABLE_MAX_ROWS;
    if frame_table.pinned_rows.load(Ordering::Acquire) & (1 << row) != 0 {
        super::compute::evict_retained_pinning_row_for_context(contexts, frame_table, ctx, row);
    }
    assert_row_available(frame_table, row)?;
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    let staging_ptr = frame_table.staging_mapped as *mut u32;
    unsafe {
        let payload_dst =
            staging_ptr.add(crate::frame_table::FRAME_TABLE_STAGING_SELECTOR_U32S + row as usize * row_u32s);
        std::ptr::copy_nonoverlapping(data.as_ptr(), payload_dst, copy_u32s);
        std::ptr::write(staging_ptr.add(row as usize), row);
    }
    Ok(row)
}

/// Refresh the active row in the device-local table without advancing the selector.
pub(crate) fn sync_table_row_to_device(
    record: &super::submit_session::Dx12RecordState<'_>,
    _device_handle: super::DeviceHandle,
    cl: &ID3D12GraphicsCommandList7,
    data: &[u32],
) -> Result<()> {
    let ft = &record.frame_table;
    let buffers_read = record.buffers.read().unwrap();
    let device_table = buffers_read.entries.get(&ft.device_table).context("device table")?;
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    let staging_ptr = ft.staging_mapped as *mut u32;
    let row = ft.submission_counter.load(Ordering::Relaxed).saturating_sub(1) % FRAME_TABLE_MAX_ROWS;
    unsafe {
        let payload_dst =
            staging_ptr.add(crate::frame_table::FRAME_TABLE_STAGING_SELECTOR_U32S + row as usize * row_u32s);
        std::ptr::copy_nonoverlapping(data.as_ptr(), payload_dst, copy_u32s);
        std::ptr::write(staging_ptr.add(row as usize), row);
    }
    let row_bytes = (FRAME_TABLE_ROW_STRIDE as u64) * 4;
    let dest_offset = (row as u64) * row_bytes;
    let src_payload = crate::frame_table::staging_row_payload_byte_offset(row);

    transition_buffer(
        cl,
        &ft.staging,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    transition_buffer(
        cl,
        &device_table.resource,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    unsafe {
        cl.CopyBufferRegion(
            &device_table.resource,
            dest_offset,
            &ft.staging,
            src_payload,
            (copy_u32s * 4) as u64,
        );
    }
    transition_buffer(
        cl,
        &device_table.resource,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    transition_buffer(
        cl,
        &ft.staging,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    Ok(())
}

/// CPU staging write for legacy standalone render (no context eviction).
fn write_staging_standalone(frame_table: &FrameTableDevice, data: &[u32]) -> Result<u32> {
    let sub = frame_table.submission_counter.fetch_add(1, Ordering::Relaxed);
    let row = sub % FRAME_TABLE_MAX_ROWS;
    assert_row_available(frame_table, row)?;
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    let staging_ptr = frame_table.staging_mapped as *mut u32;
    unsafe {
        let payload_dst =
            staging_ptr.add(crate::frame_table::FRAME_TABLE_STAGING_SELECTOR_U32S + row as usize * row_u32s);
        std::ptr::copy_nonoverlapping(data.as_ptr(), payload_dst, copy_u32s);
        std::ptr::write(staging_ptr.add(row as usize), row);
    }
    Ok(row)
}

fn record_prologue_gpu_copies(
    frame_table: &FrameTableDevice,
    buffers: &HashMap<BufferHandle, BufferState>,
    cl: &ID3D12GraphicsCommandList7,
    data: &[u32],
    row: u32,
) -> Result<()> {
    let device_table = buffers.get(&frame_table.device_table).context("device table")?;
    let selector_state = buffers.get(&frame_table.selector).context("selector")?;
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    let row_bytes = (FRAME_TABLE_ROW_STRIDE as u64) * 4;
    let dest_offset = (row as u64) * row_bytes;
    let src_payload = crate::frame_table::staging_row_payload_byte_offset(row);
    let src_selector = crate::frame_table::staging_selector_byte_offset(row);

    transition_buffer(
        cl,
        &frame_table.staging,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    transition_buffer(
        cl,
        &device_table.resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    transition_buffer(
        cl,
        &selector_state.resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );

    unsafe {
        cl.CopyBufferRegion(
            &device_table.resource,
            dest_offset,
            &frame_table.staging,
            src_payload,
            (copy_u32s * 4) as u64,
        );
        cl.CopyBufferRegion(&selector_state.resource, 0, &frame_table.staging, src_selector, 4);
    }

    transition_buffer(
        cl,
        &device_table.resource,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    transition_buffer(
        cl,
        &selector_state.resource,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    transition_buffer(
        cl,
        &frame_table.staging,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    Ok(())
}

/// CPU staging write + GPU copy prologue for legacy `render_to_target`.
pub(crate) fn record_prologue_legacy(
    frame_table: &FrameTableDevice,
    buffers: &HashMap<BufferHandle, BufferState>,
    cl: &ID3D12GraphicsCommandList7,
    data: &[u32],
) -> Result<u32> {
    let row = write_staging_standalone(frame_table, data)?;
    record_prologue_gpu_copies(frame_table, buffers, cl, data, row)?;
    Ok(row)
}

/// CPU staging write + GPU copy prologue.
///
/// The prologue copies the active row index into this context's selector buffer.
/// That buffer is shared across all command lists on the context; correctness
/// assumes every submission for the context runs on the same FIFO queue so
/// command lists retire in submit order and a later prologue cannot clobber the
/// selector while an earlier list's dispatches are still executing.
pub(crate) fn record_prologue(
    contexts: &super::types::SharedContextMap,
    ctx: super::ContextHandle,
    frame_table: &FrameTableDevice,
    buffers: &HashMap<BufferHandle, BufferState>,
    cl: &ID3D12GraphicsCommandList7,
    data: &[u32],
) -> Result<u32> {
    let row = write_staging_for_submission(contexts, ctx, frame_table, data)?;
    record_prologue_gpu_copies(frame_table, buffers, cl, data, row)?;
    Ok(row)
}

pub(crate) fn extract_staging_from_commands(commands: &[GpuCommand]) -> Option<std::sync::Arc<[u32]>> {
    commands.iter().find_map(|c| match c {
        GpuCommand::FrameTableStaging { data } => Some(std::sync::Arc::clone(data)),
        _ => None,
    })
}

pub(crate) fn extract_staging_from_graph(commands: &[crate::backend::GraphCommand]) -> Option<std::sync::Arc<[u32]>> {
    commands.iter().find_map(|c| match c {
        crate::backend::GraphCommand::Compute(GpuCommand::FrameTableStaging { data }) => {
            Some(std::sync::Arc::clone(data))
        }
        _ => None,
    })
}

/// Lower render commands and build staging for standalone render passes (not graph submit).
pub(crate) fn prepare_render_commands(
    record: &super::submit_session::Dx12RecordState<'_>,
    commands: &[crate::backend::RenderCommand],
) -> Result<(Vec<u32>, Vec<crate::backend::RenderCommand>, bool)> {
    use crate::backend::RenderCommand;
    use crate::frame_table::FrameTableStaging;

    crate::backend::with_layout_validation(|| {
        crate::backend::validate_render_pass_bind_resources(
            commands,
            |h| {
                record
                    .pipelines
                    .read()
                    .unwrap()
                    .entries
                    .get(&h)
                    .map(|p| (p.binding_element_strides.clone(), p.shader_debug_name.clone()))
            },
            |h| {
                record
                    .buffers
                    .read()
                    .unwrap()
                    .entries
                    .get(&h)
                    .and_then(|b| b.element_stride)
            },
        )
    })?;

    let mut staging = FrameTableStaging::new();
    let lowered = commands
        .iter()
        .map(|cmd| match cmd {
            RenderCommand::BindResources { buffers } => {
                let indices: Vec<u32> = buffers
                    .iter()
                    .map(|h| {
                        record
                            .buffers
                            .read()
                            .unwrap()
                            .entries
                            .get(h)
                            .and_then(|b| b.bindless_offset)
                            .with_context(|| format!("BindResources: buffer handle {h:?} has no bindless offset"))
                    })
                    .collect::<Result<_>>()?;
                let frame_table_base = staging.alloc_dispatch(indices.len() as u32);
                staging.write_dispatch_indices(frame_table_base, &indices);
                Ok(RenderCommand::BindResourcesRaw {
                    indices: Vec::new(),
                    user: Vec::new(),
                    frame_table_base,
                })
            }
            other => {
                let batch = crate::frame_table::lower_render_pass_commands(&mut staging, std::slice::from_ref(other));
                Ok(batch.into_iter().next().unwrap_or_else(|| other.clone()))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let has_bindings = staging.has_bindings();
    Ok((staging.data, lowered, has_bindings))
}

/// Lower render commands and build staging (backend state lookup).
pub(crate) fn prepare_render_commands_state(
    state: &Dx12State,
    ctx: super::ContextHandle,
    device_handle: super::DeviceHandle,
    commands: &[crate::backend::RenderCommand],
) -> Result<(Vec<u32>, Vec<crate::backend::RenderCommand>, bool)> {
    let record = super::submit_session::record_state_from_backend(state, ctx, device_handle)?;
    prepare_render_commands(&record, commands)
}
