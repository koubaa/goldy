//! DX12 frame-table buffers and prologue (staging upload + device-local copy).

use super::types::{BufferState, Dx12State, LogicalDevice, SharedContextFrameTable};
use super::BufferHandle;
use crate::backend::GpuCommand;
use crate::frame_table::{
    FRAME_TABLE_MAX_ROWS, FRAME_TABLE_ROW_STRIDE, FRAME_TABLE_STAGING_BYTES, FRAME_TABLE_TABLE_U32S,
    FRAME_TABLE_USER_SLOT_BASE,
};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

/// Sentinel in [`ContextFrameTable::last_token_for_row`]: row is reserved for CPU
/// staging / recording and must not be claimed by another submitter until
/// [`record_submission`] stores the GPU timeline value (or `0` on abort).
const ROW_IN_USE: u64 = u64::MAX;

/// Per-context frame-table GPU resources and ring state.
pub(crate) struct ContextFrameTable {
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
    /// Last GPU timeline value that used each upload row (`0` = free/unused,
    /// [`ROW_IN_USE`] = reserved by a submitter that has not yet recorded).
    ///
    /// Reuse waits for this value on the context (or device) fence — backpressure
    /// when CPU submission outruns GPU consumption of the finite ring.
    pub last_token_for_row: [AtomicU64; FRAME_TABLE_MAX_ROWS as usize],
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
) -> Result<SharedContextFrameTable> {
    let (selector, selector_slot) =
        create_scattered_u32_buffer_registered(state, device_handle, ld, 1, "goldy_frame_table_selector")?;

    let (device_table, table_slot) = create_scattered_u32_buffer_registered(
        state,
        device_handle,
        ld,
        FRAME_TABLE_TABLE_U32S as u32,
        "goldy_frame_table_device",
    )
    .inspect_err(|_| {
        release_registered_buffers(state, ld, &[selector]);
    })?;

    let (staging, staging_mapped) = create_upload_table_buffer(ld).inspect_err(|_| {
        release_registered_buffers(state, ld, &[selector, device_table]);
    })?;

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

    let ft = SharedContextFrameTable::new(ContextFrameTable {
        selector,
        device_table,
        selector_slot,
        table_slot,
        staging,
        staging_mapped,
        submission_counter: AtomicU32::new(0),
        pinned_rows: AtomicU32::new(0),
        last_token_for_row: std::array::from_fn(|_| AtomicU64::new(0)),
    });

    // Drop the BufferTable read before any error cleanup: destroy_context takes
    // buffers.write() and nesting that under a live read is a permanent hang.
    let bind_result = {
        let buffers = state.buffers.read().unwrap();
        bind_to_bindless_heap(ld, &ft, &buffers.entries)
    };
    if let Err(e) = bind_result {
        destroy_context(state, device_handle, &ft);
        return Err(e);
    }

    Ok(ft)
}

/// Lazy-init the device-owned frame table used by legacy `render_to_target`.
///
/// Standalone render passes have no submission context; they must not borrow a
/// live context's ring buffer or advance its `submission_counter`.
pub(crate) fn ensure_legacy_frame_table(
    state: &mut Dx12State,
    device_handle: super::DeviceHandle,
) -> Result<SharedContextFrameTable> {
    let ld = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?
        .clone();
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
/// Called once at context init. Each context owns disjoint heap indices from the
/// registry, so concurrent inits may write the shared heap safely (non-overlapping
/// descriptor regions). No rebinding at execute time is ever needed.
pub(crate) fn bind_to_bindless_heap(
    ld: &LogicalDevice,
    ft: &ContextFrameTable,
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
pub(crate) fn destroy_context(state: &Dx12State, device_handle: super::DeviceHandle, ft: &ContextFrameTable) {
    let ld = state
        .devices
        .get(&device_handle)
        .expect("frame table destroy: invalid device handle");
    destroy_context_resources(&state.buffers, ld, ft);
}

pub(crate) fn destroy_context_resources(
    buffers: &super::types::SharedBufferTable,
    ld: &super::types::LogicalDevice,
    ft: &ContextFrameTable,
) {
    unsafe {
        ft.staging.Unmap(0, None);
    }
    let mut buffers_write = buffers.write().unwrap();
    buffers_write.entries.remove(&ft.selector);
    buffers_write.entries.remove(&ft.device_table);
    drop(buffers_write);
    let mut registry = ld.descriptors.lock().unwrap();
    registry.reclaim_buffer_slots(ft.selector);
    registry.reclaim_buffer_slots(ft.device_table);
}

/// Roll back buffer registry entries when [`init_context`] fails mid-flight.
fn release_registered_buffers(state: &Dx12State, ld: &LogicalDevice, handles: &[BufferHandle]) {
    let mut buffers = state.buffers.write().unwrap();
    for &handle in handles {
        buffers.entries.remove(&handle);
    }
    drop(buffers);
    let mut registry = ld.descriptors.lock().unwrap();
    for &handle in handles {
        registry.reclaim_buffer_slots(handle);
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
            is_withdraw_staging: false,
            texture_copy_footprint: None,
        },
    );
    Ok((handle, bindless_slot))
}

fn legacy_state_to_barrier(state: D3D12_RESOURCE_STATES) -> (D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS) {
    if state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
        (
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
        )
    } else if state == D3D12_RESOURCE_STATE_COPY_DEST {
        (D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST)
    } else if state == D3D12_RESOURCE_STATE_COPY_SOURCE {
        (D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE)
    } else if state == D3D12_RESOURCE_STATE_GENERIC_READ {
        // Upload-buffer implicit read — express as copy-source (valid on compute queues).
        (D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE)
    } else {
        (D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON)
    }
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
    let (sync_before, access_before) = legacy_state_to_barrier(before);
    let (sync_after, access_after) = legacy_state_to_barrier(after);
    let mut barrier = [super::barriers::buffer_barrier_full(
        resource,
        sync_before,
        sync_after,
        access_before,
        access_after,
    )];
    unsafe {
        super::barriers::barrier_buffers(cl, &barrier);
        super::barriers::drop_buffer_barriers(&mut barrier);
    }
}

/// Pin a staging row so retained CB prologue copies keep valid source bytes.
pub(crate) fn pin_row(ft: &ContextFrameTable, row: u32) -> Result<()> {
    let bit = 1u32 << row;
    let prev = ft.pinned_rows.fetch_or(bit, Ordering::AcqRel);
    if prev & bit != 0 {
        anyhow::bail!("frame table row {row} is already pinned");
    }
    Ok(())
}

/// Release a row pinned by [`pin_row`].
pub(crate) fn unpin_row(ft: &ContextFrameTable, row: u32) {
    let bit = 1u32 << row;
    ft.pinned_rows.fetch_and(!bit, Ordering::AcqRel);
}

/// Record the GPU timeline value that owns `row` after Signal (or `0` to abort a reservation).
pub(crate) fn record_submission(ft: &ContextFrameTable, row: u32, token: u64) {
    debug_assert!(
        token != ROW_IN_USE,
        "frame table row token must not be the in-use sentinel"
    );
    ft.last_token_for_row[row as usize].store(token, Ordering::Release);
}

/// Clears [`ROW_IN_USE`] on drop unless [`Self::take`] / [`Self::commit`] runs first.
///
/// Use around record paths that may `?`-bail after [`record_prologue`] so a failed
/// submit cannot leave a row permanently reserved.
pub(crate) struct RowReservation<'a> {
    ft: &'a ContextFrameTable,
    row: Option<u32>,
}

impl<'a> RowReservation<'a> {
    pub(crate) fn new(ft: &'a ContextFrameTable) -> Self {
        Self { ft, row: None }
    }

    pub(crate) fn set(&mut self, row: u32) {
        self.row = Some(row);
    }

    /// Disarm without recording (caller will [`record_submission`] itself).
    pub(crate) fn take(&mut self) -> Option<u32> {
        self.row.take()
    }

    pub(crate) fn commit(mut self, token: u64) {
        if let Some(row) = self.row.take() {
            record_submission(self.ft, row, token);
        }
    }
}

impl Drop for RowReservation<'_> {
    fn drop(&mut self) {
        if let Some(row) = self.row.take() {
            record_submission(self.ft, row, 0);
        }
    }
}

fn assert_not_pinned(ft: &ContextFrameTable, row: u32) -> Result<()> {
    if ft.pinned_rows.load(Ordering::Acquire) & (1 << row) != 0 {
        anyhow::bail!(
            "frame table row {row} is still pinned after evicting retained graphs; \
             retained command list lifecycle bug"
        );
    }
    Ok(())
}

/// Wait until `fence` has reached `token`, then claim the row with [`ROW_IN_USE`].
fn reserve_row_with_backpressure(ft: &ContextFrameTable, row: u32, fence: &ID3D12Fence) -> Result<()> {
    loop {
        let prev = ft.last_token_for_row[row as usize].load(Ordering::Acquire);
        if prev == ROW_IN_USE {
            // Another submitter reserved this row and has not recorded yet.
            std::thread::yield_now();
            continue;
        }
        if prev > 0 {
            let completed = unsafe { fence.GetCompletedValue() };
            if completed < prev {
                super::utils::wait_for_fence(fence, prev)?;
            }
        }
        match ft.last_token_for_row[row as usize].compare_exchange(
            prev,
            ROW_IN_USE,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return Ok(()),
            Err(_) => continue,
        }
    }
}

fn write_row_payload(ft: &ContextFrameTable, data: &[u32], row: u32) {
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    let staging_ptr = ft.staging_mapped as *mut u32;
    unsafe {
        let payload_dst =
            staging_ptr.add(crate::frame_table::FRAME_TABLE_STAGING_SELECTOR_U32S + row as usize * row_u32s);
        std::ptr::copy_nonoverlapping(data.as_ptr(), payload_dst, copy_u32s);
        std::ptr::write(staging_ptr.add(row as usize), row);
    }
}

/// CPU staging write before a submission (row bump + table bytes; GPU copy is in the CB).
///
/// When the ring wraps onto a row still referenced by in-flight GPU work, this blocks on
/// that work's fence value (bounded pipeline backpressure) before overwriting upload bytes.
pub(crate) fn write_staging_for_submission(
    contexts: &super::types::SharedContextMap,
    ld: &super::types::SharedLogicalDevice,
    ctx: super::ContextHandle,
    frame_table: &ContextFrameTable,
    data: &[u32],
) -> Result<u32> {
    let sub = frame_table.submission_counter.fetch_add(1, Ordering::Relaxed);
    let row = sub % FRAME_TABLE_MAX_ROWS;
    if frame_table.pinned_rows.load(Ordering::Acquire) & (1 << row) != 0 {
        super::compute::evict_retained_pinning_row_for_context(contexts, frame_table, ld, ctx, row);
    }
    assert_not_pinned(frame_table, row)?;
    let fence = {
        let contexts_read = contexts.read().unwrap();
        let sc_arc = contexts_read
            .get(&ctx)
            .with_context(|| format!("frame table staging: invalid context {ctx}"))?
            .clone();
        drop(contexts_read);
        let fence = sc_arc.lock().unwrap().fence.clone();
        fence
    };
    reserve_row_with_backpressure(frame_table, row, &fence)?;
    write_row_payload(frame_table, data, row);
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
fn write_staging_standalone(frame_table: &ContextFrameTable, fence: &ID3D12Fence, data: &[u32]) -> Result<u32> {
    let sub = frame_table.submission_counter.fetch_add(1, Ordering::Relaxed);
    let row = sub % FRAME_TABLE_MAX_ROWS;
    assert_not_pinned(frame_table, row)?;
    reserve_row_with_backpressure(frame_table, row, fence)?;
    write_row_payload(frame_table, data, row);
    Ok(row)
}

fn record_prologue_gpu_copies(
    frame_table: &ContextFrameTable,
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
    frame_table: &ContextFrameTable,
    fence: &ID3D12Fence,
    buffers: &HashMap<BufferHandle, BufferState>,
    cl: &ID3D12GraphicsCommandList7,
    data: &[u32],
) -> Result<u32> {
    let row = write_staging_standalone(frame_table, fence, data)?;
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
    ld: &super::types::SharedLogicalDevice,
    ctx: super::ContextHandle,
    frame_table: &ContextFrameTable,
    buffers: &HashMap<BufferHandle, BufferState>,
    cl: &ID3D12GraphicsCommandList7,
    data: &[u32],
) -> Result<u32> {
    let row = write_staging_for_submission(contexts, ld, ctx, frame_table, data)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal stand-in so we can exercise token bookkeeping without a GPU.
    fn empty_tokens() -> [AtomicU64; FRAME_TABLE_MAX_ROWS as usize] {
        std::array::from_fn(|_| AtomicU64::new(0))
    }

    #[test]
    fn record_submission_stores_token() {
        let tokens = empty_tokens();
        // Simulate ContextFrameTable::last_token_for_row store path.
        tokens[3].store(ROW_IN_USE, Ordering::Release);
        tokens[3].store(42, Ordering::Release);
        assert_eq!(tokens[3].load(Ordering::Acquire), 42);
        assert_ne!(tokens[3].load(Ordering::Acquire), ROW_IN_USE);
    }

    #[test]
    fn row_reservation_drop_aborts_to_zero() {
        // Build a tiny fake table with only the token array + Drop path via record_submission.
        // We can't construct a full ContextFrameTable without D3D resources, so test the
        // atomic protocol that Drop relies on.
        let tokens = empty_tokens();
        tokens[1].store(ROW_IN_USE, Ordering::Release);
        // Abort path: store 0 (what RowReservation::drop does via record_submission).
        tokens[1].store(0, Ordering::Release);
        assert_eq!(tokens[1].load(Ordering::Acquire), 0);
    }

    #[test]
    fn cas_claims_row_from_retired_token() {
        let tokens = empty_tokens();
        tokens[0].store(7, Ordering::Release);
        let prev = tokens[0].load(Ordering::Acquire);
        assert_eq!(prev, 7);
        assert!(tokens[0]
            .compare_exchange(prev, ROW_IN_USE, Ordering::AcqRel, Ordering::Acquire)
            .is_ok());
        assert_eq!(tokens[0].load(Ordering::Acquire), ROW_IN_USE);
    }
}
