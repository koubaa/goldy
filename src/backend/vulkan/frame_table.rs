//! Vulkan frame-table buffers and prologue (staging upload + device-local copy).

use super::types::{
    self, BufferState, LogicalDevice, SharedBufferTable, SharedContextFrameTable, SharedContextMap,
    SharedPipelineTable, VulkanState,
};
use super::utils::{find_memory_type, with_buffer_sharing};
use super::BufferHandle;
use crate::backend::GpuCommand;
use crate::frame_table::{
    FRAME_TABLE_MAX_ROWS, FRAME_TABLE_ROW_STRIDE, FRAME_TABLE_STAGING_BYTES, FRAME_TABLE_TABLE_U32S,
    FRAME_TABLE_USER_SLOT_BASE,
};
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Sentinel in [`ContextFrameTable::last_token_for_row`]: row is reserved for CPU
/// staging / recording and must not be claimed by another submitter until
/// [`record_submission`] stores the GPU timeline value (or `0` on abort).
const ROW_IN_USE: u64 = u64::MAX;

/// Per-context frame-table GPU resources and ring state.
pub(crate) struct ContextFrameTable {
    pub selector: BufferHandle,
    pub device_table: BufferHandle,
    /// Bindless storage slot of this context's selector cell (shader `_rs1`).
    pub selector_slot: u32,
    /// Bindless storage slot of this context's device-local table (shader `_rs2`).
    pub table_slot: u32,
    pub staging: vk::Buffer,
    pub staging_memory: vk::DeviceMemory,
    pub staging_mapped: usize,
    pub submission_counter: AtomicU32,
    /// Bitmask of rows pinned by retained command buffers (must not be overwritten).
    pub pinned_rows: AtomicU32,
    /// Last GPU timeline value that used each upload row (`0` = free/unused,
    /// [`ROW_IN_USE`] = reserved by a submitter that has not yet recorded).
    ///
    /// Reuse waits for this value on the context timeline semaphore — backpressure
    /// when CPU submission outruns GPU consumption of the finite ring.
    pub last_token_for_row: [AtomicU64; FRAME_TABLE_MAX_ROWS as usize],
}

/// Stub frame table for the synthetic device-owner context (never used for recording).
pub(crate) fn device_owner_frame_table_stub() -> SharedContextFrameTable {
    use std::sync::Arc;
    Arc::new(ContextFrameTable {
        selector: 0,
        device_table: 0,
        selector_slot: 0,
        table_slot: 0,
        staging: vk::Buffer::null(),
        staging_memory: vk::DeviceMemory::null(),
        staging_mapped: 0,
        submission_counter: AtomicU32::new(0),
        pinned_rows: AtomicU32::new(0),
        last_token_for_row: std::array::from_fn(|_| AtomicU64::new(0)),
    })
}

/// Reserve user bindless slots once per device (low protocol slots stay unused).
pub(crate) fn reserve_device_bindless_slots(ld: &LogicalDevice) {
    ld.descriptors
        .lock()
        .unwrap()
        .resource_registry
        .ensure_storage_start(FRAME_TABLE_USER_SLOT_BASE);
}

/// Create per-context frame-table buffers at per-context bindless storage slots.
///
/// Slots come from the device's descriptor registry (like any other buffer), so
/// concurrent contexts never share mutable descriptor slots and no rebinding at
/// execute time is needed. The slot indices reach shaders via push constants
/// (`_rs1`/`_rs2`).
pub(crate) fn init_context(
    state: &mut VulkanState,
    instance: &ash::Instance,
    device_handle: super::DeviceHandle,
    ld: &LogicalDevice,
) -> Result<SharedContextFrameTable> {
    let (selector, selector_slot) = create_scattered_u32_buffer_registered(state, instance, device_handle, ld, 1)?;
    let (device_table, table_slot) = create_scattered_u32_buffer_registered(
        state,
        instance,
        device_handle,
        ld,
        FRAME_TABLE_TABLE_U32S as u32,
    )
    .map_err(|e| {
        release_registered_buffers(state, ld, &[selector]);
        e
    })?;

    let (staging, staging_memory, staging_mapped) =
        create_upload_table_buffer(instance, ld).map_err(|e| {
            release_registered_buffers(state, ld, &[selector, device_table]);
            e
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
        staging_memory,
        staging_mapped,
        submission_counter: AtomicU32::new(0),
        pinned_rows: AtomicU32::new(0),
        last_token_for_row: std::array::from_fn(|_| AtomicU64::new(0)),
    });

    let buffers = state.buffers.read().unwrap();
    if let Err(e) = bind_to_bindless_heap(ld, &ft, &buffers.entries) {
        destroy_context(state, ld, &ft);
        return Err(e);
    }

    Ok(ft)
}

/// Lazy-init the device-owned frame table used by legacy `render_to_target`.
pub(crate) fn ensure_legacy_frame_table(
    state: &mut VulkanState,
    instance: &ash::Instance,
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
    let ft = init_context(state, instance, device_handle, &ld)?;
    *guard = Some(std::sync::Arc::clone(&ft));
    Ok(ft)
}

/// Write this context's selector/table descriptors at its per-context slots.
///
/// Called once at context init. Each context owns disjoint heap indices from the
/// registry, so concurrent inits may write the shared descriptor set safely
/// (non-overlapping bindings). No rebinding at execute time is ever needed.
pub(crate) fn bind_to_bindless_heap(
    ld: &LogicalDevice,
    ft: &ContextFrameTable,
    buffers: &HashMap<BufferHandle, BufferState>,
) -> Result<()> {
    let Some(descriptor_set) = ld.bindless_descriptor_set else {
        return Ok(());
    };
    let selector = buffers.get(&ft.selector).context("frame table selector")?;
    let device_table = buffers.get(&ft.device_table).context("frame table device table")?;
    write_storage_at_slot(&ld.device, descriptor_set, ft.selector_slot, selector.buffer, 4)?;
    write_storage_at_slot(
        &ld.device,
        descriptor_set,
        ft.table_slot,
        device_table.buffer,
        (FRAME_TABLE_TABLE_U32S * 4) as u64,
    )?;
    Ok(())
}

fn write_storage_at_slot(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    bindless_slot: u32,
    buffer: vk::Buffer,
    range: u64,
) -> Result<()> {
    let buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(range.max(1));
    let write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(types::bindless_bindings::STORAGE_BUFFERS)
        .dst_array_element(bindless_slot)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&buffer_info));
    unsafe {
        device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
    }
    Ok(())
}

/// Destroy per-context frame-table resources: staging, the selector and table
/// buffers, and their per-context bindless slots.
///
/// Caller guarantees the context's GPU work has fully retired.
pub(crate) fn destroy_context(state: &VulkanState, ld: &LogicalDevice, ft: &ContextFrameTable) {
    destroy_context_resources(&state.buffers, ld, ft);
}

pub(crate) fn destroy_context_resources(
    buffers: &super::types::SharedBufferTable,
    ld: &LogicalDevice,
    ft: &ContextFrameTable,
) {
    unsafe {
        ld.device.destroy_buffer(ft.staging, None);
        ld.device.free_memory(ft.staging_memory, None);
    }
    let mut buffers_write = buffers.write().unwrap();
    for handle in [ft.selector, ft.device_table] {
        if let Some(entry) = buffers_write.entries.remove(&handle) {
            unsafe {
                ld.device.destroy_buffer(entry.buffer, None);
                ld.device.free_memory(entry.memory, None);
            }
        }
    }
    drop(buffers_write);
    let mut registry = ld.descriptors.lock().unwrap();
    registry.reclaim_buffer_slots(ft.selector);
    registry.reclaim_buffer_slots(ft.device_table);
}

/// Roll back buffer registry entries when [`init_context`] fails mid-flight.
fn release_registered_buffers(state: &VulkanState, ld: &LogicalDevice, handles: &[BufferHandle]) {
    let mut buffers = state.buffers.write().unwrap();
    for &handle in handles {
        if let Some(entry) = buffers.entries.remove(&handle) {
            unsafe {
                ld.device.destroy_buffer(entry.buffer, None);
                ld.device.free_memory(entry.memory, None);
            }
        }
    }
    drop(buffers);
    let mut registry = ld.descriptors.lock().unwrap();
    for &handle in handles {
        registry.reclaim_buffer_slots(handle);
    }
}

fn create_upload_table_buffer(
    instance: &ash::Instance,
    ld: &LogicalDevice,
) -> Result<(vk::Buffer, vk::DeviceMemory, usize)> {
    let size = FRAME_TABLE_STAGING_BYTES.max(256);
    let qf = ld.concurrent_queue_families();
    let buffer_info = with_buffer_sharing(
        vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC),
        qf.as_ref(),
    );

    let buffer = unsafe { ld.device.create_buffer(&buffer_info, None) }.context("frame table staging create_buffer")?;

    let mem_requirements = unsafe { ld.device.get_buffer_memory_requirements(buffer) };
    let memory_type = find_memory_type(
        instance,
        ld.physical_device,
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .context("frame table staging find_memory_type")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    let memory =
        unsafe { ld.device.allocate_memory(&alloc_info, None) }.context("frame table staging allocate_memory")?;

    unsafe { ld.device.bind_buffer_memory(buffer, memory, 0) }.context("frame table staging bind_buffer_memory")?;

    let mapped = unsafe { ld.map_memory2(memory, 0, size) }.context("frame table staging map_memory2")?;
    let ptr = mapped as *mut u8;
    if ptr.is_null() {
        anyhow::bail!("frame table staging map returned null");
    }

    Ok((buffer, memory, ptr as usize))
}

fn create_scattered_u32_buffer_registered(
    state: &mut VulkanState,
    instance: &ash::Instance,
    device_handle: super::DeviceHandle,
    ld: &LogicalDevice,
    num_u32s: u32,
) -> Result<(BufferHandle, u32)> {
    let logical_size = (num_u32s as u64) * 4;
    let allocation_size = logical_size.max(256);

    let vk_usage =
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let qf = ld.concurrent_queue_families();
    let buffer_info = with_buffer_sharing(
        vk::BufferCreateInfo::default().size(allocation_size).usage(vk_usage),
        qf.as_ref(),
    );

    let buffer = unsafe { ld.device.create_buffer(&buffer_info, None) }.context("frame table buffer create_buffer")?;

    let mem_requirements = unsafe { ld.device.get_buffer_memory_requirements(buffer) };
    let memory_type = find_memory_type(
        instance,
        ld.physical_device,
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .context("frame table buffer find_memory_type")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    let memory =
        unsafe { ld.device.allocate_memory(&alloc_info, None) }.context("frame table buffer allocate_memory")?;

    unsafe { ld.device.bind_buffer_memory(buffer, memory, 0) }.context("frame table buffer bind_buffer_memory")?;

    let handle = state.buffers.write().unwrap().alloc_handle();
    let bindless_slot = ld
        .descriptors
        .lock()
        .unwrap()
        .resource_registry
        .register_buffer(handle, true);
    state.buffers.write().unwrap().entries.insert(
        handle,
        BufferState {
            device_handle,
            buffer,
            memory,
            size: logical_size,
            allocation_size,
            bindless_index: Some(bindless_slot),
            is_storage: true,
            element_stride: Some(4),
            staging_buffer: None,
            staging_memory: None,
            is_view: false,
            host_mapped: None,
            flags: crate::types::BufferFlags::empty(),
            transient_heap_suballoc: false,
            view_byte_offset: None,
            is_sparse: false,
            sparse_block_size: 0,
            sparse_pages: Vec::new(),
            is_grant_readback: false,
            texture_copy_footprint: None,
        },
    );
    Ok((handle, bindless_slot))
}

fn prologue_pre_copy_barrier(device: &ash::Device, cmd: vk::CommandBuffer, on_graphics_queue: bool) {
    let src_graphics = if on_graphics_queue {
        vk::PipelineStageFlags2::ALL_GRAPHICS
    } else {
        vk::PipelineStageFlags2::empty()
    };
    let mem_barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | src_graphics)
        .src_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
    let dep_info = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dep_info);
    }
}

fn prologue_post_copy_barrier(device: &ash::Device, cmd: vk::CommandBuffer, on_graphics_queue: bool) {
    let dst_graphics = if on_graphics_queue {
        vk::PipelineStageFlags2::ALL_GRAPHICS
    } else {
        vk::PipelineStageFlags2::empty()
    };
    let mem_barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | dst_graphics)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
    let dep_info = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dep_info);
    }
}

struct TableCopyParams {
    row: u32,
    copy_u32s: usize,
    copy_selector: bool,
    on_graphics_queue: bool,
}

fn record_table_copies(
    ft: &ContextFrameTable,
    buffers: &SharedBufferTable,
    ld: &LogicalDevice,
    cmd: vk::CommandBuffer,
    params: TableCopyParams,
) -> Result<()> {
    let TableCopyParams {
        row,
        copy_u32s,
        copy_selector,
        on_graphics_queue,
    } = params;
    let buffers_read = buffers.read().unwrap();
    let device_table = buffers_read.entries.get(&ft.device_table).context("device table")?;
    let selector_state = buffers_read.entries.get(&ft.selector).context("selector")?;
    let row_bytes = (FRAME_TABLE_ROW_STRIDE as u64) * 4;
    let dest_offset = (row as u64) * row_bytes;
    let src_payload = crate::frame_table::staging_row_payload_byte_offset(row);
    let src_selector = crate::frame_table::staging_selector_byte_offset(row);

    prologue_pre_copy_barrier(&ld.device, cmd, on_graphics_queue);
    unsafe {
        ld.device.cmd_copy_buffer(
            cmd,
            ft.staging,
            device_table.buffer,
            std::slice::from_ref(&vk::BufferCopy {
                src_offset: src_payload,
                dst_offset: dest_offset,
                size: (copy_u32s * 4) as u64,
            }),
        );
        if copy_selector {
            ld.device.cmd_copy_buffer(
                cmd,
                ft.staging,
                selector_state.buffer,
                std::slice::from_ref(&vk::BufferCopy {
                    src_offset: src_selector,
                    dst_offset: 0,
                    size: 4,
                }),
            );
        }
    }
    prologue_post_copy_barrier(&ld.device, cmd, on_graphics_queue);
    Ok(())
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
             retained command buffer lifecycle bug"
        );
    }
    Ok(())
}

fn wait_for_timeline(device: &ash::Device, semaphore: vk::Semaphore, token: u64) -> Result<()> {
    if token == 0 {
        return Ok(());
    }
    let completed = unsafe { device.get_semaphore_counter_value(semaphore) }.unwrap_or(0);
    if completed >= token {
        return Ok(());
    }
    let wait = vk::SemaphoreWaitInfo::default()
        .semaphores(std::slice::from_ref(&semaphore))
        .values(std::slice::from_ref(&token));
    unsafe { device.wait_semaphores(&wait, u64::MAX) }.context("frame table row timeline wait")?;
    Ok(())
}

/// Wait until `semaphore` has reached the prior token, then claim the row with [`ROW_IN_USE`].
fn reserve_row_with_backpressure(
    ft: &ContextFrameTable,
    row: u32,
    device: &ash::Device,
    semaphore: vk::Semaphore,
) -> Result<()> {
    loop {
        let prev = ft.last_token_for_row[row as usize].load(Ordering::Acquire);
        if prev == ROW_IN_USE {
            std::thread::yield_now();
            continue;
        }
        if prev > 0 {
            wait_for_timeline(device, semaphore, prev)?;
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

/// Legacy path has no timeline; prior users must [`record_submission`](`0`) after `queue_wait_idle`.
fn reserve_row_legacy(ft: &ContextFrameTable, row: u32, ld: &LogicalDevice) -> Result<()> {
    loop {
        let prev = ft.last_token_for_row[row as usize].load(Ordering::Acquire);
        if prev == ROW_IN_USE {
            std::thread::yield_now();
            continue;
        }
        if prev > 0 {
            // Safety net if a prior legacy submit forgot to clear the token.
            let _guard = ld.queue_lock.lock().unwrap();
            unsafe { ld.device.queue_wait_idle(ld.queue) }.context("frame table legacy row wait")?;
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
/// that work's timeline value (bounded pipeline backpressure) before overwriting upload bytes.
fn write_staging_for_submission(
    contexts: &SharedContextMap,
    ld: &super::types::LogicalDevice,
    ctx: super::ContextHandle,
    ft: &ContextFrameTable,
    data: &[u32],
) -> Result<u32> {
    let sub = ft.submission_counter.fetch_add(1, Ordering::Relaxed);
    let row = sub % FRAME_TABLE_MAX_ROWS;
    if ft.pinned_rows.load(Ordering::Acquire) & (1 << row) != 0 {
        super::compute::evict_retained_pinning_row_for_context(contexts, ft, ld, ctx, row);
    }
    assert_not_pinned(ft, row)?;
    let (device, semaphore) = {
        let contexts_read = contexts.read().unwrap();
        let sc_arc = contexts_read
            .get(&ctx)
            .with_context(|| format!("frame table staging: invalid context {ctx}"))?;
        let sc = sc_arc.lock().unwrap();
        (ld.device.clone(), sc.timeline_semaphore)
    };
    reserve_row_with_backpressure(ft, row, &device, semaphore)?;
    write_row_payload(ft, data, row);
    Ok(row)
}

/// CPU staging write for legacy standalone render (no context eviction).
fn write_staging_standalone(ft: &ContextFrameTable, ld: &LogicalDevice, data: &[u32]) -> Result<u32> {
    let sub = ft.submission_counter.fetch_add(1, Ordering::Relaxed);
    let row = sub % FRAME_TABLE_MAX_ROWS;
    assert_not_pinned(ft, row)?;
    reserve_row_legacy(ft, row, ld)?;
    write_row_payload(ft, data, row);
    Ok(row)
}

/// CPU staging write + GPU copy prologue for legacy `render_to_target`.
pub(crate) fn record_prologue_legacy(
    frame_table: &ContextFrameTable,
    buffers: &SharedBufferTable,
    ld: &LogicalDevice,
    cmd: vk::CommandBuffer,
    data: &[u32],
) -> Result<u32> {
    let row = write_staging_standalone(frame_table, ld, data)?;
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    record_table_copies(
        frame_table,
        buffers,
        ld,
        cmd,
        TableCopyParams {
            row,
            copy_u32s,
            copy_selector: true,
            on_graphics_queue: true,
        },
    )?;
    Ok(row)
}

/// GPU resources used when recording a frame-table prologue into a command buffer.
pub(crate) struct PrologueRecording<'a> {
    pub frame_table: &'a ContextFrameTable,
    pub buffers: &'a SharedBufferTable,
    pub ld: &'a LogicalDevice,
    pub cmd: vk::CommandBuffer,
    pub on_graphics_queue: bool,
}

/// CPU staging write + GPU copy prologue.
///
/// The prologue copies the active row index into this context's selector buffer.
/// That buffer is shared across all command buffers on the context; correctness
/// assumes every submission for the context runs on the same FIFO queue so
/// command buffers retire in submit order and a later prologue cannot clobber
/// the selector while an earlier buffer's dispatches are still executing.
pub(crate) fn record_prologue(
    contexts: &SharedContextMap,
    ctx: super::ContextHandle,
    rec: PrologueRecording<'_>,
    data: &[u32],
) -> Result<u32> {
    let PrologueRecording {
        frame_table,
        buffers,
        ld,
        cmd,
        on_graphics_queue,
    } = rec;
    let row = write_staging_for_submission(contexts, ld, ctx, frame_table, data)?;
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    record_table_copies(
        frame_table,
        buffers,
        ld,
        cmd,
        TableCopyParams {
            row,
            copy_u32s,
            copy_selector: true,
            on_graphics_queue,
        },
    )?;
    Ok(row)
}

/// Refresh the active row in the device-local table without advancing the selector.
pub(crate) fn sync_table_row_to_device(
    frame_table: &SharedContextFrameTable,
    buffers: &SharedBufferTable,
    ld: &LogicalDevice,
    cmd: vk::CommandBuffer,
    data: &[u32],
    on_graphics_queue: bool,
) -> Result<()> {
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    let staging_ptr = frame_table.staging_mapped as *mut u32;
    let row = frame_table.submission_counter.load(Ordering::Relaxed).saturating_sub(1) % FRAME_TABLE_MAX_ROWS;
    unsafe {
        let payload_dst =
            staging_ptr.add(crate::frame_table::FRAME_TABLE_STAGING_SELECTOR_U32S + row as usize * row_u32s);
        std::ptr::copy_nonoverlapping(data.as_ptr(), payload_dst, copy_u32s);
        std::ptr::write(staging_ptr.add(row as usize), row);
    }
    record_table_copies(
        frame_table,
        buffers,
        ld,
        cmd,
        TableCopyParams {
            row,
            copy_u32s,
            copy_selector: false,
            on_graphics_queue,
        },
    )?;
    Ok(())
}

/// Lower render commands and build staging for standalone render passes (not graph submit).
pub(crate) fn prepare_render_commands(
    buffers: &SharedBufferTable,
    pipelines: &SharedPipelineTable,
    commands: &[crate::backend::RenderCommand],
) -> Result<(Vec<u32>, Vec<crate::backend::RenderCommand>, bool)> {
    use crate::backend::RenderCommand;
    use crate::frame_table::FrameTableStaging;

    crate::backend::with_layout_validation(|| {
        crate::backend::validate_render_pass_bind_resources(
            commands,
            |h| {
                pipelines
                    .read()
                    .unwrap()
                    .entries
                    .get(&h)
                    .map(|p| (p.binding_element_strides.clone(), p.shader_debug_name.clone()))
            },
            |h| buffers.read().unwrap().entries.get(&h).and_then(|b| b.element_stride),
        )
    })?;

    let mut staging = FrameTableStaging::new();
    let lowered = commands
        .iter()
        .map(|cmd| match cmd {
            RenderCommand::BindResources { buffers: handles } => {
                let indices: Vec<u32> = handles
                    .iter()
                    .map(|h| {
                        buffers
                            .read()
                            .unwrap()
                            .entries
                            .get(h)
                            .and_then(|b| b.bindless_index)
                            .with_context(|| format!("BindResources: buffer handle {h:?} has no bindless index"))
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

/// Merge graph-level staging with render-local staging for row refresh after an
/// earlier prologue in the same command buffer.
pub(crate) fn merge_staging_for_render_sync(graph: &[u32], render: &[u32]) -> Vec<u32> {
    let len = graph.len().max(render.len()).min(FRAME_TABLE_TABLE_U32S);
    let mut merged = vec![0u32; len];
    for (i, slot) in merged.iter_mut().enumerate().take(len) {
        *slot = graph.get(i).copied().unwrap_or(0);
        if render.get(i).is_some_and(|&v| v != 0) {
            *slot = render[i];
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_staging_overlays_render_indices_without_dropping_compute() {
        let mut graph = vec![0u32; FRAME_TABLE_TABLE_U32S];
        graph[0] = 7;
        let mut render = vec![0u32; FRAME_TABLE_TABLE_U32S];
        render[1] = 42;
        let merged = merge_staging_for_render_sync(&graph, &render);
        assert_eq!(merged[0], 7);
        assert_eq!(merged[1], 42);
    }
}

pub(crate) fn extract_staging_from_graph(commands: &[crate::backend::GraphCommand]) -> Option<std::sync::Arc<[u32]>> {
    commands.iter().find_map(|c| match c {
        crate::backend::GraphCommand::Compute(GpuCommand::FrameTableStaging { data }) => {
            Some(std::sync::Arc::clone(data))
        }
        _ => None,
    })
}
