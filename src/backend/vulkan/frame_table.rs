//! Vulkan frame-table buffers and prologue (staging upload + device-local copy).

use super::types::{self, BufferState, LogicalDevice, VulkanState};
use super::utils::find_memory_type;
use super::BufferHandle;
use crate::backend::GpuCommand;
use crate::frame_table::{
    FRAME_TABLE_DEVICE_SLOT, FRAME_TABLE_MAX_ROWS, FRAME_TABLE_ROW_STRIDE, FRAME_TABLE_SELECTOR_SLOT,
    FRAME_TABLE_STAGING_BYTES, FRAME_TABLE_TABLE_U32S, FRAME_TABLE_USER_SLOT_BASE,
};
use anyhow::{Context, Result};
use ash::vk;
use std::sync::atomic::{AtomicU32, Ordering};

/// Per-device frame-table GPU resources.
pub(crate) struct FrameTableDevice {
    pub selector: BufferHandle,
    pub device_table: BufferHandle,
    pub staging: vk::Buffer,
    pub staging_memory: vk::DeviceMemory,
    pub staging_mapped: usize,
    pub submission_counter: AtomicU32,
    /// Bitmask of rows pinned by retained command buffers (must not be overwritten).
    pub pinned_rows: AtomicU32,
    /// Last `submission_counter` value that wrote each row (ring reuse guard).
    pub last_sub_for_row: [AtomicU32; FRAME_TABLE_MAX_ROWS as usize],
}

/// Create frame-table buffers at reserved bindless storage slots 0 and 1.
pub(crate) fn init_device(
    state: &mut VulkanState,
    instance: &ash::Instance,
    device_handle: super::DeviceHandle,
    ld: &LogicalDevice,
) -> Result<()> {
    let selector =
        create_scattered_u32_buffer_at_slot(state, instance, device_handle, ld, FRAME_TABLE_SELECTOR_SLOT, 1)?;
    let device_table = create_scattered_u32_buffer_at_slot(
        state,
        instance,
        device_handle,
        ld,
        FRAME_TABLE_DEVICE_SLOT,
        FRAME_TABLE_TABLE_U32S as u32,
    )?;

    let (staging, staging_memory, staging_mapped) = create_upload_table_buffer(instance, ld)?;

    state.buffers.get_mut(&selector).unwrap().element_stride = Some(4);
    state.buffers.get_mut(&device_table).unwrap().element_stride = Some(4);

    {
        let dev = state.devices.get(&device_handle).context("init frame table")?;
        dev.ledger
            .lock()
            .unwrap()
            .resource_registry
            .ensure_storage_start(FRAME_TABLE_USER_SLOT_BASE);
    }

    state.frame_tables.insert(
        device_handle,
        FrameTableDevice {
            selector,
            device_table,
            staging,
            staging_memory,
            staging_mapped,
            submission_counter: AtomicU32::new(0),
            pinned_rows: AtomicU32::new(0),
            last_sub_for_row: std::array::from_fn(|_| AtomicU32::new(0)),
        },
    );

    Ok(())
}

/// Destroy frame-table staging resources owned outside `state.buffers`.
pub(crate) fn destroy_device(state: &mut VulkanState, device_handle: super::DeviceHandle, ld: &LogicalDevice) {
    if let Some(ft) = state.frame_tables.remove(&device_handle) {
        unsafe {
            ld.device.destroy_buffer(ft.staging, None);
            ld.device.free_memory(ft.staging_memory, None);
        }
    }
}

fn create_upload_table_buffer(
    instance: &ash::Instance,
    ld: &LogicalDevice,
) -> Result<(vk::Buffer, vk::DeviceMemory, usize)> {
    let size = FRAME_TABLE_STAGING_BYTES.max(256);
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

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

fn create_scattered_u32_buffer_at_slot(
    state: &mut VulkanState,
    instance: &ash::Instance,
    device_handle: super::DeviceHandle,
    ld: &LogicalDevice,
    bindless_slot: u32,
    num_u32s: u32,
) -> Result<BufferHandle> {
    let logical_size = (num_u32s as u64) * 4;
    let allocation_size = logical_size.max(256);

    let vk_usage =
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(allocation_size)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

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

    if let Some(descriptor_set) = ld.bindless_descriptor_set {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(logical_size.max(1));

        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(types::bindless_bindings::STORAGE_BUFFERS)
            .dst_array_element(bindless_slot)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe {
            ld.device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    }

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;
    state.buffers.insert(
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
            grant_texture_readback: None,
        },
    );
    Ok(handle)
}

fn prologue_pre_copy_barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    // Wait for prior submission's table reads before overwriting device-local rows.
    let mem_barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_GRAPHICS)
        .src_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
    let dep_info = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dep_info);
    }
}

fn prologue_post_copy_barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    let mem_barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_GRAPHICS)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
    let dep_info = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dep_info);
    }
}

fn record_table_copies(
    ft: &FrameTableDevice,
    buffers: &std::collections::HashMap<BufferHandle, BufferState>,
    ld: &LogicalDevice,
    cmd: vk::CommandBuffer,
    row: u32,
    copy_u32s: usize,
    copy_selector: bool,
) -> Result<()> {
    let device_table = buffers.get(&ft.device_table).context("device table")?;
    let selector_state = buffers.get(&ft.selector).context("selector")?;
    let row_bytes = (FRAME_TABLE_ROW_STRIDE as u64) * 4;
    let dest_offset = (row as u64) * row_bytes;
    let src_payload = crate::frame_table::staging_row_payload_byte_offset(row);
    let src_selector = crate::frame_table::staging_selector_byte_offset(row);

    prologue_pre_copy_barrier(&ld.device, cmd);
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
    prologue_post_copy_barrier(&ld.device, cmd);
    Ok(())
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

/// Row used by the most recent prologue (before counter bump on next submission).
pub(crate) fn last_prologue_row(ft: &FrameTableDevice) -> Option<u32> {
    let sub = ft.submission_counter.load(Ordering::Relaxed);
    if sub == 0 {
        None
    } else {
        Some((sub - 1) % FRAME_TABLE_MAX_ROWS)
    }
}

fn assert_row_available(ft: &FrameTableDevice, sub: u32, row: u32) -> Result<()> {
    let pinned = ft.pinned_rows.load(Ordering::Acquire);
    if pinned & (1 << row) != 0 {
        anyhow::bail!(
            "frame table row {row} is still pinned after evicting retained graphs \
             (sub={sub}); retained command list lifecycle bug"
        );
    }
    if sub >= FRAME_TABLE_MAX_ROWS {
        let prev = ft.last_sub_for_row[row as usize].load(Ordering::Acquire);
        let gap = sub - prev;
        if gap < FRAME_TABLE_MAX_ROWS {
            anyhow::bail!(
                "frame table row capacity exceeded: row {row} may still be in flight \
                 (sub={sub}, last_sub_for_row={prev}, need gap >= {FRAME_TABLE_MAX_ROWS})"
            );
        }
    }
    Ok(())
}

/// CPU staging write before a retained resubmit (row bump + table bytes; GPU copy is in the CB).
fn write_staging_for_submission_on(
    contexts: &std::collections::HashMap<super::ContextHandle, super::types::SharedSubmissionContext>,
    frame_tables: &std::collections::HashMap<super::DeviceHandle, FrameTableDevice>,
    device_handle: super::DeviceHandle,
    ft: &FrameTableDevice,
    data: &[u32],
) -> Result<u32> {
    let sub = ft.submission_counter.fetch_add(1, Ordering::Relaxed);
    let row = sub % FRAME_TABLE_MAX_ROWS;
    let row_pinned = ft.pinned_rows.load(Ordering::Acquire) & (1 << row) != 0;
    if row_pinned {
        super::compute::evict_retained_pinning_row(contexts, frame_tables, device_handle, row);
    }
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    let staging_ptr = ft.staging_mapped as *mut u32;
    assert_row_available(ft, sub, row)?;
    ft.last_sub_for_row[row as usize].store(sub, Ordering::Release);
    unsafe {
        let payload_dst =
            staging_ptr.add(crate::frame_table::FRAME_TABLE_STAGING_SELECTOR_U32S + row as usize * row_u32s);
        std::ptr::copy_nonoverlapping(data.as_ptr(), payload_dst, copy_u32s);
        std::ptr::write(staging_ptr.add(row as usize), row);
    }
    Ok(row)
}

/// CPU staging write + GPU copy prologue (split borrows for standalone render).
pub(crate) fn record_prologue_for_tables(
    contexts: &std::collections::HashMap<super::ContextHandle, super::types::SharedSubmissionContext>,
    frame_tables: &std::collections::HashMap<super::DeviceHandle, FrameTableDevice>,
    buffers: &std::collections::HashMap<BufferHandle, BufferState>,
    device_handle: super::DeviceHandle,
    ld: &LogicalDevice,
    cmd: vk::CommandBuffer,
    data: &[u32],
) -> Result<()> {
    let ft = frame_tables
        .get(&device_handle)
        .context("frame table not initialized")?;
    let row = write_staging_for_submission_on(contexts, frame_tables, device_handle, ft, data)?;
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    record_table_copies(ft, buffers, ld, cmd, row, copy_u32s, true)
}

/// CPU staging write + GPU copy prologue.
pub(crate) fn record_prologue(
    contexts: &std::collections::HashMap<super::ContextHandle, super::types::SharedSubmissionContext>,
    frame_tables: &std::collections::HashMap<super::DeviceHandle, FrameTableDevice>,
    buffers: &std::collections::HashMap<BufferHandle, BufferState>,
    device_handle: super::DeviceHandle,
    ld: &LogicalDevice,
    cmd: vk::CommandBuffer,
    data: &[u32],
) -> Result<()> {
    record_prologue_for_tables(contexts, frame_tables, buffers, device_handle, ld, cmd, data)
}

/// Refresh the active row in the device-local table without advancing the selector.
pub(crate) fn sync_table_row_to_device(
    state: &VulkanState,
    device_handle: super::DeviceHandle,
    ld: &LogicalDevice,
    cmd: vk::CommandBuffer,
    data: &[u32],
) -> Result<()> {
    let ft = state
        .frame_tables
        .get(&device_handle)
        .context("frame table not initialized")?;
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
    record_table_copies(ft, &state.buffers, ld, cmd, row, copy_u32s, false)
}

/// Lower render commands and build staging for standalone render passes (not graph submit).
pub(crate) fn prepare_render_commands(
    buffers: &std::collections::HashMap<BufferHandle, BufferState>,
    pipelines: &std::collections::HashMap<super::PipelineHandle, super::types::PipelineState>,
    commands: &[crate::backend::RenderCommand],
) -> Result<(Vec<u32>, Vec<crate::backend::RenderCommand>, bool)> {
    use crate::backend::RenderCommand;
    use crate::frame_table::FrameTableStaging;

    crate::backend::with_layout_validation(|| {
        crate::backend::validate_render_pass_bind_resources(
            commands,
            |h| {
                pipelines
                    .get(&h)
                    .map(|p| (p.binding_element_strides.clone(), p.shader_debug_name.clone()))
            },
            |h| buffers.get(&h).and_then(|b| b.element_stride),
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
