//! Compute pipeline and dispatch logic.

use super::super::shared;
use super::super::shared::DISPATCH_BATCH_STRIDE;
use super::staging;
use super::types::{ComputePipelineState, LogicalDevice, PipelineState, PushLayout, SlotKey};
use super::{BufferHandle, ComputePipelineHandle, DeviceHandle, PipelineHandle, RenderTargetHandle};
use crate::backend::{GpuCommand, GraphCommand, RenderCommand};
use crate::gpu_profiler::{self, DispatchGpuNs};
use crate::task_graph::{NodeAccessUnion, SlotUsageSet, UsageKindFlags};
use crate::timeline::TimelineValue;
use crate::tracy_zone;
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

fn slot_key_from_category(cat: crate::types::ResourceCategory, index: u32) -> Option<SlotKey> {
    use crate::types::ResourceCategory;
    match cat {
        ResourceCategory::Scattered => Some(SlotKey::StorageBuffer(index)),
        ResourceCategory::Broadcast => Some(SlotKey::UniformBuffer(index)),
        ResourceCategory::Texture => Some(SlotKey::SampledTexture(index)),
        ResourceCategory::StorageImage => Some(SlotKey::StorageImage(index)),
        ResourceCategory::Sampler => Some(SlotKey::Sampler(index)),
    }
}

fn buffer_stride_for_bindless_index(
    buffers: &HashMap<BufferHandle, super::types::BufferState>,
    device_handle: DeviceHandle,
    index: u32,
    cat: crate::types::ResourceCategory,
) -> Option<u32> {
    // Uniform and storage buffers use separate bindless array indices; both may be 0.
    for b in buffers.values() {
        if b.device_handle != device_handle {
            continue;
        }
        match cat {
            crate::types::ResourceCategory::Scattered if b.is_storage && b.bindless_index == Some(index) => {
                return b.element_stride;
            }
            crate::types::ResourceCategory::Broadcast if !b.is_storage && b.bindless_index == Some(index) => {
                return b.element_stride;
            }
            _ => {}
        }
    }
    None
}

fn collect_slots_from_raw_bind(indices: &[u32], categories: &[Option<crate::types::ResourceCategory>]) -> Vec<SlotKey> {
    let mut slots = Vec::new();
    for (i, &idx) in indices.iter().enumerate() {
        if let Some(Some(cat)) = categories.get(i) {
            if let Some(key) = slot_key_from_category(*cat, idx) {
                slots.push(key);
            }
        }
    }
    slots
}

fn collect_slot_keys_from_gpu_commands(
    commands: &[GpuCommand],
    compute_pipelines: &HashMap<ComputePipelineHandle, ComputePipelineState>,
    _buffers: &HashMap<BufferHandle, super::types::BufferState>,
) -> Vec<SlotKey> {
    let mut current_pipeline = None;
    let mut slots = Vec::new();
    for cmd in commands {
        match cmd {
            GpuCommand::SetPipeline(p) => current_pipeline = Some(*p),
            GpuCommand::BindResourcesRaw { indices, .. } => {
                if let Some(p) = current_pipeline.and_then(|h| compute_pipelines.get(&h)) {
                    slots.extend(collect_slots_from_raw_bind(indices, &p.push_constant_categories));
                }
            }
            GpuCommand::BindResourcesTyped { handles } => {
                for h in handles {
                    if let Some(key) = slot_key_from_category(h.category(), h.index()) {
                        slots.push(key);
                    }
                }
            }
            GpuCommand::DispatchBatch { arg_data, count, .. } => {
                if let Some(p) = current_pipeline.and_then(|h| compute_pipelines.get(&h)) {
                    let layout_size = std::mem::size_of::<PushLayout>();
                    for i in 0..*count as usize {
                        let base = i * DISPATCH_BATCH_STRIDE;
                        if base + layout_size <= arg_data.len() {
                            let layout: &PushLayout = bytemuck::from_bytes(&arg_data[base..base + layout_size]);
                            for (slot_i, &idx) in layout.bindless.iter().enumerate() {
                                if let Some(Some(cat)) = p.push_constant_categories.get(slot_i).copied() {
                                    if let Some(key) = slot_key_from_category(cat, idx as u32) {
                                        slots.push(key);
                                    }
                                }
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

fn collect_slot_keys_from_graph_commands(
    commands: &[GraphCommand],
    compute_pipelines: &HashMap<ComputePipelineHandle, ComputePipelineState>,
    pipelines: &HashMap<PipelineHandle, PipelineState>,
    buffers: &HashMap<BufferHandle, super::types::BufferState>,
) -> Vec<SlotKey> {
    let mut slots = Vec::new();
    let mut current_compute_pipeline = None;
    let mut current_render_pipeline = None;
    for gc in commands {
        match gc {
            GraphCommand::Compute(cmd) => match cmd {
                GpuCommand::SetPipeline(p) => current_compute_pipeline = Some(*p),
                GpuCommand::BindResourcesRaw { indices, .. } => {
                    if let Some(p) = current_compute_pipeline.and_then(|h| compute_pipelines.get(&h)) {
                        slots.extend(collect_slots_from_raw_bind(indices, &p.push_constant_categories));
                    }
                }
                GpuCommand::BindResourcesTyped { handles } => {
                    for h in handles {
                        if let Some(key) = slot_key_from_category(h.category(), h.index()) {
                            slots.push(key);
                        }
                    }
                }
                GpuCommand::DispatchBatch { arg_data, count, .. } => {
                    if let Some(p) = current_compute_pipeline.and_then(|h| compute_pipelines.get(&h)) {
                        let layout_size = std::mem::size_of::<PushLayout>();
                        for i in 0..*count as usize {
                            let base = i * DISPATCH_BATCH_STRIDE;
                            if base + layout_size <= arg_data.len() {
                                let layout: &PushLayout = bytemuck::from_bytes(&arg_data[base..base + layout_size]);
                                for (slot_i, &idx) in layout.bindless.iter().enumerate() {
                                    if let Some(Some(cat)) = p.push_constant_categories.get(slot_i).copied() {
                                        if let Some(key) = slot_key_from_category(cat, idx as u32) {
                                            slots.push(key);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            },
            GraphCommand::Render {
                commands: render_cmds, ..
            } => {
                for rc in render_cmds {
                    match rc {
                        RenderCommand::SetPipeline(p) => current_render_pipeline = Some(*p),
                        RenderCommand::BindResources { buffers: buf_handles } => {
                            for h in buf_handles {
                                if let Some(idx) = buffers.get(h).and_then(|b| b.bindless_index) {
                                    slots.push(SlotKey::StorageBuffer(idx));
                                }
                            }
                        }
                        RenderCommand::BindResourcesRaw { indices, .. } => {
                            if let Some(p) = current_render_pipeline.and_then(|h| pipelines.get(&h)) {
                                slots.extend(collect_slots_from_raw_bind(indices, &p.push_constant_categories));
                            }
                        }
                        RenderCommand::BindResourcesTyped { handles } => {
                            for h in handles {
                                if let Some(key) = slot_key_from_category(h.category(), h.index()) {
                                    slots.push(key);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    slots
}

/// Map a Koubaa producer/consumer usage set to Vulkan pipeline stage flags.
fn slot_usage_to_vk_stage(usage: &SlotUsageSet) -> vk::PipelineStageFlags2 {
    if usage.kinds.is_empty() {
        return vk::PipelineStageFlags2::ALL_COMMANDS;
    }
    let mut flags = vk::PipelineStageFlags2::empty();
    if usage.kinds.contains(UsageKindFlags::COMPUTE) {
        flags |= vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT;
    }
    if usage.kinds.contains(UsageKindFlags::TRANSFER) {
        flags |= vk::PipelineStageFlags2::TRANSFER;
    }
    if usage.kinds.contains(UsageKindFlags::RENDER) {
        flags |= vk::PipelineStageFlags2::ALL_GRAPHICS;
    }
    flags
}

/// Map a Koubaa producer/consumer usage set to Vulkan access flags.
fn slot_usage_to_vk_access(usage: &SlotUsageSet) -> vk::AccessFlags2 {
    if usage.kinds.is_empty() {
        return vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
    }
    let mut flags = vk::AccessFlags2::empty();
    if usage.kinds.contains(UsageKindFlags::COMPUTE) {
        if usage.access == NodeAccessUnion::Write {
            // we don't track writes independently of reads, so we have to be conservative here
            flags |= vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ;
        } else {
            flags |= vk::AccessFlags2::SHADER_READ;
        }
        // DRAW_INDIRECT reads are always reads.
        flags |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
    }
    if usage.kinds.contains(UsageKindFlags::TRANSFER) {
        if usage.access == NodeAccessUnion::Write {
            flags |= vk::AccessFlags2::TRANSFER_WRITE;
        } else {
            flags |= vk::AccessFlags2::TRANSFER_READ;
        }
    }
    if usage.kinds.contains(UsageKindFlags::RENDER) {
        flags |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    flags
}

/// Acquire a command buffer: recycle from the free-list or allocate a fresh one.
fn acquire_cmd_buffer(ld: &LogicalDevice, sc: &mut super::types::SubmissionContext) -> Result<vk::CommandBuffer> {
    if let Some(cb) = sc.free_cmd_buffers.pop() {
        return Ok(cb);
    }
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(sc.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cbs =
        unsafe { ld.device.allocate_command_buffers(&alloc_info) }.context("Failed to allocate command buffer")?;
    Ok(cbs[0])
}

#[derive(Debug)]
struct VulkanGpuProfilePool {
    pool: vk::QueryPool,
    query_count: u32,
    dispatch_labels: Vec<Option<&'static str>>,
    period_ns: f32,
    valid_bits: u32,
}

fn collect_dispatch_labels_compute(commands: &[GpuCommand]) -> (usize, Vec<Option<&'static str>>) {
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

fn collect_dispatch_labels_graph(commands: &[GraphCommand]) -> (usize, Vec<Option<&'static str>>) {
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

unsafe fn create_vulkan_gpu_profile_pool(
    ld: &LogicalDevice,
    defer_present: bool,
    dispatch_count: usize,
    dispatch_labels: Vec<Option<&'static str>>,
) -> Result<Option<VulkanGpuProfilePool>> {
    if defer_present || !gpu_profiler::gpu_profile_enabled() || !ld.vk_timestamp_compute_and_graphics {
        return Ok(None);
    }
    debug_assert_eq!(dispatch_labels.len(), dispatch_count);
    let query_count = 2u32.saturating_add((dispatch_count as u32).saturating_mul(2));
    let pool = ld.device.create_query_pool(
        &vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(query_count),
        None,
    )?;
    Ok(Some(VulkanGpuProfilePool {
        pool,
        query_count,
        dispatch_labels,
        period_ns: ld.vk_timestamp_period_ns,
        valid_bits: 64,
    }))
}

fn vulkan_decode_duration_ns(start: u64, end: u64, valid_bits: u32, period_ns: f32) -> u64 {
    let mask = if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };
    let a = start & mask;
    let b = end & mask;
    let delta = b.wrapping_sub(a);
    let ns_f = (delta as f64) * f64::from(period_ns);
    if ns_f <= 0.0 {
        0
    } else if ns_f >= u64::MAX as f64 {
        u64::MAX
    } else {
        ns_f as u64
    }
}

unsafe fn vulkan_finish_gpu_profile(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    device: &ash::Device,
    timeline_sem: vk::Semaphore,
    signal_value: TimelineValue,
    cmd: vk::CommandBuffer,
    profile: VulkanGpuProfilePool,
) -> Result<()> {
    let wait = vk::SemaphoreWaitInfo::default()
        .semaphores(std::slice::from_ref(&timeline_sem))
        .values(std::slice::from_ref(&signal_value));
    if let Err(e) = device.wait_semaphores(&wait, u64::MAX) {
        unsafe {
            device.destroy_query_pool(profile.pool, None);
        }
        if e == vk::Result::ERROR_DEVICE_LOST {
            state.device_lost.store(true, Ordering::Relaxed);
        }
        return Err(anyhow::anyhow!("wait_semaphores (gpu profiling): {:?}", e));
    }

    let mut raw = vec![0u64; profile.query_count as usize];
    if let Err(e) = device.get_query_pool_results(
        profile.pool,
        0,
        &mut raw,
        vk::QueryResultFlags::WAIT | vk::QueryResultFlags::TYPE_64,
    ) {
        unsafe {
            device.destroy_query_pool(profile.pool, None);
        }
        return Err(anyhow::anyhow!("get_query_pool_results: {:?}", e));
    }

    let cb_ns = vulkan_decode_duration_ns(raw[0], raw[1], profile.valid_bits, profile.period_ns);
    gpu_profiler::log_cb_timing("vulkan", signal_value, cb_ns as f64 / 1_000_000.0);

    let n = profile.dispatch_labels.len();
    if n > 0 {
        let mut dispatches = Vec::with_capacity(n);
        for i in 0..n {
            let si = 2 + 2 * i;
            let ns = vulkan_decode_duration_ns(raw[si], raw[si + 1], profile.valid_bits, profile.period_ns);
            let label = profile.dispatch_labels[i].unwrap_or("dispatch");
            dispatches.push(DispatchGpuNs { label, gpu_ns: ns });
        }
        gpu_profiler::log_dispatch_timings("vulkan", signal_value, &dispatches);
    }

    device.destroy_query_pool(profile.pool, None);
    reap_timeline_cmd_buffers_up_to(state, ctx, signal_value);
    let _ = cmd;
    Ok(())
}

/// Returns the GPU-completed timeline value for a single context by reading its
/// timeline semaphore counter directly, without consulting any other context.
///
/// Used on the submit hot path to replace `device_retired` (max-over-contexts)
/// as the reclaim gate. Because all contexts submit to the same `vk::Queue`,
/// when this context's semaphore reaches V every submit with a global timeline
/// value ≤ V — including those from other contexts — has already completed on
/// that queue. Draining with V is therefore safe and never creates a
/// cross-context dependency.
pub(super) fn ctx_completed_value(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    device_handle: super::DeviceHandle,
) -> u64 {
    let sem = state.contexts.get(&ctx).map(|sc| sc.lock().unwrap().timeline_semaphore);
    let dev = state.devices.get(&device_handle).map(|ld| &ld.device);
    match (dev, sem) {
        (Some(dev), Some(sem)) => unsafe { dev.get_semaphore_counter_value(sem).unwrap_or(0) },
        _ => 0,
    }
}

/// Reap fences that have already signaled from the compute fence pool.
///
/// Without this, every `submit_graph` that doesn't have a paired `wait_fence`
/// (the common ekrano pattern: only the *last* timeline value in a recording is
/// waited on; the intermediate ones are silently dropped) leaks a `VkFence` +
/// `VkCommandBuffer` into the pool. Over a few thousand frames the pool grows
/// large enough that the driver's command-pool / fence-pool internal arenas
/// fall over, the device is lost (`ERROR_DEVICE_LOST` on the next
/// `queue_submit`), and the unbounded HashMap teardown corrupts the heap on
/// shutdown.
fn reap_signaled_fences(state: &super::types::VulkanState) {
    let signaled: Vec<u64> = {
        let pool = state.compute_fence_pool.lock().unwrap();
        pool.iter()
            .filter_map(|(token, (device_handle, fence, _))| {
                let logical_device = state.devices.get(device_handle)?;
                let signaled = unsafe { logical_device.device.get_fence_status(*fence) }.unwrap_or(false);
                if signaled {
                    Some(*token)
                } else {
                    None
                }
            })
            .collect()
    };

    let mut pool = state.compute_fence_pool.lock().unwrap();
    for token in signaled {
        if let Some((device_handle, fence, cmd_buf)) = pool.remove(&token) {
            if let Some(logical_device) = state.devices.get(&device_handle) {
                unsafe {
                    if let Some(cb) = cmd_buf {
                        logical_device
                            .device
                            .free_command_buffers(logical_device.command_pool, &[cb]);
                    }
                    logical_device.device.destroy_fence(fence, None);
                }
            }
            // Texture staging for fence-pool submissions is pooled via
            // TextureStagingPool::release with the fence token as the timeline value;
            // reclaim happens in the next submit cycle once the pool sees completion.
        }
    }
}

/// Create a compute pipeline.
pub(super) fn create(
    devices: &HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    compute_pipelines: &mut HashMap<ComputePipelineHandle, ComputePipelineState>,
    next_compute_pipeline_handle: &mut ComputePipelineHandle,
    device_handle: DeviceHandle,
    cs_module: vk::ShaderModule,
    shader_debug_name: String,
) -> Result<ComputePipelineHandle> {
    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

    // Always use bindless pipeline layout
    let pipeline_layout = logical_device
        .bindless_pipeline_layout
        .context("Bindless pipeline layout required")?;
    let owns_layout = false; // Don't own - global bindless layout

    // Compute shader stage
    let cs_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(cs_module)
        .name(c"main");

    // Pipeline robustness (core in Vulkan 1.4): OOB descriptor access returns zero
    let mut robustness = vk::PipelineRobustnessCreateInfoEXT::default()
        .storage_buffers(vk::PipelineRobustnessBufferBehaviorEXT::ROBUST_BUFFER_ACCESS_2)
        .uniform_buffers(vk::PipelineRobustnessBufferBehaviorEXT::ROBUST_BUFFER_ACCESS_2)
        .images(vk::PipelineRobustnessImageBehaviorEXT::ROBUST_IMAGE_ACCESS_2);

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(cs_stage)
        .layout(pipeline_layout)
        .push_next(&mut robustness);

    let pipelines = unsafe {
        logical_device
            .device
            .create_compute_pipelines(logical_device.pipeline_cache, &[pipeline_info], None)
    }
    .map_err(|(_, e)| anyhow::anyhow!("Failed to create compute pipeline: {:?}", e))?;

    let handle = *next_compute_pipeline_handle;
    *next_compute_pipeline_handle += 1;

    compute_pipelines.insert(
        handle,
        ComputePipelineState {
            device_handle,
            pipeline: pipelines[0],
            layout: pipeline_layout,
            owns_layout,
            parameter_block_layouts: Vec::new(),
            push_constant_categories: Vec::new(),
            binding_element_strides: Vec::new(),
            shader_debug_name,
        },
    );

    tracing::debug!("Created compute pipeline (handle={})", handle);
    Ok(handle)
}

/// Destroy a compute pipeline.
pub(super) fn destroy(
    devices: &HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    compute_pipelines: &mut HashMap<ComputePipelineHandle, ComputePipelineState>,
    pipeline_handle: ComputePipelineHandle,
) {
    if let Some(pipeline) = compute_pipelines.remove(&pipeline_handle) {
        if let Some(logical_device) = devices.get(&pipeline.device_handle) {
            unsafe {
                logical_device.device.device_wait_idle().ok();
                logical_device.device.destroy_pipeline(pipeline.pipeline, None);
                // Only destroy layout if we own it (not the global bindless layout)
                if pipeline.owns_layout {
                    logical_device.device.destroy_pipeline_layout(pipeline.layout, None);
                }
            }
        }
    }
}

/// Submit compute commands without blocking. Returns a fence token for polling/waiting.
/// Submit a batch of GPU commands as a single compute submission.
pub(super) fn submit(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    commands: &[GpuCommand],
) -> Result<TimelineValue> {
    let device_handle = state
        .contexts
        .get(&ctx)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
        .device;
    let _tz = tracy_zone!("vk.submit");
    // Detect WriteBuffer up-front so we can skip all the staging-belt /
    // fence-pool maintenance when this submit has no host→GPU uploads.
    // The next WriteBuffer-bearing submit will reclaim/reap any debris.
    let has_write_buffer = commands.iter().any(|c| matches!(c, GpuCommand::WriteBuffer { .. }));

    // Reap any previously-submitted fences that have already signaled. Keeps
    // the pool bounded when callers (ekrano) don't wait on every intermediate submit.
    // Belt before reap: need live VkFence handles to poll completion.
    //
    // For timeline-keyed staging chunks (standalone-submit path) we also need the
    // current device timeline counter so `reclaim` knows which chunks are safe to
    // recycle without reaching into `compute_fence_pool`.
    let has_write_texture = commands.iter().any(|c| {
        matches!(
            c,
            GpuCommand::WriteTexture { .. } | GpuCommand::WriteTextureRegion { .. }
        )
    });

    if has_write_buffer || has_write_texture {
        let _rz = tracy_zone!("vk.submit.belt_reclaim");
        let completed_timeline = ctx_completed_value(state, ctx, device_handle);

        if has_write_buffer {
            if let Some(sc_arc) = state.contexts.get(&ctx) {
                sc_arc.lock().unwrap().staging_belt.reclaim(
                    &state.compute_fence_pool,
                    &state.devices,
                    completed_timeline,
                )?;
            }
            reap_signaled_fences(state);
        }

        if has_write_texture {
            if let Some(sc_arc) = state.contexts.get(&ctx) {
                sc_arc.lock().unwrap().texture_staging_pool.reclaim(completed_timeline);
            }
        }
    }

    // Belt slices for DEVICE_LOCAL WriteBuffer copies (same iteration order as command loop).
    let mut belt_slices: Vec<(vk::Buffer, u64)> = Vec::new();

    // Pre-pass: stage CPU data for WriteBuffer commands (needs mutable belt
    // access before we borrow state immutably for the command loop).
    if has_write_buffer {
        for command in commands {
            if let GpuCommand::WriteBuffer {
                buffer: buf_handle,
                offset,
                data,
            } = command
            {
                // Extract Copy fields from the buffer state so the borrow ends
                // before we take a mutable borrow of state.contexts for the belt.
                let (host_mapped, is_storage, buf_device, buf_memory) = {
                    let buf = state
                        .buffers
                        .get(buf_handle)
                        .context("WriteBuffer: invalid buffer handle")?;
                    (buf.host_mapped, buf.is_storage, buf.device_handle, buf.memory)
                };
                if let Some(base) = host_mapped {
                    let p = base as *mut u8;
                    unsafe {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), p.add(*offset as usize), data.len());
                    }
                } else if !is_storage {
                    let dev = state.devices.get(&buf_device).context("WriteBuffer: device invalid")?;
                    unsafe {
                        let ptr = dev
                            .map_memory2(buf_memory, *offset, data.len() as u64)
                            .context("WriteBuffer: map failed")?;
                        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
                        dev.unmap_memory2(buf_memory).context("WriteBuffer: unmap failed")?;
                    }
                } else {
                    let dev = state.devices.get(&buf_device).context("WriteBuffer: device invalid")?;
                    let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
                    let mut sc = sc_arc.lock().unwrap();
                    let (stg_buf, stg_off) = sc.staging_belt.write(&state.instance, dev, data)?;
                    belt_slices.push((stg_buf, stg_off));
                }
            }
        }
    }

    if commands.is_empty() {
        let signal_value = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?
            .timeline_next
            .fetch_add(1, Ordering::Relaxed);
        let timeline_sem = state
            .contexts
            .get(&ctx)
            .context("Invalid context handle")?
            .lock()
            .unwrap()
            .timeline_semaphore;
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        let queue = ld.queue;
        let queue_lock = std::sync::Arc::clone(&ld.queue_lock);
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline_sem)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let submit_info2 = vk::SubmitInfo2::default().signal_semaphore_infos(std::slice::from_ref(&signal_info));
        let r = {
            let _queue_guard = queue_lock.lock().unwrap();
            unsafe {
                ld.device
                    .queue_submit2(queue, std::slice::from_ref(&submit_info2), vk::Fence::null())
            }
        };
        r.context("Failed queue_submit2 for empty compute submit")?;
        {
            let completed = ctx_completed_value(state, ctx, device_handle);
            let ctx_batch: Vec<_> = state
                .contexts
                .get(&ctx)
                .map(|sc| sc.lock().unwrap().deletion_queue.drain_up_to(completed))
                .unwrap_or_default();
            if let Some(ld) = state.devices.get(&device_handle) {
                let ledger_arc = std::sync::Arc::clone(&ld.ledger);
                let mut ledger = ledger_arc.lock().unwrap();
                for r in ctx_batch {
                    super::types::destroy_pending_deletion(ld, &mut ledger, r);
                }
            }
        }
        if let Some(sc_arc) = state.contexts.get(&ctx) {
            sc_arc.lock().unwrap().last_submitted_seq = signal_value;
        }
        return Ok(signal_value);
    }

    let compute_pipelines = &state.compute_pipelines;
    let buffers = &state.buffers;

    let mut texture_upload_scratch: Vec<super::texture::ComputeTextureScratch> = Vec::new();
    for command in commands {
        match command {
            GpuCommand::WriteTexture {
                texture,
                data,
                width,
                height,
            } => {
                let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
                let mut sc_guard = sc_arc.lock().unwrap();
                let pool = &mut sc_guard.texture_staging_pool;
                texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                    &state.instance,
                    &state.devices,
                    &state.textures,
                    pool,
                    *texture,
                    data,
                    0,
                    0,
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
                let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
                let mut sc_guard = sc_arc.lock().unwrap();
                let pool = &mut sc_guard.texture_staging_pool;
                texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                    &state.instance,
                    &state.devices,
                    &state.textures,
                    pool,
                    *texture,
                    data,
                    *x,
                    *y,
                    *width,
                    *height,
                )?);
            }
            _ => {}
        }
    }

    let (dispatch_count, dispatch_labels) = collect_dispatch_labels_compute(commands);
    let vk_gpu_profile = unsafe {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        create_vulkan_gpu_profile_pool(ld, false, dispatch_count, dispatch_labels)?
    };

    let mut vk_gpu_profile = vk_gpu_profile;

    let cmd = {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
        let mut sc = sc_arc.lock().unwrap();
        let cb = acquire_cmd_buffer(ld, &mut sc)?;
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(e) = unsafe { ld.device.begin_command_buffer(cb, &begin_info) } {
            sc.free_cmd_buffers.push(cb);
            return Err(anyhow::anyhow!("Failed to begin command buffer: {:?}", e));
        }
        cb
    };

    let (cmd, belt_idx, _texture_upload_idx) = {
        let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

        // Cross-submission acquire: make prior submit's writes visible to this
        // CB's reads. Same-queue execution ordering is guaranteed by Vulkan but
        // memory visibility is not.
        unsafe {
            let acquire = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::TRANSFER
                        | vk::PipelineStageFlags2::DRAW_INDIRECT,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::TRANSFER_READ
                        | vk::AccessFlags2::INDIRECT_COMMAND_READ,
                );
            let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&acquire));
            logical_device.device.cmd_pipeline_barrier2(cmd, &dep);

            if let (Some(bindless_set), Some(bindless_layout)) = (
                logical_device.bindless_descriptor_set,
                logical_device.bindless_pipeline_layout,
            ) {
                logical_device.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    bindless_layout,
                    0,
                    std::slice::from_ref(&bindless_set),
                    &[],
                );
            }
        }

        let mut vk_dispatch_idx = 0usize;
        if let Some(ref prof) = vk_gpu_profile {
            unsafe {
                logical_device
                    .device
                    .cmd_reset_query_pool(cmd, prof.pool, 0, prof.query_count);
                logical_device
                    .device
                    .cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, prof.pool, 0);
            }
        }

        // Track current pipeline for resource slot binding
        let mut current_pipeline: Option<ComputePipelineHandle> = None;
        let mut belt_idx = 0usize;
        let mut texture_upload_idx = 0usize;

        // Process commands (same logic as dispatch)
        for command in commands {
            match command {
                GpuCommand::FrameTableStaging { .. } => {}
                GpuCommand::SetPipeline(handle) => {
                    let _tz = tracy_zone!("vk.set_pipeline");
                    if let Some(pipeline_state) = compute_pipelines.get(handle) {
                        unsafe {
                            logical_device.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::COMPUTE,
                                pipeline_state.pipeline,
                            );
                        }
                        current_pipeline = Some(*handle);
                    }
                }
                GpuCommand::BindResourcesRaw {
                    indices: raw_indices,
                    user: raw_user,
                    ..
                } => {
                    if let Some(pipeline) = current_pipeline.and_then(|p| compute_pipelines.get(&p)) {
                        crate::backend::validate_raw_binding_strides(
                            raw_indices,
                            &pipeline.push_constant_categories,
                            &pipeline.binding_element_strides,
                            |idx, cat| buffer_stride_for_bindless_index(buffers, device_handle, idx, cat),
                            &pipeline.shader_debug_name,
                        )?;
                        let mut layout = PushLayout::default();
                        shared::fill_raw(&mut layout, raw_indices, raw_user);
                        unsafe {
                            logical_device.device.cmd_push_constants(
                                cmd,
                                pipeline.layout,
                                vk::ShaderStageFlags::ALL,
                                0,
                                layout.as_bytes(),
                            );
                        }
                    }
                }
                GpuCommand::BindResourcesTyped { handles: typed_handles } => {
                    if let Some(pipeline) = current_pipeline.and_then(|p| compute_pipelines.get(&p)) {
                        crate::backend::validate_typed_push_constants(
                            typed_handles,
                            &pipeline.push_constant_categories,
                            &pipeline.shader_debug_name,
                        )?;
                        let mut layout = PushLayout::default();
                        shared::fill_typed(&mut layout, typed_handles.iter().copied());
                        unsafe {
                            logical_device.device.cmd_push_constants(
                                cmd,
                                pipeline.layout,
                                vk::ShaderStageFlags::ALL,
                                0,
                                layout.as_bytes(),
                            );
                        }
                    }
                }
                GpuCommand::Dispatch {
                    label: _label,
                    workgroups_x,
                    workgroups_y,
                    workgroups_z,
                } => {
                    let _tz = tracy_zone!("vk.dispatch");
                    unsafe {
                        if let Some(ref prof) = vk_gpu_profile {
                            let base = 2u32 + (vk_dispatch_idx as u32) * 2;
                            logical_device.device.cmd_write_timestamp2(
                                cmd,
                                vk::PipelineStageFlags2::TOP_OF_PIPE,
                                prof.pool,
                                base,
                            );
                        }
                        logical_device
                            .device
                            .cmd_dispatch(cmd, *workgroups_x, *workgroups_y, *workgroups_z);
                        if let Some(ref prof) = vk_gpu_profile {
                            let base = 2u32 + (vk_dispatch_idx as u32) * 2;
                            logical_device.device.cmd_write_timestamp2(
                                cmd,
                                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                                prof.pool,
                                base + 1,
                            );
                        }
                        vk_dispatch_idx += 1;
                    }
                }
                GpuCommand::DispatchBatch {
                    label: _,
                    arg_data,
                    count,
                } => {
                    let _tz = tracy_zone!("vk.dispatch_batch");
                    let push_size = std::mem::size_of::<crate::backend::shared::PushLayout>();
                    let stride = crate::backend::shared::DISPATCH_BATCH_STRIDE;
                    let pipeline_layout = current_pipeline
                        .and_then(|h| compute_pipelines.get(&h))
                        .map(|p| p.layout);
                    for i in 0..*count as usize {
                        let base = i * stride;
                        let layout_bytes = &arg_data[base..base + push_size];
                        let wg_off = base + push_size;
                        let wg_x = u32::from_ne_bytes(arg_data[wg_off..wg_off + 4].try_into().unwrap());
                        let wg_y = u32::from_ne_bytes(arg_data[wg_off + 4..wg_off + 8].try_into().unwrap());
                        let wg_z = u32::from_ne_bytes(arg_data[wg_off + 8..wg_off + 12].try_into().unwrap());
                        unsafe {
                            if let Some(layout) = pipeline_layout {
                                logical_device.device.cmd_push_constants(
                                    cmd,
                                    layout,
                                    vk::ShaderStageFlags::ALL,
                                    0,
                                    layout_bytes,
                                );
                            }
                            logical_device.device.cmd_dispatch(cmd, wg_x, wg_y, wg_z);
                        }
                        vk_dispatch_idx += 1;
                    }
                }
                GpuCommand::DispatchIndirect {
                    label: _label,
                    buffer,
                    offset,
                } => {
                    let _tz = tracy_zone!("vk.dispatch_indirect");
                    let buf_state = buffers.get(buffer).context("DispatchIndirect: invalid buffer handle")?;
                    unsafe {
                        if let Some(ref prof) = vk_gpu_profile {
                            let base = 2u32 + (vk_dispatch_idx as u32) * 2;
                            logical_device.device.cmd_write_timestamp2(
                                cmd,
                                vk::PipelineStageFlags2::TOP_OF_PIPE,
                                prof.pool,
                                base,
                            );
                        }
                        logical_device
                            .device
                            .cmd_dispatch_indirect(cmd, buf_state.buffer, *offset);

                        if let Some(ref prof) = vk_gpu_profile {
                            let base = 2u32 + (vk_dispatch_idx as u32) * 2;
                            logical_device.device.cmd_write_timestamp2(
                                cmd,
                                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                                prof.pool,
                                base + 1,
                            );
                        }
                    }
                    vk_dispatch_idx += 1;
                }
                GpuCommand::Barrier => {
                    let _tz = tracy_zone!("vk.barrier");
                    unsafe {
                        let mem_barrier = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
                        let dep_info =
                            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                    }
                }
                GpuCommand::ResourceBarrier {
                    buffers: buf_entries,
                    textures: tex_entries,
                } => {
                    let _tz = tracy_zone!("vk.resource_barrier");
                    unsafe {
                        let buf_barriers: Vec<vk::BufferMemoryBarrier2> = buf_entries
                            .iter()
                            .filter_map(|(h, usage)| {
                                buffers.get(h).map(|bs| {
                                    vk::BufferMemoryBarrier2::default()
                                        .src_stage_mask(slot_usage_to_vk_stage(&usage.src))
                                        .src_access_mask(slot_usage_to_vk_access(&usage.src))
                                        .dst_stage_mask(slot_usage_to_vk_stage(&usage.dst))
                                        .dst_access_mask(slot_usage_to_vk_access(&usage.dst))
                                        .buffer(bs.buffer)
                                        .offset(0)
                                        .size(vk::WHOLE_SIZE)
                                })
                            })
                            .collect();
                        // Textures: we don't track per-image Vulkan layout in TextureState,
                        // so use a global memory barrier per texture entry. No layout
                        // transition needed — just execution + memory dependency.
                        let tex_mem: Vec<vk::MemoryBarrier2> = tex_entries
                            .iter()
                            .map(|(_, usage)| {
                                vk::MemoryBarrier2::default()
                                    .src_stage_mask(slot_usage_to_vk_stage(&usage.src))
                                    .src_access_mask(slot_usage_to_vk_access(&usage.src))
                                    .dst_stage_mask(slot_usage_to_vk_stage(&usage.dst))
                                    .dst_access_mask(slot_usage_to_vk_access(&usage.dst))
                            })
                            .collect();
                        let dep_info = if tex_mem.is_empty() {
                            vk::DependencyInfo::default().buffer_memory_barriers(&buf_barriers)
                        } else {
                            vk::DependencyInfo::default()
                                .buffer_memory_barriers(&buf_barriers)
                                .memory_barriers(&tex_mem)
                        };
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                    }
                }
                GpuCommand::ClearBuffer { buffer, offset, size } => {
                    let _tz = tracy_zone!("vk.clear_buffer");
                    let buf_state = buffers.get(buffer).context("ClearBuffer: invalid buffer handle")?;
                    let clear_size = if *size == 0 {
                        buf_state.size.saturating_sub(*offset)
                    } else {
                        *size
                    };
                    if clear_size > 0 {
                        unsafe {
                            logical_device
                                .device
                                .cmd_fill_buffer(cmd, buf_state.buffer, *offset, clear_size, 0);

                            let mem_barrier = vk::MemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
                            let dep_info =
                                vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                            logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                        }
                    }
                }
                GpuCommand::WriteBuffer {
                    buffer: buf_handle,
                    offset,
                    data,
                } => {
                    let _tz = tracy_zone!("vk.write_buffer");
                    let buf_state = buffers.get(buf_handle).context("WriteBuffer: invalid buffer handle")?;
                    // HOST_VISIBLE / CPU_READABLE paths were handled in the pre-pass;
                    // DEVICE_LOCAL storage uses the staging belt (see pre-pass).
                    if buf_state.is_storage && buf_state.host_mapped.is_none() {
                        let (stg, stg_off) = belt_slices
                            .get(belt_idx)
                            .context("WriteBuffer: belt slice missing (internal error)")?;
                        belt_idx += 1;
                        let region = vk::BufferCopy {
                            src_offset: *stg_off,
                            dst_offset: *offset,
                            size: data.len() as u64,
                        };
                        unsafe {
                            logical_device.device.cmd_copy_buffer(
                                cmd,
                                *stg,
                                buf_state.buffer,
                                std::slice::from_ref(&region),
                            );

                            let mem_barrier = vk::MemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
                            let dep_info =
                                vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                            logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                        }
                    }
                }
                GpuCommand::WriteTexture { .. } | GpuCommand::WriteTextureRegion { .. } => {
                    let _tz = tracy_zone!("vk.write_texture");
                    let scratch = texture_upload_scratch
                        .get(texture_upload_idx)
                        .context("WriteTexture: scratch missing (internal)")?;
                    texture_upload_idx += 1;
                    super::texture::record_compute_texture_upload(&state.devices, &state.textures, cmd, scratch)?;
                }
                GpuCommand::CopyTexture { src, dst } => {
                    let _tz = tracy_zone!("vk.copy_texture");
                    let (src_image, width, height) = {
                        let ts = state.textures.get(src).context("CopyTexture: src texture not found")?;
                        (ts.image, ts.width, ts.height)
                    };
                    let dst_image = state
                        .textures
                        .get(dst)
                        .context("CopyTexture: dst texture not found")?
                        .image;

                    unsafe {
                        // Barrier: ensure compute writes to src are visible; dst may be
                        // written by compute, so also synchronise its prior writes.
                        let mem_barrier = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE);
                        let dep_info =
                            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);

                        // Both UAV textures are in GENERAL layout — copy is valid.
                        let region = vk::ImageCopy {
                            src_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            src_offset: vk::Offset3D::default(),
                            dst_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            dst_offset: vk::Offset3D::default(),
                            extent: vk::Extent3D {
                                width,
                                height,
                                depth: 1,
                            },
                        };
                        logical_device.device.cmd_copy_image(
                            cmd,
                            src_image,
                            vk::ImageLayout::GENERAL,
                            dst_image,
                            vk::ImageLayout::GENERAL,
                            std::slice::from_ref(&region),
                        );

                        // Barrier: make copy writes visible to subsequent compute/transfer.
                        let mem_barrier2 = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_COMMANDS,
                            )
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
                        let dep_info2 =
                            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier2));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info2);
                    }
                }
                GpuCommand::CopyRenderTarget { src, dst } => {
                    let _tz = tracy_zone!("vk.copy_render_target");
                    let (src_image, width, height) = {
                        let rt = state
                            .render_targets
                            .get(src)
                            .context("CopyRenderTarget: src render target not found")?;
                        (rt.image, rt.width, rt.height)
                    };
                    let dst_image = state
                        .textures
                        .get(dst)
                        .context("CopyRenderTarget: dst texture not found")?
                        .image;

                    unsafe {
                        let mem_barrier = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
                        let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep);

                        let region = vk::ImageCopy {
                            src_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            src_offset: vk::Offset3D::default(),
                            dst_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            dst_offset: vk::Offset3D::default(),
                            extent: vk::Extent3D {
                                width,
                                height,
                                depth: 1,
                            },
                        };
                        logical_device.device.cmd_copy_image(
                            cmd,
                            src_image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            dst_image,
                            vk::ImageLayout::GENERAL,
                            std::slice::from_ref(&region),
                        );

                        let mem_barrier2 = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_COMMANDS,
                            )
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
                        let dep_info2 =
                            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier2));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info2);
                    }
                }
            }
        }

        debug_assert_eq!(
            texture_upload_idx,
            texture_upload_scratch.len(),
            "WriteTexture commands mismatch texture scratch pre-pass"
        );

        // Release barrier: make this CB's writes available to subsequent submits.
        unsafe {
            let release = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::TRANSFER
                        | vk::PipelineStageFlags2::DRAW_INDIRECT
                        | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::TRANSFER_READ
                        | vk::AccessFlags2::INDIRECT_COMMAND_READ
                        | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                );
            let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&release));
            logical_device.device.cmd_pipeline_barrier2(cmd, &dep);
        }

        if let Some(ref prof) = vk_gpu_profile {
            unsafe {
                logical_device
                    .device
                    .cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, prof.pool, 1);
            }
        }

        // End command buffer
        if let Err(e) = unsafe { logical_device.device.end_command_buffer(cmd) } {
            unsafe {
                logical_device
                    .device
                    .free_command_buffers(logical_device.command_pool, &[cmd]);
            }
            return Err(anyhow::anyhow!("Failed to end command buffer: {:?}", e));
        }

        (cmd, belt_idx, texture_upload_idx)
    };

    // Standalone submit: signal device timeline semaphore (Vulkan 1.2+).
    let signal_value = {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        ld.timeline_next.fetch_add(1, Ordering::Relaxed)
    };

    let used_slots = collect_slot_keys_from_gpu_commands(commands, &state.compute_pipelines, &state.buffers);
    if let Some(ld) = state.devices.get(&device_handle) {
        ld.ledger
            .lock()
            .unwrap()
            .record_slot_usage(ctx, signal_value, used_slots);
    }

    let timeline_sem = state
        .contexts
        .get(&ctx)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
        .timeline_semaphore;
    let submit_device_core = state.devices.get(&device_handle).context("Invalid device handle")?;
    let queue_lock = std::sync::Arc::clone(&submit_device_core.queue_lock);
    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
    let signal_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(timeline_sem)
        .value(signal_value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let submit_info2 = vk::SubmitInfo2::default()
        .command_buffer_infos(std::slice::from_ref(&cmd_info))
        .signal_semaphore_infos(std::slice::from_ref(&signal_info));

    let queue_submit_result = {
        let _tz = tracy_zone!("vk.queue_submit2");
        let _queue_guard = queue_lock.lock().unwrap();
        unsafe {
            submit_device_core.device.queue_submit2(
                submit_device_core.queue,
                std::slice::from_ref(&submit_info2),
                vk::Fence::null(),
            )
        }
    };
    if let Err(e) = queue_submit_result {
        let ctx_pool = state
            .contexts
            .get(&ctx)
            .context("Invalid context handle")?
            .lock()
            .unwrap()
            .command_pool;
        unsafe {
            submit_device_core.device.free_command_buffers(ctx_pool, &[cmd]);
            if let Some(prof) = vk_gpu_profile.take() {
                submit_device_core.device.destroy_query_pool(prof.pool, None);
            }
        }
        return Err(anyhow::anyhow!("Failed to queue_submit2 command buffer: {:?}", e));
    }

    if let Some(sc_arc) = state.contexts.get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        sc.last_submitted_seq = signal_value;
        sc.timeline_cmd_buffers.entry(signal_value).or_default().push(cmd);
    }

    if !texture_upload_scratch.is_empty() {
        let entries: Vec<staging::TextureStagingEntry> = texture_upload_scratch.into_iter().map(|s| s.entry).collect();
        if let Some(sc_arc) = state.contexts.get(&ctx) {
            sc_arc
                .lock()
                .unwrap()
                .texture_staging_pool
                .release(signal_value, entries);
        }
    }

    if let Some(sc_arc) = state.contexts.get(&ctx) {
        sc_arc.lock().unwrap().staging_belt.finish(signal_value);
    }

    debug_assert_eq!(
        belt_idx,
        belt_slices.len(),
        "WriteBuffer DEVICE_LOCAL count must match belt pre-pass"
    );

    if let Some(prof) = vk_gpu_profile {
        let (device_clone, timeline_sem) = {
            let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
            let sem = state
                .contexts
                .get(&ctx)
                .context("Invalid context handle")?
                .lock()
                .unwrap()
                .timeline_semaphore;
            (ld.device.clone(), sem)
        };
        unsafe {
            vulkan_finish_gpu_profile(state, ctx, &device_clone, timeline_sem, signal_value, cmd, prof)?;
        }
    }

    {
        let completed = ctx_completed_value(state, ctx, device_handle);
        let ctx_batch: Vec<_> = state
            .contexts
            .get(&ctx)
            .map(|sc| sc.lock().unwrap().deletion_queue.drain_up_to(completed))
            .unwrap_or_default();
        if let Some(ld) = state.devices.get(&device_handle) {
            let ledger_arc = std::sync::Arc::clone(&ld.ledger);
            {
                let mut ledger = ledger_arc.lock().unwrap();
                for r in ctx_batch {
                    super::types::destroy_pending_deletion(ld, &mut ledger, r);
                }
                let completed_values =
                    super::types::snapshot_context_completed_values(&ld.device, &state.contexts, device_handle);
                ledger.drain_ready_slot_reclamations(&completed_values);
            }
        }
    }

    Ok(signal_value)
}

/// Submit mixed compute + render graph commands in a single command buffer.
///
/// Eliminates CPU waits between compute and render segments by recording
/// everything into one `VkCommandBuffer` and performing a single
/// `queue_submit2` with a timeline semaphore signal at the end.
pub(super) fn submit_graph(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    commands: &[GraphCommand],
) -> Result<TimelineValue> {
    submit_graph_impl(state, ctx, commands, None)
}

/// Inner implementation shared by `submit_graph` and `submit_graph_and_retain`.
///
/// When `retain_key` is `Some(key)`:
/// - the CB is recorded without `ONE_TIME_SUBMIT` (driver keeps it executable)
/// - after a successful submit the CB is stored in `LogicalDevice::retained_compute_cb`
///   rather than `timeline_cmd_buffers` (so it is not freed by the normal reap cycle)
/// - any WriteBuffer / WriteTexture commands cause a fallback to normal (non-retained) submit
///
/// When `retain_key` is `None` (normal path):
/// - the CB is recorded with `ONE_TIME_SUBMIT`
/// - after submit it is stored in `timeline_cmd_buffers` and freed once the GPU retires it
fn submit_graph_impl(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    commands: &[GraphCommand],
    retain_key: Option<u64>,
) -> Result<TimelineValue> {
    let device_handle = state
        .contexts
        .get(&ctx)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
        .device;
    let _tz = tracy_zone!("vk.submit_graph");
    // --- Same housekeeping as `submit` ---
    // Skip belt reclaim + fence reap when there's no host upload in this submit;
    // the next upload-bearing submit will pick up any signaled fences.
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
    let has_write_texture_graph = commands.iter().any(|c| {
        matches!(
            c,
            GraphCommand::Compute(GpuCommand::WriteTexture { .. } | GpuCommand::WriteTextureRegion { .. })
        )
    });

    if has_upload {
        let _rz = tracy_zone!("vk.submit_graph.belt_reclaim");
        let completed_timeline = ctx_completed_value(state, ctx, device_handle);

        {
            if let Some(sc_arc) = state.contexts.get(&ctx) {
                sc_arc.lock().unwrap().staging_belt.reclaim(
                    &state.compute_fence_pool,
                    &state.devices,
                    completed_timeline,
                )?;
            }
        }
        reap_signaled_fences(state);

        if has_write_texture_graph {
            if let Some(sc_arc) = state.contexts.get(&ctx) {
                sc_arc.lock().unwrap().texture_staging_pool.reclaim(completed_timeline);
            }
        }
    }

    // --- Pre-pass: stage CPU data for WriteBuffer/WriteTexture in compute segments ---
    let mut belt_slices: Vec<(vk::Buffer, u64)> = Vec::new();
    let mut texture_upload_scratch: Vec<super::texture::ComputeTextureScratch> = Vec::new();

    if has_upload {
        for graph_cmd in commands {
            if let GraphCommand::Compute(gpu_cmd) = graph_cmd {
                match gpu_cmd {
                    GpuCommand::WriteBuffer {
                        buffer: buf_handle,
                        offset,
                        data,
                    } => {
                        // Extract Copy fields so the state.buffers borrow ends
                        // before we take &mut state.contexts for the belt.
                        let (host_mapped, is_storage, buf_device, buf_memory) = {
                            let buf = state
                                .buffers
                                .get(buf_handle)
                                .context("WriteBuffer: invalid buffer handle")?;
                            (buf.host_mapped, buf.is_storage, buf.device_handle, buf.memory)
                        };
                        if let Some(base) = host_mapped {
                            let p = base as *mut u8;
                            unsafe {
                                std::ptr::copy_nonoverlapping(data.as_ptr(), p.add(*offset as usize), data.len());
                            }
                        } else if !is_storage {
                            let dev = state.devices.get(&buf_device).context("WriteBuffer: device invalid")?;
                            unsafe {
                                let ptr = dev
                                    .map_memory2(buf_memory, *offset, data.len() as u64)
                                    .context("WriteBuffer: map failed")?;
                                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
                                dev.unmap_memory2(buf_memory).context("WriteBuffer: unmap failed")?;
                            }
                        } else {
                            let dev = state.devices.get(&buf_device).context("WriteBuffer: device invalid")?;
                            let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
                            let mut sc = sc_arc.lock().unwrap();
                            let (stg_buf, stg_off) = sc.staging_belt.write(&state.instance, dev, data)?;
                            belt_slices.push((stg_buf, stg_off));
                        }
                    }
                    GpuCommand::WriteTexture {
                        texture,
                        data,
                        width,
                        height,
                    } => {
                        let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
                        let mut sc_guard = sc_arc.lock().unwrap();
                        let pool = &mut sc_guard.texture_staging_pool;
                        texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                            &state.instance,
                            &state.devices,
                            &state.textures,
                            pool,
                            *texture,
                            data,
                            0,
                            0,
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
                        let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
                        let mut sc_guard = sc_arc.lock().unwrap();
                        let pool = &mut sc_guard.texture_staging_pool;
                        texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                            &state.instance,
                            &state.devices,
                            &state.textures,
                            pool,
                            *texture,
                            data,
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
    }

    // --- Acquire and begin command buffer ---
    let cmd = {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
        let mut sc = sc_arc.lock().unwrap();
        let cb = acquire_cmd_buffer(ld, &mut sc)?;
        // Use ONE_TIME_SUBMIT for normal submits (driver hint for optimization).
        // When retaining, omit the flag so the CB stays executable after GPU completion
        // and can be resubmitted on the next frame.
        let flags = if retain_key.is_none() {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        } else {
            vk::CommandBufferUsageFlags::empty()
        };
        let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);
        if let Err(e) = unsafe { ld.device.begin_command_buffer(cb, &begin_info) } {
            sc.free_cmd_buffers.push(cb);
            return Err(anyhow::anyhow!("Failed to begin command buffer: {:?}", e));
        }
        cb
    };

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    let (dispatch_count_graph, dispatch_labels_graph) = collect_dispatch_labels_graph(commands);
    let mut vk_gpu_profile =
        unsafe { create_vulkan_gpu_profile_pool(logical_device, false, dispatch_count_graph, dispatch_labels_graph)? };

    // Cross-submission acquire: make prior submit's writes visible.
    unsafe {
        let acquire = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::TRANSFER
                    | vk::PipelineStageFlags2::DRAW_INDIRECT,
            )
            .dst_access_mask(
                vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::TRANSFER_READ
                    | vk::AccessFlags2::INDIRECT_COMMAND_READ,
            );
        let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&acquire));
        logical_device.device.cmd_pipeline_barrier2(cmd, &dep);

        if let (Some(bindless_set), Some(bindless_layout)) = (
            logical_device.bindless_descriptor_set,
            logical_device.bindless_pipeline_layout,
        ) {
            logical_device.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                bindless_layout,
                0,
                std::slice::from_ref(&bindless_set),
                &[],
            );
        }
    }

    let mut vk_dispatch_idx = 0usize;
    if let Some(ref prof) = vk_gpu_profile {
        unsafe {
            logical_device
                .device
                .cmd_reset_query_pool(cmd, prof.pool, 0, prof.query_count);
            logical_device
                .device
                .cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, prof.pool, 0);
        }
    }

    // --- Walk GraphCommands (inline to satisfy the borrow checker) ---
    let compute_pipelines = &state.compute_pipelines;
    let buffers = &state.buffers;
    let mut current_compute_pipeline: Option<ComputePipelineHandle> = None;
    let mut belt_idx = 0usize;
    let mut texture_upload_idx = 0usize;
    let mut rendered_targets: Vec<RenderTargetHandle> = Vec::new();

    for graph_cmd in commands {
        match graph_cmd {
            GraphCommand::Compute(gpu_cmd) => match gpu_cmd {
                GpuCommand::FrameTableStaging { .. } => {}
                GpuCommand::SetPipeline(handle) => {
                    let _tz = tracy_zone!("vk.set_pipeline");
                    if let Some(pipeline_state) = compute_pipelines.get(handle) {
                        unsafe {
                            logical_device.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::COMPUTE,
                                pipeline_state.pipeline,
                            );
                        }
                        current_compute_pipeline = Some(*handle);
                    }
                }
                GpuCommand::BindResourcesRaw {
                    indices: raw_indices,
                    user: raw_user,
                    ..
                } => {
                    if let Some(pipeline) = current_compute_pipeline.and_then(|p| compute_pipelines.get(&p)) {
                        crate::backend::validate_raw_binding_strides(
                            raw_indices,
                            &pipeline.push_constant_categories,
                            &pipeline.binding_element_strides,
                            |idx, cat| buffer_stride_for_bindless_index(buffers, device_handle, idx, cat),
                            &pipeline.shader_debug_name,
                        )?;
                        let mut layout = PushLayout::default();
                        shared::fill_raw(&mut layout, raw_indices, raw_user);
                        unsafe {
                            logical_device.device.cmd_push_constants(
                                cmd,
                                pipeline.layout,
                                vk::ShaderStageFlags::ALL,
                                0,
                                layout.as_bytes(),
                            );
                        }
                    }
                }
                GpuCommand::BindResourcesTyped { handles: typed_handles } => {
                    if let Some(pipeline) = current_compute_pipeline.and_then(|p| compute_pipelines.get(&p)) {
                        crate::backend::validate_typed_push_constants(
                            typed_handles,
                            &pipeline.push_constant_categories,
                            &pipeline.shader_debug_name,
                        )?;
                        let mut layout = PushLayout::default();
                        shared::fill_typed(&mut layout, typed_handles.iter().copied());
                        unsafe {
                            logical_device.device.cmd_push_constants(
                                cmd,
                                pipeline.layout,
                                vk::ShaderStageFlags::ALL,
                                0,
                                layout.as_bytes(),
                            );
                        }
                    }
                }
                GpuCommand::Dispatch {
                    label: _label,
                    workgroups_x,
                    workgroups_y,
                    workgroups_z,
                } => {
                    let _tz = tracy_zone!("vk.dispatch");
                    unsafe {
                        if let Some(ref prof) = vk_gpu_profile {
                            let base = 2u32 + (vk_dispatch_idx as u32) * 2;
                            logical_device.device.cmd_write_timestamp2(
                                cmd,
                                vk::PipelineStageFlags2::TOP_OF_PIPE,
                                prof.pool,
                                base,
                            );
                        }
                        logical_device
                            .device
                            .cmd_dispatch(cmd, *workgroups_x, *workgroups_y, *workgroups_z);
                        if let Some(ref prof) = vk_gpu_profile {
                            let base = 2u32 + (vk_dispatch_idx as u32) * 2;
                            logical_device.device.cmd_write_timestamp2(
                                cmd,
                                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                                prof.pool,
                                base + 1,
                            );
                        }
                        vk_dispatch_idx += 1;
                    }
                }
                GpuCommand::DispatchBatch {
                    label: _,
                    arg_data,
                    count,
                } => {
                    let _tz = tracy_zone!("vk.dispatch_batch");
                    let push_size = std::mem::size_of::<crate::backend::shared::PushLayout>();
                    let stride = crate::backend::shared::DISPATCH_BATCH_STRIDE;
                    let pipeline_layout = current_compute_pipeline
                        .and_then(|h| compute_pipelines.get(&h))
                        .map(|p| p.layout);
                    for i in 0..*count as usize {
                        let base = i * stride;
                        let layout_bytes = &arg_data[base..base + push_size];
                        let wg_off = base + push_size;
                        let wg_x = u32::from_ne_bytes(arg_data[wg_off..wg_off + 4].try_into().unwrap());
                        let wg_y = u32::from_ne_bytes(arg_data[wg_off + 4..wg_off + 8].try_into().unwrap());
                        let wg_z = u32::from_ne_bytes(arg_data[wg_off + 8..wg_off + 12].try_into().unwrap());
                        unsafe {
                            if let Some(layout) = pipeline_layout {
                                logical_device.device.cmd_push_constants(
                                    cmd,
                                    layout,
                                    vk::ShaderStageFlags::ALL,
                                    0,
                                    layout_bytes,
                                );
                            }
                            logical_device.device.cmd_dispatch(cmd, wg_x, wg_y, wg_z);
                        }
                        vk_dispatch_idx += 1;
                    }
                }
                GpuCommand::DispatchIndirect {
                    label: _label,
                    buffer,
                    offset,
                } => {
                    let _tz = tracy_zone!("vk.dispatch_indirect");
                    let buf_state = buffers.get(buffer).context("DispatchIndirect: invalid buffer handle")?;
                    unsafe {
                        if let Some(ref prof) = vk_gpu_profile {
                            let base = 2u32 + (vk_dispatch_idx as u32) * 2;
                            logical_device.device.cmd_write_timestamp2(
                                cmd,
                                vk::PipelineStageFlags2::TOP_OF_PIPE,
                                prof.pool,
                                base,
                            );
                        }
                        logical_device
                            .device
                            .cmd_dispatch_indirect(cmd, buf_state.buffer, *offset);

                        if let Some(ref prof) = vk_gpu_profile {
                            let base = 2u32 + (vk_dispatch_idx as u32) * 2;
                            logical_device.device.cmd_write_timestamp2(
                                cmd,
                                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                                prof.pool,
                                base + 1,
                            );
                        }
                    }
                    vk_dispatch_idx += 1;
                }
                GpuCommand::Barrier => {
                    let _tz = tracy_zone!("vk.barrier");
                    unsafe {
                        let mem_barrier = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
                        let dep_info =
                            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                    }
                }
                GpuCommand::ResourceBarrier {
                    buffers: buf_entries,
                    textures: tex_entries,
                } => {
                    let _tz = tracy_zone!("vk.resource_barrier");
                    unsafe {
                        let buf_barriers: Vec<vk::BufferMemoryBarrier2> = buf_entries
                            .iter()
                            .filter_map(|(h, usage)| {
                                buffers.get(h).map(|bs| {
                                    vk::BufferMemoryBarrier2::default()
                                        .src_stage_mask(slot_usage_to_vk_stage(&usage.src))
                                        .src_access_mask(slot_usage_to_vk_access(&usage.src))
                                        .dst_stage_mask(slot_usage_to_vk_stage(&usage.dst))
                                        .dst_access_mask(slot_usage_to_vk_access(&usage.dst))
                                        .buffer(bs.buffer)
                                        .offset(0)
                                        .size(vk::WHOLE_SIZE)
                                })
                            })
                            .collect();
                        // Textures: we don't track per-image Vulkan layout in TextureState,
                        // so use a global memory barrier per texture entry. No layout
                        // transition needed — just execution + memory dependency.
                        let tex_mem: Vec<vk::MemoryBarrier2> = tex_entries
                            .iter()
                            .map(|(_, usage)| {
                                vk::MemoryBarrier2::default()
                                    .src_stage_mask(slot_usage_to_vk_stage(&usage.src))
                                    .src_access_mask(slot_usage_to_vk_access(&usage.src))
                                    .dst_stage_mask(slot_usage_to_vk_stage(&usage.dst))
                                    .dst_access_mask(slot_usage_to_vk_access(&usage.dst))
                            })
                            .collect();
                        let dep_info = if tex_mem.is_empty() {
                            vk::DependencyInfo::default().buffer_memory_barriers(&buf_barriers)
                        } else {
                            vk::DependencyInfo::default()
                                .buffer_memory_barriers(&buf_barriers)
                                .memory_barriers(&tex_mem)
                        };
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                    }
                }
                GpuCommand::ClearBuffer { buffer, offset, size } => {
                    let _tz = tracy_zone!("vk.clear_buffer");
                    let buf_state = buffers.get(buffer).context("ClearBuffer: invalid buffer handle")?;
                    let clear_size = if *size == 0 {
                        buf_state.size.saturating_sub(*offset)
                    } else {
                        *size
                    };
                    if clear_size > 0 {
                        unsafe {
                            logical_device
                                .device
                                .cmd_fill_buffer(cmd, buf_state.buffer, *offset, clear_size, 0);
                            let mem_barrier = vk::MemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
                            let dep_info =
                                vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                            logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                        }
                    }
                }
                GpuCommand::WriteBuffer {
                    buffer: buf_handle,
                    offset,
                    data,
                } => {
                    let _tz = tracy_zone!("vk.write_buffer");
                    let buf_state = buffers.get(buf_handle).context("WriteBuffer: invalid buffer handle")?;
                    if buf_state.is_storage && buf_state.host_mapped.is_none() {
                        let (stg, stg_off) = belt_slices
                            .get(belt_idx)
                            .context("WriteBuffer: belt slice missing (internal error)")?;
                        belt_idx += 1;
                        let region = vk::BufferCopy {
                            src_offset: *stg_off,
                            dst_offset: *offset,
                            size: data.len() as u64,
                        };
                        unsafe {
                            logical_device.device.cmd_copy_buffer(
                                cmd,
                                *stg,
                                buf_state.buffer,
                                std::slice::from_ref(&region),
                            );
                            let mem_barrier = vk::MemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
                            let dep_info =
                                vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                            logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                        }
                    }
                }
                GpuCommand::WriteTexture { .. } | GpuCommand::WriteTextureRegion { .. } => {
                    let _tz = tracy_zone!("vk.write_texture");
                    let scratch = texture_upload_scratch
                        .get(texture_upload_idx)
                        .context("WriteTexture: scratch missing (internal)")?;
                    texture_upload_idx += 1;
                    super::texture::record_compute_texture_upload(&state.devices, &state.textures, cmd, scratch)?;
                }
                GpuCommand::CopyTexture { src, dst } => {
                    let _tz = tracy_zone!("vk.copy_texture");
                    let (src_image, width, height) = {
                        let ts = state.textures.get(src).context("CopyTexture: src texture not found")?;
                        (ts.image, ts.width, ts.height)
                    };
                    let dst_image = state
                        .textures
                        .get(dst)
                        .context("CopyTexture: dst texture not found")?
                        .image;
                    unsafe {
                        let mem_barrier = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE);
                        let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep);

                        let region = vk::ImageCopy {
                            src_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            src_offset: vk::Offset3D::default(),
                            dst_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            dst_offset: vk::Offset3D::default(),
                            extent: vk::Extent3D {
                                width,
                                height,
                                depth: 1,
                            },
                        };
                        logical_device.device.cmd_copy_image(
                            cmd,
                            src_image,
                            vk::ImageLayout::GENERAL,
                            dst_image,
                            vk::ImageLayout::GENERAL,
                            std::slice::from_ref(&region),
                        );

                        let mem_barrier2 = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_COMMANDS,
                            )
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
                        let dep2 = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier2));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep2);
                    }
                }
                GpuCommand::CopyRenderTarget { src, dst } => {
                    let _tz = tracy_zone!("vk.copy_render_target");
                    let (src_image, width, height) = {
                        let rt = state
                            .render_targets
                            .get(src)
                            .context("CopyRenderTarget: src render target not found")?;
                        (rt.image, rt.width, rt.height)
                    };
                    let dst_image = state
                        .textures
                        .get(dst)
                        .context("CopyRenderTarget: dst texture not found")?
                        .image;

                    unsafe {
                        let mem_barrier = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
                        let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep);

                        let region = vk::ImageCopy {
                            src_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            src_offset: vk::Offset3D::default(),
                            dst_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            dst_offset: vk::Offset3D::default(),
                            extent: vk::Extent3D {
                                width,
                                height,
                                depth: 1,
                            },
                        };
                        logical_device.device.cmd_copy_image(
                            cmd,
                            src_image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            dst_image,
                            vk::ImageLayout::GENERAL,
                            std::slice::from_ref(&region),
                        );

                        let mem_barrier2 = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_COMMANDS,
                            )
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
                        let dep2 = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier2));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep2);
                    }
                }
            },
            GraphCommand::Render {
                target,
                commands: render_cmds,
            } => {
                let _tz = tracy_zone!("vk.render_pass");
                // Flush compute writes before the render pass
                unsafe {
                    let barrier = vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                                | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::VERTEX_SHADER
                                | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                                | vk::AccessFlags2::SHADER_READ
                                | vk::AccessFlags2::MEMORY_READ,
                        );
                    let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier));
                    logical_device.device.cmd_pipeline_barrier2(cmd, &dep);
                }

                let pipelines = &state.pipelines;
                let rt_buffers = &state.buffers;
                super::render_target::record_render_pass_to_buffer(
                    &state.devices,
                    &state.render_targets,
                    device_handle,
                    *target,
                    render_cmds,
                    cmd,
                    |cb, cmds, ld, cur_pipe| {
                        super::render_commands::record(cb, cmds, ld, pipelines, rt_buffers, cur_pipe)
                    },
                )?;

                rendered_targets.push(*target);

                // Make render writes visible to subsequent compute
                unsafe {
                    let barrier = vk::MemoryBarrier2::default()
                        .src_stage_mask(
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        )
                        .src_access_mask(
                            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(
                            vk::AccessFlags2::SHADER_READ
                                | vk::AccessFlags2::SHADER_WRITE
                                | vk::AccessFlags2::TRANSFER_READ
                                | vk::AccessFlags2::TRANSFER_WRITE,
                        );
                    let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier));
                    logical_device.device.cmd_pipeline_barrier2(cmd, &dep);
                }
            }
        }
    }

    // --- Release barrier: make this CB's writes available to subsequent submits ---
    unsafe {
        let release = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::TRANSFER
                    | vk::PipelineStageFlags2::DRAW_INDIRECT
                    | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            )
            .dst_access_mask(
                vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::TRANSFER_READ
                    | vk::AccessFlags2::INDIRECT_COMMAND_READ
                    | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            );
        let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&release));
        logical_device.device.cmd_pipeline_barrier2(cmd, &dep);
    }

    if let Some(ref prof) = vk_gpu_profile {
        unsafe {
            logical_device
                .device
                .cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, prof.pool, 1);
        }
    }

    // --- End + submit ---
    if let Err(e) = unsafe { logical_device.device.end_command_buffer(cmd) } {
        unsafe {
            logical_device
                .device
                .free_command_buffers(logical_device.command_pool, &[cmd]);
        }
        return Err(anyhow::anyhow!("Failed to end command buffer: {:?}", e));
    }

    let signal_value = {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        ld.timeline_next.fetch_add(1, Ordering::Relaxed)
    };

    let used_slots =
        collect_slot_keys_from_graph_commands(commands, &state.compute_pipelines, &state.pipelines, &state.buffers);
    if let Some(ld) = state.devices.get(&device_handle) {
        ld.ledger
            .lock()
            .unwrap()
            .record_slot_usage(ctx, signal_value, used_slots.iter().copied());
    }

    let timeline_sem = state
        .contexts
        .get(&ctx)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
        .timeline_semaphore;
    let submit_device = state.devices.get(&device_handle).context("Invalid device handle")?;
    let queue_lock = std::sync::Arc::clone(&submit_device.queue_lock);
    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
    let signal_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(timeline_sem)
        .value(signal_value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let submit_info2 = vk::SubmitInfo2::default()
        .command_buffer_infos(std::slice::from_ref(&cmd_info))
        .signal_semaphore_infos(std::slice::from_ref(&signal_info));

    let queue_submit_result = {
        let _tz = tracy_zone!("vk.queue_submit2");
        let _queue_guard = queue_lock.lock().unwrap();
        unsafe {
            submit_device.device.queue_submit2(
                submit_device.queue,
                std::slice::from_ref(&submit_info2),
                vk::Fence::null(),
            )
        }
    };
    if let Err(e) = queue_submit_result {
        let ctx_pool = state
            .contexts
            .get(&ctx)
            .context("Invalid context handle")?
            .lock()
            .unwrap()
            .command_pool;
        unsafe {
            submit_device.device.free_command_buffers(ctx_pool, &[cmd]);
            if let Some(prof) = vk_gpu_profile.take() {
                submit_device.device.destroy_query_pool(prof.pool, None);
            }
        }
        return Err(anyhow::anyhow!("Failed to queue_submit2 command buffer: {:?}", e));
    }

    // Post-submit: store the CB for lifecycle management.
    if let Some(sc_arc) = state.contexts.get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        sc.last_submitted_seq = signal_value;
        if let Some(key) = retain_key {
            sc.retained_compute_cb = Some(super::types::RetainedVkCb {
                fingerprint: key,
                command_buffer: cmd,
                used_slots,
            });
        } else {
            sc.timeline_cmd_buffers.entry(signal_value).or_default().push(cmd);
        }
    }

    if !texture_upload_scratch.is_empty() {
        let entries: Vec<staging::TextureStagingEntry> = texture_upload_scratch.into_iter().map(|s| s.entry).collect();
        if let Some(sc_arc) = state.contexts.get(&ctx) {
            sc_arc
                .lock()
                .unwrap()
                .texture_staging_pool
                .release(signal_value, entries);
        }
    }

    if let Some(sc_arc) = state.contexts.get(&ctx) {
        sc_arc.lock().unwrap().staging_belt.finish(signal_value);
    }

    if let Some(prof) = vk_gpu_profile {
        let (device_clone, timeline_sem) = {
            let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
            let sem = state
                .contexts
                .get(&ctx)
                .context("Invalid context handle")?
                .lock()
                .unwrap()
                .timeline_semaphore;
            (ld.device.clone(), sem)
        };
        unsafe {
            vulkan_finish_gpu_profile(state, ctx, &device_clone, timeline_sem, signal_value, cmd, prof)?;
        }
    }

    // Mark rendered targets
    for t in rendered_targets {
        if let Some(rt) = state.render_targets.get(&t) {
            rt.has_rendered.store(true, Ordering::Relaxed);
        }
    }

    {
        let completed = ctx_completed_value(state, ctx, device_handle);
        let ctx_batch: Vec<_> = state
            .contexts
            .get(&ctx)
            .map(|sc| sc.lock().unwrap().deletion_queue.drain_up_to(completed))
            .unwrap_or_default();
        if let Some(ld) = state.devices.get(&device_handle) {
            let ledger_arc = std::sync::Arc::clone(&ld.ledger);
            {
                let mut ledger = ledger_arc.lock().unwrap();
                for r in ctx_batch {
                    super::types::destroy_pending_deletion(ld, &mut ledger, r);
                }
                let completed_values =
                    super::types::snapshot_context_completed_values(&ld.device, &state.contexts, device_handle);
                ledger.drain_ready_slot_reclamations(&completed_values);
            }
        }
    }

    Ok(signal_value)
}

/// Record, submit, and retain a dispatch command buffer keyed by `key`.
///
/// The CB is recorded without `ONE_TIME_SUBMIT` (driver keeps it executable after GPU
/// completion) and stored in `LogicalDevice::retained_compute_cb` rather than
/// `timeline_cmd_buffers`, so it survives the normal reap cycle.
/// On subsequent frames call [`try_resubmit_retained`] to re-execute without re-recording.
/// If commands contain any WriteBuffer/WriteTexture nodes the call falls back to a normal
/// (non-retained) submit via [`submit_graph`].
pub(super) fn submit_graph_and_retain(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    commands: &[GraphCommand],
    key: u64,
) -> Result<TimelineValue> {
    // Evict any previously retained CB so we start fresh.
    evict_retained(state, ctx, key);
    // Delegate to the shared inner path with retention enabled.
    submit_graph_impl(state, ctx, commands, Some(key))
}

/// Re-execute the retained dispatch CB without re-recording.
///
/// Returns `Ok(Some(tv))` if the retained CB for `key` was found and resubmitted.
/// Returns `Ok(None)` if no matching retained CB exists.
/// Safety: The caller must have confirmed that the GPU has completed the previous
/// submission of this CB (e.g. by `wait_until`) so it is in executable state.
pub(super) fn try_resubmit_retained(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    key: u64,
) -> Result<Option<TimelineValue>> {
    let (device_handle, timeline_sem) = {
        let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
        let sc = sc_arc.lock().unwrap();
        (sc.device, sc.timeline_semaphore)
    };
    let retained = {
        let sc_arc = state.contexts.get(&ctx).context("Invalid context handle")?;
        let sc = sc_arc.lock().unwrap();
        match sc.retained_compute_cb.as_ref() {
            Some(r) if r.fingerprint == key => Some((r.command_buffer, r.used_slots.clone())),
            _ => None,
        }
    };

    let Some((cmd, used_slots)) = retained else {
        return Ok(None);
    };

    let signal_value = {
        let ld = state.devices.get(&device_handle).context("Invalid device handle")?;
        ld.timeline_next.fetch_add(1, Ordering::Relaxed)
    };
    let submit_device = state.devices.get(&device_handle).context("Invalid device handle")?;
    let queue_lock = std::sync::Arc::clone(&submit_device.queue_lock);
    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
    let signal_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(timeline_sem)
        .value(signal_value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let submit_info2 = vk::SubmitInfo2::default()
        .command_buffer_infos(std::slice::from_ref(&cmd_info))
        .signal_semaphore_infos(std::slice::from_ref(&signal_info));

    {
        let _tz = tracy_zone!("vk.resubmit_retained");
        let _queue_guard = queue_lock.lock().unwrap();
        unsafe {
            submit_device.device.queue_submit2(
                submit_device.queue,
                std::slice::from_ref(&submit_info2),
                vk::Fence::null(),
            )
        }
        .context("Failed to queue_submit2 retained dispatch CB")?;
    }

    if let Some(ld) = state.devices.get(&device_handle) {
        ld.ledger
            .lock()
            .unwrap()
            .record_slot_usage(ctx, signal_value, used_slots);
    }

    {
        let completed = ctx_completed_value(state, ctx, device_handle);
        let ctx_batch: Vec<_> = state
            .contexts
            .get(&ctx)
            .map(|sc| sc.lock().unwrap().deletion_queue.drain_up_to(completed))
            .unwrap_or_default();
        if let Some(ld) = state.devices.get(&device_handle) {
            let ledger_arc = std::sync::Arc::clone(&ld.ledger);
            {
                let mut ledger = ledger_arc.lock().unwrap();
                for r in ctx_batch {
                    super::types::destroy_pending_deletion(ld, &mut ledger, r);
                }
                let completed_values =
                    super::types::snapshot_context_completed_values(&ld.device, &state.contexts, device_handle);
                ledger.drain_ready_slot_reclamations(&completed_values);
            }
        }
    }
    if let Some(sc_arc) = state.contexts.get(&ctx) {
        sc_arc.lock().unwrap().last_submitted_seq = signal_value;
    }

    Ok(Some(signal_value))
}

/// Evict the retained dispatch CB for `key` (or any retained CB if `key` doesn't match),
/// returning the `VkCommandBuffer` to `free_cmd_buffers` for pool reuse.
pub(super) fn evict_retained(state: &super::types::VulkanState, ctx: super::ContextHandle, _key: u64) {
    if let Some(sc_arc) = state.contexts.get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        if let Some(old) = sc.retained_compute_cb.take() {
            sc.free_cmd_buffers.push(old.command_buffer);
        }
    }
}

pub(super) fn reap_timeline_cmd_buffers_up_to(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    max_completed_value: u64,
) {
    let (device, pool, keys): (DeviceHandle, vk::CommandPool, Vec<u64>) = {
        let sc_arc = match state.contexts.get(&ctx) {
            Some(s) => s,
            None => return,
        };
        let sc = sc_arc.lock().unwrap();
        if sc.timeline_cmd_buffers.is_empty() {
            return;
        }
        let keys: Vec<u64> = sc
            .timeline_cmd_buffers
            .keys()
            .copied()
            .filter(|k| *k <= max_completed_value)
            .collect();
        (sc.device, sc.command_pool, keys)
    };
    if keys.is_empty() {
        return;
    }
    let cbs_to_free: Vec<vk::CommandBuffer> = {
        let sc_arc = state.contexts.get(&ctx).expect("context");
        let mut sc = sc_arc.lock().unwrap();
        keys.iter()
            .filter_map(|k| sc.timeline_cmd_buffers.remove(k))
            .flatten()
            .collect()
    };
    if let Some(ld) = state.devices.get(&device) {
        for cb in cbs_to_free {
            unsafe {
                ld.device.free_command_buffers(pool, &[cb]);
            }
        }
    }
}
