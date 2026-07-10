//! Compute pipeline and dispatch logic.

use super::super::shared;
use super::super::shared::DISPATCH_BATCH_STRIDE;
use super::staging;
use super::submit_session::{VulkanSubmitScope, VulkanSubmitView};
use super::types::{
    BufferState, ComputePipelineState, LogicalDevice, PushLayout, SharedBufferTable, SharedComputePipelineTable,
    SharedPipelineTable, SlotKey,
};
use super::{BufferHandle, ComputePipelineHandle, DeviceHandle, RenderTargetHandle};
use crate::backend::{GpuCommand, GraphCommand, RenderCommand, SubmitSync};
use crate::gpu_profiler::{self, DispatchGpuNs};
use crate::task_graph::{NodeAccessUnion, SlotUsageSet, UsageKindFlags};
use crate::timeline::TimelineValue;
use crate::tracy_zone;
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Build GPU-side timeline-semaphore waits for cross-context [`SubmitSync::waits`].
fn apply_cpu_epoch_waits(view: &VulkanSubmitView<'_>, sync: Option<&SubmitSync>) -> Result<()> {
    let Some(s) = sync else {
        return Ok(());
    };
    if s.cpu_waits.is_empty() {
        return Ok(());
    }
    for epoch in &s.cpu_waits {
        let (device_handle, sem) = {
            let contexts = view.contexts.read().unwrap();
            let sc = contexts
                .get(&epoch.context)
                .with_context(|| format!("cross-submit cpu wait: invalid context {:?}", epoch.context))?;
            let sc = sc.lock().unwrap();
            (sc.device, sc.timeline_semaphore)
        };
        let ld = view
            .devices
            .get(&device_handle)
            .with_context(|| format!("cross-submit cpu wait: invalid device {:?}", device_handle))?;
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&sem))
            .values(std::slice::from_ref(&epoch.value));
        unsafe { ld.device.wait_semaphores(&wait, u64::MAX) }.context("cross-submit cpu wait on timeline semaphore")?;
    }
    Ok(())
}

fn build_cross_submit_wait_infos(
    view: &VulkanSubmitView<'_>,
    sync: Option<&SubmitSync>,
) -> Result<Vec<vk::SemaphoreSubmitInfo<'static>>> {
    apply_cpu_epoch_waits(view, sync)?;
    let Some(s) = sync else {
        return Ok(Vec::new());
    };
    let mut wait_infos = Vec::with_capacity(s.waits.len());
    for epoch in &s.waits {
        let sem = view
            .contexts
            .read()
            .unwrap()
            .get(&epoch.context)
            .with_context(|| format!("cross-submit wait: invalid context {:?}", epoch.context))?
            .lock()
            .unwrap()
            .timeline_semaphore;
        wait_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(sem)
                .value(epoch.value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        );
    }
    Ok(wait_infos)
}

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
    buffers: &HashMap<BufferHandle, BufferState>,
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
    compute_pipelines: &SharedComputePipelineTable,
    _buffers: &SharedBufferTable,
) -> Vec<SlotKey> {
    let mut current_pipeline = None;
    let mut slots = Vec::new();
    let compute_read = compute_pipelines.read().unwrap();
    for cmd in commands {
        match cmd {
            GpuCommand::SetPipeline(p) => current_pipeline = Some(*p),
            GpuCommand::BindResourcesRaw { indices, .. } => {
                if let Some(h) = current_pipeline {
                    if let Some(p) = compute_read.entries.get(&h) {
                        slots.extend(collect_slots_from_raw_bind(indices, &p.push_constant_categories));
                    }
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
                if let Some(h) = current_pipeline {
                    if let Some(p) = compute_read.entries.get(&h) {
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
            }
            _ => {}
        }
    }
    slots
}

fn collect_slot_keys_from_graph_commands(
    commands: &[GraphCommand],
    compute_pipelines: &SharedComputePipelineTable,
    pipelines: &SharedPipelineTable,
    buffers: &SharedBufferTable,
) -> Vec<SlotKey> {
    let mut slots = Vec::new();
    let mut current_compute_pipeline = None;
    let mut current_render_pipeline = None;
    let compute_read = compute_pipelines.read().unwrap();
    let pipelines_read = pipelines.read().unwrap();
    let buffers_read = buffers.read().unwrap();
    for gc in commands {
        match gc {
            GraphCommand::Compute(cmd) => match cmd {
                GpuCommand::SetPipeline(p) => current_compute_pipeline = Some(*p),
                GpuCommand::BindResourcesRaw { indices, .. } => {
                    if let Some(h) = current_compute_pipeline {
                        if let Some(p) = compute_read.entries.get(&h) {
                            slots.extend(collect_slots_from_raw_bind(indices, &p.push_constant_categories));
                        }
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
                    if let Some(h) = current_compute_pipeline {
                        if let Some(p) = compute_read.entries.get(&h) {
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
                                if let Some(idx) = buffers_read.entries.get(h).and_then(|b| b.bindless_index) {
                                    slots.push(SlotKey::StorageBuffer(idx));
                                }
                            }
                        }
                        RenderCommand::BindResourcesRaw { indices, .. } => {
                            if let Some(h) = current_render_pipeline {
                                if let Some(p) = pipelines_read.entries.get(&h) {
                                    slots.extend(collect_slots_from_raw_bind(indices, &p.push_constant_categories));
                                }
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
///
/// When `for_buffer` is true, color-attachment and depth-stencil access flags
/// are omitted: those access types are only valid on image memory barriers, not
/// `VkBufferMemoryBarrier2`.
fn slot_usage_to_vk_access(usage: &SlotUsageSet, for_buffer: bool) -> vk::AccessFlags2 {
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
        if for_buffer {
            // Buffer read by vertex/pixel shader inside a render pass → SHADER_READ.
            flags |= vk::AccessFlags2::SHADER_READ;
        } else {
            flags |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }
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
    view: &VulkanSubmitView<'_>,
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
            view.device_lost.store(true, Ordering::Relaxed);
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
    reap_timeline_cmd_buffers_up_to_with_view(view, ctx, signal_value);
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
    view: &VulkanSubmitView<'_>,
    ctx: super::ContextHandle,
    device_handle: super::DeviceHandle,
) -> u64 {
    let sem = view
        .contexts
        .read()
        .unwrap()
        .get(&ctx)
        .map(|sc| sc.lock().unwrap().timeline_semaphore);
    let dev = view.devices.get(&device_handle).map(|ld| &ld.device);
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
fn reap_signaled_fences(view: &VulkanSubmitView<'_>) {
    let signaled: Vec<u64> = {
        let pool = view.compute_fence_pool.lock().unwrap();
        pool.iter()
            .filter_map(|(token, (device_handle, fence, _))| {
                let logical_device = view.devices.get(device_handle)?;
                let signaled = unsafe { logical_device.device.get_fence_status(*fence) }.unwrap_or(false);
                if signaled {
                    Some(*token)
                } else {
                    None
                }
            })
            .collect()
    };

    let mut pool = view.compute_fence_pool.lock().unwrap();
    for token in signaled {
        if let Some((device_handle, fence, cmd_buf)) = pool.remove(&token) {
            if let Some(logical_device) = view.devices.get(&device_handle) {
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
    compute_pipelines: &SharedComputePipelineTable,
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

    let handle = compute_pipelines.write().unwrap().alloc_handle();

    compute_pipelines.write().unwrap().entries.insert(
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
    compute_pipelines: &SharedComputePipelineTable,
    pipeline_handle: ComputePipelineHandle,
) {
    if let Some(pipeline) = compute_pipelines.write().unwrap().entries.remove(&pipeline_handle) {
        if let Some(logical_device) = devices.get(&pipeline.device_handle) {
            unsafe {
                logical_device.device_wait_idle_locked().ok();
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
pub(super) fn submit_with_scope(
    scope: &VulkanSubmitScope<'_>,
    ctx: super::ContextHandle,
    commands: &[GpuCommand],
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    scope.assert_ctx(ctx);
    let view = &scope.view;
    let device_handle = scope.device_handle;
    let mut commands = commands.to_vec();
    crate::frame_table::lower_gpu_commands(&mut commands);
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
            GpuCommand::WriteTexture { .. }
                | GpuCommand::WriteTextureRegion { .. }
                | GpuCommand::CopyBufferToTexture { .. }
        )
    });

    if has_write_buffer || has_write_texture {
        let _rz = tracy_zone!("vk.submit.belt_reclaim");
        let completed_timeline = scope.completed_timeline_value();

        if has_write_buffer {
            {
                scope.sc.lock().unwrap().staging_belt.reclaim(
                    view.compute_fence_pool,
                    view.devices,
                    completed_timeline,
                )?;
            }
            reap_signaled_fences(view);
        }

        if has_write_texture {
            {
                scope
                    .sc
                    .lock()
                    .unwrap()
                    .texture_staging_pool
                    .reclaim(completed_timeline);
            }
        }
    }

    // Belt slices for DEVICE_LOCAL WriteBuffer copies (same iteration order as command loop).
    let mut belt_slices: Vec<(vk::Buffer, u64)> = Vec::new();

    // Pre-pass: stage CPU data for WriteBuffer commands (needs mutable belt
    // access before we borrow state immutably for the command loop).
    if has_write_buffer {
        for command in &commands {
            if let GpuCommand::WriteBuffer {
                buffer: buf_handle,
                offset,
                data,
            } = command
            {
                // Extract Copy fields from the buffer state so the borrow ends
                // before we take a mutable borrow of view.contexts for the belt.
                let (host_mapped, is_storage, buf_device, buf_memory) = {
                    let buffers_read = view.buffers.read().unwrap();
                    let buf = buffers_read
                        .entries
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
                    let dev = view.devices.get(&buf_device).context("WriteBuffer: device invalid")?;
                    unsafe {
                        let ptr = dev
                            .map_memory2(buf_memory, *offset, data.len() as u64)
                            .context("WriteBuffer: map failed")?;
                        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
                        dev.unmap_memory2(buf_memory).context("WriteBuffer: unmap failed")?;
                    }
                } else {
                    let dev = view.devices.get(&buf_device).context("WriteBuffer: device invalid")?;
                    let mut sc = scope.sc.lock().unwrap();
                    let (stg_buf, stg_off) = sc.staging_belt.write(view.instance, dev, data)?;
                    belt_slices.push((stg_buf, stg_off));
                }
            }
        }
    }

    if commands.is_empty() {
        let signal_value = view
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?
            .timeline_next
            .fetch_add(1, Ordering::Relaxed);
        let timeline_sem = scope.sc.lock().unwrap().timeline_semaphore;
        let ld = view.devices.get(&device_handle).context("Invalid device handle")?;
        let queue = ld.queue;
        let queue_lock = std::sync::Arc::clone(&ld.queue_lock);
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline_sem)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let wait_infos = build_cross_submit_wait_infos(view, sync)?;
        let submit_info2 = if wait_infos.is_empty() {
            vk::SubmitInfo2::default().signal_semaphore_infos(std::slice::from_ref(&signal_info))
        } else {
            vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(std::slice::from_ref(&signal_info))
        };
        let r = {
            let _queue_guard = queue_lock.lock().unwrap();
            unsafe {
                ld.device
                    .queue_submit2(queue, std::slice::from_ref(&submit_info2), vk::Fence::null())
            }
        };
        r.context("Failed queue_submit2 for empty compute submit")?;
        {
            scope.sc.lock().unwrap().last_submitted_seq = signal_value;
        }
        return Ok(signal_value);
    }

    let compute_pipelines = &view.compute_pipelines;
    let buffers = &view.buffers;

    let mut texture_upload_scratch: Vec<super::texture::ComputeTextureScratch> = Vec::new();
    for command in &commands {
        match command {
            GpuCommand::WriteTexture {
                texture,
                data,
                width,
                height,
            } => {
                let mut sc_guard = scope.sc.lock().unwrap();
                let pool = &mut sc_guard.texture_staging_pool;
                texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                    view.instance,
                    view.devices,
                    view.textures,
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
                let mut sc_guard = scope.sc.lock().unwrap();
                let pool = &mut sc_guard.texture_staging_pool;
                texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                    view.instance,
                    view.devices,
                    view.textures,
                    pool,
                    *texture,
                    data,
                    *x,
                    *y,
                    *width,
                    *height,
                )?);
            }
            GpuCommand::CopyBufferToTexture {
                src,
                src_offset,
                dst,
                x,
                y,
                width,
                height,
                ..
            } => {
                let flat = super::texture::copy_buffer_to_texture_flat_bytes(
                    view.textures,
                    view.buffers,
                    *src,
                    *src_offset,
                    *dst,
                    *width,
                    *height,
                )?;
                let mut sc_guard = scope.sc.lock().unwrap();
                texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                    view.instance,
                    view.devices,
                    view.textures,
                    &mut sc_guard.texture_staging_pool,
                    *dst,
                    &flat,
                    *x,
                    *y,
                    *width,
                    *height,
                )?);
            }
            _ => {}
        }
    }

    let (dispatch_count, dispatch_labels) = collect_dispatch_labels_compute(&commands);
    let vk_gpu_profile = unsafe {
        let ld = view.devices.get(&device_handle).context("Invalid device handle")?;
        create_vulkan_gpu_profile_pool(ld, false, dispatch_count, dispatch_labels)?
    };

    let mut vk_gpu_profile = vk_gpu_profile;

    let cmd = {
        let ld = view.devices.get(&device_handle).context("Invalid device handle")?;
        let mut sc = scope.sc.lock().unwrap();
        let cb = acquire_cmd_buffer(ld, &mut sc)?;
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(e) = unsafe { ld.device.begin_command_buffer(cb, &begin_info) } {
            sc.free_cmd_buffers.push(cb);
            return Err(anyhow::anyhow!("Failed to begin command buffer: {:?}", e));
        }
        cb
    };

    let (cmd, belt_idx, _texture_upload_idx) = {
        let logical_device = view.devices.get(&device_handle).context("Invalid device handle")?;

        // Cross-submission acquire: make prior submit's writes visible to this
        // CB's reads. Same-queue execution ordering is guaranteed by Vulkan but
        // memory visibility is not. Skipped when epoch-driven scoped sync is active.
        if SubmitSync::use_legacy_acquire_from(sync) {
            unsafe {
                let acquire = vk::MemoryBarrier2::default()
                    .src_stage_mask(
                        vk::PipelineStageFlags2::COMPUTE_SHADER
                            | vk::PipelineStageFlags2::TRANSFER
                            | vk::PipelineStageFlags2::ALL_GRAPHICS,
                    )
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
            }
        }

        unsafe {
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
        for command in &commands {
            match command {
                GpuCommand::FrameTableStaging { data } => {
                    super::frame_table::record_prologue(
                        view.contexts,
                        ctx,
                        &scope.frame_table,
                        view.buffers,
                        logical_device,
                        cmd,
                        data,
                    )?;
                }
                GpuCommand::SetPipeline(handle) => {
                    let _tz = tracy_zone!("vk.set_pipeline");
                    if let Some(pipeline_state) = compute_pipelines.read().unwrap().entries.get(handle) {
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
                    frame_table_base,
                } => {
                    let pipelines_read = compute_pipelines.read().unwrap();
                    if let Some(pipeline) = current_pipeline.and_then(|h| pipelines_read.entries.get(&h)) {
                        crate::backend::with_layout_validation(|| {
                            crate::backend::validate_raw_binding_strides(
                                raw_indices,
                                &pipeline.push_constant_categories,
                                &pipeline.binding_element_strides,
                                |idx, cat| {
                                    buffer_stride_for_bindless_index(
                                        &buffers.read().unwrap().entries,
                                        device_handle,
                                        idx,
                                        cat,
                                    )
                                },
                                &pipeline.shader_debug_name,
                            )
                        })?;
                        let mut layout = PushLayout::default();
                        shared::fill_frame_table_dispatch(&mut layout, *frame_table_base, raw_user);
                        shared::set_frame_table_slots(
                            &mut layout,
                            scope.frame_table.selector_slot,
                            scope.frame_table.table_slot,
                        );
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
                    let pipelines_read = compute_pipelines.read().unwrap();
                    if let Some(pipeline) = current_pipeline.and_then(|h| pipelines_read.entries.get(&h)) {
                        crate::backend::validate_typed_push_constants(
                            typed_handles,
                            &pipeline.push_constant_categories,
                            &pipeline.shader_debug_name,
                        )?;
                    }
                    anyhow::bail!(
                        "GpuCommand::BindResourcesTyped must be lowered before Vulkan submit; \
                         call frame_table::lower_gpu_commands first"
                    );
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
                    let pipelines_read = compute_pipelines.read().unwrap();
                    let pipeline_layout = current_pipeline
                        .and_then(|h| pipelines_read.entries.get(&h))
                        .map(|p| p.layout);
                    // Arg data is built during context-agnostic lowering; patch in
                    // this context's frame-table slots (`_rs1`/`_rs2`) at record time.
                    let mut patched_args = arg_data.to_vec();
                    crate::backend::shared::patch_dispatch_batch_frame_table_slots(
                        &mut patched_args,
                        *count as usize,
                        scope.frame_table.selector_slot,
                        scope.frame_table.table_slot,
                    );
                    for i in 0..*count as usize {
                        let base = i * stride;
                        let layout_bytes = &patched_args[base..base + push_size];
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
                    let vk_buf = buffers
                        .read()
                        .unwrap()
                        .entries
                        .get(buffer)
                        .context("DispatchIndirect: invalid buffer handle")?
                        .buffer;
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
                        logical_device.device.cmd_dispatch_indirect(cmd, vk_buf, *offset);

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
                        let buffers_guard = buffers.read().unwrap();
                        let buf_barriers: Vec<vk::BufferMemoryBarrier2> = buf_entries
                            .iter()
                            .filter_map(|(h, usage)| {
                                buffers_guard.entries.get(h).map(|bs| {
                                    vk::BufferMemoryBarrier2::default()
                                        .src_stage_mask(slot_usage_to_vk_stage(&usage.src))
                                        .src_access_mask(slot_usage_to_vk_access(&usage.src, true))
                                        .dst_stage_mask(slot_usage_to_vk_stage(&usage.dst))
                                        .dst_access_mask(slot_usage_to_vk_access(&usage.dst, true))
                                        .buffer(bs.buffer)
                                        .offset(0)
                                        .size(vk::WHOLE_SIZE)
                                })
                            })
                            .collect();
                        // Textures: emit per-image ImageMemoryBarrier2 with layout
                        // tracking. Storage images are created UNDEFINED and must
                        // transition to GENERAL on first use; GENERAL→GENERAL on
                        // subsequent frames carries only the execution/memory
                        // dependency (same effect as the old global MemoryBarrier2
                        // but formally correct and handles cold-start).
                        let tex_img: Vec<vk::ImageMemoryBarrier2> = tex_entries
                            .iter()
                            .filter_map(|(h, usage)| {
                                view.textures.read().unwrap().entries.get(h).map(|ts| {
                                    let old_layout = ts.image_layout();
                                    ts.set_image_layout(vk::ImageLayout::GENERAL);
                                    vk::ImageMemoryBarrier2::default()
                                        .src_stage_mask(slot_usage_to_vk_stage(&usage.src))
                                        .src_access_mask(slot_usage_to_vk_access(&usage.src, false))
                                        .dst_stage_mask(slot_usage_to_vk_stage(&usage.dst))
                                        .dst_access_mask(slot_usage_to_vk_access(&usage.dst, false))
                                        .old_layout(old_layout)
                                        .new_layout(vk::ImageLayout::GENERAL)
                                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                        .image(ts.image)
                                        .subresource_range(vk::ImageSubresourceRange {
                                            aspect_mask: vk::ImageAspectFlags::COLOR,
                                            base_mip_level: 0,
                                            level_count: 1,
                                            base_array_layer: 0,
                                            layer_count: 1,
                                        })
                                })
                            })
                            .collect();
                        let dep_info = if tex_img.is_empty() {
                            vk::DependencyInfo::default().buffer_memory_barriers(&buf_barriers)
                        } else {
                            vk::DependencyInfo::default()
                                .buffer_memory_barriers(&buf_barriers)
                                .image_memory_barriers(&tex_img)
                        };
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                    }
                }
                GpuCommand::ClearBuffer { buffer, offset, size } => {
                    let _tz = tracy_zone!("vk.clear_buffer");
                    let (vk_buf, buf_size) = {
                        let buffers_guard = buffers.read().unwrap();
                        let bs = buffers_guard
                            .entries
                            .get(buffer)
                            .context("ClearBuffer: invalid buffer handle")?;
                        (bs.buffer, bs.size)
                    };
                    let clear_size = if *size == 0 {
                        buf_size.saturating_sub(*offset)
                    } else {
                        *size
                    };
                    if clear_size > 0 {
                        unsafe {
                            logical_device
                                .device
                                .cmd_fill_buffer(cmd, vk_buf, *offset, clear_size, 0);

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
                    let (is_storage, host_mapped, vk_buf) = {
                        let buffers_guard = buffers.read().unwrap();
                        let bs = buffers_guard
                            .entries
                            .get(buf_handle)
                            .context("WriteBuffer: invalid buffer handle")?;
                        (bs.is_storage, bs.host_mapped, bs.buffer)
                    };
                    // HOST_VISIBLE / CPU_READABLE paths were handled in the pre-pass;
                    // DEVICE_LOCAL storage uses the staging belt (see pre-pass).
                    if is_storage && host_mapped.is_none() {
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
                            logical_device
                                .device
                                .cmd_copy_buffer(cmd, *stg, vk_buf, std::slice::from_ref(&region));

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
                GpuCommand::WriteTexture { .. }
                | GpuCommand::WriteTextureRegion { .. }
                | GpuCommand::CopyBufferToTexture { .. } => {
                    let _tz = tracy_zone!("vk.write_texture");
                    let scratch = texture_upload_scratch
                        .get(texture_upload_idx)
                        .context("WriteTexture: scratch missing (internal)")?;
                    texture_upload_idx += 1;
                    super::texture::record_compute_texture_upload(view.devices, view.textures, cmd, scratch)?;
                }
                GpuCommand::CopyTexture { src, dst } => {
                    let _tz = tracy_zone!("vk.copy_texture");
                    let (src_image, width, height, dst_image) = {
                        let textures_read = view.textures.read().unwrap();
                        let ts = textures_read
                            .entries
                            .get(src)
                            .context("CopyTexture: src texture not found")?;
                        let dst_image = textures_read
                            .entries
                            .get(dst)
                            .context("CopyTexture: dst texture not found")?
                            .image;
                        (ts.image, ts.width, ts.height, dst_image)
                    };

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
                GpuCommand::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let _tz = tracy_zone!("vk.copy_buffer");
                    let (src_buf, dst_buf) = {
                        let buffers_read = view.buffers.read().unwrap();
                        let src_state = buffers_read.entries.get(src).context("CopyBuffer: invalid src")?;
                        let dst_state = buffers_read.entries.get(dst).context("CopyBuffer: invalid dst")?;
                        if src_offset.saturating_add(*size) > src_state.size
                            || dst_offset.saturating_add(*size) > dst_state.size
                        {
                            anyhow::bail!("CopyBuffer: size exceeds buffer bounds");
                        }
                        (src_state.buffer, dst_state.buffer)
                    };
                    unsafe {
                        let mem_barrier = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE);
                        let dep_info =
                            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                        let region = vk::BufferCopy {
                            src_offset: *src_offset,
                            dst_offset: *dst_offset,
                            size: *size,
                        };
                        logical_device
                            .device
                            .cmd_copy_buffer(cmd, src_buf, dst_buf, std::slice::from_ref(&region));
                    }
                }
                GpuCommand::CopyTextureToReadback { src, dst, layout } => {
                    let _tz = tracy_zone!("vk.copy_texture_to_readback");
                    let staging_buffer = {
                        let buffers_read = view.buffers.read().unwrap();
                        buffers_read
                            .entries
                            .get(dst)
                            .context("CopyTextureToReadback: invalid dst")?
                            .buffer
                    };
                    super::texture::record_copy_texture_to_readback(
                        cmd,
                        logical_device,
                        view.textures,
                        staging_buffer,
                        *src,
                        *layout,
                    )?;
                }
                GpuCommand::CopyRenderTarget { src, dst } => {
                    let _tz = tracy_zone!("vk.copy_render_target");
                    let (src_image, width, height, dst_image) = {
                        let render_targets_read = view.render_targets.read().unwrap();
                        let rt = render_targets_read
                            .entries
                            .get(src)
                            .context("CopyRenderTarget: src render target not found")?;
                        let textures_read = view.textures.read().unwrap();
                        let dst_image = textures_read
                            .entries
                            .get(dst)
                            .context("CopyRenderTarget: dst texture not found")?
                            .image;
                        (rt.image, rt.width, rt.height, dst_image)
                    };

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
        let ld = view.devices.get(&device_handle).context("Invalid device handle")?;
        ld.timeline_next.fetch_add(1, Ordering::Relaxed)
    };

    let used_slots = collect_slot_keys_from_gpu_commands(&commands, view.compute_pipelines, view.buffers);
    if let Some(ld) = view.devices.get(&device_handle) {
        ld.descriptors
            .lock()
            .unwrap()
            .record_slot_usage(ctx, signal_value, used_slots);
    }

    let timeline_sem = scope.sc.lock().unwrap().timeline_semaphore;
    let submit_device_core = view.devices.get(&device_handle).context("Invalid device handle")?;
    let queue_lock = std::sync::Arc::clone(&submit_device_core.queue_lock);
    let signal_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(timeline_sem)
        .value(signal_value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let wait_infos = build_cross_submit_wait_infos(view, sync)?;
    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
    let submit_info2 = if wait_infos.is_empty() {
        vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info))
    } else {
        vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .wait_semaphore_infos(&wait_infos)
            .signal_semaphore_infos(std::slice::from_ref(&signal_info))
    };

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
        let ctx_pool = scope.sc.lock().unwrap().command_pool;
        unsafe {
            submit_device_core.device.free_command_buffers(ctx_pool, &[cmd]);
            if let Some(prof) = vk_gpu_profile.take() {
                submit_device_core.device.destroy_query_pool(prof.pool, None);
            }
        }
        return Err(anyhow::anyhow!("Failed to queue_submit2 command buffer: {:?}", e));
    }

    {
        let mut sc = scope.sc.lock().unwrap();
        sc.last_submitted_seq = signal_value;
        sc.timeline_cmd_buffers.entry(signal_value).or_default().push(cmd);
    }

    if !texture_upload_scratch.is_empty() {
        let entries: Vec<staging::TextureStagingEntry> = texture_upload_scratch.into_iter().map(|s| s.entry).collect();
        {
            scope
                .sc
                .lock()
                .unwrap()
                .texture_staging_pool
                .release(signal_value, entries);
        }
    }

    {
        scope.sc.lock().unwrap().staging_belt.finish(signal_value);
    }

    debug_assert_eq!(
        belt_idx,
        belt_slices.len(),
        "WriteBuffer DEVICE_LOCAL count must match belt pre-pass"
    );

    if let Some(prof) = vk_gpu_profile {
        let (device_clone, timeline_sem) = {
            let ld = view.devices.get(&device_handle).context("Invalid device handle")?;
            let sem = scope.sc.lock().unwrap().timeline_semaphore;
            (ld.device.clone(), sem)
        };
        unsafe {
            vulkan_finish_gpu_profile(view, ctx, &device_clone, timeline_sem, signal_value, cmd, prof)?;
        }
    }

    {
        if let Some(ld) = view.devices.get(&device_handle) {
            let descriptors_arc = std::sync::Arc::clone(&ld.descriptors);
            let mut registry = descriptors_arc.lock().unwrap();
            let completed_values =
                super::types::snapshot_context_completed_values(&ld.device, view.contexts, device_handle);
            registry.drain_ready_slot_reclamations(&completed_values);
        }
    }

    Ok(signal_value)
}

pub(super) fn submit(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    commands: &[GpuCommand],
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    submit_with_scope(
        &super::submit_session::scope_from_state(state, ctx)?,
        ctx,
        commands,
        sync,
    )
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
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    submit_graph_with_scope(
        &super::submit_session::scope_from_state(state, ctx)?,
        ctx,
        commands,
        None,
        sync,
    )
}

/// Inner implementation shared by `submit_graph` and `submit_graph_and_retain`.
///
/// When `retain_key` is `Some(key)`:
/// - the CB is recorded without `ONE_TIME_SUBMIT` (driver keeps it executable)
/// - after a successful submit the CB is stored in `SubmissionContext::retained_compute_cbs`
///   rather than `timeline_cmd_buffers` (so it is not freed by the normal reap cycle)
/// - any WriteBuffer / WriteTexture commands cause a fallback to normal (non-retained) submit
///
/// When `retain_key` is `None` (normal path):
/// - the CB is recorded with `ONE_TIME_SUBMIT`
/// - after submit it is stored in `timeline_cmd_buffers` and freed once the GPU retires it
pub(super) fn submit_graph_with_scope(
    scope: &VulkanSubmitScope<'_>,
    ctx: super::ContextHandle,
    commands: &[GraphCommand],
    retain_key: Option<u64>,
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    scope.assert_ctx(ctx);
    let view = &scope.view;
    let device_handle = scope.device_handle;
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
                    | GpuCommand::CopyBufferToTexture { .. }
            )
        )
    });
    let has_write_texture_graph = commands.iter().any(|c| {
        matches!(
            c,
            GraphCommand::Compute(
                GpuCommand::WriteTexture { .. }
                    | GpuCommand::WriteTextureRegion { .. }
                    | GpuCommand::CopyBufferToTexture { .. }
            )
        )
    });

    if has_upload {
        let _rz = tracy_zone!("vk.submit_graph.belt_reclaim");
        let completed_timeline = scope.completed_timeline_value();

        {
            {
                scope.sc.lock().unwrap().staging_belt.reclaim(
                    view.compute_fence_pool,
                    view.devices,
                    completed_timeline,
                )?;
            }
        }
        reap_signaled_fences(view);

        if has_write_texture_graph {
            {
                scope
                    .sc
                    .lock()
                    .unwrap()
                    .texture_staging_pool
                    .reclaim(completed_timeline);
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
                        // Extract Copy fields so the view.buffers borrow ends
                        // before we take &mut view.contexts for the belt.
                        let (host_mapped, is_storage, buf_device, buf_memory) = {
                            let buffers_read = view.buffers.read().unwrap();
                            let buf = buffers_read
                                .entries
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
                            let dev = view.devices.get(&buf_device).context("WriteBuffer: device invalid")?;
                            unsafe {
                                let ptr = dev
                                    .map_memory2(buf_memory, *offset, data.len() as u64)
                                    .context("WriteBuffer: map failed")?;
                                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
                                dev.unmap_memory2(buf_memory).context("WriteBuffer: unmap failed")?;
                            }
                        } else {
                            let dev = view.devices.get(&buf_device).context("WriteBuffer: device invalid")?;
                            let mut sc = scope.sc.lock().unwrap();
                            let (stg_buf, stg_off) = sc.staging_belt.write(view.instance, dev, data)?;
                            belt_slices.push((stg_buf, stg_off));
                        }
                    }
                    GpuCommand::WriteTexture {
                        texture,
                        data,
                        width,
                        height,
                    } => {
                        let mut sc_guard = scope.sc.lock().unwrap();
                        let pool = &mut sc_guard.texture_staging_pool;
                        texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                            view.instance,
                            view.devices,
                            view.textures,
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
                        let mut sc_guard = scope.sc.lock().unwrap();
                        let pool = &mut sc_guard.texture_staging_pool;
                        texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                            view.instance,
                            view.devices,
                            view.textures,
                            pool,
                            *texture,
                            data,
                            *x,
                            *y,
                            *width,
                            *height,
                        )?);
                    }
                    GpuCommand::CopyBufferToTexture {
                        src,
                        src_offset,
                        dst,
                        x,
                        y,
                        width,
                        height,
                        ..
                    } => {
                        let flat = super::texture::copy_buffer_to_texture_flat_bytes(
                            view.textures,
                            view.buffers,
                            *src,
                            *src_offset,
                            *dst,
                            *width,
                            *height,
                        )?;
                        let mut sc_guard = scope.sc.lock().unwrap();
                        texture_upload_scratch.push(super::texture::allocate_compute_texture_staging(
                            view.instance,
                            view.devices,
                            view.textures,
                            &mut sc_guard.texture_staging_pool,
                            *dst,
                            &flat,
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
        let ld = view.devices.get(&device_handle).context("Invalid device handle")?;
        let mut sc = scope.sc.lock().unwrap();
        let cb = acquire_cmd_buffer(ld, &mut sc)?;
        // Use ONE_TIME_SUBMIT for normal submits (driver hint for optimization).
        // Retained CBs use SIMULTANEOUS_USE so a still-pending CB may be resubmitted
        // without a CPU wait (VUID-vkQueueSubmit2-commandBuffer-03875).
        let flags = if retain_key.is_none() {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        } else {
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE
        };
        let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);
        if let Err(e) = unsafe { ld.device.begin_command_buffer(cb, &begin_info) } {
            sc.free_cmd_buffers.push(cb);
            return Err(anyhow::anyhow!("Failed to begin command buffer: {:?}", e));
        }
        cb
    };

    let logical_device = view.devices.get(&device_handle).context("Invalid device handle")?;

    let (dispatch_count_graph, dispatch_labels_graph) = collect_dispatch_labels_graph(commands);
    let mut vk_gpu_profile =
        unsafe { create_vulkan_gpu_profile_pool(logical_device, false, dispatch_count_graph, dispatch_labels_graph)? };

    // Cross-submission acquire: make prior submit's writes visible to this graph's
    // first reads. Render-only schemes (e.g. reading a buffer written by a prior
    // compute submit) need fragment/vertex stages here — not just compute.
    if SubmitSync::use_legacy_acquire_from(sync) {
        unsafe {
            let acquire = vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::TRANSFER
                        | vk::PipelineStageFlags2::ALL_GRAPHICS,
                )
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::TRANSFER
                        | vk::PipelineStageFlags2::DRAW_INDIRECT
                        | vk::PipelineStageFlags2::VERTEX_SHADER
                        | vk::PipelineStageFlags2::FRAGMENT_SHADER
                        | vk::PipelineStageFlags2::VERTEX_INPUT,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::TRANSFER_READ
                        | vk::AccessFlags2::INDIRECT_COMMAND_READ
                        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                );
            let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&acquire));
            logical_device.device.cmd_pipeline_barrier2(cmd, &dep);
        }
    }

    unsafe {
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
    let compute_pipelines = &view.compute_pipelines;
    let buffers = &view.buffers;
    let mut current_compute_pipeline: Option<ComputePipelineHandle> = None;
    let mut belt_idx = 0usize;
    let mut texture_upload_idx = 0usize;
    let mut rendered_targets: Vec<RenderTargetHandle> = Vec::new();
    let mut frame_table_prologue_in_cb = false;
    let mut frame_table_row: Option<u32> = None;

    for graph_cmd in commands {
        match graph_cmd {
            GraphCommand::Compute(gpu_cmd) => match gpu_cmd {
                GpuCommand::FrameTableStaging { data } => {
                    frame_table_prologue_in_cb = true;
                    let row = super::frame_table::record_prologue(
                        view.contexts,
                        ctx,
                        &scope.frame_table,
                        view.buffers,
                        logical_device,
                        cmd,
                        data,
                    )?;
                    frame_table_row = Some(row);
                }
                GpuCommand::SetPipeline(handle) => {
                    let _tz = tracy_zone!("vk.set_pipeline");
                    if let Some(pipeline_state) = compute_pipelines.read().unwrap().entries.get(handle) {
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
                    frame_table_base,
                } => {
                    let pipelines_read = compute_pipelines.read().unwrap();
                    if let Some(pipeline) = current_compute_pipeline.and_then(|p| pipelines_read.entries.get(&p)) {
                        crate::backend::with_layout_validation(|| {
                            crate::backend::validate_raw_binding_strides(
                                raw_indices,
                                &pipeline.push_constant_categories,
                                &pipeline.binding_element_strides,
                                |idx, cat| {
                                    buffer_stride_for_bindless_index(
                                        &buffers.read().unwrap().entries,
                                        device_handle,
                                        idx,
                                        cat,
                                    )
                                },
                                &pipeline.shader_debug_name,
                            )
                        })?;
                        let mut layout = PushLayout::default();
                        shared::fill_frame_table_dispatch(&mut layout, *frame_table_base, raw_user);
                        shared::set_frame_table_slots(
                            &mut layout,
                            scope.frame_table.selector_slot,
                            scope.frame_table.table_slot,
                        );
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
                    let pipelines_read = compute_pipelines.read().unwrap();
                    if let Some(pipeline) = current_compute_pipeline.and_then(|p| pipelines_read.entries.get(&p)) {
                        crate::backend::validate_typed_push_constants(
                            typed_handles,
                            &pipeline.push_constant_categories,
                            &pipeline.shader_debug_name,
                        )?;
                    }
                    anyhow::bail!(
                        "GpuCommand::BindResourcesTyped must be lowered before Vulkan graph submit; \
                         call frame_table::lower_gpu_commands first"
                    );
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
                    let pipelines_read = compute_pipelines.read().unwrap();
                    let pipeline_layout = current_compute_pipeline
                        .and_then(|h| pipelines_read.entries.get(&h))
                        .map(|p| p.layout);
                    // Arg data is built during context-agnostic lowering; patch in
                    // this context's frame-table slots (`_rs1`/`_rs2`) at record time.
                    let mut patched_args = arg_data.to_vec();
                    crate::backend::shared::patch_dispatch_batch_frame_table_slots(
                        &mut patched_args,
                        *count as usize,
                        scope.frame_table.selector_slot,
                        scope.frame_table.table_slot,
                    );
                    for i in 0..*count as usize {
                        let base = i * stride;
                        let layout_bytes = &patched_args[base..base + push_size];
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
                    let vk_buf = buffers
                        .read()
                        .unwrap()
                        .entries
                        .get(buffer)
                        .context("DispatchIndirect: invalid buffer handle")?
                        .buffer;
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
                        logical_device.device.cmd_dispatch_indirect(cmd, vk_buf, *offset);

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
                        let buffers_guard = buffers.read().unwrap();
                        let buf_barriers: Vec<vk::BufferMemoryBarrier2> = buf_entries
                            .iter()
                            .filter_map(|(h, usage)| {
                                buffers_guard.entries.get(h).map(|bs| {
                                    vk::BufferMemoryBarrier2::default()
                                        .src_stage_mask(slot_usage_to_vk_stage(&usage.src))
                                        .src_access_mask(slot_usage_to_vk_access(&usage.src, true))
                                        .dst_stage_mask(slot_usage_to_vk_stage(&usage.dst))
                                        .dst_access_mask(slot_usage_to_vk_access(&usage.dst, true))
                                        .buffer(bs.buffer)
                                        .offset(0)
                                        .size(vk::WHOLE_SIZE)
                                })
                            })
                            .collect();
                        // Textures: emit per-image ImageMemoryBarrier2 with layout
                        // tracking. Storage images are created UNDEFINED and must
                        // transition to GENERAL on first use; GENERAL→GENERAL on
                        // subsequent frames carries only the execution/memory
                        // dependency (same effect as the old global MemoryBarrier2
                        // but formally correct and handles cold-start).
                        let tex_img: Vec<vk::ImageMemoryBarrier2> = tex_entries
                            .iter()
                            .filter_map(|(h, usage)| {
                                view.textures.read().unwrap().entries.get(h).map(|ts| {
                                    let old_layout = ts.image_layout();
                                    ts.set_image_layout(vk::ImageLayout::GENERAL);
                                    vk::ImageMemoryBarrier2::default()
                                        .src_stage_mask(slot_usage_to_vk_stage(&usage.src))
                                        .src_access_mask(slot_usage_to_vk_access(&usage.src, false))
                                        .dst_stage_mask(slot_usage_to_vk_stage(&usage.dst))
                                        .dst_access_mask(slot_usage_to_vk_access(&usage.dst, false))
                                        .old_layout(old_layout)
                                        .new_layout(vk::ImageLayout::GENERAL)
                                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                        .image(ts.image)
                                        .subresource_range(vk::ImageSubresourceRange {
                                            aspect_mask: vk::ImageAspectFlags::COLOR,
                                            base_mip_level: 0,
                                            level_count: 1,
                                            base_array_layer: 0,
                                            layer_count: 1,
                                        })
                                })
                            })
                            .collect();
                        let dep_info = if tex_img.is_empty() {
                            vk::DependencyInfo::default().buffer_memory_barriers(&buf_barriers)
                        } else {
                            vk::DependencyInfo::default()
                                .buffer_memory_barriers(&buf_barriers)
                                .image_memory_barriers(&tex_img)
                        };
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                    }
                }
                GpuCommand::ClearBuffer { buffer, offset, size } => {
                    let _tz = tracy_zone!("vk.clear_buffer");
                    let (vk_buf, buf_size) = {
                        let buffers_guard = buffers.read().unwrap();
                        let bs = buffers_guard
                            .entries
                            .get(buffer)
                            .context("ClearBuffer: invalid buffer handle")?;
                        (bs.buffer, bs.size)
                    };
                    let clear_size = if *size == 0 {
                        buf_size.saturating_sub(*offset)
                    } else {
                        *size
                    };
                    if clear_size > 0 {
                        unsafe {
                            logical_device
                                .device
                                .cmd_fill_buffer(cmd, vk_buf, *offset, clear_size, 0);
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
                    let (is_storage, host_mapped, vk_buf) = {
                        let buffers_guard = buffers.read().unwrap();
                        let bs = buffers_guard
                            .entries
                            .get(buf_handle)
                            .context("WriteBuffer: invalid buffer handle")?;
                        (bs.is_storage, bs.host_mapped, bs.buffer)
                    };
                    if is_storage && host_mapped.is_none() {
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
                            logical_device
                                .device
                                .cmd_copy_buffer(cmd, *stg, vk_buf, std::slice::from_ref(&region));
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
                GpuCommand::WriteTexture { .. }
                | GpuCommand::WriteTextureRegion { .. }
                | GpuCommand::CopyBufferToTexture { .. } => {
                    let _tz = tracy_zone!("vk.write_texture");
                    let scratch = texture_upload_scratch
                        .get(texture_upload_idx)
                        .context("WriteTexture: scratch missing (internal)")?;
                    texture_upload_idx += 1;
                    super::texture::record_compute_texture_upload(view.devices, view.textures, cmd, scratch)?;
                }
                GpuCommand::CopyTexture { src, dst } => {
                    let _tz = tracy_zone!("vk.copy_texture");
                    let (src_image, width, height, dst_image) = {
                        let textures_read = view.textures.read().unwrap();
                        let ts = textures_read
                            .entries
                            .get(src)
                            .context("CopyTexture: src texture not found")?;
                        let dst_image = textures_read
                            .entries
                            .get(dst)
                            .context("CopyTexture: dst texture not found")?
                            .image;
                        (ts.image, ts.width, ts.height, dst_image)
                    };
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
                GpuCommand::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let _tz = tracy_zone!("vk.copy_buffer");
                    let (src_buf, dst_buf) = {
                        let buffers_read = view.buffers.read().unwrap();
                        let src_state = buffers_read.entries.get(src).context("CopyBuffer: invalid src")?;
                        let dst_state = buffers_read.entries.get(dst).context("CopyBuffer: invalid dst")?;
                        if src_offset.saturating_add(*size) > src_state.size
                            || dst_offset.saturating_add(*size) > dst_state.size
                        {
                            anyhow::bail!("CopyBuffer: size exceeds buffer bounds");
                        }
                        (src_state.buffer, dst_state.buffer)
                    };
                    unsafe {
                        let mem_barrier = vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE);
                        let dep_info =
                            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));
                        logical_device.device.cmd_pipeline_barrier2(cmd, &dep_info);
                        let region = vk::BufferCopy {
                            src_offset: *src_offset,
                            dst_offset: *dst_offset,
                            size: *size,
                        };
                        logical_device
                            .device
                            .cmd_copy_buffer(cmd, src_buf, dst_buf, std::slice::from_ref(&region));
                    }
                }
                GpuCommand::CopyTextureToReadback { src, dst, layout } => {
                    let _tz = tracy_zone!("vk.copy_texture_to_readback");
                    let staging_buffer = {
                        let buffers_read = view.buffers.read().unwrap();
                        buffers_read
                            .entries
                            .get(dst)
                            .context("CopyTextureToReadback: invalid dst")?
                            .buffer
                    };
                    super::texture::record_copy_texture_to_readback(
                        cmd,
                        logical_device,
                        view.textures,
                        staging_buffer,
                        *src,
                        *layout,
                    )?;
                }
                GpuCommand::CopyRenderTarget { src, dst } => {
                    let _tz = tracy_zone!("vk.copy_render_target");
                    let (src_image, width, height, dst_image) = {
                        let render_targets_read = view.render_targets.read().unwrap();
                        let rt = render_targets_read
                            .entries
                            .get(src)
                            .context("CopyRenderTarget: src render target not found")?;
                        let textures_read = view.textures.read().unwrap();
                        let dst_image = textures_read
                            .entries
                            .get(dst)
                            .context("CopyRenderTarget: dst texture not found")?
                            .image;
                        (rt.image, rt.width, rt.height, dst_image)
                    };

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

                let (staging_data, lowered, has_render_bindings) =
                    super::frame_table::prepare_render_commands(buffers, view.pipelines, render_cmds)?;
                if has_render_bindings {
                    if frame_table_prologue_in_cb {
                        let graph_staging = super::frame_table::extract_staging_from_graph(commands)
                            .map(|data| data.to_vec())
                            .unwrap_or_else(|| vec![0u32; crate::frame_table::FRAME_TABLE_TABLE_U32S]);
                        let sync_data =
                            super::frame_table::merge_staging_for_render_sync(&graph_staging, &staging_data);
                        super::frame_table::sync_table_row_to_device(
                            &scope.frame_table,
                            view.buffers,
                            logical_device,
                            cmd,
                            &sync_data,
                        )?;
                    } else {
                        let row = super::frame_table::record_prologue(
                            view.contexts,
                            ctx,
                            &scope.frame_table,
                            view.buffers,
                            logical_device,
                            cmd,
                            &staging_data,
                        )?;
                        frame_table_row = Some(row);
                    }
                }

                super::render_target::record_render_pass_to_buffer(
                    view.devices,
                    view.render_targets,
                    device_handle,
                    *target,
                    &lowered,
                    cmd,
                    |cb, cmds, ld, cur_pipe| {
                        let pipelines_read = view.pipelines.read().unwrap();
                        let buffers_read = view.buffers.read().unwrap();
                        super::render_commands::record(
                            cb,
                            cmds,
                            ld,
                            &pipelines_read.entries,
                            &buffers_read.entries,
                            cur_pipe,
                            (scope.frame_table.selector_slot, scope.frame_table.table_slot),
                        )
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
        let ld = view.devices.get(&device_handle).context("Invalid device handle")?;
        ld.timeline_next.fetch_add(1, Ordering::Relaxed)
    };

    let used_slots =
        collect_slot_keys_from_graph_commands(commands, view.compute_pipelines, view.pipelines, view.buffers);
    if let Some(ld) = view.devices.get(&device_handle) {
        ld.descriptors
            .lock()
            .unwrap()
            .record_slot_usage(ctx, signal_value, used_slots.iter().copied());
    }

    let timeline_sem = scope.sc.lock().unwrap().timeline_semaphore;
    let submit_device = view.devices.get(&device_handle).context("Invalid device handle")?;
    let queue_lock = std::sync::Arc::clone(&submit_device.queue_lock);
    let signal_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(timeline_sem)
        .value(signal_value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let wait_infos = build_cross_submit_wait_infos(view, sync)?;
    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
    let submit_info2 = if wait_infos.is_empty() {
        vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info))
    } else {
        vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .wait_semaphore_infos(&wait_infos)
            .signal_semaphore_infos(std::slice::from_ref(&signal_info))
    };

    let retain_plan = if let Some(key) = retain_key {
        let ft = &scope.frame_table;
        let pin_row_index = super::frame_table::extract_staging_from_graph(commands)
            .is_some()
            .then_some(frame_table_row)
            .flatten();
        if let Some(row) = pin_row_index {
            super::frame_table::pin_row(ft, row)?;
        }
        Some((key, pin_row_index))
    } else {
        None
    };

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
        let ctx_pool = scope.sc.lock().unwrap().command_pool;
        unsafe {
            submit_device.device.free_command_buffers(ctx_pool, &[cmd]);
            if let Some(prof) = vk_gpu_profile.take() {
                submit_device.device.destroy_query_pool(prof.pool, None);
            }
        }
        if let Some((_, Some(row))) = retain_plan {
            super::frame_table::unpin_row(&scope.frame_table, row);
        }
        return Err(anyhow::anyhow!("Failed to queue_submit2 command buffer: {:?}", e));
    }

    // Post-submit: store the CB for lifecycle management.
    {
        let mut sc = scope.sc.lock().unwrap();
        sc.last_submitted_seq = signal_value;
        if let Some((key, frame_table_row)) = retain_plan {
            let pin_slots = used_slots.clone();
            let replaced = sc.retained_compute_cbs.insert(
                key,
                super::types::RetainedVkCb {
                    command_buffer: cmd,
                    used_slots,
                    frame_table_row,
                    last_signal_value: signal_value,
                },
            );
            let unpin_slots = replaced.map(|old| old.used_slots).unwrap_or_default();
            drop(sc);
            if !unpin_slots.is_empty() || !pin_slots.is_empty() {
                let ld = scope
                    .view
                    .devices
                    .get(&scope.device_handle)
                    .expect("submit scope device handle must exist");
                let mut registry = ld.descriptors.lock().unwrap();
                registry.unpin_retained_slots(unpin_slots);
                registry.pin_retained_slots(pin_slots);
            }
        } else {
            sc.timeline_cmd_buffers.entry(signal_value).or_default().push(cmd);
        }
    }

    if !texture_upload_scratch.is_empty() {
        let entries: Vec<staging::TextureStagingEntry> = texture_upload_scratch.into_iter().map(|s| s.entry).collect();
        {
            scope
                .sc
                .lock()
                .unwrap()
                .texture_staging_pool
                .release(signal_value, entries);
        }
    }

    {
        scope.sc.lock().unwrap().staging_belt.finish(signal_value);
    }

    if let Some(prof) = vk_gpu_profile {
        let (device_clone, timeline_sem) = {
            let ld = view.devices.get(&device_handle).context("Invalid device handle")?;
            let sem = scope.sc.lock().unwrap().timeline_semaphore;
            (ld.device.clone(), sem)
        };
        unsafe {
            vulkan_finish_gpu_profile(view, ctx, &device_clone, timeline_sem, signal_value, cmd, prof)?;
        }
    }

    // Mark rendered targets
    for t in rendered_targets {
        if let Some(rt) = view.render_targets.read().unwrap().entries.get(&t) {
            rt.has_rendered.store(true, Ordering::Relaxed);
        }
    }

    {
        if let Some(ld) = view.devices.get(&device_handle) {
            let descriptors_arc = std::sync::Arc::clone(&ld.descriptors);
            let mut registry = descriptors_arc.lock().unwrap();
            let completed_values =
                super::types::snapshot_context_completed_values(&ld.device, view.contexts, device_handle);
            registry.drain_ready_slot_reclamations(&completed_values);
        }
    }

    Ok(signal_value)
}

/// Record, submit, and retain a dispatch command buffer keyed by `key`.
///
/// The CB is recorded with `SIMULTANEOUS_USE` so a still-pending retained CB may be
/// resubmitted without a CPU wait (VUID-vkQueueSubmit2-commandBuffer-03875).
/// Stored in `SubmissionContext::retained_compute_cbs` rather than `timeline_cmd_buffers`.
/// On subsequent frames call [`try_resubmit_retained`] to re-execute without re-recording.
/// If commands contain any WriteBuffer/WriteTexture nodes the call falls back to a normal
/// (non-retained) submit via [`submit_graph`].
pub(super) fn submit_graph_and_retain(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    commands: &[GraphCommand],
    key: u64,
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    let scope = super::submit_session::scope_from_state(state, ctx)?;
    evict_retained_with_scope(&scope, ctx, key);
    submit_graph_with_scope(&scope, ctx, commands, Some(key), sync)
}

/// Re-execute the retained dispatch CB without re-recording.
pub(super) fn try_resubmit_retained(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    key: u64,
    sync: Option<&SubmitSync>,
) -> Result<Option<TimelineValue>> {
    try_resubmit_retained_with_scope(&super::submit_session::scope_from_state(state, ctx)?, ctx, key, sync)
}

/// Re-execute the retained dispatch CB without re-recording.
pub(super) fn try_resubmit_retained_with_scope(
    scope: &VulkanSubmitScope<'_>,
    ctx: super::ContextHandle,
    key: u64,
    sync: Option<&SubmitSync>,
) -> Result<Option<TimelineValue>> {
    scope.assert_ctx(ctx);
    let view = &scope.view;
    let device_handle = scope.device_handle;
    let (timeline_sem, retained) = {
        let sc = scope.sc.lock().unwrap();
        let timeline_sem = sc.timeline_semaphore;
        let retained = sc
            .retained_compute_cbs
            .get(&key)
            .map(|r| (r.command_buffer, r.used_slots.clone()));
        (timeline_sem, retained)
    };

    let Some((cmd, used_slots)) = retained else {
        return Ok(None);
    };

    let signal_value = {
        let ld = view.devices.get(&device_handle).context("Invalid device handle")?;
        ld.timeline_next.fetch_add(1, Ordering::Relaxed)
    };
    let submit_device = view.devices.get(&device_handle).context("Invalid device handle")?;
    let queue_lock = std::sync::Arc::clone(&submit_device.queue_lock);
    let signal_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(timeline_sem)
        .value(signal_value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let wait_infos = build_cross_submit_wait_infos(view, sync)?;
    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
    let submit_info2 = if wait_infos.is_empty() {
        vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info))
    } else {
        vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .wait_semaphore_infos(&wait_infos)
            .signal_semaphore_infos(std::slice::from_ref(&signal_info))
    };

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

    if let Some(ld) = view.devices.get(&device_handle) {
        ld.descriptors
            .lock()
            .unwrap()
            .record_slot_usage(ctx, signal_value, used_slots);
    }

    // Update the last signal value so eviction can defer-free the CB safely.
    {
        let mut sc = scope.sc.lock().unwrap();
        if let Some(retained) = sc.retained_compute_cbs.values_mut().find(|r| r.command_buffer == cmd) {
            retained.last_signal_value = signal_value;
        }
    }

    {
        if let Some(ld) = view.devices.get(&device_handle) {
            let descriptors_arc = std::sync::Arc::clone(&ld.descriptors);
            let mut registry = descriptors_arc.lock().unwrap();
            let completed_values =
                super::types::snapshot_context_completed_values(&ld.device, view.contexts, device_handle);
            registry.drain_ready_slot_reclamations(&completed_values);
        }
    }
    {
        scope.sc.lock().unwrap().last_submitted_seq = signal_value;
    }

    Ok(Some(signal_value))
}

fn evict_retained_on_context(
    frame_table: &super::frame_table::ContextFrameTable,
    ld: &super::types::LogicalDevice,
    ctx: super::ContextHandle,
    key: u64,
    contexts: &super::types::SharedContextMap,
) {
    let removed = if let Some(sc_arc) = contexts.read().unwrap().get(&ctx) {
        let mut sc = sc_arc.lock().unwrap();
        sc.retained_compute_cbs.remove(&key)
    } else {
        None
    };
    if let Some(old) = removed {
        ld.descriptors.lock().unwrap().unpin_retained_slots(old.used_slots);
        if let Some(row) = old.frame_table_row {
            super::frame_table::unpin_row(frame_table, row);
        }
        if let Some(sc_arc) = contexts.read().unwrap().get(&ctx) {
            sc_arc
                .lock()
                .unwrap()
                .timeline_cmd_buffers
                .entry(old.last_signal_value)
                .or_default()
                .push(old.command_buffer);
        }
    }
}

pub(super) fn evict_retained_pinning_row_for_context(
    contexts: &super::types::SharedContextMap,
    frame_table: &super::frame_table::ContextFrameTable,
    ld: &super::types::LogicalDevice,
    ctx: super::ContextHandle,
    row: u32,
) {
    let keys: Vec<u64> = {
        let contexts_read = contexts.read().unwrap();
        let Some(sc_arc) = contexts_read.get(&ctx) else {
            return;
        };
        let sc = sc_arc.lock().unwrap();
        sc.retained_compute_cbs
            .iter()
            .filter(|(_, g)| g.frame_table_row == Some(row))
            .map(|(k, _)| *k)
            .collect()
    };
    for key in keys {
        evict_retained_on_context(frame_table, ld, ctx, key, contexts);
    }
}

/// Evict the retained dispatch CB for `key`, returning the `VkCommandBuffer` to `free_cmd_buffers`.
pub(super) fn evict_retained_with_scope(scope: &VulkanSubmitScope<'_>, ctx: super::ContextHandle, key: u64) {
    scope.assert_ctx(ctx);
    let ld = scope
        .view
        .devices
        .get(&scope.device_handle)
        .expect("submit scope device handle must exist");
    evict_retained_on_context(&scope.frame_table, ld, ctx, key, scope.view.contexts);
}

pub(super) fn evict_retained(state: &super::types::VulkanState, ctx: super::ContextHandle, key: u64) {
    if let Ok(scope) = super::submit_session::scope_from_state(state, ctx) {
        evict_retained_with_scope(&scope, ctx, key);
    }
}

fn reap_timeline_cmd_buffers_up_to_with_view(
    view: &VulkanSubmitView<'_>,
    ctx: super::ContextHandle,
    max_completed_value: u64,
) {
    let (device, pool, keys): (DeviceHandle, vk::CommandPool, Vec<u64>) = {
        let contexts = view.contexts.read().unwrap();
        let sc_arc = match contexts.get(&ctx) {
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
        let contexts = view.contexts.read().unwrap();
        let sc_arc = contexts.get(&ctx).expect("context");
        let mut sc = sc_arc.lock().unwrap();
        keys.iter()
            .filter_map(|k| sc.timeline_cmd_buffers.remove(k))
            .flatten()
            .collect()
    };
    if let Some(ld) = view.devices.get(&device) {
        for cb in cbs_to_free {
            unsafe {
                ld.device.free_command_buffers(pool, &[cb]);
            }
        }
    }
}

pub(super) fn reap_timeline_cmd_buffers_up_to(
    state: &super::types::VulkanState,
    ctx: super::ContextHandle,
    max_completed_value: u64,
) {
    reap_timeline_cmd_buffers_up_to_with_view(&state.submit_view(), ctx, max_completed_value);
}
