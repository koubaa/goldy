//! Per-context submission stream lifecycle (Vulkan).

use super::types::{SubmissionContext, TimelineWaitTarget, VulkanState};
use super::{ContextHandle, DeviceHandle};
use crate::backend::ContextDestroyHandle;
use anyhow::{Context as _, Result};
use ash::vk;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Record which native semaphore must reach `value` before that global timeline
/// ticket is considered GPU-retired.
pub(super) fn register_timeline_wait_target(ld: &super::types::LogicalDevice, value: u64, target: TimelineWaitTarget) {
    if value == 0 {
        return;
    }
    ld.timeline_wait_targets.lock().unwrap().insert(value, target);
}

/// Reserve the next global timeline value as [`TimelineWaitTarget::DeviceOwner`].
///
/// Caller must already hold [`super::types::LogicalDevice::queue_lock`] so owner
/// signals cannot be enqueued out of order when present and render submits overlap.
pub(super) fn reserve_device_owner_timeline_locked(ld: &super::types::LogicalDevice) -> u64 {
    let value = ld.timeline_next.fetch_add(1, Ordering::Relaxed);
    register_timeline_wait_target(ld, value, TimelineWaitTarget::DeviceOwner);
    value
}

fn target_completed_value(state: &VulkanState, device: DeviceHandle, target: TimelineWaitTarget) -> u64 {
    let Some(ld) = state.devices.get(&device) else {
        return 0;
    };
    match target {
        TimelineWaitTarget::Context(ctx) => state
            .contexts
            .read()
            .unwrap()
            .get(&ctx)
            .map(|sc_arc| {
                let sem = sc_arc.lock().unwrap().timeline_semaphore;
                unsafe { ld.device.get_semaphore_counter_value(sem).unwrap_or(0) }
            })
            .unwrap_or(0),
        TimelineWaitTarget::DeviceOwner => owner_timeline_semaphore(state, device)
            .map(|sem| unsafe { ld.device.get_semaphore_counter_value(sem).unwrap_or(0) })
            .unwrap_or(0),
    }
}

fn resolve_timeline_wait_semaphore(
    state: &VulkanState,
    device: DeviceHandle,
    target: TimelineWaitTarget,
) -> Option<vk::Semaphore> {
    match target {
        TimelineWaitTarget::Context(ctx) => state
            .contexts
            .read()
            .unwrap()
            .get(&ctx)
            .map(|sc_arc| sc_arc.lock().unwrap().timeline_semaphore),
        TimelineWaitTarget::DeviceOwner => owner_timeline_semaphore(state, device),
    }
}

/// Advance and return the highest global timeline value whose owning semaphore has
/// completed, starting from the cached horizon and floored by post-destroy retirement.
pub(super) fn advance_timeline_retired(state: &VulkanState, device: DeviceHandle) -> u64 {
    let Some(ld) = state.devices.get(&device) else {
        return 0;
    };
    let floor = ld.retired_floor.load(Ordering::Relaxed);
    let mut retired = ld.timeline_retired.load(Ordering::Relaxed).max(floor);

    loop {
        let next = retired.saturating_add(1);
        let target = {
            let targets = ld.timeline_wait_targets.lock().unwrap();
            targets.get(&next).copied()
        };
        let Some(target) = target else {
            break;
        };
        if target_completed_value(state, device, target) < next {
            break;
        }
        retired = next;
    }

    if retired > ld.timeline_retired.load(Ordering::Relaxed) {
        ld.timeline_retired.store(retired, Ordering::Relaxed);
        ld.timeline_wait_targets
            .lock()
            .unwrap()
            .retain(|value, _| *value > retired);
    }
    retired.max(floor)
}

/// Latest device-global seq retired on `device` (contiguous prefix over attributed values, floored).
pub(super) fn device_retired(state: &VulkanState, device: DeviceHandle) -> u64 {
    advance_timeline_retired(state, device)
}

pub(super) fn create(state: &mut VulkanState, device: DeviceHandle) -> Result<ContextHandle> {
    let ld = state.devices.get(&device).context("Invalid device handle")?.clone();

    let queue_index = ld
        .free_compute_queue_indices
        .lock()
        .unwrap()
        .pop_front()
        .context("Vulkan per-context compute queue pool exhausted — destroy a context or create another device")?;
    let queue = ld.compute_queues[queue_index];
    let compute_family = ld.compute_queue_family;
    // One-queue devices: all contexts share the graphics/present queue and its lock.
    // Private locks would race on the same VkQueue (externally synchronized).
    let (queue_lock, register_compute_lock) = if ld.compute_queues_alias_graphics {
        (Arc::clone(&ld.queue_lock), false)
    } else {
        (Arc::new(Mutex::new(())), true)
    };
    if register_compute_lock {
        ld.register_active_compute_queue_lock(Arc::clone(&queue_lock));
    }

    let mut timeline_sem_type = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let timeline_sem_ci = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_sem_type);
    let timeline_semaphore = unsafe { ld.device.create_semaphore(&timeline_sem_ci, None) }
        .context("Failed to create per-context Vulkan timeline semaphore")?;

    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(compute_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe { ld.device.create_command_pool(&pool_info, None) }
        .context("Failed to create per-context command pool")?;

    let signal_queue = std::sync::Arc::new(crate::signal::SignalQueue::new());
    let fence_shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let timeline_semaphore_for_poll = timeline_semaphore;
    let device_for_poll = ld.device.clone();
    let signal_queue_poll = std::sync::Arc::clone(&signal_queue);
    let shutdown_poll = std::sync::Arc::clone(&fence_shutdown);
    let fence_thread = Some(crate::backend::signal_fence::spawn_fence_poller(
        crate::backend::signal_fence::FencePollerState {
            shutdown: shutdown_poll,
            signal_queue: signal_queue_poll,
            last_emitted_epoch: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            gpu_completed: std::sync::Arc::new(move || unsafe {
                device_for_poll
                    .get_semaphore_counter_value(timeline_semaphore_for_poll)
                    .unwrap_or(0)
            }),
        },
    ));

    let id = state.next_context_id;
    state.next_context_id = state.next_context_id.saturating_add(1);

    let instance = state.instance.clone();
    let frame_table = super::frame_table::init_context(state, &instance, device, &ld)?;

    state.contexts.write().unwrap().insert(
        id,
        Arc::new(Mutex::new(SubmissionContext {
            device,
            is_device_owner: false,
            queue,
            queue_family: compute_family,
            queue_index: Some(queue_index),
            queue_lock,
            timeline_semaphore,
            last_submitted_seq: 0,
            signal_queue,
            fence_shutdown,
            fence_thread,
            command_pool,
            free_cmd_buffers: Vec::new(),
            retained_compute_cbs: std::collections::HashMap::new(),
            timeline_cmd_buffers: std::collections::HashMap::new(),
            graphics_timeline_cmd_buffers: std::collections::HashMap::new(),
            staging_belt: super::staging::StagingBelt::new(super::staging::DEFAULT_STAGING_CHUNK_SIZE),
            texture_staging_pool: super::staging::TextureStagingPool::new(),
            deletion_queue: super::types::DeletionQueue::new(),
            frame_table,
            pending_gpu_profiles: Vec::new(),
        })),
    );
    Ok(id)
}

pub(super) struct VulkanContextDestroyWork {
    ctx: ContextHandle,
    sc: SubmissionContext,
    ld: super::types::SharedLogicalDevice,
    buffers: super::types::SharedBufferTable,
}

impl ContextDestroyHandle for VulkanContextDestroyWork {
    fn wait(&self) -> Result<()> {
        let _ = self.ld.submission_worker.flush();
        if self.sc.last_submitted_seq > 0 {
            let _ = self.ld.submission_worker.wait_submitted(self.sc.last_submitted_seq);
        }
        self.ld.submission_worker.check_error()?;
        if self.sc.last_submitted_seq > 0 {
            let wait = vk::SemaphoreWaitInfo::default()
                .semaphores(std::slice::from_ref(&self.sc.timeline_semaphore))
                .values(std::slice::from_ref(&self.sc.last_submitted_seq));
            unsafe { self.ld.device.wait_semaphores(&wait, u64::MAX) }.context("Vulkan context destroy wait")?;
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        finish_destroy(self);
        Ok(())
    }
}

pub(super) fn detach_for_destroy(state: &VulkanState, ctx: ContextHandle) -> Option<VulkanContextDestroyWork> {
    let sc_arc = state.contexts.write().unwrap().remove(&ctx)?;
    let sc_mutex = std::sync::Arc::try_unwrap(sc_arc)
        .unwrap_or_else(|_| panic!("context {ctx} Arc still has extra owners at destroy"));
    let sc = sc_mutex.into_inner().expect("context Mutex poisoned");
    let device = sc.device;
    let ld = state.devices.get(&device)?.clone();
    Some(VulkanContextDestroyWork {
        ctx,
        sc,
        ld,
        buffers: std::sync::Arc::clone(&state.buffers),
    })
}

fn finish_destroy(work: Box<VulkanContextDestroyWork>) {
    let VulkanContextDestroyWork { ctx, sc, ld, buffers } = *work;
    teardown_submission_context(ctx, sc, &ld, &buffers);
}

/// Destroy Vulkan objects owned by a submission context (command pool, timeline semaphore, etc.).
pub(super) fn teardown_submission_context(
    ctx: ContextHandle,
    mut sc: SubmissionContext,
    ld: &super::types::SharedLogicalDevice,
    buffers: &super::types::SharedBufferTable,
) {
    let device = sc.device;

    let completed = unsafe {
        ld.device
            .get_semaphore_counter_value(sc.timeline_semaphore)
            .unwrap_or(0)
    };
    ld.retired_floor.fetch_max(completed, Ordering::Relaxed);
    if sc.is_device_owner {
        ld.timeline_wait_targets
            .lock()
            .unwrap()
            .retain(|_, target| !matches!(target, TimelineWaitTarget::DeviceOwner));
        ld.timeline_retired.fetch_max(completed, Ordering::Relaxed);
    } else {
        ld.timeline_wait_targets
            .lock()
            .unwrap()
            .retain(|_, target| !matches!(target, TimelineWaitTarget::Context(c) if *c == ctx));
        ld.timeline_retired.fetch_max(completed, Ordering::Relaxed);
    }

    let fence_thread = sc.fence_thread.take();
    crate::backend::signal_fence::join_fence_poller(&sc.fence_shutdown, fence_thread);

    let ctx_batch = sc.deletion_queue.flush_all_drain();
    if !ctx_batch.is_empty() {
        let mut registry = ld.descriptors.lock().unwrap();
        for r in ctx_batch {
            super::types::destroy_pending_deletion(ld, &mut registry, r);
        }
    }

    {
        let mut registry = ld.descriptors.lock().unwrap();
        for (_, retained) in sc.retained_compute_cbs.drain() {
            registry.unpin_retained_slots(retained.used_slots);
        }
        let profile_completed = if sc.last_submitted_seq > 0 {
            sc.last_submitted_seq
        } else {
            completed
        };
        super::pending_submit::vulkan_drain_pending_gpu_profiles_up_to(ld, &mut sc, profile_completed);
    }

    unsafe {
        sc.staging_belt.destroy_all(ld);
        sc.texture_staging_pool.destroy_all(ld);
        let command_pool = sc.command_pool;
        for (_, cbs) in sc.timeline_cmd_buffers.drain() {
            for cb in cbs {
                ld.device.free_command_buffers(command_pool, &[cb]);
            }
        }
        for (_, cbs) in sc.graphics_timeline_cmd_buffers.drain() {
            for cb in cbs {
                ld.device.free_command_buffers(ld.command_pool, &[cb]);
            }
        }
        for cb in sc.free_cmd_buffers.drain(..) {
            ld.device.free_command_buffers(command_pool, &[cb]);
        }
        let mut rows_to_unpin = Vec::new();
        for (_, retained) in sc.retained_compute_cbs.drain() {
            rows_to_unpin.push(retained.frame_table_row);
            let pool = if retained.on_graphics_queue {
                ld.command_pool
            } else {
                command_pool
            };
            ld.device.free_command_buffers(pool, &[retained.command_buffer]);
        }
        for row in rows_to_unpin.into_iter().flatten() {
            super::frame_table::unpin_row(&sc.frame_table, row);
        }
        ld.device.destroy_command_pool(sc.command_pool, None);
        ld.device.destroy_semaphore(sc.timeline_semaphore, None);
        if !sc.is_device_owner {
            if let Some(idx) = sc.queue_index {
                ld.free_compute_queue_indices.lock().unwrap().push_back(idx);
            }
            // Shared graphics lock is never registered in active_context_queue_locks.
            if !ld.compute_queues_alias_graphics {
                ld.unregister_active_compute_queue_lock(&sc.queue_lock);
            }
        }
        if !sc.is_device_owner {
            super::frame_table::destroy_context_resources(buffers, ld, &sc.frame_table);
        }
    }
    let _ = device;
}

/// Tear down the synthetic device-owner context before `vkDestroyDevice`.
pub(super) fn destroy_device_owner(state: &VulkanState, ld: &super::types::SharedLogicalDevice, owner: ContextHandle) {
    let Some(sc_arc) = state.contexts.write().unwrap().remove(&owner) else {
        return;
    };
    let sc = std::sync::Arc::try_unwrap(sc_arc)
        .unwrap_or_else(|_| panic!("device-owner context {owner} still has extra Arc owners at device destroy"))
        .into_inner()
        .expect("device-owner context Mutex poisoned");
    teardown_submission_context(owner, sc, ld, &state.buffers);
}

/// Timeline semaphore for the synthetic device-owner (graphics-queue) context.
pub(super) fn owner_timeline_semaphore(state: &VulkanState, device: DeviceHandle) -> Option<vk::Semaphore> {
    let owner = state.device_owner_handles.get(&device)?;
    Some(
        state
            .contexts
            .read()
            .unwrap()
            .get(owner)?
            .lock()
            .unwrap()
            .timeline_semaphore,
    )
}

/// Block until the device-owner timeline has reached `seq` (graphics/present work).
pub(super) fn wait_until_owner_seq_at_least(state: &VulkanState, device: DeviceHandle, seq: u64) {
    if seq == 0 {
        return;
    }
    let Some(sem) = owner_timeline_semaphore(state, device) else {
        return;
    };
    let Some(ld) = state.devices.get(&device) else {
        return;
    };
    let wait = vk::SemaphoreWaitInfo::default()
        .semaphores(std::slice::from_ref(&sem))
        .values(std::slice::from_ref(&seq));
    let _ = unsafe { ld.device.wait_semaphores(&wait, u64::MAX) };
}

/// Block until the device-global submission sequence `seq` has been signalled on the GPU.
///
/// Looks up the native semaphore that was signalled for `seq` at submit time and waits
/// on that semaphore directly. With independent per-context compute queues, waiting on
/// an arbitrary context whose `last_submitted_seq >= seq` is unsound.
pub(super) fn wait_until_device_seq_at_least(state: &VulkanState, device: DeviceHandle, seq: u64) {
    if seq == 0 {
        return;
    }

    let Some(ld) = state.devices.get(&device) else {
        return;
    };

    if advance_timeline_retired(state, device) >= seq {
        return;
    }

    let target = lookup_timeline_wait_target(ld, seq);
    if let Some(target) = target {
        if let Some(sem) = resolve_timeline_wait_semaphore(state, device, target) {
            let wait = vk::SemaphoreWaitInfo::default()
                .semaphores(std::slice::from_ref(&sem))
                .values(std::slice::from_ref(&seq));
            let _ = unsafe { ld.device.wait_semaphores(&wait, u64::MAX) };
            let _ = advance_timeline_retired(state, device);
            return;
        }
    }

    // Seq not yet registered (transient race before submit records attribution) or owner
    // context torn down — poll until retired or a wait target appears.
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    const MAX_SPIN: u32 = 10_000;
    let deadline = std::time::Instant::now() + TIMEOUT;
    let mut spins = 0u32;
    while advance_timeline_retired(state, device) < seq {
        if std::time::Instant::now() >= deadline {
            tracing::warn!("wait_until_device_seq_at_least timed out waiting for seq {seq} on device {device}");
            return;
        }
        if let Some(target) = lookup_timeline_wait_target(ld, seq) {
            if let Some(sem) = resolve_timeline_wait_semaphore(state, device, target) {
                let wait = vk::SemaphoreWaitInfo::default()
                    .semaphores(std::slice::from_ref(&sem))
                    .values(std::slice::from_ref(&seq));
                let _ = unsafe { ld.device.wait_semaphores(&wait, u64::MAX) };
                let _ = advance_timeline_retired(state, device);
                return;
            }
        }
        if spins < MAX_SPIN {
            spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    }
}

fn lookup_timeline_wait_target(ld: &super::types::LogicalDevice, seq: u64) -> Option<TimelineWaitTarget> {
    ld.timeline_wait_targets.lock().unwrap().get(&seq).copied()
}

pub(super) fn context_device(state: &VulkanState, ctx: ContextHandle) -> DeviceHandle {
    state
        .contexts
        .read()
        .unwrap()
        .get(&ctx)
        .expect("invalid context handle")
        .lock()
        .unwrap()
        .device
}

fn contexts_on_device(state: &VulkanState, device: DeviceHandle) -> Vec<ContextHandle> {
    state
        .contexts
        .read()
        .unwrap()
        .iter()
        .filter_map(|(&h, sc_arc)| {
            let sc = sc_arc.lock().unwrap();
            (sc.device == device && !sc.is_device_owner).then_some(h)
        })
        .collect()
}

/// When exactly one live context exists on `device`, return it for destroy attribution.
pub(super) fn sole_context_on_device(state: &VulkanState, device: DeviceHandle) -> Option<ContextHandle> {
    let mut on_device = contexts_on_device(state, device);
    if on_device.len() == 1 {
        on_device.pop()
    } else {
        None
    }
}

/// Context to receive a user-initiated buffer destroy when attribution is unambiguous.
pub(super) fn destroy_attribution_context(state: &VulkanState, device: DeviceHandle) -> Option<ContextHandle> {
    sole_context_on_device(state, device)
}

/// Deferred-deletion requirement for a destroy attributed to a single context.
pub(super) fn reclamation_barrier_for_context(state: &VulkanState, ctx: ContextHandle) -> u64 {
    let Some(sc_arc) = state.contexts.read().unwrap().get(&ctx).cloned() else {
        return 0;
    };
    let sc = sc_arc.lock().expect("context Mutex poisoned");
    sc.last_submitted_seq
}

/// Per-context requirement snapshot when destroy could not be attributed to one context.
pub(super) fn reclamation_requirements_all_contexts(
    state: &VulkanState,
    device: DeviceHandle,
) -> Vec<(ContextHandle, u64)> {
    let handles = contexts_on_device(state, device);
    let contexts = state.contexts.read().unwrap();
    handles
        .into_iter()
        .filter_map(|h| {
            let sc_arc = contexts.get(&h)?;
            let sc = sc_arc.lock().expect("context Mutex poisoned");
            Some((h, sc.last_submitted_seq))
        })
        .collect()
}

/// Full requirement snapshot for a buffer/texture destroy.
pub(super) fn reclamation_requirements(
    state: &VulkanState,
    device: DeviceHandle,
    ctx_h: Option<ContextHandle>,
) -> Vec<(ContextHandle, u64)> {
    match ctx_h {
        Some(ctx) => vec![(ctx, reclamation_barrier_for_context(state, ctx))],
        None => reclamation_requirements_all_contexts(state, device),
    }
}

/// Timeline barrier for deferred buffer destruction on `device` (legacy scalar helper).
#[allow(dead_code)]
pub(super) fn reclamation_barrier(state: &VulkanState, device: DeviceHandle, ctx: Option<ContextHandle>) -> u64 {
    if let Some(ctx) = ctx {
        if let Some(sc_arc) = state.contexts.read().unwrap().get(&ctx) {
            return sc_arc.lock().unwrap().last_submitted_seq;
        }
    }
    state
        .devices
        .get(&device)
        .map(|d| d.timeline_next.load(Ordering::Relaxed).saturating_sub(1))
        .unwrap_or(0)
}
