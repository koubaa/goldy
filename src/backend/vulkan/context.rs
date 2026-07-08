//! Per-context submission stream lifecycle (Vulkan).

use super::types::{SubmissionContext, VulkanState};
use super::{ContextHandle, DeviceHandle};
use crate::backend::ContextDestroyHandle;
use anyhow::{Context as _, Result};
use ash::vk;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Latest device-global seq retired on `device` (max over live context semaphores, floored).
pub(super) fn device_retired(state: &VulkanState, device: DeviceHandle) -> u64 {
    let floor = state
        .devices
        .get(&device)
        .map(|d| d.retired_floor.load(Ordering::Relaxed))
        .unwrap_or(0);
    let Some(ld) = state.devices.get(&device) else {
        return floor;
    };
    let contexts = state.contexts.read().unwrap();
    let max_ctx = contexts
        .values()
        .filter_map(|sc_arc| {
            let sc = sc_arc.lock().unwrap();
            if sc.device != device {
                return None;
            }
            Some(unsafe {
                ld.device
                    .get_semaphore_counter_value(sc.timeline_semaphore)
                    .unwrap_or(0)
            })
        })
        .max()
        .unwrap_or(0);
    floor.max(max_ctx)
}

pub(super) fn create(state: &mut VulkanState, device: DeviceHandle) -> Result<ContextHandle> {
    let ld = state.devices.get(&device).context("Invalid device handle")?.clone();

    let mut timeline_sem_type = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let timeline_sem_ci = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_sem_type);
    let timeline_semaphore = unsafe { ld.device.create_semaphore(&timeline_sem_ci, None) }
        .context("Failed to create per-context Vulkan timeline semaphore")?;

    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ld.queue_family)
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
            timeline_semaphore,
            last_submitted_seq: 0,
            signal_queue,
            fence_shutdown,
            fence_thread,
            command_pool,
            free_cmd_buffers: Vec::new(),
            retained_compute_cbs: std::collections::HashMap::new(),
            timeline_cmd_buffers: std::collections::HashMap::new(),
            staging_belt: super::staging::StagingBelt::new(super::staging::DEFAULT_STAGING_CHUNK_SIZE),
            texture_staging_pool: super::staging::TextureStagingPool::new(),
            deletion_queue: super::types::DeletionQueue::new(),
            frame_table,
        })),
    );
    Ok(id)
}

pub(super) struct VulkanContextDestroyWork {
    sc: SubmissionContext,
    ld: super::types::SharedLogicalDevice,
    buffers: super::types::SharedBufferTable,
}

impl ContextDestroyHandle for VulkanContextDestroyWork {
    fn wait(&self) -> Result<()> {
        if self.sc.last_submitted_seq > 0 {
            let wait = vk::SemaphoreWaitInfo::default()
                .semaphores(std::slice::from_ref(&self.sc.timeline_semaphore))
                .values(std::slice::from_ref(&self.sc.last_submitted_seq));
            unsafe { self.ld.device.wait_semaphores(&wait, u64::MAX) }
                .context("Vulkan context destroy wait")?;
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
        sc,
        ld,
        buffers: std::sync::Arc::clone(&state.buffers),
    })
}

fn finish_destroy(work: Box<VulkanContextDestroyWork>) {
    let VulkanContextDestroyWork { mut sc, ld, buffers } = *work;
    let device = sc.device;

    let completed = unsafe {
        ld.device
            .get_semaphore_counter_value(sc.timeline_semaphore)
            .unwrap_or(0)
    };
    ld.retired_floor.fetch_max(completed, Ordering::Relaxed);

    let fence_thread = sc.fence_thread.take();
    crate::backend::signal_fence::join_fence_poller(&sc.fence_shutdown, fence_thread);

    let ctx_batch = sc.deletion_queue.flush_all_drain();
    if !ctx_batch.is_empty() {
        let mut registry = ld.descriptors.lock().unwrap();
        for r in ctx_batch {
            super::types::destroy_pending_deletion(&ld, &mut registry, r);
        }
    }

    unsafe {
        sc.staging_belt.destroy_all(&ld);
        sc.texture_staging_pool.destroy_all(&ld);
        let command_pool = sc.command_pool;
        for (_, cbs) in sc.timeline_cmd_buffers.drain() {
            for cb in cbs {
                ld.device.free_command_buffers(command_pool, &[cb]);
            }
        }
        for cb in sc.free_cmd_buffers.drain(..) {
            ld.device.free_command_buffers(command_pool, &[cb]);
        }
        let mut rows_to_unpin = Vec::new();
        for (_, retained) in sc.retained_compute_cbs.drain() {
            rows_to_unpin.push(retained.frame_table_row);
            ld.device.free_command_buffers(command_pool, &[retained.command_buffer]);
        }
        for row in rows_to_unpin.into_iter().flatten() {
            super::frame_table::unpin_row(&sc.frame_table, row);
        }
        ld.device.destroy_command_pool(sc.command_pool, None);
        ld.device.destroy_semaphore(sc.timeline_semaphore, None);
        super::frame_table::destroy_context_resources(&buffers, &ld, &sc.frame_table);
    }
    let _ = device;
}

/// Block until the device-global submission sequence `seq` has been signalled on the GPU.
///
/// Finds the context on `device` that last submitted at or beyond `seq` and issues a
/// `vkWaitSemaphores` on its timeline semaphore. This replaces the previous 1ms
/// sleep-poll loop, eliminating the latency floor that capped Vulkan FPS at ~1260.
pub(super) fn wait_until_device_seq_at_least(state: &VulkanState, device: DeviceHandle, seq: u64) {
    if seq == 0 {
        return;
    }

    let Some(ld) = state.devices.get(&device) else {
        return;
    };

    // Find a context on this device whose last_submitted_seq covers `seq`.
    // In the unified-context model there is typically exactly one such context.
    let sem = {
        let contexts = state.contexts.read().unwrap();
        contexts.values().find_map(|sc_arc| {
            let sc = sc_arc.lock().unwrap();
            if sc.device == device && sc.last_submitted_seq >= seq {
                Some(sc.timeline_semaphore)
            } else {
                None
            }
        })
    };

    if let Some(sem) = sem {
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&sem))
            .values(std::slice::from_ref(&seq));
        let _ = unsafe { ld.device.wait_semaphores(&wait, u64::MAX) };
    } else {
        // Seq not yet submitted on any known context — rare transient race. Poll with
        // bounded spin then sleep; re-check for a covering context before each sleep.
        const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
        const MAX_SPIN: u32 = 10_000;
        let deadline = std::time::Instant::now() + TIMEOUT;
        let mut spins = 0u32;
        while device_retired(state, device) < seq {
            if std::time::Instant::now() >= deadline {
                tracing::warn!("wait_until_device_seq_at_least timed out waiting for seq {seq} on device {device}");
                return;
            }
            if let Some(sem) = state.contexts.read().unwrap().values().find_map(|sc_arc| {
                let sc = sc_arc.lock().unwrap();
                if sc.device == device && sc.last_submitted_seq >= seq {
                    Some(sc.timeline_semaphore)
                } else {
                    None
                }
            }) {
                let wait = vk::SemaphoreWaitInfo::default()
                    .semaphores(std::slice::from_ref(&sem))
                    .values(std::slice::from_ref(&seq));
                let _ = unsafe { ld.device.wait_semaphores(&wait, u64::MAX) };
                return;
            }
            if spins < MAX_SPIN {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }
    }
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
            (sc.device == device).then_some(h)
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
