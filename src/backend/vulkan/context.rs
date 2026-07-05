//! Per-context submission stream lifecycle (Vulkan).

use super::types::{SubmissionContext, VulkanState};
use super::{ContextHandle, DeviceHandle};
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
    let ld = state.devices.get(&device).context("Invalid device handle")?;

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
        })),
    );
    Ok(id)
}

pub(super) fn destroy(state: &mut VulkanState, ctx: ContextHandle) {
    let Some(sc_arc) = state.contexts.write().unwrap().remove(&ctx) else {
        return;
    };
    let mut sc = sc_arc.lock().unwrap();
    let device = sc.device;
    let completed = unsafe {
        state
            .devices
            .get(&device)
            .and_then(|ld| ld.device.get_semaphore_counter_value(sc.timeline_semaphore).ok())
            .unwrap_or(0)
    };
    if let Some(ld) = state.devices.get(&device) {
        ld.retired_floor.fetch_max(completed, Ordering::Relaxed);
    }

    let last_seq = sc.last_submitted_seq;
    let timeline_semaphore = sc.timeline_semaphore;

    let fence_thread = sc.fence_thread.take();
    crate::backend::signal_fence::join_fence_poller(&sc.fence_shutdown, fence_thread);

    let ctx_batch = sc.deletion_queue.flush_all_drain();

    {
        let Some(ld) = state.devices.get(&device) else {
            return;
        };
        if last_seq > 0 {
            let wait = vk::SemaphoreWaitInfo::default()
                .semaphores(std::slice::from_ref(&timeline_semaphore))
                .values(std::slice::from_ref(&last_seq));
            let _ = unsafe { ld.device.wait_semaphores(&wait, u64::MAX) };
        }
        if !ctx_batch.is_empty() {
            let descriptors_arc = std::sync::Arc::clone(&ld.descriptors);
            let mut registry = descriptors_arc.lock().unwrap();
            for r in ctx_batch {
                super::types::destroy_pending_deletion(ld, &mut registry, r);
            }
        }
    }

    let Some(ld) = state.devices.get(&device) else {
        return;
    };
    unsafe {
        sc.staging_belt.destroy_all(ld);
        sc.texture_staging_pool.destroy_all(ld);
        let command_pool = sc.command_pool;
        for (_, cbs) in sc.timeline_cmd_buffers.drain() {
            for cb in cbs {
                ld.device.free_command_buffers(command_pool, &[cb]);
            }
        }
        for cb in sc.free_cmd_buffers.drain(..) {
            ld.device.free_command_buffers(command_pool, &[cb]);
        }
        for (_, retained) in sc.retained_compute_cbs.drain() {
            if let Some(row) = retained.frame_table_row {
                if let Some(ft) = state.frame_tables.read().unwrap().get(&device) {
                    super::frame_table::unpin_row(ft, row);
                }
            }
            ld.device.free_command_buffers(command_pool, &[retained.command_buffer]);
        }
        ld.device.destroy_command_pool(sc.command_pool, None);
        ld.device.destroy_semaphore(sc.timeline_semaphore, None);
    }
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
