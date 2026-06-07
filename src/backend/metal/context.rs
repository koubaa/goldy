//! Per-context submission stream lifecycle (Metal).

use super::staging::{StagingBelt, TextureStagingPool, DEFAULT_STAGING_CHUNK_SIZE};
use super::types::{MetalState, MetalSubmissionContext, TimelineWaiter};
use super::{ContextHandle, DeviceHandle};
use ::metal as mtl;
use anyhow::{Context as _, Result};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Latest device-global seq retired on `device` (max over live context shared events, floored).
///
/// Sound only because all contexts commit to a single `LogicalDevice::command_queue`
/// (FIFO order guaranteed by Metal).  If contexts ever get their own queues, the
/// `max`-over-contexts approach breaks: a slot freed once context A retires could
/// still be live in context B.  See `LogicalDevice::command_queue` for details.
pub(super) fn device_retired(state: &MetalState, device: DeviceHandle) -> u64 {
    let floor = state
        .devices
        .get(&device)
        .map(|d| d.retired_floor.load(Ordering::Relaxed))
        .unwrap_or(0);
    let max_ctx = state
        .contexts
        .values()
        .filter_map(|sc_arc| {
            let sc = sc_arc.lock().unwrap();
            if sc.device == device {
                Some(sc.timeline_event.as_ref().signaled_value())
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0);
    floor.max(max_ctx)
}

/// Returns the `ContextHandle` whose reclamation context is installed on the current thread
/// for a context on `device`, if any.
///
/// Used to route resource deletions into the owning context's deletion queue (rather than
/// the device-wide queue) so they reclaim on the context's own clock.  See issue #190.
pub(super) fn context_handle_for_thread(state: &MetalState, device: DeviceHandle) -> Option<super::ContextHandle> {
    let thread = std::thread::current().id();
    state.contexts.iter().find_map(|(h, sc_arc)| {
        let sc = sc_arc.lock().unwrap();
        if sc.device == device {
            if let Some((t, _)) = sc.reclamation_context {
                if t == thread {
                    return Some(*h);
                }
            }
        }
        None
    })
}

pub(super) fn create(state: &mut MetalState, device: DeviceHandle) -> Result<ContextHandle> {
    let ld = state.devices.get(&device).context("Invalid device handle")?;

    let timeline_event = ld.device.new_shared_event();
    let signal_queue = Arc::new(crate::signal::SignalQueue::new());
    let timeline_waiter = TimelineWaiter::new_with_signals(std::sync::Arc::clone(&signal_queue));

    let id = state.next_context_id;
    state.next_context_id = state.next_context_id.saturating_add(1);
    state.contexts.insert(
        id,
        Arc::new(Mutex::new(MetalSubmissionContext {
            device,
            timeline_event,
            timeline_waiter,
            signal_queue,
            last_submitted_seq: 0,
            in_flight_command_buffers: VecDeque::new(),
            reclamation_context: None,
            pending_swapchain_returns: Arc::new(Mutex::new(Vec::new())),
            last_committed_timeline: None,
            staging_belt: StagingBelt::new(DEFAULT_STAGING_CHUNK_SIZE),
            texture_staging_pool: TextureStagingPool::new(),
            deletion_queue: super::types::DeletionQueue::new(),
        })),
    );
    Ok(id)
}

pub(super) fn destroy(state: &mut MetalState, ctx: ContextHandle) {
    let Some(sc_arc) = state.contexts.remove(&ctx) else {
        return;
    };
    let mut sc = sc_arc.lock().unwrap();
    let device = sc.device;
    let completed = sc.timeline_event.as_ref().signaled_value();
    if let Some(ld) = state.devices.get(&device) {
        ld.retired_floor.fetch_max(completed, Ordering::Relaxed);
    }

    sc.staging_belt.destroy_all();
    sc.texture_staging_pool.destroy_all();
    for (_, cb) in sc.in_flight_command_buffers.drain(..) {
        cb.wait_until_completed();
    }
    // All CBs have completed; flush any resources still parked in the per-context
    // deletion queue.  At this point every timeline value is retired so every
    // barrier has been passed.
    sc.deletion_queue.flush_all();
}

pub(super) fn context_device(state: &MetalState, ctx: ContextHandle) -> DeviceHandle {
    state
        .contexts
        .get(&ctx)
        .expect("invalid context handle")
        .lock()
        .unwrap()
        .device
}

/// Block until the device-global submission sequence `seq` has retired.
///
/// Returns `false` if `timeout` elapses before retirement completes.
pub(super) fn wait_until_device_seq_at_least(
    state: &MetalState,
    device: DeviceHandle,
    seq: u64,
    timeout: std::time::Duration,
) -> bool {
    if seq == 0 {
        return true;
    }
    let start = std::time::Instant::now();
    while device_retired(state, device) < seq {
        if start.elapsed() >= timeout {
            return false;
        }
        let remaining = timeout.saturating_sub(start.elapsed());
        // Find any context on this device that hasn't reached seq yet and wait briefly.
        let found_waiter = state.contexts.values().find_map(|sc_arc| {
            let sc = sc_arc.lock().unwrap();
            if sc.device == device && sc.timeline_event.as_ref().signaled_value() < seq {
                Some(sc.timeline_waiter.clone())
            } else {
                None
            }
        });
        if let Some(waiter) = found_waiter {
            let chunk = remaining.min(std::time::Duration::from_millis(1));
            let _ = waiter.wait_until(seq, chunk);
        }
    }
    true
}

/// Oldest in-flight command buffer across all contexts on `device` (by timeline value).
pub(super) fn oldest_in_flight_cb(state: &MetalState, device: DeviceHandle) -> Option<mtl::CommandBuffer> {
    state
        .contexts
        .values()
        .filter_map(|sc_arc| {
            let sc = sc_arc.lock().unwrap();
            if sc.device != device {
                return None;
            }
            sc.in_flight_command_buffers
                .front()
                .map(|(tv, cb)| (*tv, cb.to_owned()))
        })
        .min_by_key(|(tv, _)| *tv)
        .map(|(_, cb)| cb)
}

/// Reclamation epoch installed on the current thread for any context on `device`.
pub(super) fn reclamation_barrier(state: &MetalState, device: DeviceHandle, gpu_idle: bool) -> u64 {
    if gpu_idle {
        return 0;
    }
    let thread = std::thread::current().id();
    for sc_arc in state.contexts.values() {
        let sc = sc_arc.lock().unwrap();
        if sc.device == device {
            if let Some((t, epoch)) = sc.reclamation_context {
                if t == thread {
                    return epoch;
                }
            }
        }
    }
    state
        .devices
        .get(&device)
        .map(|d| d.timeline_scheduled_max.load(Ordering::Relaxed))
        .unwrap_or(0)
}
