//! Per-context submission stream lifecycle (Metal).

use super::staging::{StagingBelt, TextureStagingPool, DEFAULT_STAGING_CHUNK_SIZE};
use super::types::{MetalState, MetalSubmissionContext, TimelineWaiter};
use super::{ContextHandle, DeviceHandle};
use crate::backend::ContextDestroyHandle;
use ::metal as mtl;
use anyhow::{Context as _, Result};
use mtl::MTLCommandBufferStatus;
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Best-effort retirement horizon for one context without blocking.
///
/// `MTLSharedEvent::signaled_value` can lag completion handlers and individual
/// command-buffer completion; task-graph re-record safety uses this value.
pub(super) fn context_gpu_progress(sc: &MetalSubmissionContext) -> u64 {
    let mut progress = sc.timeline_event.as_ref().signaled_value();
    progress = progress.max(sc.timeline_waiter.completed_value());
    for (tv, cb) in &sc.in_flight_command_buffers {
        if cb.status() == MTLCommandBufferStatus::Completed {
            progress = progress.max(*tv);
        } else {
            break;
        }
    }
    progress
}

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
                Some(context_gpu_progress(&sc))
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
            retained_graphs: std::collections::HashMap::new(),
        })),
    );
    Ok(id)
}

pub(super) fn destroy(state: &mut MetalState, ctx: ContextHandle) {
    if let Some(work) = detach_for_destroy(state, ctx) {
        crate::backend::run_context_destroy(Box::new(work));
    }
}

pub(super) struct MetalContextDestroyWork {
    sc: super::types::MetalSubmissionContext,
    ld: super::types::SharedLogicalDevice,
}

impl ContextDestroyHandle for MetalContextDestroyWork {
    fn wait(&self) -> Result<()> {
        for (_, cb) in self.sc.in_flight_command_buffers.iter() {
            cb.wait_until_completed();
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        finish_destroy(self);
        Ok(())
    }
}

pub(super) fn detach_for_destroy(state: &mut MetalState, ctx: ContextHandle) -> Option<MetalContextDestroyWork> {
    let sc_arc = state.contexts.remove(&ctx)?;
    let sc_mutex = std::sync::Arc::try_unwrap(sc_arc)
        .unwrap_or_else(|_| panic!("context {ctx} Arc still has extra owners at destroy"));
    let sc = sc_mutex.into_inner().expect("context Mutex poisoned");
    let device = sc.device;
    let ld = state.devices.get(&device)?.clone();
    Some(MetalContextDestroyWork { sc, ld })
}

fn finish_destroy(work: Box<MetalContextDestroyWork>) {
    let MetalContextDestroyWork { mut sc, ld } = *work;
    let device = sc.device;

    {
        let mut registry = ld.descriptors.lock().unwrap();
        for (_, graph) in sc.retained_graphs.drain() {
            registry.unpin_retained_slots(graph.used_slots);
        }
    }

    sc.retained_graphs.clear();

    sc.staging_belt.destroy_all();
    sc.texture_staging_pool.destroy_all();
    let last_seq = sc.last_submitted_seq;
    super::drain_completed_cbs(&mut sc);
    sc.in_flight_command_buffers.clear();
    let signaled_after = sc.timeline_event.as_ref().signaled_value();
    let retired_horizon = signaled_after.max(last_seq);
    ld.retired_floor.fetch_max(retired_horizon, Ordering::Relaxed);
    sc.deletion_queue.flush_all();
    let _ = device;
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
        // Drive retirement through Metal: completion-handler condvar can report
        // completion before `MTLSharedEvent::signaled_value` advances.
        if let Some(cb) = oldest_in_flight_cb(state, device) {
            cb.wait_until_completed();
            for sc_arc in state.contexts.values() {
                let mut sc = sc_arc.lock().unwrap();
                if sc.device == device {
                    super::drain_completed_cbs(&mut sc);
                }
            }
            continue;
        }
        // Wait on the completion-handler condvar; by the time it fires the shared
        // event has already been signalled for that submission.
        let found_waiter = state.contexts.values().find_map(|sc_arc| {
            let sc = sc_arc.lock().unwrap();
            if sc.device == device && sc.timeline_event.as_ref().signaled_value() < seq {
                Some(sc.timeline_waiter.clone())
            } else {
                None
            }
        });
        if let Some(waiter) = found_waiter {
            let _ = waiter.wait_until(seq, remaining);
            continue;
        }
        // No in-flight CBs and no live context below `seq`, yet `device_retired` lags.
        // Context destroy should have floored `retired_floor` before removal.
        debug_assert!(
            device_retired(state, device) >= seq,
            "device_retired ({}) lagged seq ({}) with nothing left to wait on",
            device_retired(state, device),
            seq
        );
        return false;
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

/// Snapshot completed timeline values for every context on `device`.
///
/// Call before locking descriptors so slot reclaim can evaluate last-use epochs
/// without holding both locks.
pub(super) fn snapshot_context_completed_values(
    state: &MetalState,
    device: DeviceHandle,
) -> std::collections::HashMap<ContextHandle, u64> {
    state
        .contexts
        .iter()
        .filter_map(|(&id, sc_arc)| {
            let sc = sc_arc.lock().unwrap();
            if sc.device != device {
                return None;
            }
            Some((id, context_gpu_progress(&sc)))
        })
        .collect()
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
