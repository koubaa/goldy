//! Per-context submission stream lifecycle (Metal).

use super::types::{MetalState, MetalSubmissionContext, TimelineWaiter};
use super::{ContextHandle, DeviceHandle};
use anyhow::{Context as _, Result};
use ::metal as mtl;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Latest device-global seq retired on `device` (max over live context shared events, floored).
pub(super) fn device_retired(state: &MetalState, device: DeviceHandle) -> u64 {
    let floor = state
        .devices
        .get(&device)
        .map(|d| d.retired_floor)
        .unwrap_or(0);
    let max_ctx = state
        .contexts
        .values()
        .filter(|c| c.device == device)
        .map(|c| c.timeline_event.as_ref().signaled_value())
        .max()
        .unwrap_or(0);
    floor.max(max_ctx)
}

pub(super) fn create(state: &mut MetalState, device: DeviceHandle) -> Result<ContextHandle> {
    let ld = state
        .devices
        .get(&device)
        .context("Invalid device handle")?;

    let timeline_event = ld.device.new_shared_event();
    let signal_queue = Arc::new(crate::signal::SignalQueue::new());
    let timeline_waiter =
        TimelineWaiter::new_with_signals(std::sync::Arc::clone(&signal_queue));

    let id = state.next_context_id;
    state.next_context_id = state.next_context_id.saturating_add(1);
    state.contexts.insert(
        id,
        MetalSubmissionContext {
            device,
            timeline_event,
            timeline_waiter,
            signal_queue,
            last_submitted_seq: 0,
            in_flight_command_buffers: VecDeque::new(),
            reclamation_context: None,
            pending_swapchain_returns: Arc::new(Mutex::new(Vec::new())),
            last_committed_timeline: None,
        },
    );
    Ok(id)
}

pub(super) fn destroy(state: &mut MetalState, ctx: ContextHandle) {
    let Some(mut sc) = state.contexts.remove(&ctx) else {
        return;
    };
    let device = sc.device;
    let completed = sc.timeline_event.as_ref().signaled_value();
    if let Some(ld) = state.devices.get_mut(&device) {
        ld.retired_floor = ld.retired_floor.max(completed);
    }

    for (_, cb) in sc.in_flight_command_buffers.drain(..) {
        cb.wait_until_completed();
    }
}

pub(super) fn context_device(state: &MetalState, ctx: ContextHandle) -> DeviceHandle {
    state
        .contexts
        .get(&ctx)
        .expect("invalid context handle")
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
        for sc in state.contexts.values().filter(|c| c.device == device) {
            if sc.timeline_event.as_ref().signaled_value() < seq {
                let chunk = remaining.min(std::time::Duration::from_millis(1));
                let _ = sc.timeline_waiter.wait_until(seq, chunk);
                break;
            }
        }
    }
    true
}

/// Oldest in-flight command buffer across all contexts on `device` (by timeline value).
pub(super) fn oldest_in_flight_cb(
    state: &MetalState,
    device: DeviceHandle,
) -> Option<mtl::CommandBuffer> {
    state
        .contexts
        .values()
        .filter(|c| c.device == device)
        .filter_map(|c| c.in_flight_command_buffers.front())
        .min_by_key(|(tv, _)| *tv)
        .map(|(_, cb)| cb.to_owned())
}

/// Reclamation epoch installed on the current thread for any context on `device`.
pub(super) fn reclamation_barrier(state: &MetalState, device: DeviceHandle, gpu_idle: bool) -> u64 {
    if gpu_idle {
        return 0;
    }
    let thread = std::thread::current().id();
    for sc in state.contexts.values().filter(|c| c.device == device) {
        if let Some((t, epoch)) = sc.reclamation_context {
            if t == thread {
                return epoch;
            }
        }
    }
    state
        .devices
        .get(&device)
        .map(|d| d.timeline_scheduled_max)
        .unwrap_or(0)
}
