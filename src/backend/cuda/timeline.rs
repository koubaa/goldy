//! CUDA timeline events, retirement, and blocking waits.

use super::ContextHandle;
use crate::backend::TimelineBlockingWait;
use anyhow::{Context as _, Result};
use cudarc::driver::CudaEvent;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One pre-allocated completion event for a device timeline value.
pub(super) struct LedgerEntry {
    pub context: ContextHandle,
    pub event: Arc<CudaEvent>,
    /// Set after the submission worker records the event onto its stream.
    pub recorded: bool,
}

/// Shared device-wide ledger of timeline values → completion events.
pub(super) type EventLedger = Arc<Mutex<BTreeMap<u64, LedgerEntry>>>;

/// Advance per-context high-water and device-contiguous retirement.
///
/// Context progress is the max completed timeline value submitted on that context
/// (matching DX12/Vulkan shared device counters with per-context fences).
pub(super) fn poll_retire_events(
    ledger: &EventLedger,
    context_completed: &AtomicU64,
    context: ContextHandle,
    device_retired: &AtomicU64,
    signal_queue: &crate::signal::SignalQueue,
    last_emitted: &AtomicU64,
) {
    let mut ctx_max = 0u64;
    {
        let guard = ledger.lock().unwrap();
        for (tv, entry) in guard.iter() {
            if entry.context == context && entry.recorded && entry.event.is_complete() {
                ctx_max = ctx_max.max(*tv);
            }
        }
    }

    let prev = context_completed.load(Ordering::Acquire);
    if ctx_max > prev {
        context_completed.store(ctx_max, Ordering::Release);
    }

    let mut last = last_emitted.load(Ordering::Acquire);
    let completed = context_completed.load(Ordering::Acquire);
    while last < completed {
        last += 1;
        signal_queue.push_boundary_crossed(last);
        last_emitted.store(last, Ordering::Release);
    }

    advance_device_retired(ledger, device_retired);
}

/// Device retired value is the longest contiguous prefix of recorded+complete events.
pub(super) fn advance_device_retired(ledger: &EventLedger, device_retired: &AtomicU64) {
    let guard = ledger.lock().unwrap();
    let mut retired = device_retired.load(Ordering::Acquire);
    loop {
        let next = retired + 1;
        match guard.get(&next) {
            Some(entry) if entry.recorded && entry.event.is_complete() => {
                retired = next;
                device_retired.store(retired, Ordering::Release);
            }
            _ => break,
        }
    }
}

pub(super) fn lookup_event(ledger: &EventLedger, context: ContextHandle, value: u64) -> Option<Arc<CudaEvent>> {
    let guard = ledger.lock().unwrap();
    guard.get(&value).and_then(|entry| {
        if entry.context == context {
            Some(Arc::clone(&entry.event))
        } else {
            None
        }
    })
}

pub(super) fn mark_recorded(ledger: &EventLedger, value: u64) {
    if let Some(entry) = ledger.lock().unwrap().get_mut(&value) {
        entry.recorded = true;
    }
}

pub(super) struct CudaTimelineBlockingWait {
    pub event: Arc<CudaEvent>,
}

impl TimelineBlockingWait for CudaTimelineBlockingWait {
    fn block(self: Box<Self>) -> Result<()> {
        self.event.synchronize().context("CUDA: event synchronize failed")
    }

    fn block_timeout(self: Box<Self>, timeout_ms: u32) -> Result<bool> {
        if timeout_ms == 0 {
            return Ok(self.event.is_complete());
        }
        let deadline = Instant::now() + Duration::from_millis(u64::from(timeout_ms));
        while Instant::now() < deadline {
            if self.event.is_complete() {
                return Ok(true);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(self.event.is_complete())
    }
}

/// Host-side wait on a CUDA event (used by submission-worker sidecar).
pub(super) fn host_wait_event(event: &CudaEvent) -> Result<()> {
    event.synchronize().context("CUDA: host wait on event failed")
}
