//! CUDA timeline events, retirement, and blocking waits.

use super::ContextHandle;
use crate::backend::TimelineBlockingWait;
use anyhow::{bail, Context as _, Result};
use cudarc::driver::CudaEvent;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
use super::dx12_companion::Dx12Companion;

/// How a ledger timeline value becomes observable as complete.
pub(super) enum LedgerCompletion {
    CudaEvent(Arc<CudaEvent>),
    /// DX12 companion fence signaled on the presentation DIRECT queue.
    ///
    /// `value` may be 0 until [`bind_dx12_fence_value`] runs at Signal time.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    Dx12Fence {
        companion: Arc<Dx12Companion>,
        value: u64,
    },
}

/// Snapshot used so fence/event queries do not run under the ledger mutex.
enum CompletionSnap {
    CudaEvent(Arc<CudaEvent>),
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    Dx12Fence {
        companion: Arc<Dx12Companion>,
        value: u64,
    },
}

impl CompletionSnap {
    fn from_completion(completion: &LedgerCompletion) -> Option<Self> {
        match completion {
            LedgerCompletion::CudaEvent(event) => Some(Self::CudaEvent(Arc::clone(event))),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            LedgerCompletion::Dx12Fence { companion, value } => {
                (*value > 0).then(|| Self::Dx12Fence {
                    companion: Arc::clone(companion),
                    value: *value,
                })
            }
        }
    }

    fn is_complete(&self) -> bool {
        match self {
            Self::CudaEvent(event) => event.is_complete(),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            Self::Dx12Fence { companion, value } => (unsafe { companion.fence.GetCompletedValue() }) >= *value,
        }
    }
}

/// One pre-allocated completion marker for a device timeline value.
pub(super) struct LedgerEntry {
    pub context: ContextHandle,
    pub completion: LedgerCompletion,
    /// Set after the submission worker records a CUDA event, or after a DX12 fence
    /// signal is submitted for fence-based entries.
    pub recorded: bool,
}

impl LedgerEntry {
    pub(super) fn is_complete(&self) -> bool {
        CompletionSnap::from_completion(&self.completion)
            .map(|snap| snap.is_complete())
            .unwrap_or(false)
    }
}

/// Shared device-wide ledger of timeline values → completion markers.
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
    // Snapshot under the lock; query completion outside so GetCompletedValue /
    // cudaEventQuery cannot stall insert/lookup on the present/submit hot path.
    let snaps: Vec<(u64, CompletionSnap)> = {
        let guard = ledger.lock().unwrap();
        guard
            .iter()
            .filter_map(|(tv, entry)| {
                if entry.context == context && entry.recorded {
                    CompletionSnap::from_completion(&entry.completion).map(|snap| (*tv, snap))
                } else {
                    None
                }
            })
            .collect()
    };

    let mut ctx_max = 0u64;
    for (tv, snap) in snaps {
        if snap.is_complete() {
            ctx_max = ctx_max.max(tv);
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
    loop {
        let next = device_retired.load(Ordering::Acquire) + 1;
        let snap = {
            let guard = ledger.lock().unwrap();
            match guard.get(&next) {
                Some(entry) if entry.recorded => CompletionSnap::from_completion(&entry.completion),
                _ => None,
            }
        };
        let Some(snap) = snap else {
            break;
        };
        if !snap.is_complete() {
            break;
        }
        let _ = device_retired.compare_exchange(next - 1, next, Ordering::AcqRel, Ordering::Acquire);
    }
}

pub(super) fn lookup_event(ledger: &EventLedger, context: ContextHandle, value: u64) -> Option<Arc<CudaEvent>> {
    let guard = ledger.lock().unwrap();
    guard.get(&value).and_then(|entry| {
        if entry.context == context {
            match &entry.completion {
                LedgerCompletion::CudaEvent(event) => Some(Arc::clone(event)),
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                LedgerCompletion::Dx12Fence { .. } => None,
            }
        } else {
            None
        }
    })
}

pub(super) fn lookup_completion(ledger: &EventLedger, context: ContextHandle, value: u64) -> Option<LedgerCompletion> {
    let guard = ledger.lock().unwrap();
    guard.get(&value).and_then(|entry| {
        if entry.context == context {
            Some(match &entry.completion {
                LedgerCompletion::CudaEvent(event) => LedgerCompletion::CudaEvent(Arc::clone(event)),
                #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                LedgerCompletion::Dx12Fence { companion, value } => LedgerCompletion::Dx12Fence {
                    companion: Arc::clone(companion),
                    value: *value,
                },
            })
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

/// Fill the DX12 fence value for a present ledger entry at Signal time (not earlier).
#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
pub(super) fn bind_dx12_fence_value(ledger: &EventLedger, timeline: u64, fence_value: u64) {
    if let Some(entry) = ledger.lock().unwrap().get_mut(&timeline) {
        if let LedgerCompletion::Dx12Fence { value, .. } = &mut entry.completion {
            *value = fence_value;
        }
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

#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
pub(super) struct Dx12FenceTimelineBlockingWait {
    pub companion: Arc<Dx12Companion>,
    pub value: u64,
}

#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
impl TimelineBlockingWait for Dx12FenceTimelineBlockingWait {
    fn block(self: Box<Self>) -> Result<()> {
        if self.value == 0 {
            bail!("CUDA/DX12: cannot wait on unbound present fence");
        }
        self.companion.cpu_wait(self.value)
    }

    fn block_timeout(self: Box<Self>, timeout_ms: u32) -> Result<bool> {
        if self.value == 0 {
            return Ok(false);
        }
        if timeout_ms == 0 {
            return Ok(unsafe { self.companion.fence.GetCompletedValue() } >= self.value);
        }
        let deadline = Instant::now() + Duration::from_millis(u64::from(timeout_ms));
        while Instant::now() < deadline {
            if unsafe { self.companion.fence.GetCompletedValue() } >= self.value {
                return Ok(true);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(unsafe { self.companion.fence.GetCompletedValue() } >= self.value)
    }
}

/// Wait handle for a timeline value that was never submitted (no ledger event).
///
/// [`block_timeout`](TimelineBlockingWait::block_timeout) always reports timeout so
/// `wait_until_timeout` can return [`crate::error::GoldyError::SubmitTimeout`] like DX12
/// fence waits on a never-signaled value. [`block`](TimelineBlockingWait::block) errors.
pub(super) struct CudaAbsentTimelineWait {
    pub context: ContextHandle,
    pub value: u64,
}

impl TimelineBlockingWait for CudaAbsentTimelineWait {
    fn block(self: Box<Self>) -> Result<()> {
        bail!(
            "CUDA: no completion event for context {} value {}",
            self.context,
            self.value
        )
    }

    fn block_timeout(self: Box<Self>, _timeout_ms: u32) -> Result<bool> {
        Ok(false)
    }
}

/// Host-side wait on a CUDA event (used by submission-worker sidecar).
pub(super) fn host_wait_event(event: &CudaEvent) -> Result<()> {
    event.synchronize().context("CUDA: host wait on event failed")
}
