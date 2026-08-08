//! CUDA timeline events, retirement, and blocking waits.

use super::ContextHandle;
use crate::backend::TimelineBlockingWait;
use anyhow::{bail, Context as _, Result};
use cudarc::driver::{CudaContext, CudaEvent};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
use super::dx12_companion::Dx12Companion;

/// Soft cap on recycled completion events per device (steady-state depth + headroom).
const EVENT_POOL_CAP: usize = 32;
/// Events allocated at device create so the first submits avoid `cuEventCreate`.
const EVENT_POOL_PREWARM: usize = 16;

/// Per-device recycle pool for CUDA completion events (DX12 fence analogue).
///
/// Steady-state submits should hit the free list; `cuEventCreate` / `cuEventDestroy`
/// only run on cold start, miss, or overflow beyond [`EVENT_POOL_CAP`].
pub(super) struct EventPool {
    ctx: Arc<CudaContext>,
    free: Mutex<Vec<CudaEvent>>,
}

impl EventPool {
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        Self {
            ctx,
            free: Mutex::new(Vec::new()),
        }
    }

    pub fn prewarm(&self) -> Result<()> {
        let mut free = self.free.lock().unwrap();
        while free.len() < EVENT_POOL_PREWARM {
            free.push(
                self.ctx
                    .new_event(None)
                    .context("CUDA: event pool prewarm failed")?,
            );
        }
        Ok(())
    }

    /// Take a recycled event or create one. Returned as [`Arc`] for ledger / wait cloning.
    pub fn acquire(&self) -> Result<Arc<CudaEvent>> {
        if let Some(event) = self.free.lock().unwrap().pop() {
            let _tz = crate::tracy_zone!("cuda.event_pool.acquire.hit");
            return Ok(Arc::new(event));
        }
        let _tz = crate::tracy_zone!("cuda.event_pool.acquire.miss");
        Ok(Arc::new(
            self.ctx
                .new_event(None)
                .context("CUDA: create completion event failed")?,
        ))
    }

    /// Return events whose ledger entries have retired. Drops that still have live
    /// wait clones are skipped (`Arc::try_unwrap` fails); overflow beyond the cap is
    /// destroyed **outside** the free-list lock.
    pub fn recycle_many(&self, events: Vec<Arc<CudaEvent>>) {
        if events.is_empty() {
            return;
        }
        let _tz = crate::tracy_zone!("cuda.event_pool.recycle");
        let mut overflow = Vec::new();
        {
            let mut free = self.free.lock().unwrap();
            for event in events {
                let Ok(event) = Arc::try_unwrap(event) else {
                    continue;
                };
                if free.len() < EVENT_POOL_CAP {
                    free.push(event);
                } else {
                    overflow.push(event);
                }
            }
        }
        // `cuEventDestroy` (and bind_to_thread) must not run under `free` or the ledger lock.
        drop(overflow);
    }
}

/// How a ledger timeline value becomes observable as complete.
pub(super) enum LedgerCompletion {
    CudaEvent(Arc<CudaEvent>),
    /// DX12 companion fence signaled on the presentation DIRECT queue.
    ///
    /// Present publishes [`LedgerCompletion::CudaEvent`] (bridged on `present_stream`).
    /// This variant remains for residual fence-ledger producers; `value` may be 0 until
    /// [`bind_dx12_fence_value`] runs at Signal time.
    #[allow(dead_code)]
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
            LedgerCompletion::Dx12Fence { companion, value } => (*value > 0).then(|| Self::Dx12Fence {
                companion: Arc::clone(companion),
                value: *value,
            }),
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
    event_pool: &EventPool,
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

    // Monotonic: concurrent pollers (worker + fence thread) must not clobber a
    // higher completed watermark with a stale snapshot. After prune, the ledger
    // no longer retains older complete entries to rediscover.
    if ctx_max > 0 {
        context_completed.fetch_max(ctx_max, Ordering::AcqRel);
    }

    let mut last = last_emitted.load(Ordering::Acquire);
    let completed = context_completed.load(Ordering::Acquire);
    while last < completed {
        last += 1;
        signal_queue.push_boundary_crossed(last);
        last_emitted.store(last, Ordering::Release);
    }

    advance_device_retired(ledger, device_retired);
    // Recycle (or destroy) retired events outside the ledger lock.
    let retired_events = prune_retired_entries(ledger, device_retired);
    event_pool.recycle_many(retired_events);
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

/// Remove ledger entries at or below the device retirement floor.
///
/// Returns CUDA events from pruned entries so callers can recycle them **after**
/// releasing the ledger mutex (avoids holding the lock across `cuEventDestroy`).
///
/// Callers that resolve waits must treat a missing entry with
/// `value <= device_retired` as already complete (see [`completion_for_wait`]).
pub(super) fn prune_retired_entries(
    ledger: &EventLedger,
    device_retired: &AtomicU64,
) -> Vec<Arc<CudaEvent>> {
    let retired = device_retired.load(Ordering::Acquire);
    if retired == 0 {
        return Vec::new();
    }
    let retired_map = {
        let mut guard = ledger.lock().unwrap();
        if guard.is_empty() {
            return Vec::new();
        }
        // `split_off(retired + 1)` keeps keys >= retired+1 in the returned map;
        // what remains in `guard` is the retired prefix — swap it out without dropping
        // under the lock.
        let keep = guard.split_off(&(retired + 1));
        std::mem::replace(&mut *guard, keep)
    };
    let mut events = Vec::new();
    for (_, entry) in retired_map {
        if let LedgerCompletion::CudaEvent(event) = entry.completion {
            events.push(event);
        }
    }
    events
}

/// Result of resolving a timeline wait against the event ledger.
pub(super) enum WaitCompletion {
    /// Live completion marker; caller must wait.
    Pending(LedgerCompletion),
    /// Value is already at or below `device_retired` (entry may have been pruned).
    AlreadyComplete,
    /// Not submitted / unknown — caller decides (bail vs absent wait).
    Missing,
}

/// Look up a wait target, accounting for pruned retired ledger entries.
pub(super) fn completion_for_wait(
    ledger: &EventLedger,
    device_retired: &AtomicU64,
    context: ContextHandle,
    value: u64,
) -> WaitCompletion {
    if let Some(completion) = lookup_completion(ledger, context, value) {
        return WaitCompletion::Pending(completion);
    }
    if value > 0 && value <= device_retired.load(Ordering::Acquire) {
        return WaitCompletion::AlreadyComplete;
    }
    WaitCompletion::Missing
}

#[allow(dead_code)]
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

/// Fill the DX12 fence value for a fence ledger entry at Signal time (not earlier).
#[allow(dead_code)] // paired with LedgerCompletion::Dx12Fence for residual producers
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
