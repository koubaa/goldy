//! Context-shared deposit staging pool owned by the memory exchange.
//!
//! Physical CPU-writable staging buffers are exchange-local: they are not merchant
//! parcels and do not participate in the parcel ledger. Retirement is an epoch
//! (`ready_after`) on the submission that consumed the deposit copy.

use crate::backend::BufferHandle;
use crate::context::Context;
use crate::error::GoldyError;
use crate::timeline::TimelineValue;
use crate::types::{BufferFlags, BufferKind};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

static NEXT_DEPOSIT_AFFINITY: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_deposit_affinity() -> u64 {
    NEXT_DEPOSIT_AFFINITY.fetch_add(1, Ordering::Relaxed)
}

struct DepositBacking {
    handle: BufferHandle,
    capacity: u64,
    ready_after: TimelineValue,
    claimed: bool,
    last_affinity: u64,
}

/// Shared pool of deposit staging buffers for one [`Context`].
pub(crate) struct DepositExchangePool {
    entries: Mutex<Vec<DepositBacking>>,
    alloc_count: AtomicUsize,
}

impl DepositExchangePool {
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            alloc_count: AtomicUsize::new(0),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<DepositBacking>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Checkout a settled best-fit backing, preferring `affinity`, or allocate.
    pub(crate) fn take_or_alloc(&self, ctx: &Context, need: u64, affinity: u64) -> Result<BufferHandle, GoldyError> {
        let progress = ctx.gpu_progress();
        {
            let mut entries = self.lock();
            if let Some(i) = pick_settled(&entries, progress, need, Some(affinity)) {
                entries[i].claimed = true;
                entries[i].last_affinity = affinity;
                return Ok(entries[i].handle);
            }
            if let Some(i) = pick_settled(&entries, progress, need, None) {
                entries[i].claimed = true;
                entries[i].last_affinity = affinity;
                return Ok(entries[i].handle);
            }
        }

        let device = ctx.device().inner.handle;
        let handle = {
            let mut backend = ctx.device().inner.backend.lock().unwrap();
            backend
                .create_buffer(device, need, BufferKind::Scattered, None, BufferFlags::CPU_WRITABLE)
                .map_err(|e| ctx.classify(e))?
        };
        let mut entries = self.lock();
        entries.push(DepositBacking {
            handle,
            capacity: need,
            ready_after: 0,
            claimed: true,
            last_affinity: affinity,
        });
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        Ok(handle)
    }

    pub(crate) fn return_handle(&self, handle: BufferHandle, ready_after: TimelineValue) {
        let mut entries = self.lock();
        if let Some(entry) = entries.iter_mut().find(|e| e.handle == handle) {
            entry.claimed = false;
            entry.ready_after = ready_after;
        }
    }

    pub(crate) fn write_handle(
        &self,
        ctx: &Context,
        handle: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<(), GoldyError> {
        let result = {
            let mut backend = ctx.device().inner.backend.lock().unwrap();
            backend.write_buffer(handle, offset, data)
        };
        result.map_err(|e| ctx.classify(e))
    }

    /// Park every backing after waiting for in-flight copies (context teardown).
    pub(crate) fn drain_backend(&self, device: &crate::device::Device, handle: crate::backend::ContextHandle) {
        let max_ready = {
            let entries = self.lock();
            entries.iter().map(|e| e.ready_after).max().unwrap_or(0)
        };
        if max_ready > 0 {
            let mut backend = device.inner.backend.lock().unwrap();
            let _ = backend.wait_until(handle, max_ready);
        }
        let drained: Vec<DepositBacking> = self.lock().drain(..).collect();
        let mut backend = device.inner.backend.lock().unwrap();
        for entry in drained {
            backend.destroy_buffer(entry.handle);
        }
    }

    pub(crate) fn alloc_count(&self) -> usize {
        self.alloc_count.load(Ordering::Relaxed)
    }

    pub(crate) fn live_count(&self) -> usize {
        self.lock().len()
    }

    pub(crate) fn count_for_affinity(&self, affinity: u64) -> usize {
        self.lock().iter().filter(|e| e.last_affinity == affinity).count()
    }

    pub(crate) fn handles_for_affinity(&self, affinity: u64) -> Vec<BufferHandle> {
        self.lock()
            .iter()
            .filter(|e| e.last_affinity == affinity)
            .map(|e| e.handle)
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn parked_epoch(&self, handle: BufferHandle) -> Option<TimelineValue> {
        self.lock()
            .iter()
            .find(|entry| entry.handle == handle && !entry.claimed)
            .map(|entry| entry.ready_after)
    }

    /// Force a parked backing with this affinity to look in-flight until `tv`.
    pub(crate) fn mark_affinity_inflight(&self, affinity: u64, tv: TimelineValue) -> bool {
        let mut entries = self.lock();
        if let Some(entry) = entries
            .iter_mut()
            .rev()
            .find(|e| e.last_affinity == affinity && !e.claimed)
        {
            entry.ready_after = tv;
            return true;
        }
        false
    }
}

fn pick_settled(
    entries: &[DepositBacking],
    progress: TimelineValue,
    need: u64,
    affinity: Option<u64>,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, e) in entries.iter().enumerate() {
        if e.claimed || e.ready_after > progress || e.capacity < need {
            continue;
        }
        if let Some(a) = affinity {
            if e.last_affinity != a {
                continue;
            }
        }
        match best {
            None => best = Some(i),
            Some(j) if e.capacity < entries[j].capacity => best = Some(i),
            _ => {}
        }
    }
    best
}
