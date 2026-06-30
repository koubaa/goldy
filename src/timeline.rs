//! Monotonic GPU timeline values for completion tracking.
//!
//! Each successful standalone submission or completed frame bracket is assigned a
//! [`TimelineValue`] that the GPU signals when that work finishes. Use
//! [`crate::Context::gpu_progress`] to query completion without blocking, and
//! [`crate::Context::wait_until`] / [`crate::Context::wait_until_timeout`] to block.
//!
//! ## Resource lifetime vs the timeline
//!
//! Destroying a [`crate::Buffer`], [`crate::Texture`], or similar may be **deferred** on GPU
//! backends: the handle becomes invalid immediately, but underlying GPU memory may be kept
//! alive until all work **already submitted** before the destroy has finished (the same
//! conservative rule as tagging with the latest scheduled timeline point). If you record
//! commands that use a resource and destroy it **before** submitting that recording, the
//! implementation cannot always detect the hazard — submit (or bracket a frame) before
//! dropping resources that must outlive those commands.

use crate::backend::ContextHandle;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

pub type TimelineValue = u64;

/// Raw `UsageKindFlags::COMPUTE` bits for [`ResourceSync::record_write`] (no `task_graph` import).
pub const WRITE_KINDS_COMPUTE: u8 = 0b001;

/// Raw `UsageKindFlags::TRANSFER` bits for [`ResourceSync::record_write`] (no `task_graph` import).
pub const WRITE_KINDS_TRANSFER: u8 = 0b010;

/// `WRITE_KINDS_COMPUTE | WRITE_KINDS_TRANSFER` — matches `UsageKindFlags::COMPUTE | UsageKindFlags::TRANSFER`.
///
/// Conservative default when a prior write's pipeline category is unknown (legacy stamping,
/// missing `last_write_kinds` entry).
pub const WRITE_KINDS_COMPUTE_TRANSFER: u8 = WRITE_KINDS_COMPUTE | WRITE_KINDS_TRANSFER;

/// A context-qualified timeline stamp: which context's semaphore must reach `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Epoch {
    pub context: ContextHandle,
    pub value: TimelineValue,
}

/// Inline capacity before spilling to heap for [`SmallContextMap`].
const SMALL_CONTEXT_INLINE_CAP: usize = 4;

/// Inline storage for per-context maps keyed by [`ContextHandle`].
///
/// Retained parcels and cross-submit sync typically touch 1–3 contexts; this avoids
/// heap allocation and SipHash overhead of `HashMap` at that scale. Spills to a `Vec`
/// when more than [`SMALL_CONTEXT_INLINE_CAP`] distinct contexts are recorded.
#[derive(Clone, PartialEq, Eq)]
pub struct SmallContextMap<T: Copy + Default> {
    inline: [(ContextHandle, T); SMALL_CONTEXT_INLINE_CAP],
    inline_len: u8,
    spill: Vec<(ContextHandle, T)>,
}

impl<T: Copy + Default> SmallContextMap<T> {
    pub const INLINE_CAP: usize = SMALL_CONTEXT_INLINE_CAP;

    pub fn new() -> Self {
        Self {
            inline: std::array::from_fn(|_| (0, T::default())),
            inline_len: 0,
            spill: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inline_len == 0 && self.spill.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inline_len as usize + self.spill.len()
    }

    pub fn get(&self, key: ContextHandle) -> Option<T> {
        self.find(key).map(|loc| self.read(loc))
    }

    pub fn insert(&mut self, key: ContextHandle, value: T) {
        match self.find(key) {
            Some(loc) => self.write(loc, value),
            None => self.push(key, value),
        }
    }

    pub fn get_or_insert(&mut self, key: ContextHandle, default: T) -> &mut T {
        if let Some(loc) = self.find(key) {
            return self.write_ref(loc);
        }
        self.push(key, default);
        self.write_ref(self.find(key).expect("just inserted"))
    }

    pub fn get_mut(&mut self, key: ContextHandle) -> Option<&mut T> {
        self.find(key).map(|loc| self.write_ref(loc))
    }

    pub fn iter(&self) -> impl Iterator<Item = (ContextHandle, T)> + '_ {
        (0..self.inline_len as usize)
            .map(|i| self.inline[i])
            .chain(self.spill.iter().copied())
    }

    pub fn keys(&self) -> impl Iterator<Item = ContextHandle> + '_ {
        self.iter().map(|(ctx, _)| ctx)
    }

    pub fn values(&self) -> impl Iterator<Item = T> + '_ {
        self.iter().map(|(_, v)| v)
    }

    fn find(&self, key: ContextHandle) -> Option<SlotLoc> {
        for i in 0..self.inline_len as usize {
            if self.inline[i].0 == key {
                return Some(SlotLoc::Inline(i));
            }
        }
        for (i, &(k, _)) in self.spill.iter().enumerate() {
            if k == key {
                return Some(SlotLoc::Spill(i));
            }
        }
        None
    }

    fn read(&self, loc: SlotLoc) -> T {
        match loc {
            SlotLoc::Inline(i) => self.inline[i].1,
            SlotLoc::Spill(i) => self.spill[i].1,
        }
    }

    fn write(&mut self, loc: SlotLoc, value: T) {
        match loc {
            SlotLoc::Inline(i) => self.inline[i].1 = value,
            SlotLoc::Spill(i) => self.spill[i].1 = value,
        }
    }

    fn write_ref(&mut self, loc: SlotLoc) -> &mut T {
        match loc {
            SlotLoc::Inline(i) => &mut self.inline[i].1,
            SlotLoc::Spill(i) => &mut self.spill[i].1,
        }
    }

    fn push(&mut self, key: ContextHandle, value: T) {
        if (self.inline_len as usize) < Self::INLINE_CAP {
            let i = self.inline_len as usize;
            self.inline[i] = (key, value);
            self.inline_len += 1;
        } else {
            self.spill.push((key, value));
        }
    }
}

impl SmallContextMap<TimelineValue> {
    pub fn mark_max(&mut self, key: ContextHandle, tv: TimelineValue) {
        let entry = self.get_or_insert(key, 0);
        *entry = (*entry).max(tv);
    }
}

impl SmallContextMap<u8> {
    pub fn merge_kind(&mut self, key: ContextHandle, kinds_bits: u8) {
        if let Some(v) = self.get_mut(key) {
            *v |= kinds_bits;
        } else {
            self.insert(key, kinds_bits);
        }
    }
}

impl<T: Copy + Default> Default for SmallContextMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default + std::fmt::Debug> std::fmt::Debug for SmallContextMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[derive(Copy, Clone)]
enum SlotLoc {
    Inline(usize),
    Spill(usize),
}

/// Per-context last-referencing timeline values for a retained parcel.
pub type ReferenceTable = SmallContextMap<TimelineValue>;

/// Direction-aware GPU sync epochs for one retained resource (parcel backing).
///
/// Used by cross-scheme hazard analysis: reads depend on [`Self::last_write`],
/// writes depend on both [`Self::last_write`] and [`Self::last_reads`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceSync {
    /// Last submission on `ctx` that wrote this resource.
    pub last_write: ReferenceTable,
    /// Raw `UsageKindFlags` bits for the last write per context.
    ///
    /// Stored as `u8` to avoid a circular module dependency (`timeline` ↔ `task_graph`).
    /// Callers in `task_graph` cast this to `UsageKindFlags` via `from_bits_truncate`.
    pub last_write_kinds: SmallContextMap<u8>,
    /// Last submission on `ctx` that read this resource (max per context).
    pub last_reads: ReferenceTable,
    /// Present easement read epochs on `ctx` that require a live queue wait on the next
    /// same-context write (legacy present path — GPU work not enqueued on the FIFO worker).
    pub war_read_epochs: ReferenceTable,
    /// Read epochs on `ctx` whose hazard is covered by present work enqueued on the
    /// submission worker before the next frame's compute partitions.
    pub fifo_ordered_reads: ReferenceTable,
}

impl ResourceSync {
    /// Per-context max of read and write epochs — used for reuse gating ([`crate::Parcel::is_settled`]).
    pub fn merged(&self) -> ReferenceTable {
        let mut merged = self.last_write.clone();
        for (ctx, tv) in self.last_reads.iter() {
            mark_reference(&mut merged, ctx, tv);
        }
        merged
    }

    /// Record a write with its usage kind bits (raw `UsageKindFlags::bits()`).
    ///
    /// When `tv` supersedes the existing epoch, the kinds are replaced; when equal they
    /// are ORed together; when older the existing record wins (monotonic per context).
    pub fn record_write(&mut self, ctx: ContextHandle, tv: TimelineValue, kinds_bits: u8) {
        let entry = self.last_write.get_or_insert(ctx, 0);
        match tv.cmp(entry) {
            std::cmp::Ordering::Greater => {
                *entry = tv;
                self.last_write_kinds.insert(ctx, kinds_bits);
            }
            std::cmp::Ordering::Equal => {
                self.last_write_kinds.merge_kind(ctx, kinds_bits);
            }
            std::cmp::Ordering::Less => {
                // Existing write is newer; don't update.
            }
        }
    }

    pub fn record_read(&mut self, ctx: ContextHandle, tv: TimelineValue) {
        mark_reference(&mut self.last_reads, ctx, tv);
    }

    pub fn mark_war_read(&mut self, ctx: ContextHandle, tv: TimelineValue) {
        mark_reference(&mut self.war_read_epochs, ctx, tv);
    }

    pub fn mark_fifo_ordered_read(&mut self, ctx: ContextHandle, tv: TimelineValue) {
        mark_reference(&mut self.fifo_ordered_reads, ctx, tv);
    }

    /// Conservative touch for legacy/test-only use when kinds are unknown.
    ///
    /// Uses [`WRITE_KINDS_COMPUTE_TRANSFER`] — the maximally conservative non-render write set.
    pub fn record_any(&mut self, ctx: ContextHandle, tv: TimelineValue) {
        self.record_write(ctx, tv, WRITE_KINDS_COMPUTE_TRANSFER);
        self.record_read(ctx, tv);
    }
}

/// Record `tv` for `ctx`, monotonically per context.
pub fn mark_reference(table: &mut ReferenceTable, ctx: ContextHandle, tv: TimelineValue) {
    table.mark_max(ctx, tv);
}

/// True when `progress` has retired every entry in `table`.
pub fn is_ready(table: &ReferenceTable, progress: &HashMap<ContextHandle, TimelineValue>) -> bool {
    table
        .iter()
        .all(|(ctx, tv)| progress.get(&ctx).copied().unwrap_or(0) >= tv)
}

/// Fast single-context readiness check.
pub fn is_ready_on(table: &ReferenceTable, ctx: ContextHandle, progress: TimelineValue) -> bool {
    table.get(ctx).is_none_or(|tv| progress >= tv)
}

/// Collect `table` entries as [`Epoch`] values.
pub fn epochs_from(table: &ReferenceTable) -> Vec<Epoch> {
    table
        .iter()
        .map(|(context, value)| Epoch { context, value })
        .collect()
}

const PROMISE_PENDING: u64 = 0;
const PROMISE_ABANDONED: u64 = u64::MAX;

/// Progress for a context that has been destroyed (GPU was drained at teardown).
pub(crate) const CONTEXT_DESTROYED_PROGRESS: TimelineValue = u64::MAX;

/// Pattern-match primitive for a within-context timeline promise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromiseState {
    /// The resolver has not yet resolved or abandoned this promise.
    Pending,
    /// The easement's expiry timeline value on the owning context.
    Resolved(TimelineValue),
    /// The resolver was dropped without resolving — vacuously satisfied.
    Abandoned,
}

struct PromiseCell {
    state: AtomicU64,
    park: Mutex<()>,
    wake: Condvar,
}

impl PromiseCell {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU64::new(PROMISE_PENDING),
            park: Mutex::new(()),
            wake: Condvar::new(),
        })
    }

    fn load_state(&self) -> PromiseState {
        match self.state.load(Ordering::Acquire) {
            PROMISE_PENDING => PromiseState::Pending,
            PROMISE_ABANDONED => PromiseState::Abandoned,
            tv => PromiseState::Resolved(tv),
        }
    }

    fn resolve(&self, tv: TimelineValue) {
        debug_assert!(tv != PROMISE_PENDING && tv != PROMISE_ABANDONED);
        if self
            .state
            .compare_exchange(PROMISE_PENDING, tv, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // Lock then immediately drop before notifying. This closes the lost-wakeup
            // window: block()'s inner poll and its condvar wait are both done while
            // holding `park`. Either the inner poll sees the new state (and returns
            // before sleeping), or we acquire `park` only after the waiter is already
            // parked — in which case notify_all() will wake it.
            drop(self.park.lock().unwrap());
            self.wake.notify_all();
        }
    }

    fn abandon(&self) {
        if self
            .state
            .compare_exchange(PROMISE_PENDING, PROMISE_ABANDONED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            drop(self.park.lock().unwrap());
            self.wake.notify_all();
        }
    }
}

/// Read-end of a within-context timeline promise.
///
/// Cheap to clone; registered on parcel stamps for easements whose expiry value is
/// not yet known at submit time (e.g. async present completion).
#[derive(Clone)]
pub struct TimelinePromise {
    cell: Arc<PromiseCell>,
}

impl TimelinePromise {
    /// Non-blocking poll — callers must pattern-match on [`PromiseState`].
    pub fn poll(&self) -> PromiseState {
        self.cell.load_state()
    }

    /// Opt-in block until the promise is [`PromiseState::Resolved`] or [`PromiseState::Abandoned`].
    pub fn block(&self) -> PromiseState {
        loop {
            match self.poll() {
                PromiseState::Pending => {
                    let guard = self.cell.park.lock().unwrap();
                    match self.poll() {
                        PromiseState::Pending => {
                            let _guard = self.cell.wake.wait(guard).unwrap();
                        }
                        other => return other,
                    }
                }
                other => return other,
            }
        }
    }

    /// Create a new unresolved promise pair.
    pub fn new() -> (Self, PromiseResolver) {
        let cell = PromiseCell::new();
        (
            Self {
                cell: Arc::clone(&cell),
            },
            PromiseResolver { cell },
        )
    }
}

impl std::fmt::Debug for TimelinePromise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimelinePromise").field("state", &self.poll()).finish()
    }
}

/// Write-end of a within-context timeline promise.
///
/// Dropping without [`Self::resolve`] transitions the promise to [`PromiseState::Abandoned`].
pub struct PromiseResolver {
    cell: Arc<PromiseCell>,
}

impl PromiseResolver {
    /// Resolve the promise with the easement-expiry timeline value (resolve-once).
    pub fn resolve(self, tv: TimelineValue) {
        self.cell.resolve(tv);
    }
}

impl Drop for PromiseResolver {
    fn drop(&mut self) {
        self.cell.abandon();
    }
}

impl std::fmt::Debug for PromiseResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromiseResolver")
            .field("state", &self.cell.load_state())
            .finish()
    }
}

/// Reuse-gate result for a parcel, including outstanding timeline promises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Settle {
    /// All resolved epochs are retired and no live pending promises remain.
    Ready,
    /// A resolved epoch on this context has not yet been retired by the GPU.
    Waiting(TimelineValue),
    /// At least one promise is still unresolved — the expiry value is unknown.
    Pending,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn record_write_tracks_kinds_per_context() {
        let mut sync = ResourceSync::default();
        sync.record_write(1, 5, WRITE_KINDS_COMPUTE);
        sync.record_write(2, 7, WRITE_KINDS_TRANSFER);
        assert_eq!(sync.last_write.get(1), Some(5));
        assert_eq!(sync.last_write_kinds.get(1), Some(WRITE_KINDS_COMPUTE));
        assert_eq!(sync.last_write_kinds.get(2), Some(WRITE_KINDS_TRANSFER));
    }

    #[test]
    fn newer_write_replaces_kinds() {
        let mut sync = ResourceSync::default();
        sync.record_write(1, 5, WRITE_KINDS_COMPUTE);
        sync.record_write(1, 9, WRITE_KINDS_TRANSFER);
        assert_eq!(sync.last_write.get(1), Some(9));
        // Newer epoch fully supersedes the kinds of the older one.
        assert_eq!(sync.last_write_kinds.get(1), Some(WRITE_KINDS_TRANSFER));
    }

    #[test]
    fn equal_epoch_writes_or_kinds() {
        let mut sync = ResourceSync::default();
        sync.record_write(1, 5, WRITE_KINDS_COMPUTE);
        sync.record_write(1, 5, WRITE_KINDS_TRANSFER);
        assert_eq!(sync.last_write.get(1), Some(5));
        assert_eq!(sync.last_write_kinds.get(1), Some(WRITE_KINDS_COMPUTE_TRANSFER));
    }

    #[test]
    fn older_write_is_ignored() {
        let mut sync = ResourceSync::default();
        sync.record_write(1, 9, WRITE_KINDS_COMPUTE);
        sync.record_write(1, 3, WRITE_KINDS_TRANSFER);
        assert_eq!(sync.last_write.get(1), Some(9));
        // Stale write must not overwrite the newer kinds.
        assert_eq!(sync.last_write_kinds.get(1), Some(WRITE_KINDS_COMPUTE));
    }

    #[test]
    fn record_any_uses_conservative_kinds() {
        let mut sync = ResourceSync::default();
        sync.record_any(1, 4);
        assert_eq!(sync.last_write_kinds.get(1), Some(WRITE_KINDS_COMPUTE_TRANSFER));
        assert_eq!(sync.last_reads.get(1), Some(4));
    }

    #[test]
    fn promise_new_is_pending() {
        let (promise, _resolver) = TimelinePromise::new();
        assert_eq!(promise.poll(), PromiseState::Pending);
    }

    #[test]
    fn promise_resolve_becomes_resolved() {
        let (promise, resolver) = TimelinePromise::new();
        resolver.resolve(42);
        assert_eq!(promise.poll(), PromiseState::Resolved(42));
    }

    #[test]
    fn promise_drop_resolver_abandons() {
        let (promise, resolver) = TimelinePromise::new();
        drop(resolver);
        assert_eq!(promise.poll(), PromiseState::Abandoned);
    }

    #[test]
    fn promise_resolve_is_once() {
        let (promise, resolver) = TimelinePromise::new();
        resolver.resolve(10);
        assert_eq!(promise.poll(), PromiseState::Resolved(10));
    }

    #[test]
    fn promise_block_returns_resolved() {
        let (promise, resolver) = TimelinePromise::new();
        let reader = Arc::new(promise);
        let clone = Arc::clone(&reader);
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            resolver.resolve(99);
        });
        assert_eq!(clone.block(), PromiseState::Resolved(99));
        handle.join().unwrap();
    }

    #[test]
    fn promise_block_returns_abandoned() {
        let (promise, resolver) = TimelinePromise::new();
        let reader = Arc::new(promise);
        let clone = Arc::clone(&reader);
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            drop(resolver);
        });
        assert_eq!(clone.block(), PromiseState::Abandoned);
        handle.join().unwrap();
    }

    #[test]
    fn promise_block_never_returns_pending() {
        let (promise, resolver) = TimelinePromise::new();
        resolver.resolve(7);
        assert_ne!(promise.block(), PromiseState::Pending);
    }
}
