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

pub type TimelineValue = u64;

/// A context-qualified timeline stamp: which context's semaphore must reach `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Epoch {
    pub context: ContextHandle,
    pub value: TimelineValue,
}

/// Per-context last-referencing timeline values for a retained parcel.
pub type ReferenceTable = HashMap<ContextHandle, TimelineValue>;

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
    pub last_write_kinds: HashMap<ContextHandle, u8>,
    /// Last submission on `ctx` that read this resource (max per context).
    pub last_reads: ReferenceTable,
}

impl ResourceSync {
    /// Per-context max of read and write epochs — used for reuse gating ([`crate::Parcel::is_settled`]).
    pub fn merged(&self) -> ReferenceTable {
        let mut merged = self.last_write.clone();
        for (&ctx, &tv) in &self.last_reads {
            mark_reference(&mut merged, ctx, tv);
        }
        merged
    }

    /// Record a write with its usage kind bits (raw `UsageKindFlags::bits()`).
    ///
    /// When `tv` supersedes the existing epoch, the kinds are replaced; when equal they
    /// are ORed together; when older the existing record wins (monotonic per context).
    pub fn record_write(&mut self, ctx: ContextHandle, tv: TimelineValue, kinds_bits: u8) {
        let entry = self.last_write.entry(ctx).or_insert(0);
        match tv.cmp(entry) {
            std::cmp::Ordering::Greater => {
                *entry = tv;
                self.last_write_kinds.insert(ctx, kinds_bits);
            }
            std::cmp::Ordering::Equal => {
                self.last_write_kinds
                    .entry(ctx)
                    .and_modify(|k| *k |= kinds_bits)
                    .or_insert(kinds_bits);
            }
            std::cmp::Ordering::Less => {
                // Existing write is newer; don't update.
            }
        }
    }

    pub fn record_read(&mut self, ctx: ContextHandle, tv: TimelineValue) {
        mark_reference(&mut self.last_reads, ctx, tv);
    }

    /// Conservative touch for legacy/test-only use when kinds are unknown.
    ///
    /// Uses `0b011` (COMPUTE | TRANSFER) — the maximally conservative set for
    /// non-render writes, matching what the barrier code previously hardcoded.
    pub fn record_any(&mut self, ctx: ContextHandle, tv: TimelineValue) {
        self.record_write(ctx, tv, 0b011); // COMPUTE | TRANSFER
        self.record_read(ctx, tv);
    }
}

/// Record `tv` for `ctx`, monotonically per context.
pub fn mark_reference(table: &mut ReferenceTable, ctx: ContextHandle, tv: TimelineValue) {
    table.entry(ctx).and_modify(|v| *v = (*v).max(tv)).or_insert(tv);
}

/// True when `progress` has retired every entry in `table`.
pub fn is_ready(table: &ReferenceTable, progress: &HashMap<ContextHandle, TimelineValue>) -> bool {
    table
        .iter()
        .all(|(ctx, &tv)| progress.get(ctx).copied().unwrap_or(0) >= tv)
}

/// Fast single-context readiness check.
pub fn is_ready_on(table: &ReferenceTable, ctx: ContextHandle, progress: TimelineValue) -> bool {
    table.get(&ctx).is_none_or(|&tv| progress >= tv)
}

/// Collect `table` entries as [`Epoch`] values.
pub fn epochs_from(table: &ReferenceTable) -> Vec<Epoch> {
    table
        .iter()
        .map(|(&context, &value)| Epoch { context, value })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPUTE: u8 = 0b001;
    const TRANSFER: u8 = 0b010;

    #[test]
    fn record_write_tracks_kinds_per_context() {
        let mut sync = ResourceSync::default();
        sync.record_write(1, 5, COMPUTE);
        sync.record_write(2, 7, TRANSFER);
        assert_eq!(sync.last_write.get(&1), Some(&5));
        assert_eq!(sync.last_write_kinds.get(&1), Some(&COMPUTE));
        assert_eq!(sync.last_write_kinds.get(&2), Some(&TRANSFER));
    }

    #[test]
    fn newer_write_replaces_kinds() {
        let mut sync = ResourceSync::default();
        sync.record_write(1, 5, COMPUTE);
        sync.record_write(1, 9, TRANSFER);
        assert_eq!(sync.last_write.get(&1), Some(&9));
        // Newer epoch fully supersedes the kinds of the older one.
        assert_eq!(sync.last_write_kinds.get(&1), Some(&TRANSFER));
    }

    #[test]
    fn equal_epoch_writes_or_kinds() {
        let mut sync = ResourceSync::default();
        sync.record_write(1, 5, COMPUTE);
        sync.record_write(1, 5, TRANSFER);
        assert_eq!(sync.last_write.get(&1), Some(&5));
        assert_eq!(sync.last_write_kinds.get(&1), Some(&(COMPUTE | TRANSFER)));
    }

    #[test]
    fn older_write_is_ignored() {
        let mut sync = ResourceSync::default();
        sync.record_write(1, 9, COMPUTE);
        sync.record_write(1, 3, TRANSFER);
        assert_eq!(sync.last_write.get(&1), Some(&9));
        // Stale write must not overwrite the newer kinds.
        assert_eq!(sync.last_write_kinds.get(&1), Some(&COMPUTE));
    }

    #[test]
    fn record_any_uses_conservative_kinds() {
        let mut sync = ResourceSync::default();
        sync.record_any(1, 4);
        assert_eq!(sync.last_write_kinds.get(&1), Some(&(COMPUTE | TRANSFER)));
        assert_eq!(sync.last_reads.get(&1), Some(&4));
    }
}
