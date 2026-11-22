//! Optional command-buffer replay ledger.
//!
//! Separated from [`super::graph::CompiledCacheEntry`] (CPU schedule/emission cache)
//! so that when replay is disabled (`GOLDY_DISABLE_CB_REUSE`, GPU profiling, or an
//! explicit off path) Goldy never computes retention fingerprints, stores closed
//! command lists, waits for retained allocators, or registers topology-dirty edges.

use std::collections::HashSet;

use crate::timeline::TimelineValue;

/// Per-partition backend CB retention bookkeeping.
///
/// Present only when CB replay is enabled. Absence makes retention structurally
/// impossible for the submit path — not merely a failed resubmit attempt.
#[derive(Debug, Default)]
pub(crate) struct CbReplayState {
    /// `Some(key)` when partition `i` was last successfully retained with that key.
    pub partition_keys: Vec<Option<u64>>,
    /// Timeline value from the most recent submission of each partition.
    /// Gates `ensure_partition_retired_before_rerecord` before replacing a retained CB.
    pub partition_last_tv: Vec<Option<TimelineValue>>,
    /// Per-partition set of present-slot combination keys (swapchain image variants).
    pub partition_slot_keys: Vec<Option<HashSet<u64>>>,
    /// Retained command buffers this ledger dropped. Folded into the owning scheme.
    pub evictions: u64,
}

impl CbReplayState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ensure_partition_vecs(&mut self, ctx: &crate::Context, partition_count: usize) {
        if self.partition_keys.len() != partition_count {
            self.evict_all(ctx);
            self.partition_keys = vec![None; partition_count];
            self.partition_slot_keys = vec![None; partition_count];
            self.partition_last_tv = vec![None; partition_count];
        }
    }

    /// Drop cached retention keys so the next submit re-records retained partitions.
    ///
    /// The backend command buffers are evicted too. Their rows and bindless slots
    /// would otherwise stay pinned after the ledger forgot them. Topology or
    /// structural dirtiness forces re-record even when the GPU has not retired
    /// the prior partition timeline — do not gate re-record on stale TVs.
    pub fn invalidate(&mut self, ctx: &crate::Context) {
        self.evict_all(ctx);
        self.partition_keys.fill(None);
        for keys in &mut self.partition_slot_keys {
            *keys = None;
        }
        self.partition_last_tv.fill(None);
    }

    /// Point `indices` at `new_key`, evicting whatever backend buffer they named before.
    pub fn install_keys(&mut self, ctx: &crate::Context, indices: &[usize], new_key: u64) {
        let mut doomed = HashSet::new();
        for &index in indices {
            if let Some(old) = self.partition_keys.get(index).copied().flatten() {
                if old != new_key {
                    doomed.insert(old);
                }
            }
        }
        for key in doomed {
            self.evict_one(ctx, key);
        }
        for &index in indices {
            if let Some(slot) = self.partition_keys.get_mut(index) {
                *slot = Some(new_key);
            }
        }
    }

    fn evict_all(&mut self, ctx: &crate::Context) {
        let keys = self.all_backend_keys();
        for key in keys {
            self.evict_one(ctx, key);
        }
    }

    fn evict_one(&mut self, ctx: &crate::Context, key: u64) {
        ctx.submit_session().evict_retained(ctx.backend_handle(), key);
        self.evictions += 1;
    }

    pub fn record_last_tv(&mut self, part_idx: usize, tv: TimelineValue) {
        self.partition_last_tv[part_idx] = Some(tv);
    }

    pub fn record_merged_last_tvs(&mut self, part_idx: usize, tv: TimelineValue) {
        self.partition_last_tv[part_idx] = Some(tv);
        self.partition_last_tv[part_idx + 1] = Some(tv);
    }

    pub fn last_tvs(&self) -> &[Option<TimelineValue>] {
        &self.partition_last_tv
    }

    /// Collect all backend retention keys currently referenced by this ledger.
    pub fn all_backend_keys(&self) -> HashSet<u64> {
        let mut keys = HashSet::new();
        for key in self.partition_keys.iter().flatten() {
            keys.insert(*key);
        }
        for slot_keys in self.partition_slot_keys.iter().flatten() {
            keys.extend(slot_keys.iter().copied());
        }
        keys
    }

    /// Evict backend retained command lists, then clear the ledger.
    pub fn release_backend(&mut self, ctx: &crate::Context) {
        self.invalidate(ctx);
    }
}

/// True when Goldy must not retain or resubmit closed command buffers.
#[inline]
pub(crate) fn cb_replay_disabled() -> bool {
    crate::validation_env::retained_cb_reuse_disabled()
}
