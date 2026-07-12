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
}

impl CbReplayState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ensure_partition_vecs(&mut self, partition_count: usize) {
        if self.partition_keys.len() != partition_count {
            self.partition_keys = vec![None; partition_count];
            self.partition_slot_keys = vec![None; partition_count];
            self.partition_last_tv = vec![None; partition_count];
        }
    }

    /// Drop cached retention keys so the next submit re-records retained partitions.
    ///
    /// Topology/structural dirtiness forces re-record even when the GPU has not
    /// retired the prior partition timeline — do not gate re-record on stale TVs.
    pub fn invalidate(&mut self) {
        for key in &mut self.partition_keys {
            *key = None;
        }
        for keys in &mut self.partition_slot_keys {
            *keys = None;
        }
        for tv in &mut self.partition_last_tv {
            *tv = None;
        }
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
        let handle = ctx.backend_handle();
        let session = ctx.submit_session();
        for key in self.all_backend_keys() {
            session.evict_retained(handle, key);
        }
        self.invalidate();
    }
}

/// True when Goldy must not retain or resubmit closed command buffers.
#[inline]
pub(crate) fn cb_replay_disabled() -> bool {
    crate::validation_env::retained_cb_reuse_disabled()
}
