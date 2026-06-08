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
