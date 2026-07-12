//! Helpers for relocating reuse waits and deferred host writes to the submission worker.

use crate::backend::{DeferredHostWrite, SubmitSync};
use crate::timeline::{mark_reference, Epoch, ReferenceTable};

/// Merge `from` into `into`, keeping the maximum timeline value per context.
pub fn merge_epochs(into: &mut Vec<Epoch>, from: &[Epoch]) {
    for e in from {
        if let Some(w) = into.iter_mut().find(|x| x.context == e.context) {
            w.value = w.value.max(e.value);
        } else {
            into.push(*e);
        }
    }
}

/// Accumulate reuse epochs from a parcel reference table (max per context).
pub fn merge_reference_table(into: &mut ReferenceTable, from: &ReferenceTable) {
    for (ctx, tv) in from.iter() {
        mark_reference(into, ctx, tv);
    }
}

/// Build an owned [`SubmitSync`] for one partition submit, merging cross-submit sync with
/// per-scheme extra queue waits. Host-observed waits and deferred writes attach once.
pub fn merge_submit_sync_for_partition(
    base: Option<&SubmitSync>,
    extra_queue_epochs: &[Epoch],
    host_observed: Vec<Epoch>,
    deferred_writes: Vec<DeferredHostWrite>,
) -> Option<SubmitSync> {
    let has_base = base.is_some_and(|s| !s.is_empty());
    let has_extra = !extra_queue_epochs.is_empty() || !host_observed.is_empty() || !deferred_writes.is_empty();
    if !has_base && !has_extra {
        return None;
    }
    let mut s = base.cloned().unwrap_or_default();
    s.merge_queue_waits(extra_queue_epochs);
    s.merge_host_observed_waits(&host_observed);
    if !deferred_writes.is_empty() {
        debug_assert!(
            s.deferred_host_writes.is_empty(),
            "merge_submit_sync_for_partition: host writes already attached"
        );
        s.deferred_host_writes = deferred_writes;
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DeferredHostWrite, SubmitSync};
    use crate::timeline::Epoch;
    use std::sync::Arc;

    #[test]
    fn merge_epochs_keeps_max_per_context() {
        let mut into = vec![Epoch { context: 1, value: 10 }];
        merge_epochs(
            &mut into,
            &[Epoch { context: 1, value: 8 }, Epoch { context: 2, value: 3 }],
        );
        assert_eq!(into.len(), 2);
        assert_eq!(into[0].value, 10);
        assert_eq!(into[1].value, 3);
    }

    #[test]
    fn merge_submit_sync_attaches_host_writes_once() {
        let base = SubmitSync {
            waits: vec![Epoch { context: 1, value: 4 }],
            ..Default::default()
        };
        let write = DeferredHostWrite {
            buffer: 99,
            offset: 0,
            data: Arc::from([1u8, 2, 3].as_slice()),
        };
        let merged = merge_submit_sync_for_partition(
            Some(&base),
            &[Epoch { context: 1, value: 7 }],
            vec![Epoch { context: 2, value: 5 }],
            vec![write],
        )
        .unwrap();
        assert_eq!(merged.waits[0].value, 7);
        assert_eq!(merged.host_observed_waits[0].value, 5);
        assert_eq!(merged.deferred_host_writes.len(), 1);
        assert_eq!(merged.deferred_host_writes[0].buffer, 99);

        let no_host = merge_submit_sync_for_partition(Some(&base), &[], vec![], vec![]).unwrap();
        assert!(no_host.deferred_host_writes.is_empty());
        assert!(no_host.host_observed_waits.is_empty());
    }

    #[test]
    fn merge_preserves_cpu_waits() {
        let base = SubmitSync {
            cpu_waits: vec![Epoch { context: 3, value: 9 }],
            ..Default::default()
        };
        let merged = merge_submit_sync_for_partition(Some(&base), &[], vec![], vec![]).unwrap();
        assert_eq!(merged.cpu_waits[0].value, 9);
    }

    #[test]
    fn merge_attach_host_once_then_queue_only() {
        let write = DeferredHostWrite {
            buffer: 7,
            offset: 4,
            data: Arc::from([9u8].as_slice()),
        };
        // First partition: attach queue + host sidecars.
        let first = merge_submit_sync_for_partition(
            None,
            &[Epoch { context: 1, value: 3 }],
            vec![Epoch { context: 1, value: 2 }],
            vec![write],
        )
        .unwrap();
        assert_eq!(first.waits[0].value, 3);
        assert_eq!(first.host_observed_waits.len(), 1);
        assert_eq!(first.deferred_host_writes.len(), 1);

        // Later partitions (SubmitSidecarState after attach): queue epochs only.
        let second = merge_submit_sync_for_partition(
            Some(&SubmitSync {
                waits: first.waits.clone(),
                ..Default::default()
            }),
            &[Epoch { context: 1, value: 5 }],
            vec![],
            vec![],
        )
        .unwrap();
        assert_eq!(second.waits[0].value, 5);
        assert!(second.host_observed_waits.is_empty());
        assert!(second.deferred_host_writes.is_empty());
    }

    #[test]
    fn submit_sync_waits_only_preserves_host_sidecar() {
        let sync = SubmitSync {
            prologue: Default::default(),
            waits: vec![Epoch { context: 1, value: 11 }],
            cpu_waits: vec![Epoch { context: 2, value: 22 }],
            host_observed_waits: vec![Epoch { context: 3, value: 33 }],
            deferred_host_writes: vec![DeferredHostWrite {
                buffer: 42,
                offset: 8,
                data: Arc::from([1u8, 2].as_slice()),
            }],
        };
        // Mirror `submit_sync_waits_only` from task_graph::graph (prologue stripped).
        let waits_only = SubmitSync {
            prologue: Default::default(),
            waits: sync.waits.clone(),
            cpu_waits: sync.cpu_waits.clone(),
            host_observed_waits: sync.host_observed_waits.clone(),
            deferred_host_writes: sync.deferred_host_writes.clone(),
        };
        assert!(waits_only.prologue.is_empty());
        assert_eq!(waits_only.waits[0].value, 11);
        assert_eq!(waits_only.cpu_waits[0].value, 22);
        assert_eq!(waits_only.host_observed_waits[0].value, 33);
        assert_eq!(waits_only.deferred_host_writes[0].buffer, 42);
        assert_eq!(waits_only.deferred_host_writes[0].offset, 8);
        assert_eq!(&*waits_only.deferred_host_writes[0].data, &[1u8, 2]);
    }
}
