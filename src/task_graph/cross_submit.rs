//! Cross-submission hazard analysis for independent [`Scheme`] / [`TaskGraph`] submits.
//!
//! Intra-submission barriers are computed by [`super::analysis`]; this module derives
//! scoped memory barriers and cross-context queue-waits from the runtime's ledger
//! (spec §5): the standing per-parcel ownership record on [`crate::parcel::ParcelStamp`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::backend::{BufferHandle, ContextHandle, SubmitSync, TextureHandle};
use crate::parcel::{InteractionEdge, InteractionRole, ParcelStamp};
use crate::task_graph::ir::{
    BarrierSet, BarrierUsage, GraphIR, NodeAccess, NodeKind, ResourceBinding, SlotUsageSet, UsageKindFlags,
};
use crate::task_graph::ResourceId;
use crate::timeline::{Epoch, ResourceSync, WRITE_KINDS_COMPUTE_TRANSFER};

/// Aggregated access for one resource within a single submission.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NetAccess {
    pub reads: bool,
    pub writes: bool,
    /// Barrier-oriented usage kinds (may map render-pass SRV reads to compute).
    pub read_kinds: UsageKindFlags,
    /// Actual pipeline categories that performed reads (render vs compute).
    pub read_pipeline_kinds: UsageKindFlags,
    pub write_kinds: UsageKindFlags,
}

impl NetAccess {
    fn absorb(&mut self, access: NodeAccess, barrier_kind: UsageKindFlags, pipeline_kind: UsageKindFlags) {
        if access.reads() {
            self.reads = true;
            self.read_kinds |= barrier_kind;
            self.read_pipeline_kinds |= pipeline_kind;
        }
        if access.writes() {
            self.writes = true;
            self.write_kinds |= barrier_kind;
        }
    }
}

/// Canonical resource key for per-unit ledger (whole buffer, buffer range, or texture).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKey {
    Buffer(BufferHandle),
    BufferRange {
        parent: BufferHandle,
        offset: u64,
        len: u64,
    },
    Texture(TextureHandle),
}

impl ResourceKey {
    pub fn from_resource_id(id: ResourceId) -> Option<Self> {
        match id {
            ResourceId::Buffer(h) => Some(Self::Buffer(h)),
            ResourceId::BufferRange { parent, offset, len } => Some(Self::BufferRange { parent, offset, len }),
            ResourceId::Texture(h) => Some(Self::Texture(h)),
            ResourceId::TransientBuffer(_) | ResourceId::TransientTexture(_) => None,
            // RenderTargets are scheme-owned leases (Lease<LeaseRenderTarget> is an opaque
            // index into the owning scheme's rt_leases Vec, with no public Clone or borrow-out
            // path). They can never appear in a *different* scheme's IR, so cross-scheme
            // hazard tracking at the RT level is structurally impossible and the exclusion here
            // is safe. SwapchainOutput and PresentLease are similarly owned by the surface
            // infrastructure and not shared as ledger-tracked resources.
            ResourceId::RenderTarget(_) | ResourceId::SwapchainOutput | ResourceId::PresentLease(_) => None,
        }
    }
}

/// FxHash-backed map keyed by [`ResourceKey`] (hot-path lookups use integer handles).
pub type ResourceKeyMap<V> = FxHashMap<ResourceKey, V>;

/// True when two ledger keys refer to overlapping GPU memory for hazard analysis.
pub(crate) fn resource_keys_alias(a: ResourceKey, b: ResourceKey) -> bool {
    use crate::task_graph::analysis::ranges_overlap;
    match (a, b) {
        (ResourceKey::Buffer(x), ResourceKey::Buffer(y)) => x == y,
        (ResourceKey::Buffer(h), ResourceKey::BufferRange { parent, .. })
        | (ResourceKey::BufferRange { parent, .. }, ResourceKey::Buffer(h)) => h == parent,
        (
            ResourceKey::BufferRange {
                parent: p1,
                offset: o1,
                len: l1,
            },
            ResourceKey::BufferRange {
                parent: p2,
                offset: o2,
                len: l2,
            },
        ) => p1 == p2 && ranges_overlap(o1, l1, o2, l2),
        (ResourceKey::Texture(x), ResourceKey::Texture(y)) => x == y,
        _ => false,
    }
}

/// Merge all ledger entries that alias `key` when there is no exact ledger key.
///
/// Returns `None` when `key` is present exactly (callers should use [`LedgerSnapshot::get`])
/// or when no aliasing entries exist.
fn merge_aliased_ledger_entries(ledger: &LedgerSnapshot, key: ResourceKey) -> Option<LedgerEntry> {
    if ledger.contains_key(&key) {
        return None;
    }
    let _tz = crate::tracy_zone!("goldy.cross_sync.compute_sync.alias_merge.scan");
    let mut merged: Option<LedgerEntry> = None;
    for (ledger_key, entry) in ledger {
        if !resource_keys_alias(*ledger_key, key) {
            continue;
        }
        match &mut merged {
            None => merged = Some(entry.clone()),
            Some(m) => {
                for (ctx, tv) in entry.sync.last_write.iter() {
                    let kinds = entry
                        .sync
                        .last_write_kinds
                        .get(ctx)
                        .unwrap_or(WRITE_KINDS_COMPUTE_TRANSFER);
                    m.sync.record_write(ctx, tv, kinds);
                }
                for (ctx, tv) in entry.sync.last_reads.iter() {
                    m.sync.record_read(ctx, tv);
                }
                for (ctx, tv) in entry.sync.fifo_ordered_reads.iter() {
                    m.sync.mark_fifo_ordered_read(ctx, tv);
                }
            }
        }
    }
    merged
}

fn merge_context_wait(waits: &mut Vec<(ContextHandle, u64)>, ctx: ContextHandle, tv: u64) {
    for entry in waits.iter_mut() {
        if entry.0 == ctx {
            entry.1 = entry.1.max(tv);
            return;
        }
    }
    waits.push((ctx, tv));
}

fn sync_has_foreign_context(sync: &ResourceSync, submitting_ctx: ContextHandle) -> bool {
    sync.last_write.iter().any(|(ctx, _)| ctx != submitting_ctx)
        || sync.last_reads.iter().any(|(ctx, _)| ctx != submitting_ctx)
}

fn merge_device_queue_wait(device_waits: &mut Vec<u64>, tv: u64) {
    match device_waits.as_slice() {
        [] => device_waits.push(tv),
        [max] if tv <= *max => {}
        _ => {
            let max = device_waits.iter().copied().max().unwrap_or(0).max(tv);
            device_waits.clear();
            device_waits.push(max);
        }
    }
}

/// Present-copy reads retire on the device fence; compute writes run on a per-context queue.
fn emit_present_easement_device_wait(
    sync: &ResourceSync,
    submitting_ctx: ContextHandle,
    device_queue_waits: &mut Vec<u64>,
) {
    if let Some(fifo_tv) = sync.fifo_ordered_reads.get(submitting_ctx) {
        merge_device_queue_wait(device_queue_waits, fifo_tv);
    }
}

/// Present blits read textures as `TRANSFER` on the device queue; the next compute write on a
/// per-context queue needs an explicit transfer→write prologue. A queue wait on the device
/// fence orders execution but does not replace that layout/access transition — without it D3D12
/// reports `0x887A002B` (write to a resource still in copy-source / read-only state).
fn emit_present_easement_texture_war_barrier(
    key: ResourceKey,
    access: NetAccess,
    sync: &ResourceSync,
    submitting_ctx: ContextHandle,
    prologue: &mut BarrierSet,
) {
    if !matches!(key, ResourceKey::Texture(_)) {
        return;
    }
    if sync.fifo_ordered_reads.get(submitting_ctx).is_none() {
        // #region agent log
        if let ResourceKey::Texture(tex) = key {
            crate::debug_session_log::write(
                "H1",
                "cross_submit.rs:emit_present_easement_texture_war_barrier",
                "present easement texture barrier skipped: no fifo_ordered_read",
                &format!(
                    r#"{{"texture":{},"submitting_ctx":{},"fifo_ordered_reads":{:?}}}"#,
                    tex,
                    submitting_ctx,
                    sync.fifo_ordered_reads
                ),
            );
        }
        // #endregion
        return;
    }
    let usage = BarrierUsage {
        src: {
            let mut s = SlotUsageSet::default();
            s.merge(NodeAccess::Read, UsageKindFlags::TRANSFER);
            s
        },
        dst: {
            let mut d = SlotUsageSet::default();
            d.merge(NodeAccess::Write, access.write_kinds);
            d
        },
    };
    merge_barrier(prologue, key, usage);
    // #region agent log
    if let ResourceKey::Texture(tex) = key {
        crate::debug_session_log::write(
            "H1",
            "cross_submit.rs:emit_present_easement_texture_war_barrier",
            "present easement transfer→write texture prologue emitted",
            &format!(
                r#"{{"texture":{},"submitting_ctx":{},"fifo_tv":{},"write_kinds":"{:?}"}}"#,
                tex,
                submitting_ctx,
                sync.fifo_ordered_reads.get(submitting_ctx).unwrap_or(0),
                access.write_kinds
            ),
        );
    }
    // #endregion
}

/// Present easement barriers must be replayed on every retained resubmit: the baked CB
/// only carries record-time prologue, while present-copy reads advance on the device queue
/// each frame.
pub(crate) fn present_easement_live_prologue(
    prologue: &BarrierSet,
    device_queue_waits: &[u64],
) -> BarrierSet {
    if device_queue_waits.is_empty() {
        return BarrierSet::default();
    }
    let mut out = BarrierSet::default();
    for (h, usage) in &prologue.textures {
        if !usage.src.kinds.contains(UsageKindFlags::TRANSFER) {
            continue;
        }
        if !usage.dst.access.writes() {
            continue;
        }
        // Recover whatever write kind the baked entry actually recorded — not just COMPUTE.
        // The prior version filtered to `dst.kinds.contains(COMPUTE)`, so any present-easement
        // write that (also or only) came from e.g. a render pass silently dropped its live
        // transfer→write replay on retained resubmit, leaving the resource in the stale
        // record-time state instead of the actual post-present TRANSFER state.
        out.textures.push((
            *h,
            BarrierUsage {
                src: {
                    let mut s = SlotUsageSet::default();
                    s.merge(NodeAccess::Read, UsageKindFlags::TRANSFER);
                    s
                },
                dst: {
                    let mut d = SlotUsageSet::default();
                    d.merge(NodeAccess::Write, usage.dst.kinds);
                    d
                },
            },
        ));
    }
    out
}

/// True when any written resource in this partition still has a registered reader edge.
/// While readers remain on the interaction set, same-context WAR is covered by FIFO queue
/// ordering and retained resubmit must not promote baked WAW prologue to a live queue wait.
pub(crate) fn partition_has_active_read_edges(
    net: &ResourceKeyMap<NetAccess>,
    resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
) -> bool {
    net.iter()
        .filter(|(_, access)| access.writes)
        .filter_map(|(key, _)| resource_stamps.get(key))
        .any(|stamp| {
            stamp
                .interaction_set
                .lock()
                .unwrap()
                .iter()
                .any(|e| e.role == InteractionRole::Reads)
        })
}

/// True when the parcel's interaction set has both foreign readers and writers (cross-scheme WAR).
fn stamp_has_cross_scheme_war(stamp: &ParcelStamp) -> bool {
    let edges = stamp.interaction_set.lock().unwrap();
    let has_read = edges.iter().any(|e| e.role == InteractionRole::Reads);
    let has_write = edges.iter().any(|e| e.role == InteractionRole::Writes);
    if !(has_read && has_write) {
        return false;
    }
    let schemes: FxHashSet<u64> = edges.iter().map(|e| e.scheme_id).collect();
    schemes.len() > 1
}

/// Bake a transfer→compute WAR prologue for same-context cross-scheme retained replay.
fn emit_cross_scheme_same_context_war_prologue(
    key: ResourceKey,
    access: NetAccess,
    sync: &ResourceSync,
    submitting_ctx: ContextHandle,
    prologue: &mut BarrierSet,
    stamp: &ParcelStamp,
) {
    if !access.writes || !stamp_has_cross_scheme_war(stamp) {
        return;
    }
    for (ctx, _tv) in sync.last_reads.iter() {
        if ctx != submitting_ctx {
            continue;
        }
        let usage = BarrierUsage {
            src: {
                let mut s = SlotUsageSet::default();
                s.merge(NodeAccess::Read, UsageKindFlags::TRANSFER);
                s
            },
            dst: {
                let mut d = SlotUsageSet::default();
                d.merge(NodeAccess::Write, access.write_kinds);
                d
            },
        };
        merge_barrier(prologue, key, usage);
    }
}

fn apply_cross_submit_hazards_single_context(
    key: ResourceKey,
    access: NetAccess,
    sync: &ResourceSync,
    submitting_ctx: ContextHandle,
    prologue: &mut BarrierSet,
    _context_waits: &mut Vec<(ContextHandle, u64)>,
    device_queue_waits: &mut Vec<u64>,
    stamp: Option<&ParcelStamp>,
) {
    if access.reads {
        if sync.last_write.get(submitting_ctx).is_some() {
            let prev_write_kinds = UsageKindFlags::from_bits_truncate(
                sync.last_write_kinds
                    .get(submitting_ctx)
                    .unwrap_or(WRITE_KINDS_COMPUTE_TRANSFER),
            );
            let usage = BarrierUsage {
                src: {
                    let mut s = SlotUsageSet::default();
                    s.merge(NodeAccess::Write, prev_write_kinds);
                    s
                },
                dst: {
                    let mut d = SlotUsageSet::default();
                    d.merge(NodeAccess::Read, access.read_kinds | access.read_pipeline_kinds);
                    d
                },
            };
            merge_barrier(prologue, key, usage);
        }
    }

    if access.writes {
        if sync.last_write.get(submitting_ctx).is_some() {
            let prev_write_kinds = UsageKindFlags::from_bits_truncate(
                sync.last_write_kinds
                    .get(submitting_ctx)
                    .unwrap_or(WRITE_KINDS_COMPUTE_TRANSFER),
            );
            let usage = BarrierUsage {
                src: {
                    let mut s = SlotUsageSet::default();
                    s.merge(NodeAccess::Write, prev_write_kinds);
                    s
                },
                dst: {
                    let mut d = SlotUsageSet::default();
                    d.merge(NodeAccess::Write, access.write_kinds);
                    d
                },
            };
            merge_barrier(prologue, key, usage);
        }
        if let Some(stamp) = stamp {
            emit_cross_scheme_same_context_war_prologue(key, access, sync, submitting_ctx, prologue, stamp);
        }
        emit_present_easement_texture_war_barrier(key, access, sync, submitting_ctx, prologue);
        emit_present_easement_device_wait(sync, submitting_ctx, device_queue_waits);
    }
}

fn apply_cross_submit_hazards_for_resource(
    key: ResourceKey,
    access: NetAccess,
    sync: &ResourceSync,
    submitting_ctx: ContextHandle,
    prologue: &mut BarrierSet,
    context_waits: &mut Vec<(ContextHandle, u64)>,
    device_queue_waits: &mut Vec<u64>,
    stamp: Option<&ParcelStamp>,
) {
    if !sync_has_foreign_context(sync, submitting_ctx) {
        apply_cross_submit_hazards_single_context(
            key,
            access,
            sync,
            submitting_ctx,
            prologue,
            context_waits,
            device_queue_waits,
            stamp,
        );
        return;
    }

    // RAW: this reads -> hazard vs last_write
    if access.reads {
        for (ctx, tv) in sync.last_write.iter() {
            if ctx == submitting_ctx {
                let prev_write_kinds = UsageKindFlags::from_bits_truncate(
                    sync.last_write_kinds.get(ctx).unwrap_or(WRITE_KINDS_COMPUTE_TRANSFER),
                );
                let usage = BarrierUsage {
                    src: {
                        let mut s = SlotUsageSet::default();
                        s.merge(NodeAccess::Write, prev_write_kinds);
                        s
                    },
                    dst: {
                        let mut d = SlotUsageSet::default();
                        d.merge(NodeAccess::Read, access.read_kinds | access.read_pipeline_kinds);
                        d
                    },
                };
                merge_barrier(prologue, key, usage);
            } else {
                merge_context_wait(context_waits, ctx, tv);
            }
        }
    }

    // WAW + WAR: this writes -> hazard vs last_write and last_reads
    if access.writes {
        for (ctx, tv) in sync.last_write.iter() {
            if ctx == submitting_ctx {
                let prev_write_kinds = UsageKindFlags::from_bits_truncate(
                    sync.last_write_kinds.get(ctx).unwrap_or(WRITE_KINDS_COMPUTE_TRANSFER),
                );
                let usage = BarrierUsage {
                    src: {
                        let mut s = SlotUsageSet::default();
                        s.merge(NodeAccess::Write, prev_write_kinds);
                        s
                    },
                    dst: {
                        let mut d = SlotUsageSet::default();
                        d.merge(NodeAccess::Write, access.write_kinds);
                        d
                    },
                };
                merge_barrier(prologue, key, usage);
            } else {
                merge_context_wait(context_waits, ctx, tv);
            }
        }
        for (ctx, tv) in sync.last_reads.iter() {
            if ctx != submitting_ctx {
                merge_context_wait(context_waits, ctx, tv);
            }
        }
        emit_present_easement_texture_war_barrier(key, access, sync, submitting_ctx, prologue);
        emit_present_easement_device_wait(sync, submitting_ctx, device_queue_waits);
    }
}

/// Snapshot of one resource's epoch ledger at submit time.
#[derive(Debug, Clone, Default)]
pub struct LedgerEntry {
    pub sync: ResourceSync,
}

/// Per-resource ledger keyed by [`ResourceKey`], populated from parcel stamps.
pub type LedgerSnapshot = ResourceKeyMap<LedgerEntry>;

/// Result of cross-submission hazard analysis for one submit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrossSubmitSync {
    /// Same-context memory barriers prepended to the submission command stream.
    pub prologue: BarrierSet,
    /// Cross-context GPU queue-waits (one per producer context, max tv).
    pub waits: Vec<Epoch>,
    /// Device-timeline waits on the shared device fence (present easement).
    pub device_queue_waits: Vec<u64>,
}

impl CrossSubmitSync {
    pub fn is_empty(&self) -> bool {
        self.prologue.is_empty() && self.waits.is_empty() && self.device_queue_waits.is_empty()
    }
}

fn node_usage_kind(node: &super::ir::TaskNode) -> UsageKindFlags {
    match &node.kind {
        NodeKind::Dispatch { .. } => UsageKindFlags::COMPUTE,
        NodeKind::RenderPass { .. } => UsageKindFlags::RENDER,
        NodeKind::ClearBuffer { .. }
        | NodeKind::WriteBuffer { .. }
        | NodeKind::CopyBuffer { .. }
        | NodeKind::CopyBufferToTexture { .. }
        | NodeKind::WriteTexture { .. }
        | NodeKind::WriteTextureRegion { .. }
        | NodeKind::CopyTexture { .. }
        | NodeKind::CopyRenderTarget { .. } => UsageKindFlags::TRANSFER,
        NodeKind::GrantRead { .. } | NodeKind::GrantPresent { .. } => UsageKindFlags::empty(),
    }
}

fn barrier_usage_kind_for_binding(
    resource: ResourceId,
    access: NodeAccess,
    node: &super::ir::TaskNode,
) -> UsageKindFlags {
    let kind = node_usage_kind(node);
    let shader_read = !access.writes();
    let non_attachment = matches!(
        resource,
        ResourceId::Buffer(_)
            | ResourceId::BufferRange { .. }
            | ResourceId::TransientBuffer(_)
            | ResourceId::Texture(_)
            | ResourceId::TransientTexture(_)
    );
    if kind.contains(UsageKindFlags::RENDER) && shader_read && non_attachment {
        UsageKindFlags::COMPUTE
    } else {
        kind
    }
}

/// Union each resource's access across all nodes in `ir`.
pub fn net_access_per_resource(ir: &GraphIR) -> ResourceKeyMap<NetAccess> {
    let mut net = ResourceKeyMap::default();
    for node in &ir.nodes {
        absorb_node_net_access(&mut net, node);
    }
    net
}

/// Union each resource's access across the nodes in `waves` (one submit partition).
pub fn net_access_for_waves_into(out: &mut ResourceKeyMap<NetAccess>, ir: &GraphIR, waves: &[super::ir::Wave]) {
    {
        let _tz = crate::tracy_zone!("goldy.cross_sync.net_access.clear");
        out.clear();
    }
    {
        let _tz = crate::tracy_zone!("goldy.cross_sync.net_access.absorb");
        for wave in waves {
            for &node_idx in &wave.node_indices {
                absorb_node_net_access(out, &ir.nodes[node_idx]);
            }
        }
    }
}

fn absorb_node_net_access(net: &mut ResourceKeyMap<NetAccess>, node: &super::ir::TaskNode) {
    if matches!(node.kind, NodeKind::GrantRead { .. } | NodeKind::GrantPresent { .. }) {
        return;
    }
    for binding in &node.bindings {
        let Some(key) = ResourceKey::from_resource_id(binding.resource) else {
            continue;
        };
        let barrier_kind = barrier_usage_kind_for_binding(binding.resource, binding.access, node);
        let pipeline_kind = node_usage_kind(node);
        net.entry(key)
            .or_default()
            .absorb(binding.access, barrier_kind, pipeline_kind);
    }
}

fn merge_barrier(barriers: &mut BarrierSet, key: ResourceKey, usage: BarrierUsage) {
    match key {
        ResourceKey::Buffer(h) | ResourceKey::BufferRange { parent: h, .. } => {
            if let Some((_, existing)) = barriers.buffers.iter_mut().find(|(bh, _)| *bh == h) {
                existing.src.merge(
                    if usage.src.access.writes() {
                        NodeAccess::Write
                    } else {
                        NodeAccess::Read
                    },
                    usage.src.kinds,
                );
                existing.dst.merge(
                    if usage.dst.access.writes() {
                        NodeAccess::Write
                    } else {
                        NodeAccess::Read
                    },
                    usage.dst.kinds,
                );
            } else {
                barriers.buffers.push((h, usage));
            }
        }
        ResourceKey::Texture(h) => {
            if let Some((_, existing)) = barriers.textures.iter_mut().find(|(th, _)| *th == h) {
                existing.src.merge(
                    if usage.src.access.writes() {
                        NodeAccess::Write
                    } else {
                        NodeAccess::Read
                    },
                    usage.src.kinds,
                );
                existing.dst.merge(
                    if usage.dst.access.writes() {
                        NodeAccess::Write
                    } else {
                        NodeAccess::Read
                    },
                    usage.dst.kinds,
                );
            } else {
                barriers.textures.push((h, usage));
            }
        }
    }
}

/// Derive cross-submission sync from this submission's net access and the resource ledger.
pub fn compute_cross_submit_sync_into(
    prologue: &mut BarrierSet,
    waits: &mut Vec<Epoch>,
    device_queue_waits: &mut Vec<u64>,
    context_waits: &mut Vec<(ContextHandle, u64)>,
    net: &ResourceKeyMap<NetAccess>,
    ledger: &LedgerSnapshot,
    submitting_ctx: ContextHandle,
    resource_stamps: Option<&ResourceKeyMap<Arc<ParcelStamp>>>,
) {
    {
        let _tz = crate::tracy_zone!("goldy.cross_sync.compute_sync.reset");
        prologue.buffers.clear();
        prologue.textures.clear();
        prologue.transient_ids.clear();
        context_waits.clear();
        device_queue_waits.clear();
    }

    {
        let _tz = crate::tracy_zone!("goldy.cross_sync.compute_sync.hazards");
        for (key, access) in net {
            let stamp = resource_stamps.and_then(|stamps| stamps.get(key).map(|s| s.as_ref()));
            if let Some(entry) = ledger.get(key) {
                apply_cross_submit_hazards_for_resource(
                    *key,
                    *access,
                    &entry.sync,
                    submitting_ctx,
                    prologue,
                    context_waits,
                    device_queue_waits,
                    stamp,
                );
            } else if let Some(entry) = merge_aliased_ledger_entries(ledger, *key) {
                let _tz = crate::tracy_zone!("goldy.cross_sync.compute_sync.hazards.alias_merge");
                apply_cross_submit_hazards_for_resource(
                    *key,
                    *access,
                    &entry.sync,
                    submitting_ctx,
                    prologue,
                    context_waits,
                    device_queue_waits,
                    stamp,
                );
            }
        }
    }

    {
        let _tz = crate::tracy_zone!("goldy.cross_sync.compute_sync.finalize");
        waits.clear();
        waits.extend(context_waits.iter().map(|&(context, value)| Epoch { context, value }));
        waits.sort_by_key(|e| (e.context, e.value));
    }
}

/// Derive cross-submission sync from this submission's net access and the resource ledger.
#[allow(
    dead_code,
    reason = "allocating convenience wrapper; hot path uses CrossSubmitScratch"
)]
pub fn compute_cross_submit_sync(
    net: &ResourceKeyMap<NetAccess>,
    ledger: &LedgerSnapshot,
    submitting_ctx: crate::backend::ContextHandle,
) -> CrossSubmitSync {
    let mut result = CrossSubmitSync::default();
    let mut context_waits = Vec::new();
    compute_cross_submit_sync_into(
        &mut result.prologue,
        &mut result.waits,
        &mut result.device_queue_waits,
        &mut context_waits,
        net,
        ledger,
        submitting_ctx,
        None,
    );
    result
}

/// Registration-time index of buffer-scoped [`ResourceKey`]s for alias discovery.
///
/// Built once when parcels register; avoids scanning all `resource_stamps` on every
/// partition when building a ledger snapshot (buffer/range aliasing is parent-local).
#[derive(Debug, Default)]
pub(crate) struct BufferStampIndex {
    by_parent: FxHashMap<BufferHandle, Vec<ResourceKey>>,
    any_aliased_buffers: bool,
}

impl BufferStampIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a newly registered buffer or buffer-range key.
    pub fn register(&mut self, key: ResourceKey) {
        let parent = match key {
            ResourceKey::Buffer(h) => h,
            ResourceKey::BufferRange { parent, .. } => parent,
            ResourceKey::Texture(_) => return,
        };
        let bucket = self.by_parent.entry(parent).or_default();
        if !bucket.contains(&key) {
            bucket.push(key);
            if bucket.len() > 1 {
                self.any_aliased_buffers = true;
            }
        }
    }

    /// True when at least one buffer parent has multiple registered keys (range aliasing).
    pub fn any_aliased_buffers(&self) -> bool {
        self.any_aliased_buffers
    }

    /// Candidate keys that might alias `query_key` (same buffer parent only).
    pub fn candidates_for(&self, query_key: ResourceKey) -> Option<&[ResourceKey]> {
        let parent = match query_key {
            ResourceKey::Buffer(h) => h,
            ResourceKey::BufferRange { parent, .. } => parent,
            ResourceKey::Texture(_) => return None,
        };
        self.by_parent.get(&parent).map(|v| v.as_slice())
    }
}

fn insert_ledger_entry(out: &mut LedgerSnapshot, stamp_key: ResourceKey, stamp: &Arc<ParcelStamp>) {
    out.entry(stamp_key).or_insert_with(|| LedgerEntry {
        sync: stamp.sync.lock().clone(),
    });
}

fn collect_ledger_aliases_for_key(
    out: &mut LedgerSnapshot,
    query_key: ResourceKey,
    resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
    buffer_index: Option<&BufferStampIndex>,
) {
    if !matches!(query_key, ResourceKey::Buffer(_) | ResourceKey::BufferRange { .. }) {
        return;
    }
    if let Some(candidates) = buffer_index.and_then(|index| index.candidates_for(query_key)) {
        for &stamp_key in candidates {
            if stamp_key == query_key {
                continue;
            }
            if resource_keys_alias(stamp_key, query_key) {
                if let Some(stamp) = resource_stamps.get(&stamp_key) {
                    insert_ledger_entry(out, stamp_key, stamp);
                }
            }
        }
        return;
    }
    let _tz = crate::tracy_zone!("goldy.cross_sync.ledger_snapshot.alias_scan");
    for (&stamp_key, stamp) in resource_stamps {
        if stamp_key == query_key {
            continue;
        }
        if resource_keys_alias(stamp_key, query_key) {
            insert_ledger_entry(out, stamp_key, stamp);
        }
    }
}

/// Build a ledger snapshot for cross-submit planning: one entry per [`NetAccess`] key
/// in `net` that has a registered stamp, plus any registered stamps whose keys alias
/// a net key (partitioned buffer ranges).
pub fn build_ledger_snapshot_for_net_into(
    out: &mut LedgerSnapshot,
    net: &ResourceKeyMap<NetAccess>,
    resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
    buffer_index: Option<&BufferStampIndex>,
) {
    {
        let _tz = crate::tracy_zone!("goldy.cross_sync.ledger_snapshot.clear");
        out.clear();
    }
    {
        let _tz = crate::tracy_zone!("goldy.cross_sync.ledger_snapshot.exact");
        for &query_key in net.keys() {
            if let Some(stamp) = resource_stamps.get(&query_key) {
                insert_ledger_entry(out, query_key, stamp);
            }
        }
    }
    {
        let _tz = crate::tracy_zone!("goldy.cross_sync.ledger_snapshot.alias");
        let skip_alias = buffer_index.is_some_and(|index| !index.any_aliased_buffers());
        if !skip_alias {
            for &query_key in net.keys() {
                collect_ledger_aliases_for_key(out, query_key, resource_stamps, buffer_index);
            }
        }
    }
}

/// Build a ledger snapshot from registered stamp bindings.
pub fn build_ledger_snapshot_into(out: &mut LedgerSnapshot, stamps: &[(ResourceKey, Arc<ParcelStamp>)]) {
    out.clear();
    for (key, stamp) in stamps {
        out.entry(*key).or_insert_with(|| LedgerEntry {
            sync: stamp.sync.lock().clone(),
        });
    }
}

/// Build a ledger snapshot from registered stamp bindings.
#[allow(
    dead_code,
    reason = "allocating convenience wrapper; hot path uses CrossSubmitScratch"
)]
pub fn build_ledger_snapshot(stamps: &[(ResourceKey, Arc<ParcelStamp>)]) -> LedgerSnapshot {
    let mut ledger = LedgerSnapshot::default();
    build_ledger_snapshot_into(&mut ledger, stamps);
    ledger
}

/// Collect unique resource keys from IR that have registered stamps.
pub fn resource_stamps_from_ir_into(
    out: &mut Vec<(ResourceKey, Arc<ParcelStamp>)>,
    seen: &mut FxHashSet<ResourceKey>,
    ir: &GraphIR,
    resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
) {
    out.clear();
    seen.clear();
    for node in &ir.nodes {
        for ResourceBinding { resource, .. } in &node.bindings {
            let Some(key) = ResourceKey::from_resource_id(*resource) else {
                continue;
            };
            if !seen.insert(key) {
                continue;
            }
            if let Some(stamp) = resource_stamps.get(&key) {
                out.push((key, Arc::clone(stamp)));
            }
        }
    }
}

/// Collect unique resource keys from IR that have registered stamps.
#[allow(
    dead_code,
    reason = "allocating convenience wrapper; hot path uses CrossSubmitScratch"
)]
pub fn resource_stamps_from_ir(
    ir: &GraphIR,
    resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
) -> Vec<(ResourceKey, Arc<ParcelStamp>)> {
    let mut seen = FxHashSet::default();
    let mut out = Vec::new();
    resource_stamps_from_ir_into(&mut out, &mut seen, ir, resource_stamps);
    out
}

/// Reusable scratch for [`CrossSubmitScratch::plan`] — cleared and refilled per partition.
///
/// Retains container capacity across frames so steady-state cross-submit planning avoids
/// repeated HashMap/Vec/HashSet allocations.
pub(crate) struct CrossSubmitScratch {
    net: ResourceKeyMap<NetAccess>,
    ledger: LedgerSnapshot,
    context_waits: Vec<(ContextHandle, u64)>,
    submit_sync: SubmitSync,
}

impl Default for CrossSubmitScratch {
    fn default() -> Self {
        let mut net = ResourceKeyMap::default();
        net.reserve(32);
        let mut ledger = ResourceKeyMap::default();
        ledger.reserve(32);
        Self {
            net,
            ledger,
            context_waits: Vec::with_capacity(4),
            submit_sync: SubmitSync::default(),
        }
    }
}

impl CrossSubmitScratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Net access for the waves last passed to [`Self::plan`].
    ///
    /// Stale (or empty) when `plan` was not called for the current partition — callers must
    /// only rely on this when they know the underlying `resource_stamps` is non-empty, since
    /// in that case a stale/empty `net` is harmless (lookups against an empty map are no-ops).
    pub fn net(&self) -> &ResourceKeyMap<NetAccess> {
        &self.net
    }

    fn clear(&mut self) {
        self.net.clear();
        self.ledger.clear();
        self.context_waits.clear();
        self.submit_sync.prologue.buffers.clear();
        self.submit_sync.prologue.textures.clear();
        self.submit_sync.prologue.transient_ids.clear();
        self.submit_sync.waits.clear();
        self.submit_sync.device_queue_waits.clear();
    }

    /// `cached_net` — when `Some`, skips the wave walk and reuses a prior partition net
    /// snapshot (valid when the partition retention fingerprint is unchanged).
    pub fn plan(
        &mut self,
        ir: &GraphIR,
        resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
        buffer_index: Option<&BufferStampIndex>,
        submitting_ctx: ContextHandle,
        waves: &[super::ir::Wave],
        cached_net: Option<ResourceKeyMap<NetAccess>>,
    ) -> &SubmitSync {
        self.clear();
        if let Some(net) = cached_net {
            let _tz = crate::tracy_zone!("goldy.cross_sync.net_access.cached");
            self.net = net;
        } else {
            let _tz = crate::tracy_zone!("goldy.cross_sync.net_access");
            net_access_for_waves_into(&mut self.net, ir, waves);
        }
        {
            let _tz = crate::tracy_zone!("goldy.cross_sync.ledger_snapshot");
            build_ledger_snapshot_for_net_into(&mut self.ledger, &self.net, resource_stamps, buffer_index);
        }
        {
            let _tz = crate::tracy_zone!("goldy.cross_sync.compute_sync");
            compute_cross_submit_sync_into(
                &mut self.submit_sync.prologue,
                &mut self.submit_sync.waits,
                &mut self.submit_sync.device_queue_waits,
                &mut self.context_waits,
                &self.net,
                &self.ledger,
                submitting_ctx,
                Some(resource_stamps),
            );
        }
        // #region agent log
        {
            let s = &self.submit_sync;
            if !s.device_queue_waits.is_empty() || !s.prologue.textures.is_empty() {
                let tex_prologue: Vec<String> = s
                    .prologue
                    .textures
                    .iter()
                    .map(|(h, u)| {
                        format!(
                            "{{\"handle\":{},\"src\":\"{:?}\",\"dst\":\"{:?}\"}}",
                            h, u.src.kinds, u.dst.kinds
                        )
                    })
                    .collect();
                crate::debug_session_log::write(
                    "H1-H4",
                    "cross_submit.rs:CrossSubmitScratch::plan",
                    "cross-submit plan with present-easement or texture prologue",
                    &format!(
                        r#"{{"submitting_ctx":{},"device_queue_waits":{:?},"waits_len":{},"prologue_textures":[{}],"prologue_buffers":{}}}"#,
                        submitting_ctx,
                        s.device_queue_waits,
                        s.waits.len(),
                        tex_prologue.join(","),
                        s.prologue.buffers.len()
                    ),
                );
                // H15: duplicate texture barriers (post H6/7/8 revert baseline — stale WAW + present easement)
                let mut per_tex: FxHashMap<u64, usize> = FxHashMap::default();
                for (h, _) in &s.prologue.textures {
                    *per_tex.entry(*h).or_insert(0) += 1;
                }
                for (h, count) in per_tex {
                    if count > 1 {
                        let entries: Vec<String> = s
                            .prologue
                            .textures
                            .iter()
                            .filter(|(tex, _)| *tex == h)
                            .map(|(_, u)| format!("{{\"src\":\"{:?}\",\"dst\":\"{:?}\"}}", u.src.kinds, u.dst.kinds))
                            .collect();
                        crate::debug_session_log::write(
                            "H15",
                            "cross_submit.rs:CrossSubmitScratch::plan",
                            "duplicate texture barriers in prologue",
                            &format!(
                                r#"{{"texture":{},"count":{},"barriers":[{}]}}"#,
                                h,
                                count,
                                entries.join(",")
                            ),
                        );
                    }
                }
            }
        }
        // #endregion
        &self.submit_sync
    }

    /// Like [`Self::plan`], but returns owned copies so callers can store `net` without
    /// holding a borrow of `submit_sync`.
    pub fn plan_owned(
        &mut self,
        ir: &GraphIR,
        resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
        buffer_index: Option<&BufferStampIndex>,
        submitting_ctx: ContextHandle,
        waves: &[super::ir::Wave],
        cached_net: Option<ResourceKeyMap<NetAccess>>,
    ) -> Option<(SubmitSync, ResourceKeyMap<NetAccess>)> {
        if resource_stamps.is_empty() {
            return None;
        }
        self.plan(ir, resource_stamps, buffer_index, submitting_ctx, waves, cached_net);
        Some((self.submit_sync.clone(), self.net.clone()))
    }
}

/// After a successful submit, record this submission's access on each touched stamp.
pub fn apply_resource_sync_updates(
    net: &ResourceKeyMap<NetAccess>,
    resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
    ctx: crate::backend::ContextHandle,
    tv: u64,
) {
    for (key, access) in net {
        let Some(stamp) = resource_stamps.get(key) else {
            continue;
        };
        let mut sync = stamp.sync.lock();
        if access.writes {
            sync.record_write(ctx, tv, access.write_kinds.bits());
        }
        if access.reads {
            sync.record_read(ctx, tv);
        }
    }
}

/// Legacy post-submit stamp: mark merged epoch for every registered stamp target.
pub fn apply_stamp_targets_legacy(targets: &[Arc<ParcelStamp>], ctx: crate::backend::ContextHandle, tv: u64) {
    for stamp in targets {
        let mut sync = stamp.sync.lock();
        sync.record_any(ctx, tv);
    }
}

/// True when every buffer/texture entry in `cmd` is already covered by `prologue`
/// with identical [`BarrierUsage`]. Used to skip intra-graph wave barriers that
/// duplicate the cross-submit prologue prepended by [`prepend_prologue`] /
/// [`super::graph::graph_commands_with_sync_prologue`] — log evidence (H19) showed
/// copy partitions emitting two COMPUTE→TRANSFER barriers on texture 1 per frame.
pub(crate) fn resource_barrier_redundant_with_prologue(
    cmd_buffers: &[(BufferHandle, BarrierUsage)],
    cmd_textures: &[(TextureHandle, BarrierUsage)],
    prologue: &BarrierSet,
) -> bool {
    if cmd_buffers.is_empty() && cmd_textures.is_empty() {
        return false;
    }
    cmd_buffers
        .iter()
        .all(|(h, u)| prologue.buffers.iter().any(|(ph, pu)| ph == h && pu == u))
        && cmd_textures
            .iter()
            .all(|(h, u)| prologue.textures.iter().any(|(ph, pu)| ph == h && pu == u))
}

/// Prepend a cross-submission prologue barrier to a command slice.
pub fn prepend_prologue(
    commands: &[crate::backend::GpuCommand],
    prologue: &BarrierSet,
) -> Vec<crate::backend::GpuCommand> {
    if prologue.is_empty() {
        return commands.to_vec();
    }
    let mut out = Vec::with_capacity(1 + commands.len());
    out.push(crate::backend::GpuCommand::ResourceBarrier {
        buffers: prologue.buffers.clone(),
        textures: prologue.textures.clone(),
    });
    let mut skip = 0usize;
    while skip < commands.len() {
        if let crate::backend::GpuCommand::ResourceBarrier { buffers, textures } = &commands[skip] {
            if resource_barrier_redundant_with_prologue(buffers, textures, prologue) {
                // #region agent log
                crate::debug_session_log::write(
                    "H19",
                    "cross_submit.rs:prepend_prologue",
                    "skipped redundant leading ResourceBarrier already covered by cross-submit prologue",
                    &format!(
                        r#"{{"cmd_buffers":{},"cmd_textures":{},"prologue_buffers":{},"prologue_textures":{}}}"#,
                        buffers.len(),
                        textures.len(),
                        prologue.buffers.len(),
                        prologue.textures.len()
                    ),
                );
                // #endregion
                skip += 1;
                continue;
            }
        }
        break;
    }
    out.extend_from_slice(&commands[skip..]);
    out
}

fn interaction_role_from_net(access: &NetAccess) -> InteractionRole {
    if access.writes {
        InteractionRole::Writes
    } else {
        InteractionRole::Reads
    }
}

fn interaction_kind_bits_from_net(access: &NetAccess) -> u8 {
    if access.writes {
        access.write_kinds.bits()
    } else {
        access.read_kinds.bits()
    }
}

fn edge_matches(edge: &InteractionEdge, role: InteractionRole, kind_bits: u8, ctx: ContextHandle) -> bool {
    edge.role == role && edge.kind_bits == kind_bits && edge.ctx == ctx
}

fn prune_dead_edges(edges: &mut Vec<InteractionEdge>) {
    edges.retain(|edge| edge.dirty_flag.upgrade().is_some());
}

fn dirty_foreign_schemes(edges: &mut [InteractionEdge], scheme_id: u64) {
    for edge in edges.iter_mut() {
        if edge.scheme_id == scheme_id {
            continue;
        }
        if let Some(flag) = edge.dirty_flag.upgrade() {
            flag.store(true, Ordering::Release);
        }
    }
}

fn edges_have_foreign_war_conflict(edges: &[InteractionEdge], scheme_id: u64, role: InteractionRole) -> bool {
    match role {
        InteractionRole::Writes => edges
            .iter()
            .any(|e| e.scheme_id != scheme_id && e.role == InteractionRole::Reads),
        InteractionRole::Reads => edges
            .iter()
            .any(|e| e.scheme_id != scheme_id && e.role == InteractionRole::Writes),
    }
}

fn dirty_self_if_foreign_war(
    edges: &[InteractionEdge],
    scheme_id: u64,
    role: InteractionRole,
    dirty_flag: &Arc<AtomicBool>,
) {
    // Only writers need a topology refresh when a foreign reader already owns the parcel.
    // Foreign writes vs this scheme's read rely on FIFO / live sync — readers must not
    // re-record (see retained_reader_observes_independent_writer_across_resubmits).
    if role != InteractionRole::Writes {
        return;
    }
    if !edges_have_foreign_war_conflict(edges, scheme_id, role) {
        return;
    }
    dirty_flag.store(true, Ordering::Release);
}

type SelfTopologyEdge = (InteractionRole, u8, ContextHandle);

/// Remove this scheme's edges from every parcel in `prev_parcels` without notifying peers.
pub(crate) fn clear_scheme_topology_registration(
    scheme_id: u64,
    prev_parcels: &[(ResourceKey, Arc<ParcelStamp>)],
) -> ResourceKeyMap<SelfTopologyEdge> {
    let mut removed = ResourceKeyMap::default();
    for (key, stamp) in prev_parcels {
        let mut edges = stamp.interaction_set.lock().unwrap();
        if let Some(idx) = edges.iter().position(|edge| edge.scheme_id == scheme_id) {
            let edge = edges.remove(idx);
            removed.insert(*key, (edge.role, edge.kind_bits, edge.ctx));
        }
        prune_dead_edges(&mut edges);
    }
    removed
}

fn topology_parcels_from_net(
    net: &ResourceKeyMap<NetAccess>,
    resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
) -> Vec<(ResourceKey, Arc<ParcelStamp>)> {
    net.keys()
        .filter_map(|key| resource_stamps.get(key).map(|stamp| (*key, Arc::clone(stamp))))
        .collect()
}

/// Insert/update this scheme's edges for the current submission and dirty foreign schemes
/// when a parcel's interaction set actually changes.
pub(crate) fn update_scheme_topology(
    net: &ResourceKeyMap<NetAccess>,
    resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
    scheme_id: u64,
    ctx: ContextHandle,
    dirty_flag: &Arc<AtomicBool>,
    previous_self_edges: &ResourceKeyMap<SelfTopologyEdge>,
) {
    let weak_dirty = Arc::downgrade(dirty_flag);

    for (key, access) in net {
        let Some(stamp) = resource_stamps.get(key) else {
            continue;
        };
        let role = interaction_role_from_net(access);
        let kind_bits = interaction_kind_bits_from_net(access);
        let new_edge = (role, kind_bits, ctx);
        let mut edges = stamp.interaction_set.lock().unwrap();
        prune_dead_edges(&mut edges);

        let existing = edges.iter().position(|edge| edge.scheme_id == scheme_id);
        let is_first_parcel_registration = previous_self_edges.get(key).is_none();
        let topology_changed = match existing {
            Some(idx) => !edge_matches(&edges[idx], role, kind_bits, ctx),
            None => previous_self_edges.get(key) != Some(&new_edge),
        };

        if topology_changed {
            dirty_foreign_schemes(&mut edges, scheme_id);
            let edge = InteractionEdge {
                scheme_id,
                role,
                kind_bits,
                ctx,
                dirty_flag: Arc::downgrade(dirty_flag),
            };
            match existing {
                Some(idx) => edges[idx] = edge,
                None => edges.push(edge),
            }
            if is_first_parcel_registration {
                dirty_self_if_foreign_war(&edges, scheme_id, role, dirty_flag);
            }
        } else if let Some(idx) = existing {
            edges[idx].dirty_flag = weak_dirty.clone();
        } else {
            edges.push(InteractionEdge {
                scheme_id,
                role,
                kind_bits,
                ctx,
                dirty_flag: weak_dirty.clone(),
            });
            if is_first_parcel_registration {
                dirty_self_if_foreign_war(&edges, scheme_id, role, dirty_flag);
            }
        }
    }
}

/// Clear prior cross-scheme registration, register the current footprint, return the new set.
pub(crate) fn reregister_scheme_topology(
    net: &ResourceKeyMap<NetAccess>,
    resource_stamps: &ResourceKeyMap<Arc<ParcelStamp>>,
    prev_parcels: &[(ResourceKey, Arc<ParcelStamp>)],
    scheme_id: u64,
    ctx: ContextHandle,
    dirty_flag: &Arc<AtomicBool>,
) -> Vec<(ResourceKey, Arc<ParcelStamp>)> {
    let previous_self_edges = clear_scheme_topology_registration(scheme_id, prev_parcels);
    update_scheme_topology(net, resource_stamps, scheme_id, ctx, dirty_flag, &previous_self_edges);
    topology_parcels_from_net(net, resource_stamps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ContextHandle;
    use crate::task_graph::ir::{GraphIR, NodeAccess, NodeKind, ResourceBinding, TaskNode};
    use crate::task_graph::ResourceId;
    use crate::timeline::{ResourceSync, WRITE_KINDS_COMPUTE_TRANSFER};
    use std::sync::Weak;

    fn buf_key(h: u64) -> ResourceKey {
        ResourceKey::Buffer(h)
    }

    fn empty_stamp() -> Arc<ParcelStamp> {
        Arc::new(ParcelStamp::new(Weak::new()))
    }

    fn ledger_with_write(ctx: ContextHandle, key: ResourceKey, tv: u64) -> LedgerSnapshot {
        let mut sync = ResourceSync::default();
        sync.record_write(ctx, tv, WRITE_KINDS_COMPUTE_TRANSFER);
        let mut ledger = LedgerSnapshot::default();
        ledger.insert(key, LedgerEntry { sync });
        ledger
    }

    fn ledger_with_write_kinds(ctx: ContextHandle, key: ResourceKey, tv: u64, kinds: UsageKindFlags) -> LedgerSnapshot {
        let mut sync = ResourceSync::default();
        sync.record_write(ctx, tv, kinds.bits());
        let mut ledger = LedgerSnapshot::default();
        ledger.insert(key, LedgerEntry { sync });
        ledger
    }

    fn ledger_with_read(ctx: ContextHandle, key: ResourceKey, tv: u64) -> LedgerSnapshot {
        let mut sync = ResourceSync::default();
        sync.record_read(ctx, tv);
        let mut ledger = LedgerSnapshot::default();
        ledger.insert(key, LedgerEntry { sync });
        ledger
    }

    fn single_binding_ir(resource: ResourceId, access: NodeAccess) -> GraphIR {
        GraphIR {
            nodes: vec![TaskNode {
                label: "n",
                bindings: vec![ResourceBinding { resource, access }],
                kind: NodeKind::Dispatch {
                    pipeline: 1,
                    resource_slots: vec![],
                    user_slots: vec![],
                    dispatch: super::super::ir::DispatchDim::Direct { x: 1, y: 1, z: 1 },
                },
            }],
        }
    }

    #[test]
    fn war_same_context_write_after_read_records_wait_not_prologue_from_reads() {
        let ctx = 1;
        let key = ResourceKey::Texture(4);
        let mut sync = ResourceSync::default();
        sync.record_write(ctx, 44, WRITE_KINDS_COMPUTE_TRANSFER);
        sync.record_read(ctx, 45);
        let mut ledger = LedgerSnapshot::default();
        ledger.insert(key, LedgerEntry { sync });

        let mut ir = GraphIR::default();
        ir.nodes.push(TaskNode {
            label: "write_tex",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Texture(4),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::Dispatch {
                pipeline: 1,
                resource_slots: vec![],
                user_slots: vec![],
                dispatch: super::super::ir::DispatchDim::Direct { x: 1, y: 1, z: 1 },
            },
        });
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);

        assert!(
            sync.waits.is_empty(),
            "same-context WAR relies on FIFO single-queue ordering, not a live wait"
        );
        assert_eq!(
            sync.prologue.textures.len(),
            1,
            "loop-carried WAW against own last_write still needs a baked prologue barrier"
        );
    }

    #[test]
    fn raw_same_context_emits_buffer_barrier_no_waits() {
        let ctx = 1;
        let key = buf_key(10);
        let ledger = ledger_with_write(ctx, key, 5);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Read);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        assert!(sync.waits.is_empty());
        assert_eq!(sync.prologue.buffers.len(), 1);
        assert_eq!(sync.prologue.buffers[0].0, 10);
    }

    /// Regression: the producer-side barrier must reflect the *actual* recorded
    /// write kind, not a hardcoded `COMPUTE | TRANSFER`. A buffer last written by
    /// a transfer (copy) only must NOT produce a COMPUTE barrier source — that is
    /// what made non-storage buffers get illegal UAV access bits on DX12 (doom).
    #[test]
    fn raw_barrier_src_uses_recorded_write_kinds_transfer_only() {
        let ctx = 1;
        let key = buf_key(10);
        let ledger = ledger_with_write_kinds(ctx, key, 5, UsageKindFlags::TRANSFER);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Read);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        assert_eq!(sync.prologue.buffers.len(), 1);
        let src_kinds = sync.prologue.buffers[0].1.src.kinds;
        assert!(src_kinds.contains(UsageKindFlags::TRANSFER));
        assert!(
            !src_kinds.contains(UsageKindFlags::COMPUTE),
            "transfer-only producer must not synthesize a COMPUTE barrier source"
        );
    }

    /// Same invariant for the WAW edge.
    #[test]
    fn waw_barrier_src_uses_recorded_write_kinds_compute_only() {
        let ctx = 1;
        let key = buf_key(10);
        let ledger = ledger_with_write_kinds(ctx, key, 2, UsageKindFlags::COMPUTE);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Write);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        assert_eq!(sync.prologue.buffers.len(), 1);
        let src_kinds = sync.prologue.buffers[0].1.src.kinds;
        assert!(src_kinds.contains(UsageKindFlags::COMPUTE));
        assert!(!src_kinds.contains(UsageKindFlags::TRANSFER));
    }

    /// `apply_resource_sync_updates` must persist the write kinds it observed so a
    /// later submission's barrier analysis can recover them.
    #[test]
    fn apply_updates_round_trips_write_kinds() {
        use std::sync::Arc;

        let ctx = 1;
        let key = buf_key(10);
        let stamp = empty_stamp();
        let mut stamps: ResourceKeyMap<Arc<ParcelStamp>> = ResourceKeyMap::default();
        stamps.insert(key, Arc::clone(&stamp));

        let mut net: ResourceKeyMap<NetAccess> = ResourceKeyMap::default();
        net.insert(
            key,
            NetAccess {
                reads: false,
                writes: true,
                read_kinds: UsageKindFlags::empty(),
                read_pipeline_kinds: UsageKindFlags::empty(),
                write_kinds: UsageKindFlags::TRANSFER,
            },
        );

        apply_resource_sync_updates(&net, &stamps, ctx, 9);

        let sync = stamp.sync.lock();
        assert_eq!(sync.last_write.get(ctx), Some(9));
        assert_eq!(sync.last_write_kinds.get(ctx), Some(UsageKindFlags::TRANSFER.bits()));
    }

    #[test]
    fn rar_same_context_empty() {
        let ctx = 1;
        let key = buf_key(10);
        let ledger = ledger_with_read(ctx, key, 5);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Read);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        assert!(sync.is_empty());
    }

    #[test]
    fn raw_cross_context_emits_wait() {
        let producer = 1;
        let consumer = 2;
        let key = buf_key(10);
        let ledger = ledger_with_write(producer, key, 7);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Read);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, consumer);
        assert!(sync.prologue.is_empty());
        assert_eq!(
            sync.waits,
            vec![Epoch {
                context: producer,
                value: 7
            }]
        );
    }

    #[test]
    fn first_use_empty_ledger() {
        let ctx = 1;
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Write);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &LedgerSnapshot::default(), ctx);
        assert!(sync.is_empty());
    }

    #[test]
    fn war_same_context_non_fifo_read_emits_no_context_wait() {
        let ctx = 1;
        let key = buf_key(10);
        let ledger = ledger_with_read(ctx, key, 3);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Write);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        assert!(sync.prologue.is_empty());
        assert!(sync.waits.is_empty());
        assert!(sync.device_queue_waits.is_empty());
    }

    #[test]
    fn war_cross_scheme_same_context_emits_war_prologue_when_interaction_set_visible() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let ctx = 1;
        let key = buf_key(10);
        let ledger = ledger_with_read(ctx, key, 3);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Write);
        let net = net_access_per_resource(&ir);

        let reader_dirty = Arc::new(AtomicBool::new(false));
        let writer_dirty = Arc::new(AtomicBool::new(false));
        let stamp = empty_stamp();
        {
            let mut edges = stamp.interaction_set.lock().unwrap();
            edges.push(InteractionEdge {
                scheme_id: 1,
                role: InteractionRole::Reads,
                kind_bits: UsageKindFlags::TRANSFER.bits(),
                ctx,
                dirty_flag: Arc::downgrade(&reader_dirty),
            });
            edges.push(InteractionEdge {
                scheme_id: 2,
                role: InteractionRole::Writes,
                kind_bits: UsageKindFlags::COMPUTE.bits(),
                ctx,
                dirty_flag: Arc::downgrade(&writer_dirty),
            });
        }
        let mut stamps: ResourceKeyMap<Arc<ParcelStamp>> = ResourceKeyMap::default();
        stamps.insert(key, Arc::clone(&stamp));

        let mut prologue = BarrierSet::default();
        let mut waits = Vec::new();
        let mut device_waits = Vec::new();
        let mut context_waits = Vec::new();
        compute_cross_submit_sync_into(
            &mut prologue,
            &mut waits,
            &mut device_waits,
            &mut context_waits,
            &net,
            &ledger,
            ctx,
            Some(&stamps),
        );
        assert!(waits.is_empty(), "same-context cross-scheme WAR is baked, not a live wait");
        assert_eq!(prologue.buffers.len(), 1, "transfer read → compute write needs a prologue barrier");
    }

    #[test]
    fn war_same_context_fifo_present_easement_emits_device_queue_wait() {
        let ctx = 1;
        let key = buf_key(10);
        let present_tv = 7u64;
        let mut sync = ResourceSync::default();
        sync.record_read(ctx, present_tv);
        sync.mark_fifo_ordered_read(ctx, present_tv);
        let mut ledger = LedgerSnapshot::default();
        ledger.insert(key, LedgerEntry { sync });
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Write);
        let net = net_access_per_resource(&ir);
        let plan = compute_cross_submit_sync(&net, &ledger, ctx);
        assert!(plan.prologue.is_empty());
        assert!(plan.waits.is_empty());
        assert_eq!(plan.device_queue_waits, vec![present_tv]);
    }

    #[test]
    fn war_fifo_present_easement_texture_emits_transfer_to_write_barrier() {
        let ctx = 1;
        let tex = 5;
        let key = ResourceKey::Texture(tex);
        let present_tv = 7u64;
        let mut sync = ResourceSync::default();
        sync.record_read(ctx, present_tv);
        sync.mark_fifo_ordered_read(ctx, present_tv);
        sync.record_write(ctx, 6, WRITE_KINDS_COMPUTE_TRANSFER);
        let mut ledger = LedgerSnapshot::default();
        ledger.insert(key, LedgerEntry { sync });
        let ir = single_binding_ir(ResourceId::Texture(tex), NodeAccess::Write);
        let net = net_access_per_resource(&ir);
        let plan = compute_cross_submit_sync(&net, &ledger, ctx);
        assert!(plan.waits.is_empty());
        assert_eq!(plan.device_queue_waits, vec![present_tv]);
        assert_eq!(plan.prologue.textures.len(), 1);
        assert!(plan.prologue.textures[0].1.src.kinds.contains(UsageKindFlags::TRANSFER));
        assert!(plan.prologue.textures[0].1.dst.kinds.contains(UsageKindFlags::COMPUTE));
    }

    #[test]
    fn present_easement_live_prologue_extracts_transfer_to_compute_texture() {
        let mut full = BarrierSet::default();
        full.textures.push((
            1,
            BarrierUsage {
                src: {
                    let mut s = SlotUsageSet::default();
                    s.merge(NodeAccess::Read, UsageKindFlags::TRANSFER | UsageKindFlags::COMPUTE);
                    s
                },
                dst: {
                    let mut d = SlotUsageSet::default();
                    d.merge(NodeAccess::Write, UsageKindFlags::COMPUTE);
                    d
                },
            },
        ));
        let live = present_easement_live_prologue(&full, &[99]);
        assert_eq!(live.textures.len(), 1);
        assert_eq!(live.textures[0].0, 1);
        assert_eq!(live.textures[0].1.src.kinds, UsageKindFlags::TRANSFER);
        assert_eq!(live.textures[0].1.dst.kinds, UsageKindFlags::COMPUTE);
        assert!(present_easement_live_prologue(&full, &[]).is_empty());
    }

    #[test]
    fn present_easement_live_prologue_recovers_non_compute_write_kinds() {
        // Regression: an earlier version filtered to `dst.kinds.contains(COMPUTE)`, which
        // silently dropped the live replay for any present-easement write that wasn't COMPUTE
        // (e.g. a RENDER-only write), leaving the resource in a stale record-time state on
        // retained resubmit.
        let mut full = BarrierSet::default();
        full.textures.push((
            1,
            BarrierUsage {
                src: {
                    let mut s = SlotUsageSet::default();
                    s.merge(NodeAccess::Read, UsageKindFlags::TRANSFER);
                    s
                },
                dst: {
                    let mut d = SlotUsageSet::default();
                    d.merge(NodeAccess::Write, UsageKindFlags::RENDER);
                    d
                },
            },
        ));
        let live = present_easement_live_prologue(&full, &[99]);
        assert_eq!(live.textures.len(), 1);
        assert_eq!(live.textures[0].1.src.kinds, UsageKindFlags::TRANSFER);
        assert_eq!(live.textures[0].1.dst.kinds, UsageKindFlags::RENDER);
    }

    #[test]
    fn waw_same_context_emits_write_barrier() {
        let ctx = 1;
        let key = buf_key(10);
        let ledger = ledger_with_write(ctx, key, 2);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Write);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        assert!(sync.waits.is_empty());
        assert_eq!(sync.prologue.buffers.len(), 1);
        assert!(sync.prologue.buffers[0].1.dst.access.writes());
    }

    #[test]
    fn no_alias_resources_empty() {
        let ctx = 1;
        let ledger = ledger_with_write(ctx, buf_key(10), 5);
        let ir = single_binding_ir(ResourceId::Buffer(20), NodeAccess::Read);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        assert!(sync.is_empty());
    }

    #[test]
    fn war_cross_context_emits_wait_from_reads() {
        let producer = 1;
        let consumer = 2;
        let key = buf_key(10);
        let ledger = ledger_with_read(producer, key, 4);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Write);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, consumer);
        assert!(sync.prologue.is_empty());
        assert_eq!(
            sync.waits,
            vec![Epoch {
                context: producer,
                value: 4
            }]
        );
    }

    #[test]
    fn waw_cross_context_emits_wait_from_write() {
        let producer = 1;
        let consumer = 2;
        let key = buf_key(10);
        let ledger = ledger_with_write(producer, key, 9);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Write);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, consumer);
        assert_eq!(
            sync.waits,
            vec![Epoch {
                context: producer,
                value: 9
            }]
        );
    }

    #[test]
    fn render_pass_read_includes_render_pipeline_kind() {
        let ctx = 1;
        let key = buf_key(10);
        let ledger = ledger_with_write(ctx, key, 1);
        let ir = GraphIR {
            nodes: vec![TaskNode {
                label: "draw",
                bindings: vec![ResourceBinding {
                    resource: ResourceId::Buffer(10),
                    access: NodeAccess::Read,
                }],
                kind: NodeKind::RenderPass {
                    target: 1,
                    commands: vec![],
                },
            }],
        };
        let net = net_access_per_resource(&ir);
        assert!(net[&key].read_pipeline_kinds.contains(UsageKindFlags::RENDER));
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        let dst = &sync.prologue.buffers[0].1.dst;
        assert!(dst.kinds.contains(UsageKindFlags::RENDER));
    }

    #[test]
    fn multiple_predecessors_merge_waits_and_prologue() {
        let ctx_a = 1;
        let ctx_b = 2;
        let consumer = 3;
        let key = buf_key(10);
        let mut sync_state = ResourceSync::default();
        sync_state.record_write(ctx_a, 3, WRITE_KINDS_COMPUTE_TRANSFER);
        sync_state.record_write(ctx_b, 7, WRITE_KINDS_COMPUTE_TRANSFER);
        let mut ledger = LedgerSnapshot::default();
        ledger.insert(key, LedgerEntry { sync: sync_state });
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Read);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, consumer);
        assert_eq!(sync.waits.len(), 2);
        assert!(sync.waits.contains(&Epoch {
            context: ctx_a,
            value: 3
        }));
        assert!(sync.waits.contains(&Epoch {
            context: ctx_b,
            value: 7
        }));
    }

    #[test]
    fn whole_resource_disjoint_ranges_barrier_on_parent() {
        let ctx = 1;
        let parent: BufferHandle = 10;
        let key = ResourceKey::Buffer(parent);
        let ledger = ledger_with_write(ctx, key, 2);
        let ir = GraphIR {
            nodes: vec![TaskNode {
                label: "read_tail",
                bindings: vec![ResourceBinding {
                    resource: ResourceId::BufferRange {
                        parent,
                        offset: 64,
                        len: 64,
                    },
                    access: NodeAccess::Read,
                }],
                kind: NodeKind::Dispatch {
                    pipeline: 1,
                    resource_slots: vec![],
                    user_slots: vec![],
                    dispatch: super::super::ir::DispatchDim::Direct { x: 1, y: 1, z: 1 },
                },
            }],
        };
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        assert_eq!(sync.prologue.buffers[0].0, parent);
    }

    #[test]
    fn mixed_read_write_considers_both_hazards() {
        let ctx = 1;
        let key = buf_key(10);
        let mut sync_state = ResourceSync::default();
        sync_state.record_write(ctx, 5, WRITE_KINDS_COMPUTE_TRANSFER);
        sync_state.record_read(ctx, 3);
        let mut ledger = LedgerSnapshot::default();
        ledger.insert(key, LedgerEntry { sync: sync_state });
        let ir = GraphIR {
            nodes: vec![TaskNode {
                label: "rw",
                bindings: vec![ResourceBinding {
                    resource: ResourceId::Buffer(10),
                    access: NodeAccess::ReadWrite,
                }],
                kind: NodeKind::Dispatch {
                    pipeline: 1,
                    resource_slots: vec![],
                    user_slots: vec![],
                    dispatch: super::super::ir::DispatchDim::Direct { x: 1, y: 1, z: 1 },
                },
            }],
        };
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        assert!(!sync.prologue.buffers.is_empty());
    }

    #[test]
    fn byte_range_disjoint_submissions_no_hazard() {
        use crate::task_graph::analysis::ranges_overlap;
        assert!(!ranges_overlap(0, 32, 64, 32));
        let ctx = 1;
        let parent: BufferHandle = 5;
        let ledger_key = ResourceKey::BufferRange {
            parent,
            offset: 0,
            len: 32,
        };
        let ledger = ledger_with_write(ctx, ledger_key, 1);
        let ir = single_binding_ir(
            ResourceId::BufferRange {
                parent,
                offset: 64,
                len: 32,
            },
            NodeAccess::Read,
        );
        let sync = compute_cross_submit_sync(&net_access_per_resource(&ir), &ledger, ctx);
        assert_eq!(sync.prologue.buffers.len(), 0, "disjoint buffer ranges must not hazard");
    }

    // ---- merge_aliased_ledger_entries: multi-producer aliasing -------------------------
    //
    // The following tests guard the alias-merge path: when a queried key aliases
    // *several* ledger entries, all of them must contribute to the hazard analysis,
    // not just the first one encountered.

    fn range_ledger_with_two_writes(
        parent: BufferHandle,
        ctx_a: ContextHandle,
        tv_a: u64,
        kinds_a: UsageKindFlags,
        ctx_b: ContextHandle,
        tv_b: u64,
        kinds_b: UsageKindFlags,
    ) -> LedgerSnapshot {
        let mut sync_a = ResourceSync::default();
        sync_a.record_write(ctx_a, tv_a, kinds_a.bits());
        let mut sync_b = ResourceSync::default();
        sync_b.record_write(ctx_b, tv_b, kinds_b.bits());
        let mut ledger = LedgerSnapshot::default();
        ledger.insert(
            ResourceKey::BufferRange {
                parent,
                offset: 0,
                len: 32,
            },
            LedgerEntry { sync: sync_a },
        );
        ledger.insert(
            ResourceKey::BufferRange {
                parent,
                offset: 32,
                len: 32,
            },
            LedgerEntry { sync: sync_b },
        );
        ledger
    }

    /// A spanning range that overlaps two disjoint producer ranges (same ctx) must
    /// barrier against the *later* of the two write epochs.
    /// Before the fix, only the first-found aliasing entry contributed: the other
    /// epoch was silently dropped, potentially emitting a barrier against a stale write.
    #[test]
    fn spanning_range_read_merges_both_same_ctx_producers_into_one_barrier() {
        let ctx = 1;
        let parent: BufferHandle = 10;
        let ledger =
            range_ledger_with_two_writes(parent, ctx, 5, UsageKindFlags::COMPUTE, ctx, 9, UsageKindFlags::COMPUTE);

        // Reader spans the full [0, 64) — overlaps both [0,32) and [32,32).
        let ir = single_binding_ir(
            ResourceId::BufferRange {
                parent,
                offset: 0,
                len: 64,
            },
            NodeAccess::Read,
        );
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);

        assert!(sync.waits.is_empty(), "same-ctx: no queue-wait expected");
        assert_eq!(sync.prologue.buffers.len(), 1, "one barrier for the parent buffer");
    }

    /// Same scenario but both producers are on different contexts: the consumer must
    /// emit a queue-wait against *both* producer contexts, not just the first alias.
    #[test]
    fn spanning_range_read_emits_waits_for_both_cross_ctx_producers() {
        let producer_a = 1;
        let producer_b = 2;
        let consumer = 3;
        let parent: BufferHandle = 10;
        let ledger = range_ledger_with_two_writes(
            parent,
            producer_a,
            5,
            UsageKindFlags::COMPUTE,
            producer_b,
            7,
            UsageKindFlags::COMPUTE,
        );

        let ir = single_binding_ir(
            ResourceId::BufferRange {
                parent,
                offset: 0,
                len: 64,
            },
            NodeAccess::Read,
        );
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, consumer);

        assert!(sync.prologue.is_empty(), "cross-ctx: no same-context barrier expected");
        assert_eq!(sync.waits.len(), 2, "must wait on both producer contexts");
        assert!(
            sync.waits.iter().any(|e| e.context == producer_a && e.value == 5),
            "wait on producer_a tv=5 missing"
        );
        assert!(
            sync.waits.iter().any(|e| e.context == producer_b && e.value == 7),
            "wait on producer_b tv=7 missing"
        );
    }

    /// A partial-overlap read (covers only [0,48)) must still see *both* aliasing
    /// producers (the second range [32,32) partially overlaps [0,48)).
    #[test]
    fn partial_overlap_read_sees_all_aliasing_producers() {
        let producer_a = 1;
        let producer_b = 2;
        let consumer = 3;
        let parent: BufferHandle = 10;
        let ledger = range_ledger_with_two_writes(
            parent,
            producer_a,
            3,
            UsageKindFlags::COMPUTE,
            producer_b,
            8,
            UsageKindFlags::TRANSFER,
        );

        // [0, 48) overlaps both [0,32) and [32,32).
        let ir = single_binding_ir(
            ResourceId::BufferRange {
                parent,
                offset: 0,
                len: 48,
            },
            NodeAccess::Read,
        );
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, consumer);

        assert_eq!(sync.waits.len(), 2, "both producers must be seen");
    }

    /// Three-producer scenario: ledger has three disjoint ranges; a whole-parent
    /// read keyed Buffer(parent) must alias all three and emit waits for all three.
    #[test]
    fn three_range_producers_all_contribute_to_whole_parent_read() {
        let ctx_a = 1;
        let ctx_b = 2;
        let ctx_c = 3;
        let consumer = 4;
        let parent: BufferHandle = 20;

        let mut sync_a = ResourceSync::default();
        sync_a.record_write(ctx_a, 2, UsageKindFlags::COMPUTE.bits());
        let mut sync_b = ResourceSync::default();
        sync_b.record_write(ctx_b, 5, UsageKindFlags::COMPUTE.bits());
        let mut sync_c = ResourceSync::default();
        sync_c.record_write(ctx_c, 9, UsageKindFlags::TRANSFER.bits());

        let mut ledger = LedgerSnapshot::default();
        ledger.insert(
            ResourceKey::BufferRange {
                parent,
                offset: 0,
                len: 32,
            },
            LedgerEntry { sync: sync_a },
        );
        ledger.insert(
            ResourceKey::BufferRange {
                parent,
                offset: 32,
                len: 32,
            },
            LedgerEntry { sync: sync_b },
        );
        ledger.insert(
            ResourceKey::BufferRange {
                parent,
                offset: 64,
                len: 32,
            },
            LedgerEntry { sync: sync_c },
        );

        // Consumer reads the whole parent — Buffer(parent) aliases all three ranges.
        let ir = single_binding_ir(ResourceId::Buffer(parent), NodeAccess::Read);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, consumer);

        assert!(sync.prologue.is_empty());
        assert_eq!(sync.waits.len(), 3, "all three cross-ctx producers must be waited on");
        assert!(sync.waits.iter().any(|e| e.context == ctx_a && e.value == 2));
        assert!(sync.waits.iter().any(|e| e.context == ctx_b && e.value == 5));
        assert!(sync.waits.iter().any(|e| e.context == ctx_c && e.value == 9));
    }

    /// Same-ctx multi-producer: two writes from the same ctx at tv=3 and tv=9;
    /// a spanning write (WAW) must see the max epoch (tv=9) as the existing write to
    /// barrier against, not a stale tv=3 from whichever alias was found first.
    #[test]
    fn spanning_write_waw_sees_max_epoch_from_two_same_ctx_producers() {
        let ctx = 1;
        let parent: BufferHandle = 10;
        let ledger =
            range_ledger_with_two_writes(parent, ctx, 3, UsageKindFlags::COMPUTE, ctx, 9, UsageKindFlags::COMPUTE);

        let ir = single_binding_ir(
            ResourceId::BufferRange {
                parent,
                offset: 0,
                len: 64,
            },
            NodeAccess::Write,
        );
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);

        assert!(sync.waits.is_empty(), "same-ctx WAW must emit a barrier, not a wait");
        assert_eq!(sync.prologue.buffers.len(), 1, "WAW barrier required");
        assert!(sync.prologue.buffers[0].1.dst.access.writes());
    }

    /// Regression guard: the existing disjoint-range no-hazard property must be
    /// preserved — the alias-merge path must not accidentally merge entries whose ranges
    /// do not alias the queried key.
    #[test]
    fn non_overlapping_range_in_multi_producer_ledger_not_merged() {
        let ctx = 1;
        let parent: BufferHandle = 10;

        // Producer writes [0,32) and [128,32) — two disjoint ranges.
        // Consumer reads [64,32) — overlaps neither.
        let mut sync_a = ResourceSync::default();
        sync_a.record_write(ctx, 5, UsageKindFlags::COMPUTE.bits());
        let mut sync_b = ResourceSync::default();
        sync_b.record_write(ctx, 9, UsageKindFlags::COMPUTE.bits());

        let mut ledger = LedgerSnapshot::default();
        ledger.insert(
            ResourceKey::BufferRange {
                parent,
                offset: 0,
                len: 32,
            },
            LedgerEntry { sync: sync_a },
        );
        ledger.insert(
            ResourceKey::BufferRange {
                parent,
                offset: 128,
                len: 32,
            },
            LedgerEntry { sync: sync_b },
        );

        let ir = single_binding_ir(
            ResourceId::BufferRange {
                parent,
                offset: 64,
                len: 32,
            },
            NodeAccess::Read,
        );
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);

        assert!(
            sync.prologue.is_empty(),
            "non-overlapping producers must not contribute barriers"
        );
        assert!(
            sync.waits.is_empty(),
            "non-overlapping producers must not contribute waits"
        );
    }

    #[test]
    fn buffer_stamp_index_produces_same_ledger_as_full_scan() {
        let parent: BufferHandle = 10;
        let whole = ResourceKey::Buffer(parent);
        let range_a = ResourceKey::BufferRange {
            parent,
            offset: 0,
            len: 32,
        };
        let range_b = ResourceKey::BufferRange {
            parent,
            offset: 64,
            len: 32,
        };
        let unrelated = ResourceKey::Buffer(99);

        let mut resource_stamps = ResourceKeyMap::default();
        resource_stamps.insert(whole, empty_stamp());
        resource_stamps.insert(range_a, empty_stamp());
        resource_stamps.insert(range_b, empty_stamp());
        resource_stamps.insert(unrelated, empty_stamp());

        let mut index = BufferStampIndex::new();
        for key in [whole, range_a, range_b, unrelated] {
            index.register(key);
        }

        let query = ResourceKey::BufferRange {
            parent,
            offset: 16,
            len: 64,
        };
        let mut net = ResourceKeyMap::default();
        net.insert(
            query,
            NetAccess {
                reads: true,
                ..Default::default()
            },
        );

        let mut with_index = LedgerSnapshot::default();
        build_ledger_snapshot_for_net_into(&mut with_index, &net, &resource_stamps, Some(&index));

        let mut full_scan = LedgerSnapshot::default();
        build_ledger_snapshot_for_net_into(&mut full_scan, &net, &resource_stamps, None);

        assert_eq!(with_index.len(), full_scan.len());
        for key in with_index.keys() {
            assert!(full_scan.contains_key(key), "missing key {key:?} in full-scan ledger");
        }
    }
}
