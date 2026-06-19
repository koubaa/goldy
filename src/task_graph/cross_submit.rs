//! Cross-submission hazard analysis for independent [`Scheme`] / [`TaskGraph`] submits.
//!
//! Intra-submission barriers are computed by [`super::analysis`]; this module derives
//! scoped memory barriers and cross-context queue-waits from the per-resource epoch
//! ledger on [`crate::parcel::ParcelStamp`].

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::backend::{BufferHandle, ContextHandle, TextureHandle};
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

/// Find and merge all ledger entries that alias `key`.
///
/// Returns an owned, merged [`LedgerEntry`] so that a range read against several
/// independently-tracked producer ranges (e.g. both fields of a partitioned buffer)
/// contributes barriers/waits from every producer, not just the first one found.
fn find_ledger_entry(ledger: &LedgerSnapshot, key: ResourceKey) -> Option<LedgerEntry> {
    // Fast path: exact match — clone once and return.
    if let Some(entry) = ledger.get(&key) {
        return Some(entry.clone());
    }
    // Fallback: merge every aliasing entry so no producer is silently dropped.
    let mut merged: Option<LedgerEntry> = None;
    for (ledger_key, entry) in ledger {
        if !resource_keys_alias(*ledger_key, key) {
            continue;
        }
        match &mut merged {
            None => merged = Some(entry.clone()),
            Some(m) => {
                for (&ctx, &tv) in &entry.sync.last_write {
                    let kinds = entry
                        .sync
                        .last_write_kinds
                        .get(&ctx)
                        .copied()
                        .unwrap_or(WRITE_KINDS_COMPUTE_TRANSFER);
                    m.sync.record_write(ctx, tv, kinds);
                }
                for (&ctx, &tv) in &entry.sync.last_reads {
                    m.sync.record_read(ctx, tv);
                }
            }
        }
    }
    merged
}

/// Snapshot of one resource's epoch ledger at submit time.
#[derive(Debug, Clone, Default)]
pub struct LedgerEntry {
    pub sync: ResourceSync,
}

/// Per-resource ledger keyed by [`ResourceKey`], populated from parcel stamps.
pub type LedgerSnapshot = HashMap<ResourceKey, LedgerEntry>;

/// Result of cross-submission hazard analysis for one submit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrossSubmitSync {
    /// Same-context memory barriers prepended to the submission command stream.
    pub prologue: BarrierSet,
    /// Cross-context GPU queue-waits (one per producer context, max tv).
    pub waits: Vec<Epoch>,
}

impl CrossSubmitSync {
    pub fn is_empty(&self) -> bool {
        self.prologue.is_empty() && self.waits.is_empty()
    }
}

fn node_usage_kind(node: &super::ir::TaskNode) -> UsageKindFlags {
    match &node.kind {
        NodeKind::Dispatch { .. } => UsageKindFlags::COMPUTE,
        NodeKind::RenderPass { .. } => UsageKindFlags::RENDER,
        NodeKind::ClearBuffer { .. }
        | NodeKind::WriteBuffer { .. }
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
pub fn net_access_per_resource(ir: &GraphIR) -> HashMap<ResourceKey, NetAccess> {
    let mut net: HashMap<ResourceKey, NetAccess> = HashMap::new();
    for node in &ir.nodes {
        absorb_node_net_access(&mut net, node);
    }
    net
}

/// Union each resource's access across the nodes in `waves` (one submit partition).
pub fn net_access_for_waves(ir: &GraphIR, waves: &[super::ir::Wave]) -> HashMap<ResourceKey, NetAccess> {
    let mut net: HashMap<ResourceKey, NetAccess> = HashMap::new();
    for wave in waves {
        for &node_idx in &wave.node_indices {
            absorb_node_net_access(&mut net, &ir.nodes[node_idx]);
        }
    }
    net
}

fn absorb_node_net_access(net: &mut HashMap<ResourceKey, NetAccess>, node: &super::ir::TaskNode) {
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

fn producer_slot_from_read(read_kinds: UsageKindFlags) -> SlotUsageSet {
    let mut src = SlotUsageSet::default();
    src.merge(NodeAccess::Read, read_kinds);
    src
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
pub fn compute_cross_submit_sync(
    net: &HashMap<ResourceKey, NetAccess>,
    ledger: &LedgerSnapshot,
    submitting_ctx: crate::backend::ContextHandle,
) -> CrossSubmitSync {
    let mut result = CrossSubmitSync::default();
    let mut wait_map: HashMap<crate::backend::ContextHandle, u64> = HashMap::new();

    for (key, access) in net {
        let Some(entry) = find_ledger_entry(ledger, *key) else {
            continue;
        };
        let sync = &entry.sync;

        // RAW: this reads -> hazard vs last_write
        if access.reads {
            for (&ctx, &tv) in &sync.last_write {
                if ctx == submitting_ctx {
                    let prev_write_kinds = UsageKindFlags::from_bits_truncate(
                        *sync.last_write_kinds.get(&ctx).unwrap_or(&WRITE_KINDS_COMPUTE_TRANSFER),
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
                    merge_barrier(&mut result.prologue, *key, usage);
                } else {
                    wait_map.entry(ctx).and_modify(|v| *v = (*v).max(tv)).or_insert(tv);
                }
            }
        }

        // WAW + WAR: this writes -> hazard vs last_write and last_reads
        if access.writes {
            for (&ctx, &tv) in &sync.last_write {
                if ctx == submitting_ctx {
                    let prev_write_kinds = UsageKindFlags::from_bits_truncate(
                        *sync.last_write_kinds.get(&ctx).unwrap_or(&WRITE_KINDS_COMPUTE_TRANSFER),
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
                    merge_barrier(&mut result.prologue, *key, usage);
                } else {
                    wait_map.entry(ctx).and_modify(|v| *v = (*v).max(tv)).or_insert(tv);
                }
            }
            for (&ctx, &tv) in &sync.last_reads {
                if ctx == submitting_ctx {
                    let usage = BarrierUsage {
                        src: producer_slot_from_read(access.read_kinds | UsageKindFlags::COMPUTE),
                        dst: {
                            let mut d = SlotUsageSet::default();
                            d.merge(NodeAccess::Write, access.write_kinds);
                            d
                        },
                    };
                    merge_barrier(&mut result.prologue, *key, usage);
                } else {
                    wait_map.entry(ctx).and_modify(|v| *v = (*v).max(tv)).or_insert(tv);
                }
            }
        }
    }

    result.waits = wait_map
        .into_iter()
        .map(|(context, value)| Epoch { context, value })
        .collect();
    result.waits.sort_by_key(|e| (e.context, e.value));
    result
}

/// Build a ledger snapshot from registered stamp bindings.
pub fn build_ledger_snapshot(stamps: &[(ResourceKey, Arc<ParcelStamp>)]) -> LedgerSnapshot {
    let mut ledger = LedgerSnapshot::new();
    for (key, stamp) in stamps {
        ledger.entry(*key).or_insert_with(|| LedgerEntry {
            sync: stamp.sync.lock().unwrap().clone(),
        });
    }
    ledger
}

/// Collect unique resource keys from IR that have registered stamps.
pub fn resource_stamps_from_ir(
    ir: &GraphIR,
    resource_stamps: &HashMap<ResourceKey, Arc<ParcelStamp>>,
) -> Vec<(ResourceKey, Arc<ParcelStamp>)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
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
    out
}

/// After a successful submit, record this submission's access on each touched stamp.
pub fn apply_resource_sync_updates(
    net: &HashMap<ResourceKey, NetAccess>,
    resource_stamps: &HashMap<ResourceKey, Arc<ParcelStamp>>,
    ctx: crate::backend::ContextHandle,
    tv: u64,
) {
    for (key, access) in net {
        let Some(stamp) = resource_stamps.get(key) else {
            continue;
        };
        let mut sync = stamp.sync.lock().unwrap();
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
        let mut sync = stamp.sync.lock().unwrap();
        sync.record_any(ctx, tv);
    }
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
    out.extend_from_slice(commands);
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

type SelfTopologyEdge = (InteractionRole, u8, ContextHandle);

/// Remove this scheme's edges from every parcel in `prev_parcels` without notifying peers.
pub(crate) fn clear_scheme_topology_registration(
    scheme_id: u64,
    prev_parcels: &[(ResourceKey, Arc<ParcelStamp>)],
) -> HashMap<ResourceKey, SelfTopologyEdge> {
    let mut removed = HashMap::new();
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
    net: &HashMap<ResourceKey, NetAccess>,
    resource_stamps: &HashMap<ResourceKey, Arc<ParcelStamp>>,
) -> Vec<(ResourceKey, Arc<ParcelStamp>)> {
    net.keys()
        .filter_map(|key| resource_stamps.get(key).map(|stamp| (*key, Arc::clone(stamp))))
        .collect()
}

/// Insert/update this scheme's edges for the current submission and dirty foreign schemes
/// when a parcel's interaction set actually changes.
pub(crate) fn update_scheme_topology(
    net: &HashMap<ResourceKey, NetAccess>,
    resource_stamps: &HashMap<ResourceKey, Arc<ParcelStamp>>,
    scheme_id: u64,
    ctx: ContextHandle,
    dirty_flag: &Arc<AtomicBool>,
    previous_self_edges: &HashMap<ResourceKey, SelfTopologyEdge>,
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
        }
    }
}

/// Clear prior cross-scheme registration, register the current footprint, return the new set.
pub(crate) fn reregister_scheme_topology(
    net: &HashMap<ResourceKey, NetAccess>,
    resource_stamps: &HashMap<ResourceKey, Arc<ParcelStamp>>,
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
        let mut ledger = LedgerSnapshot::new();
        ledger.insert(key, LedgerEntry { sync });
        ledger
    }

    fn ledger_with_write_kinds(ctx: ContextHandle, key: ResourceKey, tv: u64, kinds: UsageKindFlags) -> LedgerSnapshot {
        let mut sync = ResourceSync::default();
        sync.record_write(ctx, tv, kinds.bits());
        let mut ledger = LedgerSnapshot::new();
        ledger.insert(key, LedgerEntry { sync });
        ledger
    }

    fn ledger_with_read(ctx: ContextHandle, key: ResourceKey, tv: u64) -> LedgerSnapshot {
        let mut sync = ResourceSync::default();
        sync.record_read(ctx, tv);
        let mut ledger = LedgerSnapshot::new();
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
        use std::collections::HashMap;
        use std::sync::Arc;

        let ctx = 1;
        let key = buf_key(10);
        let stamp = empty_stamp();
        let mut stamps: HashMap<ResourceKey, Arc<ParcelStamp>> = HashMap::new();
        stamps.insert(key, Arc::clone(&stamp));

        let mut net: HashMap<ResourceKey, NetAccess> = HashMap::new();
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

        let sync = stamp.sync.lock().unwrap();
        assert_eq!(sync.last_write.get(&ctx), Some(&9));
        assert_eq!(sync.last_write_kinds.get(&ctx), Some(&UsageKindFlags::TRANSFER.bits()));
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
        let sync = compute_cross_submit_sync(&net, &LedgerSnapshot::new(), ctx);
        assert!(sync.is_empty());
    }

    #[test]
    fn war_same_context_emits_barrier() {
        let ctx = 1;
        let key = buf_key(10);
        let ledger = ledger_with_read(ctx, key, 3);
        let ir = single_binding_ir(ResourceId::Buffer(10), NodeAccess::Write);
        let net = net_access_per_resource(&ir);
        let sync = compute_cross_submit_sync(&net, &ledger, ctx);
        assert!(sync.waits.is_empty());
        assert_eq!(sync.prologue.buffers.len(), 1);
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
        let mut ledger = LedgerSnapshot::new();
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
        let mut ledger = LedgerSnapshot::new();
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

    // ---- find_ledger_entry: multi-producer aliasing -------------------------
    //
    // The following tests guard the fix to find_ledger_entry: when a queried
    // key aliases *several* ledger entries, all of them must contribute to the
    // hazard analysis, not just the first one encountered.

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
        let mut ledger = LedgerSnapshot::new();
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

        let mut ledger = LedgerSnapshot::new();
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
    /// preserved — find_ledger_entry must not accidentally merge entries whose ranges
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

        let mut ledger = LedgerSnapshot::new();
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
}
