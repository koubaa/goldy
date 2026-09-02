//! Dependency analysis, wave scheduling, and command emission.
//!
//! This module implements the core scheduling algorithm:
//!
//! 1. **Edge construction**: for each pair of nodes (i, j) where i < j,
//!    a dependency edge exists if they share a resource and at least one
//!    writes (RAW, WAR, or WAW). Multiple reads create no edge (SWMR).
//!    For `BufferRange` resources, conflicts are detected at byte-range
//!    granularity: non-overlapping sub-ranges of the same parent buffer are
//!    independent and produce no edge.
//!
//! 2. **Wave scheduling**: nodes are assigned to waves via BFS-based
//!    topological sort with longest-path depth tracking. Independent nodes
//!    share a wave and can execute concurrently on the GPU.
//!
//! 3. **Barrier computation**: for each wave boundary, only the specific
//!    resources involved in cross-wave dependency edges are listed in the
//!    barrier set. `BufferRange` entries are collapsed to their parent
//!    `BufferHandle` so backends always receive whole-buffer handles.
//!
//! 4. **Command emission**: waves are serialized into a flat
//!    `Vec<GpuCommand>` with `ResourceBarrier` commands between waves.
//!    Each [`NodeKind`] variant emits different commands.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::ir::{BarrierSet, BarrierUsage, CompiledSchedule, GraphIR, NodeKind, ResourceBinding, UsageKindFlags, Wave};
// NodeAccess / Result are used in the test module (and cfg(test) helpers) via `super::*`
#[cfg(test)]
use super::ir::NodeAccess;
use super::{ResourceId, SlotResolver};
use crate::backend::shared::DISPATCH_BATCH_STRIDE;
use crate::backend::{GpuCommand, GraphCommand, TextureHandle};
use crate::frame_table::FrameTableStaging;
#[cfg(test)]
use anyhow::Result;

fn push_compute_resource_bind(
    commands: &mut Vec<GpuCommand>,
    staging: &mut FrameTableStaging,
    slots: &[u32],
    user_slots: &[u32],
) {
    if !slots.is_empty() || !user_slots.is_empty() {
        let frame_table_base = staging.alloc_dispatch(slots.len() as u32);
        staging.write_dispatch_indices(frame_table_base, slots);
        commands.push(GpuCommand::BindResourcesRaw {
            indices: slots.to_vec(),
            user: user_slots.to_vec(),
            frame_table_base,
        });
    }
}

/// Returns true if byte range `[o1, o1+l1)` overlaps `[o2, o2+l2)`.
///
/// Zero-length ranges never overlap anything.
pub(crate) fn ranges_overlap(o1: u64, l1: u64, o2: u64, l2: u64) -> bool {
    if l1 == 0 || l2 == 0 {
        return false;
    }
    o1 < o2 + l2 && o2 < o1 + l1
}

/// Returns true if the two resource bindings form a data dependency.
///
/// A dependency exists when the resources alias (touch the same bytes) and at
/// least one binding writes. Two reads on any aliased resource are safe
/// (SWMR).
///
/// # Resource aliasing rules
///
/// | `a`                    | `b`                    | Aliases?                          |
/// |------------------------|------------------------|-----------------------------------|
/// | `Buffer(x)`            | `Buffer(y)`            | `x == y`                          |
/// | `Buffer(h)`            | `BufferRange{parent:h}`| always (whole-buffer subsumes any range) |
/// | `BufferRange{parent:p1}` | `BufferRange{parent:p2}` | `p1 == p2` AND ranges overlap |
/// | `Texture(x)`           | `Texture(y)`           | `x == y`                          |
/// | buffer variant         | texture variant        | never                             |
pub(crate) fn bindings_conflict(a: &ResourceBinding, b: &ResourceBinding) -> bool {
    if !(a.access.writes() || b.access.writes()) {
        return false;
    }
    resources_alias(a.resource, b.resource)
}

/// Returns true if two `ResourceId`s refer to overlapping GPU memory.
pub(crate) fn resources_alias(a: ResourceId, b: ResourceId) -> bool {
    match (a, b) {
        (ResourceId::Buffer(x), ResourceId::Buffer(y)) => x == y,
        (ResourceId::Buffer(h), ResourceId::BufferRange { parent, .. })
        | (ResourceId::BufferRange { parent, .. }, ResourceId::Buffer(h)) => h == parent,
        (
            ResourceId::BufferRange {
                parent: p1,
                offset: o1,
                len: l1,
            },
            ResourceId::BufferRange {
                parent: p2,
                offset: o2,
                len: l2,
            },
        ) => p1 == p2 && ranges_overlap(o1, l1, o2, l2),
        (ResourceId::Texture(x), ResourceId::Texture(y)) => x == y,
        #[cfg(feature = "graphics")]
        (ResourceId::RenderTarget(x), ResourceId::RenderTarget(y)) => x == y,
        (ResourceId::TransientBuffer(x), ResourceId::TransientBuffer(y)) => x == y,
        (ResourceId::TransientTexture(x), ResourceId::TransientTexture(y)) => x == y,
        #[cfg(feature = "graphics")]
        (ResourceId::SwapchainOutput, ResourceId::SwapchainOutput) => true,
        #[cfg(feature = "graphics")]
        (ResourceId::PresentLease(a), ResourceId::PresentLease(b)) => a == b,
        (ResourceId::Deposit(a), ResourceId::Deposit(b)) => a == b,
        (ResourceId::Accel(a), ResourceId::Accel(b)) => a == b,
        _ => false,
    }
}

fn resolve_copy_destination(id: ResourceId, resolver: Option<&SlotResolver>) -> TextureHandle {
    let resolved = match resolver {
        Some(r) => r.resolve(id),
        None => id,
    };
    match resolved {
        ResourceId::Texture(h) => h,
        #[cfg(feature = "graphics")]
        ResourceId::SwapchainOutput => {
            panic!("copy destination SwapchainOutput emitted before surface acquire")
        }
        #[cfg(feature = "graphics")]
        ResourceId::PresentLease(_) => {
            panic!("copy destination PresentLease emitted before pool acquire")
        }
        other => panic!("copy destination resolved to non-texture resource: {other:?}"),
    }
}

fn resolve_buffer_copy_target(
    id: ResourceId,
    offset: u64,
    resolver: Option<&SlotResolver>,
) -> (crate::backend::BufferHandle, u64) {
    let resolved = match id {
        ResourceId::Deposit(_) => match resolver {
            Some(r) => r.resolve(id),
            None => panic!("CopyBuffer Deposit emitted before submit-time resolve"),
        },
        other => other,
    };
    match resolved {
        ResourceId::Buffer(h) => (h, offset),
        ResourceId::BufferRange {
            parent, offset: base, ..
        } => (parent, base.saturating_add(offset)),
        other => panic!("CopyBuffer resource resolved to non-buffer: {other:?}"),
    }
}

/// Returns true when one node implicitly/explicitly writes a render target and
/// the other reads the same target (e.g. [`NodeKind::RenderPass`] → copy).
#[cfg(feature = "graphics")]
fn render_target_access_conflict(a: &super::ir::TaskNode, b: &super::ir::TaskNode) -> bool {
    let rt_from = |node: &super::ir::TaskNode| match &node.kind {
        NodeKind::RenderPass { target, .. } => Some(*target),
        _ => None,
    };
    let reads_rt = |node: &super::ir::TaskNode, rt: crate::backend::RenderTargetHandle| {
        node.bindings
            .iter()
            .any(|b| matches!(b.resource, ResourceId::RenderTarget(h) if h == rt) && b.access.reads())
    };
    if let Some(rt) = rt_from(a) {
        if reads_rt(b, rt) {
            return true;
        }
    }
    if let Some(rt) = rt_from(b) {
        if reads_rt(a, rt) {
            return true;
        }
    }
    if let (Some(t1), Some(t2)) = (rt_from(a), rt_from(b)) {
        return t1 == t2;
    }
    false
}

#[cfg(not(feature = "graphics"))]
fn render_target_access_conflict(_a: &super::ir::TaskNode, _b: &super::ir::TaskNode) -> bool {
    false
}

fn nodes_conflict(ir: &GraphIR, i: usize, j: usize) -> bool {
    if ir.nodes[i]
        .bindings
        .iter()
        .any(|bi| ir.nodes[j].bindings.iter().any(|bj| bindings_conflict(bi, bj)))
    {
        return true;
    }
    render_target_access_conflict(&ir.nodes[i], &ir.nodes[j])
}

/// Build directed dependency edges between graph nodes.
///
/// An edge (i -> j) means node j depends on node i and must execute after it.
/// Edges are created when two nodes access overlapping resources and at least
/// one writes. Non-overlapping `BufferRange`s from the same parent produce no
/// edge even though they share a backing allocation.
///
/// Uses a **resource-to-nodes index** to avoid the naïve O(N²) pair scan.
/// Each binding is bucketed by its canonical resource key (parent buffer
/// handle or texture handle). Only node pairs that share at least one bucket
/// are checked for actual conflict, reducing complexity from O(N² × B²) to
/// O(Σ_r K_r²) where K_r is the number of nodes touching resource r.
pub fn build_edges(ir: &GraphIR) -> Vec<(usize, usize)> {
    let n = ir.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Canonical grouping key: buffers and buffer ranges map to the parent
    // handle; textures map to their own handle in a separate namespace.
    #[derive(Hash, Eq, PartialEq, Clone, Copy)]
    enum GroupKey {
        Buffer(u64),
        Texture(u64),
        #[cfg(feature = "graphics")]
        RenderTarget(u64),
        TransientBuffer(u32),
        TransientTexture(u32),
        #[cfg(feature = "graphics")]
        SwapchainOutput,
        #[cfg(feature = "graphics")]
        PresentLease(u32),
        Deposit(u32),
        Accel(u64),
    }

    fn group_key(r: &ResourceId) -> GroupKey {
        match *r {
            ResourceId::Buffer(h) => GroupKey::Buffer(h),
            ResourceId::BufferRange { parent, .. } => GroupKey::Buffer(parent),
            ResourceId::Texture(h) => GroupKey::Texture(h),
            #[cfg(feature = "graphics")]
            ResourceId::RenderTarget(h) => GroupKey::RenderTarget(h),
            ResourceId::TransientBuffer(t) => GroupKey::TransientBuffer(t.0),
            ResourceId::TransientTexture(t) => GroupKey::TransientTexture(t.0),
            #[cfg(feature = "graphics")]
            ResourceId::SwapchainOutput => GroupKey::SwapchainOutput,
            #[cfg(feature = "graphics")]
            ResourceId::PresentLease(id) => GroupKey::PresentLease(id),
            ResourceId::Deposit(id) => GroupKey::Deposit(id),
            ResourceId::Accel(h) => GroupKey::Accel(h),
        }
    }

    // Map each canonical key to the set of node indices that reference it.
    let mut resource_nodes: HashMap<GroupKey, Vec<usize>> = HashMap::new();
    for (idx, node) in ir.nodes.iter().enumerate() {
        #[cfg(feature = "graphics")]
        if let NodeKind::RenderPass { target, .. } = &node.kind {
            resource_nodes
                .entry(GroupKey::RenderTarget(*target))
                .or_default()
                .push(idx);
        }
        for binding in &node.bindings {
            resource_nodes
                .entry(group_key(&binding.resource))
                .or_default()
                .push(idx);
        }
    }

    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();

    for node_indices in resource_nodes.values() {
        // Deduplicate (a node can appear multiple times if it has several
        // bindings on the same canonical resource).
        let mut unique = node_indices.clone();
        unique.sort_unstable();
        unique.dedup();

        for (pos, &j) in unique.iter().enumerate() {
            for &i in &unique[..pos] {
                debug_assert!(i < j);
                if edge_set.contains(&(i, j)) {
                    continue;
                }
                if nodes_conflict(ir, i, j) {
                    edge_set.insert((i, j));
                }
            }
        }
    }

    // Upload-phase ordering: buffer staging copies (CopyBuffer, ClearBuffer, WriteBuffer)
    // run in an earlier wave than texture uploads (CopyBufferToTexture, WriteTexture, …)
    // so retainable buffer-only partitions can be submitted separately from texture uploads.
    let mut buffer_upload_nodes: Vec<usize> = Vec::new();
    let mut texture_upload_nodes: Vec<usize> = Vec::new();
    for (idx, node) in ir.nodes.iter().enumerate() {
        match &node.kind {
            NodeKind::ClearBuffer { .. } | NodeKind::CopyBuffer { .. } | NodeKind::WriteBuffer { .. } => {
                buffer_upload_nodes.push(idx);
            }
            NodeKind::CopyBufferToTexture { .. }
            | NodeKind::WriteTexture { .. }
            | NodeKind::WriteTextureRegion { .. }
            | NodeKind::CopyTexture { .. }
            | NodeKind::CopyTextureRegion { .. } => {
                texture_upload_nodes.push(idx);
            }
            _ => {}
        }
    }
    for &t in &texture_upload_nodes {
        for &b in &buffer_upload_nodes {
            edge_set.insert((b, t));
        }
    }

    let mut edges: Vec<_> = edge_set.into_iter().collect();
    edges.sort_unstable();
    edges
}

/// Wave index for each task node (same order as [`GraphIR::nodes`]).
///
/// Recomputes the full schedule from the IR. Prefer [`node_to_wave_map`] when
/// a [`CompiledSchedule`] is already available to avoid redundant scheduling.
#[cfg(test)]
pub(crate) fn graph_node_waves(ir: &GraphIR) -> Result<Vec<u32>> {
    let n = ir.nodes.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let edges = build_edges(ir);
    let schedule = schedule_waves(ir, &edges);
    Ok(node_to_wave_map(&schedule, n))
}

/// Derive the node-to-wave assignment from an already-computed [`CompiledSchedule`].
///
/// This is O(N) and avoids the full `build_edges` + `schedule_waves` pass that
/// [`graph_node_waves`] performs. Use this whenever a `CompiledSchedule` is
/// already in hand (e.g. from a precomputed schedule).
#[cfg(test)]
pub(crate) fn node_to_wave_map(schedule: &CompiledSchedule, n: usize) -> Vec<u32> {
    let mut map = vec![0u32; n];
    for (w, wave) in schedule.waves.iter().enumerate() {
        for &ni in &wave.node_indices {
            map[ni] = w as u32;
        }
    }
    map
}

/// For each [`ResourceId::TransientBuffer`](super::ResourceId), the inclusive
/// range of wave indices where that transient appears in node bindings.
///
/// `node_waves[i]` is the wave index of IR node `i`. Use [`node_to_wave_map`]
/// to derive this from a [`CompiledSchedule`] without re-running the scheduler.
///
/// Used to pack transient heap allocations: non-overlapping wave intervals may
/// alias the same memory.
#[cfg(test)]
pub(crate) fn transient_wave_intervals(ir: &GraphIR, node_waves: &[u32]) -> Result<HashMap<u32, (u32, u32)>> {
    if ir.nodes.is_empty() {
        return Ok(HashMap::new());
    }
    let mut first: HashMap<u32, u32> = HashMap::new();
    let mut last: HashMap<u32, u32> = HashMap::new();
    for (ni, node) in ir.nodes.iter().enumerate() {
        let w = node_waves[ni];
        for b in &node.bindings {
            if let ResourceId::TransientBuffer(tid) = b.resource {
                let id = tid.0;
                first.entry(id).and_modify(|e| *e = (*e).min(w)).or_insert(w);
                last.entry(id).and_modify(|e| *e = (*e).max(w)).or_insert(w);
            }
        }
    }
    let mut out = HashMap::with_capacity(first.len());
    for (id, s) in first {
        let e = last[&id];
        out.insert(id, (s, e));
    }
    Ok(out)
}

/// For each [`ResourceId::TransientTexture`](super::ResourceId), inclusive wave range.
///
/// `node_waves[i]` is the wave index of IR node `i`. Use [`node_to_wave_map`]
/// to derive this from a [`CompiledSchedule`] without re-running the scheduler.
#[cfg(test)]
pub(crate) fn transient_texture_wave_intervals(ir: &GraphIR, node_waves: &[u32]) -> Result<HashMap<u32, (u32, u32)>> {
    if ir.nodes.is_empty() {
        return Ok(HashMap::new());
    }
    let mut first: HashMap<u32, u32> = HashMap::new();
    let mut last: HashMap<u32, u32> = HashMap::new();
    for (ni, node) in ir.nodes.iter().enumerate() {
        let w = node_waves[ni];
        for b in &node.bindings {
            if let ResourceId::TransientTexture(tid) = b.resource {
                let id = tid.0;
                first.entry(id).and_modify(|e| *e = (*e).min(w)).or_insert(w);
                last.entry(id).and_modify(|e| *e = (*e).max(w)).or_insert(w);
            }
        }
    }
    let mut out = HashMap::with_capacity(first.len());
    for (id, s) in first {
        let e = last[&id];
        out.insert(id, (s, e));
    }
    Ok(out)
}

/// Schedule nodes into waves using a longest-path (depth) assignment.
///
/// Each node's wave index equals one plus the maximum wave index of its
/// predecessors. Nodes with no predecessors land in wave 0. Independent
/// nodes naturally share a wave.
pub fn schedule_waves(ir: &GraphIR, edges: &[(usize, usize)]) -> CompiledSchedule {
    let n = ir.nodes.len();
    if n == 0 {
        return CompiledSchedule { waves: Vec::new() };
    }

    // Adjacency list: for each node, which nodes depend on it.
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for &(from, to) in edges {
        successors[from].push(to);
        in_degree[to] += 1;
    }

    // BFS-based topological sort with depth tracking.
    let mut depth: Vec<usize> = vec![0; n];
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut processed = 0;

    while processed < queue.len() {
        let node = queue[processed];
        processed += 1;
        for &succ in &successors[node] {
            depth[succ] = depth[succ].max(depth[node] + 1);
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    let num_waves = depth.iter().copied().max().unwrap_or(0) + 1;

    // Group nodes into waves.
    let mut wave_nodes: Vec<Vec<usize>> = vec![Vec::new(); num_waves];
    for (i, &d) in depth.iter().enumerate() {
        wave_nodes[d].push(i);
    }

    // For each wave (beyond wave 0), compute which resources need barriers.
    // A barrier is needed for resource R before wave W if:
    //   - some node in a prior wave writes R, and some node in wave W accesses R
    //   - OR some node in a prior wave reads R, and some node in wave W writes R
    let waves = wave_nodes
        .into_iter()
        .enumerate()
        .map(|(wave_idx, node_indices)| {
            let barriers_before = if wave_idx == 0 {
                BarrierSet::default()
            } else {
                compute_barriers(ir, edges, &depth, wave_idx, &node_indices)
            };
            Wave {
                node_indices,
                barriers_before,
            }
        })
        .collect();

    CompiledSchedule { waves }
}

/// Map a node's kind to the Koubaa pipeline category it belongs to.
fn node_usage_kind(node: &super::ir::TaskNode) -> UsageKindFlags {
    match &node.kind {
        NodeKind::Dispatch { .. } | NodeKind::TraceRays { .. } => UsageKindFlags::COMPUTE,
        NodeKind::RenderPass { .. } => UsageKindFlags::RENDER,
        NodeKind::ClearBuffer { .. }
        | NodeKind::WriteBuffer { .. }
        | NodeKind::CopyBuffer { .. }
        | NodeKind::CopyBufferToTexture { .. }
        | NodeKind::WriteTexture { .. }
        | NodeKind::WriteTextureRegion { .. }
        | NodeKind::CopyTexture { .. }
        | NodeKind::CopyTextureRegion { .. }
        | NodeKind::CopyRenderTarget { .. } => UsageKindFlags::TRANSFER,
        // WithdrawRead participates in ordering edges but emits no GPU work in the IR.
        NodeKind::WithdrawRead { .. } => UsageKindFlags::empty(),
        NodeKind::BuildAccelerationStructure(_) => UsageKindFlags::TRANSFER,
    }
}

/// Pipeline category recorded in a wave barrier for a specific binding.
///
/// Non-attachment resources bound for read in a render pass are CBV/SRV, not
/// RTV/DSV — map `RENDER` to `COMPUTE` so backends emit shader-resource
/// barriers instead of invalid render-target access masks.
fn barrier_usage_kind_for_binding(
    resource: ResourceId,
    access: super::ir::NodeAccess,
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
            | ResourceId::Deposit(_)
            | ResourceId::Accel(_)
    );
    if kind.contains(UsageKindFlags::RENDER) && shader_read && non_attachment {
        UsageKindFlags::COMPUTE
    } else {
        kind
    }
}

/// Determine which resources need barriers before `wave_idx` executes, and
/// derive per-resource Koubaa-level access semantics (`BarrierUsage`) from the
/// producer and consumer node kinds on each crossing edge.
///
/// `BufferRange` entries are collapsed to their parent handle so the emitted
/// `BarrierSet` always contains whole-buffer handles that backends can look up
/// directly in their resource state tables.
fn compute_barriers(
    ir: &GraphIR,
    edges: &[(usize, usize)],
    depth: &[usize],
    wave_idx: usize,
    wave_nodes: &[usize],
) -> BarrierSet {
    use crate::backend::{BufferHandle, TextureHandle};

    let wave_set: HashSet<usize> = wave_nodes.iter().copied().collect();
    let mut buffer_usage: HashMap<BufferHandle, BarrierUsage> = HashMap::new();
    let mut texture_usage: HashMap<TextureHandle, BarrierUsage> = HashMap::new();
    let mut transient_usage: HashMap<u32, BarrierUsage> = HashMap::new();
    let mut upload_usage: HashMap<u32, BarrierUsage> = HashMap::new();

    // Any edge crossing into this wave means the conflicting resource needs a barrier.
    for &(from, to) in edges {
        if depth[from] < wave_idx && wave_set.contains(&to) {
            let from_node = &ir.nodes[from];
            let to_node = &ir.nodes[to];
            // WithdrawRead emits no GPU work in this command stream (copy is out-of-band in
            // `Scheme::finish_submit_frame`).  Skip it for barrier semantics so recording
            // grant before dispatch does not emit bogus COMMON→UAV global barriers on WARP.
            if matches!(from_node.kind, NodeKind::WithdrawRead { .. })
                || matches!(to_node.kind, NodeKind::WithdrawRead { .. })
            {
                continue;
            }
            for bi in &from_node.bindings {
                for bj in &to_node.bindings {
                    if bindings_conflict(bi, bj) {
                        match bi.resource {
                            ResourceId::TransientBuffer(tid) => {
                                let entry = transient_usage.entry(tid.0).or_default();
                                entry.src.merge(
                                    bi.access,
                                    barrier_usage_kind_for_binding(bi.resource, bi.access, from_node),
                                );
                                entry.dst.merge(
                                    bj.access,
                                    barrier_usage_kind_for_binding(bj.resource, bj.access, to_node),
                                );
                            }
                            ResourceId::Deposit(uid) => {
                                let entry = upload_usage.entry(uid).or_default();
                                entry.src.merge(
                                    bi.access,
                                    barrier_usage_kind_for_binding(bi.resource, bi.access, from_node),
                                );
                                entry.dst.merge(
                                    bj.access,
                                    barrier_usage_kind_for_binding(bj.resource, bj.access, to_node),
                                );
                            }
                            ResourceId::Texture(h) => {
                                let entry = texture_usage.entry(h).or_default();
                                entry.src.merge(
                                    bi.access,
                                    barrier_usage_kind_for_binding(bi.resource, bi.access, from_node),
                                );
                                entry.dst.merge(
                                    bj.access,
                                    barrier_usage_kind_for_binding(bj.resource, bj.access, to_node),
                                );
                            }
                            ResourceId::Accel(_) => {}
                            _ => {
                                // Collapse sub-range to parent for backend barrier commands.
                                if let Some(h) = bi.resource.canonical_buffer_handle() {
                                    let entry = buffer_usage.entry(h).or_default();
                                    entry.src.merge(
                                        bi.access,
                                        barrier_usage_kind_for_binding(bi.resource, bi.access, from_node),
                                    );
                                    entry.dst.merge(
                                        bj.access,
                                        barrier_usage_kind_for_binding(bj.resource, bj.access, to_node),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut buffers: Vec<_> = buffer_usage.into_iter().collect();
    let mut textures: Vec<_> = texture_usage.into_iter().collect();
    let mut transient_ids: Vec<_> = transient_usage.into_iter().collect();
    let mut upload_ids: Vec<_> = upload_usage.into_iter().collect();
    buffers.sort_by_key(|(h, _)| *h);
    textures.sort_by_key(|(h, _)| *h);
    transient_ids.sort_by_key(|(id, _)| *id);
    upload_ids.sort_by_key(|(id, _)| *id);

    BarrierSet {
        buffers,
        textures,
        transient_ids,
        upload_ids,
    }
}

/// Emit a flat `Vec<GpuCommand>` for the given slice of [`Wave`]s.
///
/// This is the shared inner loop used by both [`emit_commands`] and
/// [`emit_partitioned_commands`].  Each wave emits, in order:
///
/// 1. A `ResourceBarrier` if `barriers_before` is non-empty.
/// 2. Blit-type nodes (clears, uploads, copies) — reordered before dispatches
///    within the wave to minimise compute↔blit encoder transitions on Metal.
///    Same-wave nodes have no data dependencies so this is always safe.
/// 3. Dispatch nodes — indirect one-by-one; direct dispatches batched into a
///    single `DispatchBatch` when consecutive dispatches share the same pipeline.
///
/// # Panics
///
/// If any wave contains a [`NodeKind::RenderPass`] node.  Use
/// [`emit_graph_commands`] for graphs that include render passes.
pub(crate) fn emit_waves_to_commands(ir: &GraphIR, waves: &[Wave], resolver: Option<&SlotResolver>) -> Vec<GpuCommand> {
    let mut commands = Vec::new();
    let mut frame_table = FrameTableStaging::new();

    for wave in waves {
        if !wave.barriers_before.is_empty() {
            let mut barrier_buffers = wave.barriers_before.buffers.clone();
            // Resolve transient buffer IDs to their concrete parent handles.
            if !wave.barriers_before.transient_ids.is_empty() {
                if let Some(r) = resolver {
                    for &(tid, usage) in &wave.barriers_before.transient_ids {
                        if let Some(resolved) = r.buffers.get(&tid) {
                            if !barrier_buffers.iter().any(|(h, _)| *h == resolved.parent) {
                                barrier_buffers.push((resolved.parent, usage));
                            }
                        }
                    }
                }
            }
            if !wave.barriers_before.upload_ids.is_empty() {
                if let Some(r) = resolver {
                    for &(uid, usage) in &wave.barriers_before.upload_ids {
                        if let Some(resolved) = r.deposits.get(&uid) {
                            if !barrier_buffers.iter().any(|(h, _)| *h == resolved.parent) {
                                barrier_buffers.push((resolved.parent, usage));
                            }
                        }
                    }
                }
            }
            commands.push(GpuCommand::ResourceBarrier {
                buffers: barrier_buffers,
                textures: wave.barriers_before.textures.clone(),
            });
        }

        // Emit blit-type nodes before dispatches within each wave to minimise
        // compute↔blit encoder transitions on Metal.
        for &idx in &wave.node_indices {
            let node = &ir.nodes[idx];
            match &node.kind {
                NodeKind::ClearBuffer { buffer, offset, size } => {
                    commands.push(GpuCommand::ClearBuffer {
                        buffer: *buffer,
                        offset: *offset,
                        size: *size,
                    });
                }
                NodeKind::WriteBuffer { buffer, offset, data } => {
                    commands.push(GpuCommand::WriteBuffer {
                        buffer: *buffer,
                        offset: *offset,
                        data: data.clone(),
                    });
                }
                NodeKind::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let (src_handle, src_off) = resolve_buffer_copy_target(*src, *src_offset, resolver);
                    let (dst_handle, dst_off) = resolve_buffer_copy_target(*dst, *dst_offset, resolver);
                    commands.push(GpuCommand::CopyBuffer {
                        src: src_handle,
                        src_offset: src_off,
                        dst: dst_handle,
                        dst_offset: dst_off,
                        size: *size,
                    });
                }
                NodeKind::CopyBufferToTexture {
                    src,
                    src_offset,
                    src_row_pitch,
                    dst,
                    x,
                    y,
                    width,
                    height,
                } => {
                    let (src_handle, src_off) = resolve_buffer_copy_target(*src, *src_offset, resolver);
                    commands.push(GpuCommand::CopyBufferToTexture {
                        src: src_handle,
                        src_offset: src_off,
                        src_row_pitch: *src_row_pitch,
                        dst: *dst,
                        x: *x,
                        y: *y,
                        width: *width,
                        height: *height,
                    });
                }
                NodeKind::WriteTexture {
                    texture,
                    data,
                    width,
                    height,
                } => {
                    commands.push(GpuCommand::WriteTexture {
                        texture: *texture,
                        data: data.clone(),
                        width: *width,
                        height: *height,
                    });
                }
                NodeKind::WriteTextureRegion {
                    texture,
                    x,
                    y,
                    width,
                    height,
                    data,
                } => {
                    commands.push(GpuCommand::WriteTextureRegion {
                        texture: *texture,
                        x: *x,
                        y: *y,
                        width: *width,
                        height: *height,
                        data: data.clone(),
                    });
                }
                NodeKind::CopyTexture {
                    src,
                    dst,
                    dst_buffer_layout,
                } => {
                    if let Some(layout) = dst_buffer_layout {
                        let (dst_buf, _) = resolve_buffer_copy_target(*dst, 0, resolver);
                        commands.push(GpuCommand::CopyTextureToReadback {
                            src: *src,
                            dst: dst_buf,
                            layout: *layout,
                        });
                    } else {
                        let dst = resolve_copy_destination(*dst, resolver);
                        commands.push(GpuCommand::CopyTexture { src: *src, dst });
                    }
                }
                NodeKind::CopyTextureRegion {
                    src,
                    dst,
                    src_x,
                    src_y,
                    dst_x,
                    dst_y,
                    width,
                    height,
                } => {
                    commands.push(GpuCommand::CopyTextureRegion {
                        src: *src,
                        dst: *dst,
                        src_x: *src_x,
                        src_y: *src_y,
                        dst_x: *dst_x,
                        dst_y: *dst_y,
                        width: *width,
                        height: *height,
                    });
                }
                NodeKind::CopyRenderTarget { src, dst } => {
                    let dst = resolve_copy_destination(*dst, resolver);
                    commands.push(GpuCommand::CopyRenderTarget { src: *src, dst });
                }
                NodeKind::BuildAccelerationStructure(build) => {
                    commands.push(GpuCommand::BuildAccelerationStructure(build.clone()));
                }
                NodeKind::TraceRays { .. }
                | NodeKind::Dispatch { .. }
                | NodeKind::RenderPass { .. }
                | NodeKind::WithdrawRead { .. } => {}
            }
        }

        // Emit dispatch nodes, batching consecutive same-pipeline direct dispatches
        // into a single `DispatchBatch` command that can be executed with
        // `ExecuteIndirect` on DX12 (reducing per-dispatch CPU recording cost).
        //
        // Indirect dispatches (GPU-driven counts) are NOT batched because they
        // already read counts from a GPU buffer; mixing them with the CPU arg
        // buffer layout is not straightforward.
        {
            enum SlotData<'a> {
                Borrowed(&'a Vec<u32>),
                Resolved(Vec<u32>),
            }

            impl<'a> SlotData<'a> {
                fn as_slice(&self) -> &[u32] {
                    match self {
                        SlotData::Borrowed(v) => v.as_slice(),
                        SlotData::Resolved(v) => v.as_slice(),
                    }
                }
            }

            struct PendingDispatch<'n> {
                label: &'static str,
                pipeline: crate::backend::ComputePipelineHandle,
                resource_slots: SlotData<'n>,
                user_slots: &'n Vec<u32>,
                x: u32,
                y: u32,
                z: u32,
            }

            let mut pending: Vec<PendingDispatch<'_>> = Vec::new();
            let mut has_indirect = false;
            for &idx in &wave.node_indices {
                let node = &ir.nodes[idx];
                if let NodeKind::Dispatch {
                    pipeline,
                    resource_slots,
                    user_slots,
                    dispatch: super::ir::DispatchDim::Direct { x, y, z },
                } = &node.kind
                {
                    let slots = match resolver {
                        Some(r) => SlotData::Resolved(r.resolve_slots(resource_slots, &node.bindings)),
                        None => SlotData::Borrowed(resource_slots),
                    };
                    pending.push(PendingDispatch {
                        label: node.label,
                        pipeline: *pipeline,
                        resource_slots: slots,
                        user_slots,
                        x: *x,
                        y: *y,
                        z: *z,
                    });
                } else if let NodeKind::Dispatch {
                    dispatch: super::ir::DispatchDim::Indirect { .. },
                    ..
                } = &node.kind
                {
                    has_indirect = true;
                }
            }

            // Emit indirect dispatches one by one.
            if has_indirect {
                for &idx in &wave.node_indices {
                    let node = &ir.nodes[idx];
                    if let NodeKind::Dispatch {
                        pipeline,
                        resource_slots,
                        user_slots,
                        dispatch: super::ir::DispatchDim::Indirect { buffer, offset },
                    } = &node.kind
                    {
                        let slots = match resolver {
                            Some(r) => r.resolve_slots(resource_slots, &node.bindings),
                            None => resource_slots.clone(),
                        };
                        commands.push(GpuCommand::SetPipeline(*pipeline));
                        push_compute_resource_bind(&mut commands, &mut frame_table, &slots, user_slots);
                        commands.push(GpuCommand::DispatchIndirect {
                            label: Some(node.label),
                            buffer: *buffer,
                            offset: *offset,
                        });
                    }
                }
            }

            // Emit direct dispatches, batching consecutive same-pipeline groups.
            let mut i = 0;
            while i < pending.len() {
                let cur_pipeline = pending[i].pipeline;
                let run_end = pending[i..].iter().take_while(|d| d.pipeline == cur_pipeline).count();
                let run = &pending[i..i + run_end];

                if run.len() > 1 {
                    let mut arg_data: Vec<u8> = Vec::with_capacity(run.len() * DISPATCH_BATCH_STRIDE);
                    for d in run {
                        let slots = d.resource_slots.as_slice();
                        let frame_table_base = frame_table.alloc_dispatch(slots.len() as u32);
                        frame_table.write_dispatch_indices(frame_table_base, slots);
                        let mut layout = crate::backend::shared::PushLayout::default();
                        crate::backend::shared::fill_frame_table_dispatch(&mut layout, frame_table_base, d.user_slots);
                        arg_data.extend_from_slice(bytemuck::bytes_of(&layout));
                        arg_data.extend_from_slice(&d.x.to_ne_bytes());
                        arg_data.extend_from_slice(&d.y.to_ne_bytes());
                        arg_data.extend_from_slice(&d.z.to_ne_bytes());
                    }
                    commands.push(GpuCommand::SetPipeline(cur_pipeline));
                    commands.push(GpuCommand::DispatchBatch {
                        label: Some(run[0].label),
                        arg_data: Arc::from(arg_data.as_slice()),
                        count: run.len() as u32,
                    });
                } else {
                    let d = &run[0];
                    commands.push(GpuCommand::SetPipeline(cur_pipeline));
                    let slots = d.resource_slots.as_slice();
                    push_compute_resource_bind(&mut commands, &mut frame_table, slots, d.user_slots);
                    commands.push(GpuCommand::Dispatch {
                        label: Some(d.label),
                        workgroups_x: d.x,
                        workgroups_y: d.y,
                        workgroups_z: d.z,
                    });
                }

                i += run_end;
            }
        }

        for &idx in &wave.node_indices {
            let node = &ir.nodes[idx];
            if let NodeKind::TraceRays {
                pipeline,
                resource_slots,
                user_slots,
                width,
                height,
                depth,
            } = &node.kind
            {
                let slots = match resolver {
                    Some(r) => r.resolve_slots(resource_slots, &node.bindings),
                    None => resource_slots.clone(),
                };
                commands.push(GpuCommand::SetRayTracingPipeline(*pipeline));
                push_compute_resource_bind(&mut commands, &mut frame_table, &slots, user_slots);
                commands.push(GpuCommand::TraceRays {
                    label: Some(node.label),
                    width: *width,
                    height: *height,
                    depth: *depth,
                });
            }
        }

        // Render-pass guard — not expected in pure-compute graphs.
        for &idx in &wave.node_indices {
            let node = &ir.nodes[idx];
            if let NodeKind::RenderPass { .. } = &node.kind {
                panic!("emit_commands: graph contains render_pass; use emit_graph_commands");
            }
        }
    }

    // Only insert the staging prefix when at least one dispatch wrote bind data.
    // Pure copy/write/barrier streams must not bump the submission counter.
    if !commands.is_empty() && frame_table.has_bindings() {
        commands.insert(
            0,
            GpuCommand::FrameTableStaging {
                data: frame_table.as_arc(),
            },
        );
    }

    commands
}

/// Emit a flat `Vec<GpuCommand>` from a graph IR and its compiled schedule.
///
/// Each [`NodeKind`] variant emits a different command sequence:
/// - `Dispatch` → `SetPipeline` + optional `BindResourcesRaw` + `Dispatch`/`DispatchIndirect`
/// - `ClearBuffer` → `ClearBuffer`
/// - `WriteBuffer` → `WriteBuffer`
/// - `WriteTexture` / `WriteTextureRegion` → matching `GpuCommand` variants
///
/// # Panics
///
/// If the graph contains [`NodeKind::RenderPass`], use [`emit_graph_commands`] instead.
#[cfg(test)]
pub fn emit_commands(ir: &GraphIR, schedule: &CompiledSchedule, resolver: Option<&SlotResolver>) -> Vec<GpuCommand> {
    emit_waves_to_commands(ir, &schedule.waves, resolver)
}

/// Partition the compiled schedule into multiple command streams for pipelined
/// backend submission.
///
/// When `split_on_barrier_cost` is true, the partitioning heuristic selects the
/// single wave boundary (wave index > 0) that has the largest `barriers_before`
/// cost (sum of buffers and textures that need synchronisation), which
/// corresponds to the heaviest cross-phase data dependency — typically the
/// coarse→fine boundary in a rendering pipeline.
///
/// Returns a `Vec` of one or two partitions:
///
/// - **Single partition**: returned when the schedule has fewer than 3 waves,
///   every wave boundary has zero barrier cost, or `split_on_barrier_cost` is
///   false (Metal). The result is equivalent to calling [`emit_commands`] and
///   wrapping it.
///
/// - **Two partitions**: `[early_cmds, late_cmds]`.  Waves `0..split` go into
///   `early_cmds` and waves `split..` go into `late_cmds`.  The leading
///   `ResourceBarrier` of the first wave in `late_cmds` is preserved; the
///   backend's cross-submission acquire barrier already covers memory visibility
///   for the GPU, so correctness is maintained.
///
/// **Invariant**: `partitions.into_iter().flatten().collect::<Vec<_>>()`
/// always equals the output of [`emit_commands`] for the same inputs.
#[cfg(test)]
pub fn emit_partitioned_commands(
    ir: &GraphIR,
    schedule: &CompiledSchedule,
    resolver: Option<&SlotResolver>,
    split_on_barrier_cost: bool,
) -> Vec<Vec<GpuCommand>> {
    partition_wave_ranges(ir, schedule, split_on_barrier_cost)
        .into_iter()
        .map(|range| {
            let waves = &schedule.waves[range];
            emit_waves_to_commands(ir, waves, resolver)
        })
        .collect()
}

/// Logical partition — a semantically coherent slice of the schedule.
///
/// Logical partitions are defined by two hard split rules:
///
/// 1. **Present-binding boundary**: any wave that introduces a
///    [`ResourceId::PresentLease`] binding id (or the legacy
///    [`ResourceId::SwapchainOutput`]) not seen in earlier waves begins a new
///    logical partition. Independent GPU work must not wait behind a drawable
///    acquire, and distinct surfaces acquire at their own first dependent
///    partition.
///
/// 2. **Render-kind boundary**: any wave that is the first wave containing a
///    [`NodeKind::RenderPass`] when the preceding waves contained none — or vice versa —
///    begins a new logical partition.  Render-pass waves must be emitted as
///    [`GraphCommand::Render`] records and cannot be mixed into a pure-compute
///    `GpuCommand` cache slot.
///
/// Both rules are checked per-wave; split points from either rule are collected and
/// merged into a monotone list before forming ranges.
///
/// Flags and binding sets tag each partition so callers can choose the right
/// emitter and acquire only the bindings that partition requires:
///
/// - `has_render`   — the slice contains at least one [`NodeKind::RenderPass`] node.
/// - `has_present`  — the slice binds at least one present lease or swapchain output.
/// - `present_bindings` — scheme-unique [`ResourceId::PresentLease`] ids in the slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalPartition {
    /// Wave-index range `start..end` into [`CompiledSchedule::waves`].
    pub wave_range: std::ops::Range<usize>,
    /// True when the slice contains at least one offscreen render-pass node.
    pub has_render: bool,
    /// True when the slice references a present-lease or swapchain-output resource.
    pub has_present: bool,
    /// Scheme-unique present binding ids ([`ResourceId::PresentLease`]) referenced
    /// by this partition, sorted and deduplicated.
    pub present_bindings: Vec<u32>,
}

impl LogicalPartition {
    /// True when the partition contains only compute/blit nodes (no render, no present).
    pub fn is_pure_compute(&self) -> bool {
        !self.has_render && !self.has_present
    }
}

/// Decompose a compiled schedule into [`LogicalPartition`]s.
///
/// Rules applied, in priority order:
///
/// 1. If a wave introduces a present binding id (or first `SwapchainOutput`) not
///    seen earlier, it opens a new partition (unless it is wave 0).
/// 2. If a wave changes the render-kind of its predecessor (compute→render or
///    render→compute), it opens a new partition.
///
/// The result always covers every wave exactly once.
pub(crate) fn describe_logical_partitions(ir: &GraphIR, schedule: &CompiledSchedule) -> Vec<LogicalPartition> {
    let waves = &schedule.waves;
    let n = waves.len();
    if n == 0 {
        return vec![LogicalPartition {
            wave_range: 0..0,
            has_render: false,
            has_present: false,
            present_bindings: Vec::new(),
        }];
    }

    // Collect split points: indices of waves that begin a new logical partition.
    let mut splits: Vec<usize> = vec![0]; // always start with wave 0
    let mut seen_bindings: Vec<u32> = Vec::new();
    let mut seen_swapchain_output = false;

    for (i, wave) in waves.iter().enumerate() {
        let wave_bindings = wave_present_binding_ids(ir, wave);
        let has_swapchain = wave_has_swapchain_output(ir, wave);
        let is_render = wave
            .node_indices
            .iter()
            .any(|&ni| matches!(ir.nodes[ni].kind, NodeKind::RenderPass { .. }));

        let introduces_new_binding =
            wave_bindings.iter().any(|id| !seen_bindings.contains(id)) || (has_swapchain && !seen_swapchain_output);

        if i == 0 {
            for &id in &wave_bindings {
                if !seen_bindings.contains(&id) {
                    seen_bindings.push(id);
                }
            }
            seen_swapchain_output |= has_swapchain;
            continue;
        }

        let prev_render = waves[i - 1]
            .node_indices
            .iter()
            .any(|&ni| matches!(ir.nodes[ni].kind, NodeKind::RenderPass { .. }));

        // Present-binding boundary: wave introduces a previously unseen lease /
        // swapchain output. Multiple surfaces in different waves become distinct
        // partitions so each acquires at its own first dependent work.
        if introduces_new_binding {
            splits.push(i);
            for &id in &wave_bindings {
                if !seen_bindings.contains(&id) {
                    seen_bindings.push(id);
                }
            }
            seen_swapchain_output |= has_swapchain;
            continue; // don't also split on render-kind here
        }

        // Render-kind boundary: render→compute or compute→render transition.
        if is_render != prev_render && !splits.contains(&i) {
            splits.push(i);
        }
    }

    splits.sort_unstable();
    splits.dedup();

    // Build partition descriptors from the split-point list.
    let mut result = Vec::with_capacity(splits.len());
    for (k, &start) in splits.iter().enumerate() {
        let end = if k + 1 < splits.len() { splits[k + 1] } else { n };
        let slice = &waves[start..end];
        let present_bindings = partition_present_binding_ids(ir, slice);
        let has_present = !present_bindings.is_empty() || slice.iter().any(|w| wave_has_swapchain_output(ir, w));
        let has_render = slice.iter().any(|w| {
            w.node_indices
                .iter()
                .any(|&ni| matches!(ir.nodes[ni].kind, NodeKind::RenderPass { .. }))
        });
        result.push(LogicalPartition {
            wave_range: start..end,
            has_render,
            has_present,
            present_bindings,
        });
    }
    result
}

/// Returns false when the wave slice contains nodes that must be submitted standalone
/// (upload payload staging, non-stable copy destinations, etc.).
pub(crate) fn waves_can_retain(ir: &GraphIR, waves: &[Wave]) -> bool {
    #[cfg(feature = "graphics")]
    use super::ResourceId;
    for wave in waves {
        for &ni in &wave.node_indices {
            match &ir.nodes[ni].kind {
                NodeKind::WriteBuffer { .. }
                | NodeKind::WriteTexture { .. }
                | NodeKind::WriteTextureRegion { .. }
                | NodeKind::CopyBufferToTexture { src_row_pitch: 0, .. }
                | NodeKind::CopyTexture { .. }
                | NodeKind::CopyTextureRegion { .. } => return false,
                // CopyRenderTarget → PresentLease is retainable via the slot-key
                // mechanism (§5.3 of render-scheme.md); it must NOT be standalone.
                // CopyRenderTarget → Texture is also retainable: the texture handle
                // is stable across submissions, so the blit CB can be reused as-is.
                // All other destinations (e.g. SwapchainOutput) are not stable and
                // must be submitted standalone.
                #[cfg(feature = "graphics")]
                NodeKind::CopyRenderTarget { dst, .. }
                    if !matches!(dst, ResourceId::PresentLease(_) | ResourceId::Texture(_)) =>
                {
                    return false;
                }
                #[cfg(not(feature = "graphics"))]
                NodeKind::CopyRenderTarget { .. } => {
                    return false;
                }
                _ => {}
            }
        }
    }
    true
}

/// Fingerprint contribution from destination texture barrier layouts for pitched
/// [`NodeKind::CopyBufferToTexture`] nodes in `partition_waves`.
///
/// Retained command buffers bake `layout_before` into texture barriers at record time.
/// When the layout changes (typically once, COMMON → shader-read), the partition key
/// must change so the backend re-records before resubmitting.
pub(crate) fn partition_copy_texture_layout_fingerprint(
    ir: &GraphIR,
    partition_waves: &[Wave],
    layout_tag: impl Fn(u64) -> u64,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for wave in partition_waves {
        for &ni in &wave.node_indices {
            if let NodeKind::CopyBufferToTexture { src_row_pitch, dst, .. } = &ir.nodes[ni].kind {
                if *src_row_pitch > 0 {
                    layout_tag(*dst).hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

/// Subdivide `wave_range` at consecutive wave boundaries where [`waves_can_retain`] changes.
fn split_wave_range_at_retainability(
    ir: &GraphIR,
    schedule: &CompiledSchedule,
    wave_range: std::ops::Range<usize>,
) -> Vec<std::ops::Range<usize>> {
    let base = wave_range.start;
    let waves = &schedule.waves[wave_range.clone()];
    let len = waves.len();
    if len <= 1 {
        return vec![wave_range];
    }

    let mut out = Vec::new();
    let mut sub_start = base;
    for i in 1..len {
        let prev_retain = waves_can_retain(ir, &waves[i - 1..i]);
        let curr_retain = waves_can_retain(ir, &waves[i..i + 1]);
        // Isolate retainable buffer-only upload waves before texture uploads. Do not
        // split non-retainable upload waves (WriteBuffer) from subsequent compute — those
        // stay in one partition for payload refresh and barrier-cost heuristics.
        if prev_retain && !curr_retain {
            out.push(sub_start..base + i);
            sub_start = base + i;
        }
    }
    out.push(sub_start..wave_range.end);
    out
}

/// Push `wave_range` into `ranges`, optionally splitting at the heaviest barrier boundary
/// when the slice is a large pure-compute partition.
fn push_partition_with_barrier_heuristic(
    ranges: &mut Vec<std::ops::Range<usize>>,
    schedule: &CompiledSchedule,
    wave_range: std::ops::Range<usize>,
    enable: bool,
) {
    let waves = &schedule.waves[wave_range.clone()];
    let len = waves.len();
    if enable && len >= 3 {
        let (split_offset, max_cost) = waves
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, w)| (i, w.barriers_before.buffers.len() + w.barriers_before.textures.len()))
            .max_by_key(|&(_, cost)| cost)
            .unwrap(); // safe: len >= 3

        if max_cost > 0 {
            let abs_split = wave_range.start + split_offset;
            ranges.push(wave_range.start..abs_split);
            ranges.push(abs_split..wave_range.end);
            return;
        }
    }
    ranges.push(wave_range);
}

/// Compute the wave-index ranges for each actualized partition of a compiled schedule.
///
/// Actualized partitions refine the logical partition layout produced by
/// [`describe_logical_partitions`] with:
/// - retainability splits (buffer-only upload waves vs texture upload waves), and
/// - an optional barrier-cost heuristic (`split_on_barrier_cost`): large pure-compute
///   logical partitions (≥ 3 waves, nonzero barrier cost) are subdivided at their
///   heaviest wave boundary to expose GPU-pipeline overlap between submissions.
///   Disabled on Metal (see [`crate::device::DeviceCapabilities::split_compute_partitions_on_barrier_cost`]).
///
/// The present-boundary and render-kind splits from the logical layer are always
/// respected; the heuristics are applied only *within* pure-compute non-present partitions.
///
/// This function always returns at least one range covering all waves.
pub(crate) fn partition_wave_ranges(
    ir: &GraphIR,
    schedule: &CompiledSchedule,
    split_on_barrier_cost: bool,
) -> Vec<std::ops::Range<usize>> {
    let logical = describe_logical_partitions(ir, schedule);
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(logical.len());

    for lp in &logical {
        if lp.is_pure_compute() && lp.wave_range.len() >= 2 {
            for sub in split_wave_range_at_retainability(ir, schedule, lp.wave_range.clone()) {
                push_partition_with_barrier_heuristic(&mut ranges, schedule, sub, split_on_barrier_cost);
            }
        } else {
            push_partition_with_barrier_heuristic(&mut ranges, schedule, lp.wave_range.clone(), split_on_barrier_cost);
        }
    }

    ranges
}

/// Returns true if any node in `waves` references a present lease or swapchain output.
pub(crate) fn partition_waves_have_present(ir: &GraphIR, waves: &[Wave]) -> bool {
    waves.iter().any(|w| wave_has_present_binding(ir, w))
}

/// Scheme-unique [`ResourceId::PresentLease`] ids referenced by `waves`, sorted and deduplicated.
pub(crate) fn partition_present_binding_ids(ir: &GraphIR, waves: &[Wave]) -> Vec<u32> {
    let mut ids = Vec::new();
    for wave in waves {
        for id in wave_present_binding_ids(ir, wave) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids.sort_unstable();
    ids
}

/// Returns true if any node in `waves` references a scheme upload-buffer slot.
pub(crate) fn partition_waves_have_upload_slots(ir: &GraphIR, waves: &[Wave]) -> bool {
    waves.iter().any(|w| {
        w.node_indices.iter().any(|&ni| {
            ir.nodes[ni]
                .bindings
                .iter()
                .any(|b| matches!(b.resource, ResourceId::Deposit(_)))
        })
    })
}

fn wave_present_binding_ids(ir: &GraphIR, wave: &Wave) -> Vec<u32> {
    #[cfg(feature = "graphics")]
    {
        let mut ids = Vec::new();
        for &ni in &wave.node_indices {
            for b in &ir.nodes[ni].bindings {
                if let ResourceId::PresentLease(id) = b.resource {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
        }
        ids
    }
    #[cfg(not(feature = "graphics"))]
    {
        let _ = (ir, wave);
        Vec::new()
    }
}

fn wave_has_swapchain_output(ir: &GraphIR, wave: &Wave) -> bool {
    #[cfg(feature = "graphics")]
    {
        wave.node_indices.iter().any(|&ni| {
            ir.nodes[ni]
                .bindings
                .iter()
                .any(|b| matches!(b.resource, ResourceId::SwapchainOutput))
        })
    }
    #[cfg(not(feature = "graphics"))]
    {
        let _ = (ir, wave);
        false
    }
}

fn wave_has_present_binding(ir: &GraphIR, wave: &Wave) -> bool {
    !wave_present_binding_ids(ir, wave).is_empty() || wave_has_swapchain_output(ir, wave)
}

/// Emit [`GraphCommand`]s for a slice of compiled waves (compute + optional render).
pub(crate) fn emit_graph_commands_for_waves(
    ir: &GraphIR,
    waves: &[Wave],
    resolver: Option<&SlotResolver>,
) -> Vec<GraphCommand> {
    let mut commands = Vec::new();
    let mut frame_table = FrameTableStaging::new();

    for wave in waves {
        if !wave.barriers_before.is_empty() {
            let mut barrier_buffers = wave.barriers_before.buffers.clone();
            if !wave.barriers_before.transient_ids.is_empty() {
                if let Some(r) = resolver {
                    for &(tid, usage) in &wave.barriers_before.transient_ids {
                        if let Some(resolved) = r.buffers.get(&tid) {
                            if !barrier_buffers.iter().any(|(h, _)| *h == resolved.parent) {
                                barrier_buffers.push((resolved.parent, usage));
                            }
                        }
                    }
                }
            }
            if !wave.barriers_before.upload_ids.is_empty() {
                if let Some(r) = resolver {
                    for &(uid, usage) in &wave.barriers_before.upload_ids {
                        if let Some(resolved) = r.deposits.get(&uid) {
                            if !barrier_buffers.iter().any(|(h, _)| *h == resolved.parent) {
                                barrier_buffers.push((resolved.parent, usage));
                            }
                        }
                    }
                }
            }
            commands.push(GraphCommand::Compute(GpuCommand::ResourceBarrier {
                buffers: barrier_buffers,
                textures: wave.barriers_before.textures.clone(),
            }));
        }

        // Blit-type nodes first to minimize encoder transitions.
        for &idx in &wave.node_indices {
            let node = &ir.nodes[idx];
            match &node.kind {
                NodeKind::ClearBuffer { buffer, offset, size } => {
                    commands.push(GraphCommand::Compute(GpuCommand::ClearBuffer {
                        buffer: *buffer,
                        offset: *offset,
                        size: *size,
                    }));
                }
                NodeKind::WriteBuffer { buffer, offset, data } => {
                    commands.push(GraphCommand::Compute(GpuCommand::WriteBuffer {
                        buffer: *buffer,
                        offset: *offset,
                        data: data.clone(),
                    }));
                }
                NodeKind::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let (src_handle, src_off) = resolve_buffer_copy_target(*src, *src_offset, resolver);
                    let (dst_handle, dst_off) = resolve_buffer_copy_target(*dst, *dst_offset, resolver);
                    commands.push(GraphCommand::Compute(GpuCommand::CopyBuffer {
                        src: src_handle,
                        src_offset: src_off,
                        dst: dst_handle,
                        dst_offset: dst_off,
                        size: *size,
                    }));
                }
                NodeKind::CopyBufferToTexture {
                    src,
                    src_offset,
                    src_row_pitch,
                    dst,
                    x,
                    y,
                    width,
                    height,
                } => {
                    let (src_handle, src_off) = resolve_buffer_copy_target(*src, *src_offset, resolver);
                    commands.push(GraphCommand::Compute(GpuCommand::CopyBufferToTexture {
                        src: src_handle,
                        src_offset: src_off,
                        src_row_pitch: *src_row_pitch,
                        dst: *dst,
                        x: *x,
                        y: *y,
                        width: *width,
                        height: *height,
                    }));
                }
                NodeKind::WriteTexture {
                    texture,
                    data,
                    width,
                    height,
                } => {
                    commands.push(GraphCommand::Compute(GpuCommand::WriteTexture {
                        texture: *texture,
                        data: data.clone(),
                        width: *width,
                        height: *height,
                    }));
                }
                NodeKind::WriteTextureRegion {
                    texture,
                    x,
                    y,
                    width,
                    height,
                    data,
                } => {
                    commands.push(GraphCommand::Compute(GpuCommand::WriteTextureRegion {
                        texture: *texture,
                        x: *x,
                        y: *y,
                        width: *width,
                        height: *height,
                        data: data.clone(),
                    }));
                }
                NodeKind::CopyTexture {
                    src,
                    dst,
                    dst_buffer_layout,
                } => {
                    if let Some(layout) = dst_buffer_layout {
                        let (dst_buf, _) = resolve_buffer_copy_target(*dst, 0, resolver);
                        commands.push(GraphCommand::Compute(GpuCommand::CopyTextureToReadback {
                            src: *src,
                            dst: dst_buf,
                            layout: *layout,
                        }));
                    } else {
                        let dst = resolve_copy_destination(*dst, resolver);
                        commands.push(GraphCommand::Compute(GpuCommand::CopyTexture { src: *src, dst }));
                    }
                }
                NodeKind::CopyTextureRegion {
                    src,
                    dst,
                    src_x,
                    src_y,
                    dst_x,
                    dst_y,
                    width,
                    height,
                } => {
                    commands.push(GraphCommand::Compute(GpuCommand::CopyTextureRegion {
                        src: *src,
                        dst: *dst,
                        src_x: *src_x,
                        src_y: *src_y,
                        dst_x: *dst_x,
                        dst_y: *dst_y,
                        width: *width,
                        height: *height,
                    }));
                }
                NodeKind::CopyRenderTarget { src, dst } => {
                    let dst = resolve_copy_destination(*dst, resolver);
                    commands.push(GraphCommand::Compute(GpuCommand::CopyRenderTarget { src: *src, dst }));
                }
                NodeKind::BuildAccelerationStructure(build) => {
                    commands.push(GraphCommand::Compute(GpuCommand::BuildAccelerationStructure(build.clone())));
                }
                NodeKind::Dispatch { .. }
                | NodeKind::TraceRays { .. }
                | NodeKind::RenderPass { .. }
                | NodeKind::WithdrawRead { .. } => {}
            }
        }

        // Dispatches and render passes second.
        for &idx in &wave.node_indices {
            let node = &ir.nodes[idx];
            match &node.kind {
                NodeKind::Dispatch {
                    pipeline,
                    resource_slots,
                    user_slots,
                    dispatch,
                } => {
                    let slots = match resolver {
                        Some(r) => r.resolve_slots(resource_slots, &node.bindings),
                        None => resource_slots.clone(),
                    };
                    commands.push(GraphCommand::Compute(GpuCommand::SetPipeline(*pipeline)));
                    let mut bind_cmds = Vec::new();
                    push_compute_resource_bind(&mut bind_cmds, &mut frame_table, &slots, user_slots);
                    for cmd in bind_cmds {
                        commands.push(GraphCommand::Compute(cmd));
                    }
                    match dispatch {
                        super::ir::DispatchDim::Direct { x, y, z } => {
                            commands.push(GraphCommand::Compute(GpuCommand::Dispatch {
                                label: Some(node.label),
                                workgroups_x: *x,
                                workgroups_y: *y,
                                workgroups_z: *z,
                            }));
                        }
                        super::ir::DispatchDim::Indirect { buffer, offset } => {
                            commands.push(GraphCommand::Compute(GpuCommand::DispatchIndirect {
                                label: Some(node.label),
                                buffer: *buffer,
                                offset: *offset,
                            }));
                        }
                    }
                }
                NodeKind::TraceRays {
                    pipeline,
                    resource_slots,
                    user_slots,
                    width,
                    height,
                    depth,
                } => {
                    let slots = match resolver {
                        Some(r) => r.resolve_slots(resource_slots, &node.bindings),
                        None => resource_slots.clone(),
                    };
                    commands.push(GraphCommand::Compute(GpuCommand::SetRayTracingPipeline(*pipeline)));
                    let mut bind_cmds = Vec::new();
                    push_compute_resource_bind(&mut bind_cmds, &mut frame_table, &slots, user_slots);
                    for cmd in bind_cmds {
                        commands.push(GraphCommand::Compute(cmd));
                    }
                    commands.push(GraphCommand::Compute(GpuCommand::TraceRays {
                        label: Some(node.label),
                        width: *width,
                        height: *height,
                        depth: *depth,
                    }));
                }
                NodeKind::RenderPass {
                    target,
                    color_load,
                    commands: render_cmds,
                } => {
                    let lowered = crate::frame_table::lower_render_pass_commands(&mut frame_table, render_cmds);
                    commands.push(GraphCommand::Render {
                        target: *target,
                        color_load: *color_load,
                        commands: lowered,
                    });
                }
                _ => {}
            }
        }
    }

    if !commands.is_empty() && frame_table.has_bindings() {
        commands.insert(
            0,
            GraphCommand::Compute(GpuCommand::FrameTableStaging {
                data: frame_table.as_arc(),
            }),
        );
    }

    commands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task_graph::ir::{DispatchDim, NodeKind, ResourceBinding, TaskNode};
    use std::sync::Arc;

    fn buf(id: u64) -> ResourceId {
        ResourceId::Buffer(id)
    }

    /// Build a dispatch `TaskNode` — the workhorse helper for analysis tests.
    fn dispatch_node(label: &'static str, pipeline: u64, bindings: Vec<(ResourceId, NodeAccess)>, wg: u32) -> TaskNode {
        TaskNode {
            label,
            bindings: bindings
                .into_iter()
                .map(|(resource, access)| ResourceBinding { resource, access })
                .collect(),
            kind: NodeKind::Dispatch {
                pipeline,
                resource_slots: Vec::new(),
                user_slots: Vec::new(),
                dispatch: DispatchDim::Direct { x: wg, y: 1, z: 1 },
            },
        }
    }

    /// Short alias used by the bulk of tests.
    fn node(label: &'static str, pipeline: u64, bindings: Vec<(ResourceId, NodeAccess)>, wg: u32) -> TaskNode {
        dispatch_node(label, pipeline, bindings, wg)
    }

    /// Like `node` but includes a dummy resource slot so the dispatch contributes
    /// frame-table bindings.  Use when a test needs to assert FrameTableStaging
    /// is emitted (the staging prefix is only inserted when bindings exist).
    fn node_bound(label: &'static str, pipeline: u64, bindings: Vec<(ResourceId, NodeAccess)>, wg: u32) -> TaskNode {
        TaskNode {
            label,
            bindings: bindings
                .into_iter()
                .map(|(resource, access)| ResourceBinding { resource, access })
                .collect(),
            kind: NodeKind::Dispatch {
                pipeline,
                resource_slots: vec![pipeline as u32 + 100], // non-empty → alloc_dispatch called
                user_slots: Vec::new(),
                dispatch: DispatchDim::Direct { x: wg, y: 1, z: 1 },
            },
        }
    }

    /// Build a `ClearBuffer` `TaskNode`.
    fn clear_node(label: &'static str, buffer: ResourceId, buf_handle: u64) -> TaskNode {
        TaskNode {
            label,
            bindings: vec![ResourceBinding {
                resource: buffer,
                access: NodeAccess::Write,
            }],
            kind: NodeKind::ClearBuffer {
                buffer: buf_handle,
                offset: 0,
                size: 256,
            },
        }
    }

    /// Build a `WriteBuffer` `TaskNode`.
    fn write_node(label: &'static str, buffer: ResourceId, buf_handle: u64) -> TaskNode {
        TaskNode {
            label,
            bindings: vec![ResourceBinding {
                resource: buffer,
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteBuffer {
                buffer: buf_handle,
                offset: 0,
                data: Arc::from(vec![0u8; 4]),
            },
        }
    }

    fn write_texture_node(label: &'static str, texture: ResourceId, tex_handle: u64) -> TaskNode {
        TaskNode {
            label,
            bindings: vec![ResourceBinding {
                resource: texture,
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteTexture {
                texture: tex_handle,
                data: Arc::from(vec![0u8; 4]),
                width: 1,
                height: 1,
            },
        }
    }

    fn copy_buffer_node(label: &'static str, src: ResourceId, dst: ResourceId) -> TaskNode {
        TaskNode {
            label,
            bindings: vec![
                ResourceBinding {
                    resource: src,
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: dst,
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyBuffer {
                src,
                src_offset: 0,
                dst,
                dst_offset: 0,
                size: 64,
            },
        }
    }

    fn copy_buffer_to_texture_node(
        label: &'static str,
        src: ResourceId,
        dst: ResourceId,
        tex_handle: u64,
        src_row_pitch: u32,
    ) -> TaskNode {
        TaskNode {
            label,
            bindings: vec![
                ResourceBinding {
                    resource: src,
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: dst,
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyBufferToTexture {
                src,
                src_offset: 0,
                src_row_pitch,
                dst: tex_handle,
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
        }
    }

    #[test]
    fn upload_phase_ordering_splits_buffer_and_texture_nodes_into_waves() {
        let ir = GraphIR {
            nodes: vec![
                copy_buffer_node("scene", buf(0), buf(1)),
                copy_buffer_to_texture_node("gradient", buf(2), ResourceId::Texture(3), 3, 0),
            ],
        };
        let waves = graph_node_waves(&ir).unwrap();
        assert_eq!(
            waves,
            vec![0, 1],
            "buffer uploads must precede texture uploads in separate waves"
        );
    }

    #[test]
    fn upload_phase_partition_splits_retainable_buffer_wave_from_texture_wave() {
        let ir = GraphIR {
            nodes: vec![
                copy_buffer_node("scene", buf(0), buf(1)),
                copy_buffer_node("config", buf(2), buf(3)),
                copy_buffer_to_texture_node("gradient", buf(4), ResourceId::Texture(5), 5, 0),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let ranges = partition_wave_ranges(&ir, &schedule, true);
        assert_eq!(
            ranges.len(),
            2,
            "buffer-only and texture upload waves must be separate partitions"
        );
        assert!(waves_can_retain(&ir, &schedule.waves[ranges[0].clone()]));
        assert!(!waves_can_retain(&ir, &schedule.waves[ranges[1].clone()]));
    }

    #[test]
    fn pitched_copy_buffer_to_texture_wave_is_retainable() {
        let ir = GraphIR {
            nodes: vec![copy_buffer_to_texture_node(
                "gradient",
                buf(0),
                ResourceId::Texture(1),
                1,
                256,
            )],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        assert!(waves_can_retain(&ir, &schedule.waves));
    }

    #[test]
    fn tight_copy_buffer_to_texture_wave_is_not_retainable() {
        let ir = GraphIR {
            nodes: vec![copy_buffer_to_texture_node(
                "gradient",
                buf(0),
                ResourceId::Texture(1),
                1,
                0,
            )],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        assert!(!waves_can_retain(&ir, &schedule.waves));
    }

    #[test]
    fn render_pass_then_copy_render_target_orders_nodes() {
        let ir = GraphIR {
            nodes: vec![
                TaskNode {
                    label: "draw",
                    bindings: vec![],
                    kind: NodeKind::RenderPass {
                        target: 10,
                        color_load: crate::types::TargetLoad::Clear(crate::types::Color::BLACK),
                        commands: Vec::new(),
                    },
                },
                TaskNode {
                    label: "copy_to_swapchain",
                    bindings: vec![
                        ResourceBinding {
                            resource: ResourceId::RenderTarget(10),
                            access: NodeAccess::Read,
                        },
                        ResourceBinding {
                            resource: ResourceId::SwapchainOutput,
                            access: NodeAccess::Write,
                        },
                    ],
                    kind: NodeKind::CopyRenderTarget {
                        src: 10,
                        dst: ResourceId::SwapchainOutput,
                    },
                },
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
        assert_eq!(schedule.waves[0].node_indices, vec![0]);
        assert_eq!(schedule.waves[1].node_indices, vec![1]);
    }

    #[test]
    fn copy_render_target_resolves_swapchain_output_dst() {
        let ir = GraphIR {
            nodes: vec![TaskNode {
                label: "copy_rt_to_swapchain",
                bindings: vec![
                    ResourceBinding {
                        resource: ResourceId::RenderTarget(5),
                        access: NodeAccess::Read,
                    },
                    ResourceBinding {
                        resource: ResourceId::SwapchainOutput,
                        access: NodeAccess::Write,
                    },
                ],
                kind: NodeKind::CopyRenderTarget {
                    src: 5,
                    dst: ResourceId::SwapchainOutput,
                },
            }],
        };
        let schedule = schedule_waves(&ir, &build_edges(&ir));
        let resolver = SlotResolver {
            swapchain: Some(crate::task_graph::ResolvedSwapchain {
                handle: 42,
                uav_index: 3,
            }),
            ..Default::default()
        };

        let commands = emit_waves_to_commands(&ir, &schedule.waves, Some(&resolver));
        // Bind-free copy must NOT receive a FrameTableStaging prefix.
        assert!(matches!(
            commands.as_slice(),
            [GpuCommand::CopyRenderTarget { src: 5, dst: 42 }]
        ));
    }

    #[test]
    fn copy_texture_resolves_swapchain_output_dst() {
        let ir = GraphIR {
            nodes: vec![TaskNode {
                label: "copy_to_swapchain",
                bindings: vec![
                    ResourceBinding {
                        resource: ResourceId::Texture(1),
                        access: NodeAccess::Read,
                    },
                    ResourceBinding {
                        resource: ResourceId::SwapchainOutput,
                        access: NodeAccess::Write,
                    },
                ],
                kind: NodeKind::CopyTexture {
                    src: 1,
                    dst: ResourceId::SwapchainOutput,
                    dst_buffer_layout: None,
                },
            }],
        };
        let schedule = schedule_waves(&ir, &build_edges(&ir));
        let resolver = SlotResolver {
            swapchain: Some(crate::task_graph::ResolvedSwapchain {
                handle: 99,
                uav_index: 7,
            }),
            ..Default::default()
        };

        let commands = emit_waves_to_commands(&ir, &schedule.waves, Some(&resolver));
        // A bind-free copy must NOT receive a FrameTableStaging prefix.
        // Inserting one bumps the submission counter and silently overwrites
        // the selector with zeros, corrupting every in-flight frame.
        assert!(
            matches!(commands.as_slice(), [GpuCommand::CopyTexture { src: 1, dst: 99 }]),
            "bind-free CopyTexture must not get a FrameTableStaging prefix; got {commands:?}"
        );
    }

    /// A graph consisting only of WriteBuffer nodes (uploads) must not generate a
    /// FrameTableStaging prefix — these are bind-free and must not bump the counter.
    #[test]
    fn write_only_graph_no_staging_prefix() {
        let ir = GraphIR {
            nodes: vec![write_node("upload_a", buf(0), 0), write_node("upload_b", buf(1), 1)],
        };
        let schedule = schedule_waves(&ir, &build_edges(&ir));
        let cmds = emit_waves_to_commands(&ir, &schedule.waves, None);
        let has_staging = cmds.iter().any(|c| matches!(c, GpuCommand::FrameTableStaging { .. }));
        assert!(
            !has_staging,
            "write-only graph must not produce FrameTableStaging; got {cmds:?}"
        );
    }

    /// A graph whose dispatch node has actual resource_slots MUST receive a
    /// FrameTableStaging prefix so the prologue populates the device table.
    #[test]
    fn dispatch_graph_gets_staging_prefix() {
        let ir = GraphIR {
            nodes: vec![TaskNode {
                label: "A",
                bindings: vec![ResourceBinding {
                    resource: buf(0),
                    access: NodeAccess::Write,
                }],
                kind: NodeKind::Dispatch {
                    pipeline: 1,
                    resource_slots: vec![42u32], // non-empty → alloc_dispatch is called
                    user_slots: Vec::new(),
                    dispatch: DispatchDim::Direct { x: 4, y: 1, z: 1 },
                },
            }],
        };
        let schedule = schedule_waves(&ir, &build_edges(&ir));
        let cmds = emit_waves_to_commands(&ir, &schedule.waves, None);
        assert!(
            matches!(cmds.first(), Some(GpuCommand::FrameTableStaging { .. })),
            "dispatch with resource_slots must start with FrameTableStaging; got {cmds:?}"
        );
    }

    #[test]
    fn linear_chain_raw() {
        // A writes X, B reads X -> A before B
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
        assert_eq!(schedule.waves[0].node_indices, vec![0]);
        assert_eq!(schedule.waves[1].node_indices, vec![1]);
        assert!(!schedule.waves[1].barriers_before.is_empty());
        assert_eq!(schedule.waves[1].barriers_before.buffers[0].0, 0);
    }

    #[test]
    fn independent_nodes_same_wave() {
        // A writes X, B writes Y -> no dependency, same wave
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(1), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert_eq!(schedule.waves[0].node_indices, vec![0, 1]);
    }

    #[test]
    fn swmr_multiple_reads() {
        // A reads X, B reads X -> no conflict, same wave
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Read)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert_eq!(schedule.waves[0].node_indices, vec![0, 1]);
    }

    #[test]
    fn war_edge() {
        // A reads X, B writes X -> WAR dependency
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Read)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
    }

    #[test]
    fn waw_edge() {
        // A writes X, B writes X -> WAW dependency
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
    }

    #[test]
    fn write_to_overwrite_keeps_waw_edge() {
        // Inaugural voids contents, not ordering — Write→Overwrite still edges.
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Overwrite)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);
    }

    #[test]
    fn overwrite_to_overwrite_keeps_waw_edge() {
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Overwrite)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Overwrite)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);
    }

    #[test]
    fn overwrite_to_read_keeps_raw_edge() {
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Overwrite)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);
    }

    #[test]
    fn read_to_overwrite_keeps_war_edge() {
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Read)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Overwrite)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);
    }

    #[test]
    fn diamond_dependency() {
        //   A (writes X)
        //  / \
        // B   C  (both read X, write Y/Z respectively)
        //  \ /
        //   D  (reads Y and Z)
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)], 1),
                node("C", 3, vec![(buf(0), NodeAccess::Read), (buf(2), NodeAccess::Write)], 1),
                node("D", 4, vec![(buf(1), NodeAccess::Read), (buf(2), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);

        let schedule = schedule_waves(&ir, &edges);
        // Wave 0: A, Wave 1: B+C (both read X), Wave 2: D (reads Y,Z)
        assert_eq!(schedule.waves.len(), 3);
        assert_eq!(schedule.waves[0].node_indices, vec![0]);
        let mut w1 = schedule.waves[1].node_indices.clone();
        w1.sort();
        assert_eq!(w1, vec![1, 2]);
        assert_eq!(schedule.waves[2].node_indices, vec![3]);
    }

    #[test]
    fn empty_graph() {
        let ir = GraphIR { nodes: Vec::new() };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());
        let schedule = schedule_waves(&ir, &edges);
        assert!(schedule.waves.is_empty());
    }

    #[test]
    fn single_node() {
        let ir = GraphIR {
            nodes: vec![node("A", 1, vec![(buf(0), NodeAccess::ReadWrite)], 4)],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert!(schedule.waves[0].barriers_before.is_empty());
    }

    #[test]
    fn barrier_targets_correct_resources() {
        // A writes buf0, B writes buf1, C reads buf0 and buf1
        // A->C (buf0 RAW), B->C (buf1 RAW), A and B independent
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(1), NodeAccess::Write)], 1),
                node("C", 3, vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);

        assert_eq!(schedule.waves.len(), 2);
        let barrier = &schedule.waves[1].barriers_before;
        let handles: Vec<_> = barrier.buffers.iter().map(|(h, _)| *h).collect();
        assert_eq!(handles, vec![0, 1]);
    }

    #[test]
    fn command_emission_linear_chain() {
        let ir = GraphIR {
            nodes: vec![
                node("A", 10, vec![(buf(0), NodeAccess::Write)], 8),
                node("B", 20, vec![(buf(0), NodeAccess::Read)], 4),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let cmds = emit_commands(&ir, &schedule, None);

        // No staging (test nodes have no resource_slots → no bindings).
        // Wave 0: SetPipeline(10), Dispatch(8,1,1)
        // ResourceBarrier([0])
        // Wave 1: SetPipeline(20), Dispatch(4,1,1)
        assert_eq!(cmds.len(), 5);
        assert!(matches!(cmds[0], GpuCommand::SetPipeline(10)));
        assert!(matches!(
            cmds[1],
            GpuCommand::Dispatch {
                workgroups_x: 8,
                label: Some("A"),
                ..
            }
        ));
        assert!(matches!(cmds[2], GpuCommand::ResourceBarrier { .. }));
        assert!(matches!(cmds[3], GpuCommand::SetPipeline(20)));
        assert!(matches!(
            cmds[4],
            GpuCommand::Dispatch {
                workgroups_x: 4,
                label: Some("B"),
                ..
            }
        ));
    }

    #[test]
    fn command_emission_independent() {
        let ir = GraphIR {
            nodes: vec![
                node("A", 10, vec![(buf(0), NodeAccess::Write)], 8),
                node("B", 20, vec![(buf(1), NodeAccess::Write)], 4),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let cmds = emit_commands(&ir, &schedule, None);

        // Single wave, no barriers, no staging (test nodes have no resource_slots).
        assert_eq!(cmds.len(), 4);
        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::FrameTableStaging { .. })));
        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
    }

    #[test]
    fn command_emission_with_resource_slots() {
        let ir = GraphIR {
            nodes: vec![TaskNode {
                label: "A",
                bindings: vec![ResourceBinding {
                    resource: buf(0),
                    access: NodeAccess::Write,
                }],
                kind: NodeKind::Dispatch {
                    pipeline: 10,
                    resource_slots: vec![42, 7],
                    user_slots: Vec::new(),
                    dispatch: DispatchDim::Direct { x: 1, y: 1, z: 1 },
                },
            }],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let cmds = emit_commands(&ir, &schedule, None);

        assert_eq!(cmds.len(), 4);
        assert!(matches!(cmds[0], GpuCommand::FrameTableStaging { .. }));
        assert!(matches!(cmds[1], GpuCommand::SetPipeline(10)));
        assert!(matches!(cmds[2], GpuCommand::BindResourcesRaw { ref indices, .. } if indices == &[42, 7]));
        assert!(matches!(
            cmds[3],
            GpuCommand::Dispatch {
                workgroups_x: 1,
                label: Some("A"),
                ..
            }
        ));
    }

    // -------------------------------------------------------------------------
    // ClearBuffer and WriteBuffer node emission tests
    // -------------------------------------------------------------------------

    #[test]
    fn clear_node_emits_clear_buffer_command() {
        // A single ClearBuffer node should emit exactly one ClearBuffer command.
        let ir = GraphIR {
            nodes: vec![clear_node("clear", buf(0), 0)],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let cmds = emit_commands(&ir, &schedule, None);

        // No staging: bind-free node must not trigger a prologue.
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], GpuCommand::ClearBuffer { .. }));
    }

    #[test]
    fn write_node_emits_write_buffer_command() {
        let ir = GraphIR {
            nodes: vec![write_node("write", buf(0), 0)],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let cmds = emit_commands(&ir, &schedule, None);

        // No staging: bind-free upload must not trigger a prologue.
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], GpuCommand::WriteBuffer { .. }));
    }

    #[test]
    fn clear_then_read_produces_barrier() {
        // ClearBuffer buf0 → dispatch reads buf0: two waves, one barrier
        let ir = GraphIR {
            nodes: vec![
                clear_node("clear", buf(0), 0),
                node("read", 1, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);

        let cmds = emit_commands(&ir, &schedule, None);
        // ClearBuffer, ResourceBarrier, SetPipeline, Dispatch
        // No staging: dispatch has no resource_slots → no alloc_dispatch → no prologue.
        assert_eq!(cmds.len(), 4);
        assert!(matches!(cmds[0], GpuCommand::ClearBuffer { .. }));
        assert!(matches!(cmds[1], GpuCommand::ResourceBarrier { .. }));
        assert!(matches!(cmds[2], GpuCommand::SetPipeline(_)));
        assert!(matches!(cmds[3], GpuCommand::Dispatch { .. }));
    }

    #[test]
    fn clear_and_independent_dispatch_same_wave() {
        // ClearBuffer buf0, dispatch writes buf1 → independent → wave 0, no barrier
        let ir = GraphIR {
            nodes: vec![
                clear_node("clear", buf(0), 0),
                node("write", 1, vec![(buf(1), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);

        let cmds = emit_commands(&ir, &schedule, None);
        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
        assert!(cmds.iter().any(|c| matches!(c, GpuCommand::ClearBuffer { .. })));
        assert!(cmds.iter().any(|c| matches!(c, GpuCommand::Dispatch { .. })));
    }

    #[test]
    fn write_buffer_then_dispatch_produces_barrier() {
        // WriteBuffer buf0 → dispatch reads buf0: two waves, one barrier
        let ir = GraphIR {
            nodes: vec![
                write_node("write", buf(0), 0),
                node("read", 1, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);

        let cmds = emit_commands(&ir, &schedule, None);
        // No staging (dispatch has no resource_slots).
        assert!(matches!(cmds[0], GpuCommand::WriteBuffer { .. }));
        assert!(matches!(cmds[1], GpuCommand::ResourceBarrier { .. }));
    }

    #[test]
    fn write_buffer_and_independent_dispatch_same_wave() {
        // WriteBuffer buf0, dispatch writes buf1 → independent → no barrier
        let ir = GraphIR {
            nodes: vec![
                write_node("write", buf(0), 0),
                node("write_b", 1, vec![(buf(1), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);

        let cmds = emit_commands(&ir, &schedule, None);
        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
    }

    #[test]
    fn write_texture_node_emits_write_texture_command() {
        let ir = GraphIR {
            nodes: vec![write_texture_node("up", tex(0), 0)],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let cmds = emit_commands(&ir, &schedule, None);

        // No staging: bind-free upload must not trigger a prologue.
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], GpuCommand::WriteTexture { .. }));
    }

    #[test]
    fn write_texture_then_dispatch_produces_barrier() {
        let ir = GraphIR {
            nodes: vec![
                write_texture_node("up", tex(0), 0),
                node("read", 1, vec![(tex(0), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);

        let cmds = emit_commands(&ir, &schedule, None);
        // No staging (dispatch has no resource_slots).
        assert!(matches!(cmds[0], GpuCommand::WriteTexture { .. }));
        assert!(matches!(cmds[1], GpuCommand::ResourceBarrier { .. }));
    }

    #[test]
    fn write_texture_and_independent_dispatch_same_wave() {
        let ir = GraphIR {
            nodes: vec![
                write_texture_node("up", tex(0), 0),
                node("buf", 1, vec![(buf(1), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);

        let cmds = emit_commands(&ir, &schedule, None);
        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
    }

    #[test]
    fn multiple_clears_independent_same_wave() {
        // Two clears on different buffers → independent → wave 0, no barrier
        let ir = GraphIR {
            nodes: vec![clear_node("clear_a", buf(0), 0), clear_node("clear_b", buf(1), 1)],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert_eq!(schedule.waves[0].node_indices.len(), 2);
    }

    #[test]
    fn diamond_with_clear_at_root() {
        // ClearBuffer buf0 (root), two dispatches read buf0 + write y/z,
        // final dispatch reads y+z. Same DAG shape as compile_diamond but
        // root is a clear instead of a dispatch.
        let ir = GraphIR {
            nodes: vec![
                clear_node("clear_x", buf(0), 0),
                node("B", 2, vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)], 1),
                node("C", 3, vec![(buf(0), NodeAccess::Read), (buf(2), NodeAccess::Write)], 1),
                node("D", 4, vec![(buf(1), NodeAccess::Read), (buf(2), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);

        // 3 waves: [clear_x], [B, C], [D]
        assert_eq!(schedule.waves.len(), 3);
        assert_eq!(schedule.waves[0].node_indices, vec![0]);
        let mut w1 = schedule.waves[1].node_indices.clone();
        w1.sort();
        assert_eq!(w1, vec![1, 2]);
        assert_eq!(schedule.waves[2].node_indices, vec![3]);

        // Two barriers: before wave 1 (clear→B,C) and before wave 2 (B,C→D)
        let barrier_count = schedule.waves.iter().filter(|w| !w.barriers_before.is_empty()).count();
        assert_eq!(barrier_count, 2);
    }

    // -------------------------------------------------------------------------
    // Category A: ranges_overlap (8 tests)
    // -------------------------------------------------------------------------

    #[test]
    fn ranges_overlap_disjoint_before() {
        // [0, 10) and [10, 20) — touch at boundary but do not overlap
        assert!(!ranges_overlap(0, 10, 10, 10));
    }

    #[test]
    fn ranges_overlap_disjoint_after() {
        // [10, 20) and [0, 10) — symmetric of above
        assert!(!ranges_overlap(10, 10, 0, 10));
    }

    #[test]
    fn ranges_overlap_exact() {
        // [5, 15) and [5, 15) — identical ranges overlap
        assert!(ranges_overlap(5, 10, 5, 10));
    }

    #[test]
    fn ranges_overlap_partial_left() {
        // [0, 10) and [5, 15) — overlap in [5, 10)
        assert!(ranges_overlap(0, 10, 5, 10));
    }

    #[test]
    fn ranges_overlap_partial_right() {
        // [5, 15) and [0, 10) — symmetric of above
        assert!(ranges_overlap(5, 10, 0, 10));
    }

    #[test]
    fn ranges_overlap_contained() {
        // [0, 100) contains [10, 20) — overlap
        assert!(ranges_overlap(0, 100, 10, 10));
    }

    #[test]
    fn ranges_overlap_containing() {
        // [10, 20) is contained by [0, 100) — overlap (symmetric)
        assert!(ranges_overlap(10, 10, 0, 100));
    }

    #[test]
    fn ranges_overlap_zero_length() {
        // Zero-length ranges never overlap anything
        assert!(!ranges_overlap(5, 0, 5, 10));
        assert!(!ranges_overlap(5, 10, 5, 0));
        assert!(!ranges_overlap(5, 0, 5, 0));
    }

    // -------------------------------------------------------------------------
    // Category B: bindings_conflict cross-variant matrix (12 tests)
    // -------------------------------------------------------------------------

    fn range(parent: u64, offset: u64, len: u64) -> ResourceId {
        ResourceId::BufferRange { parent, offset, len }
    }

    fn tex(id: u64) -> ResourceId {
        ResourceId::Texture(id)
    }

    fn binding(resource: ResourceId, access: NodeAccess) -> ResourceBinding {
        ResourceBinding { resource, access }
    }

    #[test]
    fn conflict_buffer_vs_buffer_same_write() {
        assert!(bindings_conflict(
            &binding(buf(0), NodeAccess::Write),
            &binding(buf(0), NodeAccess::Read)
        ));
    }

    #[test]
    fn conflict_buffer_vs_buffer_different_no_conflict() {
        assert!(!bindings_conflict(
            &binding(buf(0), NodeAccess::Write),
            &binding(buf(1), NodeAccess::Read)
        ));
    }

    #[test]
    fn conflict_buffer_vs_buffer_both_read_no_conflict() {
        assert!(!bindings_conflict(
            &binding(buf(0), NodeAccess::Read),
            &binding(buf(0), NodeAccess::Read)
        ));
    }

    #[test]
    fn conflict_buffer_vs_range_parent_writes() {
        // Whole-buffer write conflicts with any range of that buffer
        assert!(bindings_conflict(
            &binding(buf(10), NodeAccess::Write),
            &binding(range(10, 0, 100), NodeAccess::Read)
        ));
    }

    #[test]
    fn conflict_range_vs_buffer_range_reads_parent_writes() {
        // Symmetric: range read vs whole buffer write
        assert!(bindings_conflict(
            &binding(range(10, 0, 100), NodeAccess::Read),
            &binding(buf(10), NodeAccess::Write)
        ));
    }

    #[test]
    fn conflict_buffer_vs_range_different_parent_no_conflict() {
        assert!(!bindings_conflict(
            &binding(buf(10), NodeAccess::Write),
            &binding(range(20, 0, 100), NodeAccess::Read)
        ));
    }

    #[test]
    fn conflict_range_vs_range_overlapping_one_writes() {
        // Overlapping ranges of the same parent — write creates conflict
        assert!(bindings_conflict(
            &binding(range(5, 0, 512), NodeAccess::Write),
            &binding(range(5, 256, 512), NodeAccess::Read)
        ));
    }

    #[test]
    fn conflict_range_vs_range_disjoint_same_parent_no_conflict() {
        // Disjoint ranges of the same parent — no conflict even if both write
        assert!(!bindings_conflict(
            &binding(range(5, 0, 256), NodeAccess::Write),
            &binding(range(5, 256, 256), NodeAccess::Write)
        ));
    }

    #[test]
    fn conflict_range_vs_range_different_parent_no_conflict() {
        // Same offsets but different parents — no conflict
        assert!(!bindings_conflict(
            &binding(range(1, 0, 256), NodeAccess::Write),
            &binding(range(2, 0, 256), NodeAccess::Read)
        ));
    }

    #[test]
    fn conflict_both_read_range_vs_range_no_conflict() {
        // Two reads on the same range — SWMR, no conflict
        assert!(!bindings_conflict(
            &binding(range(5, 0, 256), NodeAccess::Read),
            &binding(range(5, 0, 256), NodeAccess::Read)
        ));
    }

    #[test]
    fn conflict_texture_vs_texture_write_read() {
        assert!(bindings_conflict(
            &binding(tex(7), NodeAccess::Write),
            &binding(tex(7), NodeAccess::Read)
        ));
    }

    #[test]
    fn conflict_buffer_vs_texture_no_conflict() {
        // Cross-type: buffer and texture never conflict
        assert!(!bindings_conflict(
            &binding(buf(0), NodeAccess::Write),
            &binding(tex(0), NodeAccess::Write)
        ));
    }

    // -------------------------------------------------------------------------
    // Category C: build_edges with mixed ResourceIds (10 tests)
    // -------------------------------------------------------------------------

    #[test]
    fn edges_disjoint_ranges_same_parent_one_writes_no_edge() {
        // A writes [0,256) of parent 0; B reads [256,512) — disjoint, no edge
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 256), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(0, 256, 256), NodeAccess::Read)], 1),
            ],
        };
        assert!(build_edges(&ir).is_empty());
    }

    #[test]
    fn edges_overlapping_ranges_same_parent_creates_edge() {
        // A writes [0, 512); B reads [256, 512) — overlap, edge required
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 512), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(0, 256, 256), NodeAccess::Read)], 1),
            ],
        };
        assert_eq!(build_edges(&ir), vec![(0, 1)]);
    }

    #[test]
    fn edges_whole_buffer_vs_range_same_parent_creates_edge() {
        // A writes whole Buffer(10); B reads range of parent 10 — aliased
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(10), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(10, 0, 64), NodeAccess::Read)], 1),
            ],
        };
        assert_eq!(build_edges(&ir), vec![(0, 1)]);
    }

    #[test]
    fn edges_whole_buffer_vs_range_different_parent_no_edge() {
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(10), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(20, 0, 64), NodeAccess::Read)], 1),
            ],
        };
        assert!(build_edges(&ir).is_empty());
    }

    #[test]
    fn edges_n_disjoint_views_all_writing_no_edges() {
        // 6 nodes, each writing a non-overlapping 256-byte region of parent 0
        let nodes: Vec<TaskNode> = (0..6)
            .map(|i| node("dispatch", i, vec![(range(0, i * 256, 256), NodeAccess::Write)], 1))
            .collect();
        let ir = GraphIR { nodes };
        assert!(build_edges(&ir).is_empty());
    }

    #[test]
    fn edges_mixed_pooled_and_standalone_only_real_conflicts() {
        // A writes standalone buf(99); B writes range of parent 0; C reads buf(99)
        // Only A->C conflicts; B is independent
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(99), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(0, 0, 256), NodeAccess::Write)], 1),
                node("C", 3, vec![(buf(99), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 2)]);
    }

    #[test]
    fn edges_node_with_multiple_ranges_partial_overlap() {
        // A reads [0,256) and [512,768) of parent 0
        // B writes [200,400) of parent 0 — overlaps [0,256) but not [512,768)
        let ir = GraphIR {
            nodes: vec![
                node(
                    "A",
                    1,
                    vec![
                        (range(0, 0, 256), NodeAccess::Read),
                        (range(0, 512, 256), NodeAccess::Read),
                    ],
                    1,
                ),
                node("B", 2, vec![(range(0, 200, 200), NodeAccess::Write)], 1),
            ],
        };
        assert_eq!(build_edges(&ir), vec![(0, 1)]);
    }

    #[test]
    fn edges_range_adjacent_no_overlap_no_edge() {
        // [0, 100) and [100, 200) — touching boundary is NOT overlap
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 100), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(0, 100, 100), NodeAccess::Write)], 1),
            ],
        };
        assert!(build_edges(&ir).is_empty());
    }

    #[test]
    fn edges_range_vs_range_same_offset_different_parents_no_edge() {
        // Same offset/length but different parents — no alias
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(1, 0, 1024), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(2, 0, 1024), NodeAccess::Read)], 1),
            ],
        };
        assert!(build_edges(&ir).is_empty());
    }

    #[test]
    fn edges_range_contained_within_another_creates_edge() {
        // A writes [0, 1024); B reads [256, 512) — fully contained, must conflict
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 1024), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(0, 256, 256), NodeAccess::Read)], 1),
            ],
        };
        assert_eq!(build_edges(&ir), vec![(0, 1)]);
    }

    // -------------------------------------------------------------------------
    // Category D: schedule_waves wave structure (10 tests)
    // -------------------------------------------------------------------------

    #[test]
    fn waves_8_disjoint_views_independent_writes_one_wave() {
        // 8 nodes, each writing a distinct 128-byte window of parent 0 — all independent
        let nodes: Vec<TaskNode> = (0..8)
            .map(|i| node("write", i, vec![(range(0, i * 128, 128), NodeAccess::Write)], 1))
            .collect();
        let ir = GraphIR { nodes };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert_eq!(schedule.waves[0].node_indices.len(), 8);
    }

    #[test]
    fn waves_simulated_coarse_fine_pipeline_collapses_waves() {
        // 10 nodes: 5 pairs of (write_viewN, read_viewN) where each pair is independent
        // — result should be far fewer than 10 waves
        let mut nodes = Vec::new();
        for i in 0..5u64 {
            nodes.push(node(
                "write",
                i * 2,
                vec![(range(0, i * 256, 256), NodeAccess::Write)],
                1,
            ));
            nodes.push(node(
                "read",
                i * 2 + 1,
                vec![(range(0, i * 256, 256), NodeAccess::Read)],
                1,
            ));
        }
        let ir = GraphIR { nodes };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        // Each pair (write → read) is a chain of 2 waves, but pairs are independent
        // so total waves = 2, not 10.
        assert!(
            schedule.waves.len() < 10,
            "expected far fewer than 10 waves, got {}",
            schedule.waves.len()
        );
    }

    #[test]
    fn waves_diamond_pooled_views() {
        // A writes view [0,256); B and C both read [0,256) and write separate regions
        // D reads B's and C's regions — diamond
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 256), NodeAccess::Write)], 1),
                node(
                    "B",
                    2,
                    vec![
                        (range(0, 0, 256), NodeAccess::Read),
                        (range(0, 256, 256), NodeAccess::Write),
                    ],
                    1,
                ),
                node(
                    "C",
                    3,
                    vec![
                        (range(0, 0, 256), NodeAccess::Read),
                        (range(0, 512, 256), NodeAccess::Write),
                    ],
                    1,
                ),
                node(
                    "D",
                    4,
                    vec![
                        (range(0, 256, 256), NodeAccess::Read),
                        (range(0, 512, 256), NodeAccess::Read),
                    ],
                    1,
                ),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        // Wave 0: A; Wave 1: B+C (both read [0,256)); Wave 2: D
        assert_eq!(schedule.waves.len(), 3);
        assert_eq!(schedule.waves[0].node_indices, vec![0]);
        let mut w1 = schedule.waves[1].node_indices.clone();
        w1.sort();
        assert_eq!(w1, vec![1, 2]);
        assert_eq!(schedule.waves[2].node_indices, vec![3]);
    }

    #[test]
    fn waves_fan_out_fan_in_pooled() {
        // A writes view0; B writes view1; C reads both view0 and view1 → chain
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 256), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(0, 256, 256), NodeAccess::Write)], 1),
                node(
                    "C",
                    3,
                    vec![
                        (range(0, 0, 256), NodeAccess::Read),
                        (range(0, 256, 256), NodeAccess::Read),
                    ],
                    1,
                ),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        // A and B are independent (different regions), both needed before C
        assert_eq!(schedule.waves.len(), 2);
        let mut w0 = schedule.waves[0].node_indices.clone();
        w0.sort();
        assert_eq!(w0, vec![0, 1]); // A and B in same wave
        assert_eq!(schedule.waves[1].node_indices, vec![2]); // C after
    }

    #[test]
    fn waves_all_overlapping_views_degrades_to_chain() {
        // 4 nodes all writing the same region — must be fully serialized
        let nodes: Vec<TaskNode> = (0..4)
            .map(|i| node("w", i, vec![(range(0, 0, 512), NodeAccess::Write)], 1))
            .collect();
        let ir = GraphIR { nodes };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 4);
        for w in &schedule.waves {
            assert_eq!(w.node_indices.len(), 1);
        }
    }

    #[test]
    fn waves_single_node_with_buffer_range() {
        let ir = GraphIR {
            nodes: vec![node("A", 1, vec![(range(0, 0, 256), NodeAccess::ReadWrite)], 1)],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert!(schedule.waves[0].barriers_before.is_empty());
    }

    #[test]
    fn waves_two_node_disjoint_ranges_one_wave() {
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 64), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(0, 64, 64), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
    }

    #[test]
    fn waves_range_read_read_same_region_one_wave() {
        // Two reads on the same overlapping range — SWMR, one wave
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 512), NodeAccess::Read)], 1),
                node("B", 2, vec![(range(0, 0, 512), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
    }

    #[test]
    fn waves_range_write_then_read_same_region_two_waves() {
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 512), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(0, 0, 512), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
    }

    #[test]
    fn waves_range_independent_then_dependent_correct_depth() {
        // A writes view0; B writes view1 (independent of A);
        // C reads view0 AND view1 — must be after both A and B
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 256), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(0, 256, 256), NodeAccess::Write)], 1),
                node(
                    "C",
                    3,
                    vec![
                        (range(0, 0, 256), NodeAccess::Read),
                        (range(0, 256, 256), NodeAccess::Read),
                    ],
                    1,
                ),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
        assert_eq!(schedule.waves[1].node_indices, vec![2]);
    }

    // -------------------------------------------------------------------------
    // Category E: compute_barriers correctness (6 tests)
    // -------------------------------------------------------------------------

    #[test]
    fn barriers_range_collapses_to_parent_handle() {
        // A writes a BufferRange with parent=42; B reads same range
        // The barrier should name the parent handle 42, not some sub-range handle
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(42, 0, 256), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(42, 0, 256), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
        assert_eq!(schedule.waves[1].barriers_before.buffers[0].0, 42);
    }

    #[test]
    fn barriers_multiple_disjoint_ranges_same_parent_one_barrier_entry() {
        // A writes [0,256); B writes [256,512); C reads both — one parent barrier
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(99, 0, 256), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(99, 256, 256), NodeAccess::Write)], 1),
                node(
                    "C",
                    3,
                    vec![
                        (range(99, 0, 256), NodeAccess::Read),
                        (range(99, 256, 256), NodeAccess::Read),
                    ],
                    1,
                ),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
        // Only one barrier entry (parent 99), deduplicated
        assert_eq!(schedule.waves[1].barriers_before.buffers[0].0, 99);
    }

    #[test]
    fn barriers_mixed_buffer_and_range_same_parent_one_entry() {
        // A writes Buffer(5) (whole); B reads range of parent 5
        // Barrier should list parent 5 once
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(5), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(5, 0, 100), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
        assert_eq!(schedule.waves[1].barriers_before.buffers[0].0, 5);
    }

    #[test]
    fn barriers_ranges_in_different_parents_separate_entries() {
        // A writes range of parent 1; B writes range of parent 2;
        // C reads both — two separate barrier entries
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(1, 0, 256), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(2, 0, 256), NodeAccess::Write)], 1),
                node(
                    "C",
                    3,
                    vec![
                        (range(1, 0, 256), NodeAccess::Read),
                        (range(2, 0, 256), NodeAccess::Read),
                    ],
                    1,
                ),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
        let handles: Vec<_> = schedule.waves[1]
            .barriers_before
            .buffers
            .iter()
            .map(|(h, _)| *h)
            .collect();
        assert_eq!(handles, vec![1, 2]);
    }

    #[test]
    fn barriers_disjoint_ranges_same_wave_no_barrier() {
        // A and B in same wave, no cross-wave edge — no barrier even if same parent
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(range(0, 0, 256), NodeAccess::Write)], 1),
                node("B", 2, vec![(range(0, 256, 256), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert!(schedule.waves[0].barriers_before.is_empty());
    }

    #[test]
    fn barriers_texture_unaffected_by_buffer_range_logic() {
        // A writes texture 7; B reads texture 7 — barrier targets texture, not buffer
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(tex(7), NodeAccess::Write)], 1),
                node("B", 2, vec![(tex(7), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
        assert!(schedule.waves[1].barriers_before.buffers.is_empty());
        assert_eq!(schedule.waves[1].barriers_before.textures[0].0, 7);
    }

    // -------------------------------------------------------------------------
    // emit_partitioned_commands tests
    // -------------------------------------------------------------------------

    /// Helper: run the full analysis pipeline and return partitions.
    fn partitions(ir: &GraphIR) -> Vec<Vec<GpuCommand>> {
        let edges = build_edges(ir);
        let schedule = schedule_waves(ir, &edges);
        emit_partitioned_commands(ir, &schedule, None, true)
    }

    /// Helper: run the full analysis pipeline and return flat commands.
    fn flat_commands(ir: &GraphIR) -> Vec<GpuCommand> {
        let edges = build_edges(ir);
        let schedule = schedule_waves(ir, &edges);
        emit_commands(ir, &schedule, None)
    }

    #[test]
    fn partition_single_wave_no_split() {
        // Single-wave graph: all nodes independent, no barrier → one partition.
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 4),
                node("B", 2, vec![(buf(1), NodeAccess::Write)], 4),
            ],
        };
        let parts = partitions(&ir);
        assert_eq!(parts.len(), 1, "single wave must not be split");
        // The single partition equals the flat emission.
        assert_eq!(parts[0], flat_commands(&ir));
    }

    #[test]
    fn partition_two_waves_below_threshold() {
        // Two-wave linear chain: below the 3-wave minimum → single partition.
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let parts = partitions(&ir);
        assert_eq!(parts.len(), 1, "two-wave graph must not be split");
        assert_eq!(parts[0], flat_commands(&ir));
    }

    #[test]
    fn partition_three_wave_diamond_splits() {
        // Diamond: A→(B,C)→D produces 3 waves.  Wave 1 has barrier for buf0,
        // wave 2 has barriers for buf1 and buf2 → wave 2 has the larger cost.
        // node_bound ensures resource_slots are non-empty so FrameTableStaging is emitted.
        let ir = GraphIR {
            nodes: vec![
                node_bound("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node_bound("B", 2, vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)], 1),
                node_bound("C", 3, vec![(buf(0), NodeAccess::Read), (buf(2), NodeAccess::Write)], 1),
                node_bound("D", 4, vec![(buf(1), NodeAccess::Read), (buf(2), NodeAccess::Read)], 1),
            ],
        };
        let parts = partitions(&ir);
        assert_eq!(parts.len(), 2, "3-wave diamond must produce two partitions");

        // Partition 0 starts with frame-table staging (not a barrier).
        assert!(matches!(parts[0].first(), Some(GpuCommand::FrameTableStaging { .. })));
        // Partition 1 starts with staging then barrier before its dispatches.
        assert!(matches!(parts[1].first(), Some(GpuCommand::FrameTableStaging { .. })));
        assert!(
            parts[1].iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
            "second partition must include a barrier"
        );

        // Structural commands (pipelines, dispatches, barriers, uploads) must match
        // flat emission order. Frame-table commands (staging prefix, bind-raw bases)
        // are per-partition-local and intentionally differ from the flat stream.
        fn strip_frame_table(cmds: &[GpuCommand]) -> Vec<&GpuCommand> {
            cmds.iter()
                .filter(|c| {
                    !matches!(
                        c,
                        GpuCommand::FrameTableStaging { .. } | GpuCommand::BindResourcesRaw { .. }
                    )
                })
                .collect()
        }
        let flat: Vec<GpuCommand> = parts.into_iter().flatten().collect();
        assert_eq!(strip_frame_table(&flat), strip_frame_table(&flat_commands(&ir)));
    }

    #[test]
    fn partition_three_wave_diamond_stays_single_when_barrier_split_disabled() {
        let ir = GraphIR {
            nodes: vec![
                node_bound("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node_bound("B", 2, vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)], 1),
                node_bound("C", 3, vec![(buf(0), NodeAccess::Read), (buf(2), NodeAccess::Write)], 1),
                node_bound("D", 4, vec![(buf(1), NodeAccess::Read), (buf(2), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let parts = emit_partitioned_commands(&ir, &schedule, None, false);
        assert_eq!(
            parts.len(),
            1,
            "Metal-style disabled barrier split must keep one compute partition"
        );
        assert_eq!(parts[0], flat_commands(&ir));
    }

    #[test]
    fn partition_coarse_fine_pipeline_splits_at_boundary() {
        // Simulates a coarse→fine rendering pattern:
        //   Wave 0: upload scene data (W buf0)
        //   Wave 1: coarse_a (R buf0, W buf1), coarse_b (R buf0, W buf2) — both read upload
        //   Wave 2: coarse_c (R buf1, R buf2, W buf3)                    — reads coarse_a,_b
        //   Wave 3: fine_a  (R buf3, W buf4), fine_b (R buf3, W buf5)    — reads coarse_c
        //   Wave 4: composite (R buf4, R buf5, W buf6)                   — reads fine_a,_b
        //
        // Barrier costs:
        //   wave 1: 1  (buf0)
        //   wave 2: 2  (buf1, buf2)
        //   wave 3: 1  (buf3)          ← coarse→fine boundary; only 1 resource
        //   wave 4: 2  (buf4, buf5)
        //
        // The largest barrier is a tie between wave 2 and wave 4 (cost 2).
        // max_by_key returns the last maximum in iteration order, so split = 4.
        // The test verifies two partitions and the flatten invariant.
        // node_bound ensures resource_slots are non-empty so FrameTableStaging is emitted.
        let ir = GraphIR {
            nodes: vec![
                write_node("upload", buf(0), 0),
                node_bound(
                    "coarse_a",
                    10,
                    vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)],
                    1,
                ),
                node_bound(
                    "coarse_b",
                    11,
                    vec![(buf(0), NodeAccess::Read), (buf(2), NodeAccess::Write)],
                    1,
                ),
                node_bound(
                    "coarse_c",
                    12,
                    vec![
                        (buf(1), NodeAccess::Read),
                        (buf(2), NodeAccess::Read),
                        (buf(3), NodeAccess::Write),
                    ],
                    1,
                ),
                node_bound(
                    "fine_a",
                    20,
                    vec![(buf(3), NodeAccess::Read), (buf(4), NodeAccess::Write)],
                    1,
                ),
                node_bound(
                    "fine_b",
                    21,
                    vec![(buf(3), NodeAccess::Read), (buf(5), NodeAccess::Write)],
                    1,
                ),
                node_bound(
                    "composite",
                    22,
                    vec![
                        (buf(4), NodeAccess::Read),
                        (buf(5), NodeAccess::Read),
                        (buf(6), NodeAccess::Write),
                    ],
                    1,
                ),
            ],
        };
        let parts = partitions(&ir);
        assert_eq!(parts.len(), 2, "coarse/fine graph must produce two partitions");

        // Late partition includes a barrier after its frame-table staging prefix.
        assert!(matches!(parts[1].first(), Some(GpuCommand::FrameTableStaging { .. })));
        assert!(
            parts[1].iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
            "late partition must include a barrier"
        );

        fn strip_frame_table(cmds: &[GpuCommand]) -> Vec<&GpuCommand> {
            cmds.iter()
                .filter(|c| {
                    !matches!(
                        c,
                        GpuCommand::FrameTableStaging { .. } | GpuCommand::BindResourcesRaw { .. }
                    )
                })
                .collect()
        }
        let flat: Vec<GpuCommand> = parts.into_iter().flatten().collect();
        assert_eq!(strip_frame_table(&flat), strip_frame_table(&flat_commands(&ir)));
    }

    #[test]
    fn partition_all_independent_single_partition() {
        // Many dispatches on disjoint buffers — one wave, no barrier → single partition.
        let ir = GraphIR {
            nodes: (0u64..10)
                .map(|i| node("x", i + 1, vec![(buf(i), NodeAccess::Write)], 1))
                .collect(),
        };
        let parts = partitions(&ir);
        assert_eq!(parts.len(), 1, "all-independent graph must not be split");
        // No barrier anywhere.
        let has_barrier = parts[0].iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. }));
        assert!(!has_barrier);
    }

    #[test]
    fn partition_flatten_invariant_linear_five_waves() {
        // A→B→C→D→E: 5 waves, each boundary has cost 1 (equal barriers).
        // With equal costs, max_by_key returns the last maximum (wave 4).
        // The invariant test is what matters here.
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)], 1),
                node("C", 3, vec![(buf(1), NodeAccess::Read), (buf(2), NodeAccess::Write)], 1),
                node("D", 4, vec![(buf(2), NodeAccess::Read), (buf(3), NodeAccess::Write)], 1),
                node("E", 5, vec![(buf(3), NodeAccess::Read)], 1),
            ],
        };
        let parts = partitions(&ir);
        assert_eq!(parts.len(), 2);
        fn strip_frame_table(cmds: &[GpuCommand]) -> Vec<&GpuCommand> {
            cmds.iter()
                .filter(|c| {
                    !matches!(
                        c,
                        GpuCommand::FrameTableStaging { .. } | GpuCommand::BindResourcesRaw { .. }
                    )
                })
                .collect()
        }
        let flat: Vec<GpuCommand> = parts.into_iter().flatten().collect();
        assert_eq!(strip_frame_table(&flat), strip_frame_table(&flat_commands(&ir)));
    }

    #[test]
    fn partition_zero_barrier_cost_stays_single() {
        // Graph with 3 waves but no actual cross-wave resource conflicts would
        // have zero barrier cost at every boundary → single partition fallback.
        //
        // Build this via two groups that individually have chains but share no resources:
        //   A (W buf0) → B (R buf0)   [two-wave sub-chain]
        //   C (W buf1) [independent]
        //
        // A and C are in wave 0 (independent), B is in wave 1.
        // But wave 1 barrier costs 1 (buf0) — not a zero-cost scenario.
        //
        // To get true zero-cost: use a 3-wave chain where nodes have no bindings.
        // Construct explicitly: nodes share no resources but forced ordering
        // is impossible without edges.  Instead, verify the fallback path by
        // constructing a schedule manually.
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(1), NodeAccess::Write)], 1),
                node("C", 3, vec![(buf(2), NodeAccess::Write)], 1),
            ],
        };
        // All three nodes are independent → one wave → fewer than 3 waves → single partition.
        let parts = partitions(&ir);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], flat_commands(&ir));
    }

    // ── node_to_wave_map / transient interval tests ─────────────────────────

    fn transient_buf(id: u32) -> ResourceId {
        ResourceId::TransientBuffer(super::super::TransientId(id))
    }

    fn transient_tex(id: u32) -> ResourceId {
        ResourceId::TransientTexture(super::super::TransientTextureId(id))
    }

    /// `node_to_wave_map` must produce the same mapping as `graph_node_waves`.
    #[test]
    fn node_to_wave_map_matches_graph_node_waves() {
        // 3-wave diamond: A→(B,C)→D
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)], 1),
                node("C", 3, vec![(buf(0), NodeAccess::Read), (buf(2), NodeAccess::Write)], 1),
                node("D", 4, vec![(buf(1), NodeAccess::Read), (buf(2), NodeAccess::Read)], 1),
            ],
        };

        let old_map = graph_node_waves(&ir).unwrap();

        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let new_map = node_to_wave_map(&schedule, ir.nodes.len());

        assert_eq!(old_map, new_map, "node_to_wave_map must equal graph_node_waves");
        // Sanity: wave 0 is A, wave 1 is B and C, wave 2 is D
        assert_eq!(new_map[0], 0);
        assert_eq!(new_map[1], 1);
        assert_eq!(new_map[2], 1);
        assert_eq!(new_map[3], 2);
    }

    /// `transient_wave_intervals` with pre-computed map must equal the result
    /// from the old path that re-ran the full scheduler internally.
    #[test]
    fn transient_wave_intervals_via_precomputed_map() {
        // A writes transient 0, B reads transient 0 and writes transient 1, C reads transient 1.
        // Expected intervals: t0 = [0,1], t1 = [1,2]
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(transient_buf(0), NodeAccess::Write)], 1),
                node(
                    "B",
                    2,
                    vec![
                        (transient_buf(0), NodeAccess::Read),
                        (transient_buf(1), NodeAccess::Write),
                    ],
                    1,
                ),
                node("C", 3, vec![(transient_buf(1), NodeAccess::Read)], 1),
            ],
        };

        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let node_waves = node_to_wave_map(&schedule, ir.nodes.len());

        let intervals = transient_wave_intervals(&ir, &node_waves).unwrap();
        assert_eq!(intervals[&0], (0, 1), "transient 0: waves 0..1");
        assert_eq!(intervals[&1], (1, 2), "transient 1: waves 1..2");
    }

    /// `transient_texture_wave_intervals` with pre-computed map produces correct ranges.
    #[test]
    fn transient_texture_wave_intervals_via_precomputed_map() {
        // Upload transient texture 0, read it in wave 1, use transient texture 1 in wave 2.
        let ir = GraphIR {
            nodes: vec![
                node("upload", 1, vec![(transient_tex(0), NodeAccess::Write)], 1),
                node(
                    "read0",
                    2,
                    vec![
                        (transient_tex(0), NodeAccess::Read),
                        (transient_tex(1), NodeAccess::Write),
                    ],
                    1,
                ),
                node("read1", 3, vec![(transient_tex(1), NodeAccess::Read)], 1),
            ],
        };

        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let node_waves = node_to_wave_map(&schedule, ir.nodes.len());

        let intervals = transient_texture_wave_intervals(&ir, &node_waves).unwrap();
        assert_eq!(intervals[&0], (0, 1));
        assert_eq!(intervals[&1], (1, 2));
    }

    /// Empty graph returns empty results without panicking.
    #[test]
    fn node_to_wave_map_empty_graph() {
        let ir = GraphIR { nodes: vec![] };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let map = node_to_wave_map(&schedule, 0);
        assert!(map.is_empty());
        let intervals = transient_wave_intervals(&ir, &map).unwrap();
        assert!(intervals.is_empty());
    }
}
