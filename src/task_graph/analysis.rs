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

use super::ir::{BarrierSet, CompiledSchedule, GraphIR, NodeKind, ResourceBinding, Wave};
// NodeAccess is used in the test module via `super::*`
#[cfg(test)]
use super::ir::NodeAccess;
use super::ResourceId;
use crate::backend::shared::DISPATCH_BATCH_STRIDE;
use crate::backend::{GpuCommand, GraphCommand};
use anyhow::Result;

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
        (ResourceId::TransientBuffer(x), ResourceId::TransientBuffer(y)) => x == y,
        (ResourceId::TransientTexture(x), ResourceId::TransientTexture(y)) => x == y,
        _ => false,
    }
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
        TransientBuffer(u32),
        TransientTexture(u32),
    }

    fn group_key(r: &ResourceId) -> GroupKey {
        match *r {
            ResourceId::Buffer(h) => GroupKey::Buffer(h),
            ResourceId::BufferRange { parent, .. } => GroupKey::Buffer(parent),
            ResourceId::Texture(h) => GroupKey::Texture(h),
            ResourceId::TransientBuffer(t) => GroupKey::TransientBuffer(t.0),
            ResourceId::TransientTexture(t) => GroupKey::TransientTexture(t.0),
        }
    }

    // Map each canonical key to the set of node indices that reference it.
    let mut resource_nodes: HashMap<GroupKey, Vec<usize>> = HashMap::new();
    for (idx, node) in ir.nodes.iter().enumerate() {
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
                let conflict = ir.nodes[i].bindings.iter().any(|bi| {
                    ir.nodes[j]
                        .bindings
                        .iter()
                        .any(|bj| bindings_conflict(bi, bj))
                });
                if conflict {
                    edge_set.insert((i, j));
                }
            }
        }
    }

    let mut edges: Vec<_> = edge_set.into_iter().collect();
    edges.sort_unstable();
    edges
}

/// Wave index for each task node (same order as [`GraphIR::nodes`]).
pub(crate) fn graph_node_waves(ir: &GraphIR) -> Result<Vec<u32>> {
    let n = ir.nodes.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let edges = build_edges(ir);
    let schedule = schedule_waves(ir, &edges);
    let mut node_to_wave: Vec<Option<u32>> = vec![None; n];
    for (w, wave) in schedule.waves.iter().enumerate() {
        let w = u32::try_from(w).map_err(|_| anyhow::anyhow!("wave index overflow"))?;
        for &ni in &wave.node_indices {
            node_to_wave[ni] = Some(w);
        }
    }
    for (i, slot) in node_to_wave.iter().enumerate() {
        if slot.is_none() {
            anyhow::bail!("internal: task node {} was not assigned a wave", i);
        }
    }
    Ok(node_to_wave.into_iter().map(|x| x.unwrap()).collect())
}

/// For each [`ResourceId::TransientBuffer`](super::ResourceId), the inclusive
/// range of wave indices where that transient appears in node bindings.
///
/// Used to pack transient heap allocations: non-overlapping wave intervals may
/// alias the same memory.
pub(crate) fn transient_wave_intervals(ir: &GraphIR) -> Result<HashMap<u32, (u32, u32)>> {
    let n = ir.nodes.len();
    if n == 0 {
        return Ok(HashMap::new());
    }
    let waves = graph_node_waves(ir)?;
    let mut first: HashMap<u32, u32> = HashMap::new();
    let mut last: HashMap<u32, u32> = HashMap::new();
    for (ni, node) in ir.nodes.iter().enumerate() {
        let w = waves[ni];
        for b in &node.bindings {
            if let ResourceId::TransientBuffer(tid) = b.resource {
                let id = tid.0;
                first
                    .entry(id)
                    .and_modify(|e| *e = (*e).min(w))
                    .or_insert(w);
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
pub(crate) fn transient_texture_wave_intervals(ir: &GraphIR) -> Result<HashMap<u32, (u32, u32)>> {
    let n = ir.nodes.len();
    if n == 0 {
        return Ok(HashMap::new());
    }
    let waves = graph_node_waves(ir)?;
    let mut first: HashMap<u32, u32> = HashMap::new();
    let mut last: HashMap<u32, u32> = HashMap::new();
    for (ni, node) in ir.nodes.iter().enumerate() {
        let w = waves[ni];
        for b in &node.bindings {
            if let ResourceId::TransientTexture(tid) = b.resource {
                let id = tid.0;
                first
                    .entry(id)
                    .and_modify(|e| *e = (*e).min(w))
                    .or_insert(w);
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

/// Determine which resources need barriers before `wave_idx` executes.
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
    let mut barrier_buffers: HashSet<BufferHandle> = HashSet::new();
    let mut barrier_textures: HashSet<TextureHandle> = HashSet::new();

    // Any edge crossing into this wave means the conflicting resource needs a barrier.
    for &(from, to) in edges {
        if depth[from] < wave_idx && wave_set.contains(&to) {
            for bi in &ir.nodes[from].bindings {
                for bj in &ir.nodes[to].bindings {
                    if bindings_conflict(bi, bj) {
                        // Collapse sub-range to parent for backend barrier commands.
                        match bi.resource.canonical_buffer_handle() {
                            Some(h) => {
                                barrier_buffers.insert(h);
                            }
                            None => {
                                if let ResourceId::Texture(h) = bi.resource {
                                    barrier_textures.insert(h);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut buffers: Vec<_> = barrier_buffers.into_iter().collect();
    let mut textures: Vec<_> = barrier_textures.into_iter().collect();
    buffers.sort();
    textures.sort();

    BarrierSet { buffers, textures }
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
pub fn emit_commands(ir: &GraphIR, schedule: &CompiledSchedule) -> Vec<GpuCommand> {
    let mut commands = Vec::new();

    for wave in &schedule.waves {
        if !wave.barriers_before.is_empty() {
            commands.push(GpuCommand::ResourceBarrier {
                buffers: wave.barriers_before.buffers.clone(),
                textures: wave.barriers_before.textures.clone(),
            });
        }

        // Emit blit-type nodes (clears, uploads) before dispatches within each
        // wave to minimize compute↔blit encoder transitions on Metal. Nodes in
        // the same wave have no data dependencies, so reordering is safe.
        for &idx in &wave.node_indices {
            let node = &ir.nodes[idx];
            match &node.kind {
                NodeKind::ClearBuffer {
                    buffer,
                    offset,
                    size,
                } => {
                    commands.push(GpuCommand::ClearBuffer {
                        buffer: *buffer,
                        offset: *offset,
                        size: *size,
                    });
                }
                NodeKind::WriteBuffer {
                    buffer,
                    offset,
                    data,
                } => {
                    commands.push(GpuCommand::WriteBuffer {
                        buffer: *buffer,
                        offset: *offset,
                        data: data.clone(),
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
                NodeKind::CopyTexture { src, dst } => {
                    commands.push(GpuCommand::CopyTexture {
                        src: *src,
                        dst: *dst,
                    });
                }
                NodeKind::Dispatch { .. } | NodeKind::RenderPass { .. } => {}
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
            // Build a list of (pipeline, resource_slots, user_slots, dispatch) tuples
            // for this wave's dispatch nodes, in order.
            struct PendingDispatch<'n> {
                label: &'static str,
                pipeline: crate::backend::ComputePipelineHandle,
                resource_slots: &'n Vec<u32>,
                user_slots: &'n Vec<u32>,
                x: u32,
                y: u32,
                z: u32,
            }

            // Collect all DIRECT dispatch nodes in this wave (excluding indirect).
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
                    pending.push(PendingDispatch {
                        label: node.label,
                        pipeline: *pipeline,
                        resource_slots,
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

            // Emit indirect dispatches one by one (existing path).
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
                        commands.push(GpuCommand::SetPipeline(*pipeline));
                        if !resource_slots.is_empty() || !user_slots.is_empty() {
                            commands.push(GpuCommand::BindResourcesRaw {
                                indices: resource_slots.clone(),
                                user: user_slots.clone(),
                            });
                        }
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
                // Find the run of dispatches sharing the same pipeline.
                let run_end = pending[i..]
                    .iter()
                    .take_while(|d| d.pipeline == cur_pipeline)
                    .count();
                let run = &pending[i..i + run_end];

                if run.len() > 1 {
                    // Build the flat argument buffer: [PushLayout | wg_x | wg_y | wg_z] per entry.
                    let mut arg_data: Vec<u8> =
                        Vec::with_capacity(run.len() * DISPATCH_BATCH_STRIDE);
                    for d in run {
                        // Reconstruct PushLayout bytes from resource/user slots.
                        let mut layout = crate::backend::shared::PushLayout::default();
                        crate::backend::shared::fill_raw(
                            &mut layout,
                            d.resource_slots,
                            d.user_slots,
                        );
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
                    // Single dispatch — use existing per-dispatch path.
                    let d = &run[0];
                    commands.push(GpuCommand::SetPipeline(cur_pipeline));
                    if !d.resource_slots.is_empty() || !d.user_slots.is_empty() {
                        commands.push(GpuCommand::BindResourcesRaw {
                            indices: d.resource_slots.clone(),
                            user: d.user_slots.clone(),
                        });
                    }
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

        // Handle RenderPass nodes (not expected here — checked for completeness).
        for &idx in &wave.node_indices {
            let node = &ir.nodes[idx];
            if let NodeKind::RenderPass { .. } = &node.kind {
                panic!(
                    "emit_commands: graph contains render_pass; use emit_graph_commands / TaskGraph::compile_graph_commands"
                );
            }
        }
    }

    commands
}

/// Emit [`GraphCommand`]s (compute + optional offscreen render) from the analyzed graph.
pub fn emit_graph_commands(ir: &GraphIR, schedule: &CompiledSchedule) -> Vec<GraphCommand> {
    let mut commands = Vec::new();

    for wave in &schedule.waves {
        if !wave.barriers_before.is_empty() {
            commands.push(GraphCommand::Compute(GpuCommand::ResourceBarrier {
                buffers: wave.barriers_before.buffers.clone(),
                textures: wave.barriers_before.textures.clone(),
            }));
        }

        // Blit-type nodes first to minimize encoder transitions.
        for &idx in &wave.node_indices {
            let node = &ir.nodes[idx];
            match &node.kind {
                NodeKind::ClearBuffer {
                    buffer,
                    offset,
                    size,
                } => {
                    commands.push(GraphCommand::Compute(GpuCommand::ClearBuffer {
                        buffer: *buffer,
                        offset: *offset,
                        size: *size,
                    }));
                }
                NodeKind::WriteBuffer {
                    buffer,
                    offset,
                    data,
                } => {
                    commands.push(GraphCommand::Compute(GpuCommand::WriteBuffer {
                        buffer: *buffer,
                        offset: *offset,
                        data: data.clone(),
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
                NodeKind::CopyTexture { src, dst } => {
                    commands.push(GraphCommand::Compute(GpuCommand::CopyTexture {
                        src: *src,
                        dst: *dst,
                    }));
                }
                NodeKind::Dispatch { .. } | NodeKind::RenderPass { .. } => {}
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
                    commands.push(GraphCommand::Compute(GpuCommand::SetPipeline(*pipeline)));
                    if !resource_slots.is_empty() || !user_slots.is_empty() {
                        commands.push(GraphCommand::Compute(GpuCommand::BindResourcesRaw {
                            indices: resource_slots.clone(),
                            user: user_slots.clone(),
                        }));
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
                NodeKind::RenderPass {
                    target,
                    commands: render_cmds,
                } => {
                    commands.push(GraphCommand::Render {
                        target: *target,
                        commands: render_cmds.clone(),
                    });
                }
                _ => {}
            }
        }
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
    fn dispatch_node(
        label: &'static str,
        pipeline: u64,
        bindings: Vec<(ResourceId, NodeAccess)>,
        wg: u32,
    ) -> TaskNode {
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
    fn node(
        label: &'static str,
        pipeline: u64,
        bindings: Vec<(ResourceId, NodeAccess)>,
        wg: u32,
    ) -> TaskNode {
        dispatch_node(label, pipeline, bindings, wg)
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
        assert_eq!(schedule.waves[1].barriers_before.buffers, vec![0]);
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
    fn diamond_dependency() {
        //   A (writes X)
        //  / \
        // B   C  (both read X, write Y/Z respectively)
        //  \ /
        //   D  (reads Y and Z)
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node(
                    "B",
                    2,
                    vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)],
                    1,
                ),
                node(
                    "C",
                    3,
                    vec![(buf(0), NodeAccess::Read), (buf(2), NodeAccess::Write)],
                    1,
                ),
                node(
                    "D",
                    4,
                    vec![(buf(1), NodeAccess::Read), (buf(2), NodeAccess::Read)],
                    1,
                ),
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
                node(
                    "C",
                    3,
                    vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Read)],
                    1,
                ),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);

        assert_eq!(schedule.waves.len(), 2);
        let barrier = &schedule.waves[1].barriers_before;
        assert_eq!(barrier.buffers, vec![0, 1]);
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
        let cmds = emit_commands(&ir, &schedule);

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
        let cmds = emit_commands(&ir, &schedule);

        // Single wave, no barriers
        assert_eq!(cmds.len(), 4);
        assert!(!cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
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
        let cmds = emit_commands(&ir, &schedule);

        assert_eq!(cmds.len(), 3);
        assert!(matches!(cmds[0], GpuCommand::SetPipeline(10)));
        assert!(
            matches!(cmds[1], GpuCommand::BindResourcesRaw { ref indices, .. } if indices == &[42, 7])
        );
        assert!(matches!(
            cmds[2],
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
        let cmds = emit_commands(&ir, &schedule);

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
        let cmds = emit_commands(&ir, &schedule);

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

        let cmds = emit_commands(&ir, &schedule);
        // ClearBuffer, ResourceBarrier, SetPipeline, Dispatch
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

        let cmds = emit_commands(&ir, &schedule);
        assert!(!cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
        assert!(cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::ClearBuffer { .. })));
        assert!(cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::Dispatch { .. })));
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

        let cmds = emit_commands(&ir, &schedule);
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

        let cmds = emit_commands(&ir, &schedule);
        assert!(!cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
    }

    #[test]
    fn write_texture_node_emits_write_texture_command() {
        let ir = GraphIR {
            nodes: vec![write_texture_node("up", tex(0), 0)],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let cmds = emit_commands(&ir, &schedule);

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

        let cmds = emit_commands(&ir, &schedule);
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

        let cmds = emit_commands(&ir, &schedule);
        assert!(!cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
    }

    #[test]
    fn multiple_clears_independent_same_wave() {
        // Two clears on different buffers → independent → wave 0, no barrier
        let ir = GraphIR {
            nodes: vec![
                clear_node("clear_a", buf(0), 0),
                clear_node("clear_b", buf(1), 1),
            ],
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
                node(
                    "B",
                    2,
                    vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)],
                    1,
                ),
                node(
                    "C",
                    3,
                    vec![(buf(0), NodeAccess::Read), (buf(2), NodeAccess::Write)],
                    1,
                ),
                node(
                    "D",
                    4,
                    vec![(buf(1), NodeAccess::Read), (buf(2), NodeAccess::Read)],
                    1,
                ),
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
        let barrier_count = schedule
            .waves
            .iter()
            .filter(|w| !w.barriers_before.is_empty())
            .count();
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
        ResourceId::BufferRange {
            parent,
            offset,
            len,
        }
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
            .map(|i| {
                node(
                    "dispatch",
                    i,
                    vec![(range(0, i * 256, 256), NodeAccess::Write)],
                    1,
                )
            })
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
            .map(|i| {
                node(
                    "write",
                    i,
                    vec![(range(0, i * 128, 128), NodeAccess::Write)],
                    1,
                )
            })
            .collect();
        let ir = GraphIR { nodes };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert_eq!(schedule.waves[0].node_indices.len(), 8);
    }

    #[test]
    fn waves_simulated_vello_pipeline_collapses_waves() {
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
            nodes: vec![node(
                "A",
                1,
                vec![(range(0, 0, 256), NodeAccess::ReadWrite)],
                1,
            )],
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
        assert_eq!(schedule.waves[1].barriers_before.buffers, vec![42]);
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
        assert_eq!(schedule.waves[1].barriers_before.buffers, vec![99]);
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
        assert_eq!(schedule.waves[1].barriers_before.buffers, vec![5]);
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
        assert_eq!(schedule.waves[1].barriers_before.buffers, vec![1, 2]);
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
        assert_eq!(schedule.waves[1].barriers_before.textures, vec![7]);
    }
}
