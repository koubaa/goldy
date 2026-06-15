//! `TaskGraph` — analyzed GPU task graph with automatic barrier insertion.

use super::analysis;
use super::ir::{CompiledSchedule, DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding, TaskNode, Wave};
use super::{
    ResourceId, SwapchainOutputHandle, TransientBufferSpec, TransientId, TransientTextureId, TransientTextureKey,
    TransientTextureSpec,
};
use crate::backend::{
    BufferHandle, GpuBackend, GpuCommand, GraphCommand, RenderCommand, RenderTargetHandle, TextureHandle,
};
use crate::buffer::{Buffer, BufferSource, BufferView};
use crate::compute::ComputePipeline;
use crate::device::Device;
use crate::error::GoldyError;
use crate::pipeline::RenderPipeline;
use crate::render_target::RenderTarget;
use crate::sampler::Sampler;
use crate::texture::Texture;
use crate::timeline::TimelineValue;
use crate::types::{Color, IndexFormat, ResourceAccess, ResourceHandle, TextureFormat};
use anyhow::Result;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Build a map from each upload command index in `commands` back to its IR node index.
///
/// On a cache hit we use this to refresh `Arc<[u8]>` payloads from the current IR.
fn build_upload_remap(ir: &GraphIR, commands: &[GpuCommand]) -> Vec<(usize, usize)> {
    let mut remap = Vec::with_capacity(8);
    let mut consumed = vec![false; ir.nodes.len()];
    for (cmd_idx, cmd) in commands.iter().enumerate() {
        if let Some(n_idx) = find_upload_node(ir, cmd, &consumed) {
            consumed[n_idx] = true;
            remap.push((cmd_idx, n_idx));
        }
    }
    remap
}

/// Build the upload remap for a partitioned command set.
///
/// Each entry is `(partition_index, command_index_within_partition, ir_node_index)`.
fn build_partitioned_upload_remap(ir: &GraphIR, partitions: &[Vec<GpuCommand>]) -> Vec<(usize, usize, usize)> {
    let mut remap = Vec::with_capacity(8);
    let mut consumed = vec![false; ir.nodes.len()];
    for (part_idx, commands) in partitions.iter().enumerate() {
        for (cmd_idx, cmd) in commands.iter().enumerate() {
            if let Some(n_idx) = find_upload_node(ir, cmd, &consumed) {
                consumed[n_idx] = true;
                remap.push((part_idx, cmd_idx, n_idx));
            }
        }
    }
    remap
}

/// Find the IR node index that corresponds to an upload command.
fn find_upload_node(ir: &GraphIR, cmd: &GpuCommand, consumed: &[bool]) -> Option<usize> {
    for (n_idx, node) in ir.nodes.iter().enumerate() {
        if consumed[n_idx] {
            continue;
        }
        let matches = match (cmd, &node.kind) {
            (
                GpuCommand::WriteBuffer {
                    buffer: cb, offset: co, ..
                },
                NodeKind::WriteBuffer {
                    buffer: nb, offset: no, ..
                },
            ) => cb == nb && co == no,
            (GpuCommand::WriteTexture { texture: ct, .. }, NodeKind::WriteTexture { texture: nt, .. }) => ct == nt,
            (
                GpuCommand::WriteTextureRegion {
                    texture: ct,
                    x: cx,
                    y: cy,
                    ..
                },
                NodeKind::WriteTextureRegion {
                    texture: nt,
                    x: nx,
                    y: ny,
                    ..
                },
            ) => ct == nt && cx == nx && cy == ny,
            _ => false,
        };
        if matches {
            return Some(n_idx);
        }
    }
    None
}

fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return a | b;
    }
    a / gcd(a, b) * b
}

/// A task graph that analyzes dependencies at submit time.
///
/// Build a DAG of task nodes — compute dispatches, buffer clears, and buffer
/// writes — with per-resource access declarations, then submit. Goldy analyzes
/// the graph, inserts minimal barriers, and executes with maximum parallelism.
///
/// # Example
///
/// ```rust,ignore
/// let mut graph = TaskGraph::new();
///
/// // Zero-fill the pool buffer (analyzed as a Write on the pool buffer)
/// graph.clear_buffer(&pool_backing, 0, pool.capacity());
///
/// // Dispatch nodes declare what they read/write so the analyzer can insert
/// // barriers automatically.
/// graph.node("write_data", &pipeline_a)
///     .bind_buffer(&buf, NodeAccess::Write)
///     .bind_resources_raw_slice(&[buf_idx])
///     .dispatch(64, 1, 1);
///
/// graph.node("read_data", &pipeline_b)
///     .bind_buffer(&buf, NodeAccess::Read)
///     .bind_resources_raw_slice(&[buf_idx])
///     .dispatch(64, 1, 1);
///
/// let tv = graph.submit(&ctx)?;
/// context.wait_until(tv)?;
/// ```
pub struct TaskGraph {
    ir: GraphIR,
    pub(crate) transient_specs: Vec<TransientBufferSpec>,
    next_transient_id: u32,
    pub(crate) transient_texture_specs: Vec<TransientTextureSpec>,
    next_transient_texture_id: u32,
    /// Previous frame's transient buffer spec snapshot for declaration-order stability
    /// telemetry. Updated in [`Self::clear`].
    #[cfg(debug_assertions)]
    prev_transient_shapes: Vec<(u64, u32)>,
    /// Previous frame's transient texture spec snapshot.
    #[cfg(debug_assertions)]
    prev_transient_texture_keys: Vec<TransientTextureKey>,
    /// Cached schedule + emitted command list, keyed on binding fingerprint.
    ///
    /// The fingerprint is a hash of the graph's binding structure (node count + per-node
    /// resource-access pairs).  When the graph is rebuilt with the same bindings but
    /// different data payloads (e.g. same shader dispatch topology but new scene data),
    /// the schedule and emitted commands are reused, skipping `build_edges` +
    /// `schedule_waves` + `emit_commands`.
    ///
    /// On a cache hit, upload command payloads (`WriteBuffer`, `WriteTexture`,
    /// `WriteTextureRegion`) are refreshed from the current IR via `upload_remap`
    /// — the binding fingerprint excludes data bytes, so cached `Arc<[u8]>` Arcs
    /// would otherwise be stale.  The Arc swap is a single atomic refcount bump.
    schedule_cache: Option<CompiledCacheEntry>,
    /// Node count at the time the schedule cache was last validated.
    /// When the node count hasn't changed and the cache already holds a
    /// schedule, we skip the expensive `binding_fingerprint` hash.
    schedule_validated_node_count: usize,
    /// Stamp cells for [`crate::Parcel`]s bound via [`NodeBuilder::bind_parcel`].
    /// Cleared in [`Self::clear`]; stamped in [`Self::apply_reference_stamps`] at submit.
    stamp_targets: Vec<Arc<crate::parcel::ParcelStamp>>,
}

/// Stamp every parcel bound via [`NodeBuilder::bind_parcel`] with a context-qualified timeline.
pub(crate) fn apply_stamp_targets(
    targets: &[Arc<crate::parcel::ParcelStamp>],
    ctx: crate::backend::ContextHandle,
    submit_device: &std::sync::Arc<crate::device::DeviceInner>,
    tv: TimelineValue,
) {
    for stamp in targets {
        if let Some(home) = stamp.home_device.upgrade() {
            debug_assert!(
                std::sync::Arc::ptr_eq(&home, submit_device),
                "parcel home_device must match submitting context's device"
            );
        }
        let mut table = stamp.references.lock().unwrap();
        crate::timeline::mark_reference(&mut table, ctx, tv);
    }
}

// ---- Free functions over GraphIR -------------------------------------------
//
// These are pure algorithms: they take GraphIR + backend data and produce a
// result. They carry no TaskGraph dependency and are the actual logic shared
// with Scheme via IrSubmitState.

/// Compile a `GraphIR` to backend `GraphCommand`s.
pub(crate) fn compile_graph_commands_for_ir(ir: &GraphIR) -> Vec<GraphCommand> {
    let edges = analysis::build_edges(ir);
    let schedule = analysis::schedule_waves(ir, &edges);
    analysis::emit_graph_commands(ir, &schedule, None)
}

/// Fingerprint of the graph's *binding structure* (node count + per-node resource-access pairs).
/// Suitable for schedule caching only — not for CB retention (does not cover pipeline/slots/dims).
pub(crate) fn binding_fingerprint(ir: &GraphIR) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    ir.nodes.len().hash(&mut h);
    for node in &ir.nodes {
        node.bindings.len().hash(&mut h);
        for b in &node.bindings {
            b.hash(&mut h);
        }
    }
    h.finish()
}

/// Hash dispatch `resource_slots` for retention fingerprints.
///
/// Late-bound present/swapchain placeholders are normalized to a single sentinel
/// tag so slot indices do not affect the fingerprint.
fn hash_resource_slots_for_fingerprint(slots: &[u32], h: &mut impl std::hash::Hasher) {
    use super::{PRESENT_LEASE_SLOT_PLACEHOLDER, SWAPCHAIN_SLOT_PLACEHOLDER};
    slots.len().hash(h);
    for &slot in slots {
        let tag = match slot {
            PRESENT_LEASE_SLOT_PLACEHOLDER => 0u8,
            SWAPCHAIN_SLOT_PLACEHOLDER => 1u8,
            _ => {
                2u8.hash(h);
                slot.hash(h);
                continue;
            }
        };
        tag.hash(h);
    }
}

/// Fingerprint of all state that affects the *recorded* command buffer.
///
/// Hashes everything `binding_fingerprint` covers plus per-`Dispatch` node:
/// pipeline handle, `resource_slots`, `user_slots`, and dispatch dimensions.
/// Upload/copy nodes are excluded — graphs containing them fall back to a
/// plain submit (their data is staged on every submission, not retained).
pub(crate) fn retention_fingerprint(ir: &GraphIR) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    ir.nodes.len().hash(&mut h);
    for node in &ir.nodes {
        node.bindings.len().hash(&mut h);
        for b in &node.bindings {
            b.hash(&mut h);
        }
        match &node.kind {
            NodeKind::Dispatch {
                pipeline,
                resource_slots,
                user_slots,
                dispatch,
            } => {
                0u8.hash(&mut h);
                pipeline.hash(&mut h);
                hash_resource_slots_for_fingerprint(resource_slots, &mut h);
                user_slots.hash(&mut h);
                match dispatch {
                    DispatchDim::Direct { x, y, z } => {
                        0u8.hash(&mut h);
                        x.hash(&mut h);
                        y.hash(&mut h);
                        z.hash(&mut h);
                    }
                    DispatchDim::Indirect { buffer, offset } => {
                        1u8.hash(&mut h);
                        buffer.hash(&mut h);
                        offset.hash(&mut h);
                    }
                }
            }
            NodeKind::ClearBuffer { buffer, offset, size } => {
                1u8.hash(&mut h);
                buffer.hash(&mut h);
                offset.hash(&mut h);
                size.hash(&mut h);
            }
            NodeKind::GrantRead { grant_id } => {
                3u8.hash(&mut h);
                grant_id.hash(&mut h);
            }
            NodeKind::GrantPresent { grant_id } => {
                4u8.hash(&mut h);
                grant_id.hash(&mut h);
            }
            _ => {
                2u8.hash(&mut h);
            }
        }
    }
    h.finish()
}

/// Per-partition retention fingerprint.
///
/// Hashes the same fields as [`retention_fingerprint`] but restricted to
/// the nodes assigned to `partition_idx` in the compiled schedule.  Two
/// partitions from different IRs will have the same key if and only if their
/// node sets are structurally identical — pipeline handles, resource-access
/// bindings, slot arrays, and dispatch dimensions.
///
/// The hash also folds in the partition index itself so that two consecutive
/// identical partitions receive distinct keys.
pub(crate) fn partition_fingerprint(ir: &GraphIR, schedule: &CompiledSchedule, partition_waves: &[Wave]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    // Collect node indices in this partition's waves in stable order.
    let mut node_indices: Vec<usize> = partition_waves
        .iter()
        .flat_map(|w| w.node_indices.iter().copied())
        .collect();
    node_indices.sort_unstable();
    node_indices.len().hash(&mut h);
    // Include partition's wave count to distinguish same-sized different positions.
    partition_waves.len().hash(&mut h);
    for &ni in &node_indices {
        let node = &ir.nodes[ni];
        // Wave depth of this node (its position in the schedule).
        let wave_depth = schedule
            .waves
            .iter()
            .position(|w| w.node_indices.contains(&ni))
            .unwrap_or(0);
        wave_depth.hash(&mut h);
        node.bindings.len().hash(&mut h);
        for b in &node.bindings {
            b.hash(&mut h);
        }
        match &node.kind {
            NodeKind::Dispatch {
                pipeline,
                resource_slots,
                user_slots,
                dispatch,
            } => {
                0u8.hash(&mut h);
                pipeline.hash(&mut h);
                hash_resource_slots_for_fingerprint(resource_slots, &mut h);
                user_slots.hash(&mut h);
                match dispatch {
                    DispatchDim::Direct { x, y, z } => {
                        0u8.hash(&mut h);
                        x.hash(&mut h);
                        y.hash(&mut h);
                        z.hash(&mut h);
                    }
                    DispatchDim::Indirect { buffer, offset } => {
                        1u8.hash(&mut h);
                        buffer.hash(&mut h);
                        offset.hash(&mut h);
                    }
                }
            }
            NodeKind::ClearBuffer { buffer, offset, size } => {
                1u8.hash(&mut h);
                buffer.hash(&mut h);
                offset.hash(&mut h);
                size.hash(&mut h);
            }
            NodeKind::GrantRead { grant_id } => {
                3u8.hash(&mut h);
                grant_id.hash(&mut h);
            }
            NodeKind::GrantPresent { grant_id } => {
                4u8.hash(&mut h);
                grant_id.hash(&mut h);
            }
            _ => {
                2u8.hash(&mut h);
            }
        }
    }
    h.finish()
}

/// True when all nodes in the given waves are retainable (no uploads).
///
/// Upload nodes (WriteBuffer, WriteTexture, etc.) must be staged on every
/// submit, so a partition containing them is submitted standalone rather than
/// retained.
fn partition_waves_can_retain(ir: &GraphIR, waves: &[Wave]) -> bool {
    use super::ResourceId;
    for wave in waves {
        for &ni in &wave.node_indices {
            match &ir.nodes[ni].kind {
                NodeKind::WriteBuffer { .. }
                | NodeKind::WriteTexture { .. }
                | NodeKind::WriteTextureRegion { .. }
                | NodeKind::CopyTexture { .. } => return false,
                NodeKind::CopyRenderTarget { dst, .. } => {
                    if !matches!(dst, ResourceId::PresentLease(_)) {
                        return false;
                    }
                }
                _ => {}
            }
        }
    }
    true
}

/// True when the partition's waves contain at least one [`NodeKind::RenderPass`] node.
fn partition_waves_have_render(ir: &GraphIR, waves: &[Wave]) -> bool {
    waves.iter().any(|w| {
        w.node_indices
            .iter()
            .any(|&ni| matches!(ir.nodes[ni].kind, NodeKind::RenderPass { .. }))
    })
}

/// Emit graph commands for a retainable partition from the cache or by re-emitting.
///
/// For render-pass partitions the cached `GraphCommand` list is used; re-emit only
/// happens on the very first submission (before the cache is warm) or after a
/// present-slot resolver is provided.  Compute-only partitions use the cached
/// `GpuCommand` slice wrapped in `GraphCommand::Compute`.
fn partition_graph_commands_for_retain(
    ir: &GraphIR,
    cache: &CompiledCacheEntry,
    waves: &[Wave],
    part_idx: usize,
    has_render: bool,
    resolver: Option<&super::SlotResolver>,
) -> Vec<GraphCommand> {
    // When a resolver is given (present partition), we must re-emit to patch in the
    // concrete drawable handle — the cached form was emitted without a resolver.
    if resolver.is_some() {
        if has_render {
            return analysis::emit_graph_commands_for_waves(ir, waves, resolver);
        }
        return analysis::emit_waves_to_commands(ir, waves, resolver)
            .into_iter()
            .map(GraphCommand::Compute)
            .collect();
    }

    // No resolver: use the cache.
    if has_render {
        if let Some(cmds) = cache.partitioned_graph_commands.get(part_idx).and_then(|o| o.as_ref()) {
            return cmds.clone();
        }
        // Cache not yet warm — emit fresh (will be cached next call).
        return analysis::emit_graph_commands_for_waves(ir, waves, None);
    }

    if let Some(parts) = cache.partitioned_commands.as_ref() {
        return parts[part_idx].iter().cloned().map(GraphCommand::Compute).collect();
    }
    // Cache not yet warm.
    analysis::emit_waves_to_commands(ir, waves, None)
        .into_iter()
        .map(GraphCommand::Compute)
        .collect()
}

/// Emit standalone (non-retained) commands for a non-retainable partition.
fn partition_standalone_commands(
    ir: &GraphIR,
    cache: &CompiledCacheEntry,
    waves: &[Wave],
    part_idx: usize,
    has_render: bool,
    has_present: bool,
    resolver: Option<&super::SlotResolver>,
) -> Result<Vec<GpuCommand>> {
    if has_render {
        anyhow::bail!(
            "retained submit: standalone partition contains render_pass nodes; \
             render-pass partitions must always be retainable (no upload nodes)"
        );
    }
    if has_present {
        return Ok(analysis::emit_waves_to_commands(ir, waves, resolver));
    }
    if let Some(parts) = cache.partitioned_commands.as_ref() {
        return Ok(parts[part_idx].clone());
    }
    Ok(analysis::emit_waves_to_commands(ir, waves, None))
}

/// Submit `ir`, partitioned into wave groups.
pub(crate) fn submit_resolved_ir(
    cache: &mut Option<CompiledCacheEntry>,
    context: &crate::Context,
    backend: &mut dyn GpuBackend,
    ir: &GraphIR,
) -> Result<TimelineValue> {
    let has_render = ir.nodes.iter().any(|n| matches!(n.kind, NodeKind::RenderPass { .. }));

    if has_render {
        let g = compile_graph_commands_for_ir(ir);
        return backend.submit_graph(context.backend_handle(), &g);
    }

    let fp = binding_fingerprint(ir);
    TaskGraph::get_or_build_partitioned_commands(cache, ir, fp);
    // Iterate over the partitions we just built. Render partitions have an empty
    // compute slot; they are submitted below via `partitioned_graph_commands`.
    let mut last_tv = backend.gpu_progress(context.backend_handle());
    let n_parts = cache
        .as_ref()
        .unwrap()
        .partitioned_commands
        .as_ref()
        .map(|p| p.len())
        .unwrap_or(0);
    for part_idx in 0..n_parts {
        let _tz = crate::tracy_zone!("goldy.submit_partition");
        let cache_ref = cache.as_ref().unwrap();
        if let Some(graph_cmds) = cache_ref
            .partitioned_graph_commands
            .get(part_idx)
            .and_then(|o| o.as_ref())
        {
            last_tv = backend.submit_graph(context.backend_handle(), graph_cmds)?;
        } else {
            let cmds = &cache_ref.partitioned_commands.as_ref().unwrap()[part_idx];
            last_tv = backend.submit_standalone(context.backend_handle(), cmds)?;
        }
    }
    Ok(last_tv)
}

/// Outcome of [`submit_resolved_ir_and_retain`]: how many partitions were re-recorded
/// versus resubmitted from the retained cache.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PartitionSubmitResult {
    /// Partitions that were re-recorded (and retained) this call.
    pub records: usize,
    /// Partitions that were resubmitted from the cached command list without re-recording.
    pub resubmit_hits: usize,
}

impl PartitionSubmitResult {
    /// True when every retainable partition was served from cache (zero re-records).
    pub fn all_from_cache(&self) -> bool {
        self.records == 0
    }
}

/// Submit `ir` to the backend with per-partition (slice-aware) retention.
///
/// The IR is split into the same partitions as [`submit_resolved_ir`] — one or
/// two wave groups depending on barrier cost or swapchain presence.  Each
/// partition is treated independently:
///
/// - **Upload partition** (contains `WriteBuffer`/`WriteTexture`/copy nodes):
///   submitted via `submit_standalone` on every call — data is staged fresh
///   each frame and cannot be retained.
/// - **Render partition** (contains `RenderPass` nodes):
///   submitted via `submit_graph_and_retain`; the backend retains the closed
///   list and can resubmit it on cache hit.
/// - **Pure-compute partition**: submitted via `submit_graph_and_retain`; the
///   retained command list is reused whenever the partition fingerprint matches.
///
/// Upload partitions do not prevent retention of adjacent pure-compute
/// partitions — this is the key improvement over the previous whole-IR bail-out.
///
/// Returns both the final timeline value and a [`PartitionSubmitResult`] so callers
/// can decide whether to count this call as a record or a resubmit.
pub(crate) fn submit_resolved_ir_and_retain(
    cache: &mut Option<CompiledCacheEntry>,
    context: &crate::Context,
    backend: &mut dyn GpuBackend,
    ir: &GraphIR,
) -> Result<(TimelineValue, PartitionSubmitResult)> {
    let fp = binding_fingerprint(ir);

    // Ensure schedule exists in the cache.
    TaskGraph::get_or_build_schedule(cache, ir, fp);

    // Compute partition wave ranges from the cached schedule (same split logic as
    // `emit_partitioned_commands`).
    let wave_ranges = analysis::partition_wave_ranges(ir, &cache.as_ref().unwrap().schedule);

    // Compute per-partition fingerprints up front so we can check them against the
    // cached keys without borrowing `cache` mutably yet.
    let schedule = &cache.as_ref().unwrap().schedule;
    let partition_fps: Vec<u64> = wave_ranges
        .iter()
        .enumerate()
        .map(|(part_idx, range)| {
            let waves = &schedule.waves[range.clone()];
            let raw_fp = partition_fingerprint(ir, schedule, waves);
            // Fold in partition index so identical adjacent partitions have distinct keys.
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            raw_fp.hash(&mut h);
            part_idx.hash(&mut h);
            h.finish()
        })
        .collect();

    // Ensure the partition retention key vecs are sized correctly.
    // On a topology change (fp miss) the schedule was just rebuilt; reset keys.
    {
        let entry = cache.as_mut().unwrap();
        if entry.partition_retention_keys.len() != wave_ranges.len() {
            entry.partition_retention_keys = vec![None; wave_ranges.len()];
            entry.partition_slot_keys = vec![None; wave_ranges.len()];
        }
    }

    // Build partitioned commands (cached; only emits on topology change).
    TaskGraph::get_or_build_partitioned_commands(cache, ir, fp);

    let ctx = context.backend_handle();
    let mut last_tv = backend.gpu_progress(ctx);
    let mut result = PartitionSubmitResult::default();

    for part_idx in 0..wave_ranges.len() {
        let part_fp = partition_fps[part_idx];
        let range = wave_ranges[part_idx].clone();
        let cached_key = cache.as_ref().unwrap().partition_retention_keys[part_idx];
        let waves = &cache.as_ref().unwrap().schedule.waves[range.clone()];
        let can_retain = partition_waves_can_retain(ir, waves);
        let has_render = partition_waves_have_render(ir, waves);

        if !can_retain {
            // Upload/copy partition: always submit standalone, never retain.
            let cache_entry = cache.as_ref().unwrap();
            let cmds = partition_standalone_commands(ir, cache_entry, waves, part_idx, has_render, false, None)?;
            let _tz = crate::tracy_zone!("goldy.submit_partition");
            last_tv = backend.submit_standalone(ctx, &cmds)?;
            // Leave partition_retention_keys[part_idx] as None.
            continue;
        }

        // Try to resubmit from the retained cache if the fingerprint matches.
        if cached_key == Some(part_fp) {
            if let Some(tv) = backend.try_resubmit_retained(ctx, part_fp)? {
                last_tv = tv;
                result.resubmit_hits += 1;
                continue;
            }
            // Backend evicted the entry (e.g. out of slots); fall through to re-record.
        }

        // Re-record this partition.
        let cache_entry = cache.as_ref().unwrap();
        let graph_cmds = partition_graph_commands_for_retain(ir, cache_entry, waves, part_idx, has_render, None);
        let _tz = crate::tracy_zone!("goldy.submit_partition");
        last_tv = backend.submit_graph_and_retain(ctx, &graph_cmds, part_fp)?;
        cache.as_mut().unwrap().partition_retention_keys[part_idx] = Some(part_fp);
        result.records += 1;
    }

    Ok((last_tv, result))
}

/// Resolved swapchain drawable for one present lease at submit time.
pub(crate) struct ResolvedPresentSlot {
    pub lease_id: u32,
    pub slot_id: u32,
    pub handle: crate::backend::TextureHandle,
    pub uav_index: u32,
}

/// Like [`submit_resolved_ir_and_retain`], but resolves [`ResourceId::PresentLease`]
/// bindings through `present_slots` and retains present-touching partitions per
/// backing slot (immutable CB per swapchain image index).
pub(crate) fn submit_resolved_ir_and_retain_with_presents(
    cache: &mut Option<CompiledCacheEntry>,
    context: &crate::Context,
    backend: &mut dyn GpuBackend,
    ir: &GraphIR,
    present_slots: &[ResolvedPresentSlot],
) -> Result<(TimelineValue, PartitionSubmitResult)> {
    use super::{ResolvedSwapchain, SlotResolver};

    let fp = binding_fingerprint(ir);
    TaskGraph::get_or_build_schedule(cache, ir, fp);

    let wave_ranges = analysis::partition_wave_ranges(ir, &cache.as_ref().unwrap().schedule);
    let schedule = &cache.as_ref().unwrap().schedule;
    let partition_fps: Vec<u64> = wave_ranges
        .iter()
        .enumerate()
        .map(|(part_idx, range)| {
            let waves = &schedule.waves[range.clone()];
            let raw_fp = partition_fingerprint(ir, schedule, waves);
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            raw_fp.hash(&mut h);
            part_idx.hash(&mut h);
            h.finish()
        })
        .collect();

    {
        let entry = cache.as_mut().unwrap();
        if entry.partition_retention_keys.len() != wave_ranges.len() {
            entry.partition_retention_keys = vec![None; wave_ranges.len()];
            entry.partition_slot_keys = vec![None; wave_ranges.len()];
        }
    }

    TaskGraph::get_or_build_partitioned_commands(cache, ir, fp);

    let mut resolver = SlotResolver::new();
    for slot in present_slots {
        resolver.present_leases.insert(
            slot.lease_id,
            ResolvedSwapchain {
                handle: slot.handle,
                uav_index: slot.uav_index,
            },
        );
    }

    let ctx = context.backend_handle();
    let mut last_tv = backend.gpu_progress(ctx);
    let mut result = PartitionSubmitResult::default();

    for part_idx in 0..wave_ranges.len() {
        let part_fp = partition_fps[part_idx];
        let range = wave_ranges[part_idx].clone();
        let waves = &cache.as_ref().unwrap().schedule.waves[range.clone()];
        let can_retain = partition_waves_can_retain(ir, waves);
        let has_render = partition_waves_have_render(ir, waves);
        let has_present = analysis::partition_waves_have_present(ir, waves);

        if !can_retain {
            let cache_entry = cache.as_ref().unwrap();
            let cmds = partition_standalone_commands(
                ir,
                cache_entry,
                waves,
                part_idx,
                has_render,
                has_present,
                if has_present { Some(&resolver) } else { None },
            )?;
            let _tz = crate::tracy_zone!("goldy.submit_partition");
            last_tv = backend.submit_standalone(ctx, &cmds)?;
            continue;
        }

        if has_present {
            if present_slots.is_empty() {
                return Err(anyhow::anyhow!(
                    "present partition requires at least one resolved present slot"
                ));
            }

            if !backend.retains_present_partitions() {
                let cache_entry = cache.as_ref().unwrap();
                let cmds =
                    partition_standalone_commands(ir, cache_entry, waves, part_idx, has_render, true, Some(&resolver))?;
                let _tz = crate::tracy_zone!("goldy.submit_partition");
                last_tv = backend.submit_standalone(ctx, &cmds)?;
                continue;
            }

            // Derive a single retention key from ALL present-slot assignments so that
            // multi-grant schemes produce distinct keys for each (slot_A, slot_B, …)
            // combination, rather than colliding on the first slot only.
            let slot_key = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                part_fp.hash(&mut h);
                for slot in present_slots {
                    slot.lease_id.hash(&mut h);
                    slot.slot_id.hash(&mut h);
                }
                h.finish()
            };

            let already_retained = cache.as_ref().unwrap().partition_slot_keys[part_idx]
                .as_ref()
                .map(|s| s.contains(&slot_key))
                .unwrap_or(false);

            if already_retained {
                if let Some(tv) = backend.try_resubmit_retained(ctx, slot_key)? {
                    last_tv = tv;
                    result.resubmit_hits += 1;
                    continue;
                }
            }

            let graph_cmds = partition_graph_commands_for_retain(
                ir,
                cache.as_ref().unwrap(),
                waves,
                part_idx,
                has_render,
                Some(&resolver),
            );
            let _tz = crate::tracy_zone!("goldy.submit_partition");
            last_tv = backend.submit_graph_and_retain(ctx, &graph_cmds, slot_key)?;
            {
                let entry = cache.as_mut().unwrap();
                entry.partition_slot_keys[part_idx]
                    .get_or_insert_with(std::collections::HashSet::new)
                    .insert(slot_key);
            }
            result.records += 1;
            continue;
        }

        let cached_key = cache.as_ref().unwrap().partition_retention_keys[part_idx];
        if cached_key == Some(part_fp) {
            if let Some(tv) = backend.try_resubmit_retained(ctx, part_fp)? {
                last_tv = tv;
                result.resubmit_hits += 1;
                continue;
            }
        }

        let graph_cmds =
            partition_graph_commands_for_retain(ir, cache.as_ref().unwrap(), waves, part_idx, has_render, None);
        let _tz = crate::tracy_zone!("goldy.submit_partition");
        last_tv = backend.submit_graph_and_retain(ctx, &graph_cmds, part_fp)?;
        cache.as_mut().unwrap().partition_retention_keys[part_idx] = Some(part_fp);
        result.records += 1;
    }

    Ok((last_tv, result))
}

/// Schedule cache + parcel stamp targets for a retained [`GraphIR`] submission.
///
/// Shared by [`crate::Scheme`]; [`TaskGraph`] carries an equivalent bundle inline.
pub(crate) struct IrSubmitState {
    schedule_cache: Option<CompiledCacheEntry>,
    stamp_targets: Vec<Arc<crate::parcel::ParcelStamp>>,
}

impl IrSubmitState {
    pub fn new() -> Self {
        Self {
            schedule_cache: None,
            stamp_targets: Vec::new(),
        }
    }

    pub fn register_parcel_stamp(&mut self, parcel: &crate::Parcel) {
        self.stamp_targets.push(parcel.stamp_handle());
    }

    pub fn register_stamp(&mut self, stamp: Arc<crate::parcel::ParcelStamp>) {
        self.stamp_targets.push(stamp);
    }

    pub fn apply_reference_stamps(
        &self,
        ctx: crate::backend::ContextHandle,
        submit_device: &std::sync::Arc<crate::device::DeviceInner>,
        tv: TimelineValue,
    ) {
        apply_stamp_targets(&self.stamp_targets, ctx, submit_device, tv);
    }

    /// Clear compiled schedule cache and stamp targets for in-place scheme re-record.
    pub fn reset(&mut self) {
        self.schedule_cache = None;
        self.stamp_targets.clear();
    }

    /// Record and retain the command list for `ir` on `ctx`.
    ///
    /// When `present_slots` is empty, every partition uses the standard single-key
    /// retention path. Present leases are resolved through `present_slots` at emit time.
    pub fn submit_pipelined_and_retain_with_presents(
        &mut self,
        ctx: &crate::Context,
        ir: &GraphIR,
        present_slots: &[ResolvedPresentSlot],
    ) -> Result<(TimelineValue, PartitionSubmitResult)> {
        let mut backend = ctx.device().inner.backend.lock().unwrap();
        submit_resolved_ir_and_retain_with_presents(&mut self.schedule_cache, ctx, backend.as_mut(), ir, present_slots)
    }
}

/// Cache entry holding both the wave schedule and the emitted compute command stream.
pub(crate) struct CompiledCacheEntry {
    fp: u64,
    schedule: CompiledSchedule,
    /// `None` when the graph contains render-pass nodes (caller uses `emit_graph_commands`)
    /// or when no compute commands have been emitted yet for this fingerprint.
    commands: Option<Vec<GpuCommand>>,
    /// `(command_index, ir_node_index)` for each upload command in `commands`.
    /// Used to refresh `Arc<[u8]>` payloads from the current IR on cache hit.
    upload_remap: Vec<(usize, usize)>,
    /// Cached partitioned emission output.  Populated on first call to
    /// `get_or_build_partitioned_commands`.
    ///
    /// For pure-compute partitions, the entry holds the compute `GpuCommand` list.
    /// For render-pass partitions, the entry is an empty `Vec` — the `GraphCommand`
    /// form is stored in `partitioned_graph_commands` instead.
    partitioned_commands: Option<Vec<Vec<GpuCommand>>>,
    /// `(partition_idx, cmd_idx, ir_node_idx)` for upload commands across partitions.
    partitioned_upload_remap: Vec<(usize, usize, usize)>,
    /// Cached `GraphCommand` lists for render-pass partitions.
    ///
    /// Indexed in parallel with `partitioned_commands`.  `Some(cmds)` when partition
    /// `i` has render-pass nodes; `None` when partition `i` is pure-compute.
    partitioned_graph_commands: Vec<Option<Vec<GraphCommand>>>,
    /// Per-partition retention keys for slice-aware retention.
    ///
    /// `Some(key)` when partition `i` was last successfully retained with that key;
    /// `None` when the partition has not yet been retained (e.g. it contains upload
    /// nodes and is submitted standalone rather than retained).
    partition_retention_keys: Vec<Option<u64>>,
    /// Per-partition set of slot-combination keys for present-aware retention.
    ///
    /// Each entry is the set of `slot_key` values (derived from all present lease
    /// slot assignments for that frame) that have been successfully retained.
    /// A cache hit occurs when the current frame's `slot_key` is already in the set
    /// and the backend can resubmit the retained command buffer.
    partition_slot_keys: Vec<Option<std::collections::HashSet<u64>>>,
}

/// Per-page cache of a fully-lowered [`GraphIR`].
///
/// Keyed on `(spec_fp, base_offset)` where `spec_fp` is a hash of the transient
/// spec set and `base_offset` is the page's deterministic heap offset. When both
/// match, the IR is returned directly, skipping all lowering work.
///
/// At pipeline depth D, at most D entries are live simultaneously.
impl TaskGraph {
    pub fn new() -> Self {
        Self {
            ir: GraphIR::default(),
            transient_specs: Vec::new(),
            next_transient_id: 0,
            transient_texture_specs: Vec::new(),
            next_transient_texture_id: 0,
            schedule_cache: None,
            schedule_validated_node_count: 0,
            stamp_targets: Vec::new(),
            #[cfg(debug_assertions)]
            prev_transient_shapes: Vec::new(),
            #[cfg(debug_assertions)]
            prev_transient_texture_keys: Vec::new(),
        }
    }

    /// Register a transient GPU buffer suballocation for this graph.
    ///
    /// The backing memory is a single device buffer allocated for the duration of
    /// [`crate::Context::submit`]. Transients whose live ranges (in the compiled wave
    /// schedule) do not overlap may alias within that heap to reduce allocation size.
    /// Graphs using transients **block until the submit completes** when using
    /// [`crate::Context::submit`] (so the CPU does not record overlapping standalone graphs that
    /// reuse the same placement-heap protocol). For pipelined multi-submit frames, use
    /// [`crate::Context::submit_pipelined`] or the surface path / [`crate::FrameOrchestrator`].
    pub fn transient_buffer(&mut self, size: u64) -> TransientId {
        self.transient_buffer_with_stride(size, 4)
    }

    /// Like [`Self::transient_buffer`] but with an explicit element stride for the
    /// structured buffer descriptor. The stride is forwarded to
    /// [`crate::Buffer::create_view`] when the transient is materialised.
    ///
    /// ## Stable slot identity contract
    ///
    /// The returned [`TransientId`] encodes only the declaration order within this
    /// recording phase (0, 1, 2, …). [`Self::clear`] resets the counter to 0, so the
    /// **N-th call** to any `transient_buffer*` method in the next frame produces
    /// `TransientId(N)`. The [`crate::placement_heap::PlacementHeap`] view cache relies
    /// on this: it keys cached `BufferView` objects on `(slot_id, shape, placement)`,
    /// where `slot_id == N`.
    ///
    /// **Recordings must be deterministic in steady state.** If the declaration order
    /// is data-dependent, slots that diverge will always be cache misses. Debug builds
    /// emit a `tracing::debug!` warning when a slot's shape changes between frames.
    pub fn transient_buffer_with_stride(&mut self, size: u64, stride: u32) -> TransientId {
        let id = self.next_transient_id;
        self.next_transient_id += 1;

        #[cfg(debug_assertions)]
        if (id as usize) < self.prev_transient_shapes.len() {
            let (prev_size, prev_stride) = self.prev_transient_shapes[id as usize];
            if prev_size != size || prev_stride != stride {
                tracing::debug!(
                    slot = id,
                    prev_size,
                    prev_stride,
                    new_size = size,
                    new_stride = stride,
                    "transient buffer slot shape changed: view cache miss for slot {id}"
                );
            }
        }

        self.transient_specs.push(TransientBufferSpec { id, size, stride });
        TransientId(id)
    }

    /// Register a transient texture (same dimensions and format) for this graph.
    ///
    /// Non-overlapping wave lifetimes may alias onto one backing texture; see
    /// [`Self::transient_buffer`] for scheduling behavior. [`crate::Context::submit`]
    /// waits until completion when transients are used; use [`crate::Context::submit_pipelined`]
    /// for overlapping submissions in a managed frame loop.
    ///
    /// ## Stable slot identity contract
    ///
    /// The returned [`TransientTextureId`] encodes the declaration order within this
    /// recording phase. The texture cache in [`crate::placement_heap::PlacementHeap`]
    /// keys on the graph-coloring color index, which is derived from stable spec ordering.
    /// Recordings must be deterministic; debug builds warn when a slot's shape changes.
    pub fn transient_texture(&mut self, width: u32, height: u32, format: TextureFormat) -> TransientTextureId {
        let id = self.next_transient_texture_id;
        self.next_transient_texture_id += 1;

        #[cfg(debug_assertions)]
        if (id as usize) < self.prev_transient_texture_keys.len() {
            let prev = self.prev_transient_texture_keys[id as usize];
            if prev.width != width || prev.height != height || prev.format != format {
                tracing::debug!(
                    slot = id,
                    prev_width = prev.width,
                    prev_height = prev.height,
                    ?prev.format,
                    new_width = width,
                    new_height = height,
                    ?format,
                    "transient texture slot shape changed: texture cache miss for slot {id}"
                );
            }
        }

        self.transient_texture_specs.push(TransientTextureSpec {
            id,
            width,
            height,
            format,
        });
        TransientTextureId(id)
    }

    fn needs_transient_gpu_wait(&self) -> bool {
        !self.transient_specs.is_empty() || !self.transient_texture_specs.is_empty()
    }
    /// Returns `(total_size, required_base_alignment, offset_map)`.
    ///
    /// `required_base_alignment` is the LCM of all per-color alignments. When
    /// the layout is placed at a non-zero base offset (e.g. inside a ring
    /// buffer), that base must be a multiple of this value so that every
    /// internal offset remains stride-aligned for its buffer view descriptor.
    pub(crate) fn transient_heap_size_and_layout(&self, node_waves: &[u32]) -> Result<(u64, u64, HashMap<u32, u64>)> {
        Self::transient_heap_layout(&self.transient_specs, &self.ir, node_waves)
    }

    pub(crate) fn submit_with_backend(
        &mut self,
        context: &crate::Context,
        backend: &mut dyn GpuBackend,
        _transient_buffer_ranges: Option<&HashMap<u32, (BufferHandle, u64, u64)>>,
        _transient_texture_handles: &HashMap<u32, TextureHandle>,
        wait_for_transient_completion: bool,
    ) -> Result<TimelineValue> {
        debug_assert!(
            self.transient_specs.is_empty() && self.transient_texture_specs.is_empty(),
            "submit_with_backend: transient resources must go through submit_ir_with_resolver"
        );

        let tv = submit_resolved_ir(&mut self.schedule_cache, context, backend, &self.ir)?;
        if wait_for_transient_completion && self.needs_transient_gpu_wait() {
            backend.wait_until(context.backend_handle(), tv)?;
        }
        Ok(tv)
    }

    /// Like [`Self::submit_with_backend`] but retains the closed command list keyed by
    /// the graph's [`Self::retention_fingerprint`].
    ///
    /// Graphs with transient resources, render passes, or upload nodes are not eligible for
    /// retention and silently fall back to a normal submit.
    /// Called from [`Context::submit_pipelined_and_retain`].
    pub(crate) fn submit_with_backend_and_retain(
        &mut self,
        context: &crate::Context,
        backend: &mut dyn GpuBackend,
    ) -> Result<TimelineValue> {
        // Only retain pure compute graphs (no transients, no render passes, no uploads).
        if self.has_transient_resources() {
            return submit_resolved_ir(&mut self.schedule_cache, context, backend, &self.ir);
        }
        submit_resolved_ir_and_retain(&mut self.schedule_cache, context, backend, &self.ir).map(|(tv, _)| tv)
    }

    /// Pack transient buffers into a heap using wave live ranges: transients whose
    /// lifetimes do not overlap (in the compiled wave schedule) may alias the same bytes.
    fn transient_heap_layout(
        specs: &[TransientBufferSpec],
        ir: &GraphIR,
        node_waves: &[u32],
    ) -> Result<(u64, u64, HashMap<u32, u64>)> {
        if specs.is_empty() {
            return Ok((0, 256, HashMap::new()));
        }

        let intervals = analysis::transient_wave_intervals(ir, node_waves)?;
        for s in specs {
            if !intervals.contains_key(&s.id) {
                anyhow::bail!("transient_buffer id {} is never referenced by any graph node", s.id);
            }
        }

        #[derive(Clone)]
        struct Item {
            id: u32,
            start: u32,
            end: u32,
        }

        let mut items: Vec<Item> = specs
            .iter()
            .map(|s| {
                let (st, en) = intervals[&s.id];
                Item {
                    id: s.id,
                    start: st,
                    end: en,
                }
            })
            .collect();
        items.sort_by_key(|i| (i.end, i.start));

        fn wave_intervals_overlap(a: (u32, u32), b: (u32, u32)) -> bool {
            !(a.1 < b.0 || b.1 < a.0)
        }

        let mut colors: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut id_to_color: HashMap<u32, usize> = HashMap::new();

        for it in items {
            let iv = (it.start, it.end);
            let mut chosen = None;
            for (c, assigned) in colors.iter().enumerate() {
                if assigned.iter().all(|&other| !wave_intervals_overlap(iv, other)) {
                    chosen = Some(c);
                    break;
                }
            }
            let c = match chosen {
                Some(c) => c,
                None => {
                    colors.push(Vec::new());
                    colors.len() - 1
                }
            };
            colors[c].push(iv);
            id_to_color.insert(it.id, c);
        }

        let mut color_max: Vec<u64> = vec![0; colors.len()];
        let mut color_align: Vec<u64> = vec![256; colors.len()];
        for s in specs {
            let c = id_to_color[&s.id];
            color_max[c] = color_max[c].max(s.size);
            let stride = s.stride.max(1) as u64;
            color_align[c] = lcm(color_align[c], stride);
        }

        let mut max_align = 256u64;
        let mut next_off = 0u64;
        let mut color_base: Vec<u64> = vec![0; colors.len()];
        for c in 0..colors.len() {
            let a = color_align[c];
            max_align = lcm(max_align, a);
            next_off = next_off.div_ceil(a) * a;
            color_base[c] = next_off;
            next_off = next_off
                .checked_add(color_max[c])
                .ok_or_else(|| anyhow::anyhow!("transient heap layout overflow"))?;
        }

        let mut m = HashMap::new();
        for s in specs {
            let c = id_to_color[&s.id];
            m.insert(s.id, color_base[c]);
        }

        Ok((next_off, max_align, m))
    }

    /// Submit the original (unlowered) IR to the backend, resolving transient
    /// and swapchain slots at emission time via `resolver`.
    ///
    /// The schedule is cached (graph topology is invariant); commands are emitted
    /// fresh each frame through the resolver.
    pub(crate) fn submit_ir_with_resolver(
        &mut self,
        context: &crate::Context,
        backend: &mut dyn GpuBackend,
        resolver: &super::SlotResolver,
        wait_for_transient_completion: bool,
    ) -> Result<TimelineValue> {
        let has_render = self
            .ir
            .nodes
            .iter()
            .any(|n| matches!(n.kind, NodeKind::RenderPass { .. }));

        // Build the schedule from disjoint fields to avoid borrow conflicts.
        let fp = Self::binding_fingerprint(&self.ir);
        let schedule = Self::get_or_build_schedule(&mut self.schedule_cache, &self.ir, fp);

        if has_render {
            let g = analysis::emit_graph_commands(&self.ir, schedule, Some(resolver));
            let tv = backend.submit_graph(context.backend_handle(), &g)?;
            if wait_for_transient_completion && self.needs_transient_gpu_wait() {
                backend.wait_until(context.backend_handle(), tv)?;
            }
            return Ok(tv);
        }

        let partitions = analysis::emit_partitioned_commands(&self.ir, schedule, Some(resolver));
        let mut last_tv = backend.gpu_progress(context.backend_handle());
        for partition in &partitions {
            let _tz = crate::tracy_zone!("goldy.submit_partition");
            last_tv = backend.submit_standalone(context.backend_handle(), partition)?;
        }
        if wait_for_transient_completion && self.needs_transient_gpu_wait() {
            backend.wait_until(context.backend_handle(), last_tv)?;
        }
        Ok(last_tv)
    }

    /// Like the old `allocate_transient_textures` but uses the heap's texture cache.
    ///
    /// Textures are owned by the `PlacementHeap` across frames; callers do not
    /// receive a keepalive `Vec<Texture>`. The heap evicts stale entries via
    /// `defer_release` when shapes change or the heap is grown.
    pub(crate) fn resolve_transient_textures_with_heap(
        &self,
        device: &Device,
        heap: &mut crate::placement_heap::PlacementHeap,
        node_waves: &[u32],
        page_slot: usize,
        retired_timeline: crate::timeline::TimelineValue,
    ) -> Result<HashMap<u32, TextureHandle>> {
        if self.transient_texture_specs.is_empty() {
            return Ok(HashMap::new());
        }
        let intervals = analysis::transient_texture_wave_intervals(&self.ir, node_waves)?;
        for s in &self.transient_texture_specs {
            if !intervals.contains_key(&s.id) {
                anyhow::bail!("transient_texture id {} is never referenced by any graph node", s.id);
            }
        }
        let (id_to_color, color_keys) = Self::transient_texture_coloring(&self.transient_texture_specs, &intervals)?;
        let per_color_handles = heap.get_or_create_textures(device, &color_keys, page_slot, retired_timeline)?;
        let mut out = HashMap::new();
        for s in &self.transient_texture_specs {
            let c = id_to_color[&s.id];
            out.insert(s.id, per_color_handles[c]);
        }
        Ok(out)
    }

    fn transient_texture_key(spec: &TransientTextureSpec) -> TransientTextureKey {
        TransientTextureKey {
            width: spec.width,
            height: spec.height,
            format: spec.format,
        }
    }

    fn transient_texture_coloring(
        specs: &[TransientTextureSpec],
        intervals: &HashMap<u32, (u32, u32)>,
    ) -> Result<(HashMap<u32, usize>, Vec<TransientTextureKey>)> {
        #[derive(Clone)]
        struct Item {
            id: u32,
            start: u32,
            end: u32,
            key: TransientTextureKey,
        }
        let mut items: Vec<Item> = specs
            .iter()
            .map(|s| {
                let (st, en) = intervals[&s.id];
                Item {
                    id: s.id,
                    start: st,
                    end: en,
                    key: Self::transient_texture_key(s),
                }
            })
            .collect();
        items.sort_by_key(|i| (i.end, i.start));
        fn wave_intervals_overlap(a: (u32, u32), b: (u32, u32)) -> bool {
            !(a.1 < b.0 || b.1 < a.0)
        }
        let mut colors: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut color_keys: Vec<TransientTextureKey> = Vec::new();
        let mut id_to_color: HashMap<u32, usize> = HashMap::new();
        for it in items {
            let iv = (it.start, it.end);
            let mut chosen = None;
            for (c, assigned) in colors.iter().enumerate() {
                if color_keys[c] != it.key {
                    continue;
                }
                if assigned.iter().all(|&other| !wave_intervals_overlap(iv, other)) {
                    chosen = Some(c);
                    break;
                }
            }
            let c = match chosen {
                Some(c) => c,
                None => {
                    colors.push(Vec::new());
                    color_keys.push(it.key);
                    colors.len() - 1
                }
            };
            colors[c].push(iv);
            id_to_color.insert(it.id, c);
        }
        Ok((id_to_color, color_keys))
    }

    /// True if the graph has any transient resources (buffers and/or textures).
    pub fn has_transient_resources(&self) -> bool {
        !self.transient_specs.is_empty() || !self.transient_texture_specs.is_empty()
    }

    /// True if the graph contains at least one transient buffer (graph-colored).
    pub(crate) fn has_transient_buffers(&self) -> bool {
        !self.transient_specs.is_empty()
    }

    /// Access the transient buffer specs (id + size) for coloring/layout.
    pub(crate) fn transient_specs(&self) -> &[TransientBufferSpec] {
        &self.transient_specs
    }

    /// Declare that this graph will write to a swapchain output.
    ///
    /// Returns a [`SwapchainOutputHandle`] that must be passed to
    /// [`NodeBuilder::bind_swapchain_output`] when recording the final
    /// (fine-pass) dispatch node.  The concrete `TextureHandle` is resolved
    /// at submit time inside [`Surface::submit_graph`](crate::Surface::submit_graph).
    ///
    /// Call this exactly once per graph before recording fine-pass nodes.
    pub fn declare_swapchain_output(&mut self) -> SwapchainOutputHandle {
        SwapchainOutputHandle
    }

    /// Returns `true` if the graph contains any node that binds `SwapchainOutput`.
    pub(crate) fn has_swapchain_output(&self) -> bool {
        self.ir
            .nodes
            .iter()
            .any(|n| n.bindings.iter().any(|b| b.resource == ResourceId::SwapchainOutput))
    }

    fn has_render_passes_in_ir(ir: &GraphIR) -> bool {
        ir.nodes.iter().any(|n| matches!(n.kind, NodeKind::RenderPass { .. }))
    }

    pub fn has_render_passes(&self) -> bool {
        Self::has_render_passes_in_ir(&self.ir)
    }

    /// Add a compute dispatch node to the graph. The returned [`NodeBuilder`] must
    /// be finalized with [`NodeBuilder::dispatch`] or [`NodeBuilder::dispatch_indirect`].
    pub fn node<'a>(&'a mut self, label: &'static str, pipeline: &ComputePipeline) -> NodeBuilder<'a> {
        NodeBuilder {
            graph: self,
            label,
            pipeline: pipeline.handle,
            bindings: Vec::new(),
            resource_slots: Vec::new(),
            user_slots: Vec::new(),
        }
    }

    /// Add a zero-fill node for `buffer[offset..offset+size]`.
    ///
    /// The node is declared as `NodeAccess::Write` on the buffer so the analyzer
    /// can insert barriers between this clear and any subsequent reader.
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: u64) {
        self.ir.nodes.push(TaskNode {
            label: "clear_buffer",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Buffer(buffer.handle),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::ClearBuffer {
                buffer: buffer.gpu_buffer_handle(),
                offset,
                size,
            },
        });
    }

    /// Add a zero-fill node for a view region.
    ///
    /// `offset` is relative to the view's start; the absolute parent-buffer
    /// offset is computed internally. If `size` is 0, clears from `offset`
    /// to the end of the view.
    ///
    /// The node is declared as `NodeAccess::Write` on the view's `BufferRange`
    /// so the analyzer can detect conflicts with overlapping views or the full
    /// parent buffer while allowing independent (non-overlapping) views to run
    /// concurrently in the same wave.
    pub fn clear_buffer_view(&mut self, view: &BufferView, offset: u64, size: u64) {
        let clear_size = if size == 0 {
            view.size().saturating_sub(offset)
        } else {
            size
        };
        self.ir.nodes.push(TaskNode {
            label: "clear_buffer_view",
            bindings: vec![ResourceBinding {
                resource: ResourceId::BufferRange {
                    parent: view.parent_handle(),
                    offset: view.offset(),
                    len: view.size(),
                },
                access: NodeAccess::Write,
            }],
            kind: NodeKind::ClearBuffer {
                buffer: view.parent_handle(),
                offset: view.offset() + offset,
                size: clear_size,
            },
        });
    }

    /// Add a CPU→GPU write node for `buffer`.
    ///
    /// The data is uploaded to the buffer in the same GPU submission as the
    /// surrounding dispatches. The node is declared as `NodeAccess::Write` so
    /// the analyzer inserts the necessary barrier between this write and any
    /// subsequent reader, and serializes it after any prior reader (WAR).
    pub fn write_buffer(&mut self, buffer: &Buffer, offset: u64, data: Vec<u8>) {
        self.ir.nodes.push(TaskNode {
            label: "write_buffer",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Buffer(buffer.handle),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteBuffer {
                buffer: buffer.gpu_buffer_handle(),
                offset,
                data: Arc::from(data),
            },
        });
    }

    /// Add a CPU→GPU write node for a retained buffer [`crate::Parcel`].
    ///
    /// Like [`Self::write_buffer`], but accepts an opaque parcel instead of a
    /// [`Buffer`] handle. Valid only for non-mosaic buffer parcels (the same
    /// restriction as a direct buffer write). The analyzer inserts
    /// barriers between this write and any subsequent reader in the graph.
    pub fn write_parcel(&mut self, parcel: &crate::Parcel, offset: u64, data: Vec<u8>) -> Result<()> {
        let (buffer, resource) = parcel.write_buffer_target()?;
        self.ir.nodes.push(TaskNode {
            label: "write_parcel",
            bindings: vec![ResourceBinding {
                resource,
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteBuffer {
                buffer,
                offset,
                data: Arc::from(data),
            },
        });
        Ok(())
    }

    /// Add a CPU→GPU texture upload node (full image).
    ///
    /// Data length must match [`Texture::byte_size`]. The upload is batched with
    /// the same submission as surrounding graph nodes; the analyzer inserts barriers
    /// before any node that reads the texture.
    pub fn write_texture(&mut self, texture: &Texture, data: Vec<u8>) -> Result<()> {
        let expected = texture.byte_size();
        if data.len() != expected {
            anyhow::bail!("write_texture: expected {} bytes, got {}", expected, data.len());
        }
        let width = texture.width();
        let height = texture.height();
        let th = texture.gpu_handle();
        self.ir.nodes.push(TaskNode {
            label: "write_texture",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Texture(th),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteTexture {
                texture: th,
                data: Arc::from(data),
                width,
                height,
            },
        });
        Ok(())
    }

    /// Add a GPU-side full-texture copy node from `src` to `dst`.
    ///
    /// Both textures must have identical dimensions and compatible formats.
    /// The backend inserts appropriate memory barriers and layout transitions.
    /// `src` should have [`crate::types::TextureFlags::COPY_SRC`] and
    /// `dst` should have [`crate::types::TextureFlags::COPY_DST`].
    pub fn copy_texture(&mut self, src: &Texture, dst: &Texture) {
        let src_h = src.gpu_handle();
        let dst_h = dst.gpu_handle();
        self.ir.nodes.push(TaskNode {
            label: "copy_texture",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::Texture(src_h),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::Texture(dst_h),
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyTexture {
                src: src_h,
                dst: ResourceId::Texture(dst_h),
            },
        });
    }

    /// Add a GPU-side full-texture copy node from `src` to the late-bound swapchain output.
    ///
    /// The concrete swapchain image is resolved by [`Surface::submit_graph`](crate::Surface::submit_graph)
    /// after acquire. This keeps swapchain presentation as an abstract graph resource while allowing
    /// expensive producer work to run before WSI image availability.
    pub fn copy_texture_to_swapchain(&mut self, src: &Texture, _dst: SwapchainOutputHandle) {
        let src_h = src.gpu_handle();
        self.ir.nodes.push(TaskNode {
            label: "copy_texture_to_swapchain",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::Texture(src_h),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::SwapchainOutput,
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyTexture {
                src: src_h,
                dst: ResourceId::SwapchainOutput,
            },
        });
    }

    /// Copy an offscreen [`crate::RenderTarget`] color buffer to the late-bound swapchain output.
    ///
    /// Record this after a [`Self::render_pass`] that targets the same `src` render target.
    /// The analyzer orders the copy after the render pass via the shared
    /// render-target resource binding.
    pub fn copy_render_target_to_swapchain(&mut self, src: &RenderTarget, _dst: SwapchainOutputHandle) {
        let src_h = src.backend_handle();
        self.ir.nodes.push(TaskNode {
            label: "copy_render_target_to_swapchain",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::RenderTarget(src_h),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::SwapchainOutput,
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyRenderTarget {
                src: src_h,
                dst: ResourceId::SwapchainOutput,
            },
        });
    }

    /// Add a CPU→GPU texture upload node for a subrectangle.
    pub fn write_texture_region(
        &mut self,
        texture: &Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: Vec<u8>,
    ) -> Result<()> {
        if x + width > texture.width() || y + height > texture.height() {
            anyhow::bail!(
                "write_texture_region: {}x{} at ({},{}) exceeds {}x{} texture",
                width,
                height,
                x,
                y,
                texture.width(),
                texture.height()
            );
        }
        let expected = (width * height * texture.format().bytes_per_pixel()) as usize;
        if data.len() != expected {
            anyhow::bail!(
                "write_texture_region: expected {} bytes for {}x{} region, got {}",
                expected,
                width,
                height,
                data.len()
            );
        }
        let th = texture.gpu_handle();
        self.ir.nodes.push(TaskNode {
            label: "write_texture_region",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Texture(th),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteTextureRegion {
                texture: th,
                x,
                y,
                width,
                height,
                data: Arc::from(data),
            },
        });
        Ok(())
    }

    /// Push a fully recorded task node (used by [`super::record::RenderPassRecord`] and tests).
    pub(crate) fn push_task_node(&mut self, node: super::ir::TaskNode) {
        self.ir.nodes.push(node);
    }

    pub(crate) fn extend_stamp_targets(&mut self, stamps: Vec<Arc<crate::parcel::ParcelStamp>>) {
        self.stamp_targets.extend(stamps);
    }

    /// Begin building an offscreen [`crate::RenderTarget`] render pass node.
    pub fn render_pass<'a>(&'a mut self, label: &'static str, target: &RenderTarget) -> RenderPassBuilder<'a> {
        RenderPassBuilder {
            graph: self,
            label,
            target: target.backend_handle(),
            bindings: Vec::new(),
            commands: Vec::new(),
            push_constant_handles: Vec::new(),
        }
    }

    /// Analyze the graph and submit all tasks with optimal barriers.
    /// Returns the device [`TimelineValue`] to pass to [`Context::wait_until`](crate::Context::wait_until).
    pub fn submit(&mut self, context: &crate::Context) -> Result<TimelineValue, GoldyError> {
        context.submit(self)
    }

    /// Analyze the graph, submit, and block until complete.
    pub fn dispatch(&mut self, context: &crate::Context) -> Result<(), GoldyError> {
        context.dispatch(self)
    }

    /// Number of task nodes in the graph.
    pub fn len(&self) -> usize {
        self.ir.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ir.nodes.is_empty()
    }

    /// Returns `true` if the graph contains any `WriteBuffer` nodes.
    ///
    /// Used by the command-list retention path to detect staging-belt uploads.
    /// A command list that contains `CopyBufferRegion` commands sourced from the
    /// staging belt cannot be safely retained across frames because the staging
    /// belt recycles its chunks once the GPU fence is met.  Callers should
    /// fall back to a normal (non-retained) submit when this returns `true`.
    pub fn has_write_buffer(&self) -> bool {
        self.ir
            .nodes
            .iter()
            .any(|n| matches!(n.kind, NodeKind::WriteBuffer { .. }))
    }

    /// Reset the graph to empty while retaining all heap allocations for reuse.
    ///
    /// ## DEPRECATION PLANNED (retained-scheme design §2) — and a misnomer
    ///
    /// This is **not** a collection clear. A caller of `clear()` on "a data structure
    /// containing nodes" expects it to resize to zero nodes; this method additionally
    /// performs an end-of-frame *ritual*: it snapshots transient shapes for telemetry,
    /// resets the slot-identity counters (the `TransientId(N)` = "N-th declaration"
    /// contract below), drops parcel stamp targets, and invalidates the retention key.
    /// None of that is predictable from the name, and its existence teaches the
    /// `clear()`+rebuild-per-frame anti-pattern that defeats retained submission
    /// (every frame pays full record cost; nothing can resubmit).
    ///
    /// The `#[deprecated]` attribute lands once `Scheme` exists as the migration
    /// target (deprecating with no alternative would strand clients).
    ///
    /// ## Slot identity reset
    ///
    /// Resetting `next_transient_id` to 0 is what makes transient slot identity
    /// deterministic across frames: the N-th call to `transient_buffer*` / `transient_texture`
    /// in the next recording always produces the same `TransientId(N)` /
    /// `TransientTextureId(N)`. The placement-heap view cache depends on this contract;
    /// do not alter the reset order or skip it between frames.
    pub fn clear(&mut self) {
        // Snapshot current specs for debug-assertions telemetry before clearing.
        #[cfg(debug_assertions)]
        {
            self.prev_transient_shapes = self.transient_specs.iter().map(|s| (s.size, s.stride)).collect();
            self.prev_transient_texture_keys = self
                .transient_texture_specs
                .iter()
                .map(|s| TransientTextureKey {
                    width: s.width,
                    height: s.height,
                    format: s.format,
                })
                .collect();
        }
        self.ir.nodes.clear();
        self.transient_specs.clear();
        self.transient_texture_specs.clear();
        self.next_transient_id = 0;
        self.next_transient_texture_id = 0;
        self.stamp_targets.clear();
    }

    /// Stamp every [`crate::Parcel`] bound via [`NodeBuilder::bind_parcel`] with the
    /// context-qualified timeline value of the submission that just completed.
    pub(crate) fn apply_reference_stamps(
        &self,
        ctx: crate::backend::ContextHandle,
        submit_device: &std::sync::Arc<crate::device::DeviceInner>,
        tv: TimelineValue,
    ) {
        apply_stamp_targets(&self.stamp_targets, ctx, submit_device, tv);
    }

    /// Move parcel stamp cells off the graph for surface submit (applied at [`crate::Frame::submit_frame`]).
    pub(crate) fn take_stamp_targets(&mut self) -> Vec<Arc<crate::parcel::ParcelStamp>> {
        std::mem::take(&mut self.stamp_targets)
    }

    /// Access the raw IR for internal use (e.g. transient lowering from outside the task_graph module).
    pub(crate) fn ir(&self) -> &GraphIR {
        &self.ir
    }

    /// Compute a fingerprint of the graph's binding structure for **schedule caching only**.
    ///
    /// Hashes node count and per-node `ResourceBinding { resource_id, access }` pairs.
    /// This is exactly what the wave scheduler needs: two graphs with the same binding
    /// fingerprint produce the same wave schedule and emitted `GpuCommand` stream.
    ///
    /// **Not suitable for CB retention.** It does not hash `NodeKind` fields (pipeline
    /// handle, `resource_slots`, `user_slots`, dispatch dimensions). Two graphs with
    /// identical bindings but different pipelines or dispatch sizes would share this
    /// fingerprint but record different `VkCommandBuffer` contents. Use
    /// [`Self::compute_retention_fingerprint`] when keying retained command lists.
    pub fn compute_binding_fingerprint(&self) -> u64 {
        Self::binding_fingerprint(&self.ir)
    }

    /// Hashes `(resource_id, access)` pairs per node. Data payloads (`WriteBuffer`
    /// bytes, dispatch dimensions) are intentionally excluded — the schedule depends
    /// only on which resources are read/written, not what values they carry.
    fn binding_fingerprint(ir: &GraphIR) -> u64 {
        binding_fingerprint(ir)
    }

    /// Compute a fingerprint of the transient spec set (buffers + textures).
    ///
    /// Used by [`Self::get_or_lower_resolved_ir`] to detect shape changes between
    /// frames. A mismatch invalidates the resolved IR cache.
    fn schedule_fp(&mut self) -> u64 {
        let n = self.ir.nodes.len();
        if let Some(ref entry) = self.schedule_cache {
            if n == self.schedule_validated_node_count {
                return entry.fp;
            }
        }
        let fp = Self::binding_fingerprint(&self.ir);
        self.schedule_validated_node_count = n;
        fp
    }

    /// Compute a fingerprint of all state that affects the *recorded* command buffer.
    ///
    /// Hashes everything `binding_fingerprint` captures, plus per-`Dispatch` node:
    /// pipeline handle, `resource_slots`, `user_slots`, and dispatch dimensions.
    /// Also hashes `ClearBuffer` target identity.
    ///
    /// Upload nodes (`WriteBuffer`, `WriteTexture`, `WriteTextureRegion`, `CopyTexture`)
    /// are excluded because their data is replayed via staging on every submit; retaining
    /// a CB that contains upload commands is currently unsupported (those graphs fall back
    /// to a normal submit).
    ///
    /// Pass this value to [`crate::Context::try_resubmit_retained`] to attempt zero-cost
    /// resubmission; [`crate::Context::submit_pipelined_and_retain`] derives and stores the
    /// same key internally.
    pub fn compute_retention_fingerprint(&self) -> u64 {
        Self::retention_fingerprint(&self.ir)
    }

    pub(crate) fn retention_fingerprint(ir: &GraphIR) -> u64 {
        retention_fingerprint(ir)
    }

    /// Return a reference to the compiled schedule for `ir`, using the cache when possible.
    ///
    /// On a miss the schedule is built and stored; on a hit it is returned directly.
    fn get_or_build_schedule<'c>(
        cache: &'c mut Option<CompiledCacheEntry>,
        ir: &GraphIR,
        fp: u64,
    ) -> &'c CompiledSchedule {
        if cache.as_ref().is_some_and(|e| e.fp == fp) {
            tracing::trace!(target: "goldy::schedule_cache", hit = true, fp, "schedule");
            return &cache.as_ref().unwrap().schedule;
        }
        tracing::trace!(target: "goldy::schedule_cache", hit = false, fp, "schedule");
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        *cache = Some(CompiledCacheEntry {
            fp,
            schedule,
            commands: None,
            upload_remap: Vec::new(),
            partitioned_commands: None,
            partitioned_upload_remap: Vec::new(),
            partition_retention_keys: Vec::new(),
            partition_slot_keys: Vec::new(),
            partitioned_graph_commands: Vec::new(),
        });
        &cache.as_ref().unwrap().schedule
    }

    /// Return cached emitted commands for `ir`, building them if necessary.
    ///
    /// Used by the standalone-compute submit path to skip `emit_commands` when the
    /// graph topology is unchanged.  Upload command payloads are refreshed from
    /// the current IR on hit; `Arc<[u8]>` swaps are a single atomic refcount bump.
    fn get_or_build_compute_commands<'c>(
        cache: &'c mut Option<CompiledCacheEntry>,
        ir: &GraphIR,
        fp: u64,
    ) -> &'c [GpuCommand] {
        let needs_build = match cache.as_ref() {
            Some(e) => e.fp != fp || e.commands.is_none(),
            None => true,
        };

        tracing::trace!(target: "goldy::schedule_cache", hit = !needs_build, fp, "compute_commands");

        if !needs_build {
            // Hit: refresh upload `Arc<[u8]>` payloads from the current IR.
            let entry = cache.as_mut().unwrap();
            if let Some(commands) = entry.commands.as_mut() {
                for &(cmd_idx, node_idx) in &entry.upload_remap {
                    let node = &ir.nodes[node_idx];
                    match (&mut commands[cmd_idx], &node.kind) {
                        (GpuCommand::WriteBuffer { data, .. }, NodeKind::WriteBuffer { data: src, .. }) => {
                            *data = src.clone()
                        }
                        (GpuCommand::WriteTexture { data, .. }, NodeKind::WriteTexture { data: src, .. }) => {
                            *data = src.clone()
                        }
                        (
                            GpuCommand::WriteTextureRegion { data, .. },
                            NodeKind::WriteTextureRegion { data: src, .. },
                        ) => *data = src.clone(),
                        _ => {} // mismatch should not occur if fp matches
                    }
                }
            }
            return cache.as_ref().unwrap().commands.as_deref().unwrap();
        }

        // Miss: rebuild schedule (if needed) and emit commands fresh.
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        let commands = analysis::emit_commands(ir, &schedule, None);
        let upload_remap = build_upload_remap(ir, &commands);
        *cache = Some(CompiledCacheEntry {
            fp,
            schedule,
            commands: Some(commands),
            upload_remap,
            partitioned_commands: None,
            partitioned_upload_remap: Vec::new(),
            partition_retention_keys: Vec::new(),
            partition_slot_keys: Vec::new(),
            partitioned_graph_commands: Vec::new(),
        });
        cache.as_ref().unwrap().commands.as_deref().unwrap()
    }

    /// Return cached partitioned commands for `ir`, building them if necessary.
    ///
    /// Partitions that contain only compute nodes are stored as `Vec<GpuCommand>`.
    /// Partitions that contain render-pass nodes are stored as `Vec<GraphCommand>` in
    /// `CompiledCacheEntry::partitioned_graph_commands` and represented by an empty
    /// `Vec` in the parallel `partitioned_commands` slot.
    ///
    /// On cache hit only upload `Arc<[u8]>` payloads in the compute slots are refreshed;
    /// render-pass commands are immutable (data arrives via bound parcels, not uploads).
    fn get_or_build_partitioned_commands(cache: &mut Option<CompiledCacheEntry>, ir: &GraphIR, fp: u64) {
        let _tz = crate::tracy_zone!("goldy.compile_partitioned");

        // Ensure schedule exists.
        let needs_schedule = match cache.as_ref() {
            Some(e) => e.fp != fp,
            None => true,
        };
        if needs_schedule {
            let edges = analysis::build_edges(ir);
            let schedule = analysis::schedule_waves(ir, &edges);
            *cache = Some(CompiledCacheEntry {
                fp,
                schedule,
                commands: None,
                upload_remap: Vec::new(),
                partitioned_commands: None,
                partitioned_upload_remap: Vec::new(),
                partition_retention_keys: Vec::new(),
                partition_slot_keys: Vec::new(),
                partitioned_graph_commands: Vec::new(),
            });
        }

        let needs_build = cache.as_ref().is_none_or(|e| e.partitioned_commands.is_none());

        tracing::trace!(target: "goldy::schedule_cache", hit = !needs_build, fp, "partitioned_commands");

        if !needs_build {
            // Hit: refresh upload payloads in compute-only partitions.
            let entry = cache.as_mut().unwrap();
            if let Some(parts) = entry.partitioned_commands.as_mut() {
                for &(part_idx, cmd_idx, node_idx) in &entry.partitioned_upload_remap {
                    let node = &ir.nodes[node_idx];
                    match (&mut parts[part_idx][cmd_idx], &node.kind) {
                        (GpuCommand::WriteBuffer { data, .. }, NodeKind::WriteBuffer { data: src, .. }) => {
                            *data = src.clone()
                        }
                        (GpuCommand::WriteTexture { data, .. }, NodeKind::WriteTexture { data: src, .. }) => {
                            *data = src.clone()
                        }
                        (
                            GpuCommand::WriteTextureRegion { data, .. },
                            NodeKind::WriteTextureRegion { data: src, .. },
                        ) => *data = src.clone(),
                        _ => {}
                    }
                }
            }
            return;
        }

        // Miss: emit each partition with the correct emitter.
        //
        // Present partitions are skipped — their commands are always re-emitted at
        // submit time with a concrete SlotResolver (the drawable handle isn't known
        // until the OS grants the swapchain image).  An empty Vec placeholder is
        // stored so the slot indices remain aligned with wave_ranges.
        let entry = cache.as_mut().unwrap();
        let wave_ranges = analysis::partition_wave_ranges(ir, &entry.schedule);

        let mut compute_partitions: Vec<Vec<GpuCommand>> = Vec::with_capacity(wave_ranges.len());
        let mut graph_partitions: Vec<Option<Vec<GraphCommand>>> = Vec::with_capacity(wave_ranges.len());

        for range in &wave_ranges {
            let waves = &entry.schedule.waves[range.clone()];
            let has_present = analysis::partition_waves_have_present(ir, waves);
            let has_render = waves.iter().any(|w| {
                w.node_indices
                    .iter()
                    .any(|&ni| matches!(ir.nodes[ni].kind, NodeKind::RenderPass { .. }))
            });

            if has_present {
                // Deferred: commands emitted fresh at submit time with a resolver.
                compute_partitions.push(Vec::new());
                graph_partitions.push(None);
            } else if has_render {
                compute_partitions.push(Vec::new());
                graph_partitions.push(Some(analysis::emit_graph_commands_for_waves(ir, waves, None)));
            } else {
                compute_partitions.push(analysis::emit_waves_to_commands(ir, waves, None));
                graph_partitions.push(None);
            }
        }

        let remap = build_partitioned_upload_remap(ir, &compute_partitions);
        entry.partitioned_commands = Some(compute_partitions);
        entry.partitioned_upload_remap = remap;
        entry.partitioned_graph_commands = graph_partitions;
    }

    /// Compile the graph into a flat command stream.
    ///
    /// Runs the dependency analyzer, schedules waves, inserts `ResourceBarrier`
    /// commands at wave boundaries, and emits the final [`GpuCommand`](crate::backend::GpuCommand) sequence.
    ///
    /// # Panics
    ///
    /// If the graph contains render-pass nodes or transient buffers, use
    /// [`Self::compile_graph_commands`] or [`Context::submit`](crate::Context::submit) instead.
    pub fn compile_commands(&mut self) -> Vec<crate::backend::GpuCommand> {
        assert!(
            self.transient_specs.is_empty(),
            "compile_commands: graph uses transient_buffer; use Device::submit"
        );
        assert!(
            self.transient_texture_specs.is_empty(),
            "compile_commands: graph uses transient_texture; use Device::submit"
        );
        if Self::has_render_passes_in_ir(&self.ir) {
            panic!("compile_commands: graph contains render_pass; use compile_graph_commands or Device::submit");
        }
        let fp = self.schedule_fp();
        Self::get_or_build_compute_commands(&mut self.schedule_cache, &self.ir, fp).to_vec()
    }

    /// Compile a pre-lowered [`GraphIR`] into a flat GPU command stream.
    ///
    /// Unlike [`Self::compile_commands`], this operates directly on a resolved IR that
    /// contains no transient specs — callers are responsible for lowering transients first.
    #[allow(dead_code)]
    pub(crate) fn compile_ir_to_gpu_commands(ir: &GraphIR) -> Vec<crate::backend::GpuCommand> {
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        analysis::emit_commands(ir, &schedule, None)
    }

    /// Return a clone of the cached [`CompiledSchedule`] (building it first if
    /// necessary) and the wave index where the swapchain-output split occurs.
    ///
    /// The split wave is the index of the first wave that contains a node
    /// binding [`ResourceId::SwapchainOutput`].  If no such wave exists, the
    /// split equals the total wave count (i.e., no early partition).
    ///
    /// Callers use the schedule to re-emit early and final wave slices from
    /// different IRs (pre-lowered vs. post-swapchain-lowered).
    pub(crate) fn schedule_and_split_wave(&mut self) -> (CompiledSchedule, usize) {
        let fp = self.schedule_fp();
        let schedule = Self::get_or_build_schedule(&mut self.schedule_cache, &self.ir, fp).clone();
        let split = schedule
            .waves
            .iter()
            .enumerate()
            .find(|(_, w)| {
                w.node_indices.iter().any(|&ni| {
                    self.ir.nodes[ni]
                        .bindings
                        .iter()
                        .any(|b| b.resource == ResourceId::SwapchainOutput)
                })
            })
            .map(|(idx, _)| idx)
            .unwrap_or(schedule.waves.len());
        (schedule, split)
    }

    /// Like [`Self::compile_commands`] but allows graphs that include render-pass nodes.
    pub fn compile_graph_commands(&mut self) -> Vec<GraphCommand> {
        assert!(
            self.transient_specs.is_empty(),
            "compile_graph_commands: graph uses transient_buffer; use Device::submit"
        );
        assert!(
            self.transient_texture_specs.is_empty(),
            "compile_graph_commands: graph uses transient_texture; use Device::submit"
        );
        let fp = self.schedule_fp();
        let schedule = Self::get_or_build_schedule(&mut self.schedule_cache, &self.ir, fp);
        analysis::emit_graph_commands(&self.ir, schedule, None)
    }
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for a single compute dispatch node within a [`TaskGraph`].
///
/// Created by [`TaskGraph::node`]. Must be finalized with
/// [`dispatch`](NodeBuilder::dispatch) or [`dispatch_indirect`](NodeBuilder::dispatch_indirect).
pub struct NodeBuilder<'a> {
    graph: &'a mut TaskGraph,
    label: &'static str,
    pipeline: crate::backend::ComputePipelineHandle,
    bindings: Vec<ResourceBinding>,
    resource_slots: Vec<u32>,
    user_slots: Vec<u32>,
}

impl<'a> NodeBuilder<'a> {
    /// Declare that this node accesses a buffer with the given logical access.
    pub fn bind_buffer(mut self, buf: &Buffer, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::Buffer(buf.handle),
            access,
        });
        self
    }

    /// Declare that this node accesses a buffer view with the given logical access.
    ///
    /// Records the view's exact byte range `[offset, offset+size)` within the
    /// parent buffer, so the scheduler can determine whether two views alias at
    /// byte-range granularity. Non-overlapping views of the same pool produce no
    /// dependency edge and can execute in the same wave.
    ///
    /// Barriers are still emitted against the parent buffer handle so backends
    /// require no changes.
    pub fn bind_buffer_view(mut self, view: &BufferView, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::BufferRange {
                parent: view.parent_handle(),
                offset: view.offset(),
                len: view.size(),
            },
            access,
        });
        self
    }

    /// Declare that this node accesses a texture with the given logical access.
    pub fn bind_texture(mut self, tex: &Texture, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::Texture(tex.handle),
            access,
        });
        self
    }

    /// Declare that this node accesses a retained [`crate::Parcel`] (buffer or texture).
    ///
    /// The backend resource handle is resolved inside the runtime; the client does not
    /// pass a raw handle.
    pub fn bind_parcel(mut self, parcel: &crate::Parcel, access: NodeAccess) -> Self {
        self.graph.stamp_targets.push(parcel.stamp_handle());
        self.bindings.push(ResourceBinding {
            resource: parcel.resource_id(),
            access,
        });
        self
    }

    pub fn bind_transient_buffer(mut self, id: TransientId, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::TransientBuffer(id),
            access,
        });
        self
    }

    pub fn bind_transient_texture(mut self, id: TransientTextureId, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::TransientTexture(id),
            access,
        });
        self
    }

    /// Declare that this node writes to the swapchain output.
    ///
    /// The concrete `TextureHandle` is resolved at submit time by
    /// [`Surface::submit_graph`](crate::Surface::submit_graph).  The caller (ekrano) must place
    /// [`super::SWAPCHAIN_SLOT_PLACEHOLDER`] in `resource_slots` at the corresponding
    /// binding position so `TaskGraph::lower_swapchain_output` can patch it with the
    /// real UAV bindless index after `surface.begin()`.
    pub fn bind_swapchain_output(mut self, _handle: SwapchainOutputHandle, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::SwapchainOutput,
            access,
        });
        self
    }

    /// Set the bindless resource slot indices for this node's dispatch (region A).
    /// Accepts an owned `Vec` to avoid re-allocation when the caller already has one.
    pub fn bind_resources_raw(mut self, indices: Vec<u32>) -> Self {
        self.resource_slots = indices;
        self
    }

    /// Convenience wrapper that copies a slice into owned storage.
    pub fn bind_resources_raw_slice(self, indices: &[u32]) -> Self {
        self.bind_resources_raw(indices.to_vec())
    }

    /// Bind buffer resource slots and declare read/write dependencies.
    ///
    /// Slot indices are each buffer's UAV bindless index in shader parameter order.
    pub fn bind_resources(mut self, buffers: &[&Buffer]) -> Self {
        use crate::types::ResourceAccess;
        let mut indices = Vec::with_capacity(buffers.len());
        for buf in buffers {
            self.bindings.push(ResourceBinding {
                resource: ResourceId::Buffer(buf.handle),
                access: NodeAccess::ReadWrite,
            });
            let idx = buf
                .resource_index(ResourceAccess::ReadWrite)
                .or_else(|| buf.resource_index(ResourceAccess::Read))
                .expect("bind_resources: buffer has no bindless index");
            indices.push(idx);
        }
        self.resource_slots = indices;
        self
    }

    /// Bind resource slots from typed [`ResourceHandle`]s (region A indices only).
    pub fn bind_resources_typed(mut self, handles: &[ResourceHandle]) -> Self {
        self.resource_slots = handles.iter().map(|h| h.index()).collect();
        self
    }

    /// Set user scalar parameters for this node's dispatch (region B).
    /// Accepts an owned `Vec` for indices to avoid re-allocation.
    pub fn bind_resources_raw_with_user(mut self, indices: Vec<u32>, user: &[u32]) -> Self {
        self.resource_slots = indices;
        self.user_slots = user.to_vec();
        self
    }

    /// Finalize the node with fixed workgroup dimensions.
    pub fn dispatch(self, x: u32, y: u32, z: u32) {
        let node = TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::Dispatch {
                pipeline: self.pipeline,
                resource_slots: self.resource_slots,
                user_slots: self.user_slots,
                dispatch: DispatchDim::Direct { x, y, z },
            },
        };
        self.graph.ir.nodes.push(node);
    }

    /// Finalize the node with indirect dispatch (dimensions read from `buf` at `offset`).
    pub fn dispatch_indirect(self, buf: &Buffer, offset: u64) {
        let node = TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::Dispatch {
                pipeline: self.pipeline,
                resource_slots: self.resource_slots,
                user_slots: self.user_slots,
                dispatch: DispatchDim::Indirect {
                    buffer: buf.handle,
                    offset,
                },
            },
        };
        self.graph.ir.nodes.push(node);
    }
}

/// One bindless push-constant slot in shader virtual-main parameter order.
///
/// Use with [`RenderPassBuilder::bind_shader_resources`]. Mosaic parcels belong in
/// [`RenderPassBuilder::bind_parcel_mut`] (graph dependency + vertex views), not here.
pub enum ShaderResourceSlot<'a> {
    Parcel {
        parcel: &'a crate::Parcel,
        access: NodeAccess,
    },
    Sampler(&'a Sampler),
}

fn node_access_to_resource_access(access: NodeAccess) -> ResourceAccess {
    match access {
        NodeAccess::Read => ResourceAccess::Read,
        NodeAccess::Write => ResourceAccess::Write,
        NodeAccess::ReadWrite => ResourceAccess::ReadWrite,
    }
}

/// Builder for a render pass targeting an offscreen [`crate::RenderTarget`].
pub struct RenderPassBuilder<'a> {
    graph: &'a mut TaskGraph,
    label: &'static str,
    target: RenderTargetHandle,
    bindings: Vec<ResourceBinding>,
    commands: Vec<RenderCommand>,
    push_constant_handles: Vec<ResourceHandle>,
}

impl<'a> RenderPassBuilder<'a> {
    /// Declare push-constant slots in shader parameter order and register graph bindings.
    ///
    /// [`Self::set_pipeline`] emits [`RenderCommand::BindResourcesTyped`] from these
    /// handles before each pipeline bind.
    pub fn bind_shader_resources(&mut self, slots: &[ShaderResourceSlot<'_>]) -> &mut Self {
        for slot in slots {
            match slot {
                ShaderResourceSlot::Parcel { parcel, access } => {
                    let resource_access = node_access_to_resource_access(*access);
                    self.graph.stamp_targets.push(parcel.stamp_handle());
                    self.bindings.push(ResourceBinding {
                        resource: parcel.resource_id(),
                        access: *access,
                    });
                    let handle = parcel.handle(resource_access).unwrap_or_else(|| {
                        panic!(
                            "ShaderResourceSlot::Parcel: mosaic parcels cannot be push-constant slots; \
                             use bind_parcel_mut for geometry and bind views at draw time"
                        )
                    });
                    self.push_constant_handles.push(handle);
                }
                ShaderResourceSlot::Sampler(sampler) => {
                    let handle = sampler
                        .handle(ResourceAccess::Read)
                        .expect("ShaderResourceSlot::Sampler: missing bindless sampler index");
                    self.push_constant_handles.push(handle);
                }
            }
        }
        self
    }

    /// Like [`Self::bind_parcel`] but for use while recording on `&mut self`.
    pub fn bind_parcel_mut(&mut self, parcel: &crate::Parcel, access: NodeAccess) -> &mut Self {
        self.graph.stamp_targets.push(parcel.stamp_handle());
        self.bindings.push(ResourceBinding {
            resource: parcel.resource_id(),
            access,
        });
        self
    }

    /// Like [`Self::bind_buffer`] but for use while recording on `&mut self`.
    pub fn bind_buffer_mut(&mut self, buf: &Buffer, access: NodeAccess) -> &mut Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::Buffer(buf.handle),
            access,
        });
        self
    }

    /// Like [`Self::bind_texture`] but for use while recording on `&mut self`.
    pub fn bind_texture_mut(&mut self, tex: &Texture, access: NodeAccess) -> &mut Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::Texture(tex.handle),
            access,
        });
        self
    }

    /// Like [`Self::bind_buffer_view`] but for use while recording on `&mut self`.
    pub fn bind_buffer_view_mut(&mut self, view: &BufferView, access: NodeAccess) -> &mut Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::BufferRange {
                parent: view.parent_handle(),
                offset: view.offset(),
                len: view.size(),
            },
            access,
        });
        self
    }

    pub fn clear(&mut self, color: Color) -> &mut Self {
        self.commands.push(RenderCommand::Clear(color));
        self
    }

    pub fn clear_depth(&mut self, depth: f32) -> &mut Self {
        self.commands.push(RenderCommand::ClearDepth(depth));
        self
    }

    /// Set the active pipeline and bind [`Self::bind_shader_resources`] slots when declared.
    ///
    /// Pipeline is bound before root constants (required on D3D12: root signature first).
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) -> &mut Self {
        self.commands.push(RenderCommand::SetPipeline(pipeline.handle));
        if !self.push_constant_handles.is_empty() {
            self.commands.push(RenderCommand::BindResourcesTyped {
                handles: self.push_constant_handles.clone(),
            });
        }
        self
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &impl BufferSource) -> &mut Self {
        self.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.source_handle(),
            offset: buffer.source_offset(),
        });
        self
    }

    pub fn set_index_buffer(&mut self, buffer: &impl BufferSource, format: IndexFormat) -> &mut Self {
        self.commands.push(RenderCommand::SetIndexBuffer {
            buffer: buffer.source_handle(),
            offset: buffer.source_offset(),
            format,
        });
        self
    }

    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) -> &mut Self {
        self.commands.push(RenderCommand::DrawIndexed {
            index_count: indices.end - indices.start,
            instance_count: instances.end - instances.start,
            first_index: indices.start,
            base_vertex,
            first_instance: instances.start,
        });
        self
    }

    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) -> &mut Self {
        self.commands.push(RenderCommand::Draw {
            vertex_count: vertices.end - vertices.start,
            instance_count: instances.end - instances.start,
            first_vertex: vertices.start,
            first_instance: instances.start,
        });
        self
    }

    pub fn draw_fullscreen(&mut self) -> &mut Self {
        self.draw(0..3, 0..1)
    }

    pub fn draw_quads(&mut self, count: u32) -> &mut Self {
        self.draw(0..6, 0..count)
    }

    pub fn bind_resources(&mut self, buffers: &[&Buffer]) -> &mut Self {
        self.commands.push(RenderCommand::BindResources {
            buffers: buffers.iter().map(|b| b.handle).collect(),
        });
        self
    }

    pub fn bind_resources_raw(&mut self, indices: &[u32]) -> &mut Self {
        self.commands.push(RenderCommand::BindResourcesRaw {
            indices: indices.to_vec(),
            user: Vec::new(),
            frame_table_base: 0,
        });
        self
    }

    pub fn bind_resources_typed(&mut self, handles: &[ResourceHandle]) -> &mut Self {
        self.commands.push(RenderCommand::BindResourcesTyped {
            handles: handles.to_vec(),
        });
        self
    }

    pub fn bind_buffer(mut self, buf: &Buffer, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::Buffer(buf.handle),
            access,
        });
        self
    }

    pub fn bind_buffer_view(mut self, view: &BufferView, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::BufferRange {
                parent: view.parent_handle(),
                offset: view.offset(),
                len: view.size(),
            },
            access,
        });
        self
    }

    pub fn bind_texture(mut self, tex: &Texture, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::Texture(tex.handle),
            access,
        });
        self
    }

    pub fn bind_transient_buffer(mut self, id: TransientId, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::TransientBuffer(id),
            access,
        });
        self
    }

    pub fn bind_transient_texture(mut self, id: TransientTextureId, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::TransientTexture(id),
            access,
        });
        self
    }

    /// Declare that this render pass accesses a retained [`crate::Parcel`].
    ///
    /// Like [`NodeBuilder::bind_parcel`], the parcel is stamped at graph submit
    /// time so [`crate::Parcel::last_referenced`] is updated automatically.
    pub fn bind_parcel(mut self, parcel: &crate::Parcel, access: NodeAccess) -> Self {
        self.graph.stamp_targets.push(parcel.stamp_handle());
        self.bindings.push(ResourceBinding {
            resource: parcel.resource_id(),
            access,
        });
        self
    }

    /// Finalize the node with commands recorded via [`Self::clear`], [`Self::set_pipeline`], etc.
    pub fn finish_recorded(self) {
        let RenderPassBuilder {
            graph,
            label,
            target,
            bindings,
            commands,
            push_constant_handles: _,
        } = self;
        graph.ir.nodes.push(TaskNode {
            label,
            bindings,
            kind: NodeKind::RenderPass { target, commands },
        });
    }

    /// Finalize the node with a pre-built [`RenderCommand`] list.
    pub fn finish(self, commands: Vec<RenderCommand>) {
        self.push_node(commands);
    }

    fn push_node(self, commands: Vec<RenderCommand>) {
        let RenderPassBuilder {
            graph,
            label,
            target,
            bindings,
            push_constant_handles: _,
            ..
        } = self;
        graph.ir.nodes.push(TaskNode {
            label,
            bindings,
            kind: NodeKind::RenderPass { target, commands },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::backend::GpuCommand;
    use crate::backend::GraphCommand;
    use crate::buffer::BufferPool;
    use crate::device::Device;
    use crate::render_target::RenderTarget;
    use crate::shader::ShaderModule;
    use crate::types::{Color, TextureFormat};
    use crate::Texture;

    fn count_logical_dispatches(cmds: &[GpuCommand]) -> usize {
        cmds.iter()
            .map(|c| match c {
                GpuCommand::Dispatch { .. } => 1,
                GpuCommand::DispatchBatch { count, .. } => *count as usize,
                _ => 0,
            })
            .sum()
    }

    fn mock_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn mock_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(device, "void main() {}").unwrap()
    }

    fn mock_pipeline(device: &Device, shader: &ShaderModule) -> crate::compute::ComputePipeline {
        crate::compute::ComputePipeline::new(device, shader).unwrap()
    }

    fn mock_render_pipeline(device: &Device, shader: &ShaderModule) -> RenderPipeline {
        RenderPipeline::new(
            device,
            shader,
            shader,
            &crate::RenderPipelineDesc {
                target_format: TextureFormat::Rgba8Unorm,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn compile_mixed_compute_render_inserts_barrier_and_submits_graph() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();
        let target = RenderTarget::new(&device, 8, 8, TextureFormat::Rgba8Unorm).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("compute_write", &pipeline)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[1])
            .dispatch(1, 1, 1);

        let mut pass = graph.render_pass("draw", &target);
        pass.bind_buffer_mut(&buf, NodeAccess::Read);
        pass.clear(Color::RED);
        pass.finish_recorded();

        let gcs = graph.compile_graph_commands();
        assert!(
            gcs.iter()
                .any(|c| matches!(c, GraphCommand::Compute(GpuCommand::ResourceBarrier { .. }))),
            "expected ResourceBarrier between compute write and render read"
        );

        graph.submit(&ctx).unwrap();
    }

    #[test]
    fn render_pass_finish_recorded_auto_binds_shader_resources() {
        use crate::backend::RenderCommand;

        let device = mock_device();
        let mut pool = crate::RetainedPool::new(Arc::new(device.clone()));
        let parcel = retained_buffer_parcel(&mut pool);
        let target = RenderTarget::new(&device, 8, 8, TextureFormat::Rgba8Unorm).unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_render_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        let mut pass = graph.render_pass("draw", &target);
        pass.bind_shader_resources(&[ShaderResourceSlot::Parcel {
            parcel: &parcel,
            access: NodeAccess::Read,
        }]);
        pass.clear(Color::GREEN);
        pass.set_pipeline(&pipeline);
        pass.finish_recorded();

        let gcs = graph.compile_graph_commands();
        let render_cmds = gcs
            .iter()
            .find_map(|c| match c {
                GraphCommand::Render { commands, .. } => Some(commands.as_slice()),
                _ => None,
            })
            .expect("render pass command");
        assert!(
            render_cmds
                .iter()
                .any(|c| matches!(c, RenderCommand::BindResourcesRaw { .. })),
            "set_pipeline should emit lowered BindResourcesRaw from bind_shader_resources"
        );
        let set_pipe = render_cmds
            .iter()
            .position(|c| matches!(c, RenderCommand::SetPipeline(_)))
            .expect("SetPipeline");
        let bind = render_cmds
            .iter()
            .position(|c| matches!(c, RenderCommand::BindResourcesRaw { .. }))
            .expect("BindResourcesRaw");
        assert!(set_pipe < bind, "D3D12 requires SetPipeline before BindResourcesRaw");
    }

    #[test]
    fn write_parcel_then_render_pass_inserts_barrier() {
        let device = mock_device();
        let mut pool = crate::RetainedPool::new(Arc::new(device.clone()));
        let parcel = retained_buffer_parcel(&mut pool);
        let target = RenderTarget::new(&device, 8, 8, TextureFormat::Rgba8Unorm).unwrap();

        let mut graph = TaskGraph::new();
        graph.write_parcel(&parcel, 0, vec![1, 2, 3, 4]).unwrap();

        let mut pass = graph.render_pass("draw", &target);
        pass.bind_parcel_mut(&parcel, NodeAccess::Read);
        pass.clear(Color::RED);
        pass.finish_recorded();

        let gcs = graph.compile_graph_commands();
        assert!(
            gcs.iter()
                .any(|c| matches!(c, GraphCommand::Compute(GpuCommand::ResourceBarrier { .. }))),
            "expected ResourceBarrier between write_parcel and render read"
        );
    }

    #[test]
    fn write_parcel_rejects_mosaic_parcel() {
        let device = Arc::new(mock_device());
        let mut pool = crate::RetainedPool::new(device.clone());
        let mut mosaic = pool.mosaic();
        let _slot = mosaic.emplace(&[0u32, 1, 2, 3]);
        let parcel = mosaic.build().unwrap();

        let mut graph = TaskGraph::new();
        let err = graph.write_parcel(&parcel, 0, vec![0; 4]).unwrap_err();
        assert!(
            err.to_string().contains("non-mosaic buffer parcels"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn render_pass_then_copy_render_target_emits_in_order() {
        let device = mock_device();
        let target = RenderTarget::new(&device, 8, 8, TextureFormat::Rgba8Unorm).unwrap();

        let mut graph = TaskGraph::new();
        let mut pass = graph.render_pass("draw", &target);
        pass.clear(Color::RED);
        pass.finish_recorded();
        let sc = graph.declare_swapchain_output();
        graph.copy_render_target_to_swapchain(&target, sc);

        let (schedule, split_wave) = graph.schedule_and_split_wave();
        assert_eq!(schedule.waves.len(), 2, "render and copy must be in separate waves");
        assert_eq!(split_wave, 1, "swapchain copy must be in the late partition");
        assert_eq!(schedule.waves[0].node_indices.len(), 1);
        assert_eq!(schedule.waves[1].node_indices.len(), 1);
    }

    #[test]
    fn render_pass_bind_parcel_submit_stamps_last_referenced() {
        let device = Arc::new(mock_device());
        let ctx = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device.clone());
        let parcel = retained_buffer_parcel(&mut pool);
        let target = RenderTarget::new(&device, 8, 8, TextureFormat::Rgba8Unorm).unwrap();

        let mut graph = TaskGraph::new();
        let mut pass = graph.render_pass("draw", &target);
        pass.bind_parcel_mut(&parcel, NodeAccess::Read);
        pass.clear(Color::BLUE);
        pass.finish_recorded();

        let tv = graph.submit(&ctx).unwrap();
        assert_eq!(parcel.last_referenced_on(ctx.backend_handle()), Some(tv));
    }

    #[test]
    fn transient_buffer_submit_succeeds_on_mock() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let mut graph = TaskGraph::new();
        let t = graph.transient_buffer(256);
        graph
            .node("touch", &pipeline)
            .bind_transient_buffer(t, NodeAccess::Write)
            .bind_resources_raw_slice(&[0])
            .dispatch(1, 1, 1);
        graph.submit(&ctx).unwrap();
    }

    #[test]
    fn transient_texture_submit_succeeds_on_mock() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let mut graph = TaskGraph::new();
        let tt = graph.transient_texture(4, 4, TextureFormat::Rgba8Unorm);
        graph
            .node("touch_tex", &pipeline)
            .bind_transient_texture(tt, NodeAccess::Write)
            .bind_resources_raw_slice(&[0])
            .dispatch(1, 1, 1);
        graph.submit(&ctx).unwrap();
    }

    #[test]
    fn transient_texture_heap_aliases_non_overlapping_waves() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 4, crate::BufferKind::Scattered).unwrap();
        let mut graph = TaskGraph::new();
        let t0 = graph.transient_texture(2, 2, TextureFormat::Rgba8Unorm);
        let t1 = graph.transient_texture(2, 2, TextureFormat::Rgba8Unorm);
        graph
            .node("w0", &pipeline)
            .bind_transient_texture(t0, NodeAccess::Write)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[0])
            .dispatch(1, 1, 1);
        graph
            .node("w1", &pipeline)
            .bind_buffer(&buf, NodeAccess::Read)
            .bind_transient_texture(t1, NodeAccess::Write)
            .bind_resources_raw_slice(&[0])
            .dispatch(1, 1, 1);
        graph.submit(&ctx).unwrap();
    }

    #[test]
    fn transient_heap_aliases_non_overlapping_waves() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 4, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        let t0 = graph.transient_buffer(256);
        let t1 = graph.transient_buffer(256);
        graph
            .node("wave0", &pipeline)
            .bind_transient_buffer(t0, NodeAccess::Write)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[0])
            .dispatch(1, 1, 1);
        graph
            .node("wave1", &pipeline)
            .bind_buffer(&buf, NodeAccess::Read)
            .bind_transient_buffer(t1, NodeAccess::Write)
            .bind_resources_raw_slice(&[0])
            .dispatch(1, 1, 1);

        let (schedule, _) = graph.schedule_and_split_wave();
        let node_waves = crate::task_graph::analysis::node_to_wave_map(&schedule, graph.ir().nodes.len());
        let (total, _, layout) = graph.transient_heap_size_and_layout(&node_waves).unwrap();
        assert_eq!(total, 256, "sequential transients should pack into one 256-byte slot");
        assert_eq!(layout[&t0.0], layout[&t1.0]);
        graph.submit(&ctx).unwrap();
    }

    #[test]
    fn transient_heap_separates_concurrent_waves() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        let t0 = graph.transient_buffer(256);
        let t1 = graph.transient_buffer(256);
        graph
            .node("a", &pipeline)
            .bind_transient_buffer(t0, NodeAccess::Write)
            .bind_resources_raw_slice(&[0])
            .dispatch(1, 1, 1);
        graph
            .node("b", &pipeline)
            .bind_transient_buffer(t1, NodeAccess::Write)
            .bind_resources_raw_slice(&[0])
            .dispatch(1, 1, 1);

        let (schedule, _) = graph.schedule_and_split_wave();
        let node_waves = crate::task_graph::analysis::node_to_wave_map(&schedule, graph.ir().nodes.len());
        let (total, _, layout) = graph.transient_heap_size_and_layout(&node_waves).unwrap();
        assert!(
            total >= 512,
            "concurrent transients need disjoint heap regions, got {}",
            total
        );
        assert_ne!(layout[&t0.0], layout[&t1.0]);
        graph.submit(&ctx).unwrap();
    }

    #[test]
    fn compile_linear_chain() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf_a = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("write", &pipeline)
            .bind_buffer(&buf_a, NodeAccess::Write)
            .bind_resources_raw_slice(&[42])
            .dispatch(8, 1, 1);
        graph
            .node("read_write", &pipeline)
            .bind_buffer(&buf_a, NodeAccess::Read)
            .bind_buffer(&buf_b, NodeAccess::Write)
            .bind_resources_raw_slice(&[43])
            .dispatch(4, 1, 1);

        let cmds = graph.compile_commands();

        // FrameTableStaging + wave 0: SetPipeline, BindResourcesRaw, Dispatch
        // ResourceBarrier
        // Wave 1: SetPipeline, BindResourcesRaw, Dispatch
        assert_eq!(cmds.len(), 8);
        assert!(matches!(cmds[0], GpuCommand::FrameTableStaging { .. }));
        assert!(matches!(cmds[1], GpuCommand::SetPipeline(_)));
        assert!(matches!(cmds[2], GpuCommand::BindResourcesRaw { .. }));
        assert!(matches!(cmds[3], GpuCommand::Dispatch { workgroups_x: 8, .. }));
        assert!(matches!(cmds[4], GpuCommand::ResourceBarrier { .. }));
        assert!(matches!(cmds[5], GpuCommand::SetPipeline(_)));
        assert!(matches!(cmds[6], GpuCommand::BindResourcesRaw { .. }));
        assert!(matches!(cmds[7], GpuCommand::Dispatch { workgroups_x: 4, .. }));
    }

    #[test]
    fn compile_independent_no_barrier() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf_a = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("write_a", &pipeline)
            .bind_buffer(&buf_a, NodeAccess::Write)
            .dispatch(8, 1, 1);
        graph
            .node("write_b", &pipeline)
            .bind_buffer(&buf_b, NodeAccess::Write)
            .dispatch(4, 1, 1);

        let cmds = graph.compile_commands();

        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
        assert_eq!(count_logical_dispatches(&cmds), 2);
    }

    #[test]
    fn compile_empty_graph() {
        let mut graph = TaskGraph::new();
        let cmds = graph.compile_commands();
        assert!(cmds.is_empty());
    }

    #[test]
    fn submit_via_mock() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_buffer(&buf, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);

        let tv = graph.submit(&ctx).unwrap();
        assert!(ctx.gpu_progress() >= tv);
        ctx.wait_until(tv).unwrap();
    }

    #[test]
    fn compile_diamond() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let p1 = mock_pipeline(&device, &shader);
        let p2 = mock_pipeline(&device, &shader);
        let p3 = mock_pipeline(&device, &shader);
        let p4 = mock_pipeline(&device, &shader);

        let buf_x = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();
        let buf_y = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();
        let buf_z = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        // A writes X
        graph
            .node("A", &p1)
            .bind_buffer(&buf_x, NodeAccess::Write)
            .dispatch(1, 1, 1);
        // B reads X, writes Y
        graph
            .node("B", &p2)
            .bind_buffer(&buf_x, NodeAccess::Read)
            .bind_buffer(&buf_y, NodeAccess::Write)
            .dispatch(1, 1, 1);
        // C reads X, writes Z
        graph
            .node("C", &p3)
            .bind_buffer(&buf_x, NodeAccess::Read)
            .bind_buffer(&buf_z, NodeAccess::Write)
            .dispatch(1, 1, 1);
        // D reads Y and Z
        graph
            .node("D", &p4)
            .bind_buffer(&buf_y, NodeAccess::Read)
            .bind_buffer(&buf_z, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        // 3 waves: [A], [B,C], [D]
        // Wave 0: pipeline+dispatch for A
        // ResourceBarrier (buf_x)
        // Wave 1: pipeline+dispatch for B, pipeline+dispatch for C
        // ResourceBarrier (buf_y, buf_z)
        // Wave 2: pipeline+dispatch for D

        let barrier_count = cmds
            .iter()
            .filter(|c| matches!(c, GpuCommand::ResourceBarrier { .. }))
            .count();
        assert_eq!(barrier_count, 2);

        let dispatch_count = cmds.iter().filter(|c| matches!(c, GpuCommand::Dispatch { .. })).count();
        assert_eq!(dispatch_count, 4);
    }

    #[test]
    fn len_and_is_empty() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);

        graph
            .node("A", &pipeline)
            .bind_buffer(&buf, NodeAccess::Read)
            .dispatch(1, 1, 1);
        assert!(!graph.is_empty());
        assert_eq!(graph.len(), 1);
    }

    // -------------------------------------------------------------------------
    // clear_buffer and write_buffer node tests
    // -------------------------------------------------------------------------

    #[test]
    fn clear_buffer_then_read_produces_barrier() {
        // clear_buffer declares Write on buf; a read dispatch creates a RAW edge.
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.clear_buffer(&buf, 0, 256);
        graph
            .node("read", &pipeline)
            .bind_buffer(&buf, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        // ClearBuffer, ResourceBarrier, SetPipeline, Dispatch
        // No staging: dispatch has no bindless resource_slots (barrier-only bindings).
        assert_eq!(cmds.len(), 4);
        assert!(matches!(cmds[0], GpuCommand::ClearBuffer { .. }));
        assert!(matches!(cmds[1], GpuCommand::ResourceBarrier { .. }));
        assert!(matches!(cmds[2], GpuCommand::SetPipeline(_)));
        assert!(matches!(cmds[3], GpuCommand::Dispatch { .. }));
    }

    #[test]
    fn clear_buffer_independent_of_unrelated_dispatch_same_wave() {
        // Clear of buf_a and a dispatch writing buf_b are independent.
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf_a = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.clear_buffer(&buf_a, 0, 256);
        graph
            .node("write_b", &pipeline)
            .bind_buffer(&buf_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
        assert!(cmds.iter().any(|c| matches!(c, GpuCommand::ClearBuffer { .. })));
        assert!(cmds.iter().any(|c| matches!(c, GpuCommand::Dispatch { .. })));
    }

    #[test]
    fn write_buffer_then_read_produces_barrier() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.write_buffer(&buf, 0, vec![0u8; 256]);
        graph
            .node("read", &pipeline)
            .bind_buffer(&buf, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        // No staging (dispatch uses bind_buffer for barrier-tracking only, no bindless slots).
        assert!(matches!(cmds[0], GpuCommand::WriteBuffer { .. }));
        assert!(
            cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
            "expected a barrier between write_buffer and dispatch"
        );
    }

    #[test]
    fn write_buffer_independent_of_unrelated_dispatch_same_wave() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf_a = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.write_buffer(&buf_a, 0, vec![0u8; 4]);
        graph
            .node("write_b", &pipeline)
            .bind_buffer(&buf_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
    }

    #[test]
    fn write_texture_then_dispatch_produces_barrier() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let tex = Texture::new(
            &device,
            4,
            4,
            crate::types::TextureFormat::Rgba8Unorm,
            crate::types::TextureKind::Interpolated,
            crate::types::TextureFlags::COPY_DST,
        )
        .unwrap();

        let mut graph = TaskGraph::new();
        graph.write_texture(&tex, vec![0u8; 4 * 4 * 4]).unwrap();
        graph
            .node("read_tex", &pipeline)
            .bind_texture(&tex, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();
        // No staging (dispatch uses bind_texture for barrier-tracking only, no bindless slots).
        assert!(matches!(cmds[0], GpuCommand::WriteTexture { .. }));
        assert!(
            cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
            "expected a barrier between write_texture and dispatch"
        );
    }

    #[test]
    fn write_texture_independent_of_unrelated_dispatch_same_wave() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let tex = Texture::new(
            &device,
            2,
            2,
            crate::types::TextureFormat::Rgba8Unorm,
            crate::types::TextureKind::Interpolated,
            crate::types::TextureFlags::COPY_DST,
        )
        .unwrap();
        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.write_texture(&tex, vec![0u8; 16]).unwrap();
        graph
            .node("writes_buf", &pipeline)
            .bind_buffer(&buf, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();
        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
    }

    #[test]
    fn multiple_clears_independent_same_wave() {
        let device = mock_device();
        let buf_a = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.clear_buffer(&buf_a, 0, 256);
        graph.clear_buffer(&buf_b, 0, 256);

        let cmds = graph.compile_commands();

        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, GpuCommand::ClearBuffer { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn is_empty_with_clear_node() {
        let device = mock_device();
        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        assert!(graph.is_empty());
        graph.clear_buffer(&buf, 0, 256);
        assert!(!graph.is_empty());
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn is_empty_with_write_node() {
        let device = mock_device();
        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        assert!(graph.is_empty());
        graph.write_buffer(&buf, 0, vec![1, 2, 3, 4]);
        assert!(!graph.is_empty());
        assert_eq!(graph.len(), 1);
    }

    // -------------------------------------------------------------------------
    // Category F: TaskGraph + MockBackend with real BufferPool / BufferView
    // -------------------------------------------------------------------------

    /// Create a pool of `total_size` bytes and return it together with
    /// a device, shader, and pipeline for convenience.
    fn make_pool_setup(total_size: u64) -> (Device, ShaderModule, crate::compute::ComputePipeline, BufferPool) {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let pool = BufferPool::new(&device, total_size).unwrap();
        (device, shader, pipeline, pool)
    }

    #[test]
    fn pool_two_disjoint_views_independent_writes_one_wave() {
        // Two nodes each writing a distinct pool region — should land in one wave
        let (device, shader, _, mut pool) = make_pool_setup(1024);
        let pipeline = mock_pipeline(&device, &shader);

        let view_a = pool.alloc::<u32>(64).unwrap(); // [0, 256)
        let view_b = pool.alloc::<u32>(64).unwrap(); // [256, 512)

        let mut graph = TaskGraph::new();
        graph
            .node("write_a", &pipeline)
            .bind_buffer_view(&view_a, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("write_b", &pipeline)
            .bind_buffer_view(&view_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        // No barrier — independent regions
        assert!(!cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
        assert_eq!(count_logical_dispatches(&cmds), 2);
    }

    #[test]
    fn pool_raw_dependency_two_waves_one_barrier() {
        // A writes view, B reads the same view — two waves, one barrier
        let (device, shader, _, mut pool) = make_pool_setup(512);
        let pipeline = mock_pipeline(&device, &shader);

        let view = pool.alloc::<u32>(64).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("write", &pipeline)
            .bind_buffer_view(&view, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("read", &pipeline)
            .bind_buffer_view(&view, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, GpuCommand::ResourceBarrier { .. }))
                .count(),
            1
        );
        assert_eq!(count_logical_dispatches(&cmds), 2);
    }

    #[test]
    fn mix_owned_buffer_and_pooled_view_correct_edge_detection() {
        // A writes an owned buffer; B writes a pool view; C reads both —
        // C depends on both A and B, but A and B are independent.
        let (device, shader, _, mut pool) = make_pool_setup(512);
        let p1 = mock_pipeline(&device, &shader);
        let p2 = mock_pipeline(&device, &shader);
        let p3 = mock_pipeline(&device, &shader);

        let owned = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();
        let view = pool.alloc::<u32>(64).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("A", &p1)
            .bind_buffer(&owned, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("B", &p2)
            .bind_buffer_view(&view, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("C", &p3)
            .bind_buffer(&owned, NodeAccess::Read)
            .bind_buffer_view(&view, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        // Wave 0: A+B (independent); Wave 1: C — one barrier between them
        let barrier_count = cmds
            .iter()
            .filter(|c| matches!(c, GpuCommand::ResourceBarrier { .. }))
            .count();
        assert_eq!(barrier_count, 1);

        let dispatch_count = cmds.iter().filter(|c| matches!(c, GpuCommand::Dispatch { .. })).count();
        assert_eq!(dispatch_count, 3);
    }

    #[test]
    fn pool_diamond_four_views() {
        // A writes v0; B reads v0 + writes v1; C reads v0 + writes v2;
        // D reads v1 + v2 — diamond with pooled views
        let (device, shader, _, mut pool) = make_pool_setup(2048);
        let p1 = mock_pipeline(&device, &shader);
        let p2 = mock_pipeline(&device, &shader);
        let p3 = mock_pipeline(&device, &shader);
        let p4 = mock_pipeline(&device, &shader);

        let v0 = pool.alloc::<u32>(64).unwrap();
        let v1 = pool.alloc::<u32>(64).unwrap();
        let v2 = pool.alloc::<u32>(64).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("A", &p1)
            .bind_buffer_view(&v0, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("B", &p2)
            .bind_buffer_view(&v0, NodeAccess::Read)
            .bind_buffer_view(&v1, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("C", &p3)
            .bind_buffer_view(&v0, NodeAccess::Read)
            .bind_buffer_view(&v2, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("D", &p4)
            .bind_buffer_view(&v1, NodeAccess::Read)
            .bind_buffer_view(&v2, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        // 3 waves: [A], [B+C], [D] — 2 barriers
        let barrier_count = cmds
            .iter()
            .filter(|c| matches!(c, GpuCommand::ResourceBarrier { .. }))
            .count();
        assert_eq!(barrier_count, 2);

        let dispatch_count = cmds.iter().filter(|c| matches!(c, GpuCommand::Dispatch { .. })).count();
        assert_eq!(dispatch_count, 4);
    }

    #[test]
    fn bind_buffer_on_pool_backing_aliases_all_views() {
        // A writes the whole backing buffer; B reads a view — must have an edge
        let (device, shader, _, mut pool) = make_pool_setup(512);
        let p1 = mock_pipeline(&device, &shader);
        let p2 = mock_pipeline(&device, &shader);

        let view = pool.alloc::<u32>(64).unwrap();
        let backing = pool.backing_buffer();

        let mut graph = TaskGraph::new();
        graph
            .node("A", &p1)
            .bind_buffer(backing, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("B", &p2)
            .bind_buffer_view(&view, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, GpuCommand::ResourceBarrier { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn pool_10_independent_views_one_wave() {
        // 10 nodes, each writing a distinct 64-byte region — all independent, 1 wave
        let (device, shader, _, mut pool) = make_pool_setup(10 * 256 + 256);
        let pipeline = mock_pipeline(&device, &shader);

        let views: Vec<_> = (0..10).map(|_| pool.alloc::<u32>(64).unwrap()).collect();

        let mut graph = TaskGraph::new();
        for view in &views {
            graph
                .node("write", &pipeline)
                .bind_buffer_view(view, NodeAccess::Write)
                .dispatch(1, 1, 1);
        }

        let cmds = graph.compile_commands();

        assert!(
            !cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
            "10 independent pool views should produce zero barriers"
        );
        assert_eq!(count_logical_dispatches(&cmds), 10);
    }

    #[test]
    fn pool_write_chain_barrier_uses_parent_handle() {
        // A writes view; B reads view. Barrier must name the parent buffer handle.
        let (device, shader, _, mut pool) = make_pool_setup(512);
        let p1 = mock_pipeline(&device, &shader);
        let p2 = mock_pipeline(&device, &shader);

        let view = pool.alloc::<u32>(64).unwrap();
        let parent_handle = view.parent_handle();

        let mut graph = TaskGraph::new();
        graph
            .node("write", &p1)
            .bind_buffer_view(&view, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("read", &p2)
            .bind_buffer_view(&view, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        let barrier = cmds
            .iter()
            .find(|c| matches!(c, GpuCommand::ResourceBarrier { .. }))
            .expect("expected a barrier");

        if let GpuCommand::ResourceBarrier { buffers, .. } = barrier {
            assert!(
                buffers.iter().any(|(h, _)| *h == parent_handle),
                "barrier should reference the parent buffer handle {}, got {:?}",
                parent_handle,
                buffers
            );
        }
    }

    #[test]
    fn clear_then_pool_view_dispatch_disjoint_no_spurious_barrier() {
        // A ClearBuffer node on the backing buffer followed by pool-view dispatches.
        // Two independent view writes should still be in one wave (after the clear wave).
        let (device, shader, _, mut pool) = make_pool_setup(1024);
        let pipeline = mock_pipeline(&device, &shader);

        let view_a = pool.alloc::<u32>(64).unwrap();
        let view_b = pool.alloc::<u32>(64).unwrap();

        let mut graph = TaskGraph::new();
        // Clear the whole backing buffer — writes to the whole Buffer handle
        graph.clear_buffer(pool.backing_buffer(), 0, 1024);
        // The two pool-view dispatches write disjoint *ranges* of the backing buffer.
        // But clear_buffer uses ResourceId::Buffer(parent) (whole-buffer), so it
        // aliases every range. The two view dispatches therefore depend on the clear
        // (RAW: clear writes whole, view writes range of same parent), forming:
        //   Wave 0: clear
        //   Wave 1: write_a, write_b (independent of each other)
        graph
            .node("write_a", &pipeline)
            .bind_buffer_view(&view_a, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("write_b", &pipeline)
            .bind_buffer_view(&view_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        // Exactly one barrier (clear → view dispatches), no barrier between the two views
        let barrier_count = cmds
            .iter()
            .filter(|c| matches!(c, GpuCommand::ResourceBarrier { .. }))
            .count();
        assert_eq!(barrier_count, 1, "expected exactly one barrier (clear → views)");

        // ClearBuffer first, then dispatches with one barrier.
        // No staging (dispatches use bind_buffer_view for barrier-tracking only).
        assert!(matches!(cmds[0], GpuCommand::ClearBuffer { .. }));

        // Two dispatches present
        assert_eq!(count_logical_dispatches(&cmds), 2);
    }

    #[test]
    fn clear_view_then_dispatch_on_same_view_produces_barrier() {
        // clear_buffer_view on view_a, then dispatch reads view_a — must barrier
        let (device, shader, _, mut pool) = make_pool_setup(512);
        let pipeline = mock_pipeline(&device, &shader);

        let view_a = pool.alloc::<u32>(64).unwrap();

        let mut graph = TaskGraph::new();
        graph.clear_buffer_view(&view_a, 0, 0); // size=0 → clear to end of view
        graph
            .node("read_a", &pipeline)
            .bind_buffer_view(&view_a, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, GpuCommand::ResourceBarrier { .. }))
                .count(),
            1
        );
        // No staging (dispatch uses bind_buffer_view for barrier-tracking only).
        assert!(matches!(cmds[0], GpuCommand::ClearBuffer { .. }));
    }

    #[test]
    fn clear_view_and_dispatch_on_disjoint_view_same_wave_no_barrier() {
        // clear_buffer_view on view_a, dispatch writes view_b (disjoint) — no barrier
        let (device, shader, _, mut pool) = make_pool_setup(1024);
        let pipeline = mock_pipeline(&device, &shader);

        let view_a = pool.alloc::<u32>(64).unwrap(); // [0, 256)
        let view_b = pool.alloc::<u32>(64).unwrap(); // [256, 512)

        let mut graph = TaskGraph::new();
        graph.clear_buffer_view(&view_a, 0, 0);
        graph
            .node("write_b", &pipeline)
            .bind_buffer_view(&view_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        assert!(
            !cmds.iter().any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
            "disjoint views should produce no barrier"
        );
    }

    // ---- retention fingerprint tests --------------------------------------------------

    /// A graph rebuilt with identical topology, pipeline, and dispatch dimensions must
    /// produce the same retention fingerprint, allowing the retained CB to be resubmitted.
    #[test]
    fn retention_fingerprint_stable_on_identical_graph() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut g1 = TaskGraph::new();
        g1.node("dispatch", &pipeline)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[1])
            .dispatch(4, 4, 1);

        let mut g2 = TaskGraph::new();
        g2.node("dispatch", &pipeline)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[1])
            .dispatch(4, 4, 1);

        assert_eq!(
            g1.compute_retention_fingerprint(),
            g2.compute_retention_fingerprint(),
            "identical graphs must share retention fingerprint"
        );
    }

    /// Changing dispatch dimensions must change the retention fingerprint but NOT the
    /// binding fingerprint, because wave scheduling depends only on resource access.
    #[test]
    fn retention_fingerprint_changes_on_dispatch_dim_change() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut g1 = TaskGraph::new();
        g1.node("dispatch", &pipeline)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[1])
            .dispatch(4, 4, 1);

        let mut g2 = TaskGraph::new();
        g2.node("dispatch", &pipeline)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[1])
            .dispatch(8, 8, 1); // different dims

        assert_ne!(
            g1.compute_retention_fingerprint(),
            g2.compute_retention_fingerprint(),
            "dispatch dim change must invalidate retention fingerprint"
        );
        // Schedule (binding) fingerprint should be stable across dispatch dim changes.
        assert_eq!(
            g1.compute_binding_fingerprint(),
            g2.compute_binding_fingerprint(),
            "binding fingerprint must be unaffected by dispatch dim change"
        );
    }

    /// Changing the pipeline on an otherwise identical binding pattern must invalidate
    /// the retention fingerprint.
    #[test]
    fn retention_fingerprint_changes_on_pipeline_change() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let p1 = mock_pipeline(&device, &shader);
        let p2 = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut g1 = TaskGraph::new();
        g1.node("dispatch", &p1)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[1])
            .dispatch(4, 4, 1);

        let mut g2 = TaskGraph::new();
        g2.node("dispatch", &p2) // different pipeline
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[1])
            .dispatch(4, 4, 1);

        let fp1 = g1.compute_retention_fingerprint();
        let fp2 = g2.compute_retention_fingerprint();

        // Pipelines created from the same shader by MockBackend get distinct handles only
        // when the backend increments its counter — check conditionally so the test is
        // valid regardless of mock handle allocation.
        if p1.handle != p2.handle {
            assert_ne!(
                fp1, fp2,
                "different pipeline handles must produce different retention fingerprints"
            );
        }

        // Binding fingerprint must still be stable (same resources/accesses).
        assert_eq!(
            g1.compute_binding_fingerprint(),
            g2.compute_binding_fingerprint(),
            "pipeline change must not affect binding fingerprint"
        );
    }

    /// Changing resource_slots (push-constant bindless indices) must invalidate
    /// the retention fingerprint.
    #[test]
    fn retention_fingerprint_changes_on_resource_slots_change() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::BufferKind::Scattered).unwrap();

        let mut g1 = TaskGraph::new();
        g1.node("dispatch", &pipeline)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[1])
            .dispatch(4, 4, 1);

        let mut g2 = TaskGraph::new();
        g2.node("dispatch", &pipeline)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[2]) // different bindless slot
            .dispatch(4, 4, 1);

        assert_ne!(
            g1.compute_retention_fingerprint(),
            g2.compute_retention_fingerprint(),
            "resource slot change must invalidate retention fingerprint"
        );
        assert_eq!(
            g1.compute_binding_fingerprint(),
            g2.compute_binding_fingerprint(),
            "resource slot change must not affect binding fingerprint"
        );
    }

    // -----------------------------------------------------------------------
    // Parcel reference stamping at submit
    // -----------------------------------------------------------------------

    fn retained_buffer_parcel(pool: &mut crate::RetainedPool) -> crate::Parcel {
        pool.acquire_buffer(
            256,
            crate::BufferKind::Scattered,
            None,
            crate::types::BufferFlags::empty(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn bind_parcel_submit_stamps_last_referenced() {
        let device = Arc::new(mock_device());
        let ctx = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device.clone());
        let parcel = retained_buffer_parcel(&mut pool);
        assert!(parcel.last_referenced().is_empty());

        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_parcel(&parcel, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);

        let tv = graph.submit(&ctx).unwrap();
        assert_eq!(parcel.last_referenced_on(ctx.backend_handle()), Some(tv));
    }

    #[test]
    fn bind_parcel_monotonic_max_across_submits() {
        let device = Arc::new(mock_device());
        let ctx = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device.clone());
        let parcel = retained_buffer_parcel(&mut pool);
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_parcel(&parcel, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let tv1 = graph.submit(&ctx).unwrap();

        // TODO(retained-graph): clear()+rebuild identical nodes — anti-pattern; see `TaskGraph::clear`.
        graph.clear();
        graph
            .node("work", &pipeline)
            .bind_parcel(&parcel, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let tv2 = graph.submit(&ctx).unwrap();
        assert!(tv2 > tv1);
        assert_eq!(parcel.last_referenced_on(ctx.backend_handle()), Some(tv2));

        // Lower epoch must not regress the stamp.
        graph.apply_reference_stamps(ctx.backend_handle(), &device.inner, tv1);
        assert_eq!(parcel.last_referenced_on(ctx.backend_handle()), Some(tv2));
    }

    #[test]
    fn bind_parcel_multiple_all_stamped() {
        let device = Arc::new(mock_device());
        let ctx = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device.clone());
        let p1 = retained_buffer_parcel(&mut pool);
        let p2 = retained_buffer_parcel(&mut pool);
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_parcel(&p1, NodeAccess::Write)
            .bind_parcel(&p2, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let tv = graph.submit(&ctx).unwrap();
        assert_eq!(p1.last_referenced_on(ctx.backend_handle()), Some(tv));
        assert_eq!(p2.last_referenced_on(ctx.backend_handle()), Some(tv));
    }

    #[test]
    fn unreferenced_parcel_stays_none_after_submit() {
        let device = Arc::new(mock_device());
        let ctx = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device.clone());
        let bound = retained_buffer_parcel(&mut pool);
        let unbound = retained_buffer_parcel(&mut pool);
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_parcel(&bound, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let tv = graph.submit(&ctx).unwrap();
        assert_eq!(bound.last_referenced_on(ctx.backend_handle()), Some(tv));
        assert!(unbound.last_referenced().is_empty());
    }

    #[test]
    fn transfer_out_ready_after_after_submit() {
        let device = Arc::new(mock_device());
        let ctx = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device.clone());
        let parcel = retained_buffer_parcel(&mut pool);
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_parcel(&parcel, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let tv = graph.submit(&ctx).unwrap();

        let stamped = pool.transfer_out(&ctx, parcel);
        assert_eq!(stamped.ready_after.get(&ctx.backend_handle()), Some(&tv));
    }

    #[test]
    fn transfer_out_unreferenced_has_none_ready_after() {
        let device = Arc::new(mock_device());
        let ctx = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device);
        let parcel = retained_buffer_parcel(&mut pool);
        let stamped = pool.transfer_out(&ctx, parcel);
        assert!(stamped.ready_after.is_empty());
    }

    #[test]
    fn clear_empties_stamp_targets() {
        let device = Arc::new(mock_device());
        let ctx = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device.clone());
        let parcel = retained_buffer_parcel(&mut pool);
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_parcel(&parcel, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        graph.clear();

        let tv = graph.submit(&ctx).unwrap();
        assert!(parcel.last_referenced().is_empty());
        let _ = tv;
    }

    #[test]
    fn submit_pipelined_stamps_bound_parcel() {
        let device = Arc::new(mock_device());
        let ctx = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device.clone());
        let parcel = retained_buffer_parcel(&mut pool);
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_parcel(&parcel, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);

        let tv = ctx.submit_pipelined(&mut graph).unwrap();
        assert_eq!(parcel.last_referenced_on(ctx.backend_handle()), Some(tv));
    }

    #[test]
    fn submit_pipelined_and_retain_stamps_bound_parcel() {
        let device = Arc::new(mock_device());
        let ctx = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device.clone());
        let parcel = retained_buffer_parcel(&mut pool);
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_parcel(&parcel, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);

        let tv = ctx.submit_pipelined_and_retain(&mut graph).unwrap();
        assert_eq!(parcel.last_referenced_on(ctx.backend_handle()), Some(tv));
    }

    #[test]
    fn bind_parcel_stamp_is_context_specific() {
        let device = Arc::new(mock_device());
        let ctx_a = device.create_context().unwrap();
        let ctx_b = device.create_context().unwrap();
        let mut pool = crate::RetainedPool::new(device.clone());
        let parcel = retained_buffer_parcel(&mut pool);
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_parcel(&parcel, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let tv_a = graph.submit(&ctx_a).unwrap();

        assert_eq!(parcel.last_referenced_on(ctx_a.backend_handle()), Some(tv_a));
        assert_eq!(parcel.last_referenced_on(ctx_b.backend_handle()), None);
        assert_eq!(ctx_b.gpu_progress(), 0, "context B must not observe A's submit");
        assert!(
            ctx_b.parcel_ready(&parcel.last_referenced()),
            "readiness checks the stamping context's progress, not the caller's"
        );
    }
}

// ---------------------------------------------------------------------------
// Slice-aware retention tests
//
// These test the per-partition retain/resubmit logic in `submit_resolved_ir_and_retain`.
// All tests run against the mock backend.  Submission calls use `IrSubmitState` (which
// acquires the backend lock internally) and stats are read via `Device::with_mock` after
// each call (a separate lock acquisition) to avoid holding the lock across a submit.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod slice_retention_tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::buffer::Buffer;
    use crate::compute::ComputePipeline;
    use crate::device::Device;
    use crate::shader::ShaderModule;
    use crate::task_graph::{IrSubmitState, NodeAccess};
    use std::sync::Arc;

    fn mock_device() -> Arc<Device> {
        Arc::new(Device::from_backend(Box::new(MockBackend::new())).unwrap())
    }

    fn mock_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(device, "void main() {}").unwrap()
    }

    fn mock_pipeline(device: &Device, shader: &ShaderModule) -> ComputePipeline {
        ComputePipeline::new(device, shader).unwrap()
    }

    fn mock_buf(device: &Device) -> Buffer {
        Buffer::new(device, 256, crate::BufferKind::Scattered).unwrap()
    }

    /// Read `retained_resubmit_count` from the mock backend.
    fn resubmit_count(device: &Device) -> usize {
        device.with_mock(|m| m.retained_resubmit_count)
    }

    /// Read the number of live retained graph entries.
    fn retained_count(device: &Device) -> usize {
        device.with_mock(|m| m.retained_graph_count())
    }

    fn do_submit(state: &mut IrSubmitState, ctx: &crate::Context, ir: &GraphIR) {
        state.submit_pipelined_and_retain_with_presents(ctx, ir, &[]).unwrap();
    }

    // ------------------------------------------------------------------
    // Baseline: a single-partition (1 or 2 wave) scheme retains as one slice
    // and resubmits from it on subsequent clean submits.
    // ------------------------------------------------------------------

    #[test]
    fn single_partition_retains_and_resubmits() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p = mock_pipeline(&device, &shader);
        let buf = mock_buf(&device);

        let mut ir = GraphIR::default();
        ir.nodes.push(TaskNode {
            label: "a",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Buffer(buf.handle),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::Dispatch {
                pipeline: p.handle,
                resource_slots: vec![],
                user_slots: vec![],
                dispatch: DispatchDim::Direct { x: 1, y: 1, z: 1 },
            },
        });

        let mut state = IrSubmitState::new();

        // First submit: records and retains.
        do_submit(&mut state, &ctx, &ir);
        assert_eq!(resubmit_count(&device), 0, "first submit records, does not resubmit");
        assert_eq!(retained_count(&device), 1, "one slice retained");

        // Second submit (unchanged IR): should resubmit from cache.
        do_submit(&mut state, &ctx, &ir);
        assert_eq!(resubmit_count(&device), 1, "second submit resubmits retained slice");
        assert_eq!(retained_count(&device), 1, "still one slice retained");
    }

    // ------------------------------------------------------------------
    // Two-partition IR: a 3-wave linear chain produces two partitions.
    // Both are retained on first submit; both resubmit on second submit.
    // ------------------------------------------------------------------

    /// Build a 3-wave linear-chain IR: A writes buf0 → B reads buf0 writes buf1 → C reads buf1.
    /// With 3 waves the `emit_partitioned_commands` split produces two partitions:
    ///   partition 0 = waves 0..split, partition 1 = waves split..3.
    fn three_wave_ir(
        p_a: &ComputePipeline,
        p_b: &ComputePipeline,
        p_c: &ComputePipeline,
        buf0: &Buffer,
        buf1: &Buffer,
    ) -> GraphIR {
        let mut ir = GraphIR::default();
        ir.nodes.push(TaskNode {
            label: "a",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Buffer(buf0.handle),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::Dispatch {
                pipeline: p_a.handle,
                resource_slots: vec![],
                user_slots: vec![],
                dispatch: DispatchDim::Direct { x: 1, y: 1, z: 1 },
            },
        });
        ir.nodes.push(TaskNode {
            label: "b",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::Buffer(buf0.handle),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::Buffer(buf1.handle),
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::Dispatch {
                pipeline: p_b.handle,
                resource_slots: vec![],
                user_slots: vec![],
                dispatch: DispatchDim::Direct { x: 1, y: 1, z: 1 },
            },
        });
        ir.nodes.push(TaskNode {
            label: "c",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Buffer(buf1.handle),
                access: NodeAccess::Read,
            }],
            kind: NodeKind::Dispatch {
                pipeline: p_c.handle,
                resource_slots: vec![],
                user_slots: vec![],
                dispatch: DispatchDim::Direct { x: 1, y: 1, z: 1 },
            },
        });
        ir
    }

    #[test]
    fn two_partition_ir_retains_both_slices() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p_a = mock_pipeline(&device, &shader);
        let p_b = mock_pipeline(&device, &shader);
        let p_c = mock_pipeline(&device, &shader);
        let buf0 = mock_buf(&device);
        let buf1 = mock_buf(&device);

        let ir = three_wave_ir(&p_a, &p_b, &p_c, &buf0, &buf1);
        let mut state = IrSubmitState::new();

        // First submit: both partitions record and retain.
        do_submit(&mut state, &ctx, &ir);
        assert_eq!(resubmit_count(&device), 0, "first submit records all partitions");
        assert_eq!(retained_count(&device), 2, "two slices retained for two partitions");

        // Second submit: both partitions resubmit.
        do_submit(&mut state, &ctx, &ir);
        assert_eq!(
            resubmit_count(&device),
            2,
            "second submit resubmits both retained slices"
        );
        assert_eq!(retained_count(&device), 2, "still two slices retained");
    }

    // ------------------------------------------------------------------
    // Selective re-record: change only partition 1 (node C's pipeline),
    // assert partition 0 is resubmitted while partition 1 re-records.
    // ------------------------------------------------------------------

    #[test]
    fn changing_second_partition_only_rerecords_second_partition() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p_a = mock_pipeline(&device, &shader);
        let p_b = mock_pipeline(&device, &shader);
        let p_c = mock_pipeline(&device, &shader);
        let p_c2 = mock_pipeline(&device, &shader);
        let buf0 = mock_buf(&device);
        let buf1 = mock_buf(&device);

        // The mock assigns distinct handles per pipeline.
        assert_ne!(p_c.handle, p_c2.handle, "test requires distinct pipeline handles");

        let ir = three_wave_ir(&p_a, &p_b, &p_c, &buf0, &buf1);
        let mut state = IrSubmitState::new();

        // Frame 1: record all partitions.
        do_submit(&mut state, &ctx, &ir);
        assert_eq!(resubmit_count(&device), 0);

        // Frame 2: only node C's pipeline changed → only partition 1 re-records;
        //          partition 0 resubmits from retained cache (one resubmit hit).
        let ir2 = three_wave_ir(&p_a, &p_b, &p_c2, &buf0, &buf1);
        do_submit(&mut state, &ctx, &ir2);
        assert_eq!(
            resubmit_count(&device),
            1,
            "partition 0 resubmits; partition 1 re-records — one resubmit total"
        );

        // Frame 3: both partitions are now cached (partition 1 was retained in frame 2).
        do_submit(&mut state, &ctx, &ir2);
        assert_eq!(resubmit_count(&device), 3, "third submit resubmits both partitions");
    }

    // ------------------------------------------------------------------
    // Changing partition 0 invalidates only partition 0; partition 1 resubmits.
    // ------------------------------------------------------------------

    #[test]
    fn changing_first_partition_only_rerecords_first_partition() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p_a = mock_pipeline(&device, &shader);
        let p_a2 = mock_pipeline(&device, &shader);
        let p_b = mock_pipeline(&device, &shader);
        let p_c = mock_pipeline(&device, &shader);
        let buf0 = mock_buf(&device);
        let buf1 = mock_buf(&device);

        assert_ne!(p_a.handle, p_a2.handle);

        let ir = three_wave_ir(&p_a, &p_b, &p_c, &buf0, &buf1);
        let mut state = IrSubmitState::new();

        do_submit(&mut state, &ctx, &ir);
        assert_eq!(resubmit_count(&device), 0);

        // Change only node A → partition 0 re-records; partition 1 resubmits.
        let ir2 = three_wave_ir(&p_a2, &p_b, &p_c, &buf0, &buf1);
        do_submit(&mut state, &ctx, &ir2);
        assert_eq!(
            resubmit_count(&device),
            1,
            "partition 1 resubmits; partition 0 re-records"
        );
    }

    // ------------------------------------------------------------------
    // Upload node in one partition does not prevent retention of the other
    // partition. The upload partition is submitted standalone each frame;
    // the pure-compute partition is retained.
    // ------------------------------------------------------------------

    #[test]
    fn upload_in_one_partition_does_not_prevent_other_partition_retention() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p_b = mock_pipeline(&device, &shader);
        let p_c = mock_pipeline(&device, &shader);
        let buf0 = mock_buf(&device);
        let buf1 = mock_buf(&device);
        let buf2 = mock_buf(&device);

        // Three-wave IR so `emit_partitioned_commands` splits into two partitions.
        //
        // Wave 0: upload writes buf0          → partition 0 (upload present, not retainable)
        // Wave 1: compute_b reads buf0        → partition 0 (same partition as wave 0)
        //         writes buf1
        // Wave 2: compute_c reads buf1        → partition 1 (pure compute, retained)
        //         writes buf2
        //
        // The split point is the wave boundary with highest barrier cost; for this chain
        // both wave-1 and wave-2 each have one buffer barrier so `max_by_key` (last-max
        // semantics) picks split = 2 → partition 0 = waves 0..2, partition 1 = wave 2.
        let mut ir = GraphIR::default();
        ir.nodes.push(TaskNode {
            label: "upload",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Buffer(buf0.handle),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteBuffer {
                buffer: buf0.handle,
                offset: 0,
                data: Arc::from(vec![0u8; 4]),
            },
        });
        ir.nodes.push(TaskNode {
            label: "compute_b",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::Buffer(buf0.handle),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::Buffer(buf1.handle),
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::Dispatch {
                pipeline: p_b.handle,
                resource_slots: vec![],
                user_slots: vec![],
                dispatch: DispatchDim::Direct { x: 1, y: 1, z: 1 },
            },
        });
        ir.nodes.push(TaskNode {
            label: "compute_c",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::Buffer(buf1.handle),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::Buffer(buf2.handle),
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::Dispatch {
                pipeline: p_c.handle,
                resource_slots: vec![],
                user_slots: vec![],
                dispatch: DispatchDim::Direct { x: 1, y: 1, z: 1 },
            },
        });

        let mut state = IrSubmitState::new();

        // Frame 1: partition 0 (upload) submitted standalone; partition 1 (compute) retained.
        do_submit(&mut state, &ctx, &ir);
        assert_eq!(resubmit_count(&device), 0, "first submit: no resubmits");
        assert_eq!(retained_count(&device), 1, "one retained slice (partition 1 only)");

        // Frame 2: partition 0 re-runs standalone; partition 1 resubmits from cache.
        do_submit(&mut state, &ctx, &ir);
        assert_eq!(
            resubmit_count(&device),
            1,
            "compute partition resubmits on second submit"
        );
    }
}

// ---------------------------------------------------------------------------
// Partitioning tests (no GPU submit)
//
// These tests operate at the *logical partition* level using
// `analysis::describe_logical_partitions`.  They assert invariants that must
// hold regardless of the actualized split count or heuristic tuning:
//
//  • Present-boundary invariant: all pre-present logical partitions have
//    has_present == false; the present partition (if any) comes last.
//  • Render-kind invariant: every render-pass logical partition has
//    has_render == true; pure-compute ones have has_render == false.
//  • Cache-kind invariant: render partitions use the graph-command cache slot;
//    compute partitions use the GpuCommand cache slot; present partitions use
//    neither (deferred to submit time).
//  • Upload-remap invariant: every WriteBuffer/WriteTexture node in the IR
//    appears exactly once in the upload remap across all compute partitions.
//  • Cache-stability invariant: repeated calls with the same fingerprint
//    reuse the same allocated Vecs; a fingerprint change causes a rebuild.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod partitioning_tests {
    use super::*;
    use crate::backend::GpuCommand;
    use crate::task_graph::analysis::{self, LogicalPartition};
    use std::sync::Arc;

    // ------------------------------------------------------------------
    // IR construction helpers
    // ------------------------------------------------------------------

    fn buf(id: u64) -> ResourceId {
        ResourceId::Buffer(id)
    }

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

    fn render_pass_node(label: &'static str, target: RenderTargetHandle) -> TaskNode {
        TaskNode {
            label,
            bindings: vec![],
            kind: NodeKind::RenderPass {
                target,
                commands: Vec::new(),
            },
        }
    }

    fn copy_to_dst_node(label: &'static str, src: RenderTargetHandle, dst: ResourceId) -> TaskNode {
        TaskNode {
            label,
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::RenderTarget(src),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: dst,
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyRenderTarget { src, dst },
        }
    }

    fn grant_present_node(label: &'static str, grant_id: u32) -> TaskNode {
        TaskNode {
            label,
            bindings: vec![],
            kind: NodeKind::GrantPresent { grant_id },
        }
    }

    // ------------------------------------------------------------------
    // Analysis helpers — logical partitions and actualized cache
    // ------------------------------------------------------------------

    fn logical_partitions(ir: &GraphIR) -> Vec<LogicalPartition> {
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        analysis::describe_logical_partitions(ir, &schedule)
    }

    fn build_cache(ir: &GraphIR) -> CompiledCacheEntry {
        let mut cache: Option<CompiledCacheEntry> = None;
        let fp = binding_fingerprint(ir);
        TaskGraph::get_or_build_partitioned_commands(&mut cache, ir, fp);
        cache.unwrap()
    }

    // ------------------------------------------------------------------
    // Logical-partition invariant helpers
    // ------------------------------------------------------------------

    /// Assert the present-boundary invariant:
    ///   - At most one logical partition has has_present == true.
    ///   - The present partition (if any) must be the last one.
    fn assert_present_boundary_invariant(parts: &[LogicalPartition]) {
        let present_indices: Vec<usize> = parts
            .iter()
            .enumerate()
            .filter(|(_, p)| p.has_present)
            .map(|(i, _)| i)
            .collect();
        assert!(
            present_indices.len() <= 1,
            "at most one present partition expected, got indices {:?}",
            present_indices
        );
        if let Some(&pi) = present_indices.first() {
            assert_eq!(
                pi,
                parts.len() - 1,
                "present partition must be the last logical partition"
            );
        }
    }

    /// Assert the render-kind invariant:
    ///   - No partition mixes render and present.
    ///   - Each partition's has_render flag matches whether it actually
    ///     contains render-pass waves.
    fn assert_render_kind_invariant(parts: &[LogicalPartition]) {
        for (i, p) in parts.iter().enumerate() {
            assert!(
                !(p.has_render && p.has_present),
                "partition {i} must not mix render and present"
            );
        }
    }

    /// Assert the cache-kind invariant against an actualized CompiledCacheEntry:
    ///   - Render partitions: compute slot empty, graph slot Some.
    ///   - Present partitions: both slots empty/None (deferred to submit time).
    ///   - Pure-compute partitions: compute slot non-empty, graph slot None.
    fn assert_cache_kind_invariant(ir: &GraphIR, entry: &CompiledCacheEntry) {
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        let ranges = analysis::partition_wave_ranges(ir, &schedule);
        let parts = entry.partitioned_commands.as_ref().unwrap();

        for (i, range) in ranges.iter().enumerate() {
            let waves = &entry.schedule.waves[range.clone()];
            let has_render = partition_waves_have_render(ir, waves);
            let has_present = analysis::partition_waves_have_present(ir, waves);

            if has_present {
                assert!(
                    parts[i].is_empty(),
                    "present partition {i}: compute slot must be empty (deferred)"
                );
                assert!(
                    entry.partitioned_graph_commands[i].is_none(),
                    "present partition {i}: graph slot must be None (deferred)"
                );
            } else if has_render {
                assert!(parts[i].is_empty(), "render partition {i}: compute slot must be empty");
                assert!(
                    entry.partitioned_graph_commands[i].is_some(),
                    "render partition {i}: graph slot must be Some"
                );
                assert!(
                    entry.partitioned_graph_commands[i]
                        .as_ref()
                        .unwrap()
                        .iter()
                        .any(|c| matches!(c, GraphCommand::Render { .. })),
                    "render partition {i}: graph slot must contain a Render command"
                );
            } else {
                assert!(
                    !parts[i].is_empty(),
                    "compute partition {i}: compute slot must be non-empty"
                );
                assert!(
                    entry.partitioned_graph_commands[i].is_none(),
                    "compute partition {i}: graph slot must be None"
                );
            }
        }
    }

    /// Assert upload-remap invariant: every WriteBuffer/WriteTexture/WriteTextureRegion
    /// node in the IR appears exactly once in the remap table.
    fn assert_upload_remap_invariant(ir: &GraphIR, entry: &CompiledCacheEntry) {
        let upload_node_indices: Vec<usize> = ir
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                matches!(
                    n.kind,
                    NodeKind::WriteBuffer { .. } | NodeKind::WriteTexture { .. } | NodeKind::WriteTextureRegion { .. }
                )
            })
            .map(|(i, _)| i)
            .collect();

        let mut remap_node_indices: Vec<usize> = entry.partitioned_upload_remap.iter().map(|&(_, _, ni)| ni).collect();
        remap_node_indices.sort_unstable();

        let mut expected = upload_node_indices.clone();
        expected.sort_unstable();

        assert_eq!(
            remap_node_indices, expected,
            "upload remap must cover every upload node exactly once"
        );
    }

    // ------------------------------------------------------------------
    // Group 1: logical partitions — pure-compute schemes
    // ------------------------------------------------------------------

    #[test]
    fn single_dispatch_one_pure_compute_logical_partition() {
        let ir = GraphIR {
            nodes: vec![dispatch_node("a", 1, vec![(buf(0), NodeAccess::Write)], 4)],
        };
        let parts = logical_partitions(&ir);
        assert_eq!(parts.len(), 1);
        assert!(parts[0].is_pure_compute());
        assert_present_boundary_invariant(&parts);
        assert_render_kind_invariant(&parts);
    }

    #[test]
    fn independent_dispatches_share_wave_one_logical_partition() {
        let ir = GraphIR {
            nodes: vec![
                dispatch_node("a", 1, vec![(buf(0), NodeAccess::Write)], 4),
                dispatch_node("b", 2, vec![(buf(1), NodeAccess::Write)], 4),
            ],
        };
        let parts = logical_partitions(&ir);
        // Two independent dispatches land in the same wave → one logical partition.
        assert_eq!(parts.len(), 1);
        assert!(parts[0].is_pure_compute());
        assert_present_boundary_invariant(&parts);
    }

    #[test]
    fn linear_chain_stays_one_pure_compute_logical_partition() {
        // A→B→C linear chain: 3 waves, no render, no present.
        // The barrier-cost split is an *actualized* concern; logically it is one unit.
        let ir = GraphIR {
            nodes: vec![
                dispatch_node("a", 1, vec![(buf(0), NodeAccess::Write)], 1),
                dispatch_node("b", 2, vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)], 1),
                dispatch_node("c", 3, vec![(buf(1), NodeAccess::Read)], 1),
            ],
        };
        let parts = logical_partitions(&ir);
        assert_eq!(parts.len(), 1, "pure-compute linear chain is one logical partition");
        assert!(parts[0].is_pure_compute());
        assert_present_boundary_invariant(&parts);
    }

    // ------------------------------------------------------------------
    // Group 2: logical partitions — render-pass schemes
    // ------------------------------------------------------------------

    #[test]
    fn render_pass_alone_one_render_logical_partition() {
        let ir = GraphIR {
            nodes: vec![render_pass_node("draw", 10)],
        };
        let parts = logical_partitions(&ir);
        assert_eq!(parts.len(), 1);
        assert!(parts[0].has_render);
        assert!(!parts[0].has_present);
        assert_present_boundary_invariant(&parts);
        assert_render_kind_invariant(&parts);
    }

    #[test]
    fn render_then_copy_present_two_logical_partitions() {
        // RenderPass → CopyRenderTarget(PresentLease) → GrantPresent
        // Logical split: render partition | present partition.
        let ir = GraphIR {
            nodes: vec![
                render_pass_node("draw", 10),
                copy_to_dst_node("copy", 10, ResourceId::PresentLease(0)),
                grant_present_node("grant", 0),
            ],
        };
        let parts = logical_partitions(&ir);
        // Must have at least: one render partition + one present partition.
        assert!(parts.len() >= 2, "render→present must produce ≥ 2 logical partitions");
        let render_count = parts.iter().filter(|p| p.has_render).count();
        let present_count = parts.iter().filter(|p| p.has_present).count();
        assert_eq!(render_count, 1);
        assert_eq!(present_count, 1);
        assert_present_boundary_invariant(&parts);
        assert_render_kind_invariant(&parts);
    }

    #[test]
    fn compute_then_render_two_logical_partitions() {
        // The render pass reads from a buffer that the dispatch writes, so it lands
        // in a later wave — a render-kind boundary then forces a logical split.
        let ir = GraphIR {
            nodes: vec![
                dispatch_node("pre", 1, vec![(buf(0), NodeAccess::Write)], 1),
                TaskNode {
                    label: "draw",
                    bindings: vec![ResourceBinding {
                        resource: buf(0),
                        access: NodeAccess::Read,
                    }],
                    kind: NodeKind::RenderPass {
                        target: 10,
                        commands: Vec::new(),
                    },
                },
            ],
        };
        let parts = logical_partitions(&ir);
        assert!(parts.len() >= 2, "compute→render must produce ≥ 2 logical partitions");
        let has_pure_compute = parts.iter().any(|p| p.is_pure_compute());
        let has_render = parts.iter().any(|p| p.has_render);
        assert!(has_pure_compute, "must have at least one pure-compute partition");
        assert!(has_render, "must have at least one render partition");
        assert_present_boundary_invariant(&parts);
        assert_render_kind_invariant(&parts);
    }

    // ------------------------------------------------------------------
    // Group 3: logical partitions — present-lease schemes
    // ------------------------------------------------------------------

    #[test]
    fn present_lease_copy_only_has_present_partition() {
        let ir = GraphIR {
            nodes: vec![copy_to_dst_node("copy", 5, ResourceId::PresentLease(0))],
        };
        let parts = logical_partitions(&ir);
        let present_count = parts.iter().filter(|p| p.has_present).count();
        assert_eq!(present_count, 1, "must have exactly one present logical partition");
        assert!(!parts.iter().any(|p| p.has_render), "no render partitions expected");
        assert_present_boundary_invariant(&parts);
    }

    #[test]
    fn compute_then_present_present_is_last_logical_partition() {
        let ir = GraphIR {
            nodes: vec![
                dispatch_node("pre", 1, vec![(buf(0), NodeAccess::Write)], 1),
                dispatch_node("post", 2, vec![(buf(0), NodeAccess::Read)], 1),
                copy_to_dst_node("copy", 5, ResourceId::PresentLease(0)),
                grant_present_node("grant", 0),
            ],
        };
        let parts = logical_partitions(&ir);
        assert!(parts.last().unwrap().has_present, "present partition must be last");
        assert!(parts.iter().take(parts.len() - 1).all(|p| !p.has_present));
        assert_present_boundary_invariant(&parts);
        assert_render_kind_invariant(&parts);
    }

    #[test]
    fn render_then_compute_then_present_ordering() {
        // A richer scheme: render pass → compute post-process → present.
        let ir = GraphIR {
            nodes: vec![
                render_pass_node("draw", 10),
                dispatch_node("post", 1, vec![(buf(0), NodeAccess::Write)], 1),
                copy_to_dst_node("copy", 10, ResourceId::PresentLease(0)),
                grant_present_node("grant", 0),
            ],
        };
        let parts = logical_partitions(&ir);
        // Whatever the exact count, the present partition must be last.
        assert_present_boundary_invariant(&parts);
        assert_render_kind_invariant(&parts);
        assert!(parts.iter().any(|p| p.has_render));
        assert!(parts.last().unwrap().has_present);
    }

    // ------------------------------------------------------------------
    // Group 4: actualized cache — kind correctness
    // ------------------------------------------------------------------

    #[test]
    fn cache_kind_invariant_pure_compute() {
        let ir = GraphIR {
            nodes: vec![
                dispatch_node("a", 1, vec![(buf(0), NodeAccess::Write)], 1),
                dispatch_node("b", 2, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let entry = build_cache(&ir);
        assert_cache_kind_invariant(&ir, &entry);
    }

    #[test]
    fn cache_kind_invariant_render_only() {
        let ir = GraphIR {
            nodes: vec![render_pass_node("draw", 10)],
        };
        let entry = build_cache(&ir);
        assert_cache_kind_invariant(&ir, &entry);
    }

    #[test]
    fn cache_kind_invariant_present_deferred() {
        // Present partitions must have both cache slots empty after build_cache.
        let ir = GraphIR {
            nodes: vec![
                dispatch_node("pre", 1, vec![(buf(0), NodeAccess::Write)], 1),
                copy_to_dst_node("copy", 5, ResourceId::PresentLease(0)),
                grant_present_node("grant", 0),
            ],
        };
        let entry = build_cache(&ir);
        assert_cache_kind_invariant(&ir, &entry);
    }

    #[test]
    fn cache_kind_invariant_mixed_render_and_present() {
        let ir = GraphIR {
            nodes: vec![
                dispatch_node("a", 1, vec![(buf(0), NodeAccess::Write)], 1),
                render_pass_node("draw", 10),
                copy_to_dst_node("copy", 10, ResourceId::PresentLease(0)),
                grant_present_node("grant", 0),
            ],
        };
        let entry = build_cache(&ir);
        assert_cache_kind_invariant(&ir, &entry);
    }

    // ------------------------------------------------------------------
    // Group 5: upload remap coverage
    // ------------------------------------------------------------------

    #[test]
    fn write_then_dispatch_upload_remap_covers_all_uploads() {
        let ir = GraphIR {
            nodes: vec![
                write_node("upload", buf(0), 0),
                dispatch_node("a", 1, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let entry = build_cache(&ir);
        assert_upload_remap_invariant(&ir, &entry);
    }

    #[test]
    fn two_uploads_remap_covers_both() {
        let ir = GraphIR {
            nodes: vec![
                write_node("upload_a", buf(0), 0),
                write_node("upload_b", buf(1), 1),
                dispatch_node("a", 1, vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Read)], 1),
            ],
        };
        let entry = build_cache(&ir);
        assert_upload_remap_invariant(&ir, &entry);
        assert_eq!(entry.partitioned_upload_remap.len(), 2);
    }

    #[test]
    fn no_uploads_remap_is_empty() {
        let ir = GraphIR {
            nodes: vec![
                dispatch_node("a", 1, vec![(buf(0), NodeAccess::Write)], 1),
                dispatch_node("b", 2, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let entry = build_cache(&ir);
        assert_eq!(entry.partitioned_upload_remap.len(), 0);
    }

    // ------------------------------------------------------------------
    // Group 6: cache stability
    // ------------------------------------------------------------------

    #[test]
    fn cache_hit_reuses_allocated_vecs() {
        let ir = GraphIR {
            nodes: vec![
                dispatch_node("a", 1, vec![(buf(0), NodeAccess::Write)], 1),
                dispatch_node("b", 2, vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)], 1),
                dispatch_node("c", 3, vec![(buf(1), NodeAccess::Read)], 1),
            ],
        };
        let mut cache: Option<CompiledCacheEntry> = None;
        let fp = binding_fingerprint(&ir);
        TaskGraph::get_or_build_partitioned_commands(&mut cache, &ir, fp);
        let ptr_before = cache.as_ref().unwrap().partitioned_commands.as_ref().unwrap().as_ptr();

        TaskGraph::get_or_build_partitioned_commands(&mut cache, &ir, fp);
        let ptr_after = cache.as_ref().unwrap().partitioned_commands.as_ref().unwrap().as_ptr();
        assert_eq!(
            ptr_before, ptr_after,
            "cache hit must not reallocate the partitioned_commands Vec"
        );
    }

    #[test]
    fn fingerprint_change_rebuilds_cache() {
        // Changing the binding structure (different buffer id) changes the fingerprint.
        let ir_v1 = GraphIR {
            nodes: vec![dispatch_node("a", 1, vec![(buf(0), NodeAccess::Write)], 1)],
        };
        let ir_v2 = GraphIR {
            nodes: vec![dispatch_node("a", 1, vec![(buf(1), NodeAccess::Write)], 1)],
        };
        let fp1 = binding_fingerprint(&ir_v1);
        let fp2 = binding_fingerprint(&ir_v2);
        assert_ne!(fp1, fp2, "test requires distinct binding fingerprints");

        let mut cache: Option<CompiledCacheEntry> = None;
        TaskGraph::get_or_build_partitioned_commands(&mut cache, &ir_v1, fp1);
        assert_eq!(cache.as_ref().unwrap().fp, fp1);

        TaskGraph::get_or_build_partitioned_commands(&mut cache, &ir_v2, fp2);
        assert_eq!(
            cache.as_ref().unwrap().fp,
            fp2,
            "cache must rebuild on fingerprint change"
        );
        assert!(cache.as_ref().unwrap().partitioned_commands.is_some());
    }

    #[test]
    fn upload_payload_refreshes_on_cache_hit() {
        let mut ir = GraphIR {
            nodes: vec![
                write_node("upload", buf(0), 0),
                dispatch_node("a", 1, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let mut cache: Option<CompiledCacheEntry> = None;
        let fp = binding_fingerprint(&ir);
        TaskGraph::get_or_build_partitioned_commands(&mut cache, &ir, fp);

        let ptr_before = {
            let parts = cache.as_ref().unwrap().partitioned_commands.as_ref().unwrap();
            match &parts[0][0] {
                GpuCommand::WriteBuffer { data, .. } => Arc::as_ptr(data),
                other => panic!("expected WriteBuffer, got {other:?}"),
            }
        };

        // Mutate the upload payload — same fingerprint, different Arc.
        if let NodeKind::WriteBuffer { data, .. } = &mut ir.nodes[0].kind {
            *data = Arc::from(vec![9u8; 4]);
        }
        TaskGraph::get_or_build_partitioned_commands(&mut cache, &ir, fp);

        let ptr_after = {
            let parts = cache.as_ref().unwrap().partitioned_commands.as_ref().unwrap();
            match &parts[0][0] {
                GpuCommand::WriteBuffer { data, .. } => Arc::as_ptr(data),
                other => panic!("expected WriteBuffer, got {other:?}"),
            }
        };
        assert_ne!(
            ptr_before, ptr_after,
            "upload payload Arc must be refreshed on cache hit"
        );
        // Partition count must not change.
        assert_eq!(cache.as_ref().unwrap().partitioned_commands.as_ref().unwrap().len(), 1);
    }
}
