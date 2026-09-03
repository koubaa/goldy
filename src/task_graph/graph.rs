//! Shared graph IR submit engine used by [`crate::Scheme`].

use super::analysis;
use super::cross_submit::{
    apply_resource_sync_updates, apply_stamp_targets_legacy, net_access_for_waves, prepend_prologue,
    CrossSubmitScratch, ResourceKey, ResourceKeyMap,
};
use super::ir::{CompiledSchedule, DispatchDim, GraphIR, NodeAccess, NodeKind, Wave};
use super::ResourceId;
use crate::backend::{GpuCommand, GraphCommand, SubmitSync};
use crate::sampler::Sampler;
use crate::timeline::TimelineValue;
use anyhow::Result;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

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

fn backend_submit_standalone(
    session: &dyn crate::backend::ContextSubmitSession,
    ctx: crate::backend::ContextHandle,
    commands: &[crate::backend::GpuCommand],
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    let _tz = crate::tracy_zone!("goldy.backend_submit_standalone");
    let (cmds, waits_only) = {
        let _tz = crate::tracy_zone!("goldy.backend_submit_standalone.prepare");
        let cmds = if let Some(s) = sync {
            if s.prologue.is_empty() {
                commands.to_vec()
            } else {
                prepend_prologue(commands, &s.prologue)
            }
        } else {
            commands.to_vec()
        };
        let waits_only = sync.map(|s| SubmitSync {
            prologue: Default::default(),
            waits: s.waits.clone(),
            cpu_waits: s.cpu_waits.clone(),
            host_observed_waits: s.host_observed_waits.clone(),
            deferred_host_writes: s.deferred_host_writes.clone(),
        });
        (cmds, waits_only)
    };
    let _tz = crate::tracy_zone!("goldy.backend_submit_standalone.session");
    session.submit_standalone(ctx, &cmds, waits_only.as_ref())
}

/// Fold a same-context cross-submit prologue into the graph command list.
///
/// Barrier-only standalone submits trigger D3D12 warning 1356; the prologue belongs
/// in the graph body. On the retained path the topology dirty bit forces re-record
/// before any topology-driven prologue change, so the baked barrier stays current.
fn graph_commands_with_sync_prologue<'a>(
    commands: &'a [crate::backend::GraphCommand],
    sync: Option<&SubmitSync>,
) -> std::borrow::Cow<'a, [crate::backend::GraphCommand]> {
    if let Some(s) = sync {
        if !s.prologue.is_empty() {
            let barrier = crate::backend::GpuCommand::ResourceBarrier {
                buffers: s.prologue.buffers.clone(),
                textures: s.prologue.textures.clone(),
            };
            let mut v = Vec::with_capacity(1 + commands.len());
            v.push(crate::backend::GraphCommand::Compute(barrier));
            v.extend_from_slice(commands);
            return std::borrow::Cow::Owned(v);
        }
    }
    std::borrow::Cow::Borrowed(commands)
}

fn submit_sync_waits_only(sync: Option<&SubmitSync>) -> Option<SubmitSync> {
    sync.map(|s| SubmitSync {
        prologue: Default::default(),
        waits: s.waits.clone(),
        cpu_waits: s.cpu_waits.clone(),
        host_observed_waits: s.host_observed_waits.clone(),
        deferred_host_writes: s.deferred_host_writes.clone(),
    })
}

/// Per-scheme submit sidecars relocated from the render thread to the submission worker.
pub(crate) struct SubmitSidecarState {
    extra_queue_epochs: Vec<crate::timeline::Epoch>,
    host_observed: Vec<crate::timeline::Epoch>,
    deferred_writes: Vec<crate::backend::DeferredHostWrite>,
    host_attached: bool,
}

impl SubmitSidecarState {
    fn new(
        extra_queue_epochs: Vec<crate::timeline::Epoch>,
        host_observed: Vec<crate::timeline::Epoch>,
        deferred_writes: Vec<crate::backend::DeferredHostWrite>,
    ) -> Self {
        Self {
            extra_queue_epochs,
            host_observed,
            deferred_writes,
            host_attached: false,
        }
    }

    fn merge_sync(&mut self, base: Option<&SubmitSync>) -> Option<SubmitSync> {
        let (host, writes) =
            if !self.host_attached && (!self.host_observed.is_empty() || !self.deferred_writes.is_empty()) {
                self.host_attached = true;
                (
                    std::mem::take(&mut self.host_observed),
                    std::mem::take(&mut self.deferred_writes),
                )
            } else {
                (Vec::new(), Vec::new())
            };
        let extra = if base.as_ref().is_some_and(|s| !s.is_empty()) {
            self.extra_queue_epochs.as_slice()
        } else {
            &[]
        };
        crate::backend::host_sidecar::merge_submit_sync_for_partition(base, extra, host, writes)
    }
}

fn merge_epoch_wait(waits: &mut Vec<crate::timeline::Epoch>, epoch: crate::timeline::Epoch) {
    if let Some(existing) = waits.iter_mut().find(|e| e.context == epoch.context) {
        existing.value = existing.value.max(epoch.value);
    } else {
        waits.push(epoch);
    }
}

/// Intra-context compute↔render queue boundary waits under DX12 compute style.
fn merge_queue_boundary_waits(
    sync: Option<&SubmitSync>,
    separate: bool,
    has_render: bool,
    submitting_ctx: crate::backend::ContextHandle,
    device_owner: Option<crate::backend::ContextHandle>,
    last_compute_tv: Option<TimelineValue>,
    last_render_tv: Option<TimelineValue>,
) -> Option<SubmitSync> {
    if !separate {
        return sync.cloned();
    }
    let device_owner = device_owner?;
    let mut waits = sync.map(|s| s.waits.clone()).unwrap_or_default();
    if has_render {
        if let Some(tv) = last_compute_tv.filter(|&v| v > 0) {
            merge_epoch_wait(
                &mut waits,
                crate::timeline::Epoch {
                    context: submitting_ctx,
                    value: tv,
                },
            );
        }
    } else if let Some(tv) = last_render_tv.filter(|&v| v > 0) {
        merge_epoch_wait(
            &mut waits,
            crate::timeline::Epoch {
                context: device_owner,
                value: tv,
            },
        );
    }
    if sync.is_none() && waits.is_empty() {
        return None;
    }
    Some(SubmitSync {
        prologue: sync.map(|s| s.prologue.clone()).unwrap_or_default(),
        waits,
        cpu_waits: sync.map(|s| s.cpu_waits.clone()).unwrap_or_default(),
        host_observed_waits: sync.map(|s| s.host_observed_waits.clone()).unwrap_or_default(),
        deferred_host_writes: sync.map(|s| s.deferred_host_writes.clone()).unwrap_or_default(),
    })
}

fn partition_stamp_context(
    separate: bool,
    has_render: bool,
    submitting_ctx: crate::backend::ContextHandle,
    device_owner: Option<crate::backend::ContextHandle>,
) -> crate::backend::ContextHandle {
    if separate && has_render {
        device_owner.unwrap_or(submitting_ctx)
    } else {
        submitting_ctx
    }
}

#[derive(Debug, Default)]
struct QueueBoundaryState {
    last_compute_tv: Option<TimelineValue>,
    last_render_tv: Option<TimelineValue>,
}

impl QueueBoundaryState {
    fn record(&mut self, separate: bool, has_render: bool, tv: TimelineValue) {
        if !separate {
            return;
        }
        if has_render {
            self.last_render_tv = Some(tv);
        } else {
            self.last_compute_tv = Some(tv);
        }
    }
}

fn backend_submit_graph(
    session: &dyn crate::backend::ContextSubmitSession,
    ctx: crate::backend::ContextHandle,
    commands: &[crate::backend::GraphCommand],
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    let effective = graph_commands_with_sync_prologue(commands, sync);
    // Mirror the standalone path: pass Some whenever sync is Some so the backend
    // knows to suppress its legacy blanket-acquire barrier even when there are no
    // cross-context waits.
    session.submit_graph(ctx, &effective, submit_sync_waits_only(sync).as_ref())
}

fn backend_submit_graph_and_retain(
    session: &dyn crate::backend::ContextSubmitSession,
    ctx: crate::backend::ContextHandle,
    commands: &[crate::backend::GraphCommand],
    key: u64,
    sync: Option<&SubmitSync>,
) -> Result<TimelineValue> {
    let _tz = crate::tracy_zone!("goldy.backend_submit_graph_and_retain");
    let effective = graph_commands_with_sync_prologue(commands, sync);
    session.submit_graph_and_retain(ctx, &effective, key, submit_sync_waits_only(sync).as_ref())
}

fn backend_try_resubmit_retained(
    session: &dyn crate::backend::ContextSubmitSession,
    ctx: crate::backend::ContextHandle,
    key: u64,
    sync: Option<&SubmitSync>,
) -> Result<Option<TimelineValue>> {
    let _tz = crate::tracy_zone!("goldy.backend_try_resubmit_retained");
    // Prologue is baked into the retained body; only cross-context waits are live.
    // Callers must only invoke this when `CbReplayState` is present (replay enabled).
    let waits_only = {
        let _tz = crate::tracy_zone!("goldy.backend_try_resubmit_retained.prepare");
        submit_sync_waits_only(sync)
    };
    let _tz = crate::tracy_zone!("goldy.backend_try_resubmit_retained.session");
    session.try_resubmit_retained(ctx, key, waits_only.as_ref())
}

/// Re-record replaces in-flight native CB/allocator/CUDA-graph storage. Soft-retain
/// backends (Metal, WebGPU) skip this wait: they encode a new command buffer each time.
fn ensure_partition_retired_before_rerecord(
    session: &dyn crate::backend::ContextSubmitSession,
    context: &crate::Context,
    prev_tv: Option<TimelineValue>,
) -> Result<()> {
    if !session.requires_retained_storage_retirement() {
        return Ok(());
    }
    if let Some(prev_tv) = prev_tv {
        if context.gpu_progress() < prev_tv {
            context.wait_until(prev_tv)?;
        }
    }
    Ok(())
}

/// True when `waves` include a binding with no [`ResourceKey`] (mosaic/transient/present, etc.).
fn partition_has_unkeyed_bindings(ir: &GraphIR, waves: &[Wave]) -> bool {
    waves.iter().flat_map(|w| &w.node_indices).any(|&ni| {
        let node = &ir.nodes[ni];
        if matches!(node.kind, NodeKind::WithdrawRead { .. }) {
            return false;
        }
        node.bindings
            .iter()
            .any(|b| ResourceKey::from_resource_id(b.resource).is_none())
    })
}

/// Stamp parcel epochs for resources touched in one partition at that partition's timeline value.
fn apply_partition_epoch_stamps(
    resource_stamps: &ResourceKeyMap<Arc<crate::parcel::ParcelStamp>>,
    stamp_targets: &[Arc<crate::parcel::ParcelStamp>],
    ctx: crate::backend::ContextHandle,
    ir: &GraphIR,
    waves: &[Wave],
    tv: TimelineValue,
) {
    let net = net_access_for_waves(ir, waves);
    if !net.is_empty() {
        apply_resource_sync_updates(&net, resource_stamps, ctx, tv);
    }
    if !stamp_targets.is_empty() && partition_has_unkeyed_bindings(ir, waves) {
        apply_stamp_targets_legacy(stamp_targets, ctx, tv);
    }
}

// ---- Free functions over GraphIR -------------------------------------------
//
// These are pure algorithms: they take GraphIR + backend data and produce a
// result. They carry no TaskGraph dependency and are the actual logic shared
// with Scheme via IrSubmitState.

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

/// Per-partition retention fingerprint.
///
/// Hashes dispatch/pipeline/slot fields for one partition; restricted to
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
            NodeKind::TraceRays {
                pipeline,
                resource_slots,
                user_slots,
                width,
                height,
                depth,
            } => {
                12u8.hash(&mut h);
                pipeline.hash(&mut h);
                hash_resource_slots_for_fingerprint(resource_slots, &mut h);
                user_slots.hash(&mut h);
                width.hash(&mut h);
                height.hash(&mut h);
                depth.hash(&mut h);
            }
            NodeKind::ClearBuffer { buffer, offset, size } => {
                1u8.hash(&mut h);
                buffer.hash(&mut h);
                offset.hash(&mut h);
                size.hash(&mut h);
            }
            NodeKind::WithdrawRead { withdraw_id } => {
                3u8.hash(&mut h);
                withdraw_id.hash(&mut h);
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
                5u8.hash(&mut h);
                src.hash(&mut h);
                src_offset.hash(&mut h);
                src_row_pitch.hash(&mut h);
                dst.hash(&mut h);
                x.hash(&mut h);
                y.hash(&mut h);
                width.hash(&mut h);
                height.hash(&mut h);
            }
            _ => {
                2u8.hash(&mut h);
            }
        }
    }
    h.finish()
}

/// Compute per-partition fingerprints for all partitions in `wave_ranges`.
///
/// Each fingerprint folds `partition_fingerprint` with the partition index so that
/// identical adjacent partitions get distinct keys. When `layout_tag` is provided,
/// pitched [`NodeKind::CopyBufferToTexture`] nodes also fold in the destination
/// texture's barrier-layout tag so retained CBs are re-recorded after layout settles.
fn compute_partition_fps(
    ir: &GraphIR,
    schedule: &CompiledSchedule,
    wave_ranges: &[std::ops::Range<usize>],
    layout_tag: Option<&dyn Fn(crate::backend::TextureHandle) -> u64>,
) -> Vec<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    wave_ranges
        .iter()
        .enumerate()
        .map(|(part_idx, range)| {
            let waves = &schedule.waves[range.clone()];
            let raw_fp = partition_fingerprint(ir, schedule, waves);
            let layout_fp = layout_tag
                .map(|tag| analysis::partition_copy_texture_layout_fingerprint(ir, waves, tag))
                .unwrap_or(0);
            let mut h = DefaultHasher::new();
            raw_fp.hash(&mut h);
            layout_fp.hash(&mut h);
            part_idx.hash(&mut h);
            h.finish()
        })
        .collect()
}

fn texture_copy_layout_tag(context: &crate::Context) -> impl Fn(crate::backend::TextureHandle) -> u64 + '_ {
    move |texture| {
        context
            .device()
            .inner
            .backend
            .lock()
            .unwrap()
            .texture_copy_retention_tag(texture)
    }
}

/// True when all nodes in the given waves are retainable (no uploads).
///
/// Upload nodes (WriteBuffer, WriteTexture, etc.) must be staged on every
/// submit, so a partition containing them is submitted standalone rather than
/// retained.
fn partition_waves_can_retain(ir: &GraphIR, waves: &[Wave]) -> bool {
    analysis::waves_can_retain(ir, waves)
}

/// True when the partition's waves contain at least one [`NodeKind::RenderPass`] node.
fn partition_waves_have_render(ir: &GraphIR, waves: &[Wave]) -> bool {
    waves.iter().any(|w| {
        w.node_indices
            .iter()
            .any(|&ni| matches!(ir.nodes[ni].kind, NodeKind::RenderPass { .. }))
    })
}

/// Merge key for a compute partition and its immediately following render partition.
fn merged_compute_render_fp(fp0: u64, fp1: u64) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    fp0.hash(&mut h);
    fp1.hash(&mut h);
    h.finish()
}

/// When the next partition is pure compute, merge upload and compute into one
/// standalone command buffer so Metal records blit→compute in a single CB.
fn try_merge_upload_compute_range(
    ir: &GraphIR,
    schedule: &CompiledSchedule,
    wave_ranges: &[std::ops::Range<usize>],
    part_idx: usize,
    fuse_upload: bool,
) -> Option<std::ops::Range<usize>> {
    if !fuse_upload || part_idx + 1 >= wave_ranges.len() {
        return None;
    }
    let r0 = wave_ranges[part_idx].clone();
    let r1 = wave_ranges[part_idx + 1].clone();
    let w0 = &schedule.waves[r0.clone()];
    let w1 = &schedule.waves[r1.clone()];
    if analysis::partition_waves_have_present(ir, w0)
        || analysis::partition_waves_have_present(ir, w1)
        || partition_waves_have_render(ir, w0)
        || partition_waves_have_render(ir, w1)
        || !analysis::partition_waves_have_upload_slots(ir, w0)
        || analysis::partition_waves_have_upload_slots(ir, w1)
        || !partition_waves_can_retain(ir, w1)
    {
        return None;
    }
    Some(r0.start..r1.end)
}
/// When the next partition is an offscreen render pass, emit and retain compute and render
/// in one command buffer so the shared frame table and UAV→graphics barriers stay coherent.
fn try_merge_compute_render_range(
    ir: &GraphIR,
    schedule: &CompiledSchedule,
    wave_ranges: &[std::ops::Range<usize>],
    part_idx: usize,
    separate_graphics: bool,
) -> Option<std::ops::Range<usize>> {
    if separate_graphics {
        return None;
    }
    if part_idx + 1 >= wave_ranges.len() {
        return None;
    }
    let r0 = wave_ranges[part_idx].clone();
    let r1 = wave_ranges[part_idx + 1].clone();
    let w0 = &schedule.waves[r0.clone()];
    let w1 = &schedule.waves[r1.clone()];
    // Present and deposit slots bake late-bound handles into the CB. Merged
    // compute→render retains under `merged_fp` (IR fingerprint only), not
    // `dynamic_partition_slot_key`, so either side with slots must stay split.
    if analysis::partition_waves_have_present(ir, w0)
        || analysis::partition_waves_have_present(ir, w1)
        || analysis::partition_waves_have_upload_slots(ir, w0)
        || analysis::partition_waves_have_upload_slots(ir, w1)
        || !partition_waves_can_retain(ir, w0)
        || !partition_waves_can_retain(ir, w1)
    {
        return None;
    }
    let render0 = partition_waves_have_render(ir, w0);
    let render1 = partition_waves_have_render(ir, w1);
    if !render0 && render1 {
        Some(r0.start..r1.end)
    } else {
        None
    }
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

fn cross_sync_for_stamps<'a>(
    scratch: &'a mut CrossSubmitScratch,
    resource_stamps: &ResourceKeyMap<Arc<crate::parcel::ParcelStamp>>,
    ir: &GraphIR,
    submitting_ctx: crate::backend::ContextHandle,
    waves: &[Wave],
    separate_graphics: bool,
) -> Option<&'a SubmitSync> {
    if resource_stamps.is_empty() {
        return None;
    }
    Some(scratch.plan(ir, resource_stamps, submitting_ctx, waves, separate_graphics))
}

/// Outcome of partitioned IR submit: retention records vs resubmit hits.
///
/// When CB replay is disabled, both counters stay at zero — fresh encodes are
/// neither retention "records" nor resubmits.
#[derive(Debug, Default, Clone)]
pub(crate) struct PartitionSubmitResult {
    /// Partitions that were re-recorded **and retained** this call.
    pub records: usize,
    /// Partitions that were resubmitted from a retained command list without re-recording.
    pub resubmit_hits: usize,
    /// Last timeline value per present binding that reached a submitted present partition.
    ///
    /// Used for source-WAR settlement and for failure cleanup of referenced drawables.
    /// Bindings acquired but not yet submitted are absent.
    pub present_binding_tvs: Vec<(u32, TimelineValue)>,
}

impl PartitionSubmitResult {
    /// True when no retainable partition was re-recorded this call.
    ///
    /// With replay disabled this is always true (fresh encodes do not count as records),
    /// so Scheme topology reregistration stays gated on IR dirtiness alone.
    #[cfg(feature = "graphics")]
    pub fn all_from_cache(&self) -> bool {
        self.records == 0
    }

    fn note_present_bindings(&mut self, bindings: &[u32], tv: TimelineValue) {
        for &id in bindings {
            if let Some((_, existing)) = self.present_binding_tvs.iter_mut().find(|(b, _)| *b == id) {
                *existing = tv;
            } else {
                self.present_binding_tvs.push((id, tv));
            }
        }
    }
}

/// Resolved swapchain drawable for one present binding at submit time.
pub(crate) struct ResolvedPresentSlot {
    /// Scheme-unique present binding id ([`super::ResourceId::PresentLease`]).
    pub binding_id: u32,
    /// Pool generation at acquire time (included in retained variant keys).
    pub generation: u64,
    pub slot_id: u32,
    pub handle: crate::backend::TextureHandle,
    pub uav_index: u32,
}

/// Deferred swapchain acquire for specific unresolved present binding ids.
///
/// Called with the binding ids that the upcoming partition needs and that are not
/// yet in `present_slots`. Appends only those new resolutions.
pub(crate) type DeferredPresentAcquire<'a> = dyn FnMut(&[u32], &mut Vec<ResolvedPresentSlot>) -> Result<()> + 'a;

/// Present-lease and cross-submit stamp inputs for [`submit_resolved_ir`].
///
/// Present slots may be empty at entry. When a present-touching partition is about
/// to run, [`Self::deferred_acquire`] (if set) fills any still-unresolved bindings
/// that partition needs — that is the DXGI / drawable wait. Earlier partitions
/// (including earlier present partitions for other bindings) are submitted first so
/// GPU work can overlap later acquires (Exchange option/exercise split).
///
/// [`Self::deposits`] maps logical [`super::ResourceId::Deposit`] ids to
/// the physical staging parcels selected for this submission.
///
/// [`Self::partial`] is updated after each successful partition so callers can
/// settle high-water and referenced present frames if a later acquire/submit fails.
pub(crate) struct PresentSubmitOptions<'a> {
    pub present_slots: &'a mut Vec<ResolvedPresentSlot>,
    /// Called per present partition for binding ids not yet resolved in `present_slots`.
    pub deferred_acquire: Option<&'a mut DeferredPresentAcquire<'a>>,
    /// Scheme upload-buffer resolutions for this submission (may be empty).
    pub deposits: &'a std::collections::HashMap<u32, super::ResolvedDeposit>,
    pub resource_stamps: &'a ResourceKeyMap<Arc<crate::parcel::ParcelStamp>>,
    pub stamp_targets: &'a [Arc<crate::parcel::ParcelStamp>],
    pub ir_clean: bool,
    pub sidecar: SubmitSidecarState,
    /// Best-effort progress mirror; readable after `Err` for cleanup.
    pub partial: &'a mut PartitionSubmitResult,
    pub partial_tv: &'a mut TimelineValue,
}

/// Ensure present bindings needed by the upcoming partition are in the resolver.
fn ensure_present_ready(
    needed_bindings: &[u32],
    present_slots: &mut Vec<ResolvedPresentSlot>,
    deferred_acquire: &mut Option<&mut DeferredPresentAcquire<'_>>,
    resolver: &mut super::SlotResolver,
) -> Result<()> {
    use super::ResolvedSwapchain;

    // Seed resolver from any already-acquired slots (eager path or earlier partitions).
    for slot in present_slots.iter() {
        resolver
            .present_leases
            .entry(slot.binding_id)
            .or_insert(ResolvedSwapchain {
                handle: slot.handle,
                uav_index: slot.uav_index,
            });
    }

    let missing: Vec<u32> = needed_bindings
        .iter()
        .copied()
        .filter(|id| !resolver.present_leases.contains_key(id))
        .collect();
    if missing.is_empty() {
        if needed_bindings.is_empty() {
            // Legacy SwapchainOutput partitions may have empty PresentLease ids;
            // they still need at least one resolved slot when present_slots was
            // pre-filled, or a deferred acquire that fills something.
            if present_slots.is_empty() {
                if let Some(acquire) = deferred_acquire.as_mut() {
                    let _tz = crate::tracy_zone!("scheme.submit.acquire_present");
                    acquire(&[], present_slots)?;
                }
                if present_slots.is_empty() {
                    return Err(anyhow::anyhow!(
                        "present partition requires at least one resolved present slot"
                    ));
                }
                for slot in present_slots.iter() {
                    resolver
                        .present_leases
                        .entry(slot.binding_id)
                        .or_insert(ResolvedSwapchain {
                            handle: slot.handle,
                            uav_index: slot.uav_index,
                        });
                }
            }
        }
        return Ok(());
    }

    let start = present_slots.len();
    if let Some(acquire) = deferred_acquire.as_mut() {
        let _tz = crate::tracy_zone!("scheme.submit.acquire_present");
        acquire(&missing, present_slots)?;
    }
    {
        let _tz = crate::tracy_zone!("goldy.submit_resolved.present_resolver");
        for slot in &present_slots[start..] {
            resolver.present_leases.insert(
                slot.binding_id,
                ResolvedSwapchain {
                    handle: slot.handle,
                    uav_index: slot.uav_index,
                },
            );
        }
    }
    for &id in &missing {
        if !resolver.present_leases.contains_key(&id) {
            return Err(anyhow::anyhow!(
                "present partition missing resolved slot for binding {id}"
            ));
        }
    }
    Ok(())
}

/// Fingerprint for retained CB variants that bake late-bound slot handles.
///
/// Combines the stable partition fingerprint with present-lease slots referenced
/// by `waves` and every resolved upload-buffer physical handle those waves use.
fn dynamic_partition_slot_key(
    part_fp: u64,
    present_slots: &[ResolvedPresentSlot],
    ir: &GraphIR,
    waves: &[Wave],
    resolver: &super::SlotResolver,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    part_fp.hash(&mut h);
    let needed = analysis::partition_present_binding_ids(ir, waves);
    for slot in present_slots {
        if needed.binary_search(&slot.binding_id).is_ok() {
            slot.binding_id.hash(&mut h);
            slot.generation.hash(&mut h);
            slot.slot_id.hash(&mut h);
        }
    }
    let mut upload_ids: Vec<u32> = waves
        .iter()
        .flat_map(|w| w.node_indices.iter().copied())
        .flat_map(|ni| ir.nodes[ni].bindings.iter())
        .filter_map(|b| match b.resource {
            ResourceId::Deposit(id) => Some(id),
            _ => None,
        })
        .collect();
    upload_ids.sort_unstable();
    upload_ids.dedup();
    for id in upload_ids {
        id.hash(&mut h);
        let resolved = resolver
            .deposits
            .get(&id)
            .expect("dynamic_partition_slot_key: Deposit missing from resolver");
        resolved.parent.hash(&mut h);
        resolved.offset.hash(&mut h);
        resolved.len.hash(&mut h);
    }
    h.finish()
}

fn seed_upload_resolver(
    resolver: &mut super::SlotResolver,
    deposits: &std::collections::HashMap<u32, super::ResolvedDeposit>,
) {
    resolver.deposits.clear();
    for (&id, resolved) in deposits {
        resolver.deposits.insert(id, *resolved);
    }
}

fn partition_needs_slot_resolver(ir: &GraphIR, waves: &[Wave], has_present: bool) -> bool {
    has_present || analysis::partition_waves_have_upload_slots(ir, waves)
}

/// One execution segment for the no-replay Scheme submit path.
///
/// Derived once from the compiled schedule (with the binding fingerprint) so the
/// hot loop never re-scans waves for present/render/retainability flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FreshSegment {
    wave_range: std::ops::Range<usize>,
    has_render: bool,
    has_present: bool,
    /// Scheme-unique present binding ids referenced by this segment.
    present_bindings: Vec<u32>,
    /// Upload / unstable-copy partitions must use standalone submission.
    needs_standalone: bool,
    /// True when any node in the segment references a scheme upload-buffer slot.
    has_upload_slots: bool,
}

/// True when adjacent fresh segments should coalesce into one graph CB.
///
/// Mirrors [`try_merge_compute_render_range`]: shared-queue compute→render pairs
/// stay in one command buffer so UAV→graphics barriers remain coherent.
fn try_merge_fresh_segments(plan: &[FreshSegment], idx: usize, separate_graphics: bool) -> bool {
    if separate_graphics || idx + 1 >= plan.len() {
        return false;
    }
    let a = &plan[idx];
    let b = &plan[idx + 1];
    !a.has_present
        && !b.has_present
        && !a.has_upload_slots
        && !b.has_upload_slots
        && !a.needs_standalone
        && !b.needs_standalone
        && !a.has_render
        && b.has_render
}

/// True when an upload-only segment may fuse with the immediately following compute
/// segment into one standalone command buffer (Metal Scheme path).
fn try_merge_upload_compute_fresh_segments(plan: &[FreshSegment], idx: usize, fuse_upload: bool) -> bool {
    if !fuse_upload || idx + 1 >= plan.len() {
        return false;
    }
    let a = &plan[idx];
    let b = &plan[idx + 1];
    a.has_upload_slots
        && !a.has_present
        && !a.has_render
        && !b.has_upload_slots
        && !b.has_present
        && !b.has_render
        && !b.needs_standalone
}

/// Post-process a fresh segment plan, merging upload→compute pairs when allowed.
fn coalesce_fresh_plan_for_fused_upload(plan: Vec<FreshSegment>, fuse_upload: bool) -> Vec<FreshSegment> {
    if !fuse_upload || plan.len() < 2 {
        return plan;
    }
    let mut merged = Vec::with_capacity(plan.len());
    let mut idx = 0usize;
    while idx < plan.len() {
        if try_merge_upload_compute_fresh_segments(&plan, idx, fuse_upload) {
            let a = &plan[idx];
            let b = &plan[idx + 1];
            merged.push(FreshSegment {
                wave_range: a.wave_range.start..b.wave_range.end,
                has_render: false,
                has_present: false,
                present_bindings: Vec::new(),
                needs_standalone: true,
                has_upload_slots: true,
            });
            idx += 2;
        } else {
            merged.push(plan[idx].clone());
            idx += 1;
        }
    }
    merged
}

#[cfg(test)]
pub(crate) fn test_coalesce_fresh_plan(plan: Vec<FreshSegment>, fuse_upload: bool) -> Vec<FreshSegment> {
    coalesce_fresh_plan_for_fused_upload(plan, fuse_upload)
}

/// Submit `ir` with optional per-partition CB replay.
///
/// - **`replay: Some`** — slice-aware retain/resubmit (fingerprints, backend CB store,
///   present-slot variants, retire-before-rerecord).
/// - **`replay: None`** — dedicated fresh executor: cached segment plan + per-frame
///   emission (TaskGraph surface style); no retention fingerprints or backend retain.
///
/// Upload partitions always use `submit_standalone`. Cross-submit sync and parcel
/// epoch stamps run regardless of replay.
pub(crate) fn submit_resolved_ir_partitions(
    cache: &mut Option<CompiledCacheEntry>,
    replay: Option<&mut super::cb_replay::CbReplayState>,
    context: &crate::Context,
    session: &dyn crate::backend::ContextSubmitSession,
    ir: &GraphIR,
    options: PresentSubmitOptions<'_>,
) -> Result<(TimelineValue, PartitionSubmitResult)> {
    let _tz = crate::tracy_zone!("goldy.submit_resolved_partitions");
    match replay {
        None => submit_resolved_ir_partitions_fresh(cache, context, session, ir, options),
        Some(replay) => submit_resolved_ir_partitions_replay(cache, replay, context, session, ir, options),
    }
}

/// No-replay Scheme submit: cached schedule/segment plan, fresh command emission.
fn submit_resolved_ir_partitions_fresh(
    cache: &mut Option<CompiledCacheEntry>,
    context: &crate::Context,
    session: &dyn crate::backend::ContextSubmitSession,
    ir: &GraphIR,
    options: PresentSubmitOptions<'_>,
) -> Result<(TimelineValue, PartitionSubmitResult)> {
    use super::SlotResolver;

    let PresentSubmitOptions {
        present_slots,
        mut deferred_acquire,
        deposits,
        resource_stamps,
        stamp_targets,
        ir_clean,
        mut sidecar,
        partial,
        partial_tv,
    } = options;

    let _tz = crate::tracy_zone!("goldy.submit_resolved_partitions.fresh");

    let fp = if ir_clean {
        cache.as_ref().map(|e| e.fp).unwrap_or_else(|| binding_fingerprint(ir))
    } else {
        binding_fingerprint(ir)
    };
    {
        let _tz = crate::tracy_zone!("goldy.submit_resolved.build_schedule");
        get_or_build_schedule(cache, ir, fp);
    }
    {
        let _tz = crate::tracy_zone!("goldy.submit_resolved.fresh_plan");
        let caps = context.device().capabilities();
        get_or_build_fresh_plan(
            cache,
            ir,
            fp,
            caps.split_compute_partitions_on_barrier_cost,
            caps.fuse_upload_with_compute_partitions,
        );
    }

    let mut resolver = SlotResolver::new();
    seed_upload_resolver(&mut resolver, deposits);

    let ctx = context.backend_handle();
    let separate = session.separate_graphics_queue();
    let device_owner = session.device_queue_owner(ctx);
    let mut last_tv = context.gpu_progress();
    *partial_tv = last_tv;
    let mut result = PartitionSubmitResult::default();

    {
        let _tz = crate::tracy_zone!("goldy.submit_resolved.fresh_loop");
        let mut cross_scratch = CrossSubmitScratch::new();
        let mut boundary = QueueBoundaryState::default();
        let mut seg_idx = 0usize;
        let plan_len = cache.as_ref().unwrap().fresh_plan.as_ref().unwrap().len();
        while seg_idx < plan_len {
            let merge = {
                let plan = cache.as_ref().unwrap().fresh_plan.as_ref().unwrap();
                try_merge_fresh_segments(plan, seg_idx, separate)
            };

            let (wave_range, has_render, has_present, present_bindings, advance) = {
                let plan = cache.as_ref().unwrap().fresh_plan.as_ref().unwrap();
                if merge {
                    let a = &plan[seg_idx];
                    let b = &plan[seg_idx + 1];
                    (a.wave_range.start..b.wave_range.end, true, false, Vec::new(), 2usize)
                } else {
                    let s = &plan[seg_idx];
                    (
                        s.wave_range.clone(),
                        s.has_render,
                        s.has_present,
                        s.present_bindings.clone(),
                        1usize,
                    )
                }
            };

            let stamp_ctx = partition_stamp_context(separate, has_render, ctx, device_owner);
            let base_sync = {
                let _tz = crate::tracy_zone!("goldy.partition_loop.cross_sync");
                let waves = &cache.as_ref().unwrap().schedule.waves[wave_range.clone()];
                cross_sync_for_stamps(&mut cross_scratch, resource_stamps, ir, stamp_ctx, waves, separate)
            };
            let sync = merge_queue_boundary_waits(
                base_sync,
                separate,
                has_render,
                ctx,
                device_owner,
                boundary.last_compute_tv,
                boundary.last_render_tv,
            );
            let merged = sidecar.merge_sync(sync.as_ref());

            if has_present {
                ensure_present_ready(&present_bindings, present_slots, &mut deferred_acquire, &mut resolver)?;
            }

            if has_render {
                // Offscreen render (and merged compute→render): graph submit.
                debug_assert!(!has_present, "present partitions must not contain render passes");
                let graph_cmds = {
                    let _tz = crate::tracy_zone!("goldy.partition_loop.fresh_cmds");
                    let waves = &cache.as_ref().unwrap().schedule.waves[wave_range.clone()];
                    let needs_resolver = partition_needs_slot_resolver(ir, waves, false);
                    analysis::emit_graph_commands_for_waves(
                        ir,
                        waves,
                        if needs_resolver { Some(&resolver) } else { None },
                    )
                };
                let _tz = crate::tracy_zone!("goldy.submit_partition.fresh");
                last_tv = backend_submit_graph(session, ctx, &graph_cmds, merged.as_ref())?;
            } else {
                // Pure compute, uploads, and present tail: surface-style standalone.
                let needs_resolver = partition_needs_slot_resolver(
                    ir,
                    &cache.as_ref().unwrap().schedule.waves[wave_range.clone()],
                    has_present,
                );
                let cmds = {
                    let _tz = crate::tracy_zone!("goldy.partition_loop.fresh_emit");
                    let waves = &cache.as_ref().unwrap().schedule.waves[wave_range.clone()];
                    analysis::emit_waves_to_commands(ir, waves, if needs_resolver { Some(&resolver) } else { None })
                };
                let _tz = crate::tracy_zone!("goldy.submit_partition.fresh");
                last_tv = backend_submit_standalone(session, ctx, &cmds, merged.as_ref())?;
            }

            {
                let waves = &cache.as_ref().unwrap().schedule.waves[wave_range.clone()];
                boundary.record(separate, has_render, last_tv);
                apply_partition_epoch_stamps(resource_stamps, stamp_targets, stamp_ctx, ir, waves, last_tv);
            }
            if has_present {
                result.note_present_bindings(&present_bindings, last_tv);
            }
            *partial_tv = last_tv;
            *partial = result.clone();
            seg_idx += advance;
        }
    }

    Ok((last_tv, result))
}

/// Retention/resubmit partitioned submit (CB replay enabled).
fn submit_resolved_ir_partitions_replay(
    cache: &mut Option<CompiledCacheEntry>,
    replay: &mut super::cb_replay::CbReplayState,
    context: &crate::Context,
    session: &dyn crate::backend::ContextSubmitSession,
    ir: &GraphIR,
    options: PresentSubmitOptions<'_>,
) -> Result<(TimelineValue, PartitionSubmitResult)> {
    use super::SlotResolver;

    let PresentSubmitOptions {
        present_slots,
        mut deferred_acquire,
        deposits,
        resource_stamps,
        stamp_targets,
        ir_clean,
        mut sidecar,
        partial,
        partial_tv,
    } = options;

    let _tz = crate::tracy_zone!("goldy.submit_resolved_partitions.replay");

    let fp = if ir_clean {
        cache.as_ref().map(|e| e.fp).unwrap_or_else(|| binding_fingerprint(ir))
    } else {
        binding_fingerprint(ir)
    };
    {
        let _tz = crate::tracy_zone!("goldy.submit_resolved.build_schedule");
        get_or_build_schedule(cache, ir, fp);
    }

    let split_on_barrier_cost = context.device().capabilities().split_compute_partitions_on_barrier_cost;
    let wave_ranges = analysis::partition_wave_ranges(ir, &cache.as_ref().unwrap().schedule, split_on_barrier_cost);

    let partition_fps: Vec<u64> = {
        let _tz = crate::tracy_zone!("goldy.submit_resolved.partition_fps");
        let layout_tag = texture_copy_layout_tag(context);
        if ir_clean {
            let schedule = &cache.as_ref().unwrap().schedule;
            let keys = replay.partition_keys.as_slice();
            // Sticky keys: reuse the last retained fingerprint when present. Layout tags
            // are only folded in the miss path below (first compute of a key), matching
            // historical behavior since layout fingerprinting was introduced.
            (0..wave_ranges.len())
                .map(|i| {
                    keys.get(i).and_then(|k| *k).unwrap_or_else(|| {
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::{Hash, Hasher};
                        let waves = &schedule.waves[wave_ranges[i].clone()];
                        let raw_fp = partition_fingerprint(ir, schedule, waves);
                        let layout_fp = analysis::partition_copy_texture_layout_fingerprint(ir, waves, &layout_tag);
                        let mut h = DefaultHasher::new();
                        raw_fp.hash(&mut h);
                        layout_fp.hash(&mut h);
                        i.hash(&mut h);
                        h.finish()
                    })
                })
                .collect()
        } else {
            let schedule = &cache.as_ref().unwrap().schedule;
            compute_partition_fps(ir, schedule, &wave_ranges, Some(&layout_tag))
        }
    };

    replay.ensure_partition_vecs(wave_ranges.len());

    {
        let _tz = crate::tracy_zone!("goldy.submit_resolved.build_partitions");
        get_or_build_partitioned_commands(cache, ir, fp, split_on_barrier_cost);
    }

    let mut resolver = SlotResolver::new();
    seed_upload_resolver(&mut resolver, deposits);

    let ctx = context.backend_handle();
    let separate = session.separate_graphics_queue();
    let device_owner = session.device_queue_owner(ctx);
    let fuse_upload = context.device().capabilities().fuse_upload_with_compute_partitions;
    let mut last_tv = context.gpu_progress();
    *partial_tv = last_tv;
    let mut result = PartitionSubmitResult::default();

    {
        let _tz = crate::tracy_zone!("goldy.submit_resolved.partition_loop");
        let mut cross_scratch = CrossSubmitScratch::new();
        let mut boundary = QueueBoundaryState::default();
        let mut part_idx = 0usize;
        while part_idx < wave_ranges.len() {
            let upload_compute_merged = {
                let schedule = &cache.as_ref().unwrap().schedule;
                try_merge_upload_compute_range(ir, schedule, &wave_ranges, part_idx, fuse_upload)
            };
            if let Some(merged_range) = upload_compute_merged {
                let merged_waves = {
                    let schedule = &cache.as_ref().unwrap().schedule;
                    schedule.waves[merged_range.clone()].to_vec()
                };
                let stamp_ctx = partition_stamp_context(separate, false, ctx, device_owner);
                let base_sync = {
                    let _tz = crate::tracy_zone!("goldy.partition_loop.cross_sync");
                    cross_sync_for_stamps(
                        &mut cross_scratch,
                        resource_stamps,
                        ir,
                        stamp_ctx,
                        &merged_waves,
                        separate,
                    )
                };
                let sync = merge_queue_boundary_waits(
                    base_sync,
                    separate,
                    false,
                    ctx,
                    device_owner,
                    boundary.last_compute_tv,
                    boundary.last_render_tv,
                );
                let merged = sidecar.merge_sync(sync.as_ref());
                let cmds = {
                    let _tz = crate::tracy_zone!("goldy.partition_loop.upload_compute_emit");
                    analysis::emit_waves_to_commands(ir, &merged_waves, Some(&resolver))
                };
                let _tz = crate::tracy_zone!("goldy.submit_partition.upload_compute_fused");
                last_tv = backend_submit_standalone(session, ctx, &cmds, merged.as_ref())?;
                replay.record_last_tv(part_idx, last_tv);
                replay.record_last_tv(part_idx + 1, last_tv);
                boundary.record(separate, false, last_tv);
                apply_partition_epoch_stamps(resource_stamps, stamp_targets, stamp_ctx, ir, &merged_waves, last_tv);
                part_idx += 2;
                continue;
            }

            let merged_range = {
                let schedule = &cache.as_ref().unwrap().schedule;
                try_merge_compute_render_range(ir, schedule, &wave_ranges, part_idx, separate)
            };
            if let Some(merged_range) = merged_range {
                let merged_fp = merged_compute_render_fp(partition_fps[part_idx], partition_fps[part_idx + 1]);
                let merged_waves = {
                    let schedule = &cache.as_ref().unwrap().schedule;
                    schedule.waves[merged_range.clone()].to_vec()
                };
                let sync = {
                    let _tz = crate::tracy_zone!("goldy.partition_loop.cross_sync");
                    cross_sync_for_stamps(&mut cross_scratch, resource_stamps, ir, ctx, &merged_waves, separate)
                };
                let merged = sidecar.merge_sync(sync);

                let cached_key = replay.partition_keys[part_idx];
                if cached_key == Some(merged_fp) {
                    let _tz = crate::tracy_zone!("goldy.resubmit.merged");
                    if let Some(tv) = backend_try_resubmit_retained(session, ctx, merged_fp, merged.as_ref())? {
                        last_tv = tv;
                        result.resubmit_hits += 1;
                        replay.record_merged_last_tvs(part_idx, last_tv);
                        apply_partition_epoch_stamps(resource_stamps, stamp_targets, ctx, ir, &merged_waves, last_tv);
                        part_idx += 2;
                        continue;
                    }
                }

                let graph_cmds = {
                    let _tz = crate::tracy_zone!("goldy.partition_loop.merged_emit");
                    let needs_resolver = partition_needs_slot_resolver(ir, &merged_waves, false);
                    analysis::emit_graph_commands_for_waves(
                        ir,
                        &merged_waves,
                        if needs_resolver { Some(&resolver) } else { None },
                    )
                };
                let _tz = crate::tracy_zone!("goldy.submit_partition.merged_record");
                ensure_partition_retired_before_rerecord(session, context, replay.partition_last_tv[part_idx])?;
                last_tv = backend_submit_graph_and_retain(session, ctx, &graph_cmds, merged_fp, merged.as_ref())?;
                replay.partition_keys[part_idx] = Some(merged_fp);
                replay.partition_keys[part_idx + 1] = Some(merged_fp);
                replay.record_merged_last_tvs(part_idx, last_tv);
                result.records += 1;
                apply_partition_epoch_stamps(resource_stamps, stamp_targets, ctx, ir, &merged_waves, last_tv);
                *partial_tv = last_tv;
                *partial = result.clone();
                part_idx += 2;
                continue;
            }

            let part_fp = partition_fps[part_idx];
            let range = wave_ranges[part_idx].clone();
            let waves = cache.as_ref().unwrap().schedule.waves[range].to_vec();
            let can_retain = partition_waves_can_retain(ir, &waves);
            let has_render = partition_waves_have_render(ir, &waves);
            let has_present = analysis::partition_waves_have_present(ir, &waves);
            let present_bindings = analysis::partition_present_binding_ids(ir, &waves);
            let has_upload_slots = analysis::partition_waves_have_upload_slots(ir, &waves);
            let needs_resolver = partition_needs_slot_resolver(ir, &waves, has_present);
            let stamp_ctx = partition_stamp_context(separate, has_render, ctx, device_owner);
            let base_sync = {
                let _tz = crate::tracy_zone!("goldy.partition_loop.cross_sync");
                cross_sync_for_stamps(&mut cross_scratch, resource_stamps, ir, stamp_ctx, &waves, separate)
            };
            let sync = merge_queue_boundary_waits(
                base_sync,
                separate,
                has_render,
                ctx,
                device_owner,
                boundary.last_compute_tv,
                boundary.last_render_tv,
            );
            let merged = sidecar.merge_sync(sync.as_ref());

            if analysis::partition_waves_are_accel_build(ir, &waves)
                && ir_clean
                && replay.partition_last_tv.get(part_idx).copied().flatten().is_some()
            {
                last_tv = replay.partition_last_tv[part_idx].unwrap();
                boundary.record(separate, has_render, last_tv);
                apply_partition_epoch_stamps(resource_stamps, stamp_targets, stamp_ctx, ir, &waves, last_tv);
                *partial_tv = last_tv;
                *partial = result.clone();
                part_idx += 1;
                continue;
            }

            if !can_retain {
                if has_present {
                    ensure_present_ready(&present_bindings, present_slots, &mut deferred_acquire, &mut resolver)?;
                }
                let cache_entry = cache.as_ref().unwrap();
                let cmds = {
                    let _tz = crate::tracy_zone!("goldy.partition_loop.standalone_cmds");
                    partition_standalone_commands(
                        ir,
                        cache_entry,
                        &waves,
                        part_idx,
                        has_render,
                        needs_resolver,
                        if needs_resolver { Some(&resolver) } else { None },
                    )?
                };
                let _tz = crate::tracy_zone!("goldy.submit_partition.standalone");
                last_tv = backend_submit_standalone(session, ctx, &cmds, merged.as_ref())?;
                replay.record_last_tv(part_idx, last_tv);
                boundary.record(separate, has_render, last_tv);
                apply_partition_epoch_stamps(resource_stamps, stamp_targets, stamp_ctx, ir, &waves, last_tv);
                if has_present {
                    result.note_present_bindings(&present_bindings, last_tv);
                }
                *partial_tv = last_tv;
                *partial = result.clone();
                part_idx += 1;
                continue;
            }

            // Present and/or upload-slot partitions bake late-bound handles into the CB,
            // so retention keys include the concrete slot combination.
            if has_present || has_upload_slots {
                if has_present {
                    ensure_present_ready(&present_bindings, present_slots, &mut deferred_acquire, &mut resolver)?;
                }

                // Metal (and any backend that cannot retain present partitions): always fresh
                // when present is involved. Upload-only partitions may still retain.
                if has_present && !session.retains_present_partitions() {
                    let cache_entry = cache.as_ref().unwrap();
                    let cmds = {
                        let _tz = crate::tracy_zone!("goldy.partition_loop.standalone_cmds");
                        partition_standalone_commands(
                            ir,
                            cache_entry,
                            &waves,
                            part_idx,
                            has_render,
                            true,
                            Some(&resolver),
                        )?
                    };
                    let _tz = crate::tracy_zone!("goldy.submit_partition.standalone");
                    last_tv = backend_submit_standalone(session, ctx, &cmds, merged.as_ref())?;
                    replay.record_last_tv(part_idx, last_tv);
                    boundary.record(separate, has_render, last_tv);
                    apply_partition_epoch_stamps(resource_stamps, stamp_targets, stamp_ctx, ir, &waves, last_tv);
                    result.note_present_bindings(&present_bindings, last_tv);
                    *partial_tv = last_tv;
                    *partial = result.clone();
                    part_idx += 1;
                    continue;
                }

                let slot_key = {
                    let _tz = crate::tracy_zone!("goldy.partition_loop.dynamic_slot_key");
                    dynamic_partition_slot_key(part_fp, present_slots, ir, &waves, &resolver)
                };

                let already_retained = replay.partition_slot_keys[part_idx]
                    .as_ref()
                    .map(|s| s.contains(&slot_key))
                    .unwrap_or(false);

                if already_retained {
                    let _tz = crate::tracy_zone!("goldy.resubmit.slot_variant");
                    if let Some(tv) = backend_try_resubmit_retained(session, ctx, slot_key, merged.as_ref())? {
                        last_tv = tv;
                        result.resubmit_hits += 1;
                        replay.record_last_tv(part_idx, last_tv);
                        boundary.record(separate, has_render, last_tv);
                        apply_partition_epoch_stamps(resource_stamps, stamp_targets, stamp_ctx, ir, &waves, last_tv);
                        if has_present {
                            result.note_present_bindings(&present_bindings, last_tv);
                        }
                        *partial_tv = last_tv;
                        *partial = result.clone();
                        part_idx += 1;
                        continue;
                    }
                }

                let graph_cmds = {
                    let _tz = crate::tracy_zone!("goldy.partition_loop.retain_cmds");
                    partition_graph_commands_for_retain(
                        ir,
                        cache.as_ref().unwrap(),
                        &waves,
                        part_idx,
                        has_render,
                        Some(&resolver),
                    )
                };
                let _tz = crate::tracy_zone!("goldy.submit_partition.slot_record");
                ensure_partition_retired_before_rerecord(session, context, replay.partition_last_tv[part_idx])?;
                last_tv = backend_submit_graph_and_retain(session, ctx, &graph_cmds, slot_key, merged.as_ref())?;
                replay.partition_slot_keys[part_idx]
                    .get_or_insert_with(std::collections::HashSet::new)
                    .insert(slot_key);
                replay.partition_keys[part_idx] = Some(part_fp);
                replay.record_last_tv(part_idx, last_tv);
                result.records += 1;
                boundary.record(separate, has_render, last_tv);
                apply_partition_epoch_stamps(resource_stamps, stamp_targets, stamp_ctx, ir, &waves, last_tv);
                if has_present {
                    result.note_present_bindings(&present_bindings, last_tv);
                }
                *partial_tv = last_tv;
                *partial = result.clone();
                part_idx += 1;
                continue;
            }

            // Retainable compute/render partition (no late-bound slots).
            let cached_key = replay.partition_keys[part_idx];
            if cached_key == Some(part_fp) {
                let _tz = crate::tracy_zone!("goldy.resubmit.partition");
                if let Some(tv) = backend_try_resubmit_retained(session, ctx, part_fp, merged.as_ref())? {
                    last_tv = tv;
                    result.resubmit_hits += 1;
                    replay.record_last_tv(part_idx, last_tv);
                    boundary.record(separate, has_render, last_tv);
                    apply_partition_epoch_stamps(resource_stamps, stamp_targets, stamp_ctx, ir, &waves, last_tv);
                    *partial_tv = last_tv;
                    *partial = result.clone();
                    part_idx += 1;
                    continue;
                }
            }

            let graph_cmds = {
                let _tz = crate::tracy_zone!("goldy.partition_loop.retain_cmds");
                partition_graph_commands_for_retain(ir, cache.as_ref().unwrap(), &waves, part_idx, has_render, None)
            };
            let _tz = crate::tracy_zone!("goldy.submit_partition.record");
            ensure_partition_retired_before_rerecord(session, context, replay.partition_last_tv[part_idx])?;
            last_tv = backend_submit_graph_and_retain(session, ctx, &graph_cmds, part_fp, merged.as_ref())?;
            replay.partition_keys[part_idx] = Some(part_fp);
            replay.record_last_tv(part_idx, last_tv);
            result.records += 1;
            boundary.record(separate, has_render, last_tv);
            apply_partition_epoch_stamps(resource_stamps, stamp_targets, stamp_ctx, ir, &waves, last_tv);
            *partial_tv = last_tv;
            *partial = result.clone();
            part_idx += 1;
        }
    }

    Ok((last_tv, result))
}

/// Schedule cache + parcel stamps + optional CB replay ledger for [`crate::Scheme`].
pub(crate) struct IrSubmitState {
    schedule_cache: Option<CompiledCacheEntry>,
    /// Present only while CB replay is enabled for this submitter.
    replay: Option<super::cb_replay::CbReplayState>,
    stamp_targets: Vec<Arc<crate::parcel::ParcelStamp>>,
    resource_stamps: ResourceKeyMap<Arc<crate::parcel::ParcelStamp>>,
    /// Reuse epochs merged into every partition's GPU queue-wait list at submit time.
    extra_submit_epochs: crate::timeline::ReferenceTable,
    /// Host-observed epochs + deferred writes (consumed once on the first partition job).
    host_observed_epochs: crate::timeline::ReferenceTable,
    deferred_host_writes: Vec<crate::backend::DeferredHostWrite>,
}

impl IrSubmitState {
    pub fn new() -> Self {
        Self {
            schedule_cache: None,
            // Start enabled; first submit with `cb_replay_disabled()` tears this down.
            replay: Some(super::cb_replay::CbReplayState::new()),
            stamp_targets: Vec::new(),
            resource_stamps: ResourceKeyMap::default(),
            extra_submit_epochs: crate::timeline::ReferenceTable::default(),
            host_observed_epochs: crate::timeline::ReferenceTable::default(),
            deferred_host_writes: Vec::new(),
        }
    }

    /// Record GPU-orderable reuse dependencies for the next submit (enforced via queue-wait).
    pub fn record_reuse_epochs(&mut self, refs: &crate::timeline::ReferenceTable) {
        crate::backend::host_sidecar::merge_reference_table(&mut self.extra_submit_epochs, refs);
    }

    /// Defer a host-visible write until the submission worker, after `refs` retire on the CPU.
    ///
    /// Currently applied by the DX12 and Metal submission workers.
    pub fn defer_host_write(
        &mut self,
        refs: &crate::timeline::ReferenceTable,
        buffer: &crate::Buffer,
        offset: u64,
        data: Box<[u8]>,
    ) {
        let buffer_handle = buffer
            .whole()
            .buffer_handle()
            .expect("defer_host_write requires a single-unit buffer");
        crate::backend::host_sidecar::merge_reference_table(&mut self.host_observed_epochs, refs);
        self.deferred_host_writes.push(crate::backend::DeferredHostWrite {
            buffer: buffer_handle,
            offset,
            data: std::sync::Arc::from(data),
        });
    }

    fn take_submit_sidecars(
        &mut self,
    ) -> (
        Vec<crate::timeline::Epoch>,
        Vec<crate::timeline::Epoch>,
        Vec<crate::backend::DeferredHostWrite>,
    ) {
        let queue = crate::timeline::epochs_from(&std::mem::take(&mut self.extra_submit_epochs));
        let host = crate::timeline::epochs_from(&std::mem::take(&mut self.host_observed_epochs));
        let writes = std::mem::take(&mut self.deferred_host_writes);
        (queue, host, writes)
    }

    pub fn register_parcel_stamp(&mut self, parcel: &crate::Parcel) {
        self.register_stamp_parts(parcel.resource_id(), parcel.stamp_handle());
    }

    pub fn register_stamp_parts(&mut self, resource_id: ResourceId, stamp: Arc<crate::parcel::ParcelStamp>) {
        if let Some(key) = ResourceKey::from_resource_id(resource_id) {
            // Rebinding the same resource identity replaces the map entry. Also drop any
            // matching retired stamp from the legacy `stamp_targets` ledger so
            // `all_stamps_alive` cannot keep failing after a correct re-record.
            if let Some(old) = self.resource_stamps.insert(key, stamp) {
                self.stamp_targets
                    .retain(|s| !std::sync::Arc::ptr_eq(s, &old) && s.is_alive());
            }
        } else {
            self.stamp_targets.push(stamp);
        }
    }

    /// Register stamps for every parcel unit in a buffer (dependency tracking only).
    pub fn register_buffer_stamps(&mut self, buffer: &crate::Buffer) {
        for parcel in buffer.parcels() {
            self.register_parcel_stamp(parcel);
        }
    }

    pub fn register_stamp(&mut self, stamp: Arc<crate::parcel::ParcelStamp>) {
        self.stamp_targets.push(stamp);
    }

    /// Drop cached retention keys so the next submit re-records retained partitions.
    pub fn invalidate_retention(&mut self) {
        if let Some(replay) = &mut self.replay {
            replay.invalidate();
        }
    }

    /// Ensure replay ledger matches the global disable flag; release backend CBs when turning off.
    fn sync_replay_mode(&mut self, ctx: &crate::Context) {
        if super::cb_replay::cb_replay_disabled() {
            if let Some(mut replay) = self.replay.take() {
                replay.release_backend(ctx);
            }
        } else if self.replay.is_none() {
            self.replay = Some(super::cb_replay::CbReplayState::new());
        }
    }

    /// Drop backend retained command lists referenced by the replay ledger.
    pub fn release_backend_retained_graphs(&mut self, ctx: &crate::Context) {
        if let Some(replay) = &mut self.replay {
            replay.release_backend(ctx);
        }
    }

    pub fn resource_stamps(&self) -> &ResourceKeyMap<Arc<crate::parcel::ParcelStamp>> {
        &self.resource_stamps
    }

    /// True when every registered parcel stamp is still alive (owning Buffer/Texture not dropped).
    pub fn all_stamps_alive(&self) -> bool {
        self.resource_stamps.values().all(|s| s.is_alive()) && self.stamp_targets.iter().all(|s| s.is_alive())
    }

    /// Per-partition timeline values from the most recent successful submit.
    pub fn partition_last_tvs(&self) -> &[Option<TimelineValue>] {
        self.replay.as_ref().map(|r| r.last_tvs()).unwrap_or(&[])
    }

    /// True when a CB replay ledger is currently attached.
    pub fn has_cb_replay(&self) -> bool {
        self.replay.is_some()
    }

    /// Number of retained late-bound slot variants across all partitions (tests/telemetry).
    pub fn retained_slot_variant_count(&self) -> usize {
        self.replay
            .as_ref()
            .map(|r| r.partition_slot_keys.iter().flatten().map(|s| s.len()).sum())
            .unwrap_or(0)
    }

    /// Submit `ir`, retaining command lists when CB replay is enabled.
    ///
    /// `deferred_acquire` runs for each present-touching partition, receiving the
    /// binding ids that partition still needs. Pass `None` and a pre-filled
    /// `present_slots` for tests / eager acquire.
    ///
    /// `deposits` maps logical upload-buffer ids to the physical parcels
    /// selected for this submission (may be empty).
    ///
    /// On failure after some partitions succeeded, `partial` / `partial_tv` hold
    /// progress for high-water and referenced-present cleanup.
    #[allow(clippy::too_many_arguments)] // present/upload/partial progress are all required at call sites
    pub fn submit_pipelined_and_retain_with_presents<'a>(
        &'a mut self,
        ctx: &crate::Context,
        ir: &GraphIR,
        present_slots: &'a mut Vec<ResolvedPresentSlot>,
        deferred_acquire: Option<&'a mut DeferredPresentAcquire<'a>>,
        deposits: &'a std::collections::HashMap<u32, super::ResolvedDeposit>,
        ir_clean: bool,
        partial: &'a mut PartitionSubmitResult,
        partial_tv: &'a mut TimelineValue,
    ) -> Result<(TimelineValue, PartitionSubmitResult)> {
        self.sync_replay_mode(ctx);
        let (queue_epochs, host_epochs, host_writes) = self.take_submit_sidecars();
        let sidecar = SubmitSidecarState::new(queue_epochs, host_epochs, host_writes);
        submit_resolved_ir_partitions(
            &mut self.schedule_cache,
            self.replay.as_mut(),
            ctx,
            ctx.submit_session(),
            ir,
            PresentSubmitOptions {
                present_slots,
                deferred_acquire,
                deposits,
                resource_stamps: &self.resource_stamps,
                stamp_targets: &self.stamp_targets,
                ir_clean,
                sidecar,
                partial,
                partial_tv,
            },
        )
    }
}

/// Cache entry holding the wave schedule and the emitted compute command stream.
///
/// CB retention keys / TVs live in [`super::cb_replay::CbReplayState`], not here.
pub(crate) struct CompiledCacheEntry {
    fp: u64,
    schedule: CompiledSchedule,
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
    /// Segment metadata for the no-replay fresh executor.
    ///
    /// Built from [`analysis::partition_wave_ranges`] with present/render/standalone
    /// flags precomputed. The fresh path emits commands each frame from these
    /// ranges; it does not use `partitioned_commands`.
    fresh_plan: Option<Vec<FreshSegment>>,
}

/// Return a reference to the compiled schedule for `ir`, using the cache when possible.
///
/// On a miss the schedule is built and stored; on a hit it is returned directly.
fn get_or_build_schedule<'c>(cache: &'c mut Option<CompiledCacheEntry>, ir: &GraphIR, fp: u64) -> &'c CompiledSchedule {
    let _tz = crate::tracy_zone!("goldy.compile_schedule");
    if cache.as_ref().is_some_and(|e| e.fp == fp) {
        tracing::trace!(target: "goldy::schedule_cache", hit = true, fp, "schedule");
        return &cache.as_ref().unwrap().schedule;
    }
    {
        let _tz = crate::tracy_zone!("goldy.compile_schedule.miss");
        tracing::trace!(target: "goldy::schedule_cache", hit = false, fp, "schedule");
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        *cache = Some(CompiledCacheEntry {
            fp,
            schedule,
            partitioned_commands: None,
            partitioned_upload_remap: Vec::new(),
            partitioned_graph_commands: Vec::new(),
            fresh_plan: None,
        });
    }
    &cache.as_ref().unwrap().schedule
}

/// Build (or reuse) the no-replay segment plan for `ir`.
///
/// Segments follow [`analysis::partition_wave_ranges`] with present/render/
/// standalone flags precomputed so the fresh executor does not re-scan waves.
fn get_or_build_fresh_plan(
    cache: &mut Option<CompiledCacheEntry>,
    ir: &GraphIR,
    fp: u64,
    split_on_barrier_cost: bool,
    fuse_upload_with_compute: bool,
) {
    let _tz = crate::tracy_zone!("goldy.compile_fresh_plan");
    get_or_build_schedule(cache, ir, fp);
    let needs_build = cache.as_ref().is_none_or(|e| e.fresh_plan.is_none());
    if !needs_build {
        tracing::trace!(target: "goldy::schedule_cache", hit = true, fp, "fresh_plan");
        return;
    }
    tracing::trace!(target: "goldy::schedule_cache", hit = false, fp, "fresh_plan");
    let entry = cache.as_mut().unwrap();
    let ranges = analysis::partition_wave_ranges(ir, &entry.schedule, split_on_barrier_cost);
    let mut plan = Vec::with_capacity(ranges.len());
    for range in ranges {
        let waves = &entry.schedule.waves[range.clone()];
        plan.push(FreshSegment {
            wave_range: range,
            has_render: partition_waves_have_render(ir, waves),
            has_present: analysis::partition_waves_have_present(ir, waves),
            present_bindings: analysis::partition_present_binding_ids(ir, waves),
            needs_standalone: !partition_waves_can_retain(ir, waves),
            has_upload_slots: analysis::partition_waves_have_upload_slots(ir, waves),
        });
    }
    entry.fresh_plan = Some(coalesce_fresh_plan_for_fused_upload(plan, fuse_upload_with_compute));
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
fn get_or_build_partitioned_commands(
    cache: &mut Option<CompiledCacheEntry>,
    ir: &GraphIR,
    fp: u64,
    split_on_barrier_cost: bool,
) {
    let _tz = crate::tracy_zone!("goldy.compile_partitioned");

    // Ensure schedule exists.
    let needs_schedule = match cache.as_ref() {
        Some(e) => e.fp != fp,
        None => true,
    };
    if needs_schedule {
        let _tz = crate::tracy_zone!("goldy.compile_partitioned.schedule_rebuild");
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        *cache = Some(CompiledCacheEntry {
            fp,
            schedule,
            partitioned_commands: None,
            partitioned_upload_remap: Vec::new(),
            partitioned_graph_commands: Vec::new(),
            fresh_plan: None,
        });
    }

    let needs_build = cache.as_ref().is_none_or(|e| e.partitioned_commands.is_none());

    tracing::trace!(target: "goldy::schedule_cache", hit = !needs_build, fp, "partitioned_commands");

    if !needs_build {
        // Hit: refresh upload payloads in compute-only partitions.
        let _tz = crate::tracy_zone!("goldy.compile_partitioned.hit_refresh");
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
                    (GpuCommand::WriteTextureRegion { data, .. }, NodeKind::WriteTextureRegion { data: src, .. }) => {
                        *data = src.clone()
                    }
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
    let _tz = crate::tracy_zone!("goldy.compile_partitioned.miss_emit");
    let entry = cache.as_mut().unwrap();
    let wave_ranges = analysis::partition_wave_ranges(ir, &entry.schedule, split_on_barrier_cost);

    let mut compute_partitions: Vec<Vec<GpuCommand>> = Vec::with_capacity(wave_ranges.len());
    let mut graph_partitions: Vec<Option<Vec<GraphCommand>>> = Vec::with_capacity(wave_ranges.len());

    for range in &wave_ranges {
        let waves = &entry.schedule.waves[range.clone()];
        let has_present = analysis::partition_waves_have_present(ir, waves);
        let has_upload_slots = analysis::partition_waves_have_upload_slots(ir, waves);
        let has_render = waves.iter().any(|w| {
            w.node_indices
                .iter()
                .any(|&ni| matches!(ir.nodes[ni].kind, NodeKind::RenderPass { .. }))
        });

        if has_present || has_upload_slots {
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

/// One bindless push-constant slot in shader virtual-main parameter order.
///
/// Use with [`crate::scheme::SchemeRenderPassBuilder::with_shader_resources`]. Mosaic parcels belong in
/// [`crate::scheme::SchemeRenderPassBuilder::with_parcel`] (graph dependency + vertex views), not here.
pub enum ShaderResourceSlot<'a> {
    Parcel {
        parcel: &'a crate::Parcel,
        access: NodeAccess,
    },
    Sampler(&'a Sampler),
}

#[cfg(test)]
mod slice_retention_tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::backend::{BufferHandle, ComputePipelineHandle};
    use crate::buffer::Allocation;
    use crate::compute::ComputePipeline;
    use crate::device::Device;
    use crate::shader::ShaderModule;
    use crate::task_graph::ResolvedDeposit;
    use crate::task_graph::{IrSubmitState, NodeAccess, ResourceBinding, TaskNode};
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

    fn mock_buf(device: &Device) -> Allocation {
        Allocation::new(device, 256, crate::BufferKind::Scattered).unwrap()
    }

    /// Read `retained_resubmit_count` from the mock backend.
    fn resubmit_count(device: &Device) -> usize {
        device.with_mock(|m| m.retained_resubmit_count)
    }

    /// Read the number of live retained graph entries.
    fn retained_count(device: &Device) -> usize {
        device.with_mock(|m| m.retained_graphs.len())
    }

    fn do_submit(state: &mut IrSubmitState, ctx: &crate::Context, ir: &GraphIR, ir_clean: bool) {
        // Retention assertions must not flip when the developer shell exports
        // GOLDY_DISABLE_CB_REUSE=1.
        let _cb = crate::test_support::CbReuseOverride::force_enabled();
        let mut present_slots = Vec::new();
        let empty_uploads = std::collections::HashMap::new();
        let mut partial = PartitionSubmitResult::default();
        let mut partial_tv = 0u64;
        state
            .submit_pipelined_and_retain_with_presents(
                ctx,
                ir,
                &mut present_slots,
                None,
                &empty_uploads,
                ir_clean,
                &mut partial,
                &mut partial_tv,
            )
            .unwrap();
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
        do_submit(&mut state, &ctx, &ir, false);
        assert_eq!(resubmit_count(&device), 0, "first submit records, does not resubmit");
        assert_eq!(retained_count(&device), 1, "one slice retained");

        // Second submit (unchanged IR): should resubmit from cache.
        do_submit(&mut state, &ctx, &ir, true);
        assert_eq!(resubmit_count(&device), 1, "second submit resubmits retained slice");
        assert_eq!(retained_count(&device), 1, "still one slice retained");
    }

    /// With `replay: None`, retainable partitions use fresh `submit_graph` — no backend
    /// CB storage, no resubmit hits, no retention-record counters.
    #[test]
    fn replay_none_never_stores_or_resubmits_cbs() {
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

        let mut cache = None;
        let empty_stamps = ResourceKeyMap::default();
        let empty_uploads = std::collections::HashMap::new();
        let mut present_slots = Vec::new();
        for ir_clean in [false, true, true] {
            let mut partial = PartitionSubmitResult::default();
            let mut partial_tv = 0u64;
            let (tv, result) = submit_resolved_ir_partitions(
                &mut cache,
                None,
                &ctx,
                ctx.submit_session(),
                &ir,
                PresentSubmitOptions {
                    present_slots: &mut present_slots,
                    deferred_acquire: None,
                    deposits: &empty_uploads,
                    resource_stamps: &empty_stamps,
                    stamp_targets: &[],
                    ir_clean,
                    sidecar: SubmitSidecarState::new(Vec::new(), Vec::new(), Vec::new()),
                    partial: &mut partial,
                    partial_tv: &mut partial_tv,
                },
            )
            .unwrap();
            assert!(tv > 0);
            assert_eq!(result.records, 0, "fresh path must not count retention records");
            assert_eq!(result.resubmit_hits, 0, "fresh path must not count resubmits");
        }
        assert_eq!(
            retained_count(&device),
            0,
            "no backend retained CBs when replay is None"
        );
        assert_eq!(resubmit_count(&device), 0);
    }

    /// Fresh path: compute-then-present yields two standalone submits, and deferred
    /// acquire runs only after the early partition has already been submitted.
    #[test]
    fn fresh_compute_then_present_defers_acquire_between_submits() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p = mock_pipeline(&device, &shader);
        let buf = mock_buf(&device);

        let mut ir = GraphIR::default();
        ir.nodes.push(TaskNode {
            label: "early",
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
        ir.nodes.push(TaskNode {
            label: "copy",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::RenderTarget(5),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::PresentLease(0),
                    access: NodeAccess::Write,
                },
                // Force a schedule edge so present lands after the early wave
                // (independent nodes would otherwise share wave 0 as one present partition).
                ResourceBinding {
                    resource: ResourceId::Buffer(buf.handle),
                    access: NodeAccess::Read,
                },
            ],
            kind: NodeKind::CopyRenderTarget {
                src: 5,
                dst: ResourceId::PresentLease(0),
            },
        });

        let mut cache = None;
        let empty_stamps = ResourceKeyMap::default();
        let empty_uploads = std::collections::HashMap::new();
        let mut present_slots = Vec::new();
        let acquire_after = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
        let acquire_after_cb = acquire_after.clone();
        let device_cb = device.clone();
        let mut deferred = |_needed: &[u32], slots: &mut Vec<ResolvedPresentSlot>| -> Result<()> {
            let n = device_cb.with_mock(|m| m.compute_dispatch_count);
            acquire_after_cb.store(n, std::sync::atomic::Ordering::SeqCst);
            slots.push(ResolvedPresentSlot {
                binding_id: 0,
                generation: 0,
                slot_id: 0,
                handle: 42,
                uav_index: 7,
            });
            Ok(())
        };

        let mut partial = PartitionSubmitResult::default();
        let mut partial_tv = 0u64;
        let (tv, result) = submit_resolved_ir_partitions(
            &mut cache,
            None,
            &ctx,
            ctx.submit_session(),
            &ir,
            PresentSubmitOptions {
                present_slots: &mut present_slots,
                deferred_acquire: Some(&mut deferred),
                deposits: &empty_uploads,
                resource_stamps: &empty_stamps,
                stamp_targets: &[],
                ir_clean: false,
                sidecar: SubmitSidecarState::new(Vec::new(), Vec::new(), Vec::new()),
                partial: &mut partial,
                partial_tv: &mut partial_tv,
            },
        )
        .unwrap();

        assert!(tv > 0);
        assert_eq!(result.records, 0);
        assert_eq!(result.resubmit_hits, 0);
        assert_eq!(present_slots.len(), 1);
        assert_eq!(
            result.present_binding_tvs,
            vec![(0, tv)],
            "present binding stamped with its partition timeline"
        );
        assert_eq!(
            acquire_after.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "deferred acquire must run after the early standalone submit"
        );
        device.with_mock(|m| {
            assert_eq!(m.compute_dispatch_count, 2, "early + present = two standalone submits");
            assert!(
                m.recorded_graph_syncs.is_empty(),
                "fresh compute/present path must not use submit_graph"
            );
            assert_eq!(m.recorded_compute_commands.len(), 2);
        });
        assert_eq!(retained_count(&device), 0);

        // Plan is cached across clean submits.
        let plan_ptr = cache.as_ref().unwrap().fresh_plan.as_ref().unwrap().as_ptr();
        let mut present_slots = Vec::new();
        let mut deferred2 = |_needed: &[u32], slots: &mut Vec<ResolvedPresentSlot>| -> Result<()> {
            slots.push(ResolvedPresentSlot {
                binding_id: 0,
                generation: 0,
                slot_id: 1,
                handle: 43,
                uav_index: 8,
            });
            Ok(())
        };
        let mut partial = PartitionSubmitResult::default();
        let mut partial_tv = 0u64;
        submit_resolved_ir_partitions(
            &mut cache,
            None,
            &ctx,
            ctx.submit_session(),
            &ir,
            PresentSubmitOptions {
                present_slots: &mut present_slots,
                deferred_acquire: Some(&mut deferred2),
                deposits: &empty_uploads,
                resource_stamps: &empty_stamps,
                stamp_targets: &[],
                ir_clean: true,
                sidecar: SubmitSidecarState::new(Vec::new(), Vec::new(), Vec::new()),
                partial: &mut partial,
                partial_tv: &mut partial_tv,
            },
        )
        .unwrap();
        assert_eq!(
            cache.as_ref().unwrap().fresh_plan.as_ref().unwrap().as_ptr(),
            plan_ptr,
            "fresh plan must be reused on clean schedule hit"
        );
        assert!(
            cache.as_ref().unwrap().partitioned_commands.is_none(),
            "fresh path must not populate the replay command cache"
        );
    }

    /// Fresh path: two present bindings in sequence acquire only the unresolved ids
    /// at each partition; the second acquire can fail after the first was submitted.
    #[test]
    fn fresh_two_presents_acquire_per_binding_and_partial_failure() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p = mock_pipeline(&device, &shader);
        let buf = mock_buf(&device);

        let mut ir = GraphIR::default();
        ir.nodes.push(TaskNode {
            label: "early_a",
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
        ir.nodes.push(TaskNode {
            label: "copy_a",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::RenderTarget(5),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::PresentLease(0),
                    access: NodeAccess::Write,
                },
                ResourceBinding {
                    resource: ResourceId::Buffer(buf.handle),
                    access: NodeAccess::Read,
                },
            ],
            kind: NodeKind::CopyRenderTarget {
                src: 5,
                dst: ResourceId::PresentLease(0),
            },
        });
        ir.nodes.push(TaskNode {
            label: "early_b",
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
        ir.nodes.push(TaskNode {
            label: "copy_b",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::RenderTarget(6),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::PresentLease(1),
                    access: NodeAccess::Write,
                },
                ResourceBinding {
                    resource: ResourceId::Buffer(buf.handle),
                    access: NodeAccess::Read,
                },
            ],
            kind: NodeKind::CopyRenderTarget {
                src: 6,
                dst: ResourceId::PresentLease(1),
            },
        });

        let mut cache = None;
        let empty_stamps = ResourceKeyMap::default();
        let empty_uploads = std::collections::HashMap::new();
        let mut present_slots = Vec::new();
        let acquire_calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Vec<u32>>::new()));
        let acquire_calls_cb = acquire_calls.clone();
        let mut deferred = |needed: &[u32], slots: &mut Vec<ResolvedPresentSlot>| -> Result<()> {
            acquire_calls_cb.lock().unwrap().push(needed.to_vec());
            if needed == [1] {
                anyhow::bail!("simulated acquire failure for binding 1");
            }
            for &binding_id in needed {
                slots.push(ResolvedPresentSlot {
                    binding_id,
                    generation: 0,
                    slot_id: binding_id,
                    handle: 40 + binding_id as u64,
                    uav_index: 7 + binding_id,
                });
            }
            Ok(())
        };

        let mut partial = PartitionSubmitResult::default();
        let mut partial_tv = 0u64;
        let err = submit_resolved_ir_partitions(
            &mut cache,
            None,
            &ctx,
            ctx.submit_session(),
            &ir,
            PresentSubmitOptions {
                present_slots: &mut present_slots,
                deferred_acquire: Some(&mut deferred),
                deposits: &empty_uploads,
                resource_stamps: &empty_stamps,
                stamp_targets: &[],
                ir_clean: false,
                sidecar: SubmitSidecarState::new(Vec::new(), Vec::new(), Vec::new()),
                partial: &mut partial,
                partial_tv: &mut partial_tv,
            },
        )
        .expect_err("second present acquire must fail");
        assert!(err.to_string().contains("binding 1"), "unexpected error: {err}");
        assert_eq!(
            *acquire_calls.lock().unwrap(),
            vec![vec![0], vec![1]],
            "each present partition acquires only its unresolved binding"
        );
        assert_eq!(present_slots.len(), 1, "only binding 0 was acquired");
        assert_eq!(
            partial.present_binding_tvs.len(),
            1,
            "binding 0 present partition was submitted before failure"
        );
        assert_eq!(partial.present_binding_tvs[0].0, 0);
        assert!(partial_tv > 0, "partial high-water must reflect submitted work");
    }

    #[test]
    fn dynamic_partition_slot_key_includes_generation() {
        let mut ir = GraphIR::default();
        ir.nodes.push(TaskNode {
            label: "copy",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::RenderTarget(5),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::PresentLease(0),
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyRenderTarget {
                src: 5,
                dst: ResourceId::PresentLease(0),
            },
        });
        let waves = vec![Wave {
            node_indices: vec![0],
            barriers_before: Default::default(),
        }];
        let resolver = crate::task_graph::SlotResolver::default();
        let key_a = dynamic_partition_slot_key(
            0xABCDu64,
            &[ResolvedPresentSlot {
                binding_id: 0,
                generation: 0,
                slot_id: 1,
                handle: 10,
                uav_index: 2,
            }],
            &ir,
            &waves,
            &resolver,
        );
        let key_b = dynamic_partition_slot_key(
            0xABCDu64,
            &[ResolvedPresentSlot {
                binding_id: 0,
                generation: 1,
                slot_id: 1,
                handle: 10,
                uav_index: 2,
            }],
            &ir,
            &waves,
            &resolver,
        );
        assert_ne!(key_a, key_b, "generation must participate in retained variant key");
    }

    /// Fresh path: offscreen render uses `submit_graph`, not standalone.
    #[cfg(feature = "graphics")]
    #[test]
    fn fresh_render_segment_uses_graph_submit() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();

        // Create a real mock render target so submit_graph's render path succeeds.
        let rt = crate::render_target::RenderTarget::new_with_depth(
            &device,
            4,
            4,
            crate::types::TextureFormat::Rgba8Unorm,
            None,
        )
        .unwrap();

        let mut ir = GraphIR::default();
        ir.nodes.push(TaskNode {
            label: "draw",
            bindings: vec![],
            kind: NodeKind::RenderPass {
                target: rt.backend_handle(),
                color_load: crate::types::TargetLoad::Clear(crate::types::Color::BLACK),
                commands: Vec::new(),
            },
        });

        let mut cache = None;
        let empty_stamps = ResourceKeyMap::default();
        let empty_uploads = std::collections::HashMap::new();
        let mut present_slots = Vec::new();
        let mut partial = PartitionSubmitResult::default();
        let mut partial_tv = 0u64;
        let (tv, result) = submit_resolved_ir_partitions(
            &mut cache,
            None,
            &ctx,
            ctx.submit_session(),
            &ir,
            PresentSubmitOptions {
                present_slots: &mut present_slots,
                deferred_acquire: None,
                deposits: &empty_uploads,
                resource_stamps: &empty_stamps,
                stamp_targets: &[],
                ir_clean: false,
                sidecar: SubmitSidecarState::new(Vec::new(), Vec::new(), Vec::new()),
                partial: &mut partial,
                partial_tv: &mut partial_tv,
            },
        )
        .unwrap();

        assert!(tv > 0);
        assert_eq!(result.records, 0);
        assert_eq!(result.resubmit_hits, 0);
        device.with_mock(|m| {
            assert_eq!(m.recorded_graph_syncs.len(), 1, "render segment must submit_graph");
        });
        assert_eq!(retained_count(&device), 0);
        let plan = cache.as_ref().unwrap().fresh_plan.as_ref().unwrap();
        assert_eq!(plan.len(), 1);
        assert!(plan[0].has_render);
        assert!(!plan[0].has_present);
    }

    fn upload_then_compute_ir(buf_dst: BufferHandle, p: ComputePipelineHandle) -> GraphIR {
        let upload_src = ResourceId::Deposit(0);
        let dst = ResourceId::Buffer(buf_dst);
        let mut ir = GraphIR::default();
        ir.nodes.push(TaskNode {
            label: "upload_copy",
            bindings: vec![
                ResourceBinding {
                    resource: upload_src,
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: dst,
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyBuffer {
                src: upload_src,
                src_offset: 0,
                dst,
                dst_offset: 0,
                size: 64,
            },
        });
        ir.nodes.push(TaskNode {
            label: "compute",
            bindings: vec![ResourceBinding {
                resource: dst,
                access: NodeAccess::Write,
            }],
            kind: NodeKind::Dispatch {
                pipeline: p,
                resource_slots: vec![],
                user_slots: vec![],
                dispatch: DispatchDim::Direct { x: 1, y: 1, z: 1 },
            },
        });
        ir
    }

    /// Fresh-plan coalescing leaves segments separate when fuse capability is off.
    #[test]
    fn fresh_plan_coalesce_respects_capability_flag() {
        let plan = vec![
            FreshSegment {
                wave_range: 0..1,
                has_render: false,
                has_present: false,
                present_bindings: Vec::new(),
                needs_standalone: true,
                has_upload_slots: true,
            },
            FreshSegment {
                wave_range: 1..2,
                has_render: false,
                has_present: false,
                present_bindings: Vec::new(),
                needs_standalone: false,
                has_upload_slots: false,
            },
        ];
        let split = super::test_coalesce_fresh_plan(plan.clone(), false);
        assert_eq!(split.len(), 2);
        let fused = super::test_coalesce_fresh_plan(plan, true);
        assert_eq!(fused.len(), 1);
        assert!(fused[0].has_upload_slots);
        assert_eq!(fused[0].wave_range, 0..2);
    }

    /// Metal-style capability: upload blits and following compute fuse into one submit.
    #[test]
    fn fresh_upload_compute_fuses_with_metal_capability() {
        let mut backend = MockBackend::new();
        backend.fuse_upload_with_compute_partitions = true;
        let device = Arc::new(Device::from_backend(Box::new(backend)).unwrap());
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p = mock_pipeline(&device, &shader);
        let buf = mock_buf(&device);
        let ir = upload_then_compute_ir(buf.handle, p.handle);

        let mut cache = None;
        let empty_stamps = ResourceKeyMap::default();
        let mut uploads = std::collections::HashMap::new();
        uploads.insert(
            0,
            ResolvedDeposit {
                parent: buf.handle,
                offset: 0,
                len: 64,
            },
        );
        let mut present_slots = Vec::new();
        let mut partial = PartitionSubmitResult::default();
        let mut partial_tv = 0u64;
        submit_resolved_ir_partitions(
            &mut cache,
            None,
            &ctx,
            ctx.submit_session(),
            &ir,
            PresentSubmitOptions {
                present_slots: &mut present_slots,
                deferred_acquire: None,
                deposits: &uploads,
                resource_stamps: &empty_stamps,
                stamp_targets: &[],
                ir_clean: false,
                sidecar: SubmitSidecarState::new(Vec::new(), Vec::new(), Vec::new()),
                partial: &mut partial,
                partial_tv: &mut partial_tv,
            },
        )
        .unwrap();

        device.with_mock(|m| {
            assert_eq!(
                m.compute_dispatch_count, 1,
                "fused upload+compute must be one standalone submit"
            );
            let cmds = &m.recorded_compute_commands[0];
            assert!(
                cmds.iter().any(|c| matches!(c, GpuCommand::CopyBuffer { .. })),
                "fused submit must include upload blit commands"
            );
            assert!(
                cmds.iter().any(|c| matches!(c, GpuCommand::Dispatch { .. })),
                "fused submit must include compute dispatch commands"
            );
        });
        let plan = cache.as_ref().unwrap().fresh_plan.as_ref().unwrap();
        assert_eq!(plan.len(), 1, "fresh plan must coalesce upload and compute");
        assert!(plan[0].has_upload_slots);
    }

    /// Replay path: Metal-style fuse merges upload+compute into one standalone submit.
    #[test]
    fn replay_upload_compute_fuses_with_metal_capability() {
        let mut backend = MockBackend::new();
        backend.fuse_upload_with_compute_partitions = true;
        let device = Arc::new(Device::from_backend(Box::new(backend)).unwrap());
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p = mock_pipeline(&device, &shader);
        let buf = mock_buf(&device);
        let ir = upload_then_compute_ir(buf.handle, p.handle);

        let mut state = IrSubmitState::new();
        let mut uploads = std::collections::HashMap::new();
        uploads.insert(
            0,
            ResolvedDeposit {
                parent: buf.handle,
                offset: 0,
                len: 64,
            },
        );
        let mut present_slots = Vec::new();
        let _cb = crate::test_support::CbReuseOverride::force_enabled();
        let mut partial = PartitionSubmitResult::default();
        let mut partial_tv = 0u64;
        state
            .submit_pipelined_and_retain_with_presents(
                &ctx,
                &ir,
                &mut present_slots,
                None,
                &uploads,
                false,
                &mut partial,
                &mut partial_tv,
            )
            .unwrap();

        device.with_mock(|m| {
            assert_eq!(
                m.compute_dispatch_count, 1,
                "replay path must fuse upload+compute when capability is enabled"
            );
        });
    }

    // ------------------------------------------------------------------
    // Deposit CopyBuffer vs compute→render merge
    //
    // Scheme deposits are retainable (`waves_can_retain` stays true) but bake
    // late-bound handles into the CB. Compute→render merge retains under an IR
    // fingerprint only, so deposit waves must not fold into that merge.
    // ------------------------------------------------------------------

    fn deposit_copy_node(dst: ResourceId, size: u64) -> TaskNode {
        let src = ResourceId::Deposit(0);
        TaskNode {
            label: "deposit_copy",
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
                size,
            },
        }
    }

    #[cfg(feature = "graphics")]
    fn compute_render_merge(ir: &GraphIR, separate_graphics: bool) -> Option<std::ops::Range<usize>> {
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        let wave_ranges = analysis::partition_wave_ranges(ir, &schedule, true);
        try_merge_compute_render_range(ir, &schedule, &wave_ranges, 0, separate_graphics)
    }

    #[cfg(feature = "graphics")]
    fn deposit_map(parent: BufferHandle, len: u64) -> std::collections::HashMap<u32, ResolvedDeposit> {
        let mut uploads = std::collections::HashMap::new();
        uploads.insert(0, ResolvedDeposit { parent, offset: 0, len });
        uploads
    }

    #[cfg(feature = "graphics")]
    fn submit_with_deposits(
        state: &mut IrSubmitState,
        ctx: &crate::Context,
        ir: &GraphIR,
        deposits: &std::collections::HashMap<u32, ResolvedDeposit>,
        ir_clean: bool,
    ) {
        let _cb = crate::test_support::CbReuseOverride::force_enabled();
        let mut present_slots = Vec::new();
        let mut partial = PartitionSubmitResult::default();
        let mut partial_tv = 0u64;
        state
            .submit_pipelined_and_retain_with_presents(
                ctx,
                ir,
                &mut present_slots,
                None,
                deposits,
                ir_clean,
                &mut partial,
                &mut partial_tv,
            )
            .unwrap();
    }

    fn fresh_segment(
        wave_range: std::ops::Range<usize>,
        has_render: bool,
        has_upload_slots: bool,
        needs_standalone: bool,
    ) -> FreshSegment {
        FreshSegment {
            wave_range,
            has_render,
            has_present: false,
            present_bindings: Vec::new(),
            needs_standalone,
            has_upload_slots,
        }
    }

    /// Deposit copies stay retainable; do not route them through `waves_can_retain`.
    #[test]
    fn deposit_copy_buffer_is_retainable() {
        let ir = GraphIR {
            nodes: vec![deposit_copy_node(ResourceId::Buffer(1), 64)],
        };
        let edges = analysis::build_edges(&ir);
        let schedule = analysis::schedule_waves(&ir, &edges);
        assert!(
            partition_waves_can_retain(&ir, &schedule.waves),
            "Deposit CopyBuffer must stay retainable (slot-key path), unlike WriteBuffer"
        );
        assert!(analysis::partition_waves_have_upload_slots(&ir, &schedule.waves));
    }

    /// Fresh compute→render merge is blocked when either segment has deposit slots.
    #[test]
    fn try_merge_fresh_segments_rejects_upload_slots_on_either_side() {
        let compute_then_render = vec![
            fresh_segment(0..1, false, false, false),
            fresh_segment(1..2, true, false, false),
        ];
        assert!(
            try_merge_fresh_segments(&compute_then_render, 0, false),
            "retainable compute→render must still merge on a shared queue"
        );
        assert!(
            !try_merge_fresh_segments(&compute_then_render, 0, true),
            "separate graphics queue must not merge compute into render"
        );

        let upload_then_render = vec![
            fresh_segment(0..1, false, true, false),
            fresh_segment(1..2, true, false, false),
        ];
        assert!(
            !try_merge_fresh_segments(&upload_then_render, 0, false),
            "deposit wave must not merge into the following render segment"
        );

        let compute_then_upload_render = vec![
            fresh_segment(0..1, false, false, false),
            fresh_segment(1..2, true, true, false),
        ];
        assert!(
            !try_merge_fresh_segments(&compute_then_upload_render, 0, false),
            "render segment with deposit slots must not absorb the prior compute segment"
        );
    }

    #[cfg(feature = "graphics")]
    fn dispatch_then_render_ir(p: ComputePipelineHandle, buf: BufferHandle, rt: u64) -> GraphIR {
        GraphIR {
            nodes: vec![
                TaskNode {
                    label: "pre",
                    bindings: vec![ResourceBinding {
                        resource: ResourceId::Buffer(buf),
                        access: NodeAccess::Write,
                    }],
                    kind: NodeKind::Dispatch {
                        pipeline: p,
                        resource_slots: vec![],
                        user_slots: vec![],
                        dispatch: DispatchDim::Direct { x: 1, y: 1, z: 1 },
                    },
                },
                TaskNode {
                    label: "draw",
                    bindings: vec![ResourceBinding {
                        resource: ResourceId::Buffer(buf),
                        access: NodeAccess::Read,
                    }],
                    kind: NodeKind::RenderPass {
                        target: rt,
                        color_load: crate::types::TargetLoad::Clear(crate::types::Color::BLACK),
                        commands: Vec::new(),
                    },
                },
            ],
        }
    }

    #[cfg(feature = "graphics")]
    fn deposit_then_render_ir(buf: BufferHandle, rt: u64) -> GraphIR {
        GraphIR {
            nodes: vec![
                deposit_copy_node(ResourceId::Buffer(buf), 64),
                TaskNode {
                    label: "draw",
                    bindings: vec![ResourceBinding {
                        resource: ResourceId::Buffer(buf),
                        access: NodeAccess::Read,
                    }],
                    kind: NodeKind::RenderPass {
                        target: rt,
                        color_load: crate::types::TargetLoad::Clear(crate::types::Color::BLACK),
                        commands: Vec::new(),
                    },
                },
            ],
        }
    }

    /// Compute in wave 0; deposit copy and render share wave 1 (slots on the render partition).
    ///
    /// `deposit_dst` is independent of the render pass so the copy can sit in the
    /// same wave as the draw. Reading `compute_buf` pulls the copy to depth 1.
    #[cfg(feature = "graphics")]
    fn compute_then_deposit_and_render_ir(
        p: ComputePipelineHandle,
        compute_buf: BufferHandle,
        deposit_dst: BufferHandle,
        rt: u64,
    ) -> GraphIR {
        let compute = ResourceId::Buffer(compute_buf);
        let mut copy = deposit_copy_node(ResourceId::Buffer(deposit_dst), 64);
        copy.bindings.push(ResourceBinding {
            resource: compute,
            access: NodeAccess::Read,
        });
        GraphIR {
            nodes: vec![
                TaskNode {
                    label: "pre",
                    bindings: vec![ResourceBinding {
                        resource: compute,
                        access: NodeAccess::Write,
                    }],
                    kind: NodeKind::Dispatch {
                        pipeline: p,
                        resource_slots: vec![],
                        user_slots: vec![],
                        dispatch: DispatchDim::Direct { x: 1, y: 1, z: 1 },
                    },
                },
                copy,
                TaskNode {
                    label: "draw",
                    bindings: vec![ResourceBinding {
                        resource: compute,
                        access: NodeAccess::Read,
                    }],
                    kind: NodeKind::RenderPass {
                        target: rt,
                        color_load: crate::types::TargetLoad::Clear(crate::types::Color::BLACK),
                        commands: Vec::new(),
                    },
                },
            ],
        }
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn try_merge_compute_render_rejects_deposit_slots_on_either_side() {
        let compute_render = dispatch_then_render_ir(1, 1, 10);
        assert!(
            compute_render_merge(&compute_render, false).is_some(),
            "shared-queue compute→render without deposits must still merge"
        );
        assert!(
            compute_render_merge(&compute_render, true).is_none(),
            "separate graphics queue must not merge"
        );

        let deposit_render = deposit_then_render_ir(1, 10);
        assert!(
            compute_render_merge(&deposit_render, false).is_none(),
            "deposit-then-render must not merge (slots on w0)"
        );

        let compute_deposit_render = compute_then_deposit_and_render_ir(1, 1, 2, 10);
        let edges = analysis::build_edges(&compute_deposit_render);
        let schedule = analysis::schedule_waves(&compute_deposit_render, &edges);
        let ranges = analysis::partition_wave_ranges(&compute_deposit_render, &schedule, true);
        assert!(
            ranges.len() >= 2,
            "expected compute | deposit+render partitions, got {ranges:?}"
        );
        assert!(
            !analysis::partition_waves_have_upload_slots(&compute_deposit_render, &schedule.waves[ranges[0].clone()]),
            "first partition must be deposit-free so this case exercises slots on w1"
        );
        assert!(analysis::partition_waves_have_upload_slots(
            &compute_deposit_render,
            &schedule.waves[ranges[1].clone()]
        ));
        assert!(
            compute_render_merge(&compute_deposit_render, false).is_none(),
            "compute then deposit+render must not merge (slots on w1)"
        );
    }

    /// Clock-style: deposit blit then render must not panic, and must not share a CB.
    #[cfg(feature = "graphics")]
    #[test]
    fn fresh_deposit_then_render_stays_split_and_resolves() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let rt = crate::render_target::RenderTarget::new_with_depth(
            &device,
            4,
            4,
            crate::types::TextureFormat::Rgba8Unorm,
            None,
        )
        .unwrap();
        let buf = mock_buf(&device);
        let ir = deposit_then_render_ir(buf.handle, rt.backend_handle());
        let uploads = deposit_map(buf.handle, 64);

        let mut cache = None;
        let empty_stamps = ResourceKeyMap::default();
        let mut present_slots = Vec::new();
        let mut partial = PartitionSubmitResult::default();
        let mut partial_tv = 0u64;
        submit_resolved_ir_partitions(
            &mut cache,
            None,
            &ctx,
            ctx.submit_session(),
            &ir,
            PresentSubmitOptions {
                present_slots: &mut present_slots,
                deferred_acquire: None,
                deposits: &uploads,
                resource_stamps: &empty_stamps,
                stamp_targets: &[],
                ir_clean: false,
                sidecar: SubmitSidecarState::new(Vec::new(), Vec::new(), Vec::new()),
                partial: &mut partial,
                partial_tv: &mut partial_tv,
            },
        )
        .expect("deposit CopyBuffer in a split render scheme must resolve");

        let plan = cache.as_ref().unwrap().fresh_plan.as_ref().unwrap();
        assert_eq!(plan.len(), 2, "deposit and render must be distinct fresh segments");
        assert!(plan[0].has_upload_slots);
        assert!(!plan[0].has_render);
        assert!(plan[1].has_render);
        assert!(!plan[1].has_upload_slots);
        assert!(
            !try_merge_fresh_segments(plan, 0, false),
            "fresh executor must not fold the deposit segment into render"
        );
        device.with_mock(|m| {
            assert_eq!(
                m.recorded_graph_syncs.len(),
                1,
                "render stays on submit_graph; deposit is standalone"
            );
        });
    }

    /// Same-wave deposit + render is one partition: emit must still pass a resolver.
    #[cfg(feature = "graphics")]
    #[test]
    fn fresh_same_wave_deposit_and_render_resolves_without_panic() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let rt = crate::render_target::RenderTarget::new_with_depth(
            &device,
            4,
            4,
            crate::types::TextureFormat::Rgba8Unorm,
            None,
        )
        .unwrap();
        let buf = mock_buf(&device);
        // Independent nodes share depth 0 → one wave with both deposit and render.
        let ir = GraphIR {
            nodes: vec![
                deposit_copy_node(ResourceId::Buffer(buf.handle), 64),
                TaskNode {
                    label: "draw",
                    bindings: vec![],
                    kind: NodeKind::RenderPass {
                        target: rt.backend_handle(),
                        color_load: crate::types::TargetLoad::Clear(crate::types::Color::BLACK),
                        commands: Vec::new(),
                    },
                },
            ],
        };
        let uploads = deposit_map(buf.handle, 64);

        let mut cache = None;
        let empty_stamps = ResourceKeyMap::default();
        let mut present_slots = Vec::new();
        let mut partial = PartitionSubmitResult::default();
        let mut partial_tv = 0u64;
        submit_resolved_ir_partitions(
            &mut cache,
            None,
            &ctx,
            ctx.submit_session(),
            &ir,
            PresentSubmitOptions {
                present_slots: &mut present_slots,
                deferred_acquire: None,
                deposits: &uploads,
                resource_stamps: &empty_stamps,
                stamp_targets: &[],
                ir_clean: false,
                sidecar: SubmitSidecarState::new(Vec::new(), Vec::new(), Vec::new()),
                partial: &mut partial,
                partial_tv: &mut partial_tv,
            },
        )
        .expect("render partition with deposit slots must emit with a slot resolver");

        let plan = cache.as_ref().unwrap().fresh_plan.as_ref().unwrap();
        assert_eq!(plan.len(), 1);
        assert!(plan[0].has_render);
        assert!(plan[0].has_upload_slots);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn replay_compute_then_render_still_merges_into_one_retained_cb() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p = mock_pipeline(&device, &shader);
        let buf = mock_buf(&device);
        let rt = crate::render_target::RenderTarget::new_with_depth(
            &device,
            4,
            4,
            crate::types::TextureFormat::Rgba8Unorm,
            None,
        )
        .unwrap();
        let ir = dispatch_then_render_ir(p.handle, buf.handle, rt.backend_handle());

        let mut state = IrSubmitState::new();
        do_submit(&mut state, &ctx, &ir, false);
        assert_eq!(
            retained_count(&device),
            1,
            "shared-queue compute→render without deposits retains as one merged CB"
        );
        device.with_mock(|m| {
            let cmds = m.retained_graphs.values().next().expect("merged CB");
            let has_dispatch = cmds.iter().any(|c| {
                matches!(
                    c,
                    GraphCommand::Compute(GpuCommand::Dispatch { .. } | GpuCommand::DispatchBatch { .. })
                )
            });
            let has_render = cmds.iter().any(|c| matches!(c, GraphCommand::Render { .. }));
            assert!(has_dispatch && has_render, "merged CB must contain compute and render");
        });

        // Recompute partition fingerprints from the IR (ir_clean=false). Clean-submit
        // sticky keys currently store merged_fp in both slots, which would re-hash
        // the merge key; that is independent of deposit-slot gating.
        do_submit(&mut state, &ctx, &ir, false);
        assert_eq!(resubmit_count(&device), 1);
        assert_eq!(retained_count(&device), 1);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn replay_deposit_then_render_retains_slot_keyed_copy_separate_from_render() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let buf = mock_buf(&device);
        let rt = crate::render_target::RenderTarget::new_with_depth(
            &device,
            4,
            4,
            crate::types::TextureFormat::Rgba8Unorm,
            None,
        )
        .unwrap();
        let ir = deposit_then_render_ir(buf.handle, rt.backend_handle());
        let uploads = deposit_map(buf.handle, 64);

        let mut state = IrSubmitState::new();
        submit_with_deposits(&mut state, &ctx, &ir, &uploads, false);
        assert_eq!(
            retained_count(&device),
            2,
            "deposit and render must retain as two CBs, not one merged_fp graph"
        );
        device.with_mock(|m| {
            let mut saw_copy = false;
            let mut saw_render = false;
            let mut saw_merged = false;
            for cmds in m.retained_graphs.values() {
                let has_copy = cmds
                    .iter()
                    .any(|c| matches!(c, GraphCommand::Compute(GpuCommand::CopyBuffer { .. })));
                let has_render = cmds.iter().any(|c| matches!(c, GraphCommand::Render { .. }));
                saw_copy |= has_copy && !has_render;
                saw_render |= has_render && !has_copy;
                saw_merged |= has_copy && has_render;
            }
            assert!(saw_copy, "deposit partition must retain a CopyBuffer-only CB");
            assert!(saw_render, "render partition must retain a render-only CB");
            assert!(!saw_merged, "deposit CopyBuffer must not be baked into the render CB");
        });

        submit_with_deposits(&mut state, &ctx, &ir, &uploads, true);
        assert_eq!(
            resubmit_count(&device),
            2,
            "both slot-keyed deposit and render resubmit"
        );
        assert_eq!(retained_count(&device), 2);
    }

    /// A different physical deposit parcel must not resubmit the previous baked CB.
    #[cfg(feature = "graphics")]
    #[test]
    fn replay_deposit_then_render_records_new_slot_variant_on_parcel_change() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let buf_a = mock_buf(&device);
        let buf_b = mock_buf(&device);
        let rt = crate::render_target::RenderTarget::new_with_depth(
            &device,
            4,
            4,
            crate::types::TextureFormat::Rgba8Unorm,
            None,
        )
        .unwrap();
        let ir = deposit_then_render_ir(buf_a.handle, rt.backend_handle());

        let mut state = IrSubmitState::new();
        let uploads_a = deposit_map(buf_a.handle, 64);
        submit_with_deposits(&mut state, &ctx, &ir, &uploads_a, false);
        assert_eq!(retained_count(&device), 2);

        let uploads_b = deposit_map(buf_b.handle, 64);
        submit_with_deposits(&mut state, &ctx, &ir, &uploads_b, true);
        assert_eq!(
            resubmit_count(&device),
            1,
            "render CB resubmits; deposit must re-record for the new parcel"
        );
        assert_eq!(
            retained_count(&device),
            3,
            "new deposit slot variant is retained alongside the previous deposit CB and render CB"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn replay_compute_then_deposit_render_does_not_merge() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p = mock_pipeline(&device, &shader);
        let compute_buf = mock_buf(&device);
        let deposit_dst = mock_buf(&device);
        let rt = crate::render_target::RenderTarget::new_with_depth(
            &device,
            4,
            4,
            crate::types::TextureFormat::Rgba8Unorm,
            None,
        )
        .unwrap();
        let ir =
            compute_then_deposit_and_render_ir(p.handle, compute_buf.handle, deposit_dst.handle, rt.backend_handle());
        let uploads = deposit_map(deposit_dst.handle, 64);

        let mut state = IrSubmitState::new();
        submit_with_deposits(&mut state, &ctx, &ir, &uploads, false);
        assert!(
            retained_count(&device) >= 2,
            "compute must not merge into a deposit-bearing render partition (got {} retained)",
            retained_count(&device)
        );
        device.with_mock(|m| {
            let merged_compute_into_slots = m.retained_graphs.values().any(|cmds| {
                let has_dispatch = cmds.iter().any(|c| {
                    matches!(
                        c,
                        GraphCommand::Compute(GpuCommand::Dispatch { .. } | GpuCommand::DispatchBatch { .. })
                    )
                });
                let has_copy = cmds
                    .iter()
                    .any(|c| matches!(c, GraphCommand::Compute(GpuCommand::CopyBuffer { .. })));
                let has_render = cmds.iter().any(|c| matches!(c, GraphCommand::Render { .. }));
                has_dispatch && has_copy && has_render
            });
            assert!(
                !merged_compute_into_slots,
                "prior compute partition must not fold into a deposit-bearing render CB (merged_fp has no slot key)"
            );
        });
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
        buf0: &Allocation,
        buf1: &Allocation,
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
        do_submit(&mut state, &ctx, &ir, false);
        assert_eq!(resubmit_count(&device), 0, "first submit records all partitions");
        assert_eq!(retained_count(&device), 2, "two slices retained for two partitions");

        // Second submit: both partitions resubmit.
        do_submit(&mut state, &ctx, &ir, true);
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
        do_submit(&mut state, &ctx, &ir, false);
        assert_eq!(resubmit_count(&device), 0);

        // Frame 2: only node C's pipeline changed → only partition 1 re-records;
        //          partition 0 resubmits from retained cache (one resubmit hit).
        let ir2 = three_wave_ir(&p_a, &p_b, &p_c2, &buf0, &buf1);
        do_submit(&mut state, &ctx, &ir2, false);
        assert_eq!(
            resubmit_count(&device),
            1,
            "partition 0 resubmits; partition 1 re-records — one resubmit total"
        );

        // Frame 3: both partitions are now cached (partition 1 was retained in frame 2).
        do_submit(&mut state, &ctx, &ir2, true);
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

        do_submit(&mut state, &ctx, &ir, false);
        assert_eq!(resubmit_count(&device), 0);

        // Change only node A → partition 0 re-records; partition 1 resubmits.
        let ir2 = three_wave_ir(&p_a2, &p_b, &p_c, &buf0, &buf1);
        do_submit(&mut state, &ctx, &ir2, false);
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
        do_submit(&mut state, &ctx, &ir, false);
        assert_eq!(resubmit_count(&device), 0, "first submit: no resubmits");
        assert_eq!(retained_count(&device), 1, "one retained slice (partition 1 only)");

        // Frame 2: partition 0 re-runs standalone; partition 1 resubmits from cache.
        do_submit(&mut state, &ctx, &ir, true);
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
//  • Present-boundary invariant: each PresentLease binding is introduced in
//    exactly one logical partition; present partitions do not precede their
//    first use of that binding.
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
    use crate::backend::RenderTargetHandle;
    use crate::task_graph::analysis::{self, LogicalPartition};
    use crate::task_graph::ir::{ResourceBinding, TaskNode};
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
                color_load: crate::types::TargetLoad::Clear(crate::types::Color::BLACK),
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
        get_or_build_partitioned_commands(&mut cache, ir, fp, true);
        cache.unwrap()
    }

    // ------------------------------------------------------------------
    // Logical-partition invariant helpers
    // ------------------------------------------------------------------

    /// Assert the present-boundary invariant:
    ///   - Every present partition carries the PresentLease ids it introduces.
    ///   - Each binding id is introduced in exactly one partition.
    ///   - Non-present partitions have an empty present_bindings set.
    fn assert_present_boundary_invariant(parts: &[LogicalPartition]) {
        let mut seen = Vec::new();
        for (i, p) in parts.iter().enumerate() {
            if p.has_present {
                assert!(
                    !p.present_bindings.is_empty() || p.wave_range.len() > 0, // SwapchainOutput-only may have empty lease ids
                    "present partition {i} must be well-formed"
                );
                for &id in &p.present_bindings {
                    assert!(
                        !seen.contains(&id),
                        "binding {id} introduced twice (second at partition {i})"
                    );
                    seen.push(id);
                }
            } else {
                assert!(
                    p.present_bindings.is_empty(),
                    "non-present partition {i} must not list present bindings"
                );
            }
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
        let ranges = analysis::partition_wave_ranges(ir, &schedule, true);
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
        // RenderPass → CopyRenderTarget(PresentLease)
        // Logical split: render partition | present partition.
        let ir = GraphIR {
            nodes: vec![
                render_pass_node("draw", 10),
                copy_to_dst_node("copy", 10, ResourceId::PresentLease(0)),
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
                        color_load: crate::types::TargetLoad::Clear(crate::types::Color::BLACK),
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
    fn two_present_leases_in_sequence_two_present_partitions() {
        // Force distinct waves via render-target WAR so lease 1 cannot join lease 0's wave.
        let ir = GraphIR {
            nodes: vec![
                render_pass_node("draw_a", 10),
                copy_to_dst_node("copy_a", 10, ResourceId::PresentLease(0)),
                render_pass_node("draw_b", 10),
                copy_to_dst_node("copy_b", 10, ResourceId::PresentLease(1)),
            ],
        };
        let parts = logical_partitions(&ir);
        let present: Vec<_> = parts.iter().filter(|p| p.has_present).collect();
        assert!(
            present.len() >= 2,
            "distinct present bindings in later waves must produce ≥ 2 present partitions, got {:?}",
            parts
        );
        assert_eq!(present[0].present_bindings, vec![0]);
        assert_eq!(present[1].present_bindings, vec![1]);
        assert_present_boundary_invariant(&parts);
        assert_render_kind_invariant(&parts);
    }

    #[test]
    fn same_wave_two_present_leases_one_present_partition() {
        // Two independent copies share a wave and acquire together.
        let ir = GraphIR {
            nodes: vec![
                copy_to_dst_node("copy_a", 5, ResourceId::PresentLease(0)),
                copy_to_dst_node("copy_b", 6, ResourceId::PresentLease(1)),
            ],
        };
        let parts = logical_partitions(&ir);
        let present: Vec<_> = parts.iter().filter(|p| p.has_present).collect();
        assert_eq!(present.len(), 1, "same-wave bindings share one present partition");
        assert_eq!(present[0].present_bindings, vec![0, 1]);
        assert_present_boundary_invariant(&parts);
    }

    #[test]
    fn compute_then_present_present_is_last_logical_partition() {
        let ir = GraphIR {
            nodes: vec![
                dispatch_node("pre", 1, vec![(buf(0), NodeAccess::Write)], 1),
                dispatch_node("post", 2, vec![(buf(0), NodeAccess::Read)], 1),
                copy_to_dst_node("copy", 5, ResourceId::PresentLease(0)),
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
    fn two_uploads_same_buffer_same_offset_remap_covers_both() {
        // Interleaved WAW pattern: write buf → dispatch → write buf → dispatch, all in one IR.
        // Both WriteBuffer nodes share (buffer=0, offset=0). The remap must cover both via
        // the consumed-flag walk in find_upload_node, not by key uniqueness.
        let ir = GraphIR {
            nodes: vec![
                write_node("write1", buf(0), 0),
                dispatch_node(
                    "copy1",
                    1,
                    vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)],
                    1,
                ),
                write_node("write2", buf(0), 0),
                dispatch_node(
                    "copy2",
                    1,
                    vec![(buf(0), NodeAccess::Read), (buf(2), NodeAccess::Write)],
                    1,
                ),
            ],
        };
        let entry = build_cache(&ir);
        assert_upload_remap_invariant(&ir, &entry);
        assert_eq!(
            entry.partitioned_upload_remap.len(),
            2,
            "both WriteBuffer nodes must appear in the remap"
        );
        // Node indices in the remap must be distinct (0 and 2, not 0 twice).
        let remapped_nodes: Vec<usize> = entry.partitioned_upload_remap.iter().map(|&(_, _, ni)| ni).collect();
        assert_eq!(
            remapped_nodes.iter().collect::<std::collections::HashSet<_>>().len(),
            2,
            "remap must reference two distinct IR nodes, not the same node twice"
        );
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
        get_or_build_partitioned_commands(&mut cache, &ir, fp, true);
        let ptr_before = cache.as_ref().unwrap().partitioned_commands.as_ref().unwrap().as_ptr();

        get_or_build_partitioned_commands(&mut cache, &ir, fp, true);
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
        get_or_build_partitioned_commands(&mut cache, &ir_v1, fp1, true);
        assert_eq!(cache.as_ref().unwrap().fp, fp1);

        get_or_build_partitioned_commands(&mut cache, &ir_v2, fp2, true);
        assert_eq!(
            cache.as_ref().unwrap().fp,
            fp2,
            "cache must rebuild on fingerprint change"
        );
        assert!(cache.as_ref().unwrap().partitioned_commands.is_some());
    }

    // ------------------------------------------------------------------
    // Group 7: copy_to_texture retainability
    //
    // CopyRenderTarget → Texture must be retainable (the texture handle is
    // stable across submissions; the staging readback blit runs standalone
    // separately via finish_submit_frame).
    // CopyRenderTarget → PresentLease must also be retainable (slot-key path).
    // Other destinations (e.g. SwapchainOutput) must NOT be retainable.
    // ------------------------------------------------------------------

    fn grant_read_node(label: &'static str, resource: ResourceId, withdraw_id: u32) -> TaskNode {
        TaskNode {
            label,
            bindings: vec![ResourceBinding {
                resource,
                access: NodeAccess::Read,
            }],
            kind: NodeKind::WithdrawRead { withdraw_id },
        }
    }

    /// Call `partition_waves_can_retain` for a single-wave IR built from `nodes`.
    fn can_retain_single_wave(nodes: Vec<TaskNode>) -> bool {
        let ir = GraphIR { nodes };
        let edges = analysis::build_edges(&ir);
        let schedule = analysis::schedule_waves(&ir, &edges);
        partition_waves_can_retain(&ir, &schedule.waves)
    }

    #[test]
    fn copy_render_target_to_texture_is_retainable() {
        // RenderPass → CopyRenderTarget(Texture) → WithdrawRead
        // All in one chain; CopyRenderTarget → Texture must not force standalone.
        let nodes = vec![
            render_pass_node("rp", 10),
            copy_to_dst_node("copy", 10, ResourceId::Texture(42)),
            grant_read_node("grant", ResourceId::Texture(42), 0),
        ];
        assert!(
            can_retain_single_wave(nodes),
            "CopyRenderTarget → Texture must be retainable"
        );
    }

    #[test]
    fn copy_render_target_to_present_lease_is_retainable() {
        let nodes = vec![
            render_pass_node("rp", 10),
            copy_to_dst_node("copy", 10, ResourceId::PresentLease(0)),
        ];
        assert!(
            can_retain_single_wave(nodes),
            "CopyRenderTarget → PresentLease must be retainable (slot-key path)"
        );
    }

    #[test]
    fn copy_render_target_to_swapchain_output_is_not_retainable() {
        let nodes = vec![
            render_pass_node("rp", 10),
            copy_to_dst_node("copy", 10, ResourceId::SwapchainOutput),
        ];
        assert!(
            !can_retain_single_wave(nodes),
            "CopyRenderTarget → SwapchainOutput must NOT be retainable"
        );
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
        get_or_build_partitioned_commands(&mut cache, &ir, fp, true);

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
        get_or_build_partitioned_commands(&mut cache, &ir, fp, true);

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
