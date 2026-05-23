//! `TaskGraph` — analyzed GPU task graph with automatic barrier insertion.

use super::analysis;
use super::ir::{
    CompiledSchedule, DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding, TaskNode,
};
use super::{
    ResourceId, SwapchainOutputHandle, TransientBufferSpec, TransientId, TransientTextureId,
    TransientTextureKey, TransientTextureSpec,
};
use crate::backend::{
    BufferHandle, GpuBackend, GpuCommand, GraphCommand, RenderCommand, RenderTargetHandle,
    TextureHandle,
};
use crate::buffer::{Buffer, BufferView};
use crate::compute::ComputePipeline;
use crate::device::Device;
use crate::encoder::CommandEncoder;
use crate::error::GoldyError;
use crate::render_target::RenderTarget;
use crate::texture::Texture;
use crate::timeline::TimelineValue;
use crate::types::TextureFormat;
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
fn build_partitioned_upload_remap(
    ir: &GraphIR,
    partitions: &[Vec<GpuCommand>],
) -> Vec<(usize, usize, usize)> {
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
                    buffer: cb,
                    offset: co,
                    ..
                },
                NodeKind::WriteBuffer {
                    buffer: nb,
                    offset: no,
                    ..
                },
            ) => cb == nb && co == no,
            (
                GpuCommand::WriteTexture { texture: ct, .. },
                NodeKind::WriteTexture { texture: nt, .. },
            ) => ct == nt,
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
/// let tv = graph.submit(&device)?;
/// device.wait_until(tv)?;
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
    partitioned_commands: Option<Vec<Vec<GpuCommand>>>,
    /// `(partition_idx, cmd_idx, ir_node_idx)` for upload commands across partitions.
    partitioned_upload_remap: Vec<(usize, usize, usize)>,
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
            #[cfg(debug_assertions)]
            prev_transient_shapes: Vec::new(),
            #[cfg(debug_assertions)]
            prev_transient_texture_keys: Vec::new(),
        }
    }

    /// Register a transient GPU buffer suballocation for this graph.
    ///
    /// The backing memory is a single device buffer allocated for the duration of
    /// [`crate::Device::submit`]. Transients whose live ranges (in the compiled wave
    /// schedule) do not overlap may alias within that heap to reduce allocation size.
    /// Graphs using transients **block until the submit completes** when using
    /// [`crate::Device::submit`] (so the CPU does not record overlapping standalone graphs that
    /// reuse the same placement-heap protocol). For pipelined multi-submit frames, use
    /// [`crate::Device::submit_pipelined`] or the surface path / [`crate::FrameOrchestrator`].
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

        self.transient_specs
            .push(TransientBufferSpec { id, size, stride });
        TransientId(id)
    }

    /// Register a transient texture (same dimensions and format) for this graph.
    ///
    /// Non-overlapping wave lifetimes may alias onto one backing texture; see
    /// [`Self::transient_buffer`] for scheduling behavior. [`crate::Device::submit`]
    /// waits until completion when transients are used; use [`crate::Device::submit_pipelined`]
    /// for overlapping submissions in a managed frame loop.
    ///
    /// ## Stable slot identity contract
    ///
    /// The returned [`TransientTextureId`] encodes the declaration order within this
    /// recording phase. The texture cache in [`crate::placement_heap::PlacementHeap`]
    /// keys on the graph-coloring color index, which is derived from stable spec ordering.
    /// Recordings must be deterministic; debug builds warn when a slot's shape changes.
    pub fn transient_texture(
        &mut self,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> TransientTextureId {
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
    pub(crate) fn transient_heap_size_and_layout(
        &self,
        node_waves: &[u32],
    ) -> Result<(u64, u64, HashMap<u32, u64>)> {
        Self::transient_heap_layout(&self.transient_specs, &self.ir, node_waves)
    }

    pub(crate) fn submit_with_backend(
        &mut self,
        device: &Device,
        backend: &mut dyn GpuBackend,
        _transient_buffer_ranges: Option<&HashMap<u32, (BufferHandle, u64, u64)>>,
        _transient_texture_handles: &HashMap<u32, TextureHandle>,
        wait_for_transient_completion: bool,
    ) -> Result<TimelineValue> {
        debug_assert!(
            self.transient_specs.is_empty() && self.transient_texture_specs.is_empty(),
            "submit_with_backend: transient resources must go through submit_ir_with_resolver"
        );

        let tv = Self::submit_resolved_ir(&mut self.schedule_cache, device, backend, &self.ir)?;
        if wait_for_transient_completion && self.needs_transient_gpu_wait() {
            backend.wait_until(device.inner.handle, tv)?;
        }
        Ok(tv)
    }

    fn submit_resolved_ir(
        cache: &mut Option<CompiledCacheEntry>,
        device: &Device,
        backend: &mut dyn GpuBackend,
        ir: &GraphIR,
    ) -> Result<TimelineValue> {
        let has_render = ir
            .nodes
            .iter()
            .any(|n| matches!(n.kind, NodeKind::RenderPass { .. }));

        if has_render {
            // Render-pass graphs: single submission (mixed compute+render CBs).
            let g = Self::compile_graph_commands_for_ir(ir);
            return backend.submit_graph(device.inner.handle, &g);
        }

        // Compute-only: partition the wave schedule and submit each group.
        // Early partitions are submitted immediately so the GPU can start
        // executing coarse work while the CPU records the next partition.
        // The last partition's timeline value is returned to the caller.
        let fp = Self::binding_fingerprint(ir);
        let partitions = Self::get_or_build_partitioned_commands(cache, ir, fp);
        let mut last_tv = backend.gpu_progress(device.inner.handle);
        for partition in partitions {
            let _tz = crate::tracy_zone!("goldy.submit_partition");
            last_tv = backend.submit_standalone(device.inner.handle, partition)?;
        }
        Ok(last_tv)
    }

    /// Submit `ir` to the backend and retain the closed command list for future resubmission.
    ///
    /// The retention key is always derived from [`Self::retention_fingerprint`] so it
    /// reflects every field that affects the recorded command buffer.  Render-pass graphs
    /// and graphs containing upload nodes fall back to a plain submit (no retention).
    fn submit_resolved_ir_and_retain(
        cache: &mut Option<CompiledCacheEntry>,
        device: &Device,
        backend: &mut dyn GpuBackend,
        ir: &GraphIR,
    ) -> Result<TimelineValue> {
        let has_render = ir
            .nodes
            .iter()
            .any(|n| matches!(n.kind, NodeKind::RenderPass { .. }));

        if has_render {
            // Render-pass graphs cannot be retained (mixed command lists); fall back.
            let g = Self::compile_graph_commands_for_ir(ir);
            return backend.submit_graph(device.inner.handle, &g);
        }

        let has_upload = ir.nodes.iter().any(|n| {
            matches!(
                n.kind,
                NodeKind::WriteBuffer { .. }
                    | NodeKind::WriteTexture { .. }
                    | NodeKind::WriteTextureRegion { .. }
                    | NodeKind::CopyTexture { .. }
            )
        });
        let fp = Self::binding_fingerprint(ir);
        if has_upload {
            // Upload commands use staging memory that is not re-encodable; fall back.
            let cmds = Self::get_or_build_compute_commands(cache, ir, fp);
            return backend.submit_standalone(device.inner.handle, cmds);
        }

        // Derive the retention key from full CB content so it is always correct.
        let key = Self::retention_fingerprint(ir);
        let cmds = Self::get_or_build_compute_commands(cache, ir, fp);
        let graph_cmds: Vec<GraphCommand> =
            cmds.iter().cloned().map(GraphCommand::Compute).collect();
        backend.submit_graph_and_retain(device.inner.handle, &graph_cmds, key)
    }

    /// Like [`Self::submit_with_backend`] but retains the closed command list keyed by
    /// the graph's [`Self::retention_fingerprint`].
    ///
    /// Graphs with transient resources, render passes, or upload nodes are not eligible for
    /// retention and silently fall back to a normal submit.
    /// Called from [`Device::submit_pipelined_and_retain`].
    pub(crate) fn submit_with_backend_and_retain(
        &mut self,
        device: &Device,
        backend: &mut dyn GpuBackend,
    ) -> Result<TimelineValue> {
        // Only retain pure compute graphs (no transients, no render passes, no uploads).
        if self.has_transient_resources() {
            return Self::submit_resolved_ir(&mut self.schedule_cache, device, backend, &self.ir);
        }
        Self::submit_resolved_ir_and_retain(&mut self.schedule_cache, device, backend, &self.ir)
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
                anyhow::bail!(
                    "transient_buffer id {} is never referenced by any graph node",
                    s.id
                );
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
                if assigned
                    .iter()
                    .all(|&other| !wave_intervals_overlap(iv, other))
                {
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
        device: &Device,
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
        let schedule = Self::get_or_build_schedule(
            &mut self.schedule_cache,
            &self.ir,
            fp,
        );

        if has_render {
            let g = analysis::emit_graph_commands(&self.ir, schedule, Some(resolver));
            let tv = backend.submit_graph(device.inner.handle, &g)?;
            if wait_for_transient_completion && self.needs_transient_gpu_wait() {
                backend.wait_until(device.inner.handle, tv)?;
            }
            return Ok(tv);
        }

        let partitions = analysis::emit_partitioned_commands(&self.ir, schedule, Some(resolver));
        let mut last_tv = backend.gpu_progress(device.inner.handle);
        for partition in &partitions {
            let _tz = crate::tracy_zone!("goldy.submit_partition");
            last_tv = backend.submit_standalone(device.inner.handle, partition)?;
        }
        if wait_for_transient_completion && self.needs_transient_gpu_wait() {
            backend.wait_until(device.inner.handle, last_tv)?;
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
    ) -> Result<HashMap<u32, TextureHandle>> {
        if self.transient_texture_specs.is_empty() {
            return Ok(HashMap::new());
        }
        let intervals = analysis::transient_texture_wave_intervals(&self.ir, node_waves)?;
        for s in &self.transient_texture_specs {
            if !intervals.contains_key(&s.id) {
                anyhow::bail!(
                    "transient_texture id {} is never referenced by any graph node",
                    s.id
                );
            }
        }
        let (id_to_color, color_keys) =
            Self::transient_texture_coloring(&self.transient_texture_specs, &intervals)?;
        let per_color_handles = heap.get_or_create_textures(device, &color_keys)?;
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
                if assigned
                    .iter()
                    .all(|&other| !wave_intervals_overlap(iv, other))
                {
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
        self.ir.nodes.iter().any(|n| {
            n.bindings
                .iter()
                .any(|b| b.resource == ResourceId::SwapchainOutput)
        })
    }

    pub(crate) fn compile_graph_commands_for_ir(ir: &GraphIR) -> Vec<GraphCommand> {
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        analysis::emit_graph_commands(ir, &schedule, None)
    }

    fn has_render_passes_in_ir(ir: &GraphIR) -> bool {
        ir.nodes
            .iter()
            .any(|n| matches!(n.kind, NodeKind::RenderPass { .. }))
    }

    pub fn has_render_passes(&self) -> bool {
        Self::has_render_passes_in_ir(&self.ir)
    }

    /// Add a compute dispatch node to the graph. The returned [`NodeBuilder`] must
    /// be finalized with [`NodeBuilder::dispatch`] or [`NodeBuilder::dispatch_indirect`].
    pub fn node<'a>(
        &'a mut self,
        label: &'static str,
        pipeline: &ComputePipeline,
    ) -> NodeBuilder<'a> {
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

    /// Add a CPU→GPU texture upload node (full image).
    ///
    /// Data length must match [`Texture::byte_size`]. The upload is batched with
    /// the same submission as surrounding graph nodes; the analyzer inserts barriers
    /// before any node that reads the texture.
    pub fn write_texture(&mut self, texture: &Texture, data: Vec<u8>) -> Result<()> {
        let expected = texture.byte_size();
        if data.len() != expected {
            anyhow::bail!(
                "write_texture: expected {} bytes, got {}",
                expected,
                data.len()
            );
        }
        let width = texture.width();
        let height = texture.height();
        let th = texture.handle();
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
        let src_h = src.handle();
        let dst_h = dst.handle();
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
                dst: dst_h,
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
        let th = texture.handle();
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

    /// Begin building an offscreen [`crate::RenderTarget`] render pass node.
    pub fn render_pass<'a>(
        &'a mut self,
        label: &'static str,
        target: &RenderTarget,
    ) -> RenderPassBuilder<'a> {
        RenderPassBuilder {
            graph: self,
            label,
            target: target.backend_handle(),
            bindings: Vec::new(),
        }
    }

    /// Analyze the graph and submit all tasks with optimal barriers.
    /// Returns the device [`TimelineValue`] to pass to [`Device::wait_until`].
    pub fn submit(&mut self, device: &Device) -> Result<TimelineValue, GoldyError> {
        device.submit(self)
    }

    /// Analyze the graph, submit, and block until complete.
    pub fn dispatch(&mut self, device: &Device) -> Result<(), GoldyError> {
        device.dispatch(self)
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
    /// Equivalent to `*self = TaskGraph::new()` but avoids freeing and
    /// re-allocating the internal `Vec` buffers. Call this after submitting the
    /// graph to reuse its capacity for the next frame's recording pass.
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
            self.prev_transient_shapes = self
                .transient_specs
                .iter()
                .map(|s| (s.size, s.stride))
                .collect();
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
    /// Pass this value to [`crate::Device::try_resubmit_retained`] to attempt zero-cost
    /// resubmission; [`crate::Device::submit_pipelined_and_retain`] derives and stores the
    /// same key internally.
    pub fn compute_retention_fingerprint(&self) -> u64 {
        Self::retention_fingerprint(&self.ir)
    }

    fn retention_fingerprint(ir: &GraphIR) -> u64 {
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
                    resource_slots.hash(&mut h);
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
                NodeKind::ClearBuffer {
                    buffer,
                    offset,
                    size,
                } => {
                    1u8.hash(&mut h);
                    buffer.hash(&mut h);
                    offset.hash(&mut h);
                    size.hash(&mut h);
                }
                // Upload / copy nodes: excluded intentionally (data varies per frame;
                // graphs containing these nodes are not eligible for retention).
                _ => {
                    2u8.hash(&mut h);
                }
            }
        }
        h.finish()
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
                        (
                            GpuCommand::WriteBuffer { data, .. },
                            NodeKind::WriteBuffer { data: src, .. },
                        ) => *data = src.clone(),
                        (
                            GpuCommand::WriteTexture { data, .. },
                            NodeKind::WriteTexture { data: src, .. },
                        ) => *data = src.clone(),
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
        });
        cache.as_ref().unwrap().commands.as_deref().unwrap()
    }

    /// Return cached partitioned commands for `ir`, building them if necessary.
    ///
    /// Like [`Self::get_or_build_compute_commands`] but returns a partitioned
    /// `Vec<Vec<GpuCommand>>` suitable for multi-submission.  On cache hit only
    /// the upload `Arc<[u8]>` payloads are refreshed.
    fn get_or_build_partitioned_commands<'c>(
        cache: &'c mut Option<CompiledCacheEntry>,
        ir: &GraphIR,
        fp: u64,
    ) -> &'c [Vec<GpuCommand>] {
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
            });
        }

        let needs_build = cache
            .as_ref()
            .is_none_or(|e| e.partitioned_commands.is_none());

        tracing::trace!(target: "goldy::schedule_cache", hit = !needs_build, fp, "partitioned_commands");

        if !needs_build {
            // Hit: refresh upload payloads.
            let entry = cache.as_mut().unwrap();
            if let Some(parts) = entry.partitioned_commands.as_mut() {
                for &(part_idx, cmd_idx, node_idx) in &entry.partitioned_upload_remap {
                    let node = &ir.nodes[node_idx];
                    match (&mut parts[part_idx][cmd_idx], &node.kind) {
                        (
                            GpuCommand::WriteBuffer { data, .. },
                            NodeKind::WriteBuffer { data: src, .. },
                        ) => *data = src.clone(),
                        (
                            GpuCommand::WriteTexture { data, .. },
                            NodeKind::WriteTexture { data: src, .. },
                        ) => *data = src.clone(),
                        (
                            GpuCommand::WriteTextureRegion { data, .. },
                            NodeKind::WriteTextureRegion { data: src, .. },
                        ) => *data = src.clone(),
                        _ => {}
                    }
                }
            }
            return cache
                .as_ref()
                .unwrap()
                .partitioned_commands
                .as_deref()
                .unwrap();
        }

        // Miss: emit partitioned commands from the cached schedule.
        let entry = cache.as_mut().unwrap();
        let partitions = analysis::emit_partitioned_commands(ir, &entry.schedule, None);
        let remap = build_partitioned_upload_remap(ir, &partitions);
        entry.partitioned_commands = Some(partitions);
        entry.partitioned_upload_remap = remap;
        cache
            .as_ref()
            .unwrap()
            .partitioned_commands
            .as_deref()
            .unwrap()
    }

    /// Compile the graph into a flat command stream.
    ///
    /// Runs the dependency analyzer, schedules waves, inserts `ResourceBarrier`
    /// commands at wave boundaries, and emits the final [`GpuCommand`](crate::backend::GpuCommand) sequence.
    ///
    /// # Panics
    ///
    /// If the graph contains render-pass nodes or transient buffers, use
    /// [`Self::compile_graph_commands`] or [`Device::submit`](crate::Device::submit) instead.
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
    /// [`SWAPCHAIN_SLOT_PLACEHOLDER`] in `resource_slots` at the corresponding
    /// binding position so `TaskGraph::lower_swapchain_output` can patch it with the
    /// real UAV bindless index after `surface.begin()`.
    pub fn bind_swapchain_output(
        mut self,
        _handle: SwapchainOutputHandle,
        access: NodeAccess,
    ) -> Self {
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

/// Builder for a render pass targeting an offscreen [`crate::RenderTarget`].
pub struct RenderPassBuilder<'a> {
    graph: &'a mut TaskGraph,
    label: &'static str,
    target: RenderTargetHandle,
    bindings: Vec<ResourceBinding>,
}

impl<'a> RenderPassBuilder<'a> {
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

    /// Finalize the node with recorded [`RenderCommand`]s (e.g. from [`CommandEncoder::finish`](crate::encoder::CommandEncoder::finish)).
    pub fn finish(self, commands: Vec<RenderCommand>) {
        self.graph.ir.nodes.push(TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::RenderPass {
                target: self.target,
                commands,
            },
        });
    }

    /// Convenience: [`CommandEncoder::finish`](crate::encoder::CommandEncoder::finish) then [`Self::finish`].
    pub fn finish_encoder(self, encoder: CommandEncoder) {
        self.finish(encoder.finish())
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
    use crate::encoder::CommandEncoder;
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

    #[test]
    fn compile_mixed_compute_render_inserts_barrier_and_submits_graph() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let target = RenderTarget::new(&device, 8, 8, TextureFormat::Rgba8Unorm).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("compute_write", &pipeline)
            .bind_buffer(&buf, NodeAccess::Write)
            .bind_resources_raw_slice(&[1])
            .dispatch(1, 1, 1);

        let mut enc = CommandEncoder::new();
        {
            let mut pass = enc.begin_render_pass();
            pass.clear(Color::RED);
        }
        graph
            .render_pass("draw", &target)
            .bind_buffer(&buf, NodeAccess::Read)
            .finish_encoder(enc);

        let gcs = graph.compile_graph_commands();
        assert!(
            gcs.iter()
                .any(|c| matches!(c, GraphCommand::Compute(GpuCommand::ResourceBarrier { .. }))),
            "expected ResourceBarrier between compute write and render read"
        );

        graph.submit(&device).unwrap();
    }

    #[test]
    fn transient_buffer_submit_succeeds_on_mock() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let mut graph = TaskGraph::new();
        let t = graph.transient_buffer(256);
        graph
            .node("touch", &pipeline)
            .bind_transient_buffer(t, NodeAccess::Write)
            .bind_resources_raw_slice(&[0])
            .dispatch(1, 1, 1);
        graph.submit(&device).unwrap();
    }

    #[test]
    fn transient_texture_submit_succeeds_on_mock() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let mut graph = TaskGraph::new();
        let tt = graph.transient_texture(4, 4, TextureFormat::Rgba8Unorm);
        graph
            .node("touch_tex", &pipeline)
            .bind_transient_texture(tt, NodeAccess::Write)
            .bind_resources_raw_slice(&[0])
            .dispatch(1, 1, 1);
        graph.submit(&device).unwrap();
    }

    #[test]
    fn transient_texture_heap_aliases_non_overlapping_waves() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 4, crate::DataAccess::Scattered).unwrap();
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
        graph.submit(&device).unwrap();
    }

    #[test]
    fn transient_heap_aliases_non_overlapping_waves() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 4, crate::DataAccess::Scattered).unwrap();

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
        let node_waves =
            crate::task_graph::analysis::node_to_wave_map(&schedule, graph.ir().nodes.len());
        let (total, _, layout) = graph.transient_heap_size_and_layout(&node_waves).unwrap();
        assert_eq!(
            total, 256,
            "sequential transients should pack into one 256-byte slot"
        );
        assert_eq!(layout[&t0.0], layout[&t1.0]);
        graph.submit(&device).unwrap();
    }

    #[test]
    fn transient_heap_separates_concurrent_waves() {
        let device = mock_device();
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
        let node_waves =
            crate::task_graph::analysis::node_to_wave_map(&schedule, graph.ir().nodes.len());
        let (total, _, layout) = graph.transient_heap_size_and_layout(&node_waves).unwrap();
        assert!(
            total >= 512,
            "concurrent transients need disjoint heap regions, got {}",
            total
        );
        assert_ne!(layout[&t0.0], layout[&t1.0]);
        graph.submit(&device).unwrap();
    }

    #[test]
    fn compile_linear_chain() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf_a = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

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

        // Wave 0: SetPipeline, BindResourcesRaw, Dispatch
        // ResourceBarrier
        // Wave 1: SetPipeline, BindResourcesRaw, Dispatch
        assert_eq!(cmds.len(), 7);
        assert!(matches!(cmds[0], GpuCommand::SetPipeline(_)));
        assert!(matches!(cmds[1], GpuCommand::BindResourcesRaw { .. }));
        assert!(matches!(
            cmds[2],
            GpuCommand::Dispatch {
                workgroups_x: 8,
                ..
            }
        ));
        assert!(matches!(cmds[3], GpuCommand::ResourceBarrier { .. }));
        assert!(matches!(cmds[4], GpuCommand::SetPipeline(_)));
        assert!(matches!(cmds[5], GpuCommand::BindResourcesRaw { .. }));
        assert!(matches!(
            cmds[6],
            GpuCommand::Dispatch {
                workgroups_x: 4,
                ..
            }
        ));
    }

    #[test]
    fn compile_independent_no_barrier() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf_a = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

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

        assert!(!cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
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
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("work", &pipeline)
            .bind_buffer(&buf, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);

        let tv = graph.submit(&device).unwrap();
        assert!(device.gpu_progress() >= tv);
        device.wait_until(tv).unwrap();
    }

    #[test]
    fn compile_diamond() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let p1 = mock_pipeline(&device, &shader);
        let p2 = mock_pipeline(&device, &shader);
        let p3 = mock_pipeline(&device, &shader);
        let p4 = mock_pipeline(&device, &shader);

        let buf_x = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let buf_y = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let buf_z = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

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

        let dispatch_count = cmds
            .iter()
            .filter(|c| matches!(c, GpuCommand::Dispatch { .. }))
            .count();
        assert_eq!(dispatch_count, 4);
    }

    #[test]
    fn len_and_is_empty() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

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

        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.clear_buffer(&buf, 0, 256);
        graph
            .node("read", &pipeline)
            .bind_buffer(&buf, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        // ClearBuffer, ResourceBarrier, SetPipeline, Dispatch
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

        let buf_a = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.clear_buffer(&buf_a, 0, 256);
        graph
            .node("write_b", &pipeline)
            .bind_buffer(&buf_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

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
    fn write_buffer_then_read_produces_barrier() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.write_buffer(&buf, 0, vec![0u8; 256]);
        graph
            .node("read", &pipeline)
            .bind_buffer(&buf, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        assert!(matches!(cmds[0], GpuCommand::WriteBuffer { .. }));
        assert!(
            cmds.iter()
                .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
            "expected a barrier between write_buffer and dispatch"
        );
    }

    #[test]
    fn write_buffer_independent_of_unrelated_dispatch_same_wave() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf_a = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.write_buffer(&buf_a, 0, vec![0u8; 4]);
        graph
            .node("write_b", &pipeline)
            .bind_buffer(&buf_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        assert!(!cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
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
            crate::types::SpatialAccess::Interpolated,
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
        assert!(matches!(cmds[0], GpuCommand::WriteTexture { .. }));
        assert!(
            cmds.iter()
                .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
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
            crate::types::SpatialAccess::Interpolated,
            crate::types::TextureFlags::COPY_DST,
        )
        .unwrap();
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.write_texture(&tex, vec![0u8; 16]).unwrap();
        graph
            .node("writes_buf", &pipeline)
            .bind_buffer(&buf, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();
        assert!(!cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
    }

    #[test]
    fn multiple_clears_independent_same_wave() {
        let device = mock_device();
        let buf_a = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        graph.clear_buffer(&buf_a, 0, 256);
        graph.clear_buffer(&buf_b, 0, 256);

        let cmds = graph.compile_commands();

        assert!(!cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
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
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = TaskGraph::new();
        assert!(graph.is_empty());
        graph.clear_buffer(&buf, 0, 256);
        assert!(!graph.is_empty());
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn is_empty_with_write_node() {
        let device = mock_device();
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

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
    fn make_pool_setup(
        total_size: u64,
    ) -> (
        Device,
        ShaderModule,
        crate::compute::ComputePipeline,
        BufferPool,
    ) {
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
        assert!(!cmds
            .iter()
            .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })));
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

        let owned = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
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

        let dispatch_count = cmds
            .iter()
            .filter(|c| matches!(c, GpuCommand::Dispatch { .. }))
            .count();
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

        let dispatch_count = cmds
            .iter()
            .filter(|c| matches!(c, GpuCommand::Dispatch { .. }))
            .count();
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
            !cmds
                .iter()
                .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
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
                buffers.contains(&parent_handle),
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
        assert_eq!(
            barrier_count, 1,
            "expected exactly one barrier (clear → views)"
        );

        // ClearBuffer is the first command
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
            !cmds
                .iter()
                .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
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
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

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
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

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
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

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
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

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
}
