//! `TaskGraph` — analyzed GPU task graph with automatic barrier insertion.

use super::analysis;
use super::ir::{DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding, TaskNode};
use super::{
    ResourceId, TransientBufferSpec, TransientId, TransientTextureId, TransientTextureKey,
    TransientTextureSpec,
};
use crate::backend::{
    BufferHandle, GpuBackend, GraphCommand, RenderCommand, RenderTargetHandle, TextureHandle,
};
use crate::buffer::{Buffer, BufferView};
use crate::compute::ComputePipeline;
use crate::device::Device;
use crate::encoder::CommandEncoder;
use crate::error::GoldyError;
use crate::render_target::RenderTarget;
use crate::texture::Texture;
use crate::timeline::TimelineValue;
use crate::types::{SpatialAccess, TextureFlags, TextureFormat};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
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
}

impl TaskGraph {
    pub fn new() -> Self {
        Self {
            ir: GraphIR::default(),
            transient_specs: Vec::new(),
            next_transient_id: 0,
            transient_texture_specs: Vec::new(),
            next_transient_texture_id: 0,
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
    pub fn transient_buffer_with_stride(&mut self, size: u64, stride: u32) -> TransientId {
        let id = self.next_transient_id;
        self.next_transient_id += 1;
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
    pub fn transient_texture(
        &mut self,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> TransientTextureId {
        let id = self.next_transient_texture_id;
        self.next_transient_texture_id += 1;
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
    pub(crate) fn transient_heap_size_and_layout(&self) -> Result<(u64, u64, HashMap<u32, u64>)> {
        Self::transient_heap_layout(&self.transient_specs, &self.ir)
    }

    pub(crate) fn submit_with_backend(
        &self,
        device: &Device,
        backend: &mut dyn GpuBackend,
        transient_buffer_ranges: Option<&HashMap<u32, (BufferHandle, u64, u64)>>,
        transient_texture_handles: &HashMap<u32, TextureHandle>,
        wait_for_transient_completion: bool,
    ) -> Result<TimelineValue> {
        if !self.transient_texture_specs.is_empty() {
            for s in &self.transient_texture_specs {
                if !transient_texture_handles.contains_key(&s.id) {
                    anyhow::bail!("internal: missing transient texture handle for id {}", s.id);
                }
            }
        }

        let mut ir = if self.transient_specs.is_empty() {
            debug_assert!(
                transient_buffer_ranges.is_none(),
                "transient buffer map must be None when graph has no transient buffers"
            );
            self.ir.clone()
        } else {
            let map = transient_buffer_ranges.ok_or_else(|| {
                anyhow::anyhow!(
                    "internal: transient buffer ranges required for graphs with transient buffers"
                )
            })?;
            Self::lower_transient_buffers(&self.ir, map)?
        };

        if !self.transient_texture_specs.is_empty() {
            ir = Self::lower_transient_textures(&ir, transient_texture_handles)?;
        }

        let tv = Self::submit_resolved_ir(device, backend, &ir)?;
        if wait_for_transient_completion && self.needs_transient_gpu_wait() {
            backend.wait_until(device.inner.handle, tv)?;
        }
        Ok(tv)
    }

    fn submit_resolved_ir(
        device: &Device,
        backend: &mut dyn GpuBackend,
        ir: &GraphIR,
    ) -> Result<TimelineValue> {
        let has_render = ir
            .nodes
            .iter()
            .any(|n| matches!(n.kind, NodeKind::RenderPass { .. }));

        if has_render {
            let g = Self::compile_graph_commands_for_ir(ir);
            backend.submit_graph(device.inner.handle, &g)
        } else {
            let edges = analysis::build_edges(ir);
            let schedule = analysis::schedule_waves(ir, &edges);
            let cmds = analysis::emit_commands(ir, &schedule);
            backend.submit_standalone(device.inner.handle, &cmds)
        }
    }

    /// Pack transient buffers into a heap using wave live ranges: transients whose
    /// lifetimes do not overlap (in the compiled wave schedule) may alias the same bytes.
    fn transient_heap_layout(
        specs: &[TransientBufferSpec],
        ir: &GraphIR,
    ) -> Result<(u64, u64, HashMap<u32, u64>)> {
        if specs.is_empty() {
            return Ok((0, 256, HashMap::new()));
        }

        let intervals = analysis::transient_wave_intervals(ir)?;
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

    pub(crate) fn transient_buffer_range_map_with_base(
        heap: &Buffer,
        layout: &HashMap<u32, u64>,
        specs: &[TransientBufferSpec],
        base_offset: u64,
    ) -> HashMap<u32, (BufferHandle, u64, u64)> {
        let parent = heap.gpu_buffer_handle();
        specs
            .iter()
            .map(|s| (s.id, (parent, base_offset + layout[&s.id], s.size)))
            .collect()
    }

    /// Resolve all transient buffer references in the IR: patch bindings
    /// (`TransientBuffer` → `BufferRange`) AND patch `resource_slots` in
    /// dispatch nodes (placeholder → real bindless index).
    ///
    /// `bindless_map` maps each transient id to its (UAV index, SRV index).
    pub(crate) fn lower_transient_buffers_with_bindless(
        &self,
        range_map: &HashMap<u32, (BufferHandle, u64, u64)>,
        bindless_map: &HashMap<u32, (u32, u32)>,
    ) -> Result<GraphIR> {
        Self::lower_transient_buffers_inner(&self.ir, range_map, Some(bindless_map))
    }

    /// Produce a new `TaskGraph` with all transient buffer specs cleared and
    /// the IR fully resolved (no `TransientBuffer` resource ids remain).
    /// The returned graph can be submitted via `compile_commands` or
    /// `Frame::submit_compute`.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn into_resolved(&self, resolved_ir: GraphIR) -> TaskGraph {
        TaskGraph {
            ir: resolved_ir,
            transient_specs: Vec::new(),
            next_transient_id: 0,
            transient_texture_specs: self.transient_texture_specs.clone(),
            next_transient_texture_id: self.next_transient_texture_id,
        }
    }

    fn lower_transient_buffers(
        ir: &GraphIR,
        range_map: &HashMap<u32, (BufferHandle, u64, u64)>,
    ) -> Result<GraphIR> {
        Self::lower_transient_buffers_inner(ir, range_map, None)
    }

    fn lower_transient_buffers_inner(
        ir: &GraphIR,
        range_map: &HashMap<u32, (BufferHandle, u64, u64)>,
        bindless_map: Option<&HashMap<u32, (u32, u32)>>,
    ) -> Result<GraphIR> {
        let mut nodes = Vec::with_capacity(ir.nodes.len());
        for n in &ir.nodes {
            let bindings: Result<Vec<ResourceBinding>> = n
                .bindings
                .iter()
                .map(|b| {
                    let resource = match b.resource {
                        ResourceId::TransientBuffer(t) => {
                            let (parent, offset, len) =
                                range_map.get(&t.0).copied().ok_or_else(|| {
                                    anyhow::anyhow!("unknown transient buffer id {}", t.0)
                                })?;
                            ResourceId::BufferRange {
                                parent,
                                offset,
                                len,
                            }
                        }
                        o => o,
                    };
                    Ok(ResourceBinding {
                        resource,
                        access: b.access,
                    })
                })
                .collect();

            let kind = if let (
                Some(bmap),
                NodeKind::Dispatch {
                    pipeline,
                    resource_slots,
                    user_slots,
                    dispatch,
                },
            ) = (bindless_map, &n.kind)
            {
                let mut patched_slots = resource_slots.clone();
                for (i, b) in n.bindings.iter().enumerate() {
                    if let ResourceId::TransientBuffer(t) = b.resource {
                        if let Some(&(uav_idx, srv_idx)) = bmap.get(&t.0) {
                            if i < patched_slots.len() {
                                let is_read_only = b.access == NodeAccess::Read;
                                patched_slots[i] = if is_read_only { srv_idx } else { uav_idx };
                            }
                        }
                    }
                }
                NodeKind::Dispatch {
                    pipeline: *pipeline,
                    resource_slots: patched_slots,
                    user_slots: user_slots.clone(),
                    dispatch: dispatch.clone(),
                }
            } else {
                n.kind.clone()
            };

            nodes.push(TaskNode {
                label: n.label,
                bindings: bindings?,
                kind,
            });
        }
        Ok(GraphIR { nodes })
    }

    /// Create one GPU texture per aliasing color; map every transient texture id to its handle.
    /// Returns textures that must be kept alive until the graph submit completes on the GPU.
    pub(crate) fn allocate_transient_textures(
        &self,
        device: &Device,
    ) -> Result<(Vec<Texture>, HashMap<u32, TextureHandle>)> {
        if self.transient_texture_specs.is_empty() {
            return Ok((Vec::new(), HashMap::new()));
        }
        let intervals = analysis::transient_texture_wave_intervals(&self.ir)?;
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
        let mut keep_alive: Vec<Texture> = Vec::with_capacity(color_keys.len());
        let mut per_color_handle: Vec<TextureHandle> = Vec::with_capacity(color_keys.len());
        for k in &color_keys {
            let tex = Texture::new(
                device,
                k.width,
                k.height,
                k.format,
                SpatialAccess::Direct,
                TextureFlags::COPY_DST,
            )?;
            per_color_handle.push(tex.handle());
            keep_alive.push(tex);
        }
        let mut out = HashMap::new();
        for s in &self.transient_texture_specs {
            let c = id_to_color[&s.id];
            out.insert(s.id, per_color_handle[c]);
        }
        Ok((keep_alive, out))
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

    pub(crate) fn lower_transient_textures(
        ir: &GraphIR,
        handles: &HashMap<u32, TextureHandle>,
    ) -> Result<GraphIR> {
        let mut nodes = Vec::with_capacity(ir.nodes.len());
        for n in &ir.nodes {
            let bindings: Result<Vec<ResourceBinding>> = n
                .bindings
                .iter()
                .map(|b| {
                    let resource = match b.resource {
                        ResourceId::TransientTexture(t) => {
                            let h = handles.get(&t.0).copied().ok_or_else(|| {
                                anyhow::anyhow!("unknown transient texture id {}", t.0)
                            })?;
                            ResourceId::Texture(h)
                        }
                        o => o,
                    };
                    Ok(ResourceBinding {
                        resource,
                        access: b.access,
                    })
                })
                .collect();
            nodes.push(TaskNode {
                label: n.label,
                bindings: bindings?,
                kind: n.kind.clone(),
            });
        }
        Ok(GraphIR { nodes })
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

    pub(crate) fn compile_graph_commands_for_ir(ir: &GraphIR) -> Vec<GraphCommand> {
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        analysis::emit_graph_commands(ir, &schedule)
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
    pub fn submit(&self, device: &Device) -> Result<TimelineValue, GoldyError> {
        device.submit(self)
    }

    /// Analyze the graph, submit, and block until complete.
    pub fn dispatch(&self, device: &Device) -> Result<(), GoldyError> {
        device.dispatch(self)
    }

    /// Number of task nodes in the graph.
    pub fn len(&self) -> usize {
        self.ir.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ir.nodes.is_empty()
    }

    /// Access the raw IR for internal use (e.g. transient lowering from outside the task_graph module).
    pub(crate) fn ir(&self) -> &GraphIR {
        &self.ir
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
    pub fn compile_commands(&self) -> Vec<crate::backend::GpuCommand> {
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
        let edges = analysis::build_edges(&self.ir);
        let schedule = analysis::schedule_waves(&self.ir, &edges);
        analysis::emit_commands(&self.ir, &schedule)
    }

    /// Compile a pre-lowered [`GraphIR`] into a flat GPU command stream.
    ///
    /// Unlike [`Self::compile_commands`], this operates directly on a resolved IR that
    /// contains no transient specs — callers are responsible for lowering transients first.
    /// Used by `Frame::submit_compute` to compile the resolved IR after placement-heap
    /// allocation without going through the transient-guarded public path.
    pub(crate) fn compile_ir_to_gpu_commands(ir: &GraphIR) -> Vec<crate::backend::GpuCommand> {
        let edges = analysis::build_edges(ir);
        let schedule = analysis::schedule_waves(ir, &edges);
        analysis::emit_commands(ir, &schedule)
    }

    /// Like [`Self::compile_commands`] but allows graphs that include render-pass nodes.
    pub fn compile_graph_commands(&self) -> Vec<GraphCommand> {
        assert!(
            self.transient_specs.is_empty(),
            "compile_graph_commands: graph uses transient_buffer; use Device::submit"
        );
        assert!(
            self.transient_texture_specs.is_empty(),
            "compile_graph_commands: graph uses transient_texture; use Device::submit"
        );
        let edges = analysis::build_edges(&self.ir);
        let schedule = analysis::schedule_waves(&self.ir, &edges);
        analysis::emit_graph_commands(&self.ir, &schedule)
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

        let (total, _, layout) = graph.transient_heap_size_and_layout().unwrap();
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

        let (total, _, layout) = graph.transient_heap_size_and_layout().unwrap();
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
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, GpuCommand::Dispatch { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn compile_empty_graph() {
        let graph = TaskGraph::new();
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
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, GpuCommand::Dispatch { .. }))
                .count(),
            2
        );
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
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, GpuCommand::Dispatch { .. }))
                .count(),
            2
        );
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
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, GpuCommand::Dispatch { .. }))
                .count(),
            10
        );
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
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, GpuCommand::Dispatch { .. }))
                .count(),
            2
        );
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
}
