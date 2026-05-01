//! `ComputeGraph` — Tier 1: interpreted, dynamic compute graph.

use super::analysis;
use super::ir::{DispatchKind, GraphIR, GraphNode, NodeAccess, ResourceBinding};
use super::ResourceId;
use crate::buffer::{Buffer, BufferView};
use crate::compute::ComputePipeline;
use crate::device::Device;
use crate::gpu_future::GpuFuture;
use crate::texture::Texture;
use anyhow::Result;
/// A dynamic compute graph that analyzes dependencies at submit time.
///
/// Build a DAG of dispatch nodes with per-resource access declarations,
/// then submit. Goldy analyzes the graph, inserts minimal barriers, and
/// executes with maximum parallelism.
///
/// # Example
///
/// ```rust,ignore
/// let mut graph = ComputeGraph::new();
///
/// graph.node("write_data", &pipeline_a)
///     .bind_buffer(&buf, NodeAccess::Write)
///     .push_constants_raw(&[buf_idx])
///     .dispatch(64, 1, 1);
///
/// graph.node("read_data", &pipeline_b)
///     .bind_buffer(&buf, NodeAccess::Read)
///     .push_constants_raw(&[buf_idx])
///     .dispatch(64, 1, 1);
///
/// graph.submit(&device)?.wait()?;
/// ```
pub struct ComputeGraph {
    ir: GraphIR,
    /// Prepend arbitrary compute commands **before** the analyzed graph (e.g. pool /
    /// buffer clears) so they batch into one compute submit.
    pub prelude: Vec<crate::backend::ComputeCommand>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self {
            ir: GraphIR::default(),
            prelude: Vec::new(),
        }
    }

    /// Add a dispatch node to the graph. The returned [`NodeBuilder`] must
    /// be finalized with [`NodeBuilder::dispatch`].
    pub fn node<'a>(&'a mut self, label: &str, pipeline: &ComputePipeline) -> NodeBuilder<'a> {
        NodeBuilder {
            graph: self,
            node: GraphNode {
                label: label.to_string(),
                pipeline: pipeline.handle,
                bindings: Vec::new(),
                push_constants: Vec::new(),
                dispatch: DispatchKind::Direct { x: 0, y: 0, z: 0 },
            },
        }
    }

    /// Analyze the graph and submit all dispatches with optimal barriers.
    /// Returns a [`GpuFuture`] for non-blocking completion.
    pub fn submit(&self, device: &Device) -> Result<GpuFuture> {
        let commands = self.compile_commands();
        device.submit_compute_commands(&commands)
    }

    /// Analyze the graph, submit, and block until complete.
    pub fn dispatch(&self, device: &Device) -> Result<()> {
        let commands = self.compile_commands();
        let mut backend = device.backend.lock().unwrap();
        backend.dispatch_compute(device.handle, &commands)
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.ir.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ir.nodes.is_empty() && self.prelude.is_empty()
    }

    /// Push a zero-fill of `buffer` at `[offset, offset+size)` into the prelude.
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: u64) {
        self.prelude.push(crate::backend::ComputeCommand::ClearBuffer {
            buffer: buffer.gpu_buffer_handle(),
            offset,
            size,
        });
    }

    /// Push a zero-fill of a view region into the prelude.
    ///
    /// `offset` is relative to the view's start; the absolute parent-buffer
    /// offset is computed internally. If `size` is 0, clears from `offset`
    /// to the end of the view.
    pub fn clear_buffer_view(&mut self, view: &BufferView, offset: u64, size: u64) {
        let clear_size = if size == 0 {
            view.size().saturating_sub(offset)
        } else {
            size
        };
        self.prelude.push(crate::backend::ComputeCommand::ClearBuffer {
            buffer: view.parent_handle(),
            offset: view.offset() + offset,
            size: clear_size,
        });
    }

    pub fn compile_commands(&self) -> Vec<crate::backend::ComputeCommand> {
        let mut commands = self.prelude.clone();
        let edges = analysis::build_edges(&self.ir);
        let schedule = analysis::schedule_waves(&self.ir, &edges);
        commands.extend(analysis::emit_commands(&self.ir, &schedule));
        commands
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for a single dispatch node within a [`ComputeGraph`].
///
/// Created by [`ComputeGraph::node`]. Must be finalized with [`dispatch`](NodeBuilder::dispatch).
pub struct NodeBuilder<'a> {
    graph: &'a mut ComputeGraph,
    node: GraphNode,
}

impl<'a> NodeBuilder<'a> {
    /// Declare that this node accesses a buffer with the given logical access.
    pub fn bind_buffer(mut self, buf: &Buffer, access: NodeAccess) -> Self {
        self.node.bindings.push(ResourceBinding {
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
        self.node.bindings.push(ResourceBinding {
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
        self.node.bindings.push(ResourceBinding {
            resource: ResourceId::Texture(tex.handle),
            access,
        });
        self
    }

    /// Set the push constant indices for this node's dispatch.
    pub fn push_constants_raw(mut self, indices: &[u32]) -> Self {
        self.node.push_constants = indices.to_vec();
        self
    }

    /// Finalize the node with fixed workgroup dimensions.
    pub fn dispatch(mut self, x: u32, y: u32, z: u32) {
        self.node.dispatch = DispatchKind::Direct { x, y, z };
        self.graph.ir.nodes.push(self.node);
    }

    /// Finalize the node with indirect dispatch (dimensions read from `buf` at `offset`).
    pub fn dispatch_indirect(mut self, buf: &Buffer, offset: u64) {
        self.node.dispatch = DispatchKind::Indirect {
            buffer: buf.handle,
            offset,
        };
        self.graph.ir.nodes.push(self.node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::backend::ComputeCommand;
    use crate::buffer::BufferPool;
    use crate::device::Device;
    use crate::shader::ShaderModule;

    fn mock_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn mock_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(device, "void main() {}").unwrap()
    }

    fn mock_pipeline(device: &Device, shader: &ShaderModule) -> ComputePipeline {
        ComputePipeline::new(device, shader).unwrap()
    }

    #[test]
    fn compile_linear_chain() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf_a = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = ComputeGraph::new();
        graph
            .node("write", &pipeline)
            .bind_buffer(&buf_a, NodeAccess::Write)
            .push_constants_raw(&[42])
            .dispatch(8, 1, 1);
        graph
            .node("read_write", &pipeline)
            .bind_buffer(&buf_a, NodeAccess::Read)
            .bind_buffer(&buf_b, NodeAccess::Write)
            .push_constants_raw(&[43])
            .dispatch(4, 1, 1);

        let cmds = graph.compile_commands();

        // Wave 0: SetPipeline, SetPushConstantsRaw, Dispatch
        // ResourceBarrier
        // Wave 1: SetPipeline, SetPushConstantsRaw, Dispatch
        assert_eq!(cmds.len(), 7);
        assert!(matches!(cmds[0], ComputeCommand::SetPipeline(_)));
        assert!(matches!(
            cmds[1],
            ComputeCommand::SetPushConstantsRaw { .. }
        ));
        assert!(matches!(
            cmds[2],
            ComputeCommand::Dispatch {
                workgroups_x: 8,
                ..
            }
        ));
        assert!(matches!(cmds[3], ComputeCommand::ResourceBarrier { .. }));
        assert!(matches!(cmds[4], ComputeCommand::SetPipeline(_)));
        assert!(matches!(
            cmds[5],
            ComputeCommand::SetPushConstantsRaw { .. }
        ));
        assert!(matches!(
            cmds[6],
            ComputeCommand::Dispatch {
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

        let mut graph = ComputeGraph::new();
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
            .any(|c| matches!(c, ComputeCommand::ResourceBarrier { .. })));
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, ComputeCommand::Dispatch { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn compile_empty_graph() {
        let graph = ComputeGraph::new();
        let cmds = graph.compile_commands();
        assert!(cmds.is_empty());
    }

    #[test]
    fn submit_via_mock() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = ComputeGraph::new();
        graph
            .node("work", &pipeline)
            .bind_buffer(&buf, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);

        let future = graph.submit(&device).unwrap();
        assert!(future.is_complete());
        future.wait().unwrap();
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

        let mut graph = ComputeGraph::new();
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
            .filter(|c| matches!(c, ComputeCommand::ResourceBarrier { .. }))
            .count();
        assert_eq!(barrier_count, 2);

        let dispatch_count = cmds
            .iter()
            .filter(|c| matches!(c, ComputeCommand::Dispatch { .. }))
            .count();
        assert_eq!(dispatch_count, 4);
    }

    #[test]
    fn len_and_is_empty() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut graph = ComputeGraph::new();
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
    // Category F: ComputeGraph + MockBackend with real BufferPool / BufferView
    // -------------------------------------------------------------------------

    /// Create a pool of `total_size` bytes and return it together with
    /// a device, shader, and pipeline for convenience.
    fn make_pool_setup(
        total_size: u64,
    ) -> (Device, ShaderModule, crate::compute::ComputePipeline, BufferPool) {
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

        let mut graph = ComputeGraph::new();
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
            .any(|c| matches!(c, ComputeCommand::ResourceBarrier { .. })));
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, ComputeCommand::Dispatch { .. }))
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

        let mut graph = ComputeGraph::new();
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
                .filter(|c| matches!(c, ComputeCommand::ResourceBarrier { .. }))
                .count(),
            1
        );
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, ComputeCommand::Dispatch { .. }))
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

        let mut graph = ComputeGraph::new();
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
            .filter(|c| matches!(c, ComputeCommand::ResourceBarrier { .. }))
            .count();
        assert_eq!(barrier_count, 1);

        let dispatch_count = cmds
            .iter()
            .filter(|c| matches!(c, ComputeCommand::Dispatch { .. }))
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

        let mut graph = ComputeGraph::new();
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
            .filter(|c| matches!(c, ComputeCommand::ResourceBarrier { .. }))
            .count();
        assert_eq!(barrier_count, 2);

        let dispatch_count = cmds
            .iter()
            .filter(|c| matches!(c, ComputeCommand::Dispatch { .. }))
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

        let mut graph = ComputeGraph::new();
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
                .filter(|c| matches!(c, ComputeCommand::ResourceBarrier { .. }))
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

        let mut graph = ComputeGraph::new();
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
                .any(|c| matches!(c, ComputeCommand::ResourceBarrier { .. })),
            "10 independent pool views should produce zero barriers"
        );
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, ComputeCommand::Dispatch { .. }))
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

        let mut graph = ComputeGraph::new();
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
            .find(|c| matches!(c, ComputeCommand::ResourceBarrier { .. }))
            .expect("expected a barrier");

        if let ComputeCommand::ResourceBarrier { buffers, .. } = barrier {
            assert!(
                buffers.contains(&parent_handle),
                "barrier should reference the parent buffer handle {}, got {:?}",
                parent_handle,
                buffers
            );
        }
    }

    #[test]
    fn prelude_clear_then_pool_view_dispatch_no_spurious_barrier() {
        // A ClearBuffer prelude on the backing buffer followed by pool-view dispatches.
        // Two independent writes should still be in one wave.
        let (device, shader, _, mut pool) = make_pool_setup(1024);
        let pipeline = mock_pipeline(&device, &shader);

        let view_a = pool.alloc::<u32>(64).unwrap();
        let view_b = pool.alloc::<u32>(64).unwrap();
        let backing_handle = pool.backing_buffer().gpu_buffer_handle();

        let mut graph = ComputeGraph::new();
        // Inject a clear as a prelude command (not a node)
        graph.prelude.push(ComputeCommand::ClearBuffer {
            buffer: backing_handle,
            offset: 0,
            size: 1024,
        });

        graph
            .node("write_a", &pipeline)
            .bind_buffer_view(&view_a, NodeAccess::Write)
            .dispatch(1, 1, 1);
        graph
            .node("write_b", &pipeline)
            .bind_buffer_view(&view_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let cmds = graph.compile_commands();

        // Prelude ClearBuffer comes first, then both dispatches in one wave
        assert!(matches!(cmds[0], ComputeCommand::ClearBuffer { .. }));
        let barrier_count = cmds
            .iter()
            .filter(|c| matches!(c, ComputeCommand::ResourceBarrier { .. }))
            .count();
        assert_eq!(barrier_count, 0, "disjoint views should not trigger a barrier");
    }
}
