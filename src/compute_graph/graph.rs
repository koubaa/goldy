//! `ComputeGraph` — Tier 1: interpreted, dynamic compute graph.

use super::analysis;
use super::ir::{GraphIR, GraphNode, NodeAccess, ResourceBinding};
use super::ResourceId;
use crate::buffer::Buffer;
use crate::compute::ComputePipeline;
use crate::device::Device;
use crate::gpu_future::GpuFuture;
use crate::texture::Texture;
use anyhow::Result;
use std::sync::Arc;

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
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self {
            ir: GraphIR::default(),
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
                workgroups: (0, 0, 0),
            },
        }
    }

    /// Analyze the graph and submit all dispatches with optimal barriers.
    /// Returns a [`GpuFuture`] for non-blocking completion.
    pub fn submit(&self, device: &Device) -> Result<GpuFuture> {
        let commands = self.compile_commands();
        let mut backend = device.backend.lock().unwrap();
        let token = backend.submit_compute(device.handle, &commands)?;
        Ok(GpuFuture {
            backend: Arc::clone(&device.backend),
            device: device.handle,
            fence_token: token,
        })
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
        self.ir.nodes.is_empty()
    }

    pub(crate) fn compile_commands(&self) -> Vec<crate::backend::ComputeCommand> {
        let edges = analysis::build_edges(&self.ir);
        let schedule = analysis::schedule_waves(&self.ir, &edges);
        analysis::emit_commands(&self.ir, &schedule)
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

    /// Finalize the node with the given workgroup dimensions.
    pub fn dispatch(mut self, x: u32, y: u32, z: u32) {
        self.node.workgroups = (x, y, z);
        self.graph.ir.nodes.push(self.node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::backend::ComputeCommand;
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
}
