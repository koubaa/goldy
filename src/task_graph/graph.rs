//! `TaskGraph` — analyzed GPU task graph with automatic barrier insertion.

use super::analysis;
use super::ir::{DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding, TaskNode};
use super::ResourceId;
use crate::buffer::{Buffer, BufferView};
use crate::compute::ComputePipeline;
use crate::device::Device;
use crate::gpu_future::GpuFuture;
use crate::texture::Texture;
use anyhow::Result;

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
///     .bind_resources_raw(&[buf_idx])
///     .dispatch(64, 1, 1);
///
/// graph.node("read_data", &pipeline_b)
///     .bind_buffer(&buf, NodeAccess::Read)
///     .bind_resources_raw(&[buf_idx])
///     .dispatch(64, 1, 1);
///
/// graph.submit(&device)?.wait()?;
/// ```
pub struct TaskGraph {
    ir: GraphIR,
}

impl TaskGraph {
    pub fn new() -> Self {
        Self {
            ir: GraphIR::default(),
        }
    }

    /// Add a compute dispatch node to the graph. The returned [`NodeBuilder`] must
    /// be finalized with [`NodeBuilder::dispatch`] or [`NodeBuilder::dispatch_indirect`].
    pub fn node<'a>(&'a mut self, label: &str, pipeline: &ComputePipeline) -> NodeBuilder<'a> {
        NodeBuilder {
            graph: self,
            label: label.to_string(),
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
            label: "clear_buffer".to_string(),
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
            label: "clear_buffer_view".to_string(),
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
            label: "write_buffer".to_string(),
            bindings: vec![ResourceBinding {
                resource: ResourceId::Buffer(buffer.handle),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteBuffer {
                buffer: buffer.gpu_buffer_handle(),
                offset,
                data,
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
            label: "write_texture".to_string(),
            bindings: vec![ResourceBinding {
                resource: ResourceId::Texture(th),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteTexture {
                texture: th,
                data,
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
            label: "write_texture_region".to_string(),
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
                data,
            },
        });
        Ok(())
    }

    /// Analyze the graph and submit all tasks with optimal barriers.
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

    /// Number of task nodes in the graph.
    pub fn len(&self) -> usize {
        self.ir.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ir.nodes.is_empty()
    }

    /// Compile the graph into a flat command stream.
    ///
    /// Runs the dependency analyzer, schedules waves, inserts `ResourceBarrier`
    /// commands at wave boundaries, and emits the final [`GpuCommand`](crate::backend::GpuCommand) sequence.
    pub fn compile_commands(&self) -> Vec<crate::backend::GpuCommand> {
        let edges = analysis::build_edges(&self.ir);
        let schedule = analysis::schedule_waves(&self.ir, &edges);
        analysis::emit_commands(&self.ir, &schedule)
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
    label: String,
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

    /// Set the bindless resource slot indices for this node's dispatch (region A).
    pub fn bind_resources_raw(mut self, indices: &[u32]) -> Self {
        self.resource_slots = indices.to_vec();
        self
    }

    /// Set user scalar parameters for this node's dispatch (region B).
    pub fn bind_resources_raw_with_user(mut self, indices: &[u32], user: &[u32]) -> Self {
        self.resource_slots = indices.to_vec();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::backend::GpuCommand;
    use crate::buffer::BufferPool;
    use crate::device::Device;
    use crate::shader::ShaderModule;
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
            .bind_resources_raw(&[42])
            .dispatch(8, 1, 1);
        graph
            .node("read_write", &pipeline)
            .bind_buffer(&buf_a, NodeAccess::Read)
            .bind_buffer(&buf_b, NodeAccess::Write)
            .bind_resources_raw(&[43])
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

        let mut future = graph.submit(&device).unwrap();
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
