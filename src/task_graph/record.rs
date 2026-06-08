//! Owned render-pass recorder for language bindings (no lifetime coupling to [`TaskGraph`]).
//!
//! [`RenderPassRecord`] mirrors [`super::graph::RenderPassBuilder`] but can be held across
//! FFI calls while commands accumulate, then committed with [`RenderPassRecord::commit`].

use super::graph::TaskGraph;
use super::ir::{DispatchDim, NodeKind, NodeAccess, ResourceBinding, TaskNode};
use super::ResourceId;
use crate::backend::RenderCommand;
use crate::buffer::{Buffer, BufferSource, BufferView};
use crate::compute::ComputePipeline;
use crate::pipeline::RenderPipeline;
use crate::render_target::RenderTarget;
use crate::types::{Color, IndexFormat, ResourceHandle};

/// Accumulates one offscreen render-pass node before [`Self::commit`].
pub struct RenderPassRecord {
    label: &'static str,
    target: crate::backend::RenderTargetHandle,
    bindings: Vec<ResourceBinding>,
    commands: Vec<RenderCommand>,
}

impl RenderPassRecord {
    pub fn new(label: &'static str, target: &RenderTarget) -> Self {
        Self {
            label,
            target: target.backend_handle(),
            bindings: Vec::new(),
            commands: Vec::new(),
        }
    }

    pub fn bind_buffer(&mut self, buf: &Buffer, access: NodeAccess) -> &mut Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::Buffer(buf.handle),
            access,
        });
        self
    }

    pub fn bind_buffer_view(&mut self, view: &BufferView, access: NodeAccess) -> &mut Self {
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

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) -> &mut Self {
        self.commands
            .push(RenderCommand::SetPipeline(pipeline.handle));
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

    pub fn set_vertex_buffer_offset(&mut self, slot: u32, buffer: &Buffer, offset: u64) -> &mut Self {
        self.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.source_handle(),
            offset,
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

    pub fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) -> &mut Self {
        self.commands.push(RenderCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
        self
    }

    pub fn draw_indexed(
        &mut self,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) -> &mut Self {
        self.commands.push(RenderCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
        self
    }

    pub fn draw_fullscreen(&mut self) -> &mut Self {
        self.draw(0, 3, 0, 1)
    }

    pub fn draw_quads(&mut self, count: u32) -> &mut Self {
        self.draw(0, 6, 0, count)
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
        });
        self
    }

    pub fn bind_resources_typed(&mut self, handles: &[ResourceHandle]) -> &mut Self {
        self.commands.push(RenderCommand::BindResourcesTyped {
            handles: handles.to_vec(),
        });
        self
    }

    pub fn commit(self, graph: &mut TaskGraph) {
        graph.push_task_node(TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::RenderPass {
                target: self.target,
                commands: self.commands,
            },
        });
    }
}

/// Accumulates one compute dispatch node before [`Self::commit_dispatch`].
pub struct ComputeNodeRecord {
    label: &'static str,
    pipeline: crate::backend::ComputePipelineHandle,
    bindings: Vec<ResourceBinding>,
    resource_slots: Vec<u32>,
    user_slots: Vec<u32>,
}

impl ComputeNodeRecord {
    pub fn new(label: &'static str, pipeline: &ComputePipeline) -> Self {
        Self {
            label,
            pipeline: pipeline.handle,
            bindings: Vec::new(),
            resource_slots: Vec::new(),
            user_slots: Vec::new(),
        }
    }

    pub fn bind_buffer(&mut self, buf: &Buffer, access: NodeAccess) -> &mut Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::Buffer(buf.handle),
            access,
        });
        self
    }

    pub fn bind_buffer_view(&mut self, view: &BufferView, access: NodeAccess) -> &mut Self {
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

    pub fn bind_resources_raw(&mut self, indices: &[u32]) -> &mut Self {
        self.resource_slots = indices.to_vec();
        self
    }

    pub fn commit_dispatch(self, graph: &mut TaskGraph, x: u32, y: u32, z: u32) {
        graph.push_task_node(TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::Dispatch {
                pipeline: self.pipeline,
                resource_slots: self.resource_slots,
                user_slots: self.user_slots,
                dispatch: DispatchDim::Direct { x, y, z },
            },
        });
    }
}
