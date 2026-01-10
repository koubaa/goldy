//! Command encoding for GPU operations.

use crate::backend::RenderCommand;
use crate::bind_group::BindGroup;
use crate::buffer::Buffer;
use crate::pipeline::RenderPipeline;
use crate::types::Color;

/// Command encoder for recording GPU commands.
pub struct CommandEncoder {
    pub(crate) commands: Vec<RenderCommand>,
}

impl CommandEncoder {
    /// Create a new command encoder.
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Begin a render pass.
    pub fn begin_render_pass(&mut self) -> RenderPass<'_> {
        RenderPass { encoder: self }
    }

    /// Get the recorded commands.
    pub fn finish(self) -> Vec<RenderCommand> {
        self.commands
    }
}

impl Default for CommandEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A render pass for drawing operations.
pub struct RenderPass<'a> {
    encoder: &'a mut CommandEncoder,
}

impl<'a> RenderPass<'a> {
    /// Clear the render target to a color.
    pub fn clear(&mut self, color: Color) {
        self.encoder.commands.push(RenderCommand::Clear(color));
    }

    /// Set the active render pipeline.
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        self.encoder.commands.push(RenderCommand::SetPipeline(pipeline.handle));
    }

    /// Set a vertex buffer.
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer) {
        self.encoder.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.handle,
            offset: 0,
        });
    }

    /// Set a vertex buffer (alias for set_vertex_buffer).
    pub fn set_vertex_buffer_raw(&mut self, slot: u32, buffer: &Buffer) {
        self.set_vertex_buffer(slot, buffer);
    }

    /// Set a vertex buffer with an offset.
    pub fn set_vertex_buffer_offset(&mut self, slot: u32, buffer: &Buffer, offset: u64) {
        self.encoder.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.handle,
            offset,
        });
    }

    /// Set a bind group for shader resources (uniforms, storage buffers).
    ///
    /// The `index` corresponds to the bind group set in the shader
    /// (e.g., `[[vk::binding(0, 0)]]` uses index 0).
    pub fn set_bind_group(&mut self, index: u32, bind_group: &BindGroup) {
        self.encoder.commands.push(RenderCommand::SetBindGroup {
            index,
            bind_group: bind_group.handle,
        });
    }

    /// Draw primitives.
    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        self.encoder.commands.push(RenderCommand::Draw {
            vertex_count: vertices.end - vertices.start,
            instance_count: instances.end - instances.start,
            first_vertex: vertices.start,
            first_instance: instances.start,
        });
    }
}

