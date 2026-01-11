//! Command encoding for GPU operations.

use crate::backend::RenderCommand;
use crate::bind_group::BindGroup;
use crate::buffer::Buffer;
use crate::pipeline::RenderPipeline;
use crate::types::{Color, IndexFormat};

/// Command encoder for recording GPU commands.
///
/// `CommandEncoder` is completely lock-free and does not interact with the GPU backend.
/// You can create and record commands on any thread. The actual GPU operations happen
/// when you submit the commands via [`RenderTarget::render()`](crate::RenderTarget::render)
/// or [`SurfaceFrame::render()`](crate::SurfaceFrame::render).
///
/// # Example
///
/// ```rust,no_run
/// use goldy::{CommandEncoder, Color};
///
/// let mut encoder = CommandEncoder::new();
/// let mut pass = encoder.begin_render_pass();
/// pass.clear(Color::CORNFLOWER_BLUE);
/// // ... more commands
/// drop(pass);
///
/// let commands = encoder.finish();
/// // Submit to render target or surface
/// ```
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
    /// Clear the color render target to a color.
    pub fn clear(&mut self, color: Color) {
        self.encoder.commands.push(RenderCommand::Clear(color));
    }

    /// Clear the depth buffer to a value.
    ///
    /// The default depth clear value is 1.0 (far plane).
    /// Use 0.0 for reverse-Z depth buffers.
    pub fn clear_depth(&mut self, depth: f32) {
        self.encoder.commands.push(RenderCommand::ClearDepth(depth));
    }

    /// Set the active render pipeline.
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        self.encoder
            .commands
            .push(RenderCommand::SetPipeline(pipeline.handle));
    }

    /// Set a vertex buffer.
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer) {
        self.encoder.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.handle,
            offset: 0,
        });
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

    /// Set an index buffer for indexed drawing.
    ///
    /// The buffer must have been created with `BufferUsage::INDEX`.
    pub fn set_index_buffer(&mut self, buffer: &Buffer, format: IndexFormat) {
        self.encoder.commands.push(RenderCommand::SetIndexBuffer {
            buffer: buffer.handle,
            offset: 0,
            format,
        });
    }

    /// Set an index buffer with an offset.
    pub fn set_index_buffer_offset(&mut self, buffer: &Buffer, offset: u64, format: IndexFormat) {
        self.encoder.commands.push(RenderCommand::SetIndexBuffer {
            buffer: buffer.handle,
            offset,
            format,
        });
    }

    /// Draw indexed primitives.
    ///
    /// Requires a prior call to `set_index_buffer()`.
    ///
    /// # Parameters
    /// - `indices`: Range of indices to draw
    /// - `base_vertex`: Value added to each index before fetching the vertex
    /// - `instances`: Range of instances to draw
    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        self.encoder.commands.push(RenderCommand::DrawIndexed {
            index_count: indices.end - indices.start,
            instance_count: instances.end - instances.start,
            first_index: indices.start,
            base_vertex,
            first_instance: instances.start,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_encoder_creation() {
        let encoder = CommandEncoder::new();
        assert!(encoder.commands.is_empty());
    }

    #[test]
    fn test_command_encoder_default() {
        let encoder = CommandEncoder::default();
        assert!(encoder.commands.is_empty());
    }

    #[test]
    fn test_clear_command() {
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::RED);
        }
        let commands = encoder.finish();

        assert_eq!(commands.len(), 1);
        match &commands[0] {
            RenderCommand::Clear(color) => {
                assert_eq!(color.r, 1.0);
                assert_eq!(color.g, 0.0);
                assert_eq!(color.b, 0.0);
            }
            _ => panic!("Expected Clear command"),
        }
    }

    #[test]
    fn test_draw_command() {
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.draw(0..6, 0..1);
        }
        let commands = encoder.finish();

        assert_eq!(commands.len(), 1);
        match &commands[0] {
            RenderCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => {
                assert_eq!(*vertex_count, 6);
                assert_eq!(*instance_count, 1);
                assert_eq!(*first_vertex, 0);
                assert_eq!(*first_instance, 0);
            }
            _ => panic!("Expected Draw command"),
        }
    }

    #[test]
    fn test_draw_with_offset() {
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.draw(10..16, 5..15);
        }
        let commands = encoder.finish();

        match &commands[0] {
            RenderCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => {
                assert_eq!(*vertex_count, 6);
                assert_eq!(*instance_count, 10);
                assert_eq!(*first_vertex, 10);
                assert_eq!(*first_instance, 5);
            }
            _ => panic!("Expected Draw command"),
        }
    }

    #[test]
    fn test_draw_indexed_command() {
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.draw_indexed(0..6, 0, 0..1);
        }
        let commands = encoder.finish();

        assert_eq!(commands.len(), 1);
        match &commands[0] {
            RenderCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => {
                assert_eq!(*index_count, 6);
                assert_eq!(*instance_count, 1);
                assert_eq!(*first_index, 0);
                assert_eq!(*base_vertex, 0);
                assert_eq!(*first_instance, 0);
            }
            _ => panic!("Expected DrawIndexed command"),
        }
    }

    #[test]
    fn test_draw_indexed_with_base_vertex() {
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            // Draw indices 100-106, with base vertex offset of 1000, instances 0-5
            pass.draw_indexed(100..106, 1000, 0..5);
        }
        let commands = encoder.finish();

        match &commands[0] {
            RenderCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => {
                assert_eq!(*index_count, 6);
                assert_eq!(*instance_count, 5);
                assert_eq!(*first_index, 100);
                assert_eq!(*base_vertex, 1000);
                assert_eq!(*first_instance, 0);
            }
            _ => panic!("Expected DrawIndexed command"),
        }
    }

    #[test]
    fn test_draw_indexed_negative_base_vertex() {
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.draw_indexed(0..3, -50, 0..1);
        }
        let commands = encoder.finish();

        match &commands[0] {
            RenderCommand::DrawIndexed { base_vertex, .. } => {
                assert_eq!(*base_vertex, -50);
            }
            _ => panic!("Expected DrawIndexed command"),
        }
    }

    #[test]
    fn test_multiple_commands() {
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.draw(0..3, 0..1);
            pass.draw_indexed(0..6, 0, 0..1);
        }
        let commands = encoder.finish();

        assert_eq!(commands.len(), 3);
        assert!(matches!(&commands[0], RenderCommand::Clear(_)));
        assert!(matches!(&commands[1], RenderCommand::Draw { .. }));
        assert!(matches!(&commands[2], RenderCommand::DrawIndexed { .. }));
    }
}
