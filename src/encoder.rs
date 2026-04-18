//! Command encoding for GPU operations.

use crate::backend::RenderCommand;
use crate::buffer::{Buffer, BufferSource};
use crate::pipeline::RenderPipeline;
use crate::types::{BindlessHandle, Color, IndexFormat};

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
    ///
    /// Accepts either a [`Buffer`] or [`crate::BufferView`]; for pool-allocated views,
    /// the parent buffer and offset are resolved automatically.
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &impl BufferSource) {
        self.encoder.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.source_handle(),
            offset: buffer.source_offset(),
        });
    }

    /// Set a vertex buffer with an additional offset.
    pub fn set_vertex_buffer_offset(&mut self, slot: u32, buffer: &impl BufferSource, offset: u64) {
        self.encoder.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.source_handle(),
            offset: buffer.source_offset() + offset,
        });
    }

    /// Set push constants for resource binding.
    ///
    /// Pass the buffers whose indices should be pushed to the shader.
    /// The indices are pushed in order, so `buffers[0]` becomes index 0,
    /// `buffers[1]` becomes index 1, etc.
    ///
    /// # Example
    /// ```ignore
    /// pass.set_push_constants(&[&uniform_buffer]);
    /// // In shader: g_UniformBuffers[getBufferIndex(0)].time
    /// ```
    pub fn set_push_constants(&mut self, buffers: &[&Buffer]) {
        self.encoder.commands.push(RenderCommand::SetPushConstants {
            buffers: buffers.iter().map(|b| b.handle).collect(),
        });
    }

    /// Set push constants with raw u32 indices.
    ///
    /// **Prefer [`RenderPass::set_push_constants_typed`]** for new code — the
    /// raw form bypasses per-slot category validation.
    ///
    /// # Example
    /// ```ignore
    /// let tex_idx = texture.bindless_index().unwrap();
    /// let samp_idx = sampler.bindless_index().unwrap();
    /// pass.set_push_constants_raw(&[tex_idx, samp_idx]);
    /// // In shader: GET_TEXTURE() and GET_SAMPLER() macros use these indices
    /// ```
    pub fn set_push_constants_raw(&mut self, indices: &[u32]) {
        self.encoder
            .commands
            .push(RenderCommand::SetPushConstantsRaw {
                indices: indices.to_vec(),
            });
    }

    /// Set push constants from typed [`BindlessHandle`]s.
    ///
    /// Each handle carries both the raw index and the
    /// [`crate::types::BindlessCategory`] implied by the
    /// resource. At dispatch time the backend validates each slot against the
    /// bound shader's reflection and returns an error on mismatch.
    ///
    /// # Example
    /// ```ignore
    /// let tex = texture.bindless_handle().unwrap();  // Texture
    /// let samp = sampler.bindless_handle().unwrap(); // Sampler
    /// pass.set_push_constants_typed(&[tex, samp]);
    /// ```
    pub fn set_push_constants_typed(&mut self, handles: &[BindlessHandle]) {
        self.encoder
            .commands
            .push(RenderCommand::SetPushConstantsTyped {
                handles: handles.to_vec(),
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
    /// The buffer should contain index data (u16 or u32 values).
    /// Accepts either a [`Buffer`] or [`crate::BufferView`].
    pub fn set_index_buffer(&mut self, buffer: &impl BufferSource, format: IndexFormat) {
        self.encoder.commands.push(RenderCommand::SetIndexBuffer {
            buffer: buffer.source_handle(),
            offset: buffer.source_offset(),
            format,
        });
    }

    /// Set an index buffer with an additional offset.
    pub fn set_index_buffer_offset(
        &mut self,
        buffer: &impl BufferSource,
        offset: u64,
        format: IndexFormat,
    ) {
        self.encoder.commands.push(RenderCommand::SetIndexBuffer {
            buffer: buffer.source_handle(),
            offset: buffer.source_offset() + offset,
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

    // ========================================================================
    // Convenience methods for common draw patterns
    // ========================================================================

    /// Draw a fullscreen triangle (3 vertices, no vertex buffer needed).
    ///
    /// Use with `vs_fullscreen_triangle()` from `goldy_exp.vertex` or
    /// `fullscreen_position()`/`fullscreen_uv()` from `goldy_exp.primitives`.
    ///
    /// This is more efficient than a fullscreen quad (3 verts vs 6) and
    /// eliminates vertex buffer overhead entirely.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Shader uses: vs_fullscreen_triangle(SV_VertexID)
    /// pass.set_pipeline(&fullscreen_pipeline);
    /// pass.set_push_constants(&[&uniform_buffer]);
    /// pass.draw_fullscreen();  // No vertex buffer needed!
    /// ```
    pub fn draw_fullscreen(&mut self) {
        self.draw(0..3, 0..1);
    }

    /// Draw N instances of quads (6 vertices each, no vertex buffer needed).
    ///
    /// Use with `quad_position()` from `goldy_exp.primitives` in your shader.
    /// Each instance draws a quad; the shader reads instance data from a buffer.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Shader reads from buffer: instances[SV_InstanceID]
    /// // Uses: quad_position(SV_VertexID, instance.position, instance.size)
    /// pass.set_pipeline(&instanced_pipeline);
    /// pass.set_push_constants(&[&instance_buffer]);
    /// pass.draw_quads(400);  // Draw 400 quads
    /// ```
    pub fn draw_quads(&mut self, count: u32) {
        self.draw(0..6, 0..count);
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
    fn test_clear_depth_default() {
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear_depth(1.0);
        }
        let commands = encoder.finish();

        assert_eq!(commands.len(), 1);
        match &commands[0] {
            RenderCommand::ClearDepth(depth) => {
                assert!(
                    (*depth - 1.0).abs() < f32::EPSILON,
                    "Expected depth 1.0 (far plane), got {}",
                    depth
                );
            }
            _ => panic!("Expected ClearDepth command"),
        }
    }

    #[test]
    fn test_clear_depth_reverse_z() {
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            // Reverse-Z: clear to 0.0 (far becomes 0 in reverse-Z projection)
            pass.clear_depth(0.0);
        }
        let commands = encoder.finish();

        assert_eq!(commands.len(), 1);
        match &commands[0] {
            RenderCommand::ClearDepth(depth) => {
                assert!(
                    (*depth - 0.0).abs() < f32::EPSILON,
                    "Expected depth 0.0 (reverse-Z), got {}",
                    depth
                );
            }
            _ => panic!("Expected ClearDepth command"),
        }
    }

    #[test]
    fn test_clear_color_and_depth_together() {
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.clear_depth(1.0);
        }
        let commands = encoder.finish();

        assert_eq!(commands.len(), 2);
        assert!(matches!(&commands[0], RenderCommand::Clear(_)));
        assert!(matches!(&commands[1], RenderCommand::ClearDepth(_)));
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
