//! Render pipeline management.

use crate::backend::{GpuBackend, PipelineHandle, BindGroupLayoutHandle};
use crate::bind_group::BindGroupLayout;
use crate::device::Device;
use crate::shader::ShaderModule;
use crate::types::*;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Description for creating a render pipeline.
///
/// # Format Matching
///
/// **Important**: `target_format` must match the format of the render target or surface
/// you will render to. Mismatched formats cause undefined behavior or errors.
///
/// - For `Surface`: use `surface.format()`
/// - For `RenderTarget`: use the format passed to `RenderTarget::new()`
///
/// # Example
///
/// ```rust,no_run
/// # use rag::{RenderPipelineDesc, Surface, TextureFormat};
/// # fn example(surface: &Surface) {
/// let desc = RenderPipelineDesc {
///     target_format: surface.format(), // Always match the target!
///     ..Default::default()
/// };
/// # }
/// ```
#[derive(Clone, Default)]
pub struct RenderPipelineDesc<'a> {
    /// Vertex buffer layout.
    pub vertex_layout: VertexBufferLayout,
    /// Primitive topology.
    pub topology: PrimitiveTopology,
    /// Target texture format.
    ///
    /// **Must match** the format of the Surface or RenderTarget you render to.
    /// Use `surface.format()` or the format you passed to `RenderTarget::new()`.
    pub target_format: TextureFormat,
    /// Bind group layouts used by this pipeline (optional).
    /// The order determines the set index (first = set 0, second = set 1, etc.)
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
    /// Depth/stencil state (optional, None = no depth testing).
    pub depth_stencil: Option<DepthStencilState>,
}

impl<'a> std::fmt::Debug for RenderPipelineDesc<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderPipelineDesc")
            .field("vertex_layout", &self.vertex_layout)
            .field("topology", &self.topology)
            .field("target_format", &self.target_format)
            .field("bind_group_layouts_count", &self.bind_group_layouts.len())
            .field("depth_stencil", &self.depth_stencil)
            .finish()
    }
}

impl Default for VertexBufferLayout {
    fn default() -> Self {
        Vertex2D::layout()
    }
}

impl Default for TextureFormat {
    fn default() -> Self {
        TextureFormat::Rgba8Unorm
    }
}

/// A render pipeline.
pub struct RenderPipeline {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: PipelineHandle,
}

impl RenderPipeline {
    /// Create a new render pipeline.
    pub fn new(
        device: &Device,
        vertex_shader: &ShaderModule,
        fragment_shader: &ShaderModule,
        desc: &RenderPipelineDesc,
    ) -> Result<Self> {
        let mut backend = device.backend.lock().unwrap();
        
        // Collect bind group layout handles
        let layout_handles: Vec<BindGroupLayoutHandle> = desc
            .bind_group_layouts
            .iter()
            .map(|l| l.handle)
            .collect();
        
        let handle = if desc.depth_stencil.is_some() || !desc.bind_group_layouts.is_empty() {
            // Use extended pipeline creation with depth/stencil support
            backend.create_pipeline_with_depth(
                device.handle,
                vertex_shader.handle,
                fragment_shader.handle,
                &desc.vertex_layout,
                desc.topology,
                desc.target_format,
                &layout_handles,
                desc.depth_stencil.as_ref(),
            )?
        } else {
            // Simple pipeline creation (no bind groups, no depth)
            backend.create_pipeline(
                device.handle,
                vertex_shader.handle,
                fragment_shader.handle,
                &desc.vertex_layout,
                desc.topology,
                desc.target_format,
            )?
        };

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
        })
    }
}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_pipeline(self.handle);
    }
}

