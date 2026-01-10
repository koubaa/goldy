//! Render pipeline management.

use crate::backend::{GpuBackend, PipelineHandle, BindGroupLayoutHandle};
use crate::bind_group::BindGroupLayout;
use crate::device::Device;
use crate::shader::ShaderModule;
use crate::types::*;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Description for creating a render pipeline.
#[derive(Clone, Default)]
pub struct RenderPipelineDesc<'a> {
    /// Vertex buffer layout.
    pub vertex_layout: VertexBufferLayout,
    /// Primitive topology.
    pub topology: PrimitiveTopology,
    /// Target texture format.
    pub target_format: TextureFormat,
    /// Bind group layouts used by this pipeline (optional).
    /// The order determines the set index (first = set 0, second = set 1, etc.)
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
}

impl<'a> std::fmt::Debug for RenderPipelineDesc<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderPipelineDesc")
            .field("vertex_layout", &self.vertex_layout)
            .field("topology", &self.topology)
            .field("target_format", &self.target_format)
            .field("bind_group_layouts_count", &self.bind_group_layouts.len())
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
        
        let handle = if desc.bind_group_layouts.is_empty() {
            // No bind group layouts - use simple pipeline creation
            backend.create_pipeline(
                device.handle,
                vertex_shader.handle,
                fragment_shader.handle,
                &desc.vertex_layout,
                desc.topology,
                desc.target_format,
            )?
        } else {
            // Has bind group layouts - use extended pipeline creation
            let layout_handles: Vec<BindGroupLayoutHandle> = desc
                .bind_group_layouts
                .iter()
                .map(|l| l.handle)
                .collect();
            
            backend.create_pipeline_with_layout(
                device.handle,
                vertex_shader.handle,
                fragment_shader.handle,
                &desc.vertex_layout,
                desc.topology,
                desc.target_format,
                &layout_handles,
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

