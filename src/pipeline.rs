//! Render pipeline management.

use crate::backend::{GpuBackend, PipelineHandle};
use crate::device::Device;
use crate::shader::ShaderModule;
use crate::types::*;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Description for creating a render pipeline.
#[derive(Debug, Clone)]
pub struct RenderPipelineDesc {
    /// Vertex buffer layout.
    pub vertex_layout: VertexBufferLayout,
    /// Primitive topology.
    pub topology: PrimitiveTopology,
    /// Target texture format.
    pub target_format: TextureFormat,
}

impl Default for RenderPipelineDesc {
    fn default() -> Self {
        Self {
            vertex_layout: Vertex2D::layout(),
            topology: PrimitiveTopology::TriangleList,
            target_format: TextureFormat::Rgba8Unorm,
        }
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
        let handle = backend.create_pipeline(
            device.handle,
            vertex_shader.handle,
            fragment_shader.handle,
            &desc.vertex_layout,
            desc.topology,
            desc.target_format,
        )?;

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

