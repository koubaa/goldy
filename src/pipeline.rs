//! Render pipeline management.

use crate::backend::{GpuBackend, PipelineHandle};
use crate::device::Device;
use crate::shader::ShaderModule;
use crate::types::*;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Description for creating a render pipeline.
///
/// # Format Matching
///
/// **Important**: `target_format` must match the format of the render target
/// you will render to. Mismatched formats cause undefined behavior or errors.
///
/// - For scheme-leased render targets: use the format passed to [`crate::Scheme::lease_render_target`]
///
/// # Example
///
/// ```rust,no_run
/// # use goldy::{RenderPipelineDesc, TextureFormat};
/// let desc = RenderPipelineDesc {
///     target_format: TextureFormat::Rgba8Unorm,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Default, Debug)]
pub struct RenderPipelineDesc {
    /// Vertex buffer layout.
    pub vertex_layout: VertexBufferLayout,
    /// Primitive topology.
    pub topology: PrimitiveTopology,
    /// Target texture format.
    ///
    /// **Must match** the format of the swapchain or scheme-leased render target you render to.
    /// Use the format passed to [`crate::Scheme::lease_render_target`].
    pub target_format: TextureFormat,
    /// Depth/stencil state (optional, None = no depth testing).
    pub depth_stencil: Option<DepthStencilState>,
}

impl Default for VertexBufferLayout {
    /// Returns an empty layout (no vertex attributes, stride 0).
    ///
    /// Use this for passes whose vertex shader generates geometry from `SV_VertexID`
    /// / `VertexId` without reading a vertex buffer (fullscreen quads, procedural
    /// triangles, etc.). For typed vertex input use `Vertex2D::layout()` or
    /// `VertexBufferLayout::from_formats::<T>(&[…])` explicitly.
    fn default() -> Self {
        Self {
            stride: 0,
            attributes: Vec::new(),
        }
    }
}

/// A render pipeline.
pub struct RenderPipeline {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: PipelineHandle,
    /// Per push-constant resource slot (shader-signature order), the descriptor
    /// access the shader signature requires. Used by [`crate::Scheme`] render-pass
    /// recording to pick the correct SRV/UAV descriptor independent of graph access.
    pub(crate) slot_access: Vec<Option<crate::types::ResourceAccess>>,
}

impl RenderPipeline {
    /// Create a new render pipeline.
    pub fn new(
        device: &Device,
        vertex_shader: &ShaderModule,
        fragment_shader: &ShaderModule,
        desc: &RenderPipelineDesc,
    ) -> Result<Self> {
        tracing::debug!(
            target_format = ?desc.target_format,
            topology = ?desc.topology,
            has_depth = desc.depth_stencil.is_some(),
            "Creating render pipeline"
        );

        let mut backend = device.inner.backend.lock().unwrap();

        let handle = if desc.depth_stencil.is_some() {
            backend.create_pipeline_with_depth(
                device.inner.handle,
                vertex_shader.handle,
                fragment_shader.handle,
                &desc.vertex_layout,
                desc.topology,
                desc.target_format,
                desc.depth_stencil.as_ref(),
            )?
        } else {
            backend.create_pipeline(
                device.inner.handle,
                vertex_shader.handle,
                fragment_shader.handle,
                &desc.vertex_layout,
                desc.topology,
                desc.target_format,
            )?
        };

        tracing::debug!("Render pipeline created");

        let slot_access = backend.render_pipeline_slot_access(handle);

        Ok(Self {
            _device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            slot_access,
        })
    }
}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        tracing::trace!("Destroying render pipeline");
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_pipeline(self.handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;

    fn create_test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn create_test_shader(device: &Device) -> ShaderModule {
        // Mock backend doesn't actually compile shaders, so any source works
        ShaderModule::from_slang(device, "mock shader source").unwrap()
    }

    #[test]
    fn test_render_pipeline_desc_default() {
        let desc = RenderPipelineDesc::default();

        assert_eq!(desc.target_format, TextureFormat::Rgba8Unorm);
        assert!(desc.depth_stencil.is_none());
    }

    #[test]
    fn test_render_pipeline_desc_debug() {
        let desc = RenderPipelineDesc {
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        };

        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("RenderPipelineDesc"));
        assert!(debug_str.contains("Rgba8Unorm"));
    }

    #[test]
    fn test_simple_pipeline_creation() {
        let device = create_test_device();
        let shader = create_test_shader(&device);

        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc::default()).unwrap();

        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_custom_format() {
        let device = create_test_device();
        let shader = create_test_shader(&device);

        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                target_format: TextureFormat::Bgra8UnormSrgb,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_topology() {
        let device = create_test_device();
        let shader = create_test_shader(&device);

        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_depth_stencil() {
        let device = create_test_device();
        let shader = create_test_shader(&device);

        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                depth_stencil: Some(DepthStencilState {
                    format: DepthFormat::Depth24Plus,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::Less,
                }),
                ..Default::default()
            },
        )
        .unwrap();

        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_vertex_layout() {
        use crate::types::Vertex2D;

        let device = create_test_device();
        let shader = create_test_shader(&device);

        // Test with Vertex2D layout
        let pipeline_2d = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: Vertex2D::layout(),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(pipeline_2d.handle > 0);
    }

    #[test]
    fn test_pipeline_different_vertex_fragment_shaders() {
        let device = create_test_device();

        // In real use, these would be different shaders
        let vertex_shader = create_test_shader(&device);
        let fragment_shader = create_test_shader(&device);

        let pipeline = RenderPipeline::new(
            &device,
            &vertex_shader,
            &fragment_shader,
            &RenderPipelineDesc::default(),
        )
        .unwrap();

        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_multiple_pipelines() {
        let device = create_test_device();
        let shader = create_test_shader(&device);

        // Create multiple pipelines with different configurations
        let pipeline1 = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                target_format: TextureFormat::Rgba8Unorm,
                ..Default::default()
            },
        )
        .unwrap();

        let pipeline2 = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                target_format: TextureFormat::Bgra8UnormSrgb,
                ..Default::default()
            },
        )
        .unwrap();

        // Pipelines should have different handles
        assert_ne!(pipeline1.handle, pipeline2.handle);
    }
}
