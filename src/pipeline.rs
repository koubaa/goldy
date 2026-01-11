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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::bind_group::{BindGroupLayout, BindGroupLayoutBinding};

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
        assert!(desc.bind_group_layouts.is_empty());
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
        
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc::default(),
        ).unwrap();
        
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
        ).unwrap();
        
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
        ).unwrap();
        
        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_bind_group_layout() {
        let device = create_test_device();
        let shader = create_test_shader(&device);
        
        // Create a bind group layout
        let layout = BindGroupLayout::new(&device, &[
            BindGroupLayoutBinding::uniform(0),
        ]).unwrap();
        
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                bind_group_layouts: &[&layout],
                ..Default::default()
            },
        ).unwrap();
        
        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_multiple_bind_group_layouts() {
        let device = create_test_device();
        let shader = create_test_shader(&device);
        
        // Create multiple bind group layouts
        let layout0 = BindGroupLayout::new(&device, &[
            BindGroupLayoutBinding::uniform(0),
        ]).unwrap();
        
        let layout1 = BindGroupLayout::new(&device, &[
            BindGroupLayoutBinding::storage(0, true),
            BindGroupLayoutBinding::storage(1, false),
        ]).unwrap();
        
        let layout2 = BindGroupLayout::new(&device, &[
            BindGroupLayoutBinding::texture(0),
            BindGroupLayoutBinding::sampler(1),
        ]).unwrap();
        
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                bind_group_layouts: &[&layout0, &layout1, &layout2],
                ..Default::default()
            },
        ).unwrap();
        
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
        ).unwrap();
        
        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_bind_group_layouts_and_depth() {
        let device = create_test_device();
        let shader = create_test_shader(&device);
        
        let layout = BindGroupLayout::new(&device, &[
            BindGroupLayoutBinding::uniform(0),
        ]).unwrap();
        
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                bind_group_layouts: &[&layout],
                depth_stencil: Some(DepthStencilState {
                    format: DepthFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::LessEqual,
                }),
                ..Default::default()
            },
        ).unwrap();
        
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
        ).unwrap();
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
        ).unwrap();
        
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
        ).unwrap();
        
        let pipeline2 = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                target_format: TextureFormat::Bgra8UnormSrgb,
                ..Default::default()
            },
        ).unwrap();
        
        // Pipelines should have different handles
        assert_ne!(pipeline1.handle, pipeline2.handle);
    }
}

