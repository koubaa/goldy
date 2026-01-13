//! Python wrapper for RenderPipeline.

use crate::bind_group::PyBindGroupLayout;
use crate::device::PyDevice;
use crate::error::IntoPyResult;
use crate::shader::PyShaderModule;
use crate::types::{
    PyDepthStencilState, PyPrimitiveTopology, PyTextureFormat, PyVertexBufferLayout,
};
use pyo3::prelude::*;
use std::sync::Arc;

/// Description for creating a render pipeline.
#[pyclass(name = "RenderPipelineDesc", module = "goldy")]
#[derive(Clone)]
pub struct PyRenderPipelineDesc {
    /// Vertex buffer layout.
    pub vertex_layout: Option<PyVertexBufferLayout>,
    /// Primitive topology.
    pub topology: PyPrimitiveTopology,
    /// Target texture format.
    pub target_format: PyTextureFormat,
    /// Depth/stencil state (optional).
    pub depth_stencil: Option<PyDepthStencilState>,
    /// Bind group layouts.
    pub bind_group_layouts: Vec<Arc<goldy::BindGroupLayout>>,
}

#[pymethods]
impl PyRenderPipelineDesc {
    /// Create a new render pipeline description.
    ///
    /// Args:
    ///     vertex_layout: Vertex buffer layout (default: Vertex2D layout).
    ///     topology: Primitive topology (default: TRIANGLE_LIST).
    ///     target_format: Target texture format (default: RGBA8_UNORM).
    ///     depth_stencil: Optional depth/stencil state.
    ///     bind_group_layouts: Optional list of bind group layouts.
    #[new]
    #[pyo3(signature = (vertex_layout=None, topology=PyPrimitiveTopology::TRIANGLE_LIST, target_format=PyTextureFormat::RGBA8_UNORM, depth_stencil=None, bind_group_layouts=None))]
    fn new(
        vertex_layout: Option<PyVertexBufferLayout>,
        topology: PyPrimitiveTopology,
        target_format: PyTextureFormat,
        depth_stencil: Option<PyDepthStencilState>,
        bind_group_layouts: Option<Vec<PyRef<PyBindGroupLayout>>>,
    ) -> Self {
        PyRenderPipelineDesc {
            vertex_layout,
            topology,
            target_format,
            depth_stencil,
            bind_group_layouts: bind_group_layouts
                .map(|layouts| layouts.iter().map(|l| Arc::clone(&l.inner)).collect())
                .unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RenderPipelineDesc(topology={:?}, target_format={:?})",
            self.topology, self.target_format
        )
    }
}

/// A render pipeline.
///
/// Defines the complete rendering state: shaders, vertex layout, and output format.
#[pyclass(name = "RenderPipeline", module = "goldy")]
pub struct PyRenderPipeline {
    pub(crate) inner: Arc<goldy::RenderPipeline>,
}

#[pymethods]
impl PyRenderPipeline {
    /// Create a new render pipeline.
    ///
    /// Args:
    ///     device: The GPU device.
    ///     vertex_shader: The vertex shader module.
    ///     fragment_shader: The fragment shader module.
    ///     desc: Pipeline description.
    ///
    /// Returns:
    ///     A new RenderPipeline instance.
    ///
    /// Raises:
    ///     GoldyError: If pipeline creation fails.
    #[new]
    fn new(
        device: &PyDevice,
        vertex_shader: &PyShaderModule,
        fragment_shader: &PyShaderModule,
        desc: &PyRenderPipelineDesc,
    ) -> PyResult<Self> {
        let vertex_layout = desc
            .vertex_layout
            .as_ref()
            .map(|l| l.inner.clone())
            .unwrap_or_else(goldy::Vertex2D::layout);

        // Create temporary references to BindGroupLayout
        let layout_refs: Vec<&goldy::BindGroupLayout> =
            desc.bind_group_layouts.iter().map(|l| l.as_ref()).collect();

        let rust_desc = goldy::RenderPipelineDesc {
            vertex_layout,
            topology: desc.topology.into(),
            target_format: desc.target_format.into(),
            bind_group_layouts: &layout_refs,
            depth_stencil: desc.depth_stencil.as_ref().map(|ds| ds.inner.clone()),
        };

        let pipeline = goldy::RenderPipeline::new(
            &device.inner,
            &vertex_shader.inner,
            &fragment_shader.inner,
            &rust_desc,
        )
        .into_py_result()?;

        Ok(PyRenderPipeline {
            inner: Arc::new(pipeline),
        })
    }

    fn __repr__(&self) -> String {
        "RenderPipeline()".to_string()
    }
}
