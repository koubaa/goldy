//! Python wrapper for CommandEncoder and RenderPass.

use crate::buffer::PyBuffer;
use crate::pipeline::PyRenderPipeline;
use crate::types::{PyColor, PyIndexFormat};
use pyo3::prelude::*;
use std::ops::Range;
use std::sync::Mutex;

/// Command encoder for recording GPU commands.
///
/// CommandEncoder is used to record rendering commands that will be
/// executed on the GPU. Use `begin_render_pass()` to start recording.
///
/// Example:
///     >>> encoder = goldy.CommandEncoder()
///     >>> with encoder.begin_render_pass() as rp:
///     ...     rp.clear(goldy.Color.CORNFLOWER_BLUE)
///     ...     rp.set_pipeline(pipeline)
///     ...     rp.draw(range(3))
///     >>> target.render(encoder)
#[pyclass(name = "CommandEncoder", module = "goldy")]
pub struct PyCommandEncoder {
    inner: Mutex<Option<goldy::CommandEncoder>>,
}

#[pymethods]
impl PyCommandEncoder {
    /// Create a new command encoder.
    #[new]
    fn new() -> Self {
        PyCommandEncoder {
            inner: Mutex::new(Some(goldy::CommandEncoder::new())),
        }
    }

    /// Begin a render pass.
    ///
    /// Returns a RenderPass that can be used as a context manager.
    ///
    /// Returns:
    ///     A RenderPass for recording draw commands.
    fn begin_render_pass(slf: Py<Self>) -> PyRenderPass {
        PyRenderPass { encoder: slf }
    }

    fn __repr__(&self) -> String {
        let has_encoder = self.inner.lock().unwrap().is_some();
        format!("CommandEncoder(active={})", has_encoder)
    }
}

impl PyCommandEncoder {
    /// Take the inner encoder (consumes it).
    pub fn take_inner(&self) -> goldy::CommandEncoder {
        self.inner
            .lock()
            .unwrap()
            .take()
            .expect("CommandEncoder already consumed")
    }

    /// Get mutable access to the encoder for recording commands.
    fn with_encoder<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut goldy::CommandEncoder) -> R,
    {
        let mut guard = self.inner.lock().unwrap();
        let encoder = guard.as_mut().expect("CommandEncoder already consumed");
        f(encoder)
    }
}

/// A render pass for drawing operations.
///
/// Use as a context manager to ensure proper cleanup:
///
///     >>> with encoder.begin_render_pass() as rp:
///     ...     rp.clear(goldy.Color.RED)
///     ...     rp.draw(range(3))
#[pyclass(name = "RenderPass", module = "goldy")]
pub struct PyRenderPass {
    encoder: Py<PyCommandEncoder>,
}

#[pymethods]
impl PyRenderPass {
    /// Clear the color render target to a color.
    fn clear(&self, py: Python<'_>, color: &PyColor) {
        self.encoder.borrow(py).with_encoder(|enc| {
            let mut pass = enc.begin_render_pass();
            pass.clear(color.inner);
        });
    }

    /// Clear the depth buffer to a value.
    ///
    /// The default depth clear value is 1.0 (far plane).
    /// Use 0.0 for reverse-Z depth buffers.
    fn clear_depth(&self, py: Python<'_>, depth: f32) {
        self.encoder.borrow(py).with_encoder(|enc| {
            let mut pass = enc.begin_render_pass();
            pass.clear_depth(depth);
        });
    }

    /// Set the active render pipeline.
    fn set_pipeline(&self, py: Python<'_>, pipeline: &PyRenderPipeline) {
        self.encoder.borrow(py).with_encoder(|enc| {
            let mut pass = enc.begin_render_pass();
            pass.set_pipeline(&pipeline.inner);
        });
    }

    /// Set a vertex buffer.
    ///
    /// Args:
    ///     slot: The vertex buffer slot (usually 0).
    ///     buffer: The vertex buffer.
    fn set_vertex_buffer(&self, py: Python<'_>, slot: u32, buffer: &PyBuffer) {
        self.encoder.borrow(py).with_encoder(|enc| {
            let mut pass = enc.begin_render_pass();
            pass.set_vertex_buffer(slot, buffer.inner.as_ref());
        });
    }

    /// Set an index buffer for indexed drawing.
    ///
    /// Args:
    ///     buffer: The index buffer.
    ///     format: Index format (UINT16 or UINT32).
    #[pyo3(signature = (buffer, format=PyIndexFormat::UINT16))]
    fn set_index_buffer(&self, py: Python<'_>, buffer: &PyBuffer, format: PyIndexFormat) {
        self.encoder.borrow(py).with_encoder(|enc| {
            let mut pass = enc.begin_render_pass();
            pass.set_index_buffer(buffer.inner.as_ref(), format.into());
        });
    }

    /// Set push constants for resource binding.
    ///
    /// Pass the buffers whose indices should be pushed to the shader.
    /// The indices are pushed in order, so `buffers[0]` becomes index 0,
    /// `buffers[1]` becomes index 1, etc.
    ///
    /// Args:
    ///     buffers: List of buffers to pass to the shader via push constants.
    ///
    /// Example:
    ///     >>> rp.set_push_constants([uniform_buffer])
    ///     # In shader: g_UniformBuffers[getBufferIndex(0)].time
    fn set_push_constants(&self, py: Python<'_>, buffers: Vec<PyRef<'_, PyBuffer>>) {
        self.encoder.borrow(py).with_encoder(|enc| {
            // Collect buffer references - deref the Arc to get &Buffer
            let buffer_refs: Vec<&goldy::Buffer> =
                buffers.iter().map(|b| b.inner.as_ref()).collect();

            let mut pass = enc.begin_render_pass();
            pass.set_push_constants(&buffer_refs);
        });
    }

    /// Draw primitives.
    ///
    /// Args:
    ///     vertices: Range of vertices to draw (e.g., range(3) for a triangle).
    ///     instances: Range of instances to draw (default: range(1)).
    ///
    /// Example:
    ///     >>> rp.draw(range(3))  # Draw 3 vertices
    ///     >>> rp.draw(range(6), range(10))  # Draw 6 vertices, 10 instances
    #[pyo3(signature = (vertices, instances=None))]
    fn draw(
        &self,
        py: Python<'_>,
        vertices: &Bound<'_, PyAny>,
        instances: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let vertex_range = extract_range(vertices)?;
        let instance_range = if let Some(inst) = instances {
            extract_range(inst)?
        } else {
            0..1
        };

        self.encoder.borrow(py).with_encoder(|enc| {
            let mut pass = enc.begin_render_pass();
            pass.draw(vertex_range, instance_range);
        });
        Ok(())
    }

    /// Draw indexed primitives.
    ///
    /// Requires a prior call to `set_index_buffer()`.
    ///
    /// Args:
    ///     indices: Range of indices to draw.
    ///     base_vertex: Value added to each index before fetching the vertex.
    ///     instances: Range of instances to draw (default: range(1)).
    #[pyo3(signature = (indices, base_vertex=0, instances=None))]
    fn draw_indexed(
        &self,
        py: Python<'_>,
        indices: &Bound<'_, PyAny>,
        base_vertex: i32,
        instances: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let index_range = extract_range(indices)?;
        let instance_range = if let Some(inst) = instances {
            extract_range(inst)?
        } else {
            0..1
        };

        self.encoder.borrow(py).with_encoder(|enc| {
            let mut pass = enc.begin_render_pass();
            pass.draw_indexed(index_range, base_vertex, instance_range);
        });
        Ok(())
    }

    // Context manager support
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        // Don't suppress exceptions
        false
    }

    fn __repr__(&self) -> String {
        "RenderPass()".to_string()
    }
}

/// Extract a range from a Python range object or tuple.
fn extract_range(obj: &Bound<'_, PyAny>) -> PyResult<Range<u32>> {
    // Try Python range object
    if let Ok(range) = obj.cast::<pyo3::types::PySlice>() {
        let indices = range.indices(i32::MAX as isize)?;
        return Ok(indices.start as u32..indices.stop as u32);
    }

    // Try as range() object
    if obj.hasattr("start")? && obj.hasattr("stop")? {
        let start: u32 = obj.getattr("start")?.extract()?;
        let stop: u32 = obj.getattr("stop")?.extract()?;
        return Ok(start..stop);
    }

    // Try as tuple (start, stop)
    if let Ok(tuple) = obj.cast::<pyo3::types::PyTuple>() {
        if tuple.len() == 2 {
            let start: u32 = tuple.get_item(0)?.extract()?;
            let stop: u32 = tuple.get_item(1)?.extract()?;
            return Ok(start..stop);
        }
    }

    // Try as single integer (0..n)
    if let Ok(count) = obj.extract::<u32>() {
        return Ok(0..count);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected range(), tuple (start, stop), or integer",
    ))
}
