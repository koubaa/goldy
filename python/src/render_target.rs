//! Python wrapper for RenderTarget.

use crate::device::PyDevice;
use crate::encoder::PyCommandEncoder;
use crate::error::IntoPyResult;
use crate::types::{PyDepthFormat, PyTextureFormat};
use numpy::{PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use std::sync::Arc;

/// A GPU render target that stays on the GPU until explicitly read.
///
/// RenderTarget represents a GPU texture that can be rendered to.
/// Unlike a window surface, it does not automatically copy
/// pixels to CPU memory after rendering. This enables efficient
/// multi-consumer scenarios.
#[pyclass(name = "RenderTarget", module = "goldy")]
pub struct PyRenderTarget {
    inner: Arc<goldy::RenderTarget>,
}

#[pymethods]
impl PyRenderTarget {
    /// Create a new render target without a depth buffer.
    ///
    /// Args:
    ///     device: The GPU device.
    ///     width: Width in pixels.
    ///     height: Height in pixels.
    ///     format: Pixel format for the render target.
    ///
    /// Returns:
    ///     A new RenderTarget instance.
    #[new]
    #[pyo3(signature = (device, width, height, format=PyTextureFormat::RGBA8_UNORM))]
    fn new(device: &PyDevice, width: u32, height: u32, format: PyTextureFormat) -> PyResult<Self> {
        let target = goldy::RenderTarget::new(&device.inner, width, height, format.into()).into_py_result()?;
        Ok(PyRenderTarget {
            inner: Arc::new(target),
        })
    }

    /// Create a new render target with an optional depth buffer.
    ///
    /// Args:
    ///     device: The GPU device.
    ///     width: Width in pixels.
    ///     height: Height in pixels.
    ///     color_format: Pixel format for the color buffer.
    ///     depth_format: Depth buffer format.
    ///
    /// Returns:
    ///     A new RenderTarget instance with depth buffer.
    #[staticmethod]
    fn with_depth(
        device: &PyDevice,
        width: u32,
        height: u32,
        color_format: PyTextureFormat,
        depth_format: PyDepthFormat,
    ) -> PyResult<Self> {
        let target = goldy::RenderTarget::new_with_depth(
            &device.inner,
            width,
            height,
            color_format.into(),
            Some(depth_format.into()),
        )
        .into_py_result()?;
        Ok(PyRenderTarget {
            inner: Arc::new(target),
        })
    }

    /// Get the width in pixels.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Get the height in pixels.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    /// Returns true if this render target has a depth buffer.
    fn has_depth(&self) -> bool {
        self.inner.has_depth()
    }

    /// Get the size of the pixel data in bytes.
    #[getter]
    fn buffer_size(&self) -> usize {
        self.inner.buffer_size()
    }

    /// Render commands to this target.
    ///
    /// This executes the render commands and stores the result in the GPU texture.
    /// The data stays on the GPU - no CPU copy occurs.
    ///
    /// Args:
    ///     encoder: The command encoder containing render commands.
    ///
    /// Raises:
    ///     GoldyError: If rendering fails.
    fn render(&self, encoder: &PyCommandEncoder) -> PyResult<()> {
        // Take the encoder's commands
        let rust_encoder = encoder.take_inner();
        self.inner.render(rust_encoder).into_py_result()
    }

    /// Read the rendered pixels to a numpy array.
    ///
    /// This performs a GPU-to-CPU copy, which may stall the pipeline.
    /// Only call this when you actually need the pixel data on the CPU.
    ///
    /// Returns:
    ///     A numpy array with shape (height, width, 4) and dtype uint8.
    ///
    /// Raises:
    ///     GoldyError: If the GPU-to-CPU copy fails.
    fn read_to_cpu<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let pixels = self.inner.read_to_cpu().into_py_result()?;
        let height = self.inner.height() as usize;
        let width = self.inner.width() as usize;

        // Create numpy array with shape (height, width, 4)
        // First create a 1D array, then reshape it
        let arr = numpy::PyArray1::from_vec(py, pixels);
        let reshaped = arr.reshape([height, width, 4])?;
        Ok(reshaped.to_owned())
    }

    /// Read the rendered pixels to raw bytes.
    ///
    /// Returns:
    ///     Raw pixel data as bytes.
    fn read_to_bytes(&self) -> PyResult<Vec<u8>> {
        self.inner.read_to_cpu().into_py_result()
    }

    fn __repr__(&self) -> String {
        format!(
            "RenderTarget(width={}, height={}, has_depth={})",
            self.inner.width(),
            self.inner.height(),
            self.inner.has_depth()
        )
    }
}
