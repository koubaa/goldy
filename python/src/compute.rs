//! Python wrappers for Compute pipeline and encoder.

use crate::buffer::PyBuffer;
use crate::device::PyDevice;
use crate::error::IntoPyResult;
use crate::shader::PyShaderModule;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

// =============================================================================
// ComputePipeline
// =============================================================================

/// A compute pipeline.
///
/// Compute pipelines run compute shaders on the GPU, enabling general-purpose
/// GPU computing (GPGPU). They process data in parallel across many threads.
#[pyclass(name = "ComputePipeline", module = "goldy")]
pub struct PyComputePipeline {
    pub(crate) inner: Arc<goldy::ComputePipeline>,
}

#[pymethods]
impl PyComputePipeline {
    /// Create a new compute pipeline.
    ///
    /// Args:
    ///     device: The GPU device.
    ///     compute_shader: The compute shader module.
    ///
    /// Returns:
    ///     A new ComputePipeline instance.
    #[new]
    fn new(device: &PyDevice, compute_shader: &PyShaderModule) -> PyResult<Self> {
        let pipeline =
            goldy::ComputePipeline::new(&device.inner, &compute_shader.inner).into_py_result()?;

        Ok(PyComputePipeline {
            inner: Arc::new(pipeline),
        })
    }

    fn __repr__(&self) -> String {
        "ComputePipeline()".to_string()
    }
}

// =============================================================================
// ComputeEncoder
// =============================================================================

/// Command encoder for compute operations.
///
/// Similar to CommandEncoder for graphics, but for compute workloads.
/// Commands are recorded and then dispatched.
///
/// Example:
///     >>> encoder = goldy.ComputeEncoder()
///     >>> with encoder.begin_compute_pass() as cp:
///     ...     cp.set_pipeline(pipeline)
///     ...     cp.set_push_constants([buffer])
///     ...     cp.dispatch(16, 16, 1)
///     >>> encoder.dispatch(device)
#[pyclass(name = "ComputeEncoder", module = "goldy")]
pub struct PyComputeEncoder {
    inner: Mutex<goldy::ComputeEncoder>,
}

#[pymethods]
impl PyComputeEncoder {
    /// Create a new compute encoder.
    #[new]
    fn new() -> Self {
        PyComputeEncoder {
            inner: Mutex::new(goldy::ComputeEncoder::new()),
        }
    }

    /// Begin a compute pass.
    ///
    /// Returns a ComputePass that can be used as a context manager.
    fn begin_compute_pass(slf: Py<Self>) -> PyComputePass {
        PyComputePass { encoder: slf }
    }

    /// Execute the recorded compute commands on the device.
    ///
    /// This submits the compute work to the GPU and waits for completion.
    fn dispatch(&self, device: &PyDevice) -> PyResult<()> {
        let encoder = self.inner.lock().unwrap();
        encoder.dispatch(&device.inner).into_py_result()
    }

    fn __repr__(&self) -> String {
        "ComputeEncoder()".to_string()
    }
}

impl PyComputeEncoder {
    fn with_encoder<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut goldy::ComputeEncoder) -> R,
    {
        let mut encoder = self.inner.lock().unwrap();
        f(&mut encoder)
    }
}

// =============================================================================
// ComputePass
// =============================================================================

/// A compute pass for recording compute operations.
///
/// Use as a context manager:
///
///     >>> with encoder.begin_compute_pass() as cp:
///     ...     cp.set_pipeline(pipeline)
///     ...     cp.dispatch(16, 16, 1)
#[pyclass(name = "ComputePass", module = "goldy")]
pub struct PyComputePass {
    encoder: Py<PyComputeEncoder>,
}

#[pymethods]
impl PyComputePass {
    /// Set the active compute pipeline.
    fn set_pipeline(&self, py: Python<'_>, pipeline: &PyComputePipeline) {
        self.encoder.borrow(py).with_encoder(|enc| {
            let mut pass = enc.begin_compute_pass();
            pass.set_pipeline(&pipeline.inner);
        });
    }

    /// Set push constants for compute resource binding.
    ///
    /// Pass buffer indices to shaders. The buffers' descriptor indices are pushed
    /// directly to the GPU.
    ///
    /// Args:
    ///     buffers: List of buffers to pass to the shader via push constants.
    ///
    /// Example:
    ///     >>> cp.set_push_constants([buffer_a, buffer_b])
    ///     # In shader: g_StorageBuffers[getBufferIndex(0)] and [getBufferIndex(1)]
    fn set_push_constants(&self, py: Python<'_>, buffers: Vec<PyRef<'_, PyBuffer>>) {
        self.encoder.borrow(py).with_encoder(|enc| {
            // Collect buffer references - deref the Arc to get &Buffer
            let buffer_refs: Vec<&goldy::Buffer> =
                buffers.iter().map(|b| b.inner.as_ref()).collect();

            let mut pass = enc.begin_compute_pass();
            pass.set_push_constants(&buffer_refs);
        });
    }

    /// Dispatch compute workgroups.
    ///
    /// Args:
    ///     workgroups_x: Number of workgroups in X dimension.
    ///     workgroups_y: Number of workgroups in Y dimension.
    ///     workgroups_z: Number of workgroups in Z dimension.
    ///
    /// The actual number of threads is workgroups * numthreads (from shader).
    /// For a shader with [numthreads(8, 8, 1)]:
    ///     dispatch(16, 16, 1) runs 16*8 x 16*8 = 128 x 128 = 16384 threads
    #[pyo3(signature = (workgroups_x, workgroups_y=1, workgroups_z=1))]
    fn dispatch(&self, py: Python<'_>, workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) {
        self.encoder.borrow(py).with_encoder(|enc| {
            let mut pass = enc.begin_compute_pass();
            pass.dispatch(workgroups_x, workgroups_y, workgroups_z);
        });
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
        false
    }

    fn __repr__(&self) -> String {
        "ComputePass()".to_string()
    }
}
