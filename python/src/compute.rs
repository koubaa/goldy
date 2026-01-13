//! Python wrappers for Compute pipeline and encoder.

use crate::bind_group::{PyBindGroup, PyBindGroupLayout};
use crate::device::PyDevice;
use crate::error::IntoPyResult;
use crate::shader::PyShaderModule;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

// =============================================================================
// ComputePipelineDesc
// =============================================================================

/// Description for creating a compute pipeline.
#[pyclass(name = "ComputePipelineDesc", module = "goldy")]
#[derive(Clone, Default)]
pub struct PyComputePipelineDesc {
    pub(crate) bind_group_layouts: Vec<Arc<goldy::BindGroupLayout>>,
}

#[pymethods]
impl PyComputePipelineDesc {
    /// Create a new compute pipeline description.
    #[new]
    #[pyo3(signature = (bind_group_layouts=None))]
    fn new(bind_group_layouts: Option<Vec<PyRef<PyBindGroupLayout>>>) -> Self {
        PyComputePipelineDesc {
            bind_group_layouts: bind_group_layouts
                .map(|layouts| layouts.iter().map(|l| Arc::clone(&l.inner)).collect())
                .unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ComputePipelineDesc(bind_group_layouts={})",
            self.bind_group_layouts.len()
        )
    }
}

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
    ///     desc: Pipeline description with bind group layouts.
    ///
    /// Returns:
    ///     A new ComputePipeline instance.
    #[new]
    fn new(
        device: &PyDevice,
        compute_shader: &PyShaderModule,
        desc: &PyComputePipelineDesc,
    ) -> PyResult<Self> {
        // Create temporary references to BindGroupLayout
        let layout_refs: Vec<&goldy::BindGroupLayout> =
            desc.bind_group_layouts.iter().map(|l| l.as_ref()).collect();

        let rust_desc = goldy::ComputePipelineDesc {
            bind_group_layouts: &layout_refs,
        };

        let pipeline =
            goldy::ComputePipeline::new(&device.inner, &compute_shader.inner, &rust_desc)
                .into_py_result()?;

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
///     ...     cp.set_bind_group(0, bind_group)
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

    /// Set a bind group for shader resources (storage buffers, etc.).
    ///
    /// Args:
    ///     index: The bind group set index (matches shader's [[vk::binding(N, index)]]).
    ///     bind_group: The bind group to use.
    fn set_bind_group(&self, py: Python<'_>, index: u32, bind_group: &PyBindGroup) {
        self.encoder.borrow(py).with_encoder(|enc| {
            let mut pass = enc.begin_compute_pass();
            pass.set_bind_group(index, &bind_group.inner);
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
