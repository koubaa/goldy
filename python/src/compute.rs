//! Python wrappers for ComputePipeline.

use crate::device::PyDevice;
use crate::error::IntoPyResult;
use crate::shader::PyShaderModule;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass(name = "ComputePipeline", module = "goldy")]
pub struct PyComputePipeline {
    pub(crate) inner: Arc<goldy::ComputePipeline>,
}

#[pymethods]
impl PyComputePipeline {
    #[new]
    fn new(device: &PyDevice, compute_shader: &PyShaderModule) -> PyResult<Self> {
        let pipeline = goldy::ComputePipeline::new(&device.inner, &compute_shader.inner).into_py_result()?;

        Ok(PyComputePipeline {
            inner: Arc::new(pipeline),
        })
    }

    fn __repr__(&self) -> String {
        "ComputePipeline()".to_string()
    }
}
