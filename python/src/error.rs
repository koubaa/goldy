//! Error handling for Python bindings.
//!
//! Maps Rust errors to Python exceptions.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;

// Create a custom exception type for Goldy errors
pyo3::create_exception!(goldy, GoldyError, PyException, "Goldy GPU library error.");

/// Convert anyhow::Error to PyErr
pub fn to_py_err(err: anyhow::Error) -> PyErr {
    GoldyError::new_err(err.to_string())
}

/// Extension trait for converting Result<T, anyhow::Error> to PyResult<T>
pub trait IntoPyResult<T> {
    fn into_py_result(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for anyhow::Result<T> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(to_py_err)
    }
}

