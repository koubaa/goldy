//! NumPy/bytes extraction for retained-pool uploads.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Extract bytes and element stride from a Python object (numpy array or bytes).
/// Returns `(bytes, element_stride)` where element stride is the size of each element in bytes.
pub(crate) fn extract_bytes_with_stride(data: &Bound<'_, PyAny>) -> PyResult<(Vec<u8>, u32)> {
    if let Ok(arr) = data.cast::<PyArray1<f32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 4));
    }

    if let Ok(arr) = data.cast::<PyArray1<f64>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 8));
    }

    if let Ok(arr) = data.cast::<PyArray1<i32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 4));
    }

    if let Ok(arr) = data.cast::<PyArray1<u32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 4));
    }

    if let Ok(arr) = data.cast::<PyArray1<i16>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 2));
    }

    if let Ok(arr) = data.cast::<PyArray1<u16>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 2));
    }

    if let Ok(arr) = data.cast::<PyArray1<i8>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 1));
    }

    if let Ok(arr) = data.cast::<PyArray1<u8>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((slice.to_vec(), 1));
    }

    if let Ok(bytes) = data.extract::<Vec<u8>>() {
        return Ok((bytes, 1));
    }

    Err(pyo3::exceptions::PyTypeError::new_err("Expected numpy array or bytes"))
}
