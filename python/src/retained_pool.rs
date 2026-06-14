//! Python wrapper for [`goldy::RetainedPool`] and mosaic builders.

use crate::device::PyDevice;
use crate::error::IntoPyResult;
use crate::parcel::PyParcel;
use crate::types::{PyBufferKind, PyTextureFormat, PyTextureKind};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::cell::RefCell;
use std::sync::Arc;

struct MosaicSpec {
    data: Vec<u8>,
    count: u64,
    stride: u32,
}

/// Deed-governed pool for retained GPU parcels.
#[pyclass(name = "RetainedPool", module = "goldy", unsendable)]
pub struct PyRetainedPool {
    pub(crate) inner: RefCell<goldy::RetainedPool>,
}

/// Builder for a retained mosaic parcel (one backing buffer, multiple sub-views).
#[pyclass(name = "MosaicBuilder", module = "goldy", unsendable)]
pub struct PyMosaicBuilder {
    specs: RefCell<Vec<MosaicSpec>>,
}

#[pymethods]
impl PyRetainedPool {
    #[new]
    fn new(device: &PyDevice) -> Self {
        Self {
            inner: RefCell::new(goldy::RetainedPool::new(device.inner.clone())),
        }
    }

    /// Acquire a retained buffer parcel from numpy array or bytes.
    fn acquire_buffer(&self, data: &Bound<'_, PyAny>, access: PyBufferKind) -> PyResult<PyParcel> {
        let (bytes, element_stride) = crate::bytes_util::extract_bytes_with_stride(data)?;
        let parcel = self
            .inner
            .borrow_mut()
            .acquire_buffer(
                bytes.len() as u64,
                access.into(),
                Some(element_stride),
                goldy::BufferFlags::empty(),
                Some(&bytes),
            )
            .into_py_result()?;
        Ok(PyParcel {
            inner: Arc::new(parcel),
        })
    }

    /// Acquire a retained texture parcel.
    #[pyo3(signature = (width, height, format, kind, *, copy_src = true))]
    fn acquire_texture(
        &self,
        width: u32,
        height: u32,
        format: PyTextureFormat,
        kind: PyTextureKind,
        copy_src: bool,
    ) -> PyResult<PyParcel> {
        let flags = if copy_src {
            goldy::TextureFlags::COPY_SRC
        } else {
            goldy::TextureFlags::empty()
        };
        let parcel = self
            .inner
            .borrow_mut()
            .acquire_texture(width, height, format.into(), kind.into(), flags, None)
            .into_py_result()?;
        Ok(PyParcel {
            inner: Arc::new(parcel),
        })
    }

    /// Begin building a retained mosaic parcel (one backing buffer, multiple sub-views).
    fn mosaic(&self) -> PyMosaicBuilder {
        PyMosaicBuilder {
            specs: RefCell::new(Vec::new()),
        }
    }

    fn __repr__(&self) -> String {
        "RetainedPool()".to_string()
    }
}

#[pymethods]
impl PyMosaicBuilder {
    /// Reserve a mosaic sub-view and upload numpy array or bytes.
    fn emplace(&self, data: &Bound<'_, PyAny>) -> PyResult<u32> {
        let (bytes, element_stride) = crate::bytes_util::extract_bytes_with_stride(data)?;
        let count = bytes.len() as u64 / element_stride as u64;
        let slot = self.specs.borrow().len() as u32;
        self.specs.borrow_mut().push(MosaicSpec {
            data: bytes,
            count,
            stride: element_stride,
        });
        Ok(slot)
    }

    /// Allocate the backing buffer and return the mosaic parcel.
    fn build(&self, pool: &PyRetainedPool) -> PyResult<PyParcel> {
        let specs = std::mem::take(&mut *self.specs.borrow_mut());
        let mut pool_ref = pool.inner.borrow_mut();
        let mut mosaic = pool_ref.mosaic();
        for spec in specs {
            mosaic.emplace_bytes(&spec.data, spec.count, spec.stride);
        }
        let parcel = mosaic.build().into_py_result()?;
        Ok(PyParcel {
            inner: Arc::new(parcel),
        })
    }

    fn __repr__(&self) -> String {
        "MosaicBuilder()".to_string()
    }
}
