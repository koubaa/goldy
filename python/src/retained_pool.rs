//! Python wrapper for [`goldy::RetainedPool`] and record builders.

use crate::buffer::buffer_from_owned;
use crate::buffer::PyBuffer;
use crate::device::PyDevice;
use crate::error::IntoPyResult;
use crate::parcel::parcel_from_texture;
use crate::parcel::PyParcel;
use crate::types::{PyBufferKind, PyTextureFormat, PyTextureKind};
use goldy::{field, Init, RecordField};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::cell::RefCell;

struct RecordSpec {
    name: Option<String>,
    data: Option<Vec<u8>>,
    count: u64,
    stride: u32,
}

/// Builder for a retained partitioned buffer (one backing allocation, multiple units).
#[pyclass(name = "RecordBuilder", module = "goldy", unsendable)]
pub struct PyRecordBuilder {
    specs: RefCell<Vec<RecordSpec>>,
}

/// Deed-governed pool for retained GPU buffers and texture parcels.
#[pyclass(name = "RetainedPool", module = "goldy", unsendable)]
pub struct PyRetainedPool {
    pub(crate) inner: RefCell<goldy::RetainedPool>,
}

#[pymethods]
impl PyRetainedPool {
    #[new]
    fn new(device: &PyDevice) -> Self {
        Self {
            inner: RefCell::new(goldy::RetainedPool::new(device.inner.clone())),
        }
    }

    /// Acquire a retained buffer from numpy array or bytes.
    fn acquire_buffer(&self, data: &Bound<'_, PyAny>, access: PyBufferKind) -> PyResult<PyBuffer> {
        let (bytes, element_stride) = crate::bytes_util::extract_bytes_with_stride(data)?;
        let buffer = self
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
        Ok(buffer_from_owned(buffer))
    }

    /// Acquire a retained texture parcel.
    #[pyo3(signature = (width, height, format, kind, *, copy_src = true, copy_dst = false))]
    fn acquire_texture(
        &self,
        width: u32,
        height: u32,
        format: PyTextureFormat,
        kind: PyTextureKind,
        copy_src: bool,
        copy_dst: bool,
    ) -> PyResult<PyParcel> {
        let mut flags = goldy::TextureFlags::empty();
        if copy_src {
            flags |= goldy::TextureFlags::COPY_SRC;
        }
        if copy_dst {
            flags |= goldy::TextureFlags::COPY_DST;
        }
        let parcel = self
            .inner
            .borrow_mut()
            .acquire_texture(width, height, format.into(), kind.into(), flags, None)
            .into_py_result()?;
        Ok(parcel_from_texture(parcel))
    }

    /// Begin building a partitioned buffer (one backing allocation, multiple units).
    fn acquire_record(&self) -> PyRecordBuilder {
        PyRecordBuilder {
            specs: RefCell::new(Vec::new()),
        }
    }

    fn __repr__(&self) -> String {
        "RetainedPool()".to_string()
    }
}

#[pymethods]
impl PyRecordBuilder {
    /// Upload numpy array or bytes into the next ordinal field.
    fn emplace(&self, data: &Bound<'_, PyAny>) -> PyResult<u32> {
        let (bytes, element_stride) = crate::bytes_util::extract_bytes_with_stride(data)?;
        let count = bytes.len() as u64 / element_stride as u64;
        let slot = self.specs.borrow().len() as u32;
        self.specs.borrow_mut().push(RecordSpec {
            name: None,
            data: Some(bytes),
            count,
            stride: element_stride,
        });
        Ok(slot)
    }

    /// Reserve the next ordinal field without uploading data.
    fn reserve(&self, element_count: u64, element_stride: u32) -> PyResult<u32> {
        if element_stride == 0 {
            return Err(crate::error::GoldyError::new_err("element_stride must be non-zero"));
        }
        let slot = self.specs.borrow().len() as u32;
        self.specs.borrow_mut().push(RecordSpec {
            name: None,
            data: None,
            count: element_count,
            stride: element_stride,
        });
        Ok(slot)
    }

    /// Define a named field and upload numpy array or bytes.
    #[pyo3(signature = (name, data))]
    fn emplace_field(&self, name: String, data: &Bound<'_, PyAny>) -> PyResult<u32> {
        let (bytes, element_stride) = crate::bytes_util::extract_bytes_with_stride(data)?;
        let count = bytes.len() as u64 / element_stride as u64;
        let slot = self.specs.borrow().len() as u32;
        self.specs.borrow_mut().push(RecordSpec {
            name: Some(name),
            data: Some(bytes),
            count,
            stride: element_stride,
        });
        Ok(slot)
    }

    /// Allocate the backing buffer and return the partitioned [`PyBuffer`].
    fn build(&self, pool: &PyRetainedPool) -> PyResult<PyBuffer> {
        let specs = std::mem::take(&mut *self.specs.borrow_mut());
        if specs.is_empty() {
            return Err(crate::error::GoldyError::new_err(
                "RecordBuilder requires at least one field",
            ));
        }

        let fields: Vec<RecordField> = specs
            .into_iter()
            .map(|spec| {
                let init = match spec.data {
                    Some(bytes) => Init::Data {
                        bytes,
                        count: spec.count,
                        stride: spec.stride,
                    },
                    None => Init::Reserve {
                        count: spec.count,
                        stride: spec.stride,
                    },
                };
                match spec.name {
                    Some(name) => field(name, init),
                    None => goldy::ordinal(init),
                }
            })
            .collect();

        let buffer = pool.inner.borrow_mut().acquire_record(fields).into_py_result()?;
        Ok(buffer_from_owned(buffer))
    }

    fn __repr__(&self) -> String {
        "RecordBuilder()".to_string()
    }
}
