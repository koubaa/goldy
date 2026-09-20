//! Python wrapper for [`goldy::Runtime`] and record builders.

use crate::buffer::buffer_from_owned;
use crate::buffer::PyBuffer;
use crate::error::IntoPyResult;
use crate::runtime::PyRuntime;
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
    pub(crate) specs: RefCell<Vec<RecordSpec>>,
}

impl PyRecordBuilder {
    pub(crate) fn empty() -> Self {
        Self {
            specs: RefCell::new(Vec::new()),
        }
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
    fn build(&self, runtime: &PyRuntime) -> PyResult<PyBuffer> {
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

        let buffer = runtime.inner.acquire_record(fields).into_py_result()?;
        Ok(buffer_from_owned(buffer))
    }

    fn __repr__(&self) -> String {
        "RecordBuilder()".to_string()
    }
}
