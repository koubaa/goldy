//! Python wrapper for Runtime.

use crate::buffer::buffer_from_owned;
use crate::buffer::PyBuffer;
use crate::error::IntoPyResult;
use crate::retained_pool::PyRecordBuilder;
use crate::scheme::PyContext;
use crate::texture::texture_from_owned;
use crate::texture::PyTexture;
use crate::types::{PyBufferKind, PyTextureFormat, PyTextureKind};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::cell::RefCell;
use std::sync::Arc;

/// A GPU device - used to create resources and render.
///
/// The Runtime is the primary interface for GPU operations.
#[pyclass(name = "Runtime", module = "goldy")]
#[derive(Clone)]
pub struct PyRuntime {
    pub inner: Arc<goldy::Runtime>,
}

#[pymethods]
impl PyRuntime {
    /// Get the adapter ID this device was created on.
    #[getter]
    fn adapter_id(&self) -> u32 {
        self.inner.adapter_id()
    }

    /// Check if the device is still valid.
    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    /// Check if a shader library is registered.
    fn has_library(&self, name: &str) -> bool {
        self.inner.has_library(name)
    }

    /// List all registered shader libraries.
    fn list_libraries(&self) -> Vec<String> {
        self.inner.list_libraries()
    }

    /// Register a shader library from source.
    ///
    /// Args:
    ///     name: The library name (used in `import` statements).
    ///     source: The Slang source code for the library.
    ///
    /// Raises:
    ///     GoldyError: If a library with the same name is already registered.
    fn register_library(&self, name: &str, source: &str) -> PyResult<()> {
        let library = goldy::ShaderLibrary::from_source(name, source);
        self.inner.register_library(library).into_py_result()
    }

    /// Unregister a shader library.
    ///
    /// Returns:
    ///     True if the library was found and removed, False otherwise.
    fn unregister_library(&self, name: &str) -> bool {
        self.inner.unregister_library(name)
    }

    /// Create a GPU submission context for retained schemes.
    fn create_context(&self) -> PyResult<PyContext> {
        let inner = self.inner.create_context().into_py_result()?;
        Ok(PyContext { inner })
    }

    /// Acquire a retained buffer from numpy array or bytes.
    fn acquire_buffer(&self, data: &Bound<'_, PyAny>, access: PyBufferKind) -> PyResult<PyBuffer> {
        let (bytes, element_stride) = crate::bytes_util::extract_bytes_with_stride(data)?;
        let buffer = self
            .inner
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

    /// Acquire a packed tensor from a 1-D NumPy array or bytes.
    ///
    /// Host conversion copies into GPU storage. Read results back with
    /// `SchemeSubmission.take` on `tensor.parcel()`.
    #[cfg(feature = "tensor")]
    #[pyo3(signature = (data, shape, dtype=None))]
    fn acquire_tensor(
        &self,
        data: &Bound<'_, PyAny>,
        shape: Vec<u32>,
        dtype: Option<crate::tensor::PyTensorDType>,
    ) -> PyResult<crate::tensor::PyTensor> {
        crate::tensor::acquire_tensor(
            &self.inner,
            data,
            shape,
            dtype.unwrap_or(crate::tensor::PyTensorDType::F32),
        )
    }

    /// Allocate a packed zero tensor.
    #[cfg(feature = "tensor")]
    #[pyo3(signature = (shape, dtype=None))]
    fn zeros_tensor(
        &self,
        shape: Vec<u32>,
        dtype: Option<crate::tensor::PyTensorDType>,
    ) -> PyResult<crate::tensor::PyTensor> {
        crate::tensor::zeros_tensor(&self.inner, shape, dtype.unwrap_or(crate::tensor::PyTensorDType::F32))
    }

    /// Acquire a retained texture.
    #[pyo3(signature = (width, height, format, kind, *, copy_src = true, copy_dst = false))]
    fn acquire_texture(
        &self,
        width: u32,
        height: u32,
        format: PyTextureFormat,
        kind: PyTextureKind,
        copy_src: bool,
        copy_dst: bool,
    ) -> PyResult<PyTexture> {
        let mut flags = goldy::TextureFlags::empty();
        if copy_src {
            flags |= goldy::TextureFlags::COPY_SRC;
        }
        if copy_dst {
            flags |= goldy::TextureFlags::COPY_DST;
        }
        let texture = self
            .inner
            .acquire_texture(width, height, format.into(), kind.into(), flags, None)
            .into_py_result()?;
        Ok(texture_from_owned(texture))
    }

    /// Begin building a partitioned buffer (one backing allocation, multiple units).
    fn acquire_record(&self) -> PyRecordBuilder {
        PyRecordBuilder::empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "Runtime(adapter_id={}, valid={})",
            self.inner.adapter_id(),
            self.inner.is_valid()
        )
    }
}
