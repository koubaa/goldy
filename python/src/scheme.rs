//! Python wrappers for [`goldy::Scheme`] and submission context.

use crate::compute::PyComputePipeline;
use crate::error::{GoldyError, IntoPyResult};
use crate::parcel::PyParcel;
use crate::types::{PyNodeAccess, PyResourceAccess};
use goldy::task_graph::ComputeNodeRecord;
use goldy::{Grant, GrantBuffer, GrantTexture, ReadGrant, Scheme, SchemeFrame};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::cell::RefCell;

/// GPU submission context — one per scheme.
#[pyclass(name = "Context", module = "goldy", unsendable)]
pub struct PyContext {
    pub(crate) inner: goldy::Context,
}

#[pymethods]
impl PyContext {
    fn __repr__(&self) -> String {
        "Context()".to_string()
    }
}

/// Per-submission identity returned by [`PyScheme::submit`].
#[pyclass(name = "SchemeFrame", module = "goldy", unsendable)]
pub struct PySchemeFrame {
    pub(crate) inner: SchemeFrame,
}

#[pymethods]
impl PySchemeFrame {
    fn timeline_value(&self) -> u64 {
        self.inner.timeline_value()
    }

    fn wait(&self, ctx: &PyContext) -> PyResult<()> {
        self.inner.wait(&ctx.inner).into_py_result()
    }

    fn __repr__(&self) -> String {
        format!("SchemeFrame(timeline_value={})", self.inner.timeline_value())
    }
}

pub(crate) enum PyReadGrantInner {
    Buffer(ReadGrant<GrantBuffer>),
    Texture(ReadGrant<GrantTexture>),
}

impl PyReadGrantInner {
    fn byte_size(&self) -> u64 {
        match self {
            Self::Buffer(grant) => grant.byte_size(),
            Self::Texture(grant) => grant.byte_size(),
        }
    }
}

/// Read easement grant recorded once via [`PyScheme::grant_read`] or [`PyScheme::grant_read_texture`].
#[pyclass(name = "ReadGrant", module = "goldy", unsendable)]
pub struct PyReadGrant {
    pub(crate) inner: PyReadGrantInner,
}

#[pymethods]
impl PyReadGrant {
    fn byte_size(&self) -> u64 {
        self.inner.byte_size()
    }

    /// Consumable bytes for `frame`'s submission.
    fn consume<'py>(&self, py: Python<'py>, frame: &PySchemeFrame) -> PyResult<Bound<'py, PyBytes>> {
        match &self.inner {
            PyReadGrantInner::Buffer(grant) => {
                let loan = grant.consume(&frame.inner).into_py_result()?;
                Ok(PyBytes::new(py, &loan))
            }
            PyReadGrantInner::Texture(grant) => {
                let loan = grant.consume(&frame.inner).into_py_result()?;
                Ok(PyBytes::new(py, &loan))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("ReadGrant(byte_size={})", self.inner.byte_size())
    }
}

/// Retained compute scheme bound to one [`PyContext`].
#[pyclass(name = "Scheme", module = "goldy", unsendable)]
pub struct PyScheme {
    inner: RefCell<Scheme>,
    active_compute: RefCell<Option<ComputeNodeRecord>>,
    labels: RefCell<Vec<String>>,
}

#[pymethods]
impl PyScheme {
    #[new]
    fn new(ctx: &PyContext) -> Self {
        PyScheme {
            inner: RefCell::new(Scheme::new(&ctx.inner)),
            active_compute: RefCell::new(None),
            labels: RefCell::new(Vec::new()),
        }
    }

    /// Begin recording a compute dispatch node.
    #[pyo3(signature = (label, pipeline))]
    fn node(slf: PyRefMut<'_, Self>, label: String, pipeline: &PyComputePipeline) -> PyResult<PySchemeComputeNode> {
        if slf.active_compute.borrow().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Only one compute node may be open per scheme",
            ));
        }
        let static_label = slf.intern_label(&label)?;
        *slf.active_compute.borrow_mut() = Some(ComputeNodeRecord::new(static_label, &pipeline.inner));
        Ok(PySchemeComputeNode {
            scheme: slf.into(),
            committed: RefCell::new(false),
        })
    }

    /// Record a read easement over a buffer parcel (once per scheme).
    fn grant_read(&self, parcel: &PyParcel) -> PyResult<PyReadGrant> {
        if self.active_compute.borrow().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot grant_read while recording a compute node",
            ));
        }
        let grant = self
            .inner
            .borrow_mut()
            .grant_read(parcel.inner.as_ref())
            .into_py_result()?;
        Ok(PyReadGrant {
            inner: PyReadGrantInner::Buffer(grant),
        })
    }

    /// Record a read easement over a texture parcel (once per scheme).
    #[pyo3(signature = (parcel))]
    fn grant_read_texture(&self, parcel: &PyParcel) -> PyResult<PyReadGrant> {
        if self.active_compute.borrow().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot grant_read_texture while recording a compute node",
            ));
        }
        let grant = self
            .inner
            .borrow_mut()
            .grant_read_texture(parcel.inner.as_ref())
            .into_py_result()?;
        Ok(PyReadGrant {
            inner: PyReadGrantInner::Texture(grant),
        })
    }

    /// Submit the scheme and return a per-submission frame token.
    fn submit(&self) -> PyResult<PySchemeFrame> {
        if self.active_compute.borrow().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot submit while recording a compute node",
            ));
        }
        let frame = self.inner.borrow_mut().submit().into_py_result()?;
        Ok(PySchemeFrame { inner: frame })
    }

    fn __repr__(&self) -> String {
        format!("Scheme(nodes={})", self.inner.borrow().ir_node_count())
    }
}

impl PyScheme {
    fn intern_label(&self, label: &str) -> PyResult<&'static str> {
        let mut labels = self.labels.borrow_mut();
        labels.push(label.to_string());
        let s = labels.last().unwrap();
        // SAFETY: `labels` is cleared when the scheme is dropped alongside IR node labels.
        Ok(unsafe { std::mem::transmute::<&str, &'static str>(s.as_str()) })
    }

    fn finish_compute_node(&self, workgroups: (u32, u32, u32)) -> PyResult<()> {
        let node = self
            .active_compute
            .borrow_mut()
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No compute node is being recorded"))?;
        node.commit_dispatch_scheme(&mut self.inner.borrow_mut(), workgroups.0, workgroups.1, workgroups.2);
        Ok(())
    }
}

/// Builder for one compute dispatch node on a [`PyScheme`].
#[pyclass(name = "SchemeComputeNode", module = "goldy", unsendable)]
pub struct PySchemeComputeNode {
    scheme: Py<PyScheme>,
    committed: RefCell<bool>,
}

#[pymethods]
impl PySchemeComputeNode {
    fn declare_parcel<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        parcel: &PyParcel,
        node_access: PyNodeAccess,
        resource_access: PyResourceAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        {
            let scheme = slf.scheme.borrow(py);
            let mut active = scheme.active_compute.borrow_mut();
            let node = active
                .as_mut()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No compute node is being recorded"))?;
            node.declare_parcel(parcel.inner.as_ref(), node_access.into(), resource_access.into())
                .ok_or_else(|| GoldyError::new_err("Parcel has no resource index for the requested access"))?;
        }
        Ok(slf)
    }

    #[pyo3(signature = (workgroups_x=1, workgroups_y=1, workgroups_z=1))]
    fn dispatch<'py>(
        slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        workgroups_x: u32,
        workgroups_y: u32,
        workgroups_z: u32,
    ) {
        if !*slf.committed.borrow() {
            let _ = slf
                .scheme
                .borrow(py)
                .finish_compute_node((workgroups_x, workgroups_y, workgroups_z));
            *slf.committed.borrow_mut() = true;
        }
    }

    fn __repr__(&self) -> String {
        "SchemeComputeNode(recording)".to_string()
    }
}

impl Drop for PySchemeComputeNode {
    fn drop(&mut self) {
        debug_assert!(*self.committed.borrow(), "SchemeComputeNode dropped without dispatch()");
    }
}

/// Upload CPU bytes into a retained buffer parcel via a property-only dispatch.
#[pyfunction]
#[pyo3(signature = (ctx, parcel, data))]
pub fn write_to_parcel(ctx: &PyContext, parcel: &PyParcel, data: &[u8]) -> PyResult<PySchemeFrame> {
    let mut upload = Scheme::new(&ctx.inner);
    upload
        .commit_write_parcel(parcel.inner.as_ref(), 0, data.to_vec())
        .into_py_result()?;
    let frame = upload.submit().into_py_result()?;
    Ok(PySchemeFrame { inner: frame })
}
