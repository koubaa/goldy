//! Python wrappers for [`goldy::Scheme`] and submission context.

use crate::buffer::PyBuffer;
use crate::compute::PyComputePipeline;
use crate::error::{GoldyError, IntoPyResult};
use crate::parcel::PyParcel;
use crate::texture::PyTexture;
use crate::pipeline::PyRenderPipeline;
use crate::pyutil::parse_index_range;
use crate::types::{PyColor, PyDepthFormat, PyNodeAccess, PyTextureFormat};
use goldy::scheme::{Lease, LeaseRenderTarget, PresentGrant, ReadGrant};
use goldy::swapchain_pool::PresentLease;
use goldy::task_graph::{ComputeNodeRecord, RenderPassRecord};
use goldy::{Grant, GrantBuffer, GrantTexture, Scheme, Submission};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
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
#[pyclass(name = "SchemeSubmission", module = "goldy", unsendable)]
pub struct PySchemeSubmission {
    pub(crate) inner: Submission,
}

#[pymethods]
impl PySchemeSubmission {
    fn timeline_value(&self) -> u64 {
        self.inner.timeline_value()
    }

    fn wait(&self, ctx: &PyContext) -> PyResult<()> {
        self.inner.wait(&ctx.inner).into_py_result()
    }

    fn __repr__(&self) -> String {
        format!("SchemeSubmission(timeline_value={})", self.inner.timeline_value())
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

    fn consume<'py>(&self, py: Python<'py>, submission: &PySchemeSubmission) -> PyResult<Bound<'py, PyBytes>> {
        match &self.inner {
            PyReadGrantInner::Buffer(grant) => {
                let loan = grant.consume(&submission.inner).into_py_result()?;
                Ok(PyBytes::new(py, &loan))
            }
            PyReadGrantInner::Texture(grant) => {
                let loan = grant.consume(&submission.inner).into_py_result()?;
                Ok(PyBytes::new(py, &loan))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("ReadGrant(byte_size={})", self.inner.byte_size())
    }
}

/// Stable present lease from a [`crate::swapchain_pool::PySwapchainPool`].
#[pyclass(name = "PresentLease", module = "goldy", unsendable)]
pub struct PyPresentLease {
    pub(crate) inner: PresentLease,
}

#[pymethods]
impl PyPresentLease {
    fn __repr__(&self) -> String {
        "PresentLease()".to_string()
    }
}

/// Present easement grant recorded once via [`PyScheme::grant_present`].
#[pyclass(name = "PresentGrant", module = "goldy", unsendable)]
pub struct PyPresentGrant {
    pub(crate) inner: PresentGrant,
}

#[pymethods]
impl PyPresentGrant {
    fn consume(&self, submission: &PySchemeSubmission) -> PyResult<()> {
        self.inner.consume(&submission.inner).into_py_result()
    }

    fn __repr__(&self) -> String {
        "PresentGrant()".to_string()
    }
}

/// Stable render-target lease declared on a [`PyScheme`].
#[pyclass(name = "SchemeRenderTargetLease", module = "goldy", unsendable)]
pub struct PySchemeRenderTargetLease {
    pub(crate) inner: Lease<LeaseRenderTarget>,
}

#[pymethods]
impl PySchemeRenderTargetLease {
    fn __repr__(&self) -> String {
        "SchemeRenderTargetLease()".to_string()
    }
}

/// Retained scheme bound to one [`PyContext`].
#[pyclass(name = "Scheme", module = "goldy", unsendable)]
pub struct PyScheme {
    inner: RefCell<Scheme>,
    active_compute: RefCell<Option<ComputeNodeRecord>>,
    active_render_pass: RefCell<Option<RenderPassRecord>>,
    labels: RefCell<Vec<String>>,
}

#[pymethods]
impl PyScheme {
    #[new]
    fn new(ctx: &PyContext) -> Self {
        PyScheme {
            inner: RefCell::new(Scheme::new(&ctx.inner)),
            active_compute: RefCell::new(None),
            active_render_pass: RefCell::new(None),
            labels: RefCell::new(Vec::new()),
        }
    }

    /// Begin recording a compute dispatch node.
    #[pyo3(signature = (label, pipeline))]
    fn node(slf: PyRefMut<'_, Self>, label: String, pipeline: &PyComputePipeline) -> PyResult<PySchemeComputeNode> {
        if slf.active_compute.borrow().is_some() || slf.active_render_pass.borrow().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Only one recorder may be open per scheme",
            ));
        }
        let static_label = slf.intern_label(&label)?;
        *slf.active_compute.borrow_mut() = Some(ComputeNodeRecord::new(static_label, &pipeline.inner));
        Ok(PySchemeComputeNode {
            scheme: slf.into(),
            committed: RefCell::new(false),
        })
    }

    #[pyo3(signature = (width, height, format, depth_format=None))]
    fn lease_render_target(
        &self,
        width: u32,
        height: u32,
        format: PyTextureFormat,
        depth_format: Option<PyDepthFormat>,
    ) -> PyResult<PySchemeRenderTargetLease> {
        self.ensure_no_active_recorder()?;
        let lease = self
            .inner
            .borrow_mut()
            .lease_render_target(width, height, format.into(), depth_format.map(Into::into))
            .into_py_result()?;
        Ok(PySchemeRenderTargetLease { inner: lease })
    }

    fn render_pass(
        slf: Py<Self>,
        py: Python<'_>,
        label: String,
        lease: &PySchemeRenderTargetLease,
    ) -> PyResult<PySchemeRenderPass> {
        {
            let scheme = slf.borrow_mut(py);
            if scheme.active_compute.borrow().is_some() || scheme.active_render_pass.borrow().is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Only one recorder may be open per scheme",
                ));
            }
            let static_label = scheme.intern_label(&label)?;
            let pass = RenderPassRecord::new_for_scheme_lease(static_label, &scheme.inner.borrow(), &lease.inner);
            *scheme.active_render_pass.borrow_mut() = Some(pass);
        }
        Ok(PySchemeRenderPass { scheme: slf })
    }

    fn copy_to_texture(&self, src: &PySchemeRenderTargetLease, dst: &PyTexture) -> PyResult<()> {
        self.ensure_no_active_recorder()?;
        self.inner
            .borrow_mut()
            .copy_to_texture(&src.inner, &*dst.inner)
            .into_py_result()
    }

    fn copy_to_present(&self, src: &PySchemeRenderTargetLease, dst: &PyPresentLease) -> PyResult<()> {
        self.ensure_no_active_recorder()?;
        self.inner.borrow_mut().copy_to_present(&src.inner, &dst.inner);
        Ok(())
    }

    fn grant_present(&self, lease: &PyPresentLease) -> PyResult<PyPresentGrant> {
        self.ensure_no_active_recorder()?;
        Ok(PyPresentGrant {
            inner: self.inner.borrow_mut().grant_present(&lease.inner),
        })
    }

    fn grant_read(&self, parcel: &PyParcel) -> PyResult<PyReadGrant> {
        self.ensure_no_active_recorder()?;
        let grant = self
            .inner
            .borrow_mut()
            .grant_read(parcel.inner.as_parcel())
            .into_py_result()?;
        Ok(PyReadGrant {
            inner: PyReadGrantInner::Buffer(grant),
        })
    }

    fn grant_read_texture(&self, texture: &PyTexture) -> PyResult<PyReadGrant> {
        self.ensure_no_active_recorder()?;
        let grant = self
            .inner
            .borrow_mut()
            .grant_read_texture(&*texture.inner)
            .into_py_result()?;
        Ok(PyReadGrant {
            inner: PyReadGrantInner::Texture(grant),
        })
    }

    /// Append a CPU→GPU write node for a retained buffer parcel.
    ///
    /// Marks the scheme dirty. Use an ephemeral upload scheme for per-frame
    /// uniform / vertex uploads and call `submit()` to dispatch.
    #[pyo3(signature = (parcel, data, offset=0))]
    fn commit_write_parcel(&self, parcel: &PyParcel, data: &[u8], offset: u64) -> PyResult<()> {
        self.ensure_no_active_recorder()?;
        self.inner
            .borrow_mut()
            .commit_write_parcel(parcel.inner.as_parcel(), offset, data.to_vec())
            .into_py_result()
    }

    fn submit(&self) -> PyResult<PySchemeSubmission> {
        self.ensure_no_active_recorder()?;
        let submission = self.inner.borrow_mut().submit().into_py_result()?;
        Ok(PySchemeSubmission { inner: submission })
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
        Ok(unsafe { std::mem::transmute::<&str, &'static str>(s.as_str()) })
    }

    fn ensure_no_active_recorder(&self) -> PyResult<()> {
        if self.active_compute.borrow().is_some() || self.active_render_pass.borrow().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot mutate scheme while recording a node or render pass",
            ));
        }
        Ok(())
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

    fn with_active_render_pass<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut RenderPassRecord) -> R,
    {
        let mut pass = self.active_render_pass.borrow_mut();
        let pass = pass.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "No render pass is open; use `with scheme.render_pass(...) as rp:`",
            )
        })?;
        Ok(f(pass))
    }

    fn finish_render_pass(&self) -> PyResult<()> {
        let pass = self
            .active_render_pass
            .borrow_mut()
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No render pass to finish"))?;
        pass.commit_scheme(&mut self.inner.borrow_mut());
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
    fn with_parcel<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        parcel: &PyParcel,
        node_access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        {
            let scheme = slf.scheme.borrow(py);
            let mut active = scheme.active_compute.borrow_mut();
            let node = active
                .as_mut()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No compute node is being recorded"))?;
            node.with_parcel(parcel.inner.as_parcel(), node_access.into())
                .ok_or_else(|| GoldyError::new_err("Parcel has no bindless slot for the shader binding"))?;
        }
        Ok(slf)
    }

    fn with_texture<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        texture: &PyTexture,
        node_access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        {
            let scheme = slf.scheme.borrow(py);
            let mut active = scheme.active_compute.borrow_mut();
            let node = active
                .as_mut()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No compute node is being recorded"))?;
            node.with_parcel(&*texture.inner, node_access.into())
                .ok_or_else(|| GoldyError::new_err("Texture has no bindless slot for the shader binding"))?;
        }
        Ok(slf)
    }

    #[pyo3(signature = (buffer, unit, node_access))]
    fn with_buffer_unit<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        buffer: &PyBuffer,
        unit: u32,
        node_access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        {
            let scheme = slf.scheme.borrow(py);
            let mut active = scheme.active_compute.borrow_mut();
            let node = active
                .as_mut()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No compute node is being recorded"))?;
            let idx = unit as usize;
            if idx >= buffer.inner.unit_count() {
                return Err(GoldyError::new_err(format!(
                    "buffer unit index {unit} out of range (unit_count={})",
                    buffer.inner.unit_count()
                )));
            }
            node.with_parcel(buffer.inner.unit(idx), node_access.into())
                .ok_or_else(|| GoldyError::new_err("Buffer unit has no bindless slot for the shader binding"))?;
        }
        Ok(slf)
    }

    #[pyo3(signature = (buffer, name, node_access))]
    fn with_field<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        buffer: &PyBuffer,
        name: &str,
        node_access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        {
            let scheme = slf.scheme.borrow(py);
            let mut active = scheme.active_compute.borrow_mut();
            let node = active
                .as_mut()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No compute node is being recorded"))?;
            let parcel_ptr = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                buffer.inner.field(name) as *const goldy::Parcel
            })) {
                Ok(p) => p,
                Err(_) => return Err(GoldyError::new_err(format!("unknown buffer field {name:?}"))),
            };
            node.with_parcel(unsafe { &*parcel_ptr }, node_access.into())
                .ok_or_else(|| GoldyError::new_err("Buffer field has no bindless slot for the shader binding"))?;
        }
        Ok(slf)
    }

    fn with_param<'py>(slf: PyRef<'py, Self>, py: Python<'py>, value: u32) -> PyResult<PyRef<'py, Self>> {
        {
            let scheme = slf.scheme.borrow(py);
            let mut active = scheme.active_compute.borrow_mut();
            let node = active
                .as_mut()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No compute node is being recorded"))?;
            node.with_param(value);
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

/// Records one render pass on a retained scheme.
#[pyclass(name = "SchemeRenderPass", module = "goldy", unsendable)]
pub struct PySchemeRenderPass {
    scheme: Py<PyScheme>,
}

#[pymethods]
impl PySchemeRenderPass {
    fn with_parcel<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        parcel: &PyParcel,
        access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.scheme.borrow(py).with_active_render_pass(|pass| {
            pass.with_parcel(parcel.inner.as_parcel(), access.into());
        })?;
        Ok(slf)
    }

    fn with_buffer_unit<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        buffer: &PyBuffer,
        unit: u32,
        access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        let idx = unit as usize;
        if idx >= buffer.inner.unit_count() {
            return Err(GoldyError::new_err(format!(
                "buffer unit index {unit} out of range (unit_count={})",
                buffer.inner.unit_count()
            )));
        }
        slf.scheme.borrow(py).with_active_render_pass(|pass| {
            pass.with_parcel(buffer.inner.unit(idx), access.into());
        })?;
        Ok(slf)
    }

    fn with_field<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        buffer: &PyBuffer,
        name: &str,
        access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        let parcel_ptr = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            buffer.inner.field(name) as *const goldy::Parcel
        })) {
            Ok(p) => p,
            Err(_) => return Err(GoldyError::new_err(format!("unknown buffer field {name:?}"))),
        };
        slf.scheme.borrow(py).with_active_render_pass(|pass| {
            pass.with_parcel(unsafe { &*parcel_ptr }, access.into());
        })?;
        Ok(slf)
    }

    fn clear<'py>(slf: PyRef<'py, Self>, py: Python<'py>, color: &PyColor) -> PyResult<PyRef<'py, Self>> {
        slf.scheme.borrow(py).with_active_render_pass(|pass| {
            pass.clear(color.inner);
        })?;
        Ok(slf)
    }

    fn set_pipeline<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        pipeline: &PyRenderPipeline,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.scheme.borrow(py).with_active_render_pass(|pass| {
            pass.set_pipeline(&pipeline.inner);
        })?;
        Ok(slf)
    }

    fn set_vertex_buffer_parcel<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        slot: u32,
        parcel: &PyParcel,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.scheme.borrow(py).with_active_render_pass(|pass| {
            pass.set_vertex_buffer(slot, parcel.inner.as_parcel());
        })?;
        Ok(slf)
    }

    #[pyo3(signature = (vertices=None, *, first_vertex=0, vertex_count=None, instances=None, first_instance=0, instance_count=None))]
    fn draw<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        vertices: Option<Bound<'py, PyAny>>,
        first_vertex: u32,
        vertex_count: Option<u32>,
        instances: Option<Bound<'py, PyAny>>,
        first_instance: u32,
        instance_count: Option<u32>,
    ) -> PyResult<PyRef<'py, Self>> {
        let (fv, vc) = if let Some(v) = vertices {
            parse_index_range(&v, "vertices")?
        } else {
            (first_vertex, vertex_count.unwrap_or(3))
        };
        let (fi, ic) = if let Some(i) = instances {
            parse_index_range(&i, "instances")?
        } else {
            (first_instance, instance_count.unwrap_or(1))
        };
        slf.scheme.borrow(py).with_active_render_pass(|pass| {
            pass.draw(fv, vc, fi, ic);
        })?;
        Ok(slf)
    }

    fn draw_fullscreen<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyRef<'py, Self>> {
        slf.scheme.borrow(py).with_active_render_pass(|pass| {
            pass.draw_fullscreen();
        })?;
        Ok(slf)
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.scheme.borrow(py).finish_render_pass()?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        "SchemeRenderPass(recording)".to_string()
    }
}
