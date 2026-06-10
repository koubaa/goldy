//! Python wrappers for TaskGraph and render-pass recording.

use crate::buffer::PyBuffer;
use crate::buffer_pool::PyBufferView;
use crate::compute::PyComputePipeline;
use crate::device::PyDevice;
use crate::error::{GoldyError, IntoPyResult};
use crate::parcel::PyParcel;
use crate::pipeline::PyRenderPipeline;
use crate::render_target::PyRenderTarget;
use crate::types::{PyColor, PyIndexFormat, PyNodeAccess};
use goldy::task_graph::{ComputeNodeRecord, RenderPassRecord, SwapchainOutputHandle, TaskGraph};
use goldy::types::{ResourceAccess, ResourceCategory, ResourceHandle};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::cell::RefCell;

/// Opaque token from [`PyTaskGraph::declare_swapchain_output`].
#[pyclass(name = "SwapchainOutput", module = "goldy", unsendable)]
pub struct PySwapchainOutput;

/// GPU task graph: render passes, swapchain blit, and dispatch.
#[pyclass(name = "TaskGraph", module = "goldy", unsendable)]
pub struct PyTaskGraph {
    pub(crate) inner: RefCell<TaskGraph>,
    active_pass: RefCell<Option<RenderPassRecord>>,
    active_compute: RefCell<Option<ComputeNodeRecord>>,
    labels: RefCell<Vec<String>>,
    swapchain: RefCell<Option<SwapchainOutputHandle>>,
}

impl PyTaskGraph {
    pub(crate) fn ensure_no_active_recorder(&self) -> PyResult<()> {
        if self.active_pass.borrow().is_some() || self.active_compute.borrow().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "A render pass or compute node is still open; finish it before submitting the graph",
            ));
        }
        Ok(())
    }

    fn intern_label(&self, label: &str) -> PyResult<&'static str> {
        let mut labels = self.labels.borrow_mut();
        labels.push(label.to_string());
        let s = labels.last().unwrap();
        // SAFETY: `labels` is cleared in `clear()` and dropped with the graph.
        Ok(unsafe { std::mem::transmute::<&str, &'static str>(s.as_str()) })
    }

    fn with_active_pass<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut RenderPassRecord) -> R,
    {
        let mut pass = self.active_pass.borrow_mut();
        let pass = pass.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "No render pass is open; use `with graph.render_pass(...) as rp:`",
            )
        })?;
        Ok(f(pass))
    }

    fn finish_render_pass(&self) -> PyResult<()> {
        let pass = self
            .active_pass
            .borrow_mut()
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No render pass to finish"))?;
        pass.commit(&mut self.inner.borrow_mut());
        Ok(())
    }

    fn finish_compute_node(&self, workgroups: (u32, u32, u32)) -> PyResult<()> {
        let node = self
            .active_compute
            .borrow_mut()
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No compute node to finish"))?;
        node.commit_dispatch(&mut self.inner.borrow_mut(), workgroups.0, workgroups.1, workgroups.2);
        Ok(())
    }

    fn with_active_compute<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut ComputeNodeRecord) -> R,
    {
        let mut node = self.active_compute.borrow_mut();
        let node = node
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No compute node is open"))?;
        Ok(f(node))
    }
}

/// Parse a Python `range` or `slice` into `(start, count)` with step 1.
fn parse_index_range(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<(u32, u32)> {
    let start: i64 = match obj.getattr("start") {
        Ok(v) => v.extract().unwrap_or(0),
        Err(_) => 0,
    };
    let stop: i64 = obj
        .getattr("stop")
        .map_err(|_| PyValueError::new_err(format!("{name} must be a range or slice")))?
        .extract()?;
    let step: i64 = match obj.getattr("step") {
        Ok(v) => v.extract().unwrap_or(1),
        Err(_) => 1,
    };
    if step != 1 {
        return Err(PyValueError::new_err(format!("{name} range step must be 1")));
    }
    if start < 0 || stop < start {
        return Err(PyValueError::new_err(format!(
            "invalid {name} range: start={start}, stop={stop}"
        )));
    }
    Ok((start as u32, (stop - start) as u32))
}

#[pymethods]
impl PyTaskGraph {
    #[new]
    fn new() -> Self {
        PyTaskGraph {
            inner: RefCell::new(TaskGraph::new()),
            active_pass: RefCell::new(None),
            active_compute: RefCell::new(None),
            labels: RefCell::new(Vec::new()),
            swapchain: RefCell::new(None),
        }
    }

    /// Reset the graph for the next frame while retaining internal capacity.
    fn clear(&self) -> PyResult<()> {
        self.inner.borrow_mut().clear();
        *self.active_pass.borrow_mut() = None;
        *self.active_compute.borrow_mut() = None;
        self.labels.borrow_mut().clear();
        *self.swapchain.borrow_mut() = None;
        Ok(())
    }

    /// Upload CPU bytes into a buffer via the task graph.
    fn write_buffer(&self, buffer: &PyBuffer, offset: u64, data: &[u8]) -> PyResult<()> {
        self.ensure_no_active_recorder()?;
        self.inner
            .borrow_mut()
            .write_buffer(buffer.inner.as_ref(), offset, data.to_vec());
        Ok(())
    }

    /// Upload CPU bytes into a retained buffer parcel via the task graph.
    fn write_parcel(&self, parcel: &PyParcel, offset: u64, data: &[u8]) -> PyResult<()> {
        self.ensure_no_active_recorder()?;
        self.inner
            .borrow_mut()
            .write_parcel(parcel.inner.as_ref(), offset, data.to_vec())
            .into_py_result()?;
        Ok(())
    }

    /// Begin recording an offscreen render pass. Returns a context manager.
    fn render_pass(slf: Py<Self>, py: Python<'_>, label: String, target: &PyRenderTarget) -> PyResult<PyRenderPass> {
        {
            let graph = slf.borrow_mut(py);
            if graph.active_pass.borrow().is_some() || graph.active_compute.borrow().is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Only one recorder may be open per graph",
                ));
            }
            let static_label = graph.intern_label(&label)?;
            *graph.active_pass.borrow_mut() = Some(RenderPassRecord::new(static_label, &target.inner));
        }
        Ok(PyRenderPass { graph: slf })
    }

    /// Begin recording a compute dispatch node. Returns a context manager.
    #[pyo3(signature = (label, pipeline, workgroups=(1, 1, 1)))]
    fn compute_node(
        slf: Py<Self>,
        py: Python<'_>,
        label: String,
        pipeline: &PyComputePipeline,
        workgroups: (u32, u32, u32),
    ) -> PyResult<PyComputeNode> {
        {
            let graph = slf.borrow_mut(py);
            if graph.active_pass.borrow().is_some() || graph.active_compute.borrow().is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Only one recorder may be open per graph",
                ));
            }
            let static_label = graph.intern_label(&label)?;
            *graph.active_compute.borrow_mut() = Some(ComputeNodeRecord::new(static_label, &pipeline.inner));
        }
        Ok(PyComputeNode { graph: slf, workgroups })
    }

    /// Declare the swapchain output for windowed presentation.
    fn declare_swapchain_output(&self) -> PyResult<PySwapchainOutput> {
        self.ensure_no_active_recorder()?;
        let handle = self.inner.borrow_mut().declare_swapchain_output();
        *self.swapchain.borrow_mut() = Some(handle);
        Ok(PySwapchainOutput)
    }

    /// Blit an offscreen render target to the swapchain output.
    fn copy_render_target_to_swapchain(&self, target: &PyRenderTarget, _swapchain: &PySwapchainOutput) -> PyResult<()> {
        self.ensure_no_active_recorder()?;
        let handle = self.swapchain.borrow().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Call declare_swapchain_output() before copy_render_target_to_swapchain()",
            )
        })?;
        self.inner
            .borrow_mut()
            .copy_render_target_to_swapchain(&target.inner, handle);
        Ok(())
    }

    /// Submit the graph on a device context and block until complete (headless).
    fn dispatch(&self, device: &PyDevice) -> PyResult<()> {
        self.ensure_no_active_recorder()?;
        let ctx = device.inner.create_context().into_py_result()?;
        self.inner.borrow_mut().dispatch(&ctx).into_py_result()
    }

    fn __repr__(&self) -> String {
        format!("TaskGraph(nodes={})", self.inner.borrow().len())
    }
}

/// Records draw commands for one offscreen render pass.
#[pyclass(name = "RenderPass", module = "goldy", unsendable)]
pub struct PyRenderPass {
    graph: Py<PyTaskGraph>,
}

#[pymethods]
impl PyRenderPass {
    fn bind_buffer<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        buffer: &PyBuffer,
        access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.bind_buffer(buffer.inner.as_ref(), access.into());
        })?;
        Ok(slf)
    }

    fn bind_buffer_view<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        view: &PyBufferView,
        access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.bind_buffer_view(&view.inner, access.into());
        })?;
        Ok(slf)
    }

    fn bind_parcel<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        parcel: &PyParcel,
        access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.bind_parcel(parcel.inner.as_ref(), access.into());
        })?;
        Ok(slf)
    }

    /// Graph dependency + shader push-constant slot for a retained parcel (broadcast or scattered).
    ///
    /// Combines [`Self::bind_parcel`] with [`RenderPassRecord::bind_resources_typed`] using the
    /// parcel's typed [`ResourceHandle`] (correct category for broadcast uniforms).
    fn bind_parcel_shader_resource<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        parcel: &PyParcel,
        access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.bind_parcel(parcel.inner.as_ref(), access.into());
            let resource_access = resource_access_for_shader(access);
            let handle = parcel
                .inner
                .handle(resource_access)
                .ok_or_else(|| GoldyError::new_err("bindless resource handle unavailable"))?;
            pass.bind_resources_typed(&[handle]);
            Ok(())
        })??;
        Ok(slf)
    }

    /// Bind a scattered buffer resource by bindless index (from `BufferView.resource_index`).
    fn bind_resource_index<'py>(slf: PyRef<'py, Self>, py: Python<'py>, index: u32) -> PyResult<PyRef<'py, Self>> {
        let handle = ResourceHandle::new(ResourceCategory::Scattered, index);
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.bind_resources_typed(&[handle]);
        })?;
        Ok(slf)
    }

    fn clear<'py>(slf: PyRef<'py, Self>, py: Python<'py>, color: &PyColor) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.clear(color.inner);
        })?;
        Ok(slf)
    }

    fn clear_depth<'py>(slf: PyRef<'py, Self>, py: Python<'py>, depth: f32) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.clear_depth(depth);
        })?;
        Ok(slf)
    }

    fn set_pipeline<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        pipeline: &PyRenderPipeline,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.set_pipeline(&pipeline.inner);
        })?;
        Ok(slf)
    }

    fn set_vertex_buffer<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        slot: u32,
        buffer: &PyBuffer,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.set_vertex_buffer(slot, buffer.inner.as_ref());
        })?;
        Ok(slf)
    }

    fn set_vertex_buffer_parcel<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        slot: u32,
        parcel: &PyParcel,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.set_vertex_buffer(slot, parcel.inner.as_ref());
        })?;
        Ok(slf)
    }

    #[pyo3(signature = (slot, buffer, offset))]
    fn set_vertex_buffer_offset<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        slot: u32,
        buffer: &PyBuffer,
        offset: u64,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.set_vertex_buffer_offset(slot, buffer.inner.as_ref(), offset);
        })?;
        Ok(slf)
    }

    fn set_index_buffer<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        buffer: &PyBuffer,
        format: PyIndexFormat,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.set_index_buffer(buffer.inner.as_ref(), format.into());
        })?;
        Ok(slf)
    }

    /// Draw primitives.
    ///
    /// Pass a ``range`` for vertices/instances (e.g. ``draw(range(3))``) or use
    /// explicit ``first_vertex`` / ``vertex_count`` keyword arguments.
    #[allow(clippy::too_many_arguments)]
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
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.draw(fv, vc, fi, ic);
        })?;
        Ok(slf)
    }

    /// Draw indexed primitives.
    ///
    /// Pass a ``range`` for indices/instances or use explicit count keyword arguments.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (indices=None, *, first_index=0, index_count=None, base_vertex=0, instances=None, first_instance=0, instance_count=None))]
    fn draw_indexed<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        indices: Option<Bound<'py, PyAny>>,
        first_index: u32,
        index_count: Option<u32>,
        base_vertex: i32,
        instances: Option<Bound<'py, PyAny>>,
        first_instance: u32,
        instance_count: Option<u32>,
    ) -> PyResult<PyRef<'py, Self>> {
        let (fi, ic) = if let Some(idx) = indices {
            parse_index_range(&idx, "indices")?
        } else {
            (
                first_index,
                index_count
                    .ok_or_else(|| PyValueError::new_err("draw_indexed requires indices=range(...) or index_count="))?,
            )
        };
        let (inst_start, inst_count) = if let Some(i) = instances {
            parse_index_range(&i, "instances")?
        } else {
            (first_instance, instance_count.unwrap_or(1))
        };
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.draw_indexed(fi, ic, base_vertex, inst_start, inst_count);
        })?;
        Ok(slf)
    }

    fn draw_fullscreen<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.draw_fullscreen();
        })?;
        Ok(slf)
    }

    fn draw_quads<'py>(slf: PyRef<'py, Self>, py: Python<'py>, count: u32) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.draw_quads(count);
        })?;
        Ok(slf)
    }

    fn bind_resources<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        buffers: Vec<PyRef<'py, PyBuffer>>,
    ) -> PyResult<PyRef<'py, Self>> {
        let refs: Vec<&goldy::Buffer> = buffers.iter().map(|b| b.inner.as_ref()).collect();
        slf.graph.borrow(py).with_active_pass(|pass| {
            pass.bind_resources(&refs);
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
        self.graph.borrow(py).finish_render_pass()?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        "RenderPass(recording)".to_string()
    }
}

/// Records a compute dispatch node on a task graph.
#[pyclass(name = "ComputeNode", module = "goldy", unsendable)]
pub struct PyComputeNode {
    graph: Py<PyTaskGraph>,
    workgroups: (u32, u32, u32),
}

#[pymethods]
impl PyComputeNode {
    fn bind_parcel<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        parcel: &PyParcel,
        access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_compute(|node| {
            node.bind_parcel(parcel.inner.as_ref(), access.into());
        })?;
        Ok(slf)
    }

    fn bind_buffer<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        buffer: &PyBuffer,
        access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_compute(|node| {
            node.bind_buffer(buffer.inner.as_ref(), access.into());
        })?;
        Ok(slf)
    }

    fn bind_buffer_view<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        view: &PyBufferView,
        access: PyNodeAccess,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_compute(|node| {
            node.bind_buffer_view(&view.inner, access.into());
        })?;
        Ok(slf)
    }

    fn bind_resources_raw<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        indices: Vec<u32>,
    ) -> PyResult<PyRef<'py, Self>> {
        slf.graph.borrow(py).with_active_compute(|node| {
            node.bind_resources_raw(&indices);
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
        self.graph.borrow(py).finish_compute_node(self.workgroups)?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        "ComputeNode(recording)".to_string()
    }
}
