//! Python wrappers for TaskGraph and render-pass recording.

use crate::buffer::PyBuffer;
use crate::device::PyDevice;
use crate::error::IntoPyResult;
use crate::pipeline::PyRenderPipeline;
use crate::render_target::PyRenderTarget;
use crate::types::{PyColor, PyIndexFormat, PyNodeAccess};
use goldy::task_graph::{RenderPassRecord, SwapchainOutputHandle, TaskGraph};
use pyo3::prelude::*;
use std::cell::RefCell;

/// Opaque token from [`PyTaskGraph::declare_swapchain_output`].
#[pyclass(name = "SwapchainOutput", module = "goldy", unsendable)]
pub struct PySwapchainOutput;

/// GPU task graph: render passes, swapchain blit, and dispatch.
#[pyclass(name = "TaskGraph", module = "goldy", unsendable)]
pub struct PyTaskGraph {
    pub(crate) inner: RefCell<TaskGraph>,
    active_pass: RefCell<Option<RenderPassRecord>>,
    labels: RefCell<Vec<String>>,
    swapchain: RefCell<Option<SwapchainOutputHandle>>,
}

impl PyTaskGraph {
    pub(crate) fn ensure_no_active_pass(&self) -> PyResult<()> {
        if self.active_pass.borrow().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "A render pass is still open; finish it before submitting the graph",
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
        pass.commit(&mut *self.inner.borrow_mut());
        Ok(())
    }
}

#[pymethods]
impl PyTaskGraph {
    #[new]
    fn new() -> Self {
        PyTaskGraph {
            inner: RefCell::new(TaskGraph::new()),
            active_pass: RefCell::new(None),
            labels: RefCell::new(Vec::new()),
            swapchain: RefCell::new(None),
        }
    }

    /// Reset the graph for the next frame while retaining internal capacity.
    fn clear(&self) -> PyResult<()> {
        self.inner.borrow_mut().clear();
        *self.active_pass.borrow_mut() = None;
        self.labels.borrow_mut().clear();
        *self.swapchain.borrow_mut() = None;
        Ok(())
    }

    /// Begin recording an offscreen render pass. Returns a context manager.
    fn render_pass(slf: Py<Self>, py: Python<'_>, label: String, target: &PyRenderTarget) -> PyResult<PyRenderPass> {
        {
            let graph = slf.borrow_mut(py);
            if graph.active_pass.borrow().is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Only one render pass may be open per graph",
                ));
            }
            let static_label = graph.intern_label(&label)?;
            *graph.active_pass.borrow_mut() = Some(RenderPassRecord::new(static_label, &target.inner));
        }
        Ok(PyRenderPass { graph: slf })
    }

    /// Declare the swapchain output for windowed presentation.
    fn declare_swapchain_output(&self) -> PyResult<PySwapchainOutput> {
        self.ensure_no_active_pass()?;
        let handle = self.inner.borrow_mut().declare_swapchain_output();
        *self.swapchain.borrow_mut() = Some(handle);
        Ok(PySwapchainOutput)
    }

    /// Blit an offscreen render target to the swapchain output.
    fn copy_render_target_to_swapchain(
        &self,
        target: &PyRenderTarget,
        _swapchain: &PySwapchainOutput,
    ) -> PyResult<()> {
        self.ensure_no_active_pass()?;
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
        self.ensure_no_active_pass()?;
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
    fn bind_buffer(&self, py: Python<'_>, buffer: &PyBuffer, access: PyNodeAccess) -> PyResult<()> {
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.bind_buffer(buffer.inner.as_ref(), access.into());
        })
    }

    fn clear(&self, py: Python<'_>, color: &PyColor) -> PyResult<()> {
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.clear(color.inner);
        })
    }

    fn clear_depth(&self, py: Python<'_>, depth: f32) -> PyResult<()> {
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.clear_depth(depth);
        })
    }

    fn set_pipeline(&self, py: Python<'_>, pipeline: &PyRenderPipeline) -> PyResult<()> {
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.set_pipeline(&pipeline.inner);
        })
    }

    fn set_vertex_buffer(&self, py: Python<'_>, slot: u32, buffer: &PyBuffer) -> PyResult<()> {
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.set_vertex_buffer(slot, buffer.inner.as_ref());
        })
    }

    #[pyo3(signature = (slot, buffer, offset))]
    fn set_vertex_buffer_offset(
        &self,
        py: Python<'_>,
        slot: u32,
        buffer: &PyBuffer,
        offset: u64,
    ) -> PyResult<()> {
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.set_vertex_buffer_offset(slot, buffer.inner.as_ref(), offset);
        })
    }

    fn set_index_buffer(&self, py: Python<'_>, buffer: &PyBuffer, format: PyIndexFormat) -> PyResult<()> {
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.set_index_buffer(buffer.inner.as_ref(), format.into());
        })
    }

    #[pyo3(signature = (first_vertex=0, vertex_count=3, first_instance=0, instance_count=1))]
    fn draw(
        &self,
        py: Python<'_>,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) -> PyResult<()> {
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.draw(first_vertex, vertex_count, first_instance, instance_count);
        })
    }

    #[pyo3(signature = (first_index, index_count, base_vertex=0, first_instance=0, instance_count=1))]
    fn draw_indexed(
        &self,
        py: Python<'_>,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) -> PyResult<()> {
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.draw_indexed(first_index, index_count, base_vertex, first_instance, instance_count);
        })
    }

    fn draw_fullscreen(&self, py: Python<'_>) -> PyResult<()> {
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.draw_fullscreen();
        })
    }

    fn bind_resources(&self, py: Python<'_>, buffers: Vec<PyRef<'_, PyBuffer>>) -> PyResult<()> {
        let refs: Vec<&goldy::Buffer> = buffers.iter().map(|b| b.inner.as_ref()).collect();
        self.graph.borrow(py).with_active_pass(|pass| {
            pass.bind_resources(&refs);
        })
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
