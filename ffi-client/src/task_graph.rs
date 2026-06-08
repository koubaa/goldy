use crate::buffer::Buffer;
use crate::error::{expect_ok, non_null_expect};
use crate::pipeline::RenderPipeline;
use crate::render_target::RenderTarget;
use crate::sys::{self, GoldySwapchainOutput, GoldyTaskGraph};
use crate::types::{Color, NodeAccess};
use std::ffi::CString;
use std::ops::Range;

/// GPU task graph for render passes, compute, and swapchain blits.
pub struct TaskGraph {
    ptr: *mut GoldyTaskGraph,
}

impl TaskGraph {
    pub fn new() -> Self {
        let ptr = non_null_expect(unsafe { sys::goldy_task_graph_create() });
        Self { ptr }
    }

    pub fn clear(&mut self) {
        expect_ok(unsafe { sys::goldy_task_graph_clear(self.ptr) });
    }

    pub fn declare_swapchain_output(&mut self) -> SwapchainOutputHandle {
        let ptr = non_null_expect(unsafe { sys::goldy_task_graph_declare_swapchain_output(self.ptr) });
        SwapchainOutputHandle { _token: ptr }
    }

    pub fn copy_render_target_to_swapchain(
        &mut self,
        src: &RenderTarget,
        _dst: SwapchainOutputHandle,
    ) {
        expect_ok(unsafe {
            sys::goldy_task_graph_copy_render_target_to_swapchain(
                self.ptr,
                src.as_ptr(),
                _dst._token,
            )
        });
    }

    pub fn render_pass<'a>(
        &'a mut self,
        label: &'static str,
        target: &RenderTarget,
    ) -> RenderPassBuilder<'a> {
        let label = CString::new(label).expect("render pass label contains interior null byte");
        expect_ok(unsafe {
            sys::goldy_task_graph_render_pass_begin(self.ptr, label.as_ptr(), target.as_ptr())
        });
        RenderPassBuilder {
            graph: self,
            active: true,
        }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut GoldyTaskGraph {
        self.ptr
    }
}

impl Drop for TaskGraph {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_task_graph_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Non-owning token from [`TaskGraph::declare_swapchain_output`].
///
/// Points at storage inside the parent graph; pass to [`TaskGraph::copy_render_target_to_swapchain`].
#[derive(Clone, Copy)]
pub struct SwapchainOutputHandle {
    pub(crate) _token: *const GoldySwapchainOutput,
}

/// Builder for recording one render pass on a task graph.
pub struct RenderPassBuilder<'a> {
    graph: &'a mut TaskGraph,
    active: bool,
}

impl RenderPassBuilder<'_> {
    pub fn bind_buffer_mut(&mut self, buffer: &Buffer, access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_render_pass_bind_buffer(
                self.graph.ptr,
                buffer.as_ptr(),
                access.into(),
            )
        });
        self
    }

    pub fn clear(&mut self, color: Color) -> &mut Self {
        expect_ok(unsafe { sys::goldy_task_graph_render_pass_clear(self.graph.ptr, color.into()) });
        self
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_render_pass_set_pipeline(self.graph.ptr, pipeline.as_ptr())
        });
        self
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_render_pass_set_vertex_buffer(
                self.graph.ptr,
                slot,
                buffer.as_ptr(),
            )
        });
        self
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_render_pass_draw(
                self.graph.ptr,
                vertices.start,
                vertices.end - vertices.start,
                instances.start,
                instances.end - instances.start,
            )
        });
        self
    }

    pub fn finish_recorded(mut self) {
        if self.active {
            expect_ok(unsafe { sys::goldy_task_graph_render_pass_finish(self.graph.ptr) });
            self.active = false;
        }
    }
}

impl Drop for RenderPassBuilder<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = unsafe { sys::goldy_task_graph_render_pass_finish(self.graph.ptr) };
            self.active = false;
        }
    }
}
