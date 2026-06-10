use crate::compute::ComputePipeline;
use crate::device::Device;
use crate::error::{check, expect_ok, non_null_expect, Result};
use crate::parcel::Parcel;
use crate::pipeline::RenderPipeline;
use crate::render_target::RenderTarget;
use crate::retained_pool::MosaicSlot;
use crate::sys::{self, GoldySwapchainOutput, GoldyTaskGraph};
use crate::types::{Color, IndexFormat, NodeAccess, ResourceHandle};
use std::ffi::CString;
use std::ops::Range;

/// GPU task graph for render passes, compute, and swapchain blits.
pub struct TaskGraph {
    ptr: *mut GoldyTaskGraph,
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn copy_render_target_to_swapchain(&mut self, src: &RenderTarget, _dst: SwapchainOutputHandle) {
        expect_ok(unsafe {
            sys::goldy_task_graph_copy_render_target_to_swapchain(self.ptr, src.as_ptr(), _dst._token)
        });
    }

    /// Analyze the graph, submit, and block until complete.
    pub fn dispatch(&mut self, device: &Device) -> Result<()> {
        check(unsafe { sys::goldy_task_graph_dispatch(self.ptr, device.as_ptr()) })
    }

    /// Add a CPU→GPU write node for a retained buffer [`Parcel`].
    pub fn write_parcel(&mut self, parcel: &Parcel, offset: u64, data: &[u8]) -> Result<()> {
        check(unsafe {
            sys::goldy_task_graph_write_parcel(self.ptr, parcel.as_ptr(), offset, data.as_ptr(), data.len())
        })
    }

    pub fn render_pass<'a>(&'a mut self, label: &'static str, target: &RenderTarget) -> RenderPassBuilder<'a> {
        let label = CString::new(label).expect("render pass label contains interior null byte");
        expect_ok(unsafe { sys::goldy_task_graph_render_pass_begin(self.ptr, label.as_ptr(), target.as_ptr()) });
        RenderPassBuilder {
            graph: self,
            active: true,
        }
    }

    pub fn compute_node<'a>(&'a mut self, label: &'static str, pipeline: &ComputePipeline) -> ComputeNodeBuilder<'a> {
        let label = CString::new(label).expect("compute node label contains interior null byte");
        expect_ok(unsafe { sys::goldy_task_graph_compute_node_begin(self.ptr, label.as_ptr(), pipeline.as_ptr()) });
        ComputeNodeBuilder {
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
    pub fn bind_parcel_mut(&mut self, parcel: &Parcel, access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_render_pass_bind_parcel(self.graph.ptr, parcel.as_ptr(), access.into())
        });
        self
    }

    pub fn bind_parcel_view_mut(&mut self, parcel: &Parcel, slot: MosaicSlot, access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_render_pass_bind_parcel_view(self.graph.ptr, parcel.as_ptr(), slot.0, access.into())
        });
        self
    }

    pub fn clear(&mut self, color: Color) -> &mut Self {
        expect_ok(unsafe { sys::goldy_task_graph_render_pass_clear(self.graph.ptr, color.into()) });
        self
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) -> &mut Self {
        expect_ok(unsafe { sys::goldy_task_graph_render_pass_set_pipeline(self.graph.ptr, pipeline.as_ptr()) });
        self
    }

    pub fn set_vertex_buffer_parcel(&mut self, slot: u32, parcel: &Parcel) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_render_pass_set_vertex_buffer_parcel(self.graph.ptr, slot, parcel.as_ptr())
        });
        self
    }

    pub fn set_index_buffer_parcel(&mut self, parcel: &Parcel, format: IndexFormat) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_render_pass_set_index_buffer(self.graph.ptr, parcel.as_ptr(), format.into())
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

    pub fn draw_fullscreen(&mut self) -> &mut Self {
        expect_ok(unsafe { sys::goldy_task_graph_render_pass_draw_fullscreen(self.graph.ptr) });
        self
    }

    pub fn bind_resources_typed(&mut self, handles: &[ResourceHandle]) -> &mut Self {
        let mut flat = Vec::with_capacity(handles.len() * 2);
        for h in handles {
            flat.push(h.category as u32);
            flat.push(h.index);
        }
        expect_ok(unsafe {
            sys::goldy_task_graph_render_pass_bind_resources_typed(self.graph.ptr, flat.as_ptr(), handles.len() as u32)
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

/// Builder for recording one compute dispatch node on a task graph.
pub struct ComputeNodeBuilder<'a> {
    graph: &'a mut TaskGraph,
    active: bool,
}

impl ComputeNodeBuilder<'_> {
    pub fn bind_parcel(&mut self, parcel: &Parcel, access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_compute_node_bind_parcel(self.graph.ptr, parcel.as_ptr(), access.into())
        });
        self
    }

    pub fn bind_parcel_view(&mut self, parcel: &Parcel, slot: MosaicSlot, access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_compute_node_bind_parcel_view(self.graph.ptr, parcel.as_ptr(), slot.0, access.into())
        });
        self
    }

    pub fn bind_resources_raw(&mut self, indices: &[u32]) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_task_graph_compute_node_bind_resources_raw(
                self.graph.ptr,
                indices.as_ptr(),
                indices.len() as u32,
            )
        });
        self
    }

    pub fn dispatch(mut self, workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) {
        if self.active {
            expect_ok(unsafe {
                sys::goldy_task_graph_compute_node_dispatch(self.graph.ptr, workgroups_x, workgroups_y, workgroups_z)
            });
            self.active = false;
        }
    }
}

impl Drop for ComputeNodeBuilder<'_> {
    fn drop(&mut self) {
        debug_assert!(!self.active, "ComputeNodeBuilder dropped without dispatch()");
        self.active = false;
    }
}
