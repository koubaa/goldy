use crate::buffer::Buffer;
use crate::compute::ComputePipeline;
use crate::context::Context;
use crate::error::{check, expect_ok, non_null_expect, Result};
use crate::parcel::Parcel;
use crate::pipeline::RenderPipeline;
use crate::sys::{
    self, GoldyPresentLease, GoldyReplayStats, GoldyScheme, GoldySchemeRenderTargetLease, GoldySchemeSubmission,
};
use crate::texture::Texture;
use crate::types::{DepthFormat, IndexFormat, NodeAccess, TextureFormat};
use std::ffi::CString;
use std::ops::Range;

/// Per-submission identity returned by [`Scheme::submit`].
pub struct SchemeSubmission {
    ptr: *mut GoldySchemeSubmission,
}

impl SchemeSubmission {
    pub fn is_settled(&self) -> bool {
        unsafe { sys::goldy_scheme_submission_is_settled(self.ptr) }
    }

    pub fn wait_until_settled(&self) -> Result<()> {
        check(unsafe { sys::goldy_scheme_submission_wait_until_settled(self.ptr) })
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut GoldySchemeSubmission {
        self.ptr
    }
}

impl Drop for SchemeSubmission {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_scheme_submission_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Stable render-target lease declared on a [`Scheme`].
pub struct SchemeRenderTargetLease {
    ptr: *mut GoldySchemeRenderTargetLease,
}

impl SchemeRenderTargetLease {
    pub(crate) fn as_ptr(&self) -> *const GoldySchemeRenderTargetLease {
        self.ptr
    }
}

impl Drop for SchemeRenderTargetLease {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_scheme_render_target_lease_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Stable present lease from a surface exchange.
pub struct PresentLease {
    ptr: *mut GoldyPresentLease,
}

impl PresentLease {
    pub(crate) fn from_ptr(ptr: *mut GoldyPresentLease) -> Self {
        Self { ptr }
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyPresentLease {
        self.ptr
    }
}

impl Drop for PresentLease {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_present_lease_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Retained scheme bound to one [`Context`].
pub struct Scheme {
    ptr: *mut GoldyScheme,
}

impl Scheme {
    pub fn new(ctx: &Context) -> Result<Self> {
        let ptr = non_null_expect(unsafe { sys::goldy_scheme_create(ctx.as_ptr()) });
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *mut GoldyScheme {
        self.ptr
    }

    pub fn len(&self) -> u32 {
        unsafe { sys::goldy_scheme_len(self.ptr) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_dirty(&self) -> bool {
        unsafe { sys::goldy_scheme_is_dirty(self.ptr) }
    }

    pub fn replay_stats(&self) -> Result<ReplayStats> {
        let mut stats = GoldyReplayStats::default();
        check(unsafe { sys::goldy_scheme_replay_stats(self.ptr, &mut stats) })?;
        Ok(ReplayStats {
            records: stats.records,
            resubmit_hits: stats.resubmit_hits,
        })
    }

    pub fn lease_render_target(
        &mut self,
        width: u32,
        height: u32,
        format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<SchemeRenderTargetLease> {
        let (has_depth, depth) = match depth_format {
            Some(d) => (true, d),
            None => (false, DepthFormat::Depth24Plus),
        };
        let ptr = non_null_expect(unsafe {
            sys::goldy_scheme_lease_render_target(self.ptr, width, height, format.into(), has_depth, depth.into())
        });
        Ok(SchemeRenderTargetLease { ptr })
    }

    pub fn copy_to_texture(&mut self, src: &SchemeRenderTargetLease, dst: &Texture) -> Result<()> {
        check(unsafe { sys::goldy_scheme_copy_to_texture(self.ptr, src.as_ptr(), dst.as_ptr()) })
    }

    pub fn submit(&mut self) -> Result<SchemeSubmission> {
        let mut submission = std::ptr::null_mut();
        check(unsafe { sys::goldy_scheme_submit(self.ptr, &mut submission) })?;
        Ok(SchemeSubmission { ptr: submission })
    }

    pub fn compute_node<'a>(&'a mut self, label: &'static str, pipeline: &ComputePipeline) -> ComputeNodeBuilder<'a> {
        let label = CString::new(label).expect("compute node label contains interior null byte");
        expect_ok(unsafe { sys::goldy_scheme_compute_node_begin(self.ptr, label.as_ptr(), pipeline.as_ptr()) });
        ComputeNodeBuilder {
            scheme: self,
            active: true,
        }
    }

    pub fn render_pass<'a>(
        &'a mut self,
        label: &'static str,
        target: &SchemeRenderTargetLease,
        load: crate::types::TargetLoad,
    ) -> SchemeRenderPassBuilder<'a> {
        let label = CString::new(label).expect("render pass label contains interior null byte");
        let (load_kind, clear_color) = load.to_ffi();
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_begin(self.ptr, label.as_ptr(), target.as_ptr(), load_kind, clear_color)
        });
        SchemeRenderPassBuilder {
            scheme: self,
            active: true,
        }
    }
}

/// Submission outcome counters from [`Scheme::replay_stats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayStats {
    pub records: u64,
    pub resubmit_hits: u64,
}

/// Builder for one compute dispatch node on a [`Scheme`].
pub struct ComputeNodeBuilder<'a> {
    scheme: &'a mut Scheme,
    active: bool,
}

impl ComputeNodeBuilder<'_> {
    pub fn with_parcel(&mut self, parcel: &Parcel, node_access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_compute_node_with_parcel(self.scheme.ptr, parcel.as_ptr(), node_access.into())
        });
        self
    }

    pub fn with_buffer_unit(&mut self, buffer: &Buffer, unit: u32, node_access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_compute_node_with_buffer_unit(self.scheme.ptr, buffer.as_ptr(), unit, node_access.into())
        });
        self
    }

    pub fn with_buffer(&mut self, buffer: &Buffer, node_access: NodeAccess) -> &mut Self {
        self.with_buffer_unit(buffer, 0, node_access)
    }

    pub fn with_param(&mut self, value: u32) -> &mut Self {
        expect_ok(unsafe { sys::goldy_scheme_compute_node_with_param(self.scheme.ptr, value) });
        self
    }

    pub fn dispatch(mut self, workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) {
        if self.active {
            expect_ok(unsafe {
                sys::goldy_scheme_compute_node_dispatch(self.scheme.ptr, workgroups_x, workgroups_y, workgroups_z)
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

/// Builder for recording one render pass on a [`Scheme`].
pub struct SchemeRenderPassBuilder<'a> {
    scheme: &'a mut Scheme,
    active: bool,
}

impl SchemeRenderPassBuilder<'_> {
    pub fn with_parcel(&mut self, parcel: &Parcel, access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_with_parcel(self.scheme.ptr, parcel.as_ptr(), access.into())
        });
        self
    }

    pub fn with_buffer_unit(&mut self, buffer: &Buffer, unit: u32, access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_with_buffer_unit(self.scheme.ptr, buffer.as_ptr(), unit, access.into())
        });
        self
    }

    pub fn with_buffer(&mut self, buffer: &Buffer, access: NodeAccess) -> &mut Self {
        self.with_buffer_unit(buffer, 0, access)
    }

    pub fn clear_depth(&mut self, depth: f32) -> &mut Self {
        expect_ok(unsafe { sys::goldy_scheme_render_pass_clear_depth(self.scheme.ptr, depth) });
        self
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) -> &mut Self {
        expect_ok(unsafe { sys::goldy_scheme_render_pass_set_pipeline(self.scheme.ptr, pipeline.as_ptr()) });
        self
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer) -> &mut Self {
        let parcel = buffer.field(0).expect("buffer has no field 0");
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_set_vertex_buffer_parcel(self.scheme.ptr, slot, parcel.as_ptr())
        });
        self
    }

    pub fn set_vertex_buffer_parcel(&mut self, slot: u32, parcel: &Parcel) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_set_vertex_buffer_parcel(self.scheme.ptr, slot, parcel.as_ptr())
        });
        self
    }

    pub fn set_index_buffer(&mut self, buffer: &Buffer, format: IndexFormat) -> &mut Self {
        let parcel = buffer.field(0).expect("buffer has no field 0");
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_set_index_buffer(self.scheme.ptr, parcel.as_ptr(), format.into())
        });
        self
    }

    pub fn set_index_buffer_parcel(&mut self, parcel: &Parcel, format: IndexFormat) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_set_index_buffer(self.scheme.ptr, parcel.as_ptr(), format.into())
        });
        self
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_draw(
                self.scheme.ptr,
                vertices.start,
                vertices.end - vertices.start,
                instances.start,
                instances.end - instances.start,
            )
        });
        self
    }

    pub fn draw_fullscreen(&mut self) -> &mut Self {
        expect_ok(unsafe { sys::goldy_scheme_render_pass_draw_fullscreen(self.scheme.ptr) });
        self
    }

    pub fn finish_recorded(mut self) {
        if self.active {
            expect_ok(unsafe { sys::goldy_scheme_render_pass_finish(self.scheme.ptr) });
            self.active = false;
        }
    }
}

impl Drop for SchemeRenderPassBuilder<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = unsafe { sys::goldy_scheme_render_pass_finish(self.scheme.ptr) };
            self.active = false;
        }
    }
}

impl Drop for Scheme {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_scheme_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
