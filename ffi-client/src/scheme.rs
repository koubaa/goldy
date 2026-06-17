use crate::compute::ComputePipeline;
use crate::context::Context;
use crate::error::{check, expect_ok, non_null_expect, Result};
use crate::parcel::Parcel;
use crate::pipeline::RenderPipeline;
use crate::retained_pool::MosaicSlot;
use crate::sys::{
    self, GoldyPresentGrant, GoldyPresentLease, GoldyReadGrant, GoldyReplayStats, GoldyScheme,
    GoldySchemeRenderTargetLease, GoldySchemeSubmission,
};
use crate::types::{Color, DepthFormat, IndexFormat, NodeAccess, ResourceAccess, ResourceHandle, TextureFormat};
use std::ffi::CString;
use std::ops::Range;

/// Per-submission identity returned by [`Scheme::submit`].
pub struct SchemeSubmission {
    ptr: *mut GoldySchemeSubmission,
}

impl SchemeSubmission {
    pub fn timeline_value(&self) -> u64 {
        unsafe { sys::goldy_scheme_submission_timeline_value(self.ptr) }
    }

    pub fn wait(&self, ctx: &Context) -> Result<()> {
        check(unsafe { sys::goldy_scheme_submission_wait(ctx.as_ptr(), self.ptr) })
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

/// Read easement grant recorded once via [`Scheme::grant_read`] or [`Scheme::grant_read_texture`].
pub struct ReadGrant {
    ptr: *mut GoldyReadGrant,
}

impl ReadGrant {
    pub fn byte_size(&self) -> u64 {
        unsafe { sys::goldy_read_grant_byte_size(self.ptr) }
    }

    /// Consumable bytes for `submission`'s cell (full logical buffer size).
    pub fn consume(&self, submission: &SchemeSubmission) -> Result<Vec<u8>> {
        let mut output = vec![0u8; self.byte_size() as usize];
        check(unsafe { sys::goldy_read_grant_consume(self.ptr, submission.ptr, output.as_mut_ptr(), output.len()) })?;
        Ok(output)
    }
}

impl Drop for ReadGrant {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_read_grant_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Present easement grant recorded once via [`Scheme::grant_present`].
pub struct PresentGrant {
    ptr: *mut GoldyPresentGrant,
}

impl PresentGrant {
    pub fn consume(&self, submission: &SchemeSubmission) -> Result<()> {
        check(unsafe { sys::goldy_present_grant_consume(self.ptr, submission.ptr) })
    }
}

impl Drop for PresentGrant {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_present_grant_destroy(self.ptr) };
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

/// Stable present lease from a swapchain pool.
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

    /// Record a read easement over a buffer parcel (once per scheme).
    pub fn grant_read(&mut self, parcel: &Parcel) -> Result<ReadGrant> {
        let ptr = non_null_expect(unsafe { sys::goldy_scheme_grant_read(self.ptr, parcel.as_ptr()) });
        Ok(ReadGrant { ptr })
    }

    /// Record a read easement over a texture parcel (once per scheme).
    pub fn grant_read_texture(&mut self, parcel: &Parcel) -> Result<ReadGrant> {
        let ptr = non_null_expect(unsafe { sys::goldy_scheme_grant_read_texture(self.ptr, parcel.as_ptr()) });
        Ok(ReadGrant { ptr })
    }

    pub fn copy_to_texture(&mut self, src: &SchemeRenderTargetLease, dst: &Parcel) -> Result<()> {
        check(unsafe { sys::goldy_scheme_copy_to_texture(self.ptr, src.as_ptr(), dst.as_ptr()) })
    }

    pub fn copy_to_present(&mut self, src: &SchemeRenderTargetLease, dst: &PresentLease) -> Result<()> {
        check(unsafe { sys::goldy_scheme_copy_to_present(self.ptr, src.as_ptr(), dst.as_ptr()) })
    }

    pub fn grant_present(&mut self, lease: &PresentLease) -> Result<PresentGrant> {
        let ptr = non_null_expect(unsafe { sys::goldy_scheme_grant_present(self.ptr, lease.as_ptr()) });
        Ok(PresentGrant { ptr })
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
    ) -> SchemeRenderPassBuilder<'a> {
        let label = CString::new(label).expect("render pass label contains interior null byte");
        expect_ok(unsafe { sys::goldy_scheme_render_pass_begin(self.ptr, label.as_ptr(), target.as_ptr()) });
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
    pub fn declare_parcel(
        &mut self,
        parcel: &Parcel,
        node_access: NodeAccess,
        resource_access: ResourceAccess,
    ) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_compute_node_declare_parcel(
                self.scheme.ptr,
                parcel.as_ptr(),
                node_access.into(),
                resource_access.into(),
            )
        });
        self
    }

    pub fn declare_parcel_view(
        &mut self,
        parcel: &Parcel,
        slot: MosaicSlot,
        node_access: NodeAccess,
        resource_access: ResourceAccess,
    ) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_compute_node_declare_parcel_view(
                self.scheme.ptr,
                parcel.as_ptr(),
                slot.0,
                node_access.into(),
                resource_access.into(),
            )
        });
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
    pub fn bind_parcel_mut(&mut self, parcel: &Parcel, access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_bind_parcel(self.scheme.ptr, parcel.as_ptr(), access.into())
        });
        self
    }

    pub fn bind_parcel_view_mut(&mut self, parcel: &Parcel, slot: MosaicSlot, access: NodeAccess) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_bind_parcel_view(self.scheme.ptr, parcel.as_ptr(), slot.0, access.into())
        });
        self
    }

    pub fn clear(&mut self, color: Color) -> &mut Self {
        expect_ok(unsafe { sys::goldy_scheme_render_pass_clear(self.scheme.ptr, color.into()) });
        self
    }

    pub fn clear_depth(&mut self, depth: f32) -> &mut Self {
        expect_ok(unsafe { sys::goldy_scheme_render_pass_clear_depth(self.scheme.ptr, depth) });
        self
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) -> &mut Self {
        expect_ok(unsafe { sys::goldy_scheme_render_pass_set_pipeline(self.scheme.ptr, pipeline.as_ptr()) });
        self
    }

    pub fn set_vertex_buffer_parcel(&mut self, slot: u32, parcel: &Parcel) -> &mut Self {
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_set_vertex_buffer_parcel(self.scheme.ptr, slot, parcel.as_ptr())
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

    pub fn bind_resources_typed(&mut self, handles: &[ResourceHandle]) -> &mut Self {
        let mut flat = Vec::with_capacity(handles.len() * 2);
        for h in handles {
            flat.push(h.category as u32);
            flat.push(h.index);
        }
        expect_ok(unsafe {
            sys::goldy_scheme_render_pass_bind_resources_typed(self.scheme.ptr, flat.as_ptr(), handles.len() as u32)
        });
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
