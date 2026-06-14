use crate::compute::ComputePipeline;
use crate::context::Context;
use crate::error::{check, expect_ok, non_null_expect, Result};
use crate::parcel::Parcel;
use crate::retained_pool::MosaicSlot;
use crate::sys::{self, GoldyReadGrant, GoldyReplayStats, GoldyScheme, GoldySchemeFrame};
use crate::types::{NodeAccess, ResourceAccess};
use std::ffi::CString;

/// Per-submission identity returned by [`Scheme::submit`].
pub struct SchemeFrame {
    ptr: *mut GoldySchemeFrame,
}

impl SchemeFrame {
    pub fn timeline_value(&self) -> u64 {
        unsafe { sys::goldy_scheme_frame_timeline_value(self.ptr) }
    }

    pub fn wait(&self, ctx: &Context) -> Result<()> {
        check(unsafe { sys::goldy_scheme_frame_wait(ctx.as_ptr(), self.ptr) })
    }
}

impl Drop for SchemeFrame {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_scheme_frame_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Read easement grant recorded once via [`Scheme::grant_read`].
pub struct ReadGrant {
    ptr: *mut GoldyReadGrant,
}

impl ReadGrant {
    pub fn byte_size(&self) -> u64 {
        unsafe { sys::goldy_read_grant_byte_size(self.ptr) }
    }

    /// Readable bytes for `frame`'s submission (full logical buffer size).
    pub fn read(&self, frame: &SchemeFrame) -> Result<Vec<u8>> {
        let mut output = vec![0u8; self.byte_size() as usize];
        check(unsafe { sys::goldy_read_grant_read(self.ptr, frame.ptr, output.as_mut_ptr(), output.len()) })?;
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

/// Retained compute scheme bound to one [`Context`].
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

    /// Record a read easement over a buffer parcel (once per scheme).
    pub fn grant_read(&mut self, parcel: &Parcel) -> Result<ReadGrant> {
        let ptr = non_null_expect(unsafe { sys::goldy_scheme_grant_read(self.ptr, parcel.as_ptr()) });
        Ok(ReadGrant { ptr })
    }

    pub fn submit(&mut self) -> Result<SchemeFrame> {
        let mut frame = std::ptr::null_mut();
        check(unsafe { sys::goldy_scheme_submit(self.ptr, &mut frame) })?;
        Ok(SchemeFrame { ptr: frame })
    }

    pub fn compute_node<'a>(&'a mut self, label: &'static str, pipeline: &ComputePipeline) -> ComputeNodeBuilder<'a> {
        let label = CString::new(label).expect("compute node label contains interior null byte");
        expect_ok(unsafe { sys::goldy_scheme_compute_node_begin(self.ptr, label.as_ptr(), pipeline.as_ptr()) });
        ComputeNodeBuilder {
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

impl Drop for Scheme {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_scheme_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
