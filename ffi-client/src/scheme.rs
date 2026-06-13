use crate::compute::ComputePipeline;
use crate::context::Context;
use crate::error::{check, expect_ok, non_null_expect, Result};
use crate::parcel::Parcel;
use crate::retained_pool::MosaicSlot;
use crate::sys::{self, GoldyReplayStats, GoldyScheme};
use crate::types::{NodeAccess, ResourceAccess};
use std::ffi::CString;

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

    /// Submit and block until GPU completion (temporary; see project docs).
    pub fn submit(&mut self) -> Result<()> {
        check(unsafe { sys::goldy_scheme_submit(self.ptr) })
    }

    pub fn compute_node<'a>(&'a mut self, label: &'static str, pipeline: &ComputePipeline) -> ComputeNodeBuilder<'a> {
        let label = CString::new(label).expect("compute node label contains interior null byte");
        expect_ok(unsafe {
            sys::goldy_scheme_compute_node_begin(self.ptr, label.as_ptr(), pipeline.as_ptr())
        });
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
