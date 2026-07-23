//! FFI bindings for [`goldy::Scheme`].

#[path = "scheme_graphics.rs"]
mod scheme_graphics;

pub use scheme_graphics::*;

use crate::compute::GoldyComputePipeline;
use crate::context::GoldyContext;
use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::retained_pool::{buffer_unit_at, GoldyBuffer, GoldyParcel, GoldyTexture};
use crate::types::GoldyNodeAccess;
use goldy::scheme::Scheme;
use goldy::task_graph::{ComputeNodeRecord, NodeAccess, RenderPassRecord};
use std::ffi::CStr;

/// Opaque per-submission token returned by [`goldy_scheme_submit`].
///
/// Heap-allocated; destroy with [`goldy_scheme_submission_destroy`].
pub struct GoldySchemeSubmission {
    pub(crate) inner: goldy::Submission,
}

/// Outcome counters for [`goldy_scheme_replay_stats`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GoldyReplayStats {
    pub records: u64,
    pub resubmit_hits: u64,
}

/// Opaque handle to a retained Goldy scheme.
pub struct GoldyScheme {
    pub(crate) inner: Scheme,
    active_compute: Option<ComputeNodeRecord>,
    pub(crate) active_render_pass: Option<RenderPassRecord>,
    labels: Vec<String>,
}

impl GoldyScheme {
    pub(crate) fn intern_label(&mut self, label: &str) -> &'static str {
        self.labels.push(label.to_string());
        let s = self.labels.last().unwrap();
        // SAFETY: `labels` is cleared when the scheme is dropped alongside IR node labels.
        unsafe { std::mem::transmute::<&str, &'static str>(s.as_str()) }
    }
}

fn parse_label(label: *const libc::c_char) -> Result<String, GoldyResult> {
    if label.is_null() {
        set_last_error("Scheme label is null");
        return Err(GoldyResult::NullPointer);
    }
    let s = unsafe { CStr::from_ptr(label) };
    s.to_str().map(|s| s.to_string()).map_err(|e| {
        set_last_error_from_anyhow(&anyhow::anyhow!("Invalid UTF-8 label: {e}"));
        GoldyResult::InvalidArgument
    })
}

fn map_node_access(access: GoldyNodeAccess) -> NodeAccess {
    match access {
        GoldyNodeAccess::Read => NodeAccess::Read,
        GoldyNodeAccess::Write => NodeAccess::Write,
        GoldyNodeAccess::ReadWrite => NodeAccess::ReadWrite,
        GoldyNodeAccess::Overwrite => NodeAccess::Overwrite,
    }
}

fn active_compute_mut(scheme: &mut GoldyScheme) -> Result<&mut ComputeNodeRecord, GoldyResult> {
    scheme.active_compute.as_mut().ok_or_else(|| {
        set_last_error("No compute node is being recorded; call goldy_scheme_compute_node_begin first");
        GoldyResult::InvalidArgument
    })
}

/// Create a scheme bound to `ctx`.
///
/// # Safety
/// `ctx` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_create(ctx: *const GoldyContext) -> *mut GoldyScheme {
    if ctx.is_null() {
        set_last_error("Context pointer is null");
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(GoldyScheme {
        inner: Scheme::new(&(*ctx).inner),
        active_compute: None,
        active_render_pass: None,
        labels: Vec::new(),
    }))
}

/// Destroy a scheme.
///
/// # Safety
/// `scheme` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_destroy(scheme: *mut GoldyScheme) {
    if !scheme.is_null() {
        drop(Box::from_raw(scheme));
    }
}

/// Number of nodes recorded in the scheme IR.
///
/// # Safety
/// `scheme` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_len(scheme: *const GoldyScheme) -> u32 {
    if scheme.is_null() {
        return 0;
    }
    (*scheme).inner.ir_node_count() as u32
}

/// True when the next submit must re-record.
///
/// # Safety
/// `scheme` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_is_dirty(scheme: *const GoldyScheme) -> bool {
    if scheme.is_null() {
        return false;
    }
    (*scheme).inner.is_dirty()
}

/// Submission outcome counters.
///
/// # Safety
/// `scheme` and `out_stats` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_replay_stats(
    scheme: *const GoldyScheme,
    out_stats: *mut GoldyReplayStats,
) -> GoldyResult {
    if scheme.is_null() || out_stats.is_null() {
        return GoldyResult::NullPointer;
    }
    let stats = (*scheme).inner.replay_stats();
    *out_stats = GoldyReplayStats {
        records: stats.records,
        #[cfg(not(feature = "metal"))]
        resubmit_hits: stats.resubmit_hits,
        #[cfg(feature = "metal")]
        resubmit_hits: 0,
    };
    GoldyResult::Ok
}

/// Begin recording a compute dispatch node.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_compute_node_begin(
    scheme: *mut GoldyScheme,
    label: *const libc::c_char,
    pipeline: *const GoldyComputePipeline,
) -> GoldyResult {
    if scheme.is_null() || pipeline.is_null() {
        return GoldyResult::NullPointer;
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Another scheme node is already being recorded");
        return GoldyResult::InvalidArgument;
    }
    let label = match parse_label(label) {
        Ok(l) => (*scheme).intern_label(&l),
        Err(e) => return e,
    };
    (*scheme).active_compute = Some(ComputeNodeRecord::new(label, &(*pipeline).inner));
    GoldyResult::Ok
}

/// Declare a retained parcel for the active compute node.
///
/// Registers both the graph dependency and the bindless shader slot internally —
/// callers never pass resource indices.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_compute_node_with_parcel(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
    node_access: GoldyNodeAccess,
) -> GoldyResult {
    if scheme.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *scheme) {
        Ok(n) => n,
        Err(e) => return e,
    };
    match node.with_parcel(&(*parcel).inner, map_node_access(node_access)) {
        Some(_) => GoldyResult::Ok,
        None => {
            set_last_error("Parcel has no bindless slot for the shader binding");
            GoldyResult::InvalidArgument
        }
    }
}

/// Declare a retained texture for the active compute node (shader binding + dependency).
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_compute_node_with_texture(
    scheme: *mut GoldyScheme,
    texture: *const GoldyTexture,
    node_access: GoldyNodeAccess,
) -> GoldyResult {
    if scheme.is_null() || texture.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *scheme) {
        Ok(n) => n,
        Err(e) => return e,
    };
    match node.with_parcel(&(*texture).inner, map_node_access(node_access)) {
        Some(_) => GoldyResult::Ok,
        None => {
            set_last_error("Texture has no bindless slot for the shader binding");
            GoldyResult::InvalidArgument
        }
    }
}

/// Declare a buffer unit for the active compute node (shader binding + dependency).
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_compute_node_with_buffer_unit(
    scheme: *mut GoldyScheme,
    buffer: *const GoldyBuffer,
    unit: u32,
    node_access: GoldyNodeAccess,
) -> GoldyResult {
    if scheme.is_null() || buffer.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *scheme) {
        Ok(n) => n,
        Err(e) => return e,
    };
    let parcel = match buffer_unit_at(buffer, unit) {
        Ok(p) => p,
        Err(e) => return e,
    };
    match node.with_parcel(parcel, map_node_access(node_access)) {
        Some(_) => GoldyResult::Ok,
        None => {
            set_last_error("Buffer unit has no bindless slot for the shader binding");
            GoldyResult::InvalidArgument
        }
    }
}

/// Bind one field of a partitioned retained buffer to the active compute node.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_compute_node_with_field(
    scheme: *mut GoldyScheme,
    buffer: *const GoldyBuffer,
    unit: u32,
    node_access: GoldyNodeAccess,
) -> GoldyResult {
    goldy_scheme_compute_node_with_buffer_unit(scheme, buffer, unit, node_access)
}

/// Append one scalar virtual-main parameter for the active compute node.
///
/// # Safety
/// `scheme` must be valid and a compute node must be active.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_compute_node_with_param(scheme: *mut GoldyScheme, value: u32) -> GoldyResult {
    if scheme.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *scheme) {
        Ok(n) => n,
        Err(e) => return e,
    };
    node.with_param(value);
    GoldyResult::Ok
}

/// Finalize the active compute node with a direct dispatch.
///
/// # Safety
/// `scheme` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_compute_node_dispatch(
    scheme: *mut GoldyScheme,
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
) -> GoldyResult {
    if scheme.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match (*scheme).active_compute.take() {
        Some(n) => n,
        None => {
            set_last_error("No compute node is being recorded");
            return GoldyResult::InvalidArgument;
        }
    };
    node.commit_dispatch_scheme(&mut (*scheme).inner, workgroups_x, workgroups_y, workgroups_z);
    GoldyResult::Ok
}

/// Submit the scheme and return a heap-allocated per-submission [`GoldySchemeSubmission`].
///
/// Does not block. The caller owns `*out_submission` and must call
/// [`goldy_scheme_submission_destroy`]. To read bytes from a recorded withdrawal, use
/// [`crate::goldy_withdraw_transaction_claim`] then [`crate::goldy_withdraw_claim_consume`].
///
/// # Safety
/// `scheme` and `out_submission` must be valid; `*out_submission` is written on success.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_submit(
    scheme: *mut GoldyScheme,
    out_submission: *mut *mut GoldySchemeSubmission,
) -> GoldyResult {
    if scheme.is_null() || out_submission.is_null() {
        return GoldyResult::NullPointer;
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot submit while recording a compute node");
        return GoldyResult::InvalidArgument;
    }
    match (*scheme).inner.submit() {
        Ok(submission) => {
            *out_submission = Box::into_raw(Box::new(GoldySchemeSubmission { inner: submission }));
            GoldyResult::Ok
        }
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}

/// Destroy a submission token from [`goldy_scheme_submit`].
///
/// # Safety
/// `submission` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_submission_destroy(submission: *mut GoldySchemeSubmission) {
    if !submission.is_null() {
        drop(Box::from_raw(submission));
    }
}

/// True when this submission's GPU work has retired.
///
/// # Safety
/// `submission` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_submission_is_settled(submission: *const GoldySchemeSubmission) -> bool {
    if submission.is_null() {
        return false;
    }
    (*submission).inner.is_settled()
}

/// Block until the GPU work for `submission` has completed.
///
/// Prefer [`crate::goldy_withdraw_claim_consume`] when verifying compute output through a withdrawal.
///
/// # Safety
/// `submission` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_submission_wait_until_settled(
    submission: *const GoldySchemeSubmission,
) -> GoldyResult {
    if submission.is_null() {
        return GoldyResult::NullPointer;
    }
    match (*submission).inner.wait_until_settled() {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}
