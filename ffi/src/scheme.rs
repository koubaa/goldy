//! FFI bindings for [`goldy::Scheme`].

#[path = "scheme_graphics.rs"]
mod scheme_graphics;

pub use scheme_graphics::*;

use crate::compute::GoldyComputePipeline;
use crate::context::GoldyContext;
use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::retained_pool::{buffer_unit_at, GoldyBuffer, GoldyParcel, GoldyTexture};
use crate::types::GoldyNodeAccess;
use goldy::scheme::{ReadGrant, Scheme};
use goldy::task_graph::{ComputeNodeRecord, NodeAccess, RenderPassRecord};
use goldy::{Grant, GrantBuffer, GrantTexture};
use std::ffi::CStr;

/// Opaque per-submission token returned by [`goldy_scheme_submit`].
///
/// Heap-allocated; destroy with [`goldy_scheme_submission_destroy`].
pub struct GoldySchemeSubmission {
    pub(crate) inner: goldy::Submission,
}

/// Opaque read-easement grant handle returned by [`goldy_scheme_grant_read`].
///
/// Heap-allocated; destroy with [`goldy_read_grant_destroy`].
pub struct GoldyReadGrant {
    pub(crate) inner: ReadGrantInner,
}

pub(crate) enum ReadGrantInner {
    Buffer(ReadGrant<GrantBuffer>),
    Texture(ReadGrant<GrantTexture>),
}

impl ReadGrantInner {
    fn byte_size(&self) -> u64 {
        match self {
            ReadGrantInner::Buffer(g) => g.byte_size(),
            ReadGrantInner::Texture(g) => g.byte_size(),
        }
    }

    fn consume_copy(&self, submission: &goldy::Submission, output: &mut [u8]) -> Result<(), goldy::GoldyError> {
        match self {
            ReadGrantInner::Buffer(g) => {
                let loan = g.consume(submission)?;
                output.copy_from_slice(&loan);
            }
            ReadGrantInner::Texture(g) => {
                let loan = g.consume(submission)?;
                output.copy_from_slice(&loan);
            }
        }
        Ok(())
    }
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
/// [`goldy_scheme_submission_destroy`]. To read bytes from a recorded grant, use
/// [`goldy_read_grant_consume`] with a [`GoldyReadGrant`] from [`goldy_scheme_grant_read`].
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

/// Timeline value for this submission (for debugging only).
///
/// # Safety
/// `submission` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_submission_timeline_value(submission: *const GoldySchemeSubmission) -> u64 {
    if submission.is_null() {
        return 0;
    }
    (*submission).inner.timeline_value()
}

/// Block until the GPU work for `submission` has completed.
///
/// Prefer [`goldy_read_grant_consume`] when verifying compute output through a grant.
///
/// # Safety
/// `ctx` and `submission` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_submission_wait(
    ctx: *const GoldyContext,
    submission: *const GoldySchemeSubmission,
) -> GoldyResult {
    if ctx.is_null() || submission.is_null() {
        return GoldyResult::NullPointer;
    }
    match (*submission).inner.wait(&(*ctx).inner) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}

/// Record a read-easement grant over a **buffer** parcel (once per scheme).
///
/// For texture parcels use [`goldy_scheme_grant_read_texture`].
///
/// Returns a heap-allocated [`GoldyReadGrant`]; destroy with [`goldy_read_grant_destroy`].
/// Call after the producing dispatch node(s). Marks the scheme dirty.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_grant_read(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
) -> *mut GoldyReadGrant {
    if scheme.is_null() || parcel.is_null() {
        set_last_error("Scheme or parcel pointer is null");
        return std::ptr::null_mut();
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot grant_read while recording a compute node");
        return std::ptr::null_mut();
    }
    match (*scheme).inner.grant_read(&(*parcel).inner) {
        Ok(grant) => Box::into_raw(Box::new(GoldyReadGrant {
            inner: ReadGrantInner::Buffer(grant),
        })),
        Err(e) => {
            set_last_error(format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

/// Destroy a read grant from [`goldy_scheme_grant_read`].
///
/// # Safety
/// `grant` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_read_grant_destroy(grant: *mut GoldyReadGrant) {
    if !grant.is_null() {
        drop(Box::from_raw(grant));
    }
}

/// Logical byte size of readable data for this grant.
///
/// # Safety
/// `grant` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_read_grant_byte_size(grant: *const GoldyReadGrant) -> u64 {
    if grant.is_null() {
        return 0;
    }
    (*grant).inner.byte_size()
}

/// Consume bytes for the `(grant × submission)` cell into `output`.
///
/// Blocks until this submission's GPU work (dispatch + grant staging copy) completes.
/// Each submission may be consumed at most once per grant. Drop the submission when done if you
/// rely on staging-buffer reuse.
///
/// # Safety
/// All pointers must be valid. `output` must point to at least `output_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_read_grant_consume(
    grant: *const GoldyReadGrant,
    submission: *const GoldySchemeSubmission,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    if grant.is_null() || submission.is_null() || output.is_null() {
        return GoldyResult::NullPointer;
    }
    let out = std::slice::from_raw_parts_mut(output, output_size);
    if output_size as u64 != (*grant).inner.byte_size() {
        set_last_error(format!(
            "grant readback size mismatch: expected {output_size}, grant byte size is {}",
            (*grant).inner.byte_size()
        ));
        return GoldyResult::InvalidArgument;
    }
    match (*grant).inner.consume_copy(&(*submission).inner, out) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}
