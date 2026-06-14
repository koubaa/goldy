//! FFI bindings for [`goldy::Scheme`] (compute-only surface).

use crate::compute::GoldyComputePipeline;
use crate::context::GoldyContext;
use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::retained_pool::GoldyParcel;
use crate::types::{GoldyNodeAccess, GoldyResourceAccess};
use goldy::scheme::{ReadGrant, Scheme};
use goldy::task_graph::{ComputeNodeRecord, NodeAccess};
use goldy::types::ResourceAccess;
use goldy::GrantBuffer;
use goldy::MosaicSlot;
use std::ffi::CStr;

/// Opaque per-submission frame token returned by [`goldy_scheme_submit`].
///
/// Heap-allocated; destroy with [`goldy_scheme_frame_destroy`].
pub struct GoldySchemeFrame {
    pub(crate) inner: goldy::SchemeFrame,
}

/// Opaque read-easement grant handle returned by [`goldy_scheme_grant_read`].
///
/// Heap-allocated; destroy with [`goldy_read_grant_destroy`].
pub struct GoldyReadGrant {
    pub(crate) inner: ReadGrant<GrantBuffer>,
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
    labels: Vec<String>,
}

impl GoldyScheme {
    fn has_active_recorder(&self) -> bool {
        self.active_compute.is_some()
    }

    fn intern_label(&mut self, label: &str) -> &'static str {
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
    }
}

fn map_resource_access(access: GoldyResourceAccess) -> ResourceAccess {
    access.into()
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
pub unsafe extern "C" fn goldy_scheme_compute_node_declare_parcel(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
    node_access: GoldyNodeAccess,
    resource_access: GoldyResourceAccess,
) -> GoldyResult {
    if scheme.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *scheme) {
        Ok(n) => n,
        Err(e) => return e,
    };
    let res_access = map_resource_access(resource_access);
    match node.declare_parcel(&(*parcel).inner, map_node_access(node_access), res_access) {
        Some(_) => GoldyResult::Ok,
        None => {
            set_last_error("Parcel has no resource index for the requested access");
            GoldyResult::InvalidArgument
        }
    }
}

/// Declare a mosaic sub-view for the active compute node.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_compute_node_declare_parcel_view(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
    slot: u32,
    node_access: GoldyNodeAccess,
    resource_access: GoldyResourceAccess,
) -> GoldyResult {
    if scheme.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *scheme) {
        Ok(n) => n,
        Err(e) => return e,
    };
    let res_access = map_resource_access(resource_access);
    let mosaic_slot = MosaicSlot(slot);
    match node.declare_parcel_view(&(*parcel).inner, mosaic_slot, map_node_access(node_access), res_access) {
        Some(_) => GoldyResult::Ok,
        None => {
            set_last_error("Mosaic view has no resource index for the requested access");
            GoldyResult::InvalidArgument
        }
    }
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

/// Submit the scheme and return a heap-allocated per-submission [`GoldySchemeFrame`].
///
/// Does not block. The caller owns `*out_frame` and must call
/// [`goldy_scheme_frame_destroy`]. To read bytes from a recorded grant, use
/// [`goldy_read_grant_read`] with a [`GoldyReadGrant`] from [`goldy_scheme_grant_read`].
///
/// # Safety
/// `scheme` and `out_frame` must be valid; `*out_frame` is written on success.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_submit(
    scheme: *mut GoldyScheme,
    out_frame: *mut *mut GoldySchemeFrame,
) -> GoldyResult {
    if scheme.is_null() || out_frame.is_null() {
        return GoldyResult::NullPointer;
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot submit while recording a compute node");
        return GoldyResult::InvalidArgument;
    }
    match (*scheme).inner.submit() {
        Ok(frame) => {
            *out_frame = Box::into_raw(Box::new(GoldySchemeFrame { inner: frame }));
            GoldyResult::Ok
        }
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}

/// Destroy a frame token from [`goldy_scheme_submit`].
///
/// # Safety
/// `frame` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_frame_destroy(frame: *mut GoldySchemeFrame) {
    if !frame.is_null() {
        drop(Box::from_raw(frame));
    }
}

/// Timeline value for this submission (for debugging only).
///
/// # Safety
/// `frame` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_frame_timeline_value(frame: *const GoldySchemeFrame) -> u64 {
    if frame.is_null() {
        return 0;
    }
    (*frame).inner.timeline_value()
}

/// Block until the GPU work for `frame` has completed.
///
/// Prefer [`goldy_read_grant_read`] when verifying compute output through a grant.
///
/// # Safety
/// `ctx` and `frame` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_frame_wait(
    ctx: *const GoldyContext,
    frame: *const GoldySchemeFrame,
) -> GoldyResult {
    if ctx.is_null() || frame.is_null() {
        return GoldyResult::NullPointer;
    }
    match (*frame).inner.wait(&(*ctx).inner) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}

/// Record a read-easement grant over a buffer parcel (once per scheme).
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
        Ok(grant) => Box::into_raw(Box::new(GoldyReadGrant { inner: grant })),
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

/// Read bytes for the `(grant × frame)` cell into `output`.
///
/// Blocks until this submission's GPU work (dispatch + grant staging copy) completes.
/// Each frame may be read at most once per grant. Drop the frame when done if you
/// rely on staging-buffer reuse.
///
/// # Safety
/// All pointers must be valid. `output` must point to at least `output_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_read_grant_read(
    grant: *const GoldyReadGrant,
    frame: *const GoldySchemeFrame,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    if grant.is_null() || frame.is_null() || output.is_null() {
        return GoldyResult::NullPointer;
    }
    let out = std::slice::from_raw_parts_mut(output, output_size);
    match (*grant).inner.read(&(*frame).inner) {
        Ok(loan) => {
            if loan.len() != output_size {
                set_last_error(format!(
                    "grant readback size mismatch: expected {output_size}, got {}",
                    loan.len()
                ));
                return GoldyResult::InvalidArgument;
            }
            out.copy_from_slice(&loan);
            GoldyResult::Ok
        }
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}
