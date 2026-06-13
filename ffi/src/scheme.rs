//! FFI bindings for [`goldy::Scheme`] (compute-only surface).

use crate::compute::GoldyComputePipeline;
use crate::context::GoldyContext;
use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::retained_pool::GoldyParcel;
use crate::types::{GoldyNodeAccess, GoldyResourceAccess};
use goldy::scheme::Scheme;
use goldy::task_graph::{ComputeNodeRecord, NodeAccess};
use goldy::types::ResourceAccess;
use goldy::MosaicSlot;
use std::ffi::CStr;

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
    (*scheme).inner.diagnostics().ir_node_count() as u32
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

/// Submit the scheme and block until GPU completion.
///
/// TODO: remove blocking when readback IR node exists; callers should manage sync via
/// an explicit readback node instead of implicit wait-after-submit.
///
/// # Safety
/// `scheme` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_submit(scheme: *mut GoldyScheme) -> GoldyResult {
    if scheme.is_null() {
        return GoldyResult::NullPointer;
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot submit while recording a compute node");
        return GoldyResult::InvalidArgument;
    }
    match (*scheme).inner.submit_blocking() {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}
