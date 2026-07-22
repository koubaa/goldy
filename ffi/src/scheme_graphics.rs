//! Scheme render-pass and easement FFI (`copy_to_texture`, `grant_read_texture`).

use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::pipeline::GoldyRenderPipeline;
use crate::retained_pool::{buffer_unit_at, GoldyBuffer, GoldyParcel, GoldyTexture};
use crate::scheme::{GoldyReadGrant, GoldyScheme, ReadGrantInner};
use crate::types::{
    GoldyColor, GoldyDepthFormat, GoldyIndexFormat, GoldyNodeAccess, GoldyTargetLoad, GoldyTextureFormat,
};
use goldy::scheme::{Lease, LeaseRenderTarget};
use goldy::task_graph::{NodeAccess, RenderPassRecord};
use goldy::types::TargetLoad;
use std::ffi::CStr;

/// Opaque handle to a scheme-held render-target lease.
pub struct GoldySchemeRenderTargetLease {
    pub(crate) lease: Lease<LeaseRenderTarget>,
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

fn map_target_load(load: GoldyTargetLoad, clear_color: GoldyColor) -> TargetLoad {
    match load {
        GoldyTargetLoad::Load => TargetLoad::Load,
        GoldyTargetLoad::Clear => TargetLoad::Clear(clear_color.into()),
        GoldyTargetLoad::Discard => TargetLoad::Discard,
    }
}

fn active_render_pass_mut(scheme: &mut GoldyScheme) -> Result<&mut RenderPassRecord, GoldyResult> {
    scheme.active_render_pass.as_mut().ok_or_else(|| {
        set_last_error("No render pass is being recorded; call goldy_scheme_render_pass_begin first");
        GoldyResult::InvalidArgument
    })
}

impl GoldyScheme {
    pub(crate) fn has_active_recorder(&self) -> bool {
        self.active_compute.is_some() || self.active_render_pass.is_some()
    }
}

/// Declare a render-target lease on `scheme` (N=1 backing).
///
/// Returns a heap-allocated lease handle; destroy with [`goldy_scheme_render_target_lease_destroy`].
/// The lease is valid until the scheme is destroyed.
///
/// # Safety
/// `scheme` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_lease_render_target(
    scheme: *mut GoldyScheme,
    width: u32,
    height: u32,
    format: GoldyTextureFormat,
    has_depth: bool,
    depth_format: GoldyDepthFormat,
) -> *mut GoldySchemeRenderTargetLease {
    if scheme.is_null() {
        set_last_error("Scheme pointer is null");
        return std::ptr::null_mut();
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot lease_render_target while recording a node");
        return std::ptr::null_mut();
    }
    let depth = if has_depth { Some(depth_format.into()) } else { None };
    match (*scheme).inner.lease_render_target(width, height, format.into(), depth) {
        Ok(lease) => Box::into_raw(Box::new(GoldySchemeRenderTargetLease { lease })),
        Err(e) => {
            set_last_error(format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

/// Destroy a render-target lease handle.
///
/// Does not remove the lease from the scheme; the backing remains until the scheme is dropped.
///
/// # Safety
/// `lease` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_target_lease_destroy(lease: *mut GoldySchemeRenderTargetLease) {
    if !lease.is_null() {
        drop(Box::from_raw(lease));
    }
}

/// Begin recording an offscreen render pass on a scheme-held lease.
///
/// `clear_color` is used only when `load == GoldyTargetLoad::Clear`.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_begin(
    scheme: *mut GoldyScheme,
    label: *const libc::c_char,
    lease: *const GoldySchemeRenderTargetLease,
    load: GoldyTargetLoad,
    clear_color: GoldyColor,
) -> GoldyResult {
    if scheme.is_null() || lease.is_null() {
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
    (*scheme).active_render_pass = Some(RenderPassRecord::new_for_scheme_lease(
        label,
        &(*scheme).inner,
        &(*lease).lease,
        map_target_load(load, clear_color),
    ));
    GoldyResult::Ok
}

/// Declare a buffer unit dependency for the active render pass.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_with_buffer_unit(
    scheme: *mut GoldyScheme,
    buffer: *const GoldyBuffer,
    unit: u32,
    access: GoldyNodeAccess,
) -> GoldyResult {
    if scheme.is_null() || buffer.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_render_pass_mut(&mut *scheme) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let parcel = match buffer_unit_at(buffer, unit) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.with_parcel(parcel, map_node_access(access));
    GoldyResult::Ok
}

/// Bind one field of a partitioned retained buffer to the active render pass.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_with_field(
    scheme: *mut GoldyScheme,
    buffer: *const GoldyBuffer,
    unit: u32,
    access: GoldyNodeAccess,
) -> GoldyResult {
    goldy_scheme_render_pass_with_buffer_unit(scheme, buffer, unit, access)
}

/// Declare a graph dependency on a retained parcel for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_with_parcel(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
    access: GoldyNodeAccess,
) -> GoldyResult {
    if scheme.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_render_pass_mut(&mut *scheme) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.with_parcel(&(*parcel).inner, map_node_access(access));
    GoldyResult::Ok
}

/// Clear the depth attachment in the active render pass.
///
/// # Safety
/// `scheme` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_clear_depth(scheme: *mut GoldyScheme, depth: f32) -> GoldyResult {
    if scheme.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_render_pass_mut(&mut *scheme) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.clear_depth(depth);
    GoldyResult::Ok
}

/// Set the render pipeline for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_set_pipeline(
    scheme: *mut GoldyScheme,
    pipeline: *const GoldyRenderPipeline,
) -> GoldyResult {
    if scheme.is_null() || pipeline.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_render_pass_mut(&mut *scheme) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.set_pipeline(&(*pipeline).inner);
    GoldyResult::Ok
}

/// Bind a vertex buffer slot from a retained buffer parcel.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_set_vertex_buffer_parcel(
    scheme: *mut GoldyScheme,
    slot: u32,
    parcel: *const GoldyParcel,
) -> GoldyResult {
    if scheme.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_render_pass_mut(&mut *scheme) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.set_vertex_buffer(slot, &(*parcel).inner);
    GoldyResult::Ok
}

/// Bind an index buffer parcel for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_set_index_buffer(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
    format: GoldyIndexFormat,
) -> GoldyResult {
    if scheme.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_render_pass_mut(&mut *scheme) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.set_index_buffer(&(*parcel).inner, format.into());
    GoldyResult::Ok
}

/// Draw non-indexed primitives in the active render pass.
///
/// # Safety
/// `scheme` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_draw(
    scheme: *mut GoldyScheme,
    first_vertex: u32,
    vertex_count: u32,
    first_instance: u32,
    instance_count: u32,
) -> GoldyResult {
    if scheme.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_render_pass_mut(&mut *scheme) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.draw(first_vertex, vertex_count, first_instance, instance_count);
    GoldyResult::Ok
}

/// Draw indexed primitives in the active render pass.
///
/// # Safety
/// `scheme` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_draw_indexed(
    scheme: *mut GoldyScheme,
    first_index: u32,
    index_count: u32,
    base_vertex: i32,
    first_instance: u32,
    instance_count: u32,
) -> GoldyResult {
    if scheme.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_render_pass_mut(&mut *scheme) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.draw_indexed(first_index, index_count, base_vertex, first_instance, instance_count);
    GoldyResult::Ok
}

/// Draw a fullscreen triangle in the active render pass.
///
/// # Safety
/// `scheme` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_draw_fullscreen(scheme: *mut GoldyScheme) -> GoldyResult {
    if scheme.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_render_pass_mut(&mut *scheme) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.draw_fullscreen();
    GoldyResult::Ok
}

/// Finalize the active render pass and append it to the scheme.
///
/// # Safety
/// `scheme` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_render_pass_finish(scheme: *mut GoldyScheme) -> GoldyResult {
    if scheme.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match (*scheme).active_render_pass.take() {
        Some(p) => p,
        None => {
            set_last_error("No render pass is being recorded");
            return GoldyResult::InvalidArgument;
        }
    };
    pass.commit_scheme(&mut (*scheme).inner);
    GoldyResult::Ok
}

/// Copy a scheme-held render target into a texture parcel (for grant readback).
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_copy_to_texture(
    scheme: *mut GoldyScheme,
    src_lease: *const GoldySchemeRenderTargetLease,
    dst_texture: *const GoldyTexture,
) -> GoldyResult {
    if scheme.is_null() || src_lease.is_null() || dst_texture.is_null() {
        return GoldyResult::NullPointer;
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot copy_to_texture while recording a node");
        return GoldyResult::InvalidArgument;
    }
    match (*scheme)
        .inner
        .copy_to_texture(&(*src_lease).lease, &*(*dst_texture).inner)
    {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}

///
/// Like [`goldy_scheme_grant_read`] but requires a texture parcel with [`TextureFlags::COPY_SRC`].
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_grant_read_texture(
    scheme: *mut GoldyScheme,
    texture: *const GoldyTexture,
) -> *mut GoldyReadGrant {
    if scheme.is_null() || texture.is_null() {
        set_last_error("Scheme or texture pointer is null");
        return std::ptr::null_mut();
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot grant_read_texture while recording a node");
        return std::ptr::null_mut();
    }
    match (*scheme).inner.grant_read_texture(&*(*texture).inner) {
        Ok(grant) => Box::into_raw(Box::new(GoldyReadGrant {
            inner: ReadGrantInner::Texture(grant),
        })),
        Err(e) => {
            set_last_error(format!("{e}"));
            std::ptr::null_mut()
        }
    }
}
