//! Host-claim FFI: `goldy_scheme_submission_take` → [`GoldyHostView`].

use crate::error::{set_last_error, GoldyResult};
use crate::retained_pool::{GoldyParcel, GoldyTexture};
use crate::scheme::GoldySchemeSubmission;
use goldy::HostView;
use std::ptr;

/// Typed host view of a parcel (`HostView<u8>`).
pub struct GoldyHostView {
    inner: HostView<u8>,
}

/// Realize a host read of `parcel` after `submission`.
///
/// Returns a heap-allocated view; destroy with [`goldy_host_view_destroy`].
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_submission_take(
    submission: *mut GoldySchemeSubmission,
    parcel: *const GoldyParcel,
) -> *mut GoldyHostView {
    if submission.is_null() || parcel.is_null() {
        set_last_error("submission or parcel pointer is null");
        return ptr::null_mut();
    }
    match (&mut (*submission).inner >> &(*parcel).inner).take::<u8>() {
        Ok(view) => Box::into_raw(Box::new(GoldyHostView { inner: view })),
        Err(e) => {
            set_last_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

/// Realize a host read of a texture parcel after `submission`.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_scheme_submission_take_texture(
    submission: *mut GoldySchemeSubmission,
    texture: *const GoldyTexture,
) -> *mut GoldyHostView {
    if submission.is_null() || texture.is_null() {
        set_last_error("submission or texture pointer is null");
        return ptr::null_mut();
    }
    match (&mut (*submission).inner >> &*(*texture).inner).take::<u8>() {
        Ok(view) => Box::into_raw(Box::new(GoldyHostView { inner: view })),
        Err(e) => {
            set_last_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

/// Byte length of a host view.
///
/// # Safety
/// `view` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_host_view_len(view: *const GoldyHostView) -> u64 {
    if view.is_null() {
        return 0;
    }
    let inner = &(*view).inner;
    inner.len() as u64
}

/// Pointer to host-view bytes (valid until [`goldy_host_view_destroy`]).
///
/// # Safety
/// `view` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_host_view_data(view: *const GoldyHostView) -> *const u8 {
    if view.is_null() {
        return ptr::null();
    }
    let inner = &(*view).inner;
    inner.as_ptr()
}

/// Copy host-view bytes into `output` (must be exactly [`goldy_host_view_len`] bytes).
///
/// # Safety
/// All pointers must be valid. `output` must point to at least `output_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_host_view_copy(
    view: *const GoldyHostView,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    if view.is_null() || output.is_null() {
        return GoldyResult::NullPointer;
    }
    let src: &[u8] = &(*view).inner;
    if output_size != src.len() {
        set_last_error(format!(
            "host view size mismatch: expected {}, got {output_size}",
            src.len()
        ));
        return GoldyResult::InvalidArgument;
    }
    std::slice::from_raw_parts_mut(output, output_size).copy_from_slice(src);
    GoldyResult::Ok
}

/// Destroy a host view (releases the host claim).
///
/// # Safety
/// `view` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_host_view_destroy(view: *mut GoldyHostView) {
    if !view.is_null() {
        drop(Box::from_raw(view));
    }
}
