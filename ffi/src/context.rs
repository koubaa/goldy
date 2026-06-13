//! FFI bindings for [`goldy::Context`].

use crate::device::GoldyDevice;
use crate::error::{set_last_error, GoldyResult};

/// Opaque handle to a Goldy submission context.
pub struct GoldyContext {
    pub(crate) inner: goldy::Context,
}

/// Create a context bound to `device`.
///
/// A context is the submission lifetime anchor for retained [`crate::scheme::GoldyScheme`]
/// instances on the same device.
///
/// # Safety
/// `device` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_context_create(device: *const GoldyDevice) -> *mut GoldyContext {
    if device.is_null() {
        set_last_error("Device pointer is null");
        return std::ptr::null_mut();
    }
    match (*device).inner.create_context() {
        Ok(ctx) => Box::into_raw(Box::new(GoldyContext { inner: ctx })),
        Err(e) => {
            set_last_error(format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

/// Destroy a context.
///
/// # Safety
/// `ctx` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_context_destroy(ctx: *mut GoldyContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

/// Block until the GPU has completed all work scheduled up to `timeline_value`.
///
/// # Safety
/// `ctx` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_context_wait_until(
    ctx: *const GoldyContext,
    timeline_value: u64,
) -> GoldyResult {
    if ctx.is_null() {
        return GoldyResult::NullPointer;
    }
    match (*ctx).inner.wait_until(timeline_value) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}

/// Smoke-test helper: returns [`GoldyResult::Ok`] when `ctx` is non-null.
///
/// # Safety
/// `ctx` must be valid when non-null.
#[no_mangle]
pub unsafe extern "C" fn goldy_context_is_valid(ctx: *const GoldyContext) -> GoldyResult {
    if ctx.is_null() {
        GoldyResult::NullPointer
    } else {
        GoldyResult::Ok
    }
}
