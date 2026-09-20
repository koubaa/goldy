//! FFI bindings for Runtime.

use std::ffi::{c_char, CStr};

/// Opaque handle to a Goldy Runtime.
pub struct GoldyRuntime {
    pub(crate) inner: goldy::Runtime,
}

/// Destroy a device.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_runtime_destroy(device: *mut GoldyRuntime) {
    if !device.is_null() {
        drop(Box::from_raw(device));
    }
}

/// Get the adapter ID this device was created on.
///
/// # Safety
/// The device pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_runtime_adapter_id(device: *const GoldyRuntime) -> u32 {
    if device.is_null() {
        return 0;
    }
    (*device).inner.adapter_id()
}

/// Check if the device is still valid.
///
/// # Safety
/// The device pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_runtime_is_valid(device: *const GoldyRuntime) -> bool {
    if device.is_null() {
        return false;
    }
    (*device).inner.is_valid()
}

/// Check if a shader library is registered.
///
/// # Safety
/// The device pointer and name must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_runtime_has_library(device: *const GoldyRuntime, name: *const c_char) -> bool {
    if device.is_null() || name.is_null() {
        return false;
    }

    let name = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    (*device).inner.has_library(name)
}
