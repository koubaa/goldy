//! Error handling for FFI.
//!
//! Uses a thread-local error buffer to store the last error message.

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::ptr;

/// Result codes for FFI functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyResult {
    /// Operation succeeded.
    Ok = 0,
    /// Invalid argument provided.
    InvalidArgument = 1,
    /// Null pointer provided.
    NullPointer = 2,
    /// GPU operation failed.
    GpuError = 3,
    /// Shader compilation failed.
    ShaderError = 4,
    /// Resource creation failed.
    ResourceError = 5,
    /// Internal error.
    InternalError = 6,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store an error message in the thread-local buffer.
pub(crate) fn set_last_error(msg: impl Into<String>) {
    let msg = msg.into();
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

/// Store an error from an anyhow::Error.
pub(crate) fn set_last_error_from_anyhow(err: &anyhow::Error) {
    set_last_error(format!("{:#}", err));
}

/// Get the last error message.
///
/// Returns a pointer to a null-terminated string. The pointer is valid until
/// the next FFI call on the same thread.
///
/// Returns null if no error has occurred.
#[no_mangle]
pub extern "C" fn goldy_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        match e.borrow().as_ref() {
            Some(s) => s.as_ptr(),
            None => ptr::null(),
        }
    })
}

/// Clear the last error message.
#[no_mangle]
pub extern "C" fn goldy_clear_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

