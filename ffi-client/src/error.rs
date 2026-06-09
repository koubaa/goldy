use crate::sys::{self, GoldyResult};
use std::ffi::CStr;
use std::fmt;

/// Error returned when a Goldy FFI operation fails.
#[derive(Debug, Clone)]
pub struct GoldyError {
    message: String,
}

impl GoldyError {
    pub fn from_message(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn from_last_error() -> Self {
        let message = unsafe {
            let p = sys::goldy_get_last_error();
            if p.is_null() {
                "(no message)".into()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Self { message }
    }

    pub fn check(result: GoldyResult) -> std::result::Result<(), Self> {
        if result == GoldyResult::GOLDY_RESULT_OK {
            Ok(())
        } else {
            Err(Self::from_last_error())
        }
    }
}

impl fmt::Display for GoldyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for GoldyError {}

pub type Result<T> = std::result::Result<T, GoldyError>;

pub(crate) fn check(result: GoldyResult) -> Result<()> {
    GoldyError::check(result)
}

pub(crate) fn non_null<T>(ptr: *mut T) -> Result<*mut T> {
    if ptr.is_null() {
        Err(GoldyError::from_last_error())
    } else {
        Ok(ptr)
    }
}

/// Panic on FFI failure during infallible recording paths (matches C++ throw semantics).
pub(crate) fn expect_ok(result: GoldyResult) {
    if let Err(e) = GoldyError::check(result) {
        panic!("goldy ffi: {e}");
    }
}

pub(crate) fn non_null_expect<T>(ptr: *mut T) -> *mut T {
    if ptr.is_null() {
        panic!("goldy ffi: {}", GoldyError::from_last_error());
    }
    ptr
}
