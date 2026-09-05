//! Catch Objective-C exceptions from Metal so libtest does not abort with
//! "Rust cannot catch foreign exceptions".
//!
//! Temporary diagnostics for the mesh-shader CI abort on macos-15.

use anyhow::Result;
use objc::runtime::Object;
use objc::{msg_send, sel, sel_impl};
use std::ffi::CStr;
use std::os::raw::c_char;

pub(super) fn catch_objc<T>(label: &str, f: impl FnOnce() -> T) -> Result<T> {
    eprintln!("[goldy-mesh] enter {label}");
    // `objc_exception::try` uses the ObjC runtime, not Rust `catch_unwind`.
    match unsafe { objc_exception::r#try(f) } {
        Ok(value) => {
            eprintln!("[goldy-mesh] leave {label}");
            Ok(value)
        }
        Err(exc) => {
            let owned = unsafe { objc::rc::StrongPtr::new(exc as *mut Object) };
            let msg = format_ns_exception(owned);
            eprintln!("[goldy-mesh] NSException in {label}: {msg}");
            anyhow::bail!("Metal NSException in {label}: {msg}");
        }
    }
}

fn format_ns_exception(exc: objc::rc::StrongPtr) -> String {
    unsafe {
        let obj: *mut Object = *exc;
        if obj.is_null() {
            return "<null NSException>".into();
        }
        let name: *mut Object = msg_send![obj, name];
        let reason: *mut Object = msg_send![obj, reason];
        format!("{}: {}", nsstring(name), nsstring(reason))
    }
}

unsafe fn nsstring(id: *mut Object) -> String {
    if id.is_null() {
        return "<null>".into();
    }
    let utf8: *const c_char = msg_send![id, UTF8String];
    if utf8.is_null() {
        return "<no UTF8String>".into();
    }
    CStr::from_ptr(utf8).to_string_lossy().into_owned()
}
