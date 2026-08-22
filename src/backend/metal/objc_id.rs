//! Thin objc `id` alias so Metal surface code compiles on macOS (cocoa) and iOS.

#![allow(deprecated)]
#![allow(non_upper_case_globals)]

#[cfg(target_os = "macos")]
pub use cocoa::base::{id, nil, NO, YES};

#[cfg(target_os = "ios")]
pub type id = *mut objc::runtime::Object;

#[cfg(target_os = "ios")]
pub const nil: id = std::ptr::null_mut();

#[cfg(target_os = "ios")]
pub const YES: objc::runtime::BOOL = 1;

#[cfg(target_os = "ios")]
pub const NO: objc::runtime::BOOL = 0;
