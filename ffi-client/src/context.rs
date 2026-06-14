use crate::device::Device;
use crate::error::{non_null_expect, Result};
use crate::sys::{self, GoldyContext};

/// Submission context for retained [`crate::Scheme`] instances.
pub struct Context {
    ptr: *mut GoldyContext,
}

impl Context {
    pub fn new(device: &Device) -> Result<Self> {
        let ptr = non_null_expect(unsafe { sys::goldy_context_create(device.as_ptr()) });
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyContext {
        self.ptr
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_context_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
