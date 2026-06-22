use crate::error::{non_null, Result};
use crate::sys::{self, GoldyTexture};

/// Acquired retained GPU texture.
pub struct Texture {
    ptr: *mut GoldyTexture,
}

impl Texture {
    pub(crate) fn from_ptr(ptr: *mut GoldyTexture) -> Result<Self> {
        Ok(Self { ptr: non_null(ptr)? })
    }

    pub fn byte_size(&self) -> u64 {
        unsafe { sys::goldy_texture_byte_size(self.ptr) }
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyTexture {
        self.ptr
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_texture_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
