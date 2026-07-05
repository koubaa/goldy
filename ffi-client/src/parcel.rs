use crate::error::{non_null, Result};
use crate::sys::{self, GoldyParcel};

/// Opaque retained GPU parcel (buffer units from [`Buffer::field`]; textures use [`crate::Texture`]).
pub struct Parcel {
    ptr: *mut GoldyParcel,
}

impl Parcel {
    pub(crate) fn from_ptr(ptr: *mut GoldyParcel) -> Result<Self> {
        Ok(Self { ptr: non_null(ptr)? })
    }

    pub fn byte_size(&self) -> u64 {
        unsafe { sys::goldy_parcel_byte_size(self.ptr) }
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyParcel {
        self.ptr
    }
}

impl Drop for Parcel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_parcel_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
