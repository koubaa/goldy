use crate::error::{non_null, Result};
use crate::parcel::Parcel;
use crate::sys::{self, GoldyBuffer};

/// Acquired retained GPU buffer (possibly partitioned into bindable units).
pub struct Buffer {
    ptr: *mut GoldyBuffer,
}

impl Buffer {
    pub(crate) fn from_ptr(ptr: *mut GoldyBuffer) -> Result<Self> {
        Ok(Self { ptr: non_null(ptr)? })
    }

    pub fn byte_size(&self) -> u64 {
        unsafe { sys::goldy_buffer_byte_size(self.ptr) }
    }

    pub fn unit_count(&self) -> u32 {
        unsafe { sys::goldy_buffer_unit_count(self.ptr) }
    }

    pub fn unit_byte_size(&self, unit: u32) -> u64 {
        unsafe { sys::goldy_buffer_unit_byte_size(self.ptr, unit) }
    }

    /// Borrow one bindable field unit as an owned parcel handle.
    pub fn field(&self, unit: u32) -> Result<Parcel> {
        Parcel::from_ptr(unsafe { sys::goldy_buffer_field(self.as_ptr(), unit) })
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyBuffer {
        self.ptr
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_buffer_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
