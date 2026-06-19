use crate::device::Device;
use crate::error::{check, non_null, Result};
use crate::parcel::Parcel;
use crate::sys::{self, GoldyBuffer};
use crate::types::ResourceAccess;

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

    pub fn unit_resource_index(&self, unit: u32, access: ResourceAccess) -> Result<u32> {
        let idx = unsafe { sys::goldy_buffer_unit_resource_index(self.ptr, unit, access.into()) };
        if idx == u32::MAX {
            return Err(crate::error::GoldyError::from_message(
                "buffer unit resource index unavailable for requested access",
            ));
        }
        Ok(idx)
    }

    pub fn unit_read_to_cpu(&self, unit: u32, device: &Device) -> Result<Vec<u8>> {
        let size = self.unit_byte_size(unit);
        let mut output = vec![0u8; size as usize];
        check(unsafe {
            sys::goldy_buffer_unit_read_to_cpu(self.ptr, unit, device.as_ptr(), output.as_mut_ptr(), output.len())
        })?;
        Ok(output)
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
