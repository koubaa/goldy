use crate::device::Device;
use crate::error::{non_null, Result};
use crate::sys::{self, GoldyBuffer};
use crate::types::BufferKind;

/// A GPU buffer.
pub struct Buffer {
    ptr: *mut GoldyBuffer,
}

impl Buffer {
    pub fn from_bytes(device: &Device, data: &[u8], kind: BufferKind) -> Result<Self> {
        let ptr = non_null(unsafe {
            sys::goldy_buffer_create_with_data(
                device.as_ptr(),
                data.as_ptr(),
                data.len(),
                kind.into(),
            )
        })?;
        Ok(Self { ptr })
    }

    pub fn from_slice<T>(device: &Device, data: &[T], kind: BufferKind) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        Self::from_bytes(device, bytes, kind)
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
