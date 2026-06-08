use crate::buffer::Buffer;
use crate::error::Result;
use crate::sys::GoldyDevice;
use crate::types::BufferKind;
use bytemuck::Pod;

/// A GPU device handle.
pub struct Device {
    pub(crate) ptr: *mut GoldyDevice,
}

impl Device {
    pub(crate) fn as_ptr(&self) -> *const GoldyDevice {
        self.ptr
    }

    /// Upload a typed slice into a new GPU buffer.
    pub fn alloc_buffer_with_data<T: Pod>(
        &self,
        data: &[T],
        kind: BufferKind,
    ) -> Result<Buffer> {
        Buffer::from_slice(self, data, kind)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { crate::sys::goldy_device_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
