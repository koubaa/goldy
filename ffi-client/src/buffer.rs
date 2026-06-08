use crate::device::Device;
use crate::error::{check, non_null, Result};
use crate::sys::{self, GoldyBuffer};
use crate::types::{BufferKind, ResourceAccess};
use bytemuck::Pod;

/// A GPU buffer.
pub struct Buffer {
    ptr: *mut GoldyBuffer,
}

impl Buffer {
    pub fn from_bytes(device: &Device, data: &[u8], kind: BufferKind) -> Result<Self> {
        let ptr = non_null(unsafe {
            sys::goldy_buffer_create_with_data(device.as_ptr(), data.as_ptr(), data.len(), kind.into())
        })?;
        Ok(Self { ptr })
    }

    pub fn from_slice<T: Pod>(device: &Device, data: &[T], kind: BufferKind) -> Result<Self> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        let ptr = non_null(unsafe {
            sys::goldy_buffer_create_with_data_stride(
                device.as_ptr(),
                bytes.as_ptr(),
                bytes.len(),
                kind.into(),
                std::mem::size_of::<T>() as u32,
            )
        })?;
        Ok(Self { ptr })
    }

    pub fn empty(device: &Device, size: u64, kind: BufferKind) -> Result<Self> {
        let ptr = non_null(unsafe { sys::goldy_buffer_create(device.as_ptr(), size, kind.into()) })?;
        Ok(Self { ptr })
    }

    pub fn size(&self) -> u64 {
        unsafe { sys::goldy_buffer_size(self.ptr) }
    }

    pub fn resource_index(&self, access: ResourceAccess) -> Result<u32> {
        let idx = unsafe { sys::goldy_buffer_resource_index(self.ptr, access.into()) };
        if idx == u32::MAX {
            return Err(crate::error::GoldyError::from_message(
                "buffer resource index unavailable for requested access",
            ));
        }
        Ok(idx)
    }

    pub fn read_to_cpu(&self, device: &Device) -> Result<Vec<u8>> {
        let mut output = vec![0u8; self.size() as usize];
        check(unsafe { sys::goldy_buffer_read_to_cpu(self.ptr, device.as_ptr(), output.as_mut_ptr(), output.len()) })?;
        Ok(output)
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
