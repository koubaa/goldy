use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{non_null, Result};
use crate::sys::{self, GoldyRecordBuilder};
use bytemuck::Pod;
use std::ffi::CString;

/// Builder for a retained record buffer (one backing buffer, multiple field parcels).
pub struct RecordBuilder {
    ptr: *mut GoldyRecordBuilder,
}

impl RecordBuilder {
    pub fn new() -> Result<Self> {
        let ptr = non_null(unsafe { sys::goldy_record_builder_create() })?;
        Ok(Self { ptr })
    }

    pub fn emplace_pod<T: Pod>(&mut self, data: &[T]) -> Result<u32> {
        self.emplace_named_pod(None, data)
    }

    pub fn emplace_named_pod<T: Pod>(&mut self, name: Option<&str>, data: &[T]) -> Result<u32> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        self.emplace_named_bytes(name, bytes, data.len() as u64, std::mem::size_of::<T>() as u32)
    }

    pub fn emplace_bytes(&mut self, data: &[u8], element_count: u64, element_stride: u32) -> Result<u32> {
        self.emplace_named_bytes(None, data, element_count, element_stride)
    }

    pub fn emplace_named_bytes(
        &mut self,
        name: Option<&str>,
        data: &[u8],
        element_count: u64,
        element_stride: u32,
    ) -> Result<u32> {
        let name_cstring = name
            .map(CString::new)
            .transpose()
            .map_err(|_| crate::error::GoldyError::from_message("field name contains interior null byte"))?;
        let name_ptr = name_cstring.as_ref().map_or(std::ptr::null(), |n| n.as_ptr());
        let slot = unsafe {
            sys::goldy_record_builder_emplace(
                self.ptr,
                name_ptr,
                data.as_ptr(),
                data.len(),
                element_count,
                element_stride,
            )
        };
        if slot == u32::MAX {
            return Err(crate::error::GoldyError::from_last_error());
        }
        Ok(slot)
    }

    pub fn build(self, device: &Device) -> Result<Buffer> {
        let ptr = unsafe { sys::goldy_record_builder_build(self.ptr, device.retained_pool_ptr()?) };
        std::mem::forget(self);
        Buffer::from_ptr(non_null(ptr)?)
    }
}

impl Drop for RecordBuilder {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_record_builder_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// One field for [`crate::Device::acquire_record`].
pub struct RecordField<'a> {
    pub name: Option<&'a str>,
    pub data: &'a [u8],
    pub element_count: u64,
    pub element_stride: u32,
}

impl<'a, T: Pod> From<(&'a str, &'a [T])> for RecordField<'a> {
    fn from((name, data): (&'a str, &'a [T])) -> Self {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        Self {
            name: Some(name),
            data: bytes,
            element_count: data.len() as u64,
            element_stride: std::mem::size_of::<T>() as u32,
        }
    }
}
