use crate::device::Device;
use crate::error::{non_null, Result};
use crate::parcel::Parcel;
use crate::sys::{self, GoldyRetainedPool};
use crate::types::BufferKind;
use bytemuck::Pod;

/// Deed-governed pool for retained GPU parcels.
pub struct RetainedPool {
    ptr: *mut GoldyRetainedPool,
}

impl RetainedPool {
    pub fn new(device: &Device) -> Result<Self> {
        let ptr = non_null(unsafe { sys::goldy_retained_pool_create(device.as_ptr()) })?;
        Ok(Self { ptr })
    }

    pub fn acquire_buffer_with_data<T: Pod>(&mut self, data: &[T], kind: BufferKind) -> Result<Parcel> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        self.acquire_buffer_bytes(bytes, kind, std::mem::size_of::<T>() as u32)
    }

    pub fn acquire_buffer_bytes(&mut self, data: &[u8], kind: BufferKind, element_stride: u32) -> Result<Parcel> {
        let ptr = non_null(unsafe {
            sys::goldy_retained_pool_acquire_buffer(
                self.ptr,
                data.len() as u64,
                kind.into(),
                element_stride,
                data.as_ptr(),
                data.len(),
            )
        })?;
        Parcel::from_ptr(ptr)
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut GoldyRetainedPool {
        self.ptr
    }
}

impl Drop for RetainedPool {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_retained_pool_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
