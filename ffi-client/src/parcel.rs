use crate::device::Device;
use crate::error::{check, non_null, Result};
use crate::sys::{self, GoldyParcel};
use crate::types::ResourceAccess;

/// Opaque retained GPU parcel (buffer or texture).
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

    pub fn resource_index(&self, access: ResourceAccess) -> Result<u32> {
        let idx = unsafe { sys::goldy_parcel_resource_index(self.ptr, access.into()) };
        if idx == u32::MAX {
            return Err(crate::error::GoldyError::from_message(
                "parcel resource index unavailable for requested access",
            ));
        }
        Ok(idx)
    }

    pub fn read_to_cpu(&self, device: &Device) -> Result<Vec<u8>> {
        let mut output = vec![0u8; self.byte_size() as usize];
        check(unsafe { sys::goldy_parcel_read_to_cpu(self.ptr, device.as_ptr(), output.as_mut_ptr(), output.len()) })?;
        Ok(output)
    }

    pub fn mosaic_view_resource_index(
        &self,
        slot: crate::retained_pool::MosaicSlot,
        access: ResourceAccess,
    ) -> Result<u32> {
        let idx = unsafe { sys::goldy_parcel_mosaic_view_resource_index(self.ptr, slot.0, access.into()) };
        if idx == u32::MAX {
            return Err(crate::error::GoldyError::from_message(
                "mosaic view resource index unavailable for requested access",
            ));
        }
        Ok(idx)
    }

    pub fn mosaic_view_read_to_cpu(&self, slot: crate::retained_pool::MosaicSlot, device: &Device) -> Result<Vec<u8>> {
        let size = unsafe { sys::goldy_parcel_mosaic_view_size(self.ptr, slot.0) };
        let mut output = vec![0u8; size as usize];
        check(unsafe {
            sys::goldy_parcel_mosaic_view_read_to_cpu(
                self.ptr,
                slot.0,
                device.as_ptr(),
                output.as_mut_ptr(),
                output.len(),
            )
        })?;
        Ok(output)
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
