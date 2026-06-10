use crate::sys::GoldyDevice;

/// A GPU device handle.
pub struct Device {
    pub(crate) ptr: *mut GoldyDevice,
}

impl Device {
    pub(crate) fn as_ptr(&self) -> *const GoldyDevice {
        self.ptr
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
