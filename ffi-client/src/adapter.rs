use crate::device::Device;
use crate::error::{non_null, Result};
use crate::instance::{AdapterInfo, Instance};
use crate::types::DeviceDescriptor;

/// A physical GPU adapter.
pub struct Adapter<'a> {
    instance: &'a Instance,
    info: AdapterInfo,
}

impl<'a> Adapter<'a> {
    pub(crate) fn new(instance: &'a Instance, info: AdapterInfo) -> Self {
        Self { instance, info }
    }

    /// Immutable adapter metadata.
    pub fn get_info(&self) -> &AdapterInfo {
        &self.info
    }

    /// Create a logical [`Device`] on this adapter.
    pub fn request_device(&self, desc: &DeviceDescriptor) -> Result<Device> {
        let _ = desc;
        let ptr = non_null(unsafe {
            crate::sys::goldy_instance_create_device_for_adapter(
                self.instance.as_ptr(),
                self.info.id,
            )
        })?;
        Ok(Device { ptr })
    }
}
