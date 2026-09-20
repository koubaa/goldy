use crate::error::{non_null, Result};
use crate::instance::{AdapterInfo, Instance};
use crate::runtime::Runtime;
use crate::types::RuntimeDescriptor;

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

    /// Create a logical [`Runtime`] on this adapter.
    pub fn request_runtime(&self, desc: &RuntimeDescriptor) -> Result<Runtime> {
        let _ = desc;
        let ptr = non_null(unsafe {
            crate::sys::goldy_instance_create_runtime_for_adapter(self.instance.as_ptr(), self.info.id)
        })?;
        Ok(Runtime::from_ptr(ptr))
    }
}
