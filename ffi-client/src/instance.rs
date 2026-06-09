use crate::adapter::Adapter;
use crate::device::Device;
use crate::error::{non_null, Result};
use crate::sys::{self, GoldyAdapterInfo, GoldyInstance};
use crate::types::{DeviceType, PowerPreference, RequestAdapterOptions};
use std::ffi::CStr;

/// GPU adapter metadata.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub id: u32,
    pub device_type: DeviceType,
    pub name: String,
    pub vendor: String,
}

/// Entry point for the Goldy GPU library.
pub struct Instance {
    ptr: *mut GoldyInstance,
}

impl Instance {
    pub fn new() -> Result<Self> {
        let ptr = non_null(unsafe { sys::goldy_instance_create() })?;
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *mut GoldyInstance {
        self.ptr
    }

    pub fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        let count = unsafe { sys::goldy_instance_adapter_count(self.ptr) };
        let mut adapters = Vec::with_capacity(count as usize);
        for i in 0..count {
            let mut info = GoldyAdapterInfo {
                id: 0,
                device_type: sys::GoldyDeviceType::GOLDY_DEVICE_TYPE_OTHER,
                name: [0; 256],
                vendor: [0; 64],
            };
            if unsafe { sys::goldy_instance_get_adapter(self.ptr, i, &mut info) } == sys::GoldyResult::GOLDY_RESULT_OK {
                adapters.push(AdapterInfo {
                    id: info.id,
                    device_type: info.device_type.into(),
                    name: cstr_field(&info.name),
                    vendor: cstr_field(&info.vendor),
                });
            }
        }
        adapters
    }

    /// Request an adapter matching the given options (wgpu-style).
    pub fn request_adapter<'a>(&'a self, opts: &RequestAdapterOptions) -> Result<Adapter<'a>> {
        let adapters = self.enumerate_adapters();
        if adapters.is_empty() {
            return Err(crate::error::GoldyError::from_message("No GPU adapters available"));
        }

        let selected = match opts.power_preference {
            PowerPreference::HighPerformance => adapters
                .iter()
                .find(|a| a.device_type == DeviceType::DiscreteGpu)
                .or_else(|| adapters.iter().find(|a| a.device_type == DeviceType::IntegratedGpu))
                .or_else(|| adapters.iter().find(|a| a.device_type == DeviceType::Other))
                .or(adapters.first()),
            PowerPreference::LowPower => adapters
                .iter()
                .find(|a| a.device_type == DeviceType::IntegratedGpu)
                .or_else(|| adapters.iter().find(|a| a.device_type == DeviceType::Cpu))
                .or(adapters.first()),
            PowerPreference::None => adapters.first(),
        }
        .expect("adapters non-empty");

        Ok(Adapter::new(self, selected.clone()))
    }

    pub fn create_device_for_adapter(&self, adapter_id: u32) -> Result<Device> {
        let ptr = non_null(unsafe { sys::goldy_instance_create_device_for_adapter(self.ptr, adapter_id) })?;
        Ok(Device { ptr })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_instance_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

fn cstr_field(buf: &[std::ffi::c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned() }
}
