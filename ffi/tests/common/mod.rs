//! Shared helpers for `goldy-ffi` GPU integration tests.

use goldy_ffi::{
    goldy_get_last_error, goldy_instance_adapter_count, goldy_instance_create,
    goldy_instance_create_device_for_adapter, goldy_instance_get_adapter, GoldyAdapterInfo, GoldyDevice,
    GoldyDeviceType, GoldyInstance, GoldyResult,
};
use std::ffi::CStr;

pub fn last_ffi_message() -> String {
    unsafe {
        let p = goldy_get_last_error();
        if p.is_null() {
            return "(no message)".into();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

pub unsafe fn request_device(instance: *const GoldyInstance) -> *mut GoldyDevice {
    let count = goldy_instance_adapter_count(instance);
    let mut best_id: u32 = 0;
    for i in 0..count {
        let mut info = GoldyAdapterInfo {
            id: 0,
            device_type: GoldyDeviceType::Other,
            name: [0; 256],
            vendor: [0; 64],
        };
        if goldy_instance_get_adapter(instance, i, &mut info) != GoldyResult::Ok {
            continue;
        }
        if i == 0 {
            best_id = info.id;
        }
        if info.device_type == GoldyDeviceType::DiscreteGpu {
            best_id = info.id;
            break;
        }
    }
    goldy_instance_create_device_for_adapter(instance, best_id)
}

pub unsafe fn open_device() -> (*mut GoldyInstance, *mut GoldyDevice) {
    let instance = goldy_instance_create();
    assert!(!instance.is_null(), "{}", last_ffi_message());
    let device = request_device(instance);
    assert!(!device.is_null(), "{}", last_ffi_message());
    (instance, device)
}
