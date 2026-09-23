//! Shared helpers for `goldy-ffi` GPU integration tests.

use goldy_ffi::{
    goldy_get_last_error, goldy_host_view_copy, goldy_host_view_destroy, goldy_host_view_len,
    goldy_instance_adapter_count, goldy_instance_create, goldy_instance_create_runtime_for_adapter,
    goldy_instance_get_adapter, goldy_scheme_submission_take, goldy_scheme_submission_take_texture, GoldyAdapterInfo,
    GoldyDeviceType, GoldyHostView, GoldyInstance, GoldyParcel, GoldyResult, GoldyRuntime, GoldySchemeSubmission,
    GoldyTexture,
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

unsafe fn host_view_copy(view: *mut GoldyHostView) -> Vec<u8> {
    assert!(!view.is_null(), "{}", last_ffi_message());
    let len = goldy_host_view_len(view) as usize;
    let mut out = vec![0u8; len];
    assert_eq!(
        goldy_host_view_copy(view, out.as_mut_ptr(), out.len()),
        GoldyResult::Ok,
        "{}",
        last_ffi_message()
    );
    goldy_host_view_destroy(view);
    out
}

/// Host-claim `parcel` from `submission` and copy bytes to a `Vec`.
pub unsafe fn take_parcel_copy(submission: *mut GoldySchemeSubmission, parcel: *const GoldyParcel) -> Vec<u8> {
    host_view_copy(goldy_scheme_submission_take(submission, parcel))
}

/// Host-claim `texture` from `submission` and copy bytes to a `Vec`.
pub unsafe fn take_texture_copy(submission: *mut GoldySchemeSubmission, texture: *const GoldyTexture) -> Vec<u8> {
    host_view_copy(goldy_scheme_submission_take_texture(submission, texture))
}

pub unsafe fn request_runtime(instance: *const GoldyInstance) -> *mut GoldyRuntime {
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
    goldy_instance_create_runtime_for_adapter(instance, best_id)
}

pub unsafe fn open_device() -> (*mut GoldyInstance, *mut GoldyRuntime) {
    let instance = goldy_instance_create();
    assert!(!instance.is_null(), "{}", last_ffi_message());
    let device = request_runtime(instance);
    assert!(!device.is_null(), "{}", last_ffi_message());
    (instance, device)
}
