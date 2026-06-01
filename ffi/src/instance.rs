//! FFI bindings for Instance.

use crate::device::GoldyDevice;
use crate::error::{set_last_error_from_anyhow, GoldyResult};
use crate::types::{GoldyAdapterInfo, GoldyBackendType};
use anyhow::Context;
use goldy::DeviceDescriptor;
use std::ptr;

fn device_for_adapter(
    instance: &goldy::Instance,
    adapter_id: u32,
) -> anyhow::Result<goldy::Device> {
    let adapters = instance.enumerate_adapters();
    let adapter = adapters
        .iter()
        .find(|a| a.id() == adapter_id)
        .with_context(|| format!("Adapter {adapter_id} not found"))?;
    adapter.request_device(&DeviceDescriptor::default())
}

/// Opaque handle to a Goldy Instance.
pub struct GoldyInstance {
    pub(crate) inner: goldy::Instance,
}

/// Create a new Goldy instance.
///
/// Returns a pointer to the instance, or null on failure.
/// Call `goldy_get_last_error()` to get the error message.
#[no_mangle]
pub extern "C" fn goldy_instance_create() -> *mut GoldyInstance {
    match goldy::Instance::new() {
        Ok(inner) => Box::into_raw(Box::new(GoldyInstance { inner })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Destroy an instance.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_instance_destroy(instance: *mut GoldyInstance) {
    if !instance.is_null() {
        drop(Box::from_raw(instance));
    }
}

/// Get the backend type of an instance.
///
/// # Safety
/// The instance pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_instance_backend_type(
    instance: *const GoldyInstance,
) -> GoldyBackendType {
    if instance.is_null() {
        return GoldyBackendType::Vulkan;
    }
    (*instance).inner.backend_type().into()
}

/// Get the number of available adapters.
///
/// # Safety
/// The instance pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_instance_adapter_count(instance: *const GoldyInstance) -> u32 {
    if instance.is_null() {
        return 0;
    }
    (*instance).inner.enumerate_adapters().len() as u32
}

/// Get adapter info at the given index.
///
/// # Safety
/// The instance pointer must be valid.
/// The info pointer must point to a valid GoldyAdapterInfo struct.
#[no_mangle]
pub unsafe extern "C" fn goldy_instance_get_adapter(
    instance: *const GoldyInstance,
    index: u32,
    info: *mut GoldyAdapterInfo,
) -> GoldyResult {
    if instance.is_null() || info.is_null() {
        return GoldyResult::NullPointer;
    }

    let adapters = (*instance).inner.enumerate_adapters();
    if (index as usize) >= adapters.len() {
        return GoldyResult::InvalidArgument;
    }

    *info = GoldyAdapterInfo::from_adapter(&adapters[index as usize]);
    GoldyResult::Ok
}

/// Create a device for a specific adapter ID.
///
/// Returns a pointer to the device, or null on failure.
///
/// # Safety
/// The instance pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_instance_create_device_for_adapter(
    instance: *const GoldyInstance,
    adapter_id: u32,
) -> *mut GoldyDevice {
    if instance.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Instance is null"));
        return ptr::null_mut();
    }

    match device_for_adapter(&(*instance).inner, adapter_id) {
        Ok(device) => Box::into_raw(Box::new(GoldyDevice { inner: device })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}
