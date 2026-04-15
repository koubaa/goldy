//! FFI bindings for Sampler.

use crate::device::GoldyDevice;
use crate::error::set_last_error_from_anyhow;
use crate::types::GoldySamplerDesc;
use std::ptr;

/// Opaque handle to a Goldy Sampler.
pub struct GoldySampler {
    /// Held for ownership / future accessors; C API currently only creates and destroys.
    #[allow(dead_code)]
    pub(crate) inner: goldy::Sampler,
}

/// Create a new sampler with the given descriptor.
///
/// Returns a pointer to the sampler, or null on failure.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_sampler_create(
    device: *const GoldyDevice,
    desc: *const GoldySamplerDesc,
) -> *mut GoldySampler {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }

    let sampler_desc: goldy::SamplerDesc = if desc.is_null() {
        goldy::SamplerDesc::default()
    } else {
        (*desc).into()
    };

    match goldy::Sampler::new(&(*device).inner, &sampler_desc) {
        Ok(sampler) => Box::into_raw(Box::new(GoldySampler { inner: sampler })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Create a sampler with default settings.
///
/// Returns a pointer to the sampler, or null on failure.
///
/// # Safety
/// The device pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_sampler_create_default(
    device: *const GoldyDevice,
) -> *mut GoldySampler {
    goldy_sampler_create(device, ptr::null())
}

/// Destroy a sampler.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_sampler_destroy(sampler: *mut GoldySampler) {
    if !sampler.is_null() {
        drop(Box::from_raw(sampler));
    }
}
