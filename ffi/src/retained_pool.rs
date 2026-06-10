//! FFI bindings for [`goldy::RetainedPool`] and [`goldy::Parcel`].

use crate::device::GoldyDevice;
use crate::error::{set_last_error_from_anyhow, GoldyResult};
use crate::types::{GoldyBufferKind, GoldyResourceAccess, GoldyTextureFlags, GoldyTextureFormat, GoldyTextureKind};
use std::ptr;
use std::slice;
use std::sync::Arc;

/// Opaque handle to a Goldy retained allocation pool.
pub struct GoldyRetainedPool {
    pub(crate) inner: goldy::RetainedPool,
}

/// Opaque handle to a retained [`goldy::Parcel`].
pub struct GoldyParcel {
    pub(crate) inner: goldy::Parcel,
}

/// Create a retained pool tied to `device`.
///
/// # Safety
/// The device pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_retained_pool_create(device: *const GoldyDevice) -> *mut GoldyRetainedPool {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }

    let pool = goldy::RetainedPool::new(Arc::new((*device).inner.clone()));
    Box::into_raw(Box::new(GoldyRetainedPool { inner: pool }))
}

/// Destroy a retained pool.
///
/// Parcels acquired from this pool remain valid until destroyed separately.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_retained_pool_destroy(pool: *mut GoldyRetainedPool) {
    if !pool.is_null() {
        drop(Box::from_raw(pool));
    }
}

/// Acquire a retained buffer parcel.
///
/// `element_stride` of `0` selects stride `1` (raw bytes). Pass `data == null` with
/// `data_size == 0` for an uninitialized buffer.
///
/// Returns a heap-allocated parcel handle, or null on failure.
///
/// # Safety
/// `pool` and `device` must be valid. `data` must point to at least `data_size` bytes when non-null.
#[no_mangle]
pub unsafe extern "C" fn goldy_retained_pool_acquire_buffer(
    pool: *mut GoldyRetainedPool,
    size: u64,
    access: GoldyBufferKind,
    element_stride: u32,
    data: *const u8,
    data_size: usize,
) -> *mut GoldyParcel {
    if pool.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("RetainedPool is null"));
        return ptr::null_mut();
    }
    if !data.is_null() && data_size == 0 {
        set_last_error_from_anyhow(&anyhow::anyhow!("data_size is zero with non-null data"));
        return ptr::null_mut();
    }
    if data.is_null() && data_size > 0 {
        set_last_error_from_anyhow(&anyhow::anyhow!("data is null"));
        return ptr::null_mut();
    }

    let init = if data_size > 0 {
        Some(slice::from_raw_parts(data, data_size))
    } else {
        None
    };
    let stride = if element_stride == 0 {
        None
    } else {
        Some(element_stride)
    };

    match (*pool)
        .inner
        .acquire_buffer(size, access.into(), stride, goldy::BufferFlags::empty(), init)
    {
        Ok(parcel) => Box::into_raw(Box::new(GoldyParcel { inner: parcel })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Acquire a retained texture parcel with optional initial pixel data.
///
/// `data` may be null when `data_size == 0` (uninitialized texture).
///
/// # Safety
/// `pool` must be valid. `data` must point to at least `data_size` bytes when non-null.
#[no_mangle]
pub unsafe extern "C" fn goldy_retained_pool_acquire_texture(
    pool: *mut GoldyRetainedPool,
    width: u32,
    height: u32,
    format: GoldyTextureFormat,
    access: GoldyTextureKind,
    flags: GoldyTextureFlags,
    data: *const u8,
    data_size: usize,
) -> *mut GoldyParcel {
    if pool.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("RetainedPool is null"));
        return ptr::null_mut();
    }
    if !data.is_null() && data_size == 0 {
        set_last_error_from_anyhow(&anyhow::anyhow!("data_size is zero with non-null data"));
        return ptr::null_mut();
    }
    if data.is_null() && data_size > 0 {
        set_last_error_from_anyhow(&anyhow::anyhow!("data is null"));
        return ptr::null_mut();
    }

    let init = if data_size > 0 {
        Some(slice::from_raw_parts(data, data_size))
    } else {
        None
    };

    match (*pool)
        .inner
        .acquire_texture(width, height, format.into(), access.into(), flags.into(), init)
    {
        Ok(parcel) => Box::into_raw(Box::new(GoldyParcel { inner: parcel })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Destroy a retained parcel.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_parcel_destroy(parcel: *mut GoldyParcel) {
    if !parcel.is_null() {
        drop(Box::from_raw(parcel));
    }
}

/// Approximate committed byte size of a parcel.
///
/// # Safety
/// The parcel pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_parcel_byte_size(parcel: *const GoldyParcel) -> u64 {
    if parcel.is_null() {
        return 0;
    }
    (*parcel).inner.byte_size()
}

/// Read buffer parcel contents back to CPU memory.
///
/// # Safety
/// All pointers must be valid. `output` must point to at least `output_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_parcel_read_to_cpu(
    parcel: *const GoldyParcel,
    device: *const GoldyDevice,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    if parcel.is_null() || device.is_null() || output.is_null() {
        return GoldyResult::NullPointer;
    }

    let out = slice::from_raw_parts_mut(output, output_size);
    match (*parcel).inner.read_to_cpu(&(*device).inner, out) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

/// Bindless resource slot index for shader binding.
///
/// Returns `u32::MAX` if the index is unavailable (e.g. mosaic parcels).
///
/// # Safety
/// The parcel pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_parcel_resource_index(parcel: *const GoldyParcel, access: GoldyResourceAccess) -> u32 {
    if parcel.is_null() {
        return u32::MAX;
    }
    (*parcel).inner.resource_index(access.into()).unwrap_or(u32::MAX)
}
