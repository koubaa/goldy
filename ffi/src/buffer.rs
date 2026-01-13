//! FFI bindings for Buffer.

use crate::device::GoldyDevice;
use crate::error::{set_last_error_from_anyhow, GoldyResult};
use crate::types::GoldyBufferUsage;
use std::ptr;
use std::slice;

/// Opaque handle to a Goldy Buffer.
pub struct GoldyBuffer {
    pub(crate) inner: goldy::Buffer,
}

/// Create a new buffer.
///
/// Returns a pointer to the buffer, or null on failure.
///
/// # Safety
/// The device pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_create(
    device: *const GoldyDevice,
    size: u64,
    usage: GoldyBufferUsage,
) -> *mut GoldyBuffer {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }
    
    match goldy::Buffer::new(&(*device).inner, size, usage.into()) {
        Ok(buffer) => Box::into_raw(Box::new(GoldyBuffer { inner: buffer })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Create a buffer initialized with data.
///
/// Returns a pointer to the buffer, or null on failure.
///
/// # Safety
/// The device pointer must be valid.
/// The data pointer must point to at least `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_create_with_data(
    device: *const GoldyDevice,
    data: *const u8,
    size: usize,
    usage: GoldyBufferUsage,
) -> *mut GoldyBuffer {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }
    if data.is_null() && size > 0 {
        set_last_error_from_anyhow(&anyhow::anyhow!("Data is null"));
        return ptr::null_mut();
    }
    
    let data_slice = if size > 0 {
        slice::from_raw_parts(data, size)
    } else {
        &[]
    };
    
    match goldy::Buffer::with_bytes(&(*device).inner, data_slice, usage.into()) {
        Ok(buffer) => Box::into_raw(Box::new(GoldyBuffer { inner: buffer })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Destroy a buffer.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_destroy(buffer: *mut GoldyBuffer) {
    if !buffer.is_null() {
        drop(Box::from_raw(buffer));
    }
}

/// Write data to a buffer.
///
/// # Safety
/// The buffer pointer must be valid.
/// The data pointer must point to at least `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_write(
    buffer: *const GoldyBuffer,
    offset: u64,
    data: *const u8,
    size: usize,
) -> GoldyResult {
    if buffer.is_null() {
        return GoldyResult::NullPointer;
    }
    if data.is_null() && size > 0 {
        return GoldyResult::NullPointer;
    }
    
    let data_slice = if size > 0 {
        slice::from_raw_parts(data, size)
    } else {
        &[]
    };
    
    match (*buffer).inner.write(offset, data_slice) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

/// Get the buffer size in bytes.
///
/// # Safety
/// The buffer pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_size(buffer: *const GoldyBuffer) -> u64 {
    if buffer.is_null() {
        return 0;
    }
    (*buffer).inner.size()
}

/// Get the buffer usage flags.
///
/// # Safety
/// The buffer pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_usage(buffer: *const GoldyBuffer) -> GoldyBufferUsage {
    if buffer.is_null() {
        return GoldyBufferUsage(0);
    }
    (*buffer).inner.usage().into()
}

