//! FFI bindings for Buffer.

use crate::device::GoldyDevice;
use crate::error::{set_last_error_from_anyhow, GoldyResult};
use crate::types::{GoldyBufferKind, GoldyResourceAccess};
use std::ptr;
use std::slice;

/// Opaque handle to a Goldy Buffer.
pub struct GoldyBuffer {
    pub(crate) inner: goldy::Buffer,
}

/// Create a new buffer with the specified access pattern.
///
/// # Access Patterns
/// - `Scattered` (0): Any thread can access any address (StructuredBuffer, RWStructuredBuffer)
/// - `Broadcast` (1): All threads read same address (ConstantBuffer)
///
/// Returns a pointer to the buffer, or null on failure.
///
/// # Safety
/// The device pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_create(
    device: *const GoldyDevice,
    size: u64,
    access: GoldyBufferKind,
) -> *mut GoldyBuffer {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }

    match (*device)
        .inner
        .alloc_buffer(size, access.into(), None, goldy::BufferFlags::empty())
    {
        Ok(buffer) => Box::into_raw(Box::new(GoldyBuffer { inner: buffer })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Create a buffer initialized with data.
///
/// See `goldy_buffer_create` for access pattern documentation.
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
    access: GoldyBufferKind,
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

    match (*device).inner.alloc_buffer_with_bytes(data_slice, access.into()) {
        Ok(buffer) => Box::into_raw(Box::new(GoldyBuffer { inner: buffer })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Create a buffer initialized with data and an explicit element stride.
///
/// Use stride `4` for `uint`/`float` scattered buffers; stride `1` for raw byte blobs.
///
/// # Safety
/// The device pointer must be valid.
/// The data pointer must point to at least `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_create_with_data_stride(
    device: *const GoldyDevice,
    data: *const u8,
    size: usize,
    access: GoldyBufferKind,
    element_stride: u32,
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

    match (*device)
        .inner
        .alloc_buffer_with_bytes_stride(data_slice, access.into(), element_stride)
    {
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

/// Read buffer contents to CPU memory.
///
/// # Safety
/// All pointers must be valid. `output` must point to at least `output_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_read_to_cpu(
    buffer: *const GoldyBuffer,
    device: *const GoldyDevice,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    if buffer.is_null() || device.is_null() || output.is_null() {
        return GoldyResult::NullPointer;
    }

    let out = slice::from_raw_parts_mut(output, output_size);
    match (*buffer).inner.read_to_cpu(&(*device).inner, out) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

/// Bindless resource slot index for shader binding.
///
/// Returns `u32::MAX` if the index is unavailable.
///
/// # Safety
/// The buffer pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_resource_index(buffer: *const GoldyBuffer, access: GoldyResourceAccess) -> u32 {
    if buffer.is_null() {
        return u32::MAX;
    }
    (*buffer).inner.resource_index(access.into()).unwrap_or(u32::MAX)
}

/// Get the buffer's access pattern.
///
/// # Safety
/// The buffer pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_access(buffer: *const GoldyBuffer) -> GoldyBufferKind {
    if buffer.is_null() {
        return GoldyBufferKind::Scattered;
    }
    (*buffer).inner.access().into()
}
