//! FFI bindings for [`goldy::BufferPool`] and [`goldy::BufferView`].

use crate::device::GoldyDevice;
use crate::error::{set_last_error_from_anyhow, GoldyResult};
use crate::types::GoldyResourceAccess;
use std::slice;

/// Opaque handle to a Goldy BufferPool.
pub struct GoldyBufferPool {
    pub(crate) inner: goldy::BufferPool,
}

/// Opaque handle to a Goldy BufferView (sub-range of a pool backing buffer).
pub struct GoldyBufferView {
    pub(crate) inner: goldy::BufferView,
}

/// Create a buffer pool with the given total capacity in bytes.
///
/// # Safety
/// The device pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_pool_create(device: *const GoldyDevice, capacity: u64) -> *mut GoldyBufferPool {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return std::ptr::null_mut();
    }

    match goldy::BufferPool::new(&(*device).inner, capacity) {
        Ok(pool) => Box::into_raw(Box::new(GoldyBufferPool { inner: pool })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            std::ptr::null_mut()
        }
    }
}

/// Destroy a buffer pool and its views.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_pool_destroy(pool: *mut GoldyBufferPool) {
    if !pool.is_null() {
        drop(Box::from_raw(pool));
    }
}

/// Write raw bytes into the pool's backing buffer at a byte offset.
///
/// # Safety
/// The pool pointer must be valid. `data` must point to at least `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_pool_write_backing(
    pool: *const GoldyBufferPool,
    byte_offset: u64,
    data: *const u8,
    size: usize,
) -> GoldyResult {
    if pool.is_null() {
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

    match (*pool).inner.backing_buffer().write(byte_offset, data_slice) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

/// Allocate `count` u32 elements from the pool.
///
/// Returns a heap-allocated view handle, or null on failure.
///
/// # Safety
/// The pool pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_pool_alloc_u32(pool: *mut GoldyBufferPool, count: u64) -> *mut GoldyBufferView {
    if pool.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Pool is null"));
        return std::ptr::null_mut();
    }

    match (*pool).inner.alloc::<u32>(count) {
        Ok(view) => Box::into_raw(Box::new(GoldyBufferView { inner: view })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            std::ptr::null_mut()
        }
    }
}

/// Destroy a buffer view.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_view_destroy(view: *mut GoldyBufferView) {
    if !view.is_null() {
        drop(Box::from_raw(view));
    }
}

/// View size in bytes.
///
/// # Safety
/// The view pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_view_size(view: *const GoldyBufferView) -> u64 {
    if view.is_null() {
        return 0;
    }
    (*view).inner.size()
}

/// View offset within the parent buffer in bytes.
///
/// # Safety
/// The view pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_view_offset(view: *const GoldyBufferView) -> u64 {
    if view.is_null() {
        return 0;
    }
    (*view).inner.offset()
}

/// Bindless resource slot index for shader binding.
///
/// Returns `u32::MAX` if the index is unavailable.
///
/// # Safety
/// The view pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_view_resource_index(
    view: *const GoldyBufferView,
    access: GoldyResourceAccess,
) -> u32 {
    if view.is_null() {
        return u32::MAX;
    }
    (*view).inner.resource_index(access.into()).unwrap_or(u32::MAX)
}

/// Write u32 cells into the view (`data` is `count * 4` bytes).
///
/// # Safety
/// The view pointer must be valid. `data` must point to at least `count * 4` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_view_write_u32(
    view: *const GoldyBufferView,
    data: *const u32,
    count: usize,
) -> GoldyResult {
    if view.is_null() {
        return GoldyResult::NullPointer;
    }
    if data.is_null() && count > 0 {
        return GoldyResult::NullPointer;
    }

    let cells = if count > 0 {
        slice::from_raw_parts(data, count)
    } else {
        &[]
    };

    match (*view).inner.write_data(cells) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}
