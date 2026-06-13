//! FFI bindings for [`goldy::RetainedPool`], [`goldy::Parcel`], and mosaic builders.

use crate::device::GoldyDevice;
use crate::error::{set_last_error_from_anyhow, GoldyResult};
use crate::types::{GoldyBufferKind, GoldyResourceAccess, GoldyTextureFlags, GoldyTextureFormat, GoldyTextureKind};
use goldy::MosaicSlot;
use std::ptr;
use std::slice;
use std::sync::Arc;

struct FfiMosaicSpec {
    data: Option<Vec<u8>>,
    count: u64,
    stride: u32,
}

/// Builder for a retained mosaic parcel (one backing buffer, multiple sub-views).
pub struct GoldyMosaicBuilder {
    specs: Vec<FfiMosaicSpec>,
}

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

/// Create a mosaic builder (call [`goldy_mosaic_builder_emplace`] then [`goldy_mosaic_builder_build`]).
#[no_mangle]
pub unsafe extern "C" fn goldy_mosaic_builder_create() -> *mut GoldyMosaicBuilder {
    Box::into_raw(Box::new(GoldyMosaicBuilder { specs: Vec::new() }))
}

/// Destroy a mosaic builder without building.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_mosaic_builder_destroy(builder: *mut GoldyMosaicBuilder) {
    if !builder.is_null() {
        drop(Box::from_raw(builder));
    }
}

/// Reserve a mosaic sub-view and upload `data` (`data_size` must equal `element_count * element_stride`).
///
/// Returns the slot index, or `u32::MAX` on failure.
///
/// # Safety
/// `builder` must be valid. `data` must point to at least `data_size` bytes when non-null.
#[no_mangle]
pub unsafe extern "C" fn goldy_mosaic_builder_emplace(
    builder: *mut GoldyMosaicBuilder,
    data: *const u8,
    data_size: usize,
    element_count: u64,
    element_stride: u32,
) -> u32 {
    if builder.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("MosaicBuilder is null"));
        return u32::MAX;
    }
    if element_stride == 0 {
        set_last_error_from_anyhow(&anyhow::anyhow!("element_stride is zero"));
        return u32::MAX;
    }
    let expected = element_count.saturating_mul(element_stride as u64) as usize;
    if data_size != expected {
        set_last_error_from_anyhow(&anyhow::anyhow!(
            "data_size {data_size} != element_count * element_stride ({expected})"
        ));
        return u32::MAX;
    }
    if data.is_null() && data_size > 0 {
        set_last_error_from_anyhow(&anyhow::anyhow!("data is null"));
        return u32::MAX;
    }

    let bytes = if data_size > 0 {
        slice::from_raw_parts(data, data_size).to_vec()
    } else {
        Vec::new()
    };
    let slot = (*builder).specs.len() as u32;
    (*builder).specs.push(FfiMosaicSpec {
        data: Some(bytes),
        count: element_count,
        stride: element_stride,
    });
    slot
}

/// Build a mosaic parcel from a builder and destroy the builder.
///
/// Returns a heap-allocated parcel handle, or null on failure.
///
/// # Safety
/// `builder` and `pool` must be valid. `builder` is consumed regardless of outcome.
#[no_mangle]
pub unsafe extern "C" fn goldy_mosaic_builder_build(
    builder: *mut GoldyMosaicBuilder,
    pool: *mut GoldyRetainedPool,
) -> *mut GoldyParcel {
    if builder.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("MosaicBuilder is null"));
        return ptr::null_mut();
    }
    if pool.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("RetainedPool is null"));
        drop(Box::from_raw(builder));
        return ptr::null_mut();
    }

    let ffi_builder = Box::from_raw(builder);
    let mut mosaic = (*pool).inner.mosaic();
    for spec in ffi_builder.specs {
        if let Some(data) = spec.data {
            mosaic.emplace_bytes(&data, spec.count, spec.stride);
        } else {
            mosaic.reserve_bytes(spec.count, spec.stride);
        }
    }

    match mosaic.build() {
        Ok(parcel) => Box::into_raw(Box::new(GoldyParcel { inner: parcel })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Bindless resource index for one mosaic sub-view.
///
/// Returns `u32::MAX` if unavailable.
///
/// # Safety
/// The parcel pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_parcel_mosaic_view_resource_index(
    parcel: *const GoldyParcel,
    slot: u32,
    access: GoldyResourceAccess,
) -> u32 {
    if parcel.is_null() {
        return u32::MAX;
    }
    (*parcel)
        .inner
        .view(MosaicSlot(slot))
        .resource_index(access.into())
        .unwrap_or(u32::MAX)
}

/// Read one mosaic sub-view back to CPU memory.
///
/// `output_size` must equal the sub-view byte size.
///
/// # Safety
/// All pointers must be valid. `output` must point to at least `output_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_parcel_mosaic_view_read_to_cpu(
    parcel: *const GoldyParcel,
    slot: u32,
    device: *const GoldyDevice,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    if parcel.is_null() || device.is_null() || output.is_null() {
        return GoldyResult::NullPointer;
    }

    let out = slice::from_raw_parts_mut(output, output_size);
    match (*parcel)
        .inner
        .view(MosaicSlot(slot))
        .read_to_cpu(&(*device).inner, out)
    {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

/// Byte size of one mosaic sub-view.
///
/// Returns `0` if the parcel or slot is invalid.
///
/// # Safety
/// The parcel pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_parcel_mosaic_view_size(parcel: *const GoldyParcel, slot: u32) -> u64 {
    if parcel.is_null() {
        return 0;
    }
    (*parcel).inner.view(MosaicSlot(slot)).size()
}
