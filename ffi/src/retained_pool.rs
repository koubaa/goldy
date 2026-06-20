//! FFI bindings for [`goldy::RetainedPool`], [`goldy::Buffer`], [`goldy::Parcel`], and record builders.

use crate::device::GoldyDevice;
use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::types::{GoldyBufferKind, GoldyTextureFlags, GoldyTextureFormat, GoldyTextureKind};
use goldy::{field, Init, RecordField};
use std::ptr;
use std::slice;
use std::sync::Arc;

struct FfiRecordSpec {
    name: Option<String>,
    data: Option<Vec<u8>>,
    count: u64,
    stride: u32,
}

/// Builder for a retained record buffer (one backing buffer, multiple sub-views).
pub struct GoldyRecordBuilder {
    specs: Vec<FfiRecordSpec>,
}

/// Opaque handle to a Goldy retained allocation pool.
pub struct GoldyRetainedPool {
    pub(crate) inner: goldy::RetainedPool,
}

/// Opaque handle to an acquired [`goldy::Buffer`].
pub struct GoldyBuffer {
    pub(crate) inner: goldy::Buffer,
}

/// Bounds-checked access to one parcel unit of a retained buffer.
///
/// # Safety
/// `buffer` must be a valid pointer when non-null.
pub(crate) unsafe fn buffer_unit_at<'a>(
    buffer: *const GoldyBuffer,
    unit: u32,
) -> Result<&'a goldy::Parcel, GoldyResult> {
    if buffer.is_null() {
        return Err(GoldyResult::NullPointer);
    }
    let idx = unit as usize;
    let unit_count = (*buffer).inner.unit_count();
    if idx >= unit_count {
        set_last_error(format!(
            "buffer unit index {unit} out of range (unit_count={unit_count})"
        ));
        return Err(GoldyResult::InvalidArgument);
    }
    Ok((*buffer).inner.unit(idx))
}

/// Opaque handle to a bindable [`goldy::Parcel`] (texture parcels; buffer units use [`GoldyBuffer`] + index).
pub struct GoldyParcel {
    pub(crate) inner: goldy::Parcel,
}

#[no_mangle]
pub unsafe extern "C" fn goldy_retained_pool_create(device: *const GoldyDevice) -> *mut GoldyRetainedPool {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }
    let pool = goldy::RetainedPool::new(Arc::new((*device).inner.clone()));
    Box::into_raw(Box::new(GoldyRetainedPool { inner: pool }))
}

#[no_mangle]
pub unsafe extern "C" fn goldy_retained_pool_destroy(pool: *mut GoldyRetainedPool) {
    if !pool.is_null() {
        drop(Box::from_raw(pool));
    }
}

#[no_mangle]
pub unsafe extern "C" fn goldy_retained_pool_acquire_buffer(
    pool: *mut GoldyRetainedPool,
    size: u64,
    access: GoldyBufferKind,
    element_stride: u32,
    data: *const u8,
    data_size: usize,
) -> *mut GoldyBuffer {
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
        Ok(buffer) => Box::into_raw(Box::new(GoldyBuffer { inner: buffer })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

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

#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_destroy(buffer: *mut GoldyBuffer) {
    if !buffer.is_null() {
        drop(Box::from_raw(buffer));
    }
}

#[no_mangle]
pub unsafe extern "C" fn goldy_parcel_destroy(parcel: *mut GoldyParcel) {
    if !parcel.is_null() {
        drop(Box::from_raw(parcel));
    }
}

#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_byte_size(buffer: *const GoldyBuffer) -> u64 {
    if buffer.is_null() {
        return 0;
    }
    (*buffer).inner.byte_size()
}

#[no_mangle]
pub unsafe extern "C" fn goldy_parcel_byte_size(parcel: *const GoldyParcel) -> u64 {
    if parcel.is_null() {
        return 0;
    }
    (*parcel).inner.byte_size()
}

#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_unit_count(buffer: *const GoldyBuffer) -> u32 {
    if buffer.is_null() {
        return 0;
    }
    (*buffer).inner.unit_count() as u32
}

#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_unit_byte_size(buffer: *const GoldyBuffer, unit: u32) -> u64 {
    match buffer_unit_at(buffer, unit) {
        Ok(parcel) => parcel.byte_size(),
        Err(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_unit_read_to_cpu(
    buffer: *const GoldyBuffer,
    unit: u32,
    device: *const GoldyDevice,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    if device.is_null() || output.is_null() {
        return GoldyResult::NullPointer;
    }
    let parcel = match buffer_unit_at(buffer, unit) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let out = slice::from_raw_parts_mut(output, output_size);
    match parcel.read_to_cpu(&(*device).inner, out) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn goldy_record_builder_create() -> *mut GoldyRecordBuilder {
    Box::into_raw(Box::new(GoldyRecordBuilder { specs: Vec::new() }))
}

#[no_mangle]
pub unsafe extern "C" fn goldy_record_builder_destroy(builder: *mut GoldyRecordBuilder) {
    if !builder.is_null() {
        drop(Box::from_raw(builder));
    }
}

#[no_mangle]
pub unsafe extern "C" fn goldy_record_builder_emplace(
    builder: *mut GoldyRecordBuilder,
    name: *const std::ffi::c_char,
    data: *const u8,
    data_size: usize,
    element_count: u64,
    element_stride: u32,
) -> u32 {
    if builder.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("RecordBuilder is null"));
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

    let field_name = if name.is_null() {
        None
    } else {
        Some(std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned())
    };
    let bytes = if data_size > 0 {
        slice::from_raw_parts(data, data_size).to_vec()
    } else {
        Vec::new()
    };
    let slot = (*builder).specs.len() as u32;
    (*builder).specs.push(FfiRecordSpec {
        name: field_name,
        data: Some(bytes),
        count: element_count,
        stride: element_stride,
    });
    slot
}

#[no_mangle]
pub unsafe extern "C" fn goldy_record_builder_reserve(
    builder: *mut GoldyRecordBuilder,
    name: *const std::ffi::c_char,
    element_count: u64,
    element_stride: u32,
) -> u32 {
    if builder.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("RecordBuilder is null"));
        return u32::MAX;
    }
    if element_stride == 0 {
        set_last_error_from_anyhow(&anyhow::anyhow!("element_stride is zero"));
        return u32::MAX;
    }
    let field_name = if name.is_null() {
        None
    } else {
        Some(std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned())
    };
    let slot = (*builder).specs.len() as u32;
    (*builder).specs.push(FfiRecordSpec {
        name: field_name,
        data: None,
        count: element_count,
        stride: element_stride,
    });
    slot
}

#[no_mangle]
pub unsafe extern "C" fn goldy_record_builder_build(
    builder: *mut GoldyRecordBuilder,
    pool: *mut GoldyRetainedPool,
) -> *mut GoldyBuffer {
    if builder.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("RecordBuilder is null"));
        return ptr::null_mut();
    }
    if pool.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("RetainedPool is null"));
        drop(Box::from_raw(builder));
        return ptr::null_mut();
    }

    let ffi_builder = Box::from_raw(builder);
    let fields: Vec<RecordField> = ffi_builder
        .specs
        .into_iter()
        .map(|spec| {
            let init = if let Some(data) = spec.data {
                Init::Data {
                    bytes: data,
                    count: spec.count,
                    stride: spec.stride,
                }
            } else {
                Init::Reserve {
                    count: spec.count,
                    stride: spec.stride,
                }
            };
            match spec.name {
                Some(name) => field(name, init),
                None => goldy::ordinal(init),
            }
        })
        .collect();

    match (*pool).inner.acquire_record(fields) {
        Ok(buffer) => Box::into_raw(Box::new(GoldyBuffer { inner: buffer })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Borrow one bindable unit from a retained buffer as an owned [`GoldyParcel`] handle.
///
/// The returned parcel shares dependency-tracking state with the buffer unit.
/// Destroy with [`goldy_parcel_destroy`]. The source buffer must remain alive
/// for the duration of GPU use.
#[no_mangle]
pub unsafe extern "C" fn goldy_buffer_field(buffer: *const GoldyBuffer, unit: u32) -> *mut GoldyParcel {
    let parcel = match buffer_unit_at(buffer, unit) {
        Ok(p) => p,
        Err(_) => return ptr::null_mut(),
    };
    let parcel = parcel.clone();
    Box::into_raw(Box::new(GoldyParcel { inner: parcel }))
}
