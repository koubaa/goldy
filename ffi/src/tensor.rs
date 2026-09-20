//! Tensor algebra FFI (`tensor` feature).

use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::runtime::GoldyRuntime;
use crate::scheme::GoldyScheme;
use goldy::{Tensor, TensorContext, TensorDType, TensorScalar, TensorShape};
use std::ptr;
use std::slice;

/// Dense tensor dtype.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyTensorDType {
    F32 = 0,
    U32 = 1,
    I32 = 2,
}

impl From<GoldyTensorDType> for TensorDType {
    fn from(v: GoldyTensorDType) -> Self {
        match v {
            GoldyTensorDType::F32 => TensorDType::F32,
            GoldyTensorDType::U32 => TensorDType::U32,
            GoldyTensorDType::I32 => TensorDType::I32,
        }
    }
}

impl From<TensorDType> for GoldyTensorDType {
    fn from(v: TensorDType) -> Self {
        match v {
            TensorDType::F32 => GoldyTensorDType::F32,
            TensorDType::U32 => GoldyTensorDType::U32,
            TensorDType::I32 => GoldyTensorDType::I32,
        }
    }
}

/// Concrete shape descriptor (`rank` leading entries of `dims` are live).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyTensorShape {
    pub rank: u32,
    pub dims: [u32; 4],
}

impl From<TensorShape> for GoldyTensorShape {
    fn from(s: TensorShape) -> Self {
        let mut dims = [0u32; 4];
        let d = s.dims();
        dims[..d.len()].copy_from_slice(d);
        Self {
            rank: s.rank() as u32,
            dims,
        }
    }
}

/// Opaque owned tensor.
pub struct GoldyTensor {
    pub(crate) inner: Tensor,
}

/// Prepared tensor kernels plus layout keepalive.
pub struct GoldyTensorContext {
    pub(crate) inner: TensorContext,
}

fn parse_shape(rank: u32, dims: *const u32) -> Result<TensorShape, GoldyResult> {
    if rank > 4 {
        set_last_error("tensor rank exceeds 4");
        return Err(GoldyResult::InvalidArgument);
    }
    if rank > 0 && dims.is_null() {
        set_last_error("tensor dims is null");
        return Err(GoldyResult::NullPointer);
    }
    let slice = if rank == 0 {
        &[][..]
    } else {
        unsafe { slice::from_raw_parts(dims, rank as usize) }
    };
    TensorShape::from_dims(slice).map_err(|e| {
        set_last_error(e.to_string());
        GoldyResult::InvalidArgument
    })
}

/// Acquire a packed tensor. `data` may be null to leave the buffer uninitialized.
///
/// # Safety
/// `runtime` must be valid. `dims` must have `rank` elements when `rank > 0`.
#[no_mangle]
pub unsafe extern "C" fn goldy_runtime_acquire_tensor(
    runtime: *mut GoldyRuntime,
    dtype: GoldyTensorDType,
    rank: u32,
    dims: *const u32,
    data: *const u8,
    data_size: usize,
) -> *mut GoldyTensor {
    if runtime.is_null() {
        set_last_error("Runtime is null");
        return ptr::null_mut();
    }
    let shape = match parse_shape(rank, dims) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let dt: TensorDType = dtype.into();
    let result = if data.is_null() {
        Tensor::zeros(&(*runtime).inner, shape, dt)
    } else {
        let bytes = slice::from_raw_parts(data, data_size);
        match dt {
            TensorDType::F32 => {
                if bytes.len() % 4 != 0 {
                    set_last_error("tensor F32 data is not a multiple of 4 bytes");
                    return ptr::null_mut();
                }
                let n = bytes.len() / 4;
                let vals = slice::from_raw_parts(bytes.as_ptr() as *const f32, n);
                Tensor::from_f32(&(*runtime).inner, shape, vals)
            }
            TensorDType::U32 => {
                if bytes.len() % 4 != 0 {
                    set_last_error("tensor U32 data is not a multiple of 4 bytes");
                    return ptr::null_mut();
                }
                let n = bytes.len() / 4;
                let vals = slice::from_raw_parts(bytes.as_ptr() as *const u32, n);
                Tensor::from_u32(&(*runtime).inner, shape, vals)
            }
            TensorDType::I32 => {
                if bytes.len() % 4 != 0 {
                    set_last_error("tensor I32 data is not a multiple of 4 bytes");
                    return ptr::null_mut();
                }
                let n = bytes.len() / 4;
                let vals = slice::from_raw_parts(bytes.as_ptr() as *const i32, n);
                Tensor::from_i32(&(*runtime).inner, shape, vals)
            }
        }
    };
    match result {
        Ok(t) => Box::into_raw(Box::new(GoldyTensor { inner: t })),
        Err(e) => {
            set_last_error(e.to_string());
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn goldy_tensor_destroy(tensor: *mut GoldyTensor) {
    if !tensor.is_null() {
        drop(Box::from_raw(tensor));
    }
}

#[no_mangle]
pub unsafe extern "C" fn goldy_tensor_dtype(tensor: *const GoldyTensor) -> GoldyTensorDType {
    if tensor.is_null() {
        return GoldyTensorDType::F32;
    }
    (*tensor).inner.dtype().into()
}

#[no_mangle]
pub unsafe extern "C" fn goldy_tensor_shape(tensor: *const GoldyTensor, out: *mut GoldyTensorShape) -> GoldyResult {
    if tensor.is_null() || out.is_null() {
        set_last_error("tensor or out is null");
        return GoldyResult::NullPointer;
    }
    *out = (*tensor).inner.shape().into();
    GoldyResult::Ok
}

#[no_mangle]
pub unsafe extern "C" fn goldy_tensor_context_create(runtime: *mut GoldyRuntime) -> *mut GoldyTensorContext {
    if runtime.is_null() {
        set_last_error("Runtime is null");
        return ptr::null_mut();
    }
    match TensorContext::new(&(*runtime).inner) {
        Ok(inner) => Box::into_raw(Box::new(GoldyTensorContext { inner })),
        Err(e) => {
            set_last_error(e.to_string());
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn goldy_tensor_context_destroy(ctx: *mut GoldyTensorContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

unsafe fn map_op<F>(
    ctx: *mut GoldyTensorContext,
    scheme: *mut GoldyScheme,
    label: *const libc::c_char,
    f: F,
) -> *mut GoldyTensor
where
    F: FnOnce(&mut goldy::TensorRecorder<'_>, &str) -> Result<Tensor, goldy::GoldyError>,
{
    if ctx.is_null() || scheme.is_null() {
        set_last_error("tensor context or scheme is null");
        return ptr::null_mut();
    }
    let label = if label.is_null() {
        "tensor_op"
    } else {
        match std::ffi::CStr::from_ptr(label).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error_from_anyhow(&anyhow::anyhow!("{e}"));
                return ptr::null_mut();
            }
        }
    };
    let mut rec = (*ctx).inner.recorder(&mut (*scheme).inner);
    match f(&mut rec, label) {
        Ok(t) => Box::into_raw(Box::new(GoldyTensor { inner: t })),
        Err(e) => {
            set_last_error(e.to_string());
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn goldy_tensor_add(
    ctx: *mut GoldyTensorContext,
    scheme: *mut GoldyScheme,
    label: *const libc::c_char,
    a: *const GoldyTensor,
    b: *const GoldyTensor,
) -> *mut GoldyTensor {
    if a.is_null() || b.is_null() {
        set_last_error("tensor operand is null");
        return ptr::null_mut();
    }
    map_op(ctx, scheme, label, |rec, label| {
        rec.add(label, (*a).inner.view(), (*b).inner.view())
    })
}

#[no_mangle]
pub unsafe extern "C" fn goldy_tensor_matmul(
    ctx: *mut GoldyTensorContext,
    scheme: *mut GoldyScheme,
    label: *const libc::c_char,
    a: *const GoldyTensor,
    b: *const GoldyTensor,
) -> *mut GoldyTensor {
    if a.is_null() || b.is_null() {
        set_last_error("tensor operand is null");
        return ptr::null_mut();
    }
    map_op(ctx, scheme, label, |rec, label| {
        rec.matmul(label, (*a).inner.view(), (*b).inner.view())
    })
}

#[no_mangle]
pub unsafe extern "C" fn goldy_tensor_fill_f32(
    ctx: *mut GoldyTensorContext,
    scheme: *mut GoldyScheme,
    label: *const libc::c_char,
    tensor: *mut GoldyTensor,
    value: f32,
) -> GoldyResult {
    if ctx.is_null() || scheme.is_null() || tensor.is_null() {
        set_last_error("null tensor fill argument");
        return GoldyResult::NullPointer;
    }
    let label = if label.is_null() {
        "fill"
    } else {
        match std::ffi::CStr::from_ptr(label).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error_from_anyhow(&anyhow::anyhow!("{e}"));
                return GoldyResult::InvalidArgument;
            }
        }
    };
    let mut rec = (*ctx).inner.recorder(&mut (*scheme).inner);
    match rec.fill(label, (*tensor).inner.view(), TensorScalar::F32(value)) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(e.to_string());
            GoldyResult::InvalidArgument
        }
    }
}
