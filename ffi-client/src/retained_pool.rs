use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{non_null, Result};
use crate::texture::Texture;
use crate::sys::{self, GoldyRecordBuilder, GoldyRetainedPool};
use crate::types::BufferKind;
use bytemuck::Pod;
use std::ffi::CString;

/// Builder for a retained record buffer (one backing buffer, multiple field parcels).
pub struct RecordBuilder {
    ptr: *mut GoldyRecordBuilder,
}

impl RecordBuilder {
    pub fn new() -> Result<Self> {
        let ptr = non_null(unsafe { sys::goldy_record_builder_create() })?;
        Ok(Self { ptr })
    }

    pub fn emplace_pod<T: Pod>(&mut self, data: &[T]) -> Result<u32> {
        self.emplace_named_pod(None, data)
    }

    pub fn emplace_named_pod<T: Pod>(&mut self, name: Option<&str>, data: &[T]) -> Result<u32> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        self.emplace_named_bytes(name, bytes, data.len() as u64, std::mem::size_of::<T>() as u32)
    }

    pub fn emplace_bytes(&mut self, data: &[u8], element_count: u64, element_stride: u32) -> Result<u32> {
        self.emplace_named_bytes(None, data, element_count, element_stride)
    }

    pub fn emplace_named_bytes(
        &mut self,
        name: Option<&str>,
        data: &[u8],
        element_count: u64,
        element_stride: u32,
    ) -> Result<u32> {
        let name_cstring = name
            .map(CString::new)
            .transpose()
            .map_err(|_| crate::error::GoldyError::from_message("field name contains interior null byte"))?;
        let name_ptr = name_cstring.as_ref().map_or(std::ptr::null(), |n| n.as_ptr());
        let slot = unsafe {
            sys::goldy_record_builder_emplace(
                self.ptr,
                name_ptr,
                data.as_ptr(),
                data.len(),
                element_count,
                element_stride,
            )
        };
        if slot == u32::MAX {
            return Err(crate::error::GoldyError::from_last_error());
        }
        Ok(slot)
    }

    pub fn build(self, pool: &mut RetainedPool) -> Result<Buffer> {
        let ptr = unsafe { sys::goldy_record_builder_build(self.ptr, pool.as_mut_ptr()) };
        std::mem::forget(self);
        Buffer::from_ptr(non_null(ptr)?)
    }
}

impl Drop for RecordBuilder {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_record_builder_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Deed-governed pool for retained GPU buffers and texture parcels.
pub struct RetainedPool {
    ptr: *mut GoldyRetainedPool,
}

impl RetainedPool {
    pub fn new(device: &Device) -> Result<Self> {
        let ptr = non_null(unsafe { sys::goldy_retained_pool_create(device.as_ptr()) })?;
        Ok(Self { ptr })
    }

    pub fn record(&mut self) -> Result<RecordBuilder> {
        RecordBuilder::new()
    }

    /// Build a partitioned buffer from named typed fields.
    pub fn acquire_record_pod<T: Pod>(&mut self, fields: &[(&str, &[T])]) -> Result<Buffer> {
        let specs: Vec<RecordField> = fields.iter().map(|&(name, data)| (name, data).into()).collect();
        self.acquire_record(&specs)
    }

    /// Build a partitioned buffer from named or ordinal fields.
    pub fn acquire_record(&mut self, fields: &[RecordField<'_>]) -> Result<Buffer> {
        let mut builder = self.record()?;
        for field in fields {
            builder.emplace_named_bytes(field.name, field.data, field.element_count, field.element_stride)?;
        }
        builder.build(self)
    }

    /// Acquire a retained buffer.
    ///
    /// Pass `init: None` for uninitialized storage. `element_stride` of `None` selects stride `1`.
    pub fn acquire_buffer(
        &mut self,
        size: u64,
        kind: BufferKind,
        element_stride: Option<u32>,
        init: Option<&[u8]>,
    ) -> Result<Buffer> {
        let (data, data_size) = match init {
            Some(bytes) => (bytes.as_ptr(), bytes.len()),
            None => (std::ptr::null(), 0),
        };
        let stride = element_stride.unwrap_or(0);
        let ptr = non_null(unsafe {
            sys::goldy_retained_pool_acquire_buffer(self.ptr, size, kind.into(), stride, data, data_size)
        })?;
        Buffer::from_ptr(ptr)
    }

    /// Acquire a retained buffer from a typed slice. Element stride is inferred from `T`.
    pub fn acquire_buffer_with_data<T: Pod>(&mut self, data: &[T], kind: BufferKind) -> Result<Buffer> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        self.acquire_buffer(
            bytes.len() as u64,
            kind,
            Some(std::mem::size_of::<T>() as u32),
            Some(bytes),
        )
    }

    /// Acquire a retained buffer from a raw byte slice with an explicit element stride.
    pub fn acquire_buffer_bytes(&mut self, data: &[u8], kind: BufferKind, element_stride: u32) -> Result<Buffer> {
        self.acquire_buffer(data.len() as u64, kind, Some(element_stride), Some(data))
    }

    /// Acquire a retained texture parcel.
    pub fn acquire_texture(
        &mut self,
        width: u32,
        height: u32,
        format: crate::types::TextureFormat,
        kind: crate::types::TextureKind,
        flags: crate::types::TextureFlags,
        init: Option<&[u8]>,
    ) -> Result<Texture> {
        let (data, data_size) = match init {
            Some(bytes) => (bytes.as_ptr(), bytes.len()),
            None => (std::ptr::null(), 0),
        };
        let ptr = non_null(unsafe {
            sys::goldy_retained_pool_acquire_texture(
                self.ptr,
                width,
                height,
                format.into(),
                kind.into(),
                flags.into(),
                data,
                data_size,
            )
        })?;
        Texture::from_ptr(ptr)
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut GoldyRetainedPool {
        self.ptr
    }
}

/// One field for [`RetainedPool::acquire_record`].
pub struct RecordField<'a> {
    pub name: Option<&'a str>,
    pub data: &'a [u8],
    pub element_count: u64,
    pub element_stride: u32,
}

impl<'a, T: Pod> From<(&'a str, &'a [T])> for RecordField<'a> {
    fn from((name, data): (&'a str, &'a [T])) -> Self {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        Self {
            name: Some(name),
            data: bytes,
            element_count: data.len() as u64,
            element_stride: std::mem::size_of::<T>() as u32,
        }
    }
}

impl Drop for RetainedPool {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_retained_pool_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
