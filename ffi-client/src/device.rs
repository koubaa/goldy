use crate::buffer::Buffer;
use crate::error::{non_null, Result};
use crate::retained_pool::{RecordBuilder, RecordField};
use crate::sys::{self, GoldyDevice, GoldyRetainedPool};
use crate::texture::Texture;
use crate::types::BufferKind;
use bytemuck::Pod;
use std::cell::Cell;

/// A GPU device handle.
pub struct Device {
    pub(crate) ptr: *mut GoldyDevice,
    pool: Cell<*mut GoldyRetainedPool>,
}

impl Device {
    pub(crate) fn from_ptr(ptr: *mut GoldyDevice) -> Self {
        Self {
            ptr,
            pool: Cell::new(std::ptr::null_mut()),
        }
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyDevice {
        self.ptr
    }

    pub(crate) fn retained_pool_ptr(&self) -> Result<*mut GoldyRetainedPool> {
        if self.pool.get().is_null() {
            let ptr = non_null(unsafe { sys::goldy_retained_pool_create(self.ptr) })?;
            self.pool.set(ptr);
        }
        Ok(self.pool.get())
    }

    pub fn record(&self) -> Result<RecordBuilder> {
        RecordBuilder::new()
    }

    /// Build a partitioned buffer from named typed fields.
    pub fn acquire_record_pod<T: Pod>(&self, fields: &[(&str, &[T])]) -> Result<Buffer> {
        let specs: Vec<RecordField> = fields.iter().map(|&(name, data)| (name, data).into()).collect();
        self.acquire_record(&specs)
    }

    /// Build a partitioned buffer from named or ordinal fields.
    pub fn acquire_record(&self, fields: &[RecordField<'_>]) -> Result<Buffer> {
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
        &self,
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
            sys::goldy_retained_pool_acquire_buffer(
                self.retained_pool_ptr()?,
                size,
                kind.into(),
                stride,
                data,
                data_size,
            )
        })?;
        Buffer::from_ptr(ptr)
    }

    /// Acquire a retained buffer from a typed slice. Element stride is inferred from `T`.
    pub fn acquire_buffer_with_data<T: Pod>(&self, data: &[T], kind: BufferKind) -> Result<Buffer> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        self.acquire_buffer(
            bytes.len() as u64,
            kind,
            Some(std::mem::size_of::<T>() as u32),
            Some(bytes),
        )
    }

    /// Acquire a retained buffer from a raw byte slice with an explicit element stride.
    pub fn acquire_buffer_bytes(&self, data: &[u8], kind: BufferKind, element_stride: u32) -> Result<Buffer> {
        self.acquire_buffer(data.len() as u64, kind, Some(element_stride), Some(data))
    }

    /// Acquire a retained texture parcel.
    pub fn acquire_texture(
        &self,
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
                self.retained_pool_ptr()?,
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
}

impl Drop for Device {
    fn drop(&mut self) {
        let pool = self.pool.get();
        if !pool.is_null() {
            unsafe { sys::goldy_retained_pool_destroy(pool) };
            self.pool.set(std::ptr::null_mut());
        }
        if !self.ptr.is_null() {
            unsafe { crate::sys::goldy_device_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
