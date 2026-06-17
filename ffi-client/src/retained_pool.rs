use crate::device::Device;
use crate::error::{non_null, Result};
use crate::parcel::Parcel;
use crate::sys::{self, GoldyMosaicBuilder, GoldyRetainedPool};
use crate::types::BufferKind;
use bytemuck::Pod;

/// Index into a mosaic parcel's sub-views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MosaicSlot(pub u32);

/// Builder for a retained mosaic parcel (one backing buffer, multiple sub-views).
pub struct MosaicBuilder {
    ptr: *mut GoldyMosaicBuilder,
}

impl MosaicBuilder {
    pub fn new() -> Result<Self> {
        let ptr = non_null(unsafe { sys::goldy_mosaic_builder_create() })?;
        Ok(Self { ptr })
    }

    pub fn emplace_pod<T: Pod>(&mut self, data: &[T]) -> Result<MosaicSlot> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        self.emplace_bytes(bytes, data.len() as u64, std::mem::size_of::<T>() as u32)
    }

    pub fn emplace_bytes(&mut self, data: &[u8], element_count: u64, element_stride: u32) -> Result<MosaicSlot> {
        let slot = unsafe {
            sys::goldy_mosaic_builder_emplace(self.ptr, data.as_ptr(), data.len(), element_count, element_stride)
        };
        if slot == u32::MAX {
            return Err(crate::error::GoldyError::from_last_error());
        }
        Ok(MosaicSlot(slot))
    }

    pub fn build(self, pool: &mut RetainedPool) -> Result<Parcel> {
        let ptr = unsafe { sys::goldy_mosaic_builder_build(self.ptr, pool.as_mut_ptr()) };
        std::mem::forget(self);
        Parcel::from_ptr(non_null(ptr)?)
    }
}

impl Drop for MosaicBuilder {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_mosaic_builder_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Deed-governed pool for retained GPU parcels.
pub struct RetainedPool {
    ptr: *mut GoldyRetainedPool,
}

impl RetainedPool {
    pub fn new(device: &Device) -> Result<Self> {
        let ptr = non_null(unsafe { sys::goldy_retained_pool_create(device.as_ptr()) })?;
        Ok(Self { ptr })
    }

    pub fn mosaic(&mut self) -> Result<MosaicBuilder> {
        MosaicBuilder::new()
    }

    /// Acquire a retained buffer parcel.
    ///
    /// Pass `init: None` for uninitialized storage. `element_stride` of `None` selects stride `1`.
    pub fn acquire_buffer(
        &mut self,
        size: u64,
        kind: BufferKind,
        element_stride: Option<u32>,
        init: Option<&[u8]>,
    ) -> Result<Parcel> {
        let (data, data_size) = match init {
            Some(bytes) => (bytes.as_ptr(), bytes.len()),
            None => (std::ptr::null(), 0),
        };
        let stride = element_stride.unwrap_or(0);
        let ptr = non_null(unsafe {
            sys::goldy_retained_pool_acquire_buffer(self.ptr, size, kind.into(), stride, data, data_size)
        })?;
        Parcel::from_ptr(ptr)
    }

    /// Acquire a retained buffer parcel from a typed slice. Element stride is inferred from `T`.
    pub fn acquire_buffer_with_data<T: Pod>(&mut self, data: &[T], kind: BufferKind) -> Result<Parcel> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        self.acquire_buffer(
            bytes.len() as u64,
            kind,
            Some(std::mem::size_of::<T>() as u32),
            Some(bytes),
        )
    }

    /// Acquire a retained buffer parcel from a raw byte slice with an explicit element stride.
    pub fn acquire_buffer_bytes(&mut self, data: &[u8], kind: BufferKind, element_stride: u32) -> Result<Parcel> {
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
    ) -> Result<Parcel> {
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
        Parcel::from_ptr(ptr)
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut GoldyRetainedPool {
        self.ptr
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
