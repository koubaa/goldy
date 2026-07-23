//! Gate-free retained allocation pool — the public door for deed-held GPU memory.
//!
//! [`RetainedPool::acquire_texture`], [`RetainedPool::acquire_buffer`], and
//! [`RetainedPool::acquire_record`] are the supported ways to create retained resources.
//! Buffers are acquired aggregates; bind their [`crate::Parcel`] units. Relinquish via
//! [`RetainedPool::release_buffer`] / [`RetainedPool::release_texture`] or by dropping.

use crate::buffer::{alloc_scattered_subregions, ScatteredSubregionSpec, StructuredBufferElement};
use crate::context::Context;
use crate::device::Device;
use crate::parcel::{BookkeepingGuard, Buffer, BytesByKind, Init, PoolBookkeeping, RecordField, Texture};
use crate::timeline::ReferenceTable;
use crate::types::{BufferKind, TextureFlags, TextureFormat, TextureKind};
use crate::vram_allocator::ParcelType;
use anyhow::Result;
use std::sync::Arc;

/// A resource relinquished from the retained pool, stamped for handoff to the transient pool.
pub(crate) enum RetainedHold {
    Buffer(Buffer),
    Texture(Texture),
}

/// A resource relinquished from the retained pool, stamped for handoff to the transient pool.
pub(crate) struct StampedParcel {
    pub(crate) hold: RetainedHold,
    /// Per-context timelines after which the resource may be reused; empty if never referenced.
    pub(crate) ready_after: ReferenceTable,
}

/// Deed-governed pool: allocates retained resources; no epoch gate while held.
pub struct RetainedPool {
    device: Arc<Device>,
    bookkeeping: Arc<PoolBookkeeping>,
}

impl RetainedPool {
    /// Create a pool tied to `device` (sole allocation door for retained memory in this unit).
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            bookkeeping: Arc::new(PoolBookkeeping::new()),
        }
    }

    /// Allocate a retained texture parcel. `init: Some(data)` performs a one-shot staged upload.
    pub fn acquire_texture(
        &mut self,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
        init: Option<&[u8]>,
    ) -> Result<Texture> {
        let tex = if let Some(data) = init {
            crate::texture::TextureBacking::with_data(&self.device, data, width, height, format, access, flags)?
        } else {
            self.device
                .alloc_texture(width, height, format, access, flags)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        self.wrap_texture(tex)
    }

    /// Allocate a retained buffer. `init: Some(data)` performs a one-shot staged upload.
    ///
    /// For in-place per-frame CPU rewrites, use [`crate::MemoryExchange::bind_deposit_buffer`]
    /// on the buffer's whole parcel (`&*buffer` or `buffer.whole()`).
    pub fn acquire_buffer(
        &mut self,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
        init: Option<&[u8]>,
    ) -> Result<Buffer> {
        let buf = self.alloc_raw_buffer(size, access, element_stride, flags, init)?;
        self.wrap_buffer(buf)
    }

    /// Allocate a retained buffer from a typed slice. Element stride is inferred from `T`.
    pub fn acquire_buffer_with_data<T: StructuredBufferElement>(
        &mut self,
        data: &[T],
        access: BufferKind,
    ) -> Result<Buffer> {
        self.acquire_buffer_with_data_and_flags(data, access, crate::types::BufferFlags::empty())
    }

    /// Allocate a retained buffer from a typed slice with explicit flags.
    pub fn acquire_buffer_with_data_and_flags<T: StructuredBufferElement>(
        &mut self,
        data: &[T],
        access: BufferKind,
        flags: crate::types::BufferFlags,
    ) -> Result<Buffer> {
        let stride = std::mem::size_of::<T>() as u32;
        let bytes = bytemuck::cast_slice(data);
        self.acquire_buffer(bytes.len() as u64, access, Some(stride), flags, Some(bytes))
    }

    /// Allocate an uninitialized retained buffer sized for `element_count` elements of type `T`.
    pub fn acquire_buffer_sized<T: StructuredBufferElement>(
        &mut self,
        element_count: u64,
        access: BufferKind,
        flags: crate::types::BufferFlags,
    ) -> Result<Buffer> {
        let stride = std::mem::size_of::<T>() as u32;
        self.acquire_buffer(element_count * stride as u64, access, Some(stride), flags, None)
    }

    /// Allocate a retained buffer partitioned into named or ordinal fields.
    pub fn acquire_record(&mut self, fields: impl IntoIterator<Item = RecordField>) -> Result<Buffer> {
        let fields: Vec<RecordField> = fields.into_iter().collect();
        assert!(!fields.is_empty(), "acquire_record requires at least one field");

        let regions: Vec<ScatteredSubregionSpec<'_>> = fields
            .iter()
            .map(|field| match &field.init {
                Init::Data { bytes, count, stride } => ScatteredSubregionSpec {
                    byte_size: *count * *stride as u64,
                    element_stride: *stride,
                    init: Some(bytes.as_slice()),
                },
                Init::Reserve { count, stride } => ScatteredSubregionSpec {
                    byte_size: *count * *stride as u64,
                    element_stride: *stride,
                    init: None,
                },
            })
            .collect();
        let (backing, views) = alloc_scattered_subregions(&self.device, &regions)?;

        let mut field_names = Vec::with_capacity(fields.len());
        for field in &fields {
            field_names.push(field.name.as_ref().map(|n| n.to_string()));
        }

        let bytes = backing.size();
        let kind = ParcelType::Buffer;
        self.bookkeeping.add(kind, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.bookkeeping), kind, bytes);
        Ok(Buffer::from_partitioned(
            Arc::new(backing),
            views,
            field_names,
            guard,
            self.home_device(),
        ))
    }

    fn alloc_raw_buffer(
        &self,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
        init: Option<&[u8]>,
    ) -> Result<crate::buffer::Allocation> {
        if let Some(data) = init {
            self.device
                .alloc_buffer_with_bytes_stride_and_flags(data, access, element_stride.unwrap_or(1), flags)
                .map_err(|e| anyhow::anyhow!("{e}"))
        } else {
            self.device
                .alloc_buffer(size, access, element_stride, flags)
                .map_err(|e| anyhow::anyhow!("{e}"))
        }
    }

    /// Release a held texture parcel into the context transient pool for epoch-gated reuse.
    pub fn release_texture(&mut self, ctx: &Context, texture: Texture) {
        let stamped = self.transfer_out_texture(ctx, texture);
        ctx.with_transient_pool(|pool| pool.adopt(stamped));
    }

    /// Release a held buffer into the context transient pool for epoch-gated reuse.
    pub fn release_buffer(&mut self, ctx: &Context, buffer: Buffer) {
        let stamped = self.transfer_out_buffer(ctx, buffer);
        ctx.with_transient_pool(|pool| pool.adopt(stamped));
    }

    pub(crate) fn transfer_out_texture(&mut self, ctx: &Context, mut texture: Texture) -> StampedParcel {
        if let Some(home) = texture.home_device().upgrade() {
            debug_assert!(
                Arc::ptr_eq(&home, &ctx.device().inner),
                "transfer_out: texture home_device must match submitting context's device"
            );
        }
        let ready_after = texture.last_referenced();
        texture.release_bookkeeping();
        StampedParcel {
            hold: RetainedHold::Texture(texture),
            ready_after,
        }
    }

    pub(crate) fn transfer_out_buffer(&mut self, ctx: &Context, mut buffer: Buffer) -> StampedParcel {
        if let Some(home) = buffer.home_device().upgrade() {
            debug_assert!(
                Arc::ptr_eq(&home, &ctx.device().inner),
                "transfer_out: buffer home_device must match submitting context's device"
            );
        }
        let ready_after = buffer.last_referenced();
        buffer.release_bookkeeping();
        StampedParcel {
            hold: RetainedHold::Buffer(buffer),
            ready_after,
        }
    }

    /// Committed bytes currently held through this pool (buffers vs textures).
    pub fn bytes_by_kind(&self) -> BytesByKind {
        self.bookkeeping.snapshot()
    }

    fn home_device(&self) -> std::sync::Weak<crate::device::DeviceInner> {
        Arc::downgrade(&self.device.inner)
    }

    fn wrap_texture(&self, tex: crate::texture::TextureBacking) -> Result<Texture> {
        let bytes = tex.byte_size() as u64;
        let kind = ParcelType::Texture;
        self.bookkeeping.add(kind, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.bookkeeping), kind, bytes);
        Ok(Texture::from_backing(tex, guard, self.home_device()))
    }

    fn wrap_buffer(&self, buf: crate::buffer::Allocation) -> Result<Buffer> {
        let bytes = buf.byte_size();
        let kind = ParcelType::Buffer;
        self.bookkeeping.add(kind, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.bookkeeping), kind, bytes);
        Ok(Buffer::from_single(buf, guard, self.home_device()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::parcel::{field, Init};
    use crate::types::{ResourceAccess, TextureFormat};
    use crate::MemoryExchange;

    fn test_device() -> Arc<Device> {
        Arc::new(Device::from_backend(Box::new(MockBackend::new())).expect("mock device"))
    }

    fn test_ctx(device: &Arc<Device>) -> Context {
        device.create_context().unwrap()
    }

    fn rgba_interpolated() -> (TextureFormat, TextureKind, TextureFlags) {
        (
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
        )
    }

    #[test]
    fn acquire_texture_without_init_allocates() {
        let mut pool = RetainedPool::new(test_device());
        let (fmt, acc, flags) = rgba_interpolated();
        let _p = pool.acquire_texture(64, 64, fmt, acc, flags, None).unwrap();
        assert!(pool.bytes_by_kind().texture > 0);
        assert_eq!(pool.bytes_by_kind().buffer, 0);
    }

    #[test]
    fn acquire_buffer_without_init_allocates() {
        let mut pool = RetainedPool::new(test_device());
        let b = pool
            .acquire_buffer(
                256,
                BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
                None,
            )
            .unwrap();
        assert_eq!(b.byte_size(), 256);
        assert!(pool.bytes_by_kind().buffer >= 256);
        assert_eq!(b.unit_count(), 1);
    }

    #[test]
    fn acquire_record_builds_partitioned_buffer() {
        let mut pool = RetainedPool::new(test_device());
        let cells = pool
            .acquire_record([
                field("a", Init::data(&[1u32, 2, 3])),
                field("b", Init::reserve::<u32>(4)),
            ])
            .unwrap();
        assert!(cells.is_partitioned());
        assert_eq!(cells.unit_count(), 2);
        assert!(cells["a"].resource_index(ResourceAccess::Write).is_some());
        assert!(cells["b"].resource_index(ResourceAccess::Write).is_some());
    }

    #[test]
    #[should_panic(expected = "cannot bind a partitioned buffer as one descriptor")]
    fn partitioned_buffer_whole_panics() {
        let mut pool = RetainedPool::new(test_device());
        let cells = pool
            .acquire_record([field("a", Init::data(&[1u32])), field("b", Init::reserve::<u32>(1))])
            .unwrap();
        let _ = cells.whole();
    }

    #[test]
    #[should_panic(expected = "cannot bind a partitioned buffer as one descriptor")]
    fn partitioned_buffer_deref_panics() {
        use crate::Parcel;

        let mut pool = RetainedPool::new(test_device());
        let cells = pool
            .acquire_record([field("a", Init::data(&[1u32])), field("b", Init::reserve::<u32>(1))])
            .unwrap();
        let _parcel: &Parcel = &*cells;
    }

    #[test]
    fn detach_allocation_succeeds_on_single_unit_buffer() {
        let mut pool = RetainedPool::new(test_device());
        let buffer = pool
            .acquire_buffer(
                64,
                BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
                None,
            )
            .unwrap();
        buffer.detach_allocation().expect("detach should succeed");
    }

    #[test]
    fn transfer_out_buffer_referenced_has_ready_after() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let mut pool = RetainedPool::new(device);
        let b = pool
            .acquire_buffer(
                64,
                BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
                None,
            )
            .unwrap();
        b.whole().mark_referenced(ctx.backend_handle(), 42);
        let stamped = pool.transfer_out_buffer(&ctx, b);
        assert_eq!(stamped.ready_after.get(ctx.backend_handle()), Some(42));
        assert_eq!(pool.bytes_by_kind().buffer, 0);
    }

    #[test]
    fn deposit_buffer_on_whole_buffer_parcel_succeeds() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(device);
        let buffer = pool
            .acquire_buffer(
                16,
                BufferKind::Scattered,
                Some(4),
                crate::types::BufferFlags::empty(),
                None,
            )
            .unwrap();
        let mut scheme = crate::Scheme::new(&ctx);
        let deposit = MemoryExchange::new(&ctx)
            .bind_deposit_buffer(&mut scheme, &*buffer, 16)
            .expect("bind deposit");
        deposit
            .write(&mut scheme, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))
            .expect("deposit write");
        scheme.submit().unwrap();
    }
}
