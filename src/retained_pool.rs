//! Gate-free retained allocation pool — the public door for deed-held GPU memory.
//!
//! [`RetainedPool::acquire_texture`], [`RetainedPool::acquire_buffer`], and
//! [`RetainedPool::mosaic`] are the supported ways to create retained parcels. Parcels are
//! opaque [`Parcel`] values; relinquish via [`RetainedPool::release`] or by dropping the parcel.
//!
//! Reuse-gate, transient pool, and backpressure are deferred.

use crate::buffer::{BufferPool, StructuredBufferElement};
use crate::context::Context;
use crate::device::Device;
use crate::parcel::{BookkeepingGuard, BytesByKind, MosaicSlot, Parcel, PoolBookkeeping};
use crate::timeline::ReferenceTable;
use crate::types::{BufferKind, TextureFlags, TextureFormat, TextureKind};
use crate::vram_allocator::ParcelType;
use anyhow::Result;
use std::sync::Arc;

struct MosaicSpec {
    data: Option<Vec<u8>>,
    count: u64,
    stride: u32,
}

/// Builder for a retained mosaic parcel (one backing buffer, multiple sub-views).
pub struct MosaicBuilder<'a> {
    device: &'a Arc<Device>,
    bookkeeping: &'a Arc<PoolBookkeeping>,
    specs: Vec<MosaicSpec>,
}

/// A parcel relinquished from the retained pool, stamped for handoff to the transient pool.
pub struct StampedParcel {
    pub parcel: Parcel,
    /// Per-context timelines after which the parcel may be reused; empty if never referenced.
    pub ready_after: ReferenceTable,
}

/// Deed-governed pool: allocates retained parcels; no epoch gate while held.
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

    /// Allocate a retained texture. `init: Some(data)` performs a one-shot staged upload into
    /// device-local memory (not permanently host-visible).
    pub fn acquire_texture(
        &mut self,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
        init: Option<&[u8]>,
    ) -> Result<Parcel> {
        let tex = if let Some(data) = init {
            crate::texture::Texture::with_data(&self.device, data, width, height, format, access, flags)?
        } else {
            self.device
                .alloc_texture(width, height, format, access, flags)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        self.wrap_texture(tex)
    }

    /// Begin building a retained mosaic parcel (one backing buffer, multiple sub-views).
    pub fn mosaic(&mut self) -> MosaicBuilder<'_> {
        MosaicBuilder {
            device: &self.device,
            bookkeeping: &self.bookkeeping,
            specs: Vec::new(),
        }
    }

    /// Allocate a retained buffer. `init: Some(data)` performs a one-shot staged upload.
    ///
    /// For in-place per-frame CPU rewrites, use [`Parcel::copy_into`] on the returned parcel.
    pub fn acquire_buffer(
        &mut self,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
        init: Option<&[u8]>,
    ) -> Result<Parcel> {
        let buf = self.alloc_raw_buffer(size, access, element_stride, flags, init)?;
        self.wrap_buffer(buf)
    }

    fn alloc_raw_buffer(
        &self,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
        init: Option<&[u8]>,
    ) -> Result<crate::buffer::Buffer> {
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

    /// Release a held parcel. The runtime reclaims GPU memory when safe; callers
    /// do not observe timeline tokens.
    pub fn release(&mut self, ctx: &Context, parcel: Parcel) {
        drop(self.transfer_out(ctx, parcel));
    }

    /// Internal release path. Returns stamped metadata for the future transient seam;
    /// public clients should call [`Self::release`] instead.
    pub(crate) fn transfer_out(&mut self, ctx: &Context, mut parcel: Parcel) -> StampedParcel {
        if let Some(home) = parcel.home_device().upgrade() {
            debug_assert!(
                Arc::ptr_eq(&home, &ctx.device().inner),
                "transfer_out: parcel home_device must match submitting context's device"
            );
        }
        let ready_after = parcel.last_referenced();
        parcel.release_bookkeeping();
        StampedParcel { parcel, ready_after }
    }

    /// Committed bytes currently held through this pool, by [`ParcelType`].
    pub fn bytes_by_kind(&self) -> BytesByKind {
        self.bookkeeping.snapshot()
    }

    fn home_device(&self) -> std::sync::Weak<crate::device::DeviceInner> {
        Arc::downgrade(&self.device.inner)
    }

    fn wrap_texture(&self, tex: crate::texture::Texture) -> Result<Parcel> {
        let bytes = tex.byte_size() as u64;
        let kind = ParcelType::Texture;
        self.bookkeeping.add(kind, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.bookkeeping), kind, bytes);
        Ok(Parcel::from_texture(tex, guard, self.home_device()))
    }

    fn wrap_buffer(&self, buf: crate::buffer::Buffer) -> Result<Parcel> {
        let bytes = buf.byte_size();
        let kind = ParcelType::Buffer;
        self.bookkeeping.add(kind, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.bookkeeping), kind, bytes);
        Ok(Parcel::from_buffer(buf, guard, self.home_device()))
    }
}

impl<'a> MosaicBuilder<'a> {
    /// Reserve space for `count` elements of type `T` (no initial upload).
    pub fn reserve<T: StructuredBufferElement>(&mut self, count: u64) -> MosaicSlot {
        let stride = std::mem::size_of::<T>() as u32;
        let slot = MosaicSlot(self.specs.len() as u32);
        self.specs.push(MosaicSpec {
            data: None,
            count,
            stride,
        });
        slot
    }

    /// Reserve space and upload `data` in one step.
    pub fn emplace<T: StructuredBufferElement>(&mut self, data: &[T]) -> MosaicSlot {
        let stride = std::mem::size_of::<T>() as u32;
        let slot = MosaicSlot(self.specs.len() as u32);
        self.specs.push(MosaicSpec {
            data: Some(bytemuck::cast_slice(data).to_vec()),
            count: data.len() as u64,
            stride,
        });
        slot
    }

    /// Allocate the backing buffer, carve sub-views, and return the mosaic parcel.
    pub fn build(self) -> Result<Parcel> {
        let pairs: Vec<(usize, usize)> = self
            .specs
            .iter()
            .map(|s| (s.count as usize, s.stride as usize))
            .collect();
        let total = BufferPool::padded_size(&pairs);
        let mut pool = BufferPool::new(self.device, total)?;

        let mut views = Vec::with_capacity(self.specs.len());
        for spec in &self.specs {
            let size = spec.count * spec.stride as u64;
            let view = pool.alloc_bytes(size, Some(spec.stride))?;
            if let Some(data) = &spec.data {
                view.write_data(data.as_slice())?;
            }
            views.push(view);
        }

        let bytes = pool.capacity();
        let kind = ParcelType::Buffer;
        self.bookkeeping.add(kind, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(self.bookkeeping), kind, bytes);
        Ok(Parcel::from_mosaic(
            pool,
            views,
            guard,
            Arc::downgrade(&self.device.inner),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::types::{ResourceAccess, TextureFormat};

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
        let p = pool.acquire_texture(64, 64, fmt, acc, flags, None).unwrap();
        assert_eq!(p.kind(), ParcelType::Texture);
        assert!(pool.bytes_by_kind().texture > 0);
        assert_eq!(pool.bytes_by_kind().buffer, 0);
    }

    #[test]
    fn acquire_texture_with_init_uploads() {
        let mut pool = RetainedPool::new(test_device());
        let (fmt, acc, flags) = rgba_interpolated();
        let data = vec![0u8; 32 * 32 * 4];
        let p = pool.acquire_texture(32, 32, fmt, acc, flags, Some(&data)).unwrap();
        assert_eq!(p.byte_size(), 32 * 32 * 4);
    }

    #[test]
    fn acquire_buffer_without_init_allocates() {
        let mut pool = RetainedPool::new(test_device());
        let p = pool
            .acquire_buffer(
                256,
                BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
                None,
            )
            .unwrap();
        assert_eq!(p.kind(), ParcelType::Buffer);
        assert_eq!(p.byte_size(), 256);
        assert!(pool.bytes_by_kind().buffer >= 256);
    }

    #[test]
    fn mark_referenced_is_monotonic_max() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let mut pool = RetainedPool::new(device);
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool.acquire_texture(8, 8, fmt, acc, flags, None).unwrap();
        let h = ctx.backend_handle();
        p.mark_referenced(h, 10);
        p.mark_referenced(h, 5);
        assert_eq!(p.last_referenced_on(h), Some(10));
        p.mark_referenced(h, 20);
        assert_eq!(p.last_referenced_on(h), Some(20));
    }

    #[test]
    fn transfer_out_referenced_has_ready_after() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let mut pool = RetainedPool::new(device);
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool.acquire_texture(8, 8, fmt, acc, flags, None).unwrap();
        p.mark_referenced(ctx.backend_handle(), 42);
        let stamped = pool.transfer_out(&ctx, p);
        assert_eq!(stamped.ready_after.get(&ctx.backend_handle()), Some(&42));
        assert_eq!(pool.bytes_by_kind().texture, 0);
    }

    #[test]
    fn transfer_out_unreferenced_has_none_ready_after() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let mut pool = RetainedPool::new(device);
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool.acquire_texture(8, 8, fmt, acc, flags, None).unwrap();
        let stamped = pool.transfer_out(&ctx, p);
        assert!(stamped.ready_after.is_empty());
    }

    #[test]
    fn transfer_out_preserves_texture_handle() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let mut pool = RetainedPool::new(device);
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool.acquire_texture(8, 8, fmt, acc, flags, None).unwrap();
        let h_before = p.texture_handle().unwrap();
        let stamped = pool.transfer_out(&ctx, p);
        assert_eq!(stamped.parcel.texture_handle(), Some(h_before));
    }

    #[test]
    fn bytes_by_kind_zero_after_transfer_and_drop() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let mut pool = RetainedPool::new(device);
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool.acquire_texture(16, 16, fmt, acc, flags, None).unwrap();
        assert!(pool.bytes_by_kind().texture > 0);
        let stamped = pool.transfer_out(&ctx, p);
        assert_eq!(pool.bytes_by_kind().texture, 0);
        drop(stamped);
    }

    #[test]
    fn mosaic_build_allocates_backing_and_views() {
        let mut pool = RetainedPool::new(test_device());
        let mut m = pool.mosaic();
        let a = m.emplace(&[1u32, 2, 3]);
        let b = m.reserve::<u32>(4);
        let parcel = m.build().unwrap();

        assert_eq!(parcel.kind(), ParcelType::Buffer);
        assert!(parcel.byte_size() >= 3 * 4 + 4 * 4);
        assert_eq!(parcel.view(a).size(), 12);
        assert_eq!(parcel.view(b).size(), 16);
    }

    #[test]
    fn mosaic_emplace_uploads_and_reserve_leaves_space() {
        let mut pool = RetainedPool::new(test_device());
        let mut m = pool.mosaic();
        let slot = m.emplace(&[42u32, 99]);
        let _reserved = m.reserve::<u32>(8);
        let parcel = m.build().unwrap();

        assert!(parcel.view(slot).resource_index(ResourceAccess::Write).is_some());
        assert_eq!(parcel.view(slot).offset(), 0);
        assert!(parcel.view(slot).size() > 0);
    }

    #[test]
    fn mosaic_bytes_by_kind_and_transfer_out() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let mut pool = RetainedPool::new(device);
        let mut m = pool.mosaic();
        m.emplace(&[0u32; 16]);
        let parcel = m.build().unwrap();
        let bytes_before = parcel.byte_size();
        assert!(pool.bytes_by_kind().buffer >= bytes_before);

        let h_before = parcel.buffer_handle().unwrap();
        let stamped = pool.transfer_out(&ctx, parcel);
        assert_eq!(pool.bytes_by_kind().buffer, 0);
        assert_eq!(stamped.parcel.buffer_handle(), Some(h_before));
        drop(stamped);
    }

    #[test]
    fn mosaic_mark_referenced_is_monotonic() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let mut pool = RetainedPool::new(device);
        let mut m = pool.mosaic();
        m.emplace(&[1u32]);
        let parcel = m.build().unwrap();
        let h = ctx.backend_handle();
        parcel.mark_referenced(h, 10);
        parcel.mark_referenced(h, 5);
        assert_eq!(parcel.last_referenced_on(h), Some(10));
        parcel.mark_referenced(h, 20);
        assert_eq!(parcel.last_referenced_on(h), Some(20));
    }

    #[test]
    fn copy_into_on_buffer_parcel_succeeds() {
        let mut pool = RetainedPool::new(test_device());
        let parcel = pool
            .acquire_buffer(
                16,
                BufferKind::Scattered,
                Some(4),
                crate::types::BufferFlags::empty(),
                None,
            )
            .unwrap();
        assert!(parcel.copy_into(&[1u32, 2, 3, 4]).is_ok());
    }

    #[test]
    fn copy_into_on_texture_parcel_errors() {
        let mut pool = RetainedPool::new(test_device());
        let (fmt, acc, flags) = rgba_interpolated();
        let parcel = pool.acquire_texture(4, 4, fmt, acc, flags, None).unwrap();
        let err = parcel.copy_into(&[0u32]).unwrap_err();
        assert!(err.to_string().contains("only valid for non-mosaic buffer parcels"));
    }

    #[test]
    fn copy_into_on_mosaic_parcel_errors() {
        let mut pool = RetainedPool::new(test_device());
        let mut m = pool.mosaic();
        m.emplace(&[1u32]);
        let parcel = m.build().unwrap();
        let err = parcel.copy_into(&[0u32]).unwrap_err();
        assert!(err.to_string().contains("only valid for non-mosaic buffer parcels"));
    }

    #[test]
    fn resource_index_read_on_scattered_buffer_parcel() {
        let mut pool = RetainedPool::new(test_device());
        let parcel = pool
            .acquire_buffer(
                64,
                BufferKind::Scattered,
                Some(4),
                crate::types::BufferFlags::empty(),
                None,
            )
            .unwrap();
        assert!(parcel.resource_index(ResourceAccess::Read).is_some());
    }

    #[test]
    fn bind_parcel_wires_parcel_resource_id() {
        use crate::compute::ComputePipeline;
        use crate::shader::ShaderModule;
        use crate::task_graph::{ResourceId, TaskGraph};

        let device = test_device();
        let mut pool = RetainedPool::new(device.clone());
        let (fmt, acc, flags) = rgba_interpolated();
        let parcel = pool.acquire_texture(4, 4, fmt, acc, flags, None).unwrap();
        let expected = parcel.resource_id();

        let shader = ShaderModule::from_slang(&device, "void main() {}").unwrap();
        let pipeline = ComputePipeline::new(&device, &shader).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("a", &pipeline)
            .bind_parcel(&parcel, crate::task_graph::NodeAccess::Read)
            .dispatch(1, 1, 1);

        let binding = graph.ir().nodes[0].bindings[0].resource;
        assert_eq!(binding, expected);
        match binding {
            ResourceId::Texture(h) => {
                assert_eq!(h, parcel.texture_handle().unwrap());
            }
            _ => panic!("expected texture resource"),
        }
    }
}
