//! Gate-free retained allocation pool — the public door for deed-held GPU memory.
//!
//! [`RetainedPool::acquire_texture`], [`RetainedPool::acquire_buffer`], and
//! [`RetainedPool::acquire_record`] are the supported ways to create retained resources.
//! Buffers are acquired aggregates; bind their [`Parcel`] units. Relinquish via
//! [`RetainedPool::release`] or by dropping.

use crate::buffer::{BufferPool, StructuredBufferElement};
use crate::context::Context;
use crate::device::Device;
use crate::parcel::{BookkeepingGuard, Buffer, BytesByKind, Init, Parcel, PoolBookkeeping, RecordField};
use crate::timeline::ReferenceTable;
use crate::types::{BufferKind, TextureFlags, TextureFormat, TextureKind};
use crate::vram_allocator::ParcelType;
use anyhow::Result;
use std::sync::Arc;

/// A resource relinquished from the retained pool, stamped for handoff to the transient pool.
pub enum RetainedHold {
    Buffer(Buffer),
    Texture(Parcel),
}

impl RetainedHold {
    pub fn byte_size(&self) -> u64 {
        match self {
            RetainedHold::Buffer(b) => b.byte_size(),
            RetainedHold::Texture(p) => p.byte_size(),
        }
    }

    pub fn last_referenced(&self) -> ReferenceTable {
        match self {
            RetainedHold::Buffer(b) => b.last_referenced(),
            RetainedHold::Texture(p) => p.last_referenced(),
        }
    }

    pub fn texture_descriptor(&self) -> Option<(u32, u32, TextureFormat, TextureKind, TextureFlags)> {
        match self {
            RetainedHold::Buffer(_) => None,
            RetainedHold::Texture(p) => p.texture_descriptor(),
        }
    }
}

/// A resource relinquished from the retained pool, stamped for handoff to the transient pool.
pub struct StampedParcel {
    pub hold: RetainedHold,
    /// Per-context timelines after which the resource may be reused; empty if never referenced.
    pub ready_after: ReferenceTable,
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

    /// Allocate a retained buffer. `init: Some(data)` performs a one-shot staged upload.
    ///
    /// For in-place per-frame CPU rewrites, use [`crate::TaskGraph::write_parcel`] on the
    /// buffer's whole parcel (`&*buffer` or `buffer.whole()`).
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

        let pairs: Vec<(usize, usize)> = fields
            .iter()
            .map(|f| match &f.init {
                Init::Data { count, stride, .. } | Init::Reserve { count, stride } => {
                    (*count as usize, *stride as usize)
                }
            })
            .collect();
        let total = BufferPool::padded_size(&pairs);
        let mut pool = BufferPool::new(&self.device, total)?;

        let mut views = Vec::with_capacity(fields.len());
        let mut field_names = Vec::with_capacity(fields.len());
        for field in &fields {
            let (count, stride, data) = match &field.init {
                Init::Data { bytes, count, stride } => (*count, *stride, Some(bytes.as_slice())),
                Init::Reserve { count, stride } => (*count, *stride, None),
            };
            let size = count * stride as u64;
            let view = pool.alloc_bytes(size, Some(stride))?;
            if let Some(data) = data {
                view.write_data(data)?;
            }
            views.push(view);
            field_names.push(field.name.as_ref().map(|n| n.to_string()));
        }

        let bytes = pool.capacity();
        let kind = ParcelType::Buffer;
        self.bookkeeping.add(kind, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.bookkeeping), kind, bytes);
        Ok(Buffer::from_partitioned(
            pool,
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
    pub fn release_texture(&mut self, ctx: &Context, parcel: Parcel) {
        let stamped = self.transfer_out_texture(ctx, parcel);
        ctx.with_transient_pool(|pool| pool.adopt(stamped));
    }

    /// Release a held buffer into the context transient pool for epoch-gated reuse.
    pub fn release_buffer(&mut self, ctx: &Context, buffer: Buffer) {
        let stamped = self.transfer_out_buffer(ctx, buffer);
        ctx.with_transient_pool(|pool| pool.adopt(stamped));
    }

    pub(crate) fn transfer_out_texture(&mut self, ctx: &Context, mut parcel: Parcel) -> StampedParcel {
        if let Some(home) = parcel.home_device().upgrade() {
            debug_assert!(
                Arc::ptr_eq(&home, &ctx.device().inner),
                "transfer_out: parcel home_device must match submitting context's device"
            );
        }
        let ready_after = parcel.last_referenced();
        parcel.release_bookkeeping();
        StampedParcel {
            hold: RetainedHold::Texture(parcel),
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
        assert_eq!(b.kind(), ParcelType::Buffer);
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
        b.mark_referenced(ctx.backend_handle(), 42);
        let stamped = pool.transfer_out_buffer(&ctx, b);
        assert_eq!(stamped.ready_after.get(&ctx.backend_handle()), Some(&42));
        assert_eq!(pool.bytes_by_kind().buffer, 0);
    }

    #[test]
    fn write_parcel_on_whole_buffer_parcel_succeeds() {
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
        let mut graph = crate::TaskGraph::new();
        assert!(graph
            .write_parcel(&*buffer, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]).to_vec())
            .is_ok());
        graph.dispatch(&ctx).unwrap();
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
            .with_parcel(&parcel, crate::task_graph::NodeAccess::Read)
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
