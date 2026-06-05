//! Gate-free retained allocation pool — the public door for deed-held GPU memory.
//!
//! [`RetainedPool::acquire_texture`] and [`RetainedPool::acquire_buffer`] are the only
//! supported way to create retained parcels in this unit. Parcels are opaque [`Parcel`]
//! values; relinquish via [`Self::transfer_out`] or by dropping the parcel.
//!
//! Reuse-gate, transient pool, backpressure, and host-visible `copy_into` are deferred.

use crate::device::Device;
use crate::parcel::{BookkeepingGuard, BytesByKind, Parcel, PoolBookkeeping};
use crate::timeline::TimelineValue;
use crate::types::{DataAccess, SpatialAccess, TextureFlags, TextureFormat};
use crate::vram_allocator::ParcelKind;
use anyhow::Result;
use std::sync::Arc;

/// A parcel relinquished from the retained pool, stamped for handoff to the transient pool.
pub struct StampedParcel {
    pub parcel: Parcel,
    /// Timeline after which the parcel may be reused; `None` if never referenced by GPU work.
    pub ready_after: Option<TimelineValue>,
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
        access: SpatialAccess,
        flags: TextureFlags,
        init: Option<&[u8]>,
    ) -> Result<Parcel> {
        let tex = if let Some(data) = init {
            crate::texture::Texture::with_data(
                &self.device,
                data,
                width,
                height,
                format,
                access,
                flags,
            )?
        } else {
            self.device
                .alloc_texture(width, height, format, access, flags)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        self.wrap_texture(tex)
    }

    /// Allocate a retained buffer. `init: Some(data)` performs a one-shot staged upload.
    ///
    /// Permanently host-visible buffers and repeated CPU writes are deferred (Unit 2+).
    pub fn acquire_buffer(
        &mut self,
        size: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
        init: Option<&[u8]>,
    ) -> Result<Parcel> {
        let buf = if let Some(data) = init {
            self.device
                .alloc_buffer_with_bytes_stride_and_flags(
                    data,
                    access,
                    element_stride.unwrap_or(1),
                    flags,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            self.device
                .alloc_buffer(size, access, element_stride, flags)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        self.wrap_buffer(buf)
    }

    /// Relinquish a parcel from the retained pool. The unit moves to the transient seam;
    /// `ready_after` is the last referencing timeline (if any).
    pub fn transfer_out(&mut self, mut parcel: Parcel) -> StampedParcel {
        let ready_after = parcel.last_referenced();
        parcel.release_bookkeeping();
        StampedParcel {
            parcel,
            ready_after,
        }
    }

    /// Committed bytes currently held through this pool, by [`ParcelKind`].
    pub fn bytes_by_kind(&self) -> BytesByKind {
        self.bookkeeping.snapshot()
    }

    fn wrap_texture(&self, tex: crate::texture::Texture) -> Result<Parcel> {
        let bytes = tex.byte_size() as u64;
        let kind = ParcelKind::Texture;
        self.bookkeeping.add(kind, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.bookkeeping), kind, bytes);
        Ok(Parcel::from_texture(tex, guard))
    }

    fn wrap_buffer(&self, buf: crate::buffer::Buffer) -> Result<Parcel> {
        let bytes = buf.byte_size();
        let kind = ParcelKind::Buffer;
        self.bookkeeping.add(kind, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.bookkeeping), kind, bytes);
        Ok(Parcel::from_buffer(buf, guard))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::types::TextureFormat;

    fn test_device() -> Arc<Device> {
        Arc::new(
            Device::from_backend(Box::new(MockBackend::new())).expect("mock device"),
        )
    }

    fn rgba_interpolated() -> (TextureFormat, SpatialAccess, TextureFlags) {
        (
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Interpolated,
            TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
        )
    }

    #[test]
    fn acquire_texture_without_init_allocates() {
        let mut pool = RetainedPool::new(test_device());
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool
            .acquire_texture(64, 64, fmt, acc, flags, None)
            .unwrap();
        assert_eq!(p.kind(), ParcelKind::Texture);
        assert!(pool.bytes_by_kind().texture > 0);
        assert_eq!(pool.bytes_by_kind().buffer, 0);
    }

    #[test]
    fn acquire_texture_with_init_uploads() {
        let mut pool = RetainedPool::new(test_device());
        let (fmt, acc, flags) = rgba_interpolated();
        let data = vec![0u8; 32 * 32 * 4];
        let p = pool
            .acquire_texture(32, 32, fmt, acc, flags, Some(&data))
            .unwrap();
        assert_eq!(p.byte_size(), 32 * 32 * 4);
    }

    #[test]
    fn acquire_buffer_without_init_allocates() {
        let mut pool = RetainedPool::new(test_device());
        let p = pool
            .acquire_buffer(
                256,
                DataAccess::Scattered,
                None,
                crate::types::BufferFlags::empty(),
                None,
            )
            .unwrap();
        assert_eq!(p.kind(), ParcelKind::Buffer);
        assert_eq!(p.byte_size(), 256);
        assert!(pool.bytes_by_kind().buffer >= 256);
    }

    #[test]
    fn mark_referenced_is_monotonic_max() {
        let mut pool = RetainedPool::new(test_device());
        let (fmt, acc, flags) = rgba_interpolated();
        let mut p = pool
            .acquire_texture(8, 8, fmt, acc, flags, None)
            .unwrap();
        p.mark_referenced(10);
        p.mark_referenced(5);
        assert_eq!(p.last_referenced(), Some(10));
        p.mark_referenced(20);
        assert_eq!(p.last_referenced(), Some(20));
    }

    #[test]
    fn transfer_out_referenced_has_ready_after() {
        let mut pool = RetainedPool::new(test_device());
        let (fmt, acc, flags) = rgba_interpolated();
        let mut p = pool
            .acquire_texture(8, 8, fmt, acc, flags, None)
            .unwrap();
        p.mark_referenced(42);
        let stamped = pool.transfer_out(p);
        assert_eq!(stamped.ready_after, Some(42));
        assert_eq!(pool.bytes_by_kind().texture, 0);
    }

    #[test]
    fn transfer_out_unreferenced_has_none_ready_after() {
        let mut pool = RetainedPool::new(test_device());
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool
            .acquire_texture(8, 8, fmt, acc, flags, None)
            .unwrap();
        let stamped = pool.transfer_out(p);
        assert_eq!(stamped.ready_after, None);
    }

    #[test]
    fn transfer_out_preserves_texture_handle() {
        let mut pool = RetainedPool::new(test_device());
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool
            .acquire_texture(8, 8, fmt, acc, flags, None)
            .unwrap();
        let h_before = p.texture_handle().unwrap();
        let stamped = pool.transfer_out(p);
        assert_eq!(stamped.parcel.texture_handle(), Some(h_before));
    }

    #[test]
    fn bytes_by_kind_zero_after_transfer_and_drop() {
        let mut pool = RetainedPool::new(test_device());
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool
            .acquire_texture(16, 16, fmt, acc, flags, None)
            .unwrap();
        assert!(pool.bytes_by_kind().texture > 0);
        let stamped = pool.transfer_out(p);
        assert_eq!(pool.bytes_by_kind().texture, 0);
        drop(stamped);
    }

    #[test]
    fn bind_parcel_wires_parcel_resource_id() {
        use crate::compute::ComputePipeline;
        use crate::shader::ShaderModule;
        use crate::task_graph::{NodeAccess, ResourceId, TaskGraph};

        let device = test_device();
        let mut pool = RetainedPool::new(device.clone());
        let (fmt, acc, flags) = rgba_interpolated();
        let parcel = pool
            .acquire_texture(4, 4, fmt, acc, flags, None)
            .unwrap();
        let expected = parcel.resource_id();

        let shader = ShaderModule::from_slang(&device, "void main() {}").unwrap();
        let pipeline = ComputePipeline::new(&device, &shader).unwrap();

        let mut graph = TaskGraph::new();
        graph
            .node("a", &pipeline)
            .bind_parcel(&parcel, NodeAccess::Read)
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
