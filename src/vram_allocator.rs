//! Unified GPU memory allocation interface.
//!
//! Goldy has three independent GPU memory allocation paths:
//!
//! 1. **Transient sub-allocations** — [`TransientAllocator`] → [`BufferPool`] → [`Buffer::new`].
//!    Pluggable recycling policy via the [`TransientAllocator`] trait, but no control over
//!    *where* memory comes from.
//! 2. **Standalone named buffers** — consumers call [`Buffer::new`] directly for bump readback,
//!    staging, indirect dispatch, etc.
//! 3. **Textures** — [`TexturePool`] → [`Texture::new`]. No interception point.
//!
//! [`VramAllocator`] sits **below** all three pooling systems, providing a single customization
//! point for *where* GPU memory comes from. This enables:
//!
//! - **Unified memory control** — alias transient, standalone, and texture allocations into one
//!   address space or placement heap.
//! - **Backend-native strategies** — Metal `makeAliasable` placement heaps, Vulkan sparse
//!   binding, DX12 tiled resources as first-class allocator implementations.
//! - **Budgeting / telemetry** — VRAM caps, fragmentation monitoring, eviction policies.
//! - **Defragmentation** — move allocations and update bindless descriptors atomically.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────┐
//! │  Consumers (ekrano, user code)                 │
//! │  ┌─────────────┐  ┌──────────┐  ┌───────────┐ │
//! │  │ TransientAlloc│ │ Buffer:: │  │ Texture:: │ │
//! │  │ (recycling)  │  │ new()    │  │ new()     │ │
//! │  └──────┬───────┘  └────┬─────┘  └─────┬─────┘ │
//! │         │               │              │       │
//! │  ┌──────▼───────────────▼──────────────▼──────┐│
//! │  │           VramAllocator trait               ││
//! │  │  alloc_buffer / alloc_texture / free / ...  ││
//! │  └──────────────────┬──────────────────────────┘│
//! │                     │                           │
//! │  ┌──────────────────▼──────────────────────────┐│
//! │  │           GpuBackend (Metal/Vulkan/DX12)    ││
//! │  └─────────────────────────────────────────────┘│
//! └────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! The [`Device`] holds an [`Arc<dyn VramAllocator>`]. Call
//! [`Device::set_vram_allocator`] before creating any GPU resources to install
//! a custom allocator. The default ([`DefaultVramAllocator`]) delegates directly
//! to the backend with zero overhead.
//!
//! [`TransientAllocator`]: crate::transient_allocator::TransientAllocator
//! [`BufferPool`]: crate::buffer::BufferPool
//! [`Buffer::new`]: crate::buffer::Buffer::new
//! [`Texture::new`]: crate::texture::Texture::new
//! [`TexturePool`]: crate::texture_pool::TexturePool
//! [`Device`]: crate::device::Device

use crate::buffer::Buffer;
use crate::device::Device;
use crate::texture::Texture;
use crate::types::*;
use anyhow::Result;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

// -----------------------------------------------------------------------
// Trait
// -----------------------------------------------------------------------

/// A pluggable strategy for allocating GPU memory (buffers and textures).
///
/// Implementations intercept every buffer and texture allocation, enabling unified
/// memory budgets, placement-heap strategies, and telemetry without changing call sites.
///
/// Methods take `&self` and must be internally synchronized (the trait is `Send + Sync`).
/// Use [`AtomicI64`] / [`AtomicU64`](std::sync::atomic::AtomicU64) for lock-free counters,
/// or a `Mutex` for more complex state.
pub trait VramAllocator: Send + Sync {
    /// Allocate a GPU buffer.
    ///
    /// The default implementation calls [`Buffer::new_with_stride_and_flags`] directly.
    /// Custom implementations may allocate from a placement heap, enforce a budget, or
    /// track the allocation for telemetry.
    fn alloc_buffer(
        &self,
        device: &Device,
        size: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<Buffer> {
        Buffer::new_with_stride_and_flags(device, size, access, element_stride, flags)
    }

    /// Allocate a GPU buffer with a pre-reserved capacity hint.
    ///
    /// The default implementation calls [`Buffer::new_with_capacity_hint_and_flags`].
    fn alloc_buffer_with_capacity(
        &self,
        device: &Device,
        initial_size: u64,
        expected_max: u64,
        access: DataAccess,
        flags: BufferFlags,
    ) -> Result<Buffer> {
        Buffer::new_with_capacity_hint_and_flags(device, initial_size, expected_max, access, flags)
    }

    /// Allocate a GPU texture.
    ///
    /// The default implementation calls [`Texture::new`] directly.
    fn alloc_texture(
        &self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: SpatialAccess,
        flags: TextureFlags,
    ) -> Result<Texture> {
        Texture::new(device, width, height, format, access, flags)
    }

    /// Notify the allocator that a buffer has been freed.
    ///
    /// Called automatically by [`Buffer::drop`] when the allocator is installed on the
    /// device. Implementations should decrement their tracked byte counts here.
    /// The `size` is the buffer's allocated size at the time of destruction.
    fn notify_buffer_freed(&self, _size: u64) {}

    /// Notify the allocator that a texture has been freed.
    ///
    /// Called automatically by [`Texture::drop`] when the allocator is installed on the
    /// device. The `byte_size` is [`Texture::byte_size`] at the time of destruction.
    fn notify_texture_freed(&self, _byte_size: usize) {}

    /// Net bytes allocated by this allocator (allocations minus frees).
    ///
    /// Returns 0 if the implementation does not track allocations.
    fn allocated_bytes(&self) -> u64 {
        0
    }

    /// Optional byte budget. Returns `None` if no budget is enforced.
    /// When set, [`alloc_buffer`](Self::alloc_buffer) and
    /// [`alloc_texture`](Self::alloc_texture) should return an error if
    /// the allocation would exceed the budget.
    fn budget(&self) -> Option<u64> {
        None
    }

    /// Strategy identifier for diagnostics and tracing.
    fn name(&self) -> &'static str;
}

// -----------------------------------------------------------------------
// Default implementation
// -----------------------------------------------------------------------

/// The default allocator: delegates directly to [`Buffer::new`] / [`Texture::new`]
/// with no tracking, budgeting, or overhead.
///
/// Installed automatically when a [`Device`] is created.
pub struct DefaultVramAllocator;

impl VramAllocator for DefaultVramAllocator {
    fn name(&self) -> &'static str {
        "default"
    }
}

// -----------------------------------------------------------------------
// Tracking allocator
// -----------------------------------------------------------------------

/// A `VramAllocator` that wraps another allocator and tracks total allocated bytes.
///
/// Optionally enforces a byte budget: allocations that would push the total above
/// the budget return an error instead of proceeding.
///
/// # Example
///
/// ```no_run
/// # use goldy::vram_allocator::{TrackingVramAllocator, DefaultVramAllocator};
/// # use std::sync::Arc;
/// // Track all allocations with a 512 MB budget:
/// let allocator = TrackingVramAllocator::with_budget(
///     Arc::new(DefaultVramAllocator),
///     512 * 1024 * 1024,
/// );
/// ```
pub struct TrackingVramAllocator {
    inner: Arc<dyn VramAllocator>,
    /// Signed to handle potential over-decrement from mismatched free notifications
    /// without panicking. Steady-state value is non-negative.
    live_bytes: AtomicI64,
    budget_bytes: Option<u64>,
}

impl TrackingVramAllocator {
    /// Wrap `inner` with byte-level tracking but no budget.
    pub fn new(inner: Arc<dyn VramAllocator>) -> Self {
        Self {
            inner,
            live_bytes: AtomicI64::new(0),
            budget_bytes: None,
        }
    }

    /// Wrap `inner` with tracking and a byte budget.
    pub fn with_budget(inner: Arc<dyn VramAllocator>, budget_bytes: u64) -> Self {
        Self {
            inner,
            live_bytes: AtomicI64::new(0),
            budget_bytes: Some(budget_bytes),
        }
    }

    fn check_budget(&self, additional: u64) -> Result<()> {
        if let Some(cap) = self.budget_bytes {
            let current = self.live_bytes.load(Ordering::Relaxed) as u64;
            if current.saturating_add(additional) > cap {
                anyhow::bail!(
                    "VramAllocator budget exceeded: {current} + {additional} > {cap} \
                     (allocator={}, budget={})",
                    self.inner.name(),
                    bytesize(cap),
                );
            }
        }
        Ok(())
    }
}

impl VramAllocator for TrackingVramAllocator {
    fn alloc_buffer(
        &self,
        device: &Device,
        size: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<Buffer> {
        self.check_budget(size)?;
        let buf = self
            .inner
            .alloc_buffer(device, size, access, element_stride, flags)?;
        self.live_bytes
            .fetch_add(buf.allocated_size() as i64, Ordering::Relaxed);
        Ok(buf)
    }

    fn alloc_buffer_with_capacity(
        &self,
        device: &Device,
        initial_size: u64,
        expected_max: u64,
        access: DataAccess,
        flags: BufferFlags,
    ) -> Result<Buffer> {
        self.check_budget(expected_max.max(initial_size))?;
        let buf = self.inner.alloc_buffer_with_capacity(
            device,
            initial_size,
            expected_max,
            access,
            flags,
        )?;
        self.live_bytes
            .fetch_add(buf.allocated_size() as i64, Ordering::Relaxed);
        Ok(buf)
    }

    fn alloc_texture(
        &self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: SpatialAccess,
        flags: TextureFlags,
    ) -> Result<Texture> {
        let estimated = (width as u64) * (height as u64) * (format.bytes_per_pixel() as u64);
        self.check_budget(estimated)?;
        let tex = self
            .inner
            .alloc_texture(device, width, height, format, access, flags)?;
        self.live_bytes
            .fetch_add(tex.byte_size() as i64, Ordering::Relaxed);
        Ok(tex)
    }

    fn notify_buffer_freed(&self, size: u64) {
        self.live_bytes.fetch_sub(size as i64, Ordering::Relaxed);
        self.inner.notify_buffer_freed(size);
    }

    fn notify_texture_freed(&self, byte_size: usize) {
        self.live_bytes
            .fetch_sub(byte_size as i64, Ordering::Relaxed);
        self.inner.notify_texture_freed(byte_size);
    }

    fn allocated_bytes(&self) -> u64 {
        self.live_bytes.load(Ordering::Relaxed).max(0) as u64
    }

    fn budget(&self) -> Option<u64> {
        self.budget_bytes
    }

    fn name(&self) -> &'static str {
        "tracking"
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn bytesize(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn default_allocator_creates_buffer() {
        let device = test_device();
        let alloc = DefaultVramAllocator;
        let buf = alloc
            .alloc_buffer(
                &device,
                1024,
                DataAccess::Scattered,
                None,
                BufferFlags::empty(),
            )
            .unwrap();
        assert_eq!(buf.size(), 1024);
    }

    #[test]
    fn default_allocator_creates_texture() {
        let device = test_device();
        let alloc = DefaultVramAllocator;
        let tex = alloc
            .alloc_texture(
                &device,
                64,
                64,
                TextureFormat::Rgba8Unorm,
                SpatialAccess::Interpolated,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .unwrap();
        assert_eq!(tex.width(), 64);
        assert_eq!(tex.height(), 64);
    }

    #[test]
    fn tracking_allocator_tracks_bytes() {
        let device = test_device();
        let alloc = TrackingVramAllocator::new(Arc::new(DefaultVramAllocator));

        assert_eq!(alloc.allocated_bytes(), 0);

        let buf = alloc
            .alloc_buffer(
                &device,
                4096,
                DataAccess::Scattered,
                None,
                BufferFlags::empty(),
            )
            .unwrap();
        assert!(alloc.allocated_bytes() >= 4096);

        let size = buf.allocated_size();
        drop(buf);
        alloc.notify_buffer_freed(size);
        assert_eq!(alloc.allocated_bytes(), 0);
    }

    #[test]
    fn tracking_allocator_budget_enforcement() {
        let device = test_device();
        let alloc = TrackingVramAllocator::with_budget(Arc::new(DefaultVramAllocator), 8192);

        let _buf = alloc
            .alloc_buffer(
                &device,
                4096,
                DataAccess::Scattered,
                None,
                BufferFlags::empty(),
            )
            .unwrap();

        let result = alloc.alloc_buffer(
            &device,
            8192,
            DataAccess::Scattered,
            None,
            BufferFlags::empty(),
        );
        assert!(result.is_err(), "should fail when over budget");
    }

    #[test]
    fn tracking_allocator_texture_tracking() {
        let device = test_device();
        let alloc = TrackingVramAllocator::new(Arc::new(DefaultVramAllocator));

        let tex = alloc
            .alloc_texture(
                &device,
                32,
                32,
                TextureFormat::Rgba8Unorm,
                SpatialAccess::Interpolated,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .unwrap();
        let byte_size = tex.byte_size();
        assert!(alloc.allocated_bytes() > 0);

        drop(tex);
        alloc.notify_texture_freed(byte_size);
        assert_eq!(alloc.allocated_bytes(), 0);
    }

    #[test]
    fn bytesize_formatting() {
        assert_eq!(bytesize(500), "500 B");
        assert_eq!(bytesize(1024), "1.0 KiB");
        assert_eq!(bytesize(1024 * 1024), "1.0 MiB");
        assert_eq!(bytesize(1024 * 1024 * 1024), "1.0 GiB");
    }
}
