//! Explicit texture object pool for reuse across frames.
//!
//! On backends where texture allocation is expensive (notably Direct3D 12),
//! destroying and recreating many transient textures each frame causes VRAM churn
//! and heap fragmentation. [`TexturePool`] keeps released [`crate::texture::Texture`]
//! values alive so [`TexturePool::acquire`] can reuse GPU resources instead of
//! allocating fresh ones each time.
//!
//! **Semantics:** [`Texture::drop`] always destroys owned textures. Pooling is
//! opt-in: call [`TexturePool::release`] only after GPU work using the texture
//! has completed (e.g. after [`Context::wait_until`](crate::Context::wait_until) with the timeline from [`TaskGraph::submit`](crate::task_graph::TaskGraph::submit)).

use crate::device::Device;
use crate::texture::Texture;
use crate::types::{TextureFlags, TextureFormat, TextureKind};
use anyhow::Result;
use std::collections::HashMap;

/// Configuration for [`TexturePool`].
#[derive(Debug, Clone)]
pub struct TexturePoolConfig {
    /// Maximum pooled textures per `(width, height, format, access, flags)` key.
    /// Additional textures passed to [`TexturePool::release`] are dropped (destroyed).
    pub max_per_key: usize,
}

impl Default for TexturePoolConfig {
    fn default() -> Self {
        Self { max_per_key: 8 }
    }
}

/// Snapshot of pooled texture counts and approximate memory held.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TexturePoolStats {
    /// Number of textures sitting in the pool (not yet re-acquired).
    pub entries: usize,
    /// Sum of [`Texture::byte_size`] for pooled textures (approximate VRAM).
    pub estimated_bytes: usize,
}

type PoolKey = (u32, u32, TextureFormat, TextureKind, TextureFlags);

/// User-managed pool of textures for reuse.
///
/// Unlike an implicit backend cache, the pool is explicit: acquire from the pool,
/// release back when done, query stats, or clear to free GPU memory immediately.
pub struct TexturePool {
    map: HashMap<PoolKey, Vec<Texture>>,
    config: TexturePoolConfig,
}

impl TexturePool {
    /// Create an empty pool with the given limits.
    pub fn new(config: TexturePoolConfig) -> Self {
        Self {
            map: HashMap::new(),
            config,
        }
    }

    /// Take a pooled texture matching the key, or create a new one through [`Device::alloc_texture`](crate::Device::alloc_texture).
    pub fn acquire(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<Texture> {
        let key = (width, height, format, access, flags);
        if let Some(v) = self.map.get_mut(&key) {
            if let Some(tex) = v.pop() {
                if v.is_empty() {
                    self.map.remove(&key);
                }
                return Ok(tex);
            }
        }
        device.alloc_texture(width, height, format, access, flags)
    }

    /// Return an owned texture to the pool after the GPU has finished using it.
    ///
    /// Borrowed textures ([`Texture::borrow`]) are dropped immediately and are not pooled.
    ///
    /// If the pool already holds [`TexturePoolConfig::max_per_key`] textures for this
    /// texture's key, `tex` is dropped (destroyed) instead.
    pub fn release(&mut self, tex: Texture) {
        if !tex.is_owned() {
            return;
        }
        let key = (
            tex.width(),
            tex.height(),
            tex.format(),
            tex.access(),
            tex.flags(),
        );
        let slot = self.map.entry(key).or_default();
        if slot.len() >= self.config.max_per_key {
            drop(tex);
            return;
        }
        slot.push(tex);
    }

    /// Returns pooled entry count and summed [`Texture::byte_size`] for pooled textures.
    pub fn stats(&self) -> TexturePoolStats {
        let mut entries = 0usize;
        let mut estimated_bytes = 0usize;
        for textures in self.map.values() {
            for tex in textures {
                entries += 1;
                estimated_bytes += tex.byte_size();
            }
        }
        TexturePoolStats {
            entries,
            estimated_bytes,
        }
    }

    /// Drop all pooled textures immediately (frees GPU memory for pooled entries).
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

impl Default for TexturePool {
    fn default() -> Self {
        Self::new(TexturePoolConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::device::Device;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn rgba_interpolated() -> (TextureFormat, TextureKind, TextureFlags) {
        (
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
        )
    }

    // -----------------------------------------------------------------------
    // Acquire / release basics
    // -----------------------------------------------------------------------

    /// Acquiring from an empty pool always creates a fresh texture.
    #[test]
    fn acquire_empty_pool_creates_fresh() {
        let device = test_device();
        let mut pool = TexturePool::default();
        let (fmt, acc, flags) = rgba_interpolated();

        let tex = pool.acquire(&device, 64, 64, fmt, acc, flags).unwrap();
        assert_eq!(tex.width(), 64);
        assert_eq!(tex.height(), 64);
        assert_eq!(tex.format(), fmt);
        assert_eq!(tex.access(), acc);
        assert_eq!(tex.flags(), flags);
        assert!(tex.is_owned());
    }

    /// After release, the next acquire returns the same underlying bindless handle
    /// (i.e. the same GPU resource, not a new allocation).
    #[test]
    fn released_texture_is_reused() {
        let device = test_device();
        let mut pool = TexturePool::default();
        let (fmt, acc, flags) = rgba_interpolated();

        let tex = pool.acquire(&device, 32, 32, fmt, acc, flags).unwrap();
        let handle_before = tex.gpu_handle();
        pool.release(tex);

        let tex2 = pool.acquire(&device, 32, 32, fmt, acc, flags).unwrap();
        assert_eq!(
            tex2.gpu_handle(),
            handle_before,
            "pool should return the same GPU resource on reuse"
        );
    }

    /// Releasing a texture and acquiring again empties the pool slot.
    #[test]
    fn pool_is_empty_after_reacquire() {
        let device = test_device();
        let mut pool = TexturePool::default();
        let (fmt, acc, flags) = rgba_interpolated();

        let tex = pool.acquire(&device, 8, 8, fmt, acc, flags).unwrap();
        pool.release(tex);
        assert_eq!(pool.stats().entries, 1);

        let _tex2 = pool.acquire(&device, 8, 8, fmt, acc, flags).unwrap();
        assert_eq!(
            pool.stats().entries,
            0,
            "slot should be empty after re-acquire"
        );
    }

    // -----------------------------------------------------------------------
    // Key discrimination
    // -----------------------------------------------------------------------

    /// Textures with different dimensions must NOT be mixed.
    #[test]
    fn different_dimensions_use_different_slots() {
        let device = test_device();
        let mut pool = TexturePool::default();
        let (fmt, acc, flags) = rgba_interpolated();

        let t64 = pool.acquire(&device, 64, 64, fmt, acc, flags).unwrap();
        let handle_64 = t64.gpu_handle();
        pool.release(t64);

        let t128 = pool.acquire(&device, 128, 128, fmt, acc, flags).unwrap();
        assert_ne!(
            t128.gpu_handle(),
            handle_64,
            "128x128 acquire should not reuse a 64x64 resource"
        );
        // The 64x64 texture is still pooled.
        assert_eq!(pool.stats().entries, 1);
        pool.release(t128);
    }

    /// Textures with different access patterns must NOT be mixed.
    #[test]
    fn different_access_uses_different_slots() {
        let device = test_device();
        let mut pool = TexturePool::default();
        let (fmt, _, flags) = rgba_interpolated();

        let interp = pool
            .acquire(&device, 16, 16, fmt, TextureKind::Interpolated, flags)
            .unwrap();
        let handle_interp = interp.gpu_handle();
        pool.release(interp);

        let direct = pool
            .acquire(&device, 16, 16, fmt, TextureKind::Direct, flags)
            .unwrap();
        assert_ne!(
            direct.gpu_handle(),
            handle_interp,
            "Direct access acquire should not reuse an Interpolated resource"
        );
        pool.release(direct);
    }

    /// Textures with different flags must NOT be mixed.
    #[test]
    fn different_flags_use_different_slots() {
        let device = test_device();
        let mut pool = TexturePool::default();
        let (fmt, acc, _) = rgba_interpolated();

        let src_only = pool
            .acquire(&device, 16, 16, fmt, acc, TextureFlags::COPY_SRC)
            .unwrap();
        let handle_src = src_only.gpu_handle();
        pool.release(src_only);

        let dst_only = pool
            .acquire(&device, 16, 16, fmt, acc, TextureFlags::COPY_DST)
            .unwrap();
        assert_ne!(
            dst_only.gpu_handle(),
            handle_src,
            "COPY_DST acquire must not reuse a COPY_SRC-only resource"
        );
        pool.release(dst_only);
    }

    // -----------------------------------------------------------------------
    // max_per_key eviction
    // -----------------------------------------------------------------------

    /// Releasing more textures than `max_per_key` destroys the excess immediately.
    #[test]
    fn excess_textures_are_dropped_not_pooled() {
        let device = test_device();
        let config = TexturePoolConfig { max_per_key: 2 };
        let mut pool = TexturePool::new(config);
        let (fmt, acc, flags) = rgba_interpolated();

        let t1 = pool.acquire(&device, 8, 8, fmt, acc, flags).unwrap();
        let t2 = pool.acquire(&device, 8, 8, fmt, acc, flags).unwrap();
        let t3 = pool.acquire(&device, 8, 8, fmt, acc, flags).unwrap();

        pool.release(t1);
        pool.release(t2);
        assert_eq!(pool.stats().entries, 2);

        // Releasing a third should drop it immediately, not push into the pool.
        pool.release(t3);
        assert_eq!(
            pool.stats().entries,
            2,
            "pool should not grow beyond max_per_key"
        );
    }

    // -----------------------------------------------------------------------
    // Borrowed textures are not pooled
    // -----------------------------------------------------------------------

    /// Borrowed textures (`is_owned() == false`) must be silently dropped on release,
    /// not retained in the pool.
    #[test]
    fn borrowed_texture_is_not_pooled() {
        let device = test_device();
        let mut pool = TexturePool::default();
        let (fmt, acc, flags) = rgba_interpolated();

        let owned = pool.acquire(&device, 16, 16, fmt, acc, flags).unwrap();
        let borrow = owned.borrow();
        assert!(!borrow.is_owned());

        pool.release(borrow);
        assert_eq!(
            pool.stats().entries,
            0,
            "borrowed texture must not be pooled"
        );

        // Still need to clean up the owned texture.
        pool.release(owned);
    }

    // -----------------------------------------------------------------------
    // stats
    // -----------------------------------------------------------------------

    /// `stats().estimated_bytes` is the sum of byte_size() across pooled entries.
    #[test]
    fn stats_reflect_pooled_entries() {
        let device = test_device();
        let mut pool = TexturePool::default();
        let (fmt, acc, flags) = rgba_interpolated();

        let t1 = pool.acquire(&device, 16, 16, fmt, acc, flags).unwrap();
        let t2 = pool.acquire(&device, 32, 32, fmt, acc, flags).unwrap();
        pool.release(t1);
        pool.release(t2);

        let stats = pool.stats();
        assert_eq!(stats.entries, 2);
        // 16x16x4 + 32x32x4
        assert_eq!(stats.estimated_bytes, 16 * 16 * 4 + 32 * 32 * 4);
    }

    /// An empty pool reports zero entries and zero bytes.
    #[test]
    fn stats_empty_pool() {
        let pool = TexturePool::default();
        let stats = pool.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.estimated_bytes, 0);
    }

    // -----------------------------------------------------------------------
    // clear
    // -----------------------------------------------------------------------

    /// `clear()` drops all pooled textures and resets stats to zero.
    #[test]
    fn clear_drops_all_entries() {
        let device = test_device();
        let mut pool = TexturePool::default();
        let (fmt, acc, flags) = rgba_interpolated();

        // Acquire all four first so we accumulate four distinct GPU resources,
        // then release them all to fill the pool to 4 entries.
        let textures: Vec<_> = (0..4)
            .map(|_| pool.acquire(&device, 8, 8, fmt, acc, flags).unwrap())
            .collect();
        for tex in textures {
            pool.release(tex);
        }
        assert_eq!(pool.stats().entries, 4);

        pool.clear();
        assert_eq!(pool.stats().entries, 0);
        assert_eq!(pool.stats().estimated_bytes, 0);
    }

    // -----------------------------------------------------------------------
    // Multiple keys coexist
    // -----------------------------------------------------------------------

    /// The pool can hold entries for several distinct keys simultaneously.
    #[test]
    fn multiple_keys_coexist() {
        let device = test_device();
        let mut pool = TexturePool::default();
        let (fmt, acc, flags) = rgba_interpolated();

        let sizes = [(4u32, 4u32), (8, 8), (16, 16)];
        for &(w, h) in &sizes {
            let tex = pool.acquire(&device, w, h, fmt, acc, flags).unwrap();
            pool.release(tex);
        }

        assert_eq!(pool.stats().entries, 3);

        for &(w, h) in &sizes {
            let tex = pool.acquire(&device, w, h, fmt, acc, flags).unwrap();
            assert_eq!(tex.width(), w);
            assert_eq!(tex.height(), h);
        }
        assert_eq!(pool.stats().entries, 0);
    }
}
