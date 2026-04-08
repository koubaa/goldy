//! GPU buffer management.

use crate::backend::{BufferHandle, GpuBackend};
use crate::device::Device;
use crate::types::DataAccess;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A GPU buffer.
pub struct Buffer {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: BufferHandle,
    size: u64,
    access: DataAccess,
}

impl Buffer {
    /// Create a new buffer with the specified access pattern.
    ///
    /// # Access Patterns
    ///
    /// - `DataAccess::Scattered`: Any thread can access any address (read/write).
    ///   Use for general-purpose data (StructuredBuffer, RWStructuredBuffer).
    ///
    /// - `DataAccess::Broadcast`: All threads read the same address.
    ///   Hardware optimizes for wave-wide broadcast (ConstantBuffer).
    pub fn new(device: &Device, size: u64, access: DataAccess) -> Result<Self> {
        Self::new_with_stride(device, size, access, None)
    }

    pub fn new_with_stride(
        device: &Device,
        size: u64,
        access: DataAccess,
        element_stride: Option<u32>,
    ) -> Result<Self> {
        tracing::debug!(size, ?access, element_stride, "Creating buffer");
        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_buffer(device.handle, size, access, element_stride)?;

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
            size,
            access,
        })
    }

    /// Create a buffer initialized with data.
    ///
    /// See [`Buffer::new`] for access pattern documentation.
    pub fn with_data<T: bytemuck::Pod>(
        device: &Device,
        data: &[T],
        access: DataAccess,
    ) -> Result<Self> {
        let bytes = bytemuck::cast_slice(data);
        let element_stride = std::mem::size_of::<T>() as u32;
        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_buffer(
            device.handle,
            bytes.len() as u64,
            access,
            Some(element_stride),
        )?;
        drop(backend);

        let buffer = Self {
            backend: Arc::clone(&device.backend),
            handle,
            size: bytes.len() as u64,
            access,
        };
        buffer.write(0, bytes)?;
        Ok(buffer)
    }

    /// Create a buffer initialized with raw bytes.
    ///
    /// See [`Buffer::new`] for access pattern documentation.
    pub fn with_bytes(device: &Device, data: &[u8], access: DataAccess) -> Result<Self> {
        // For raw bytes, use stride of 1 (byte-addressable)
        Self::with_bytes_stride(device, data, access, 1)
    }

    /// Create a buffer initialized with raw bytes and a custom element stride.
    ///
    /// The stride is used for creating StructuredBuffer views on DX12. For example,
    /// if the data contains u32 values, use stride=4 so the GPU can correctly
    /// interpret the buffer as `StructuredBuffer<uint>`.
    ///
    /// See [`Buffer::new`] for access pattern documentation.
    pub fn with_bytes_stride(
        device: &Device,
        data: &[u8],
        access: DataAccess,
        element_stride: u32,
    ) -> Result<Self> {
        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_buffer(
            device.handle,
            data.len() as u64,
            access,
            Some(element_stride),
        )?;
        drop(backend);

        let buffer = Self {
            backend: Arc::clone(&device.backend),
            handle,
            size: data.len() as u64,
            access,
        };
        buffer.write(0, data)?;
        Ok(buffer)
    }

    /// Write data to the buffer.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let mut backend = self.backend.lock().unwrap();
        backend.write_buffer(self.handle, offset, data)
    }

    /// Write typed data to the buffer.
    pub fn write_data<T: bytemuck::Pod>(&self, offset: u64, data: &[T]) -> Result<()> {
        self.write(offset, bytemuck::cast_slice(data))
    }

    /// Get the buffer size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the buffer's access pattern.
    pub fn access(&self) -> DataAccess {
        self.access
    }

    /// Get the buffer's index in the global bindless descriptor set.
    ///
    /// Returns `Some(index)` if this buffer is registered in the global descriptor set.
    /// All buffers with Scattered or Broadcast access are registered.
    pub fn bindless_index(&self) -> Option<u32> {
        let backend = self.backend.lock().unwrap();
        backend.buffer_bindless_index(self.handle)
    }

    /// Get the buffer's SRV (read-only) bindless index for `StructuredBuffer<T>` / `goldy_dyn_buf_ro` access.
    ///
    /// On DX12, scattered buffers have separate UAV (write) and SRV (read-only) descriptors at
    /// different heap indices. Use this index when the shader declares `StructuredBuffer<T>`.
    /// On Vulkan and Metal, returns the same value as `bindless_index()`.
    pub fn bindless_srv_index(&self) -> Option<u32> {
        let backend = self.backend.lock().unwrap();
        backend.buffer_bindless_srv_index(self.handle)
    }

    /// Read buffer contents back to CPU memory.
    ///
    /// The `output` slice must be at least `size` bytes. Reads from offset 0.
    pub fn read_to_cpu(&self, device: &Device, output: &mut [u8]) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.read_buffer_to_cpu(device.handle, self.handle, output)
    }

    /// Clear the buffer (fill with zeros) from offset for size bytes.
    pub fn clear(&self, device: &Device, offset: u64, size: u64) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.clear_buffer(device.handle, self.handle, offset, size)
    }

    /// Create a view into a sub-region of this buffer.
    ///
    /// The view gets its own bindless descriptor index, so shaders see a zero-based
    /// buffer starting at `offset`. Multiple views of the same buffer can be bound
    /// simultaneously to different push constant slots.
    ///
    /// `element_stride` sets the structured buffer stride for the view's descriptor.
    /// If `None`, defaults to 4 bytes (u32).
    pub fn create_view(
        &self,
        offset: u64,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferView> {
        let mut backend = self.backend.lock().unwrap();
        let handle = backend.create_buffer_view(self.handle, offset, size, element_stride)?;
        Ok(BufferView {
            backend: Arc::clone(&self.backend),
            handle,
            parent_handle: self.handle,
            offset,
            size,
        })
    }

    /// Create a typed view into a sub-region of this buffer.
    ///
    /// Convenience wrapper that computes the byte offset, byte size, and element stride
    /// from the type `T` and element count.
    pub fn create_typed_view<T: bytemuck::Pod>(
        &self,
        first_element: u64,
        count: u64,
    ) -> Result<BufferView> {
        let stride = std::mem::size_of::<T>() as u64;
        let offset = first_element * stride;
        let size = count * stride;
        self.create_view(offset, size, Some(stride as u32))
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        tracing::trace!(size = self.size, access = ?self.access, "Destroying buffer");
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_buffer(self.handle);
    }
}

/// Trait for types that can be bound as vertex or index buffers.
///
/// Both [`Buffer`] and [`BufferView`] implement this trait, allowing either to be passed
/// to `set_vertex_buffer` and `set_index_buffer`. For `BufferView`, the encoder binds
/// the parent buffer at the view's offset internally.
pub trait BufferSource {
    #[doc(hidden)]
    fn source_handle(&self) -> BufferHandle;
    #[doc(hidden)]
    fn source_offset(&self) -> u64;
}

impl BufferSource for Buffer {
    fn source_handle(&self) -> BufferHandle {
        self.handle
    }
    fn source_offset(&self) -> u64 {
        0
    }
}

/// A view into a sub-region of a [`Buffer`].
///
/// A `BufferView` shares the parent buffer's GPU memory but gets its own bindless
/// descriptor pointing at `[offset, offset+size)`. The shader sees the sub-region
/// as a zero-based buffer.
///
/// This enables buffer pooling: allocate one large buffer and create views for
/// each logical sub-allocation. Each view can be independently bound via push constants.
///
/// Dropping a `BufferView` unregisters its descriptor but does not free the parent's memory.
pub struct BufferView {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: BufferHandle,
    parent_handle: BufferHandle,
    offset: u64,
    size: u64,
}

impl BufferView {
    /// Get the view's index in the global bindless descriptor set.
    pub fn bindless_index(&self) -> Option<u32> {
        let backend = self.backend.lock().unwrap();
        backend.buffer_bindless_index(self.handle)
    }

    /// Get the view's SRV (read-only) bindless index.
    pub fn bindless_srv_index(&self) -> Option<u32> {
        let backend = self.backend.lock().unwrap();
        backend.buffer_bindless_srv_index(self.handle)
    }

    /// Get the view's offset within the parent buffer in bytes.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Get the view size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Write typed data into this view's region of the parent buffer.
    ///
    /// Writes starting at the view's offset. The data must fit within the view's size.
    pub fn write_data<T: bytemuck::Pod>(&self, data: &[T]) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);
        if bytes.len() as u64 > self.size {
            anyhow::bail!(
                "BufferView write overflow: {} bytes would exceed view size of {}",
                bytes.len(),
                self.size
            );
        }
        let mut backend = self.backend.lock().unwrap();
        backend.write_buffer(self.parent_handle, self.offset, bytes)
    }
}

impl BufferSource for BufferView {
    fn source_handle(&self) -> BufferHandle {
        self.parent_handle
    }
    fn source_offset(&self) -> u64 {
        self.offset
    }
}

impl Drop for BufferView {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_buffer(self.handle);
    }
}

/// GCD for alignment computation. Returns 0 if both are 0.
fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// LCM for alignment: smallest value divisible by both a and b.
fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a * b / gcd(a, b)
}

/// A GPU buffer pool that sub-allocates views from a single large buffer.
///
/// Instead of allocating many small buffers (each a separate GPU allocation),
/// create one pool and carve out typed regions. Each region gets its own
/// bindless descriptor so shaders see independent zero-based buffers.
///
/// # Example
///
/// ```rust,no_run
/// use goldy::{Instance, DeviceType, BufferPool};
///
/// let instance = Instance::new()?;
/// let device = instance.create_device(DeviceType::DiscreteGpu)?;
///
/// let mut pool = BufferPool::new(&device, 1024 * 1024)?; // 1 MB pool
///
/// let tiles = pool.alloc::<[u32; 2]>(1024)?;   // 8 KB for 1024 tiles
/// let segments = pool.alloc::<[f32; 6]>(4096)?; // 96 KB for 4096 segments
///
/// // Use views via bindless indices
/// let tile_idx = tiles.bindless_index().unwrap();
/// let seg_idx = segments.bindless_index().unwrap();
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct BufferPool {
    backing: Buffer,
    offset: u64,
    alignment: u64,
}

impl BufferPool {
    /// Compute the total pool size needed for a set of allocations.
    ///
    /// Takes `(element_count, element_size)` pairs and returns the exact byte size
    /// required, including alignment padding. Use with [`BufferPool::new`] to avoid
    /// magic padding constants.
    pub fn padded_size(allocs: &[(usize, usize)]) -> u64 {
        const ALIGNMENT: u64 = 256;
        let mut offset = 0u64;
        for &(count, stride) in allocs {
            let stride = stride as u64;
            let alloc_align = lcm(ALIGNMENT, stride);
            let aligned_offset = offset.div_ceil(alloc_align) * alloc_align;
            let size = (count as u64) * stride;
            offset = aligned_offset + size;
        }
        offset
    }

    /// Create a new buffer pool with the given total size.
    ///
    /// The backing buffer is allocated as `DataAccess::Scattered` (storage buffer)
    /// since sub-allocation only makes sense for storage buffers.
    ///
    /// `alignment` defaults to 256 bytes, which satisfies `minStorageBufferOffsetAlignment`
    /// on all known Vulkan/DX12 hardware.
    pub fn new(device: &Device, total_size: u64) -> Result<Self> {
        Self::with_alignment(device, total_size, 256)
    }

    /// Create a pool with a custom sub-allocation alignment.
    pub fn with_alignment(device: &Device, total_size: u64, alignment: u64) -> Result<Self> {
        assert!(
            alignment.is_power_of_two(),
            "alignment must be a power of two"
        );
        tracing::debug!(total_size, alignment, "Creating buffer pool");
        let backing = Buffer::new(device, total_size, DataAccess::Scattered)?;
        Ok(Self {
            backing,
            offset: 0,
            alignment,
        })
    }

    /// Allocate a typed region from the pool.
    ///
    /// Returns a `BufferView` spanning `count` elements of type `T`, with the offset
    /// aligned to the pool's alignment requirement.
    pub fn alloc<T: bytemuck::Pod>(&mut self, count: u64) -> Result<BufferView> {
        let stride = std::mem::size_of::<T>() as u64;
        let size = count * stride;
        self.alloc_bytes(size, Some(stride as u32))
    }

    /// Allocate and fill a typed region in one call.
    ///
    /// Equivalent to `alloc::<T>(data.len())` followed by `write_data(data)`.
    /// Matches the ergonomics of [`Buffer::with_data`].
    pub fn alloc_with_data<T: bytemuck::Pod>(&mut self, data: &[T]) -> Result<BufferView> {
        let view = self.alloc::<T>(data.len() as u64)?;
        view.write_data(data)?;
        Ok(view)
    }

    /// Allocate a raw byte region from the pool.
    ///
    /// `element_stride` determines the structured buffer stride for the view's descriptor.
    /// If `None`, defaults to 4 bytes (u32).
    ///
    /// Each allocation is aligned to satisfy both pool alignment (256) and
    /// `offset % element_stride == 0` (required by DX12 StructuredBuffer views).
    pub fn alloc_bytes(&mut self, size: u64, element_stride: Option<u32>) -> Result<BufferView> {
        let stride = element_stride.unwrap_or(4) as u64;
        let alloc_align = lcm(self.alignment, stride);
        let aligned_offset = self.offset.div_ceil(alloc_align) * alloc_align;

        if aligned_offset + size > self.backing.size() {
            anyhow::bail!(
                "BufferPool exhausted: need {} bytes at offset {}, pool size is {}",
                size,
                aligned_offset,
                self.backing.size()
            );
        }

        let view = self
            .backing
            .create_view(aligned_offset, size, element_stride)?;
        self.offset = aligned_offset + size;
        Ok(view)
    }

    /// Reset the pool allocator to the beginning.
    ///
    /// This does NOT invalidate existing views — their descriptors still point at the
    /// correct memory. Use this for frame-to-frame reuse where you know previous
    /// views are no longer in flight.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Bytes currently allocated from the pool.
    pub fn used(&self) -> u64 {
        self.offset
    }

    /// Total pool capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.backing.size()
    }

    /// Remaining bytes available for allocation.
    pub fn remaining(&self) -> u64 {
        self.backing.size().saturating_sub(self.offset)
    }

    /// Get a reference to the backing buffer (e.g., for bulk writes or clears).
    pub fn backing_buffer(&self) -> &Buffer {
        &self.backing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_padded_size_empty() {
        assert_eq!(BufferPool::padded_size(&[]), 0);
    }

    #[test]
    fn test_padded_size_single_allocation() {
        // 64 u32s = 256 bytes, aligned to 256, no padding
        assert_eq!(BufferPool::padded_size(&[(64, size_of::<u32>())]), 256);
    }

    #[test]
    fn test_padded_size_multiple_allocations() {
        // Simulates goldy-doom: static_vb, static_ib, sky_vb, sky_ib, decor_vb, decor_ib
        // With varying strides, alignment padding is inserted between allocs
        let size = BufferPool::padded_size(&[
            (100, size_of::<u32>()), // 400 bytes
            (200, size_of::<u32>()), // 800 bytes
            (50, 52),                // SpriteVertex-like stride
            (75, 52),
        ]);
        assert!(
            size > 400 + 800 + 50 * 52 + 75 * 52,
            "padded_size should exceed raw sum"
        );
        assert!(
            size < 400 + 800 + 50 * 52 + 75 * 52 + 4 * 8192,
            "padded_size should be tighter than naive + magic constant"
        );
    }
}
