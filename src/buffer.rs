//! GPU buffer management.

use crate::backend::{BufferHandle, GpuBackend};
use crate::device::Device;
use crate::types::{BufferFlags, BufferKind, ResourceAccess, ResourceCategory, ResourceHandle};
use crate::vram_allocator::{ParcelDeed, ParcelType};
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Types allowed as elements in [`RetainedPool::acquire_buffer_with_data`](crate::RetainedPool::acquire_buffer_with_data)
/// and [`BufferPool::alloc_with_data`].
///
/// This is implemented for common multi-byte primitives, arrays of those types, and
/// `#[repr(C)]` structs via `#[derive(goldy_derive::StructuredBufferElement)]`.
///
/// **Not** implemented for `u8` / `i8`: passing `&[u8]` (e.g. from `bytemuck::bytes_of`) would
/// set element stride to 1 while shaders usually expect a larger struct stride. Use
/// [`RetainedPool::acquire_buffer`](crate::RetainedPool::acquire_buffer) with an explicit
/// element stride or a typed slice instead.
///
/// Unit type `()` is included so empty slices type-check.
pub trait StructuredBufferElement: bytemuck::Pod {}

macro_rules! impl_structured_buffer_element_for_primitives {
    ($($t:ty),+ $(,)?) => {
        $(impl StructuredBufferElement for $t {})+
    };
}

impl_structured_buffer_element_for_primitives!((), i16, u16, i32, u32, i64, u64, i128, u128, isize, usize, f32, f64,);

impl<T: StructuredBufferElement, const N: usize> StructuredBufferElement for [T; N] where [T; N]: bytemuck::Pod {}

/// Low-level GPU buffer allocation.
pub(crate) struct Allocation {
    device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: BufferHandle,
    /// Logical byte size (API-facing; may be smaller than reserved GPU storage).
    size: u64,
    /// Reserved byte size (`MTLBuffer.length` / Vulkan allocation / …); >= [`Self::size`].
    allocated_size: u64,
    access: BufferKind,
    element_stride: Option<u32>,
    flags: BufferFlags,
    /// Peak `allocated_size` ever observed on this buffer (telemetry; for profiling/tuning hints).
    peak_committed_bytes: u64,
    /// Number of completed [`Self::resize_to`] / [`Self::resize_to_uninitialized`] calls.
    resize_count: u32,
    /// Accounting deed for observer + allocator notification on drop.
    deed: Option<ParcelDeed>,
}

#[allow(dead_code)]
impl Allocation {
    #[inline]
    pub(crate) fn gpu_buffer_handle(&self) -> BufferHandle {
        self.handle
    }

    /// Attach the accounting deed (called from [`Device::alloc_buffer`] paths only).
    pub(crate) fn set_deed(&mut self, deed: ParcelDeed) {
        self.deed = Some(deed);
    }

    /// Create a new buffer with the specified access pattern.
    ///
    /// # Access Patterns
    ///
    /// - `BufferKind::Scattered`: Any thread can access any address (read/write).
    ///   Use for general-purpose data (StructuredBuffer, RWStructuredBuffer).
    ///
    /// - `BufferKind::Broadcast`: All threads read the same address.
    ///   Hardware optimizes for wave-wide broadcast (ConstantBuffer).
    pub(crate) fn new(device: &Device, size: u64, access: BufferKind) -> Result<Self> {
        Self::new_with_stride_and_flags(device, size, access, None, BufferFlags::empty())
    }

    /// Like [`Self::new`], with a peak-capacity hint for backends that support oversize virtual
    /// reservations (e.g. Metal). `expected_max` is clamped with `initial_size`; allocation is at
    /// least `max(initial_size, expected_max)` on supporting backends.
    pub(crate) fn new_with_capacity_hint(
        device: &Device,
        initial_size: u64,
        expected_max: u64,
        access: BufferKind,
    ) -> Result<Self> {
        Self::new_with_capacity_hint_and_flags(device, initial_size, expected_max, access, BufferFlags::empty())
    }

    /// Like [`Self::new_with_capacity_hint`], with explicit [`BufferFlags`].
    ///
    /// Use [`BufferFlags::GPU_ONLY`] for device-local frame scratch pools on Metal.
    pub(crate) fn new_with_capacity_hint_and_flags(
        device: &Device,
        initial_size: u64,
        expected_max: u64,
        access: BufferKind,
        flags: BufferFlags,
    ) -> Result<Self> {
        if flags.contains(BufferFlags::GPU_ONLY) && flags.contains(BufferFlags::CPU_READABLE) {
            anyhow::bail!("BufferFlags::GPU_ONLY cannot be combined with BufferFlags::CPU_READABLE");
        }
        let capacity = expected_max.max(initial_size);
        tracing::debug!(
            initial_size,
            capacity,
            ?access,
            ?flags,
            "Creating buffer with capacity hint"
        );
        let mut backend = device.inner.backend.lock().unwrap();
        let (handle, allocated_size) =
            backend.create_buffer_with_capacity(device.inner.handle, initial_size, capacity, access, None, flags)?;
        Ok(Self {
            device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            size: initial_size,
            allocated_size,
            access,
            element_stride: None,
            flags,
            peak_committed_bytes: allocated_size,
            resize_count: 0,
            deed: None,
        })
    }
    pub(crate) fn new_with_stride(
        device: &Device,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
    ) -> Result<Self> {
        Self::new_with_stride_and_flags(device, size, access, element_stride, BufferFlags::empty())
    }

    /// Create a buffer with optional element stride and [`BufferFlags`].
    pub(crate) fn new_with_stride_and_flags(
        device: &Device,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<Self> {
        tracing::debug!(size, ?access, element_stride, ?flags, "Creating buffer");
        if flags.contains(BufferFlags::GPU_ONLY) && flags.contains(BufferFlags::CPU_READABLE) {
            anyhow::bail!("BufferFlags::GPU_ONLY cannot be combined with BufferFlags::CPU_READABLE");
        }
        let mut backend = device.inner.backend.lock().unwrap();
        let handle = backend.create_buffer(device.inner.handle, size, access, element_stride, flags)?;

        Ok(Self {
            device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            size,
            allocated_size: size,
            access,
            element_stride,
            flags,
            peak_committed_bytes: size,
            resize_count: 0,
            deed: None,
        })
    }

    /// Create a buffer initialized with data.
    ///
    /// Element stride for structured-buffer views is `size_of::<T>()`. The type parameter is
    /// load-bearing: passing a **`&[u8]`** (for example from `bytemuck::bytes_of(&uniforms)`)
    /// fixes stride at **1 byte** while shaders usually expect `size_of::<YourStruct>()`.
    /// On some backends that mismatch reads as zeros or garbage with no error. Prefer a
    /// typed slice such as `&[YourStruct]` or [`Allocation::with_bytes_stride`] /
    /// [`Allocation::with_bytes`] with an explicit stride.
    ///
    /// See [`StructuredBufferElement`] for which `T` are allowed (`u8` / `i8` are not).
    ///
    /// See [`Allocation::new`] and [`BufferKind::Scattered`] for access-pattern details.
    pub(crate) fn with_data<T: StructuredBufferElement>(
        device: &Device,
        data: &[T],
        access: BufferKind,
    ) -> Result<Self> {
        Self::with_data_and_flags(device, data, access, BufferFlags::empty())
    }

    /// Like [`Self::with_data`], with explicit [`BufferFlags`].
    pub(crate) fn with_data_and_flags<T: StructuredBufferElement>(
        device: &Device,
        data: &[T],
        access: BufferKind,
        flags: BufferFlags,
    ) -> Result<Self> {
        let bytes = bytemuck::cast_slice(data);
        let element_stride = std::mem::size_of::<T>() as u32;
        let mut backend = device.inner.backend.lock().unwrap();
        let handle = backend.create_buffer(
            device.inner.handle,
            bytes.len() as u64,
            access,
            Some(element_stride),
            flags,
        )?;
        drop(backend);

        let buffer = Self {
            device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            size: bytes.len() as u64,
            allocated_size: bytes.len() as u64,
            access,
            element_stride: Some(element_stride),
            flags,
            peak_committed_bytes: bytes.len() as u64,
            resize_count: 0,
            deed: None,
        };
        buffer.write(0, bytes)?;
        Ok(buffer)
    }

    /// Create a buffer initialized with raw bytes (element stride **1**).
    ///
    /// Use this or [`Allocation::with_bytes_stride`] when data is naturally `&[u8]`. For typed
    /// structs, prefer [`Allocation::with_data`] with `&[T]` so stride matches the shader type.
    ///
    /// See [`Allocation::new`] for access pattern documentation.
    pub(crate) fn with_bytes(device: &Device, data: &[u8], access: BufferKind) -> Result<Self> {
        // For raw bytes, use stride of 1 (byte-addressable)
        Self::with_bytes_stride_and_flags(device, data, access, 1, BufferFlags::empty())
    }

    /// Create a buffer initialized with raw bytes and a custom element stride.
    ///
    /// The stride is used for creating StructuredBuffer views on DX12. For example,
    /// if the data contains u32 values, use stride=4 so the GPU can correctly
    /// interpret the buffer as `StructuredBuffer<uint>`.
    ///
    /// See [`Allocation::new`] for access pattern documentation.
    pub(crate) fn with_bytes_stride(
        device: &Device,
        data: &[u8],
        access: BufferKind,
        element_stride: u32,
    ) -> Result<Self> {
        Self::with_bytes_stride_and_flags(device, data, access, element_stride, BufferFlags::empty())
    }

    /// Like [`Self::with_bytes_stride`], with explicit [`BufferFlags`].
    pub(crate) fn with_bytes_stride_and_flags(
        device: &Device,
        data: &[u8],
        access: BufferKind,
        element_stride: u32,
        flags: BufferFlags,
    ) -> Result<Self> {
        let mut backend = device.inner.backend.lock().unwrap();
        let handle = backend.create_buffer(
            device.inner.handle,
            data.len() as u64,
            access,
            Some(element_stride),
            flags,
        )?;
        drop(backend);

        let buffer = Self {
            device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            size: data.len() as u64,
            allocated_size: data.len() as u64,
            access,
            element_stride: Some(element_stride),
            flags,
            peak_committed_bytes: data.len() as u64,
            resize_count: 0,
            deed: None,
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

    /// Logical byte size (may be less than reserved capacity; see [`Self::allocated_size`]).
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Committed byte size for accounting (equals logical [`Self::size`] today).
    pub fn byte_size(&self) -> u64 {
        self.size
    }

    /// Reserved byte capacity (physical or virtual backing size).
    pub fn allocated_size(&self) -> u64 {
        self.allocated_size
    }
    /// Get the buffer's access pattern.
    pub fn access(&self) -> BufferKind {
        self.access
    }

    /// Creation flags (e.g. [`BufferFlags::CPU_READABLE`]).
    pub fn flags(&self) -> BufferFlags {
        self.flags
    }

    /// Element stride passed at creation (for structured-buffer descriptors), if any.
    pub fn element_stride(&self) -> Option<u32> {
        self.element_stride
    }

    /// Peak physically-committed bytes ever observed on this buffer.
    ///
    /// Equals [`Self::allocated_size`] at creation and grows monotonically each time a
    /// [`Self::resize_to`] / [`Self::resize_to_uninitialized`] call causes the backend to
    /// expand the physical backing. Useful for profiling capacity hints and detecting
    /// over-allocation.
    pub fn peak_committed_bytes(&self) -> u64 {
        self.peak_committed_bytes
    }

    /// Number of completed resize operations ([`Self::resize_to`] / [`Self::resize_to_uninitialized`]).
    ///
    /// Incremented once per call that changes the logical size. No-op calls (same size as
    /// current) are not counted.
    pub fn resize_count(&self) -> u32 {
        self.resize_count
    }

    /// Resize the buffer in place, preserving contents in `[0..min(old, new))` and zero-initialising
    /// any newly exposed bytes. [`Self::resource_index`] values and the internal resource handle stay stable.
    pub fn resize_to(&mut self, new_size: u64) -> Result<()> {
        if new_size == self.size {
            return Ok(());
        }
        self.resize_count = self.resize_count.saturating_add(1);
        if new_size <= self.allocated_size {
            let old_logical = self.size;
            let mut backend = self.backend.lock().unwrap();
            backend.set_buffer_logical_size(self.device.inner.handle, self.handle, new_size)?;
            drop(backend);
            if new_size > old_logical {
                self.clear(&self.device, old_logical, new_size.saturating_sub(old_logical))?;
            }
            self.size = new_size;
            return Ok(());
        }
        let mut backend = self.backend.lock().unwrap();
        backend.resize_buffer(self.device.inner.handle, self.handle, new_size, true)?;
        self.allocated_size = backend.buffer_capacity(self.handle);
        self.peak_committed_bytes = self.peak_committed_bytes.max(self.allocated_size);
        self.size = new_size;
        Ok(())
    }

    /// Resize without preserving or initializing existing bytes (fast path for pools about to reset).
    /// New storage may contain arbitrary data; only the handle stability contract applies.
    pub fn resize_to_uninitialized(&mut self, new_size: u64) -> Result<()> {
        if new_size == self.size {
            return Ok(());
        }
        self.resize_count = self.resize_count.saturating_add(1);
        if new_size <= self.allocated_size {
            let mut backend = self.backend.lock().unwrap();
            backend.set_buffer_logical_size(self.device.inner.handle, self.handle, new_size)?;
            self.size = new_size;
            return Ok(());
        }
        let mut backend = self.backend.lock().unwrap();
        backend.resize_buffer(self.device.inner.handle, self.handle, new_size, false)?;
        self.allocated_size = backend.buffer_capacity(self.handle);
        self.peak_committed_bytes = self.peak_committed_bytes.max(self.allocated_size);
        self.size = new_size;
        Ok(())
    }

    /// Hint that bytes at and above `offset` are not needed until written again.
    ///
    /// On Metal (shared memory), may return physical pages to the OS. Other backends may no-op.
    pub fn hint_unused_above(&mut self, offset: u64) {
        let mut backend = self.backend.lock().unwrap();
        backend.hint_buffer_unused_above(self.handle, offset);
    }

    /// Resource descriptor index for how this buffer will be accessed in the current dispatch.
    ///
    /// Returns `None` for invalid access/kind combinations (e.g. write on `Broadcast`).
    pub fn resource_index(&self, access: ResourceAccess) -> Option<u32> {
        let backend = self.backend.lock().unwrap();
        match (self.access, access) {
            (BufferKind::Broadcast, ResourceAccess::Read) => backend.buffer_bindless_index(self.handle),
            (BufferKind::Broadcast, ResourceAccess::Write | ResourceAccess::ReadWrite) => None,
            (BufferKind::Scattered, ResourceAccess::Read) => backend.buffer_bindless_srv_index(self.handle),
            (BufferKind::Scattered, ResourceAccess::Write | ResourceAccess::ReadWrite) => {
                backend.buffer_bindless_index(self.handle)
            }
        }
    }

    /// Typed resource descriptor handle for validation and dispatch wiring.
    pub fn handle(&self, access: ResourceAccess) -> Option<ResourceHandle> {
        self.resource_index(access)
            .map(|i| ResourceHandle::new(ResourceCategory::from(self.access), i))
    }

    /// Read buffer contents back to CPU memory.
    ///
    /// The `output` slice must be at least `size` bytes. Reads from offset 0.
    ///
    /// For buffers created with [`BufferFlags::CPU_READABLE`], cost differs by backend:
    /// Vulkan / Metal typically copy directly from host-visible memory (see
    /// [`crate::device::DeviceCapabilities::has_zero_copy_storage_readback`]). Direct3D 12 performs a
    /// GPU copy into a READBACK heap and waits — query capabilities to branch on behavior.
    pub fn read_to_cpu(&self, device: &Device, output: &mut [u8]) -> Result<()> {
        let _tz = crate::tracy_zone!("buffer.read_to_cpu");
        let mut backend = {
            let _lock = crate::tracy_zone!("buffer.read_to_cpu.lock");
            self.backend.lock().unwrap()
        };
        let _backend = crate::tracy_zone!("buffer.read_to_cpu.backend");
        backend.read_buffer_to_cpu(device.inner.handle, self.handle, output)
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// Clear the buffer (fill with zeros) from offset for size bytes.
    pub fn clear(&self, device: &Device, offset: u64, size: u64) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.clear_buffer(device.inner.handle, self.handle, offset, size)
    }

    /// Create a view into a sub-region of this buffer.
    ///
    /// The view gets its own bindless descriptor index, so shaders see a zero-based
    /// buffer starting at `offset`. Multiple views of the same buffer can be bound
    /// simultaneously to different resource slots.
    ///
    /// `element_stride` sets the structured buffer stride for the view's descriptor.
    /// If `None`, defaults to 4 bytes (u32).
    pub fn create_view(&self, offset: u64, size: u64, element_stride: Option<u32>) -> Result<BufferView> {
        let mut backend = self.backend.lock().unwrap();
        let handle = backend.create_buffer_view(self.handle, offset, size, element_stride)?;
        Ok(BufferView {
            _device: self.device.clone(),
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
    pub fn create_typed_view<T: bytemuck::Pod>(&self, first_element: u64, count: u64) -> Result<BufferView> {
        let stride = std::mem::size_of::<T>() as u64;
        let offset = first_element * stride;
        let size = count * stride;
        self.create_view(offset, size, Some(stride as u32))
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        tracing::trace!(size = self.size, access = ?self.access, "Destroying buffer");
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_buffer(self.handle);
        if let Some(deed) = self.deed.as_ref() {
            deed.notify_freed(self.allocated_size, self.size, ParcelType::Buffer);
        }
    }
}

/// Trait for types that can be bound as vertex or index buffers.
///
/// [`Allocation`], [`BufferView`], and non-mosaic [`crate::Parcel`] buffers implement this trait,
/// allowing any of them to be passed to `set_vertex_buffer` and `set_index_buffer`.
/// For `BufferView`, the encoder binds the parent buffer at the view's offset internally.
/// For mosaic parcels, use [`crate::Parcel::view`] instead.
pub trait BufferSource {
    #[doc(hidden)]
    fn source_handle(&self) -> BufferHandle;
    #[doc(hidden)]
    fn source_offset(&self) -> u64;
}

impl BufferSource for Allocation {
    fn source_handle(&self) -> BufferHandle {
        self.handle
    }
    fn source_offset(&self) -> u64 {
        0
    }
}

/// A view into a sub-region of an [`Allocation`].
///
/// A `BufferView` shares the parent buffer's GPU memory but gets its own bindless
/// descriptor pointing at `[offset, offset+size)`. The shader sees the sub-region
/// as a zero-based buffer.
///
/// This enables buffer pooling: allocate one large buffer and create views for
/// each logical sub-allocation. Each view can be independently bound via resource slots.
///
/// Dropping a `BufferView` unregisters its descriptor but does not free the parent's memory.
#[derive(Clone)]
pub struct BufferView {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: BufferHandle,
    parent_handle: BufferHandle,
    offset: u64,
    size: u64,
}

impl BufferView {
    /// Resource descriptor index for how this view will be accessed in the current dispatch.
    pub fn resource_index(&self, access: ResourceAccess) -> Option<u32> {
        let backend = self.backend.lock().unwrap();
        match access {
            ResourceAccess::Read => backend.buffer_bindless_srv_index(self.handle),
            ResourceAccess::Write | ResourceAccess::ReadWrite => backend.buffer_bindless_index(self.handle),
        }
    }

    /// Typed resource descriptor handle for validation and dispatch wiring.
    ///
    /// Views are always created on top of `BufferKind::Scattered` backing storage
    /// (the only access pattern for which sub-ranges make sense), so the handle
    /// is always tagged [`ResourceCategory::Scattered`].
    pub fn handle(&self, access: ResourceAccess) -> Option<ResourceHandle> {
        self.resource_index(access)
            .map(|i| ResourceHandle::new(ResourceCategory::Scattered, i))
    }

    /// Get the handle of the backing buffer that owns this view's memory.
    pub(crate) fn parent_handle(&self) -> BufferHandle {
        self.parent_handle
    }

    /// Get the view's offset within the parent buffer in bytes.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Get the view size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Clear (zero-fill) a region within this view.
    ///
    /// `offset` is relative to the view's start. If `size` is 0, clears from
    /// `offset` to the end of the view.
    pub fn clear(&self, device: &Device, offset: u64, size: u64) -> Result<()> {
        let clear_size = if size == 0 {
            self.size.saturating_sub(offset)
        } else {
            size
        };
        if offset + clear_size > self.size {
            anyhow::bail!(
                "BufferView::clear [{}, {}) exceeds view size {}",
                offset,
                offset + clear_size,
                self.size
            );
        }
        let mut backend = self.backend.lock().unwrap();
        backend.clear_buffer(
            device.inner.handle,
            self.parent_handle,
            self.offset + offset,
            clear_size,
        )
    }

    /// Read this view's contents back to CPU memory.
    ///
    /// `output` must be exactly `self.size()` bytes. Reads only the view's
    /// sub-region from the parent buffer.
    pub fn read_to_cpu(&self, device: &Device, output: &mut [u8]) -> Result<()> {
        if output.len() as u64 != self.size {
            anyhow::bail!(
                "BufferView::read_to_cpu: output len {} != view size {}",
                output.len(),
                self.size
            );
        }
        if self.size == 0 {
            return Ok(());
        }
        let mut backend = self.backend.lock().unwrap();
        let parent_size = backend.buffer_size(self.parent_handle);
        let mut full = vec![0u8; parent_size as usize];
        backend.read_buffer_to_cpu(device.inner.handle, self.parent_handle, &mut full)?;
        output.copy_from_slice(&full[self.offset as usize..self.offset as usize + self.size as usize]);
        Ok(())
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
pub(crate) fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// LCM for alignment: smallest value divisible by both a and b.
pub(crate) fn lcm(a: u64, b: u64) -> u64 {
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
/// use goldy::{BufferPool, DeviceDescriptor, Instance, RequestAdapterOptions, ResourceAccess};
///
/// let instance = Instance::new()?;
/// let device = instance
///     .request_adapter(&RequestAdapterOptions::default())?
///     .request_device(&DeviceDescriptor::default())?;
///
/// let mut pool = BufferPool::new(&device, 1024 * 1024)?; // 1 MB pool
///
/// let tiles = pool.alloc::<[u32; 2]>(1024)?;   // 8 KB for 1024 tiles
/// let segments = pool.alloc::<[f32; 6]>(4096)?; // 96 KB for 4096 segments
///
/// // Use views via resource indices
/// let tile_idx = tiles.resource_index(ResourceAccess::Write).unwrap();
/// let seg_idx = segments.resource_index(ResourceAccess::Write).unwrap();
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct BufferPool {
    backing: Allocation,
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
    /// The backing buffer is allocated as `BufferKind::Scattered` (storage buffer)
    /// since sub-allocation only makes sense for storage buffers.
    ///
    /// `alignment` defaults to 256 bytes, which satisfies `minStorageBufferOffsetAlignment`
    /// on all known Vulkan/DX12 hardware.
    pub fn new(device: &Device, total_size: u64) -> Result<Self> {
        Self::with_alignment(device, total_size, 256)
    }

    /// Create a pool with a custom sub-allocation alignment.
    pub fn with_alignment(device: &Device, total_size: u64, alignment: u64) -> Result<Self> {
        assert!(alignment.is_power_of_two(), "alignment must be a power of two");
        tracing::debug!(total_size, alignment, "Creating buffer pool");
        let backing = device.alloc_buffer(total_size, BufferKind::Scattered, None, BufferFlags::empty())?;
        Ok(Self {
            backing,
            offset: 0,
            alignment,
        })
    }

    /// Like [`Self::with_alignment`], but reserves up to `expected_max` bytes on supporting backends.
    pub fn with_alignment_and_capacity_hint(
        device: &Device,
        total_size: u64,
        expected_max: u64,
        alignment: u64,
    ) -> Result<Self> {
        Self::with_alignment_capacity_hint_and_flags(device, total_size, expected_max, alignment, BufferFlags::empty())
    }

    /// Like [`Self::with_alignment_and_capacity_hint`] with [`BufferFlags`].
    pub fn with_alignment_capacity_hint_and_flags(
        device: &Device,
        total_size: u64,
        expected_max: u64,
        alignment: u64,
        flags: BufferFlags,
    ) -> Result<Self> {
        assert!(alignment.is_power_of_two(), "alignment must be a power of two");
        tracing::debug!(
            total_size,
            expected_max,
            alignment,
            ?flags,
            "Creating buffer pool with capacity hint"
        );
        let backing = device.alloc_buffer_with_capacity(total_size, expected_max, BufferKind::Scattered, flags)?;
        Ok(Self {
            backing,
            offset: 0,
            alignment,
        })
    }

    /// Resize the backing buffer in place (stable handle) and reset the bump allocator.
    pub fn resize(&mut self, new_size: u64) -> Result<()> {
        self.backing.resize_to(new_size)?;
        self.offset = 0;
        Ok(())
    }

    /// Allocate a typed region from the pool.
    ///
    /// Returns a `BufferView` spanning `count` elements of type `T`, with the offset
    /// aligned to the pool's alignment requirement.
    pub fn alloc<T: StructuredBufferElement>(&mut self, count: u64) -> Result<BufferView> {
        let stride = std::mem::size_of::<T>() as u64;
        let size = count * stride;
        self.alloc_bytes(size, Some(stride as u32))
    }

    /// Allocate and fill a typed region in one call.
    ///
    /// Equivalent to `alloc::<T>(data.len())` followed by `write_data(data)`.
    /// Same element-stride rules as `Device::alloc_buffer_with_data`.
    pub fn alloc_with_data<T: StructuredBufferElement>(&mut self, data: &[T]) -> Result<BufferView> {
        let view = self.alloc::<T>(data.len() as u64)?;
        view.write_data(data)?;
        Ok(view)
    }

    /// Whether an [`Self::alloc_bytes`] of `size` bytes with the given `element_stride` would
    /// fit in the remaining pool capacity without growth.
    ///
    /// Uses the same alignment math as [`Self::alloc_bytes`] so the answer is exact, including
    /// for non-power-of-two strides (e.g. 12-byte `vec3<f32>`).
    pub fn would_fit(&self, size: u64, element_stride: Option<u32>) -> bool {
        let stride_u32 = element_stride.unwrap_or(4);
        if stride_u32 == 0 || !size.is_multiple_of(stride_u32 as u64) {
            return false;
        }
        let stride = stride_u32 as u64;
        let alloc_align = lcm(self.alignment, stride);
        let aligned_offset = self.offset.div_ceil(alloc_align) * alloc_align;
        aligned_offset.saturating_add(size) <= self.backing.size()
    }

    /// Allocate a raw byte region from the pool.
    ///
    /// `element_stride` determines the structured buffer stride for the view's descriptor.
    /// If `None`, defaults to 4 bytes (u32).
    ///
    /// Each allocation is aligned to satisfy both pool alignment (256) and
    /// `offset % element_stride == 0` (required by DX12 StructuredBuffer views).
    pub fn alloc_bytes(&mut self, size: u64, element_stride: Option<u32>) -> Result<BufferView> {
        let stride_u32 = element_stride.unwrap_or(4);
        if stride_u32 == 0 {
            anyhow::bail!("BufferPool alloc_bytes: element stride must be non-zero");
        }
        if !size.is_multiple_of(stride_u32 as u64) {
            anyhow::bail!(
                "BufferPool alloc_bytes: size {size} must be a multiple of element stride {stride_u32} \
                 (StructuredBuffer views require an integral element count)"
            );
        }
        let stride = stride_u32 as u64;
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

        let view = self.backing.create_view(aligned_offset, size, element_stride)?;
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
    pub(crate) fn backing_buffer(&self) -> &Allocation {
        &self.backing
    }

    /// Read the entire backing allocation to CPU memory.
    pub fn read_to_cpu(&self, device: &Device, output: &mut [u8]) -> Result<()> {
        self.backing.read_to_cpu(device, output)
    }

    /// Forward [`Allocation::hint_unused_above`] on the backing allocation.
    pub fn hint_unused_above(&mut self, offset: u64) {
        self.backing.hint_unused_above(offset);
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
