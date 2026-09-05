//! GPU buffer management.

use crate::backend::{BufferHandle, GpuBackend};
use crate::device::Device;
use crate::types::{BufferFlags, BufferKind, ResourceAccess, ResourceCategory, ResourceHandle};
use crate::vram_allocator::{ParcelDeed, ParcelType};
use anyhow::Result;
use std::borrow::Cow;
use std::sync::{Arc, Mutex};

fn bindless_cache_from_backend(
    backend: &dyn GpuBackend,
    handle: BufferHandle,
    access: BufferKind,
) -> (Option<u32>, Option<u32>, Option<u32>) {
    match access {
        BufferKind::Broadcast => (None, None, backend.buffer_bindless_index(handle)),
        BufferKind::Scattered => (
            backend.buffer_bindless_index(handle),
            backend.buffer_bindless_srv_index(handle),
            None,
        ),
    }
}

/// Types allowed as elements in [`RetainedPool::acquire_buffer_with_data`](crate::RetainedPool::acquire_buffer_with_data).
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
///
/// [`GpuType`](crate::GpuType) types pack into the Slang structured-buffer ABI in
/// [`Self::gpu_encode_slice`] / [`Self::gpu_element_stride`]. Other types memcpy as `repr(C)`.
pub trait StructuredBufferElement: bytemuck::Pod {
    /// Element stride used for structured-buffer views.
    fn gpu_element_stride() -> usize {
        std::mem::size_of::<Self>()
    }

    /// Host slice encoded for GPU upload.
    fn gpu_encode_slice(items: &[Self]) -> Cow<'_, [u8]> {
        Cow::Borrowed(bytemuck::cast_slice(items))
    }
}

macro_rules! impl_structured_buffer_element_for_primitives {
    ($($t:ty),+ $(,)?) => {
        $(impl StructuredBufferElement for $t {})+
    };
}

impl_structured_buffer_element_for_primitives!((), i16, u16, i32, u32, i64, u64, i128, u128, isize, usize, f32, f64,);

impl StructuredBufferElement for crate::types::DispatchShape {}

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
    /// Cached bindless UAV index (Scattered write / RW).
    bindless_uav: Option<u32>,
    /// Cached bindless SRV index (Scattered read).
    bindless_srv: Option<u32>,
    /// Cached bindless CBV index (Broadcast read).
    bindless_cbv: Option<u32>,
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
        let (bindless_uav, bindless_srv, bindless_cbv) = bindless_cache_from_backend(&**backend, handle, access);
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
            bindless_uav,
            bindless_srv,
            bindless_cbv,
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
        let (bindless_uav, bindless_srv, bindless_cbv) = bindless_cache_from_backend(&**backend, handle, access);

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
            bindless_uav,
            bindless_srv,
            bindless_cbv,
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
        let encoded = T::gpu_encode_slice(data);
        let bytes = encoded.as_ref();
        let element_stride = T::gpu_element_stride() as u32;
        let mut backend = device.inner.backend.lock().unwrap();
        let handle = backend.create_buffer(
            device.inner.handle,
            bytes.len() as u64,
            access,
            Some(element_stride),
            flags,
        )?;
        let (bindless_uav, bindless_srv, bindless_cbv) = bindless_cache_from_backend(&**backend, handle, access);
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
            bindless_uav,
            bindless_srv,
            bindless_cbv,
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
        let (bindless_uav, bindless_srv, bindless_cbv) = bindless_cache_from_backend(&**backend, handle, access);
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
            bindless_uav,
            bindless_srv,
            bindless_cbv,
            deed: None,
        };
        buffer.write(0, data)?;
        Ok(buffer)
    }

    /// Write data to the buffer.
    ///
    /// See [`crate::Buffer::write`] for the public contract. For
    /// [`crate::types::BufferFlags::CPU_WRITABLE`], the write must target a settled or
    /// fresh buffer; backends do not queue-order it behind in-flight GPU readers.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let mut backend = self.backend.lock().unwrap();
        backend.write_buffer(self.handle, offset, data)
    }

    /// Write typed data to the buffer (packed for [`crate::GpuType`] elements).
    pub fn write_data<T: StructuredBufferElement>(&self, offset: u64, data: &[T]) -> Result<()> {
        let encoded = T::gpu_encode_slice(data);
        self.write(offset, encoded.as_ref())
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
    /// any newly exposed bytes. Bindless slot indices and the internal resource handle stay stable.
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
    /// Crate-internal: the public binding path is [`Self::handle`] / scheme `with_parcel`.
    pub(crate) fn resource_index(&self, access: ResourceAccess) -> Option<u32> {
        match (self.access, access) {
            (BufferKind::Broadcast, ResourceAccess::Read) => self.bindless_cbv,
            (BufferKind::Broadcast, ResourceAccess::Write | ResourceAccess::ReadWrite) => None,
            (BufferKind::Scattered, ResourceAccess::Read) => self.bindless_srv,
            (BufferKind::Scattered, ResourceAccess::Write | ResourceAccess::ReadWrite) => self.bindless_uav,
        }
    }

    /// Opaque typed resource descriptor identity for validation and retention checks.
    pub fn handle(&self, access: ResourceAccess) -> Option<ResourceHandle> {
        self.resource_index(access)
            .map(|i| ResourceHandle::new(ResourceCategory::from(self.access), i))
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
        let bindless_uav = backend.buffer_bindless_index(handle);
        let bindless_srv = backend.buffer_bindless_srv_index(handle);
        Ok(BufferView {
            _device: self.device.clone(),
            backend: Arc::clone(&self.backend),
            handle,
            parent_handle: self.handle,
            offset,
            size,
            bindless_uav,
            bindless_srv,
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
/// [`BufferView`], and [`crate::Parcel`] implement this trait,
/// allowing any of them to be passed to `set_vertex_buffer` and `set_index_buffer`.
/// For `BufferView`, the encoder binds the parent buffer at the view's offset internally.
/// For partitioned buffers, bind a specific range [`crate::Parcel`] via [`crate::Buffer::field`] or indexing.
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

/// A view into a sub-region of a backing GPU buffer allocation.
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
    bindless_uav: Option<u32>,
    bindless_srv: Option<u32>,
}

impl BufferView {
    /// Resource descriptor index for how this view will be accessed in the current dispatch.
    ///
    /// Crate-internal: the public binding path is [`Self::handle`] / scheme `with_parcel`.
    pub(crate) fn resource_index(&self, access: ResourceAccess) -> Option<u32> {
        match access {
            ResourceAccess::Read => self.bindless_srv,
            ResourceAccess::Write | ResourceAccess::ReadWrite => self.bindless_uav,
        }
    }

    /// Opaque typed resource descriptor identity for validation and retention checks.
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

const SCATTERED_SUBALLOC_ALIGNMENT: u64 = 256;

/// Compute the total scattered backing size for a set of sub-regions.
fn scattered_suballoc_padded_size(allocs: &[(usize, usize)]) -> u64 {
    let mut offset = 0u64;
    for &(count, stride) in allocs {
        let stride = stride as u64;
        let alloc_align = lcm(SCATTERED_SUBALLOC_ALIGNMENT, stride);
        let aligned_offset = offset.div_ceil(alloc_align) * alloc_align;
        let size = (count as u64) * stride;
        offset = aligned_offset + size;
    }
    offset
}

/// One sub-region to carve from a single scattered backing allocation.
pub(crate) struct ScatteredSubregionSpec<'a> {
    pub byte_size: u64,
    pub element_stride: u32,
    pub init: Option<&'a [u8]>,
}

/// Allocate one `BufferKind::Scattered` backing buffer and carve typed views for each region.
pub(crate) fn alloc_scattered_subregions(
    device: &Device,
    regions: &[ScatteredSubregionSpec<'_>],
) -> Result<(Allocation, Vec<BufferView>)> {
    alloc_scattered_subregions_with_alignment(device, regions, SCATTERED_SUBALLOC_ALIGNMENT)
}

fn alloc_scattered_subregions_with_alignment(
    device: &Device,
    regions: &[ScatteredSubregionSpec<'_>],
    alignment: u64,
) -> Result<(Allocation, Vec<BufferView>)> {
    assert!(alignment.is_power_of_two(), "alignment must be a power of two");
    let pairs: Vec<(usize, usize)> = regions
        .iter()
        .map(|r| {
            let stride = r.element_stride as usize;
            let count = if stride == 0 {
                0
            } else {
                (r.byte_size / r.element_stride as u64) as usize
            };
            (count, stride)
        })
        .collect();
    let total = scattered_suballoc_padded_size(&pairs);
    let backing = device.alloc_buffer(total, BufferKind::Scattered, None, BufferFlags::empty())?;
    let mut offset = 0u64;
    let mut views = Vec::with_capacity(regions.len());
    for region in regions {
        let view = bump_scattered_subregion(
            &backing,
            &mut offset,
            alignment,
            region.byte_size,
            Some(region.element_stride),
        )?;
        if let Some(data) = region.init {
            view.write_data(data)?;
        }
        views.push(view);
    }
    Ok((backing, views))
}

fn bump_scattered_subregion(
    backing: &Allocation,
    offset: &mut u64,
    pool_alignment: u64,
    size: u64,
    element_stride: Option<u32>,
) -> Result<BufferView> {
    let stride_u32 = element_stride.unwrap_or(4);
    if stride_u32 == 0 {
        anyhow::bail!("scattered suballoc: element stride must be non-zero");
    }
    if !size.is_multiple_of(stride_u32 as u64) {
        anyhow::bail!(
            "scattered suballoc: size {size} must be a multiple of element stride {stride_u32} \
             (StructuredBuffer views require an integral element count)"
        );
    }
    let stride = stride_u32 as u64;
    let alloc_align = lcm(pool_alignment, stride);
    let aligned_offset = offset.div_ceil(alloc_align) * alloc_align;

    if aligned_offset + size > backing.size() {
        anyhow::bail!(
            "scattered suballoc exhausted: need {} bytes at offset {}, backing size is {}",
            size,
            aligned_offset,
            backing.size()
        );
    }

    let view = backing.create_view(aligned_offset, size, element_stride)?;
    *offset = aligned_offset + size;
    Ok(view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_padded_size_empty() {
        assert_eq!(scattered_suballoc_padded_size(&[]), 0);
    }

    #[test]
    fn test_padded_size_single_allocation() {
        // 64 u32s = 256 bytes, aligned to 256, no padding
        assert_eq!(scattered_suballoc_padded_size(&[(64, size_of::<u32>())]), 256);
    }

    #[test]
    fn test_padded_size_multiple_allocations() {
        // Multiple mesh buffers: static_vb, static_ib, sky_vb, sky_ib, decor_vb, decor_ib.
        // With varying strides, alignment padding is inserted between allocs
        let size = scattered_suballoc_padded_size(&[
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
