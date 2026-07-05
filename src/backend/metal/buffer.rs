//! Buffer management logic.

use super::super::{BufferHandle, DeviceHandle};
use super::types::{BufferState, MetalState, ResourceRegistry, ARGUMENT_BUFFER_SIZE, MAX_HEAP_SIZE};
use crate::backend::BufferKind;
use crate::types::BufferFlags;
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::MTLResourceOptions;

fn mtl_resource_options(flags: BufferFlags) -> MTLResourceOptions {
    if flags.contains(BufferFlags::GPU_ONLY) {
        MTLResourceOptions::StorageModePrivate
    } else {
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache
    }
}

/// Heap allocation, direct device buffer for GPU-only / jumbo sizes ([`MAX_HEAP_SIZE`]).
///
/// When the heap is saturated (overflow cap reached), performs a non-blocking
/// drain-and-retry first, then if still full, waits for the oldest in-flight command
/// buffer to complete, drains again, and retries. This makes the Metal backend
/// self-regulating: the caller never sees heap exhaustion in steady state.
fn allocate_mtl_storage_buffer(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    allocation_size: u64,
    flags: BufferFlags,
) -> Result<(mtl::Buffer, bool)> {
    let options = mtl_resource_options(flags);
    let gpu_only = flags.contains(BufferFlags::GPU_ONLY);

    {
        let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
        if gpu_only || allocation_size > MAX_HEAP_SIZE {
            let buf = logical_device.device.new_buffer(allocation_size, options);
            return Ok((buf, true));
        }

        // Attempt 1: fast path — heap has space.
        if let Some(buf) = logical_device
            .heap_allocator
            .lock()
            .unwrap()
            .allocate(allocation_size, options)
        {
            return Ok((buf, false));
        }
    }

    // Attempt 2 (non-blocking): drain any GPU work that has already signaled,
    // compact empty overflow heaps, then retry. Handles the common case where
    // the GPU finished a frame between the last flush and this allocation.
    {
        let _tz = crate::tracy_zone!("mtl.heap_allocator.drain_reclaim");
        let retired = super::context::device_retired(state, device_handle);
        let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
        logical_device.process_deletion_queue_up_to(retired);
        logical_device.heap_allocator.lock().unwrap().compact_overflow();
    }
    {
        let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
        if let Some(buf) = logical_device
            .heap_allocator
            .lock()
            .unwrap()
            .allocate(allocation_size, options)
        {
            return Ok((buf, false));
        }
    }

    // Attempt 3 (non-blocking pressure-relief valve): the overflow-heap cap is
    // reached and a synchronous drain is unsafe here — this runs with the backend
    // mutex held, so blocking on a command buffer would deadlock against the GPU
    // completion handlers that need that same lock. Instead of failing (which the
    // client turns into a retry storm that spikes VRAM), satisfy the request from a
    // standalone device buffer, exactly as the jumbo/GPU-only path above does. These
    // are transient warmup spikes (the CPU racing ahead before caches are warm, more
    // pronounced under the 2-commit-per-frame split); the direct buffer is tracked as
    // device-allocated and freed through the normal deletion queue, so steady state
    // returns to the heap once the GPU catches up.
    crate::signal::push_sync_signal(crate::signal::Signal::Oversubscribed {
        reason: crate::signal::OversubscribedReason::BufferHeap,
        size_hint: allocation_size,
    });
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
    let buf = logical_device.device.new_buffer(allocation_size, options);
    Ok((buf, true))
}

#[allow(clippy::too_many_arguments)]
fn insert_buffer_common(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    handle: BufferHandle,
    buffer: mtl::Buffer,
    logical_size: u64,
    allocation_size: u64,
    is_device_allocated: bool,
    access: BufferKind,
    element_stride: Option<u32>,
    flags: BufferFlags,
    parent_for_view: Option<BufferHandle>,
    view_byte_offset: Option<u64>,
) -> Result<()> {
    debug_assert!(logical_size <= allocation_size);
    let cpu_readable = flags.contains(BufferFlags::CPU_READABLE);
    let is_storage = access == BufferKind::Scattered;

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    let arg_buffer_index = {
        let mut registry = logical_device.descriptors.lock().unwrap();
        match access {
            BufferKind::Broadcast => registry.resource_registry.register_uniform_buffer(handle),
            BufferKind::Scattered => registry.resource_registry.register_storage_buffer(handle),
        }
    };
    let encoding_index = match access {
        BufferKind::Broadcast => ResourceRegistry::uniform_global_index(arg_buffer_index),
        BufferKind::Scattered => arg_buffer_index,
    };
    tracing::debug!(
        "Allocated buffer {} (device heap={}) at bindless index {}",
        handle,
        is_device_allocated,
        arg_buffer_index
    );

    let encoded_length = logical_device.argument_encoder.encoded_length();
    let offset = (encoding_index as u64) * encoded_length;
    if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
        logical_device
            .argument_encoder
            .set_argument_buffer(&logical_device.argument_buffer, offset);
        logical_device.argument_encoder.set_buffer(0, &buffer, 0);
        tracing::trace!(
            "Encoded buffer {} at arg buffer offset {} (slot {})",
            handle,
            offset,
            arg_buffer_index,
        );
    }

    if cpu_readable && is_storage {
        let ptr = buffer.contents() as *mut u8;
        if ptr.is_null() {
            anyhow::bail!("Metal buffer contents() returned null for CPU_READABLE");
        }
    }

    state.buffers.insert(
        handle,
        BufferState {
            device_handle,
            buffer,
            size: logical_size,
            allocation_size,
            is_device_allocated,
            arg_buffer_index,
            flags,
            element_stride,
            parent_for_view,
            access,
            view_byte_offset,
            is_grant_readback: false,
            texture_copy_footprint: None,
        },
    );

    Ok(())
}

/// Create a buffer with the given size and access pattern.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    size: u64,
    access: BufferKind,
    element_stride: Option<u32>,
    flags: BufferFlags,
) -> Result<BufferHandle> {
    let cpu_readable = flags.contains(BufferFlags::CPU_READABLE);
    let cpu_writable = flags.contains(BufferFlags::CPU_WRITABLE);
    let is_storage = access == BufferKind::Scattered;
    if cpu_readable && !is_storage {
        anyhow::bail!("BufferFlags::CPU_READABLE is only valid for BufferKind::Scattered (storage) buffers");
    }
    if cpu_writable && !is_storage {
        anyhow::bail!("BufferFlags::CPU_WRITABLE is only valid for BufferKind::Scattered (storage) buffers");
    }
    if cpu_readable && flags.contains(BufferFlags::GPU_ONLY) {
        anyhow::bail!("BufferFlags::GPU_ONLY cannot be combined with CPU_READABLE");
    }
    if cpu_writable && flags.contains(BufferFlags::GPU_ONLY) {
        anyhow::bail!("BufferFlags::GPU_ONLY cannot be combined with CPU_WRITABLE");
    }
    if cpu_writable && cpu_readable {
        anyhow::bail!("BufferFlags::CPU_WRITABLE cannot be combined with CPU_READABLE");
    }

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    let (buffer, is_device_allocated) = allocate_mtl_storage_buffer(state, device_handle, size, flags)?;

    insert_buffer_common(
        state,
        device_handle,
        handle,
        buffer,
        size,
        size,
        is_device_allocated,
        access,
        element_stride,
        flags,
        None,
        None,
    )?;

    Ok(handle)
}

/// Create with reserved capacity (`allocation_size >= logical_size`).
pub(super) fn create_with_capacity(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    logical_size: u64,
    capacity: u64,
    access: BufferKind,
    element_stride: Option<u32>,
    flags: BufferFlags,
) -> Result<(BufferHandle, u64)> {
    let cpu_readable = flags.contains(BufferFlags::CPU_READABLE);
    let cpu_writable = flags.contains(BufferFlags::CPU_WRITABLE);
    let is_storage = access == BufferKind::Scattered;
    if cpu_readable && !is_storage {
        anyhow::bail!("BufferFlags::CPU_READABLE is only valid for BufferKind::Scattered (storage) buffers");
    }
    if cpu_writable && !is_storage {
        anyhow::bail!("BufferFlags::CPU_WRITABLE is only valid for BufferKind::Scattered (storage) buffers");
    }
    if cpu_readable && flags.contains(BufferFlags::GPU_ONLY) {
        anyhow::bail!("BufferFlags::GPU_ONLY cannot be combined with CPU_READABLE");
    }
    if cpu_writable && flags.contains(BufferFlags::GPU_ONLY) {
        anyhow::bail!("BufferFlags::GPU_ONLY cannot be combined with CPU_WRITABLE");
    }
    if cpu_writable && cpu_readable {
        anyhow::bail!("BufferFlags::CPU_WRITABLE cannot be combined with CPU_READABLE");
    }
    if logical_size > capacity {
        anyhow::bail!("logical_size {logical_size} exceeds capacity {capacity}");
    }
    if logical_size == 0 || capacity == 0 {
        anyhow::bail!("buffer sizes must be non-zero");
    }

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    let (buffer, is_device_allocated) = allocate_mtl_storage_buffer(state, device_handle, capacity, flags)?;

    insert_buffer_common(
        state,
        device_handle,
        handle,
        buffer,
        logical_size,
        capacity,
        is_device_allocated,
        access,
        element_stride,
        flags,
        None,
        None,
    )?;

    Ok((handle, capacity))
}

pub(super) fn buffer_capacity(state: &MetalState, buffer_handle: BufferHandle) -> u64 {
    state
        .buffers
        .get(&buffer_handle)
        .map(|b| b.allocation_size)
        .unwrap_or(0)
}

pub(super) fn set_logical_size(
    state: &mut MetalState,
    _device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    new_logical_size: u64,
) -> Result<()> {
    let b = state.buffers.get_mut(&buffer_handle).context("Invalid buffer handle")?;
    if b.parent_for_view.is_some() {
        anyhow::bail!("cannot resize logical extent of buffer views");
    }
    if new_logical_size > b.allocation_size {
        anyhow::bail!(
            "logical size {} exceeds allocation {}",
            new_logical_size,
            b.allocation_size
        );
    }
    if new_logical_size == 0 {
        anyhow::bail!("buffer size must be non-zero");
    }
    b.size = new_logical_size;
    Ok(())
}

/// Hint kernel reclaim for pages at/above `offset` (see [`GpuBackend::hint_buffer_unused_above`]).
pub(super) fn hint_unused_above(state: &mut MetalState, buffer_handle: BufferHandle, offset: u64) {
    let Some(b) = state.buffers.get(&buffer_handle) else {
        return;
    };
    if b.flags.contains(BufferFlags::GPU_ONLY) {
        return;
    }
    if b.parent_for_view.is_some() {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        use libc::{sysconf, _SC_PAGESIZE};
        let ptr = b.buffer.contents() as *mut u8;
        if ptr.is_null() {
            return;
        }
        let page = unsafe { sysconf(_SC_PAGESIZE) } as u64;
        if page == 0 {
            return;
        }
        let page_off = offset.div_ceil(page).saturating_mul(page);
        let len = b.allocation_size.saturating_sub(page_off);
        if len == 0 {
            return;
        }
        unsafe {
            libc::madvise(ptr.add(page_off as usize).cast(), len as usize, libc::MADV_FREE);
        }
    }
}

/// Create a view into a sub-region of an existing storage buffer.
///
/// On Metal, the view encodes the parent's MTLBuffer at the view's byte offset
/// into a new argument buffer slot. The shader sees element [0] as the data at `offset`.
pub(super) fn create_view(
    state: &mut MetalState,
    parent_handle: BufferHandle,
    offset: u64,
    size: u64,
    element_stride: Option<u32>,
) -> Result<BufferHandle> {
    let parent = state
        .buffers
        .get(&parent_handle)
        .context("Invalid parent buffer handle")?;

    if offset + size > parent.size {
        anyhow::bail!(
            "View [{}, {}) exceeds parent buffer size {}",
            offset,
            offset + size,
            parent.size
        );
    }

    let device_handle = parent.device_handle;
    let parent_mtl_buffer = parent.buffer.clone();
    let parent_flags = parent.flags;

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    let arg_buffer_index = logical_device
        .descriptors
        .lock()
        .unwrap()
        .resource_registry
        .register_storage_buffer(handle);

    let encoded_length = logical_device.argument_encoder.encoded_length();
    let ab_offset = (arg_buffer_index as u64) * encoded_length;
    if ab_offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
        logical_device
            .argument_encoder
            .set_argument_buffer(&logical_device.argument_buffer, ab_offset);
        logical_device
            .argument_encoder
            .set_buffer(0, &parent_mtl_buffer, offset);
    }

    state.buffers.insert(
        handle,
        BufferState {
            device_handle,
            buffer: parent_mtl_buffer,
            size,
            allocation_size: parent.allocation_size,
            is_device_allocated: parent.is_device_allocated,
            arg_buffer_index,
            flags: parent_flags,
            element_stride,
            parent_for_view: Some(parent_handle),
            access: BufferKind::Scattered,
            view_byte_offset: Some(offset),
            is_grant_readback: false,
            texture_copy_footprint: None,
        },
    );

    Ok(handle)
}

/// Resize a root buffer in place ([`BufferHandle`] and argument-buffer slot stay stable).
pub(super) fn resize(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    new_size: u64,
    preserve_contents: bool,
) -> Result<()> {
    let old_state = state
        .buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?
        .clone();

    if old_state.parent_for_view.is_some() {
        anyhow::bail!("cannot resize buffer views");
    }
    if old_state.device_handle != device_handle {
        anyhow::bail!("buffer belongs to a different device");
    }
    if new_size == old_state.size {
        return Ok(());
    }
    if new_size == 0 {
        anyhow::bail!("buffer size must be non-zero");
    }

    let (new_buffer, is_device_allocated) =
        allocate_mtl_storage_buffer(state, device_handle, new_size, old_state.flags)?;

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    let copy_len = if preserve_contents {
        old_state.size.min(new_size)
    } else {
        0
    };

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    if copy_len > 0 {
        blit.copy_from_buffer(&old_state.buffer, 0, &new_buffer, 0, copy_len);
    }
    if preserve_contents && new_size > copy_len {
        let tail = new_size - copy_len;
        if !old_state.flags.contains(BufferFlags::GPU_ONLY) {
            unsafe {
                let ptr = (new_buffer.contents() as *mut u8).add(copy_len as usize);
                std::ptr::write_bytes(ptr, 0, tail as usize);
            }
        }
        let range = mtl::NSRange::new(copy_len, tail);
        blit.fill_buffer(&new_buffer, range, 0);
    }
    blit.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let encoded_length = logical_device.argument_encoder.encoded_length();
    let encoding_index = match old_state.access {
        BufferKind::Broadcast => ResourceRegistry::uniform_global_index(old_state.arg_buffer_index),
        BufferKind::Scattered => old_state.arg_buffer_index,
    };
    let off = (encoding_index as u64) * encoded_length;
    if off + encoded_length <= ARGUMENT_BUFFER_SIZE {
        logical_device
            .argument_encoder
            .set_argument_buffer(&logical_device.argument_buffer, off);
        logical_device.argument_encoder.set_buffer(0, &new_buffer, 0);
    }

    if old_state.flags.contains(BufferFlags::CPU_READABLE) && old_state.access == BufferKind::Scattered {
        let ptr = new_buffer.contents() as *mut u8;
        if ptr.is_null() {
            anyhow::bail!("Metal buffer contents() returned null for CPU_READABLE (resize)");
        }
    }

    let barrier = logical_device
        .timeline_scheduled_max
        .load(std::sync::atomic::Ordering::Relaxed);
    logical_device.deletion_queue.lock().unwrap().queue(
        barrier,
        super::types::PendingDeletion::Buffer {
            buffer: old_state.buffer,
        },
    );

    *state.buffers.get_mut(&buffer_handle).unwrap() = BufferState {
        device_handle,
        buffer: new_buffer,
        size: new_size,
        allocation_size: new_size,
        is_device_allocated,
        arg_buffer_index: old_state.arg_buffer_index,
        flags: old_state.flags,
        element_stride: old_state.element_stride,
        parent_for_view: None,
        access: old_state.access,
        view_byte_offset: None,
        is_grant_readback: false,
        texture_copy_footprint: None,
    };

    let new_mtl = state.buffers.get(&buffer_handle).unwrap().buffer.clone();

    let view_handles: Vec<BufferHandle> = state
        .buffers
        .iter()
        .filter(|(h, st)| **h != buffer_handle && st.parent_for_view == Some(buffer_handle))
        .map(|(h, _)| *h)
        .collect();

    let enc_len = logical_device.argument_encoder.encoded_length();
    for vh in view_handles {
        let (arg_ix, mtl_off) = {
            let st = state.buffers.get(&vh).context("view missing")?;
            (
                st.arg_buffer_index,
                st.view_byte_offset.context("internal: view_byte_offset")?,
            )
        };
        let ab_off = (arg_ix as u64) * enc_len;
        if ab_off + enc_len <= ARGUMENT_BUFFER_SIZE {
            logical_device
                .argument_encoder
                .set_argument_buffer(&logical_device.argument_buffer, ab_off);
            logical_device.argument_encoder.set_buffer(0, &new_mtl, mtl_off);
        }
        state.buffers.get_mut(&vh).unwrap().buffer = new_mtl.clone();
    }

    Ok(())
}

/// Destroy a buffer, unregistering it from the bindless registry.
/// Slot recycling is gated on GPU idleness: if any previously-submitted
/// compute command buffer is still running, the slot parks in the registry's
/// pending list and only becomes reusable after the next `wait_fence()`
/// succeeds. This prevents the CPU from overwriting an argument-buffer
/// descriptor that an in-flight shader is about to dereference (descriptor
/// aliasing = random wrong-buffer reads and MTLCommandBufferError::Internal).
pub(super) fn destroy(state: &mut MetalState, buffer_handle: BufferHandle) {
    let gpu_idle = super::gpu_is_idle(state);
    if let Some(buffer) = state.buffers.remove(&buffer_handle) {
        // When called during VRAM reclamation, the installing thread's explicit
        // reclamation epoch has already been GPU-completed, so using it as the
        // deletion-queue barrier lets the next process_deletion_queue_up_to_signaled
        // call free the Metal heap allocation immediately rather than waiting for
        // the (often much larger) timeline_scheduled_max.
        let barrier = super::context::reclamation_barrier(state, buffer.device_handle, gpu_idle);
        if let Some(device) = state.devices.get(&buffer.device_handle) {
            device
                .descriptors
                .lock()
                .unwrap()
                .resource_registry
                .unregister_buffer(buffer_handle, if gpu_idle { None } else { Some(barrier) });
        }
        let deletion = super::types::PendingDeletion::Buffer { buffer: buffer.buffer };
        // Hot path: route to the owning context's per-context deletion queue so
        // the MTL buffer is freed as soon as the context's own timeline retires,
        // without waiting for device_retired (max over all contexts).
        // Falls back to the device-level queue (async GC safety net) when there
        // is no reclamation context installed on the current thread.
        let ctx_h = super::context::context_handle_for_thread(state, buffer.device_handle);
        if let Some(h) = ctx_h {
            if let Some(sc_arc) = state.contexts.get(&h) {
                sc_arc.lock().unwrap().deletion_queue.queue(barrier, deletion);
                return;
            }
        }
        if let Some(device) = state.devices.get(&buffer.device_handle) {
            device.deletion_queue.lock().unwrap().queue(barrier, deletion);
        }
    }
}

/// View tightly packed bytes in a CPU-writable storage buffer's shared mapping.
pub(super) fn cpu_writable_flat_slice<'a>(
    buffers: &'a std::collections::HashMap<BufferHandle, super::types::BufferState>,
    buffer_handle: BufferHandle,
    offset: u64,
    len: usize,
) -> Result<&'a [u8]> {
    let buffer = buffers
        .get(&buffer_handle)
        .context("cpu_writable_flat_slice: invalid buffer handle")?;
    if !buffer.flags.contains(BufferFlags::CPU_WRITABLE) {
        anyhow::bail!("cpu_writable_flat_slice: buffer is not CPU_WRITABLE");
    }
    if offset + len as u64 > buffer.size {
        anyhow::bail!("cpu_writable_flat_slice: slice exceeds buffer bounds");
    }
    Ok(unsafe { std::slice::from_raw_parts(buffer.buffer.contents().add(offset as usize) as *const u8, len) })
}

/// Write data to a buffer at the specified offset.
///
/// See [`clear`] for the full rationale. The short version: a CPU
/// `copy_nonoverlapping` on `contents()` is **not** queue-ordered with
/// subsequent compute dispatches, so we pair it with a queue-ordered blit
/// copy (from a transient staging buffer) so the next command buffer
/// submitted to this device queue is guaranteed to observe the written
/// bytes. The observable symptom in ekrano/velato was the `config` uniform
/// buffer returning stale `config.lines_size` to the binning shader, which
/// then flagged `STAGE_FLATTEN` overflow even when the actual flatten
/// output was tiny — only visible around scene transitions where the
/// previous frame's config had happened to be re-uploaded into the same
/// pool-recycled physical buffer.
pub(super) fn write(state: &MetalState, buffer_handle: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
    let buffer = state.buffers.get(&buffer_handle).context("Invalid buffer handle")?;

    if offset + data.len() as u64 > buffer.size {
        anyhow::bail!("Write would exceed buffer bounds");
    }

    if data.is_empty() {
        return Ok(());
    }

    let gpu_only = buffer.flags.contains(BufferFlags::GPU_ONLY);
    if !gpu_only {
        unsafe {
            let ptr = buffer.buffer.contents().add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        }
    }

    let device_handle = buffer.device_handle;
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    // Allocate a transient staging buffer (shared storage) populated with the
    // same bytes and emit a blit `copy_from_buffer` into the destination on
    // the device queue. Because this blit is committed to the same queue as
    // all subsequent compute dispatches, Metal serializes it with them and
    // the GPU is guaranteed to observe the new bytes. The staging buffer is
    // retained only by the command buffer's autorelease pool, so it is
    // dropped as soon as the blit completes.
    let staging = logical_device.device.new_buffer_with_data(
        data.as_ptr() as *const _,
        data.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(&staging, 0, &buffer.buffer, offset, data.len() as u64);
    blit.end_encoding();
    command_buffer.commit();

    Ok(())
}

/// Get the size of a buffer in bytes.
pub(super) fn size(state: &MetalState, buffer_handle: BufferHandle) -> u64 {
    state.buffers.get(&buffer_handle).map(|b| b.size).unwrap_or(0)
}

/// Get the bindless index for a buffer.
pub(super) fn bindless_index(state: &MetalState, buffer_handle: BufferHandle) -> Option<u32> {
    state.buffers.get(&buffer_handle).map(|b| b.arg_buffer_index)
}

/// Read buffer contents back to CPU memory.
/// Metal buffers use StorageModeShared so contents() is always valid.
pub(super) fn read_to_cpu(
    state: &MetalState,
    _device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    output: &mut [u8],
) -> Result<()> {
    let buffer = state.buffers.get(&buffer_handle).context("Invalid buffer handle")?;

    let len = output.len() as u64;
    if len > buffer.size {
        anyhow::bail!("Read would exceed buffer bounds");
    }

    if buffer.flags.contains(BufferFlags::GPU_ONLY) {
        let device_handle = buffer.device_handle;
        let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

        let staging = logical_device.device.new_buffer(
            len,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let command_buffer = logical_device.command_queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(&buffer.buffer, 0, &staging, 0, len);
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe {
            let ptr = staging.contents() as *const u8;
            std::ptr::copy_nonoverlapping(ptr, output.as_mut_ptr(), output.len());
        }
        return Ok(());
    }

    unsafe {
        let ptr = buffer.buffer.contents() as *const u8;
        std::ptr::copy_nonoverlapping(ptr, output.as_mut_ptr(), output.len());
    }

    Ok(())
}

/// Fill buffer region with zeros.
///
/// # Why both a CPU memset and a queue-ordered blit
///
/// Metal buffers on Apple Silicon use `StorageModeShared`, so the CPU and
/// GPU ultimately observe the same bytes of physical memory. But the two
/// paths have asymmetric visibility:
///
/// * A `contents()` + `write_bytes()` is **immediately** visible to any
///   subsequent CPU read of the same buffer (tests and readback paths rely
///   on this).
/// * That same CPU store is **not** automatically serialized with the next
///   command buffer submitted to the GPU queue. Without an explicit
///   queue-ordered operation on the buffer, a compute dispatch encoded into
///   a separately-built command buffer can read the GPU L2 cache's
///   pre-clear contents. Even if the caller has just waited on prior GPU
///   work, that wait establishes GPU→CPU ordering, not the reverse.
///
/// The observable symptom we hit in ekrano/velato was a `bump_buf` that had
/// just been memset to zero still appearing non-zero to the flatten shader,
/// which then flagged `STAGE_FLATTEN` overflow (`bump.lines >
/// config.lines_size`) and triggered an endless retry cascade.
///
/// We therefore do both:
///
/// 1. `write_bytes` so CPU-side readers observe the clear immediately.
/// 2. A `fillBuffer` blit committed to the same command queue so the next
///    compute dispatch on that queue is queue-ordered after the clear and
///    observes zeros.
///
/// The blit is committed without waiting — Metal's queue guarantees
/// ordering with subsequent command buffers, and whichever future waits on
/// the final fence will also drain this one.
pub(super) fn clear(
    state: &MetalState,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    offset: u64,
    size: u64,
) -> Result<()> {
    let buffer = state.buffers.get(&buffer_handle).context("Invalid buffer handle")?;

    let clear_size = super::super::shared::resolve_clear_size(buffer.size, offset, size);

    if offset + clear_size > buffer.size {
        anyhow::bail!("Clear would exceed buffer bounds");
    }

    if clear_size == 0 {
        return Ok(());
    }

    if !buffer.flags.contains(BufferFlags::GPU_ONLY) {
        unsafe {
            let ptr = (buffer.buffer.contents() as *mut u8).add(offset as usize);
            std::ptr::write_bytes(ptr, 0, clear_size as usize);
        }
    }

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    let range = mtl::NSRange::new(offset, clear_size);
    blit.fill_buffer(&buffer.buffer, range, 0);
    blit.end_encoding();
    command_buffer.commit();

    Ok(())
}

/// Allocate a shared-storage staging buffer for grant readback (no argument-buffer slot).
pub(super) fn alloc_readback_buffer(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    size: u64,
) -> Result<BufferHandle> {
    use metal as mtl;
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
    let buffer = logical_device.device.new_buffer(
        size,
        mtl::MTLResourceOptions::StorageModeShared | mtl::MTLResourceOptions::CPUCacheModeDefaultCache,
    );
    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;
    state.buffers.insert(
        handle,
        BufferState {
            device_handle,
            buffer,
            size,
            allocation_size: size,
            is_device_allocated: true,
            arg_buffer_index: 0,
            flags: BufferFlags::empty(),
            element_stride: None,
            parent_for_view: None,
            access: BufferKind::Scattered,
            view_byte_offset: None,
            is_grant_readback: true,
            texture_copy_footprint: None,
        },
    );
    Ok(handle)
}

pub(super) fn query_texture_copy_footprint(
    width: u32,
    height: u32,
    format: crate::types::TextureFormat,
) -> crate::backend::TextureCopyFootprint {
    let row_pitch = width.saturating_mul(format.bytes_per_pixel());
    let logical_bytes = row_pitch as u64 * height as u64;
    crate::backend::TextureCopyFootprint {
        width,
        height,
        format,
        logical_bytes,
        staging_bytes: logical_bytes,
        row_pitch,
        footprint_offset: 0,
    }
}

pub(super) fn alloc_texture_readback_staging(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    layout: crate::backend::TextureCopyFootprint,
) -> Result<BufferHandle> {
    let handle = alloc_readback_buffer(state, device_handle, layout.staging_bytes)?;
    if let Some(buf) = state.buffers.get_mut(&handle) {
        buf.texture_copy_footprint = Some(layout);
    }
    Ok(handle)
}

pub(super) fn read_texture_readback_staging(
    state: &MetalState,
    buffer_handle: BufferHandle,
    layout: crate::backend::TextureCopyFootprint,
    output: &mut [u8],
) -> Result<()> {
    if output.len() as u64 != layout.logical_bytes {
        anyhow::bail!("read_texture_readback_staging size mismatch");
    }
    let buffer = state.buffers.get(&buffer_handle).context("Invalid buffer handle")?;
    if !buffer.is_grant_readback {
        anyhow::bail!("read_texture_readback_staging requires a grant readback buffer");
    }
    let row_bytes = layout.tight_row_bytes() as usize;
    let pitch = layout.row_pitch as usize;
    unsafe {
        let ptr = buffer.buffer.contents() as *const u8;
        for row in 0..layout.height as usize {
            let src_offset = layout.footprint_offset as usize + row * pitch;
            let dst_offset = row * row_bytes;
            std::ptr::copy_nonoverlapping(ptr.add(src_offset), output.as_mut_ptr().add(dst_offset), row_bytes);
        }
    }
    Ok(())
}

/// Read bytes from a grant readback staging buffer.
pub(super) fn read_readback_buffer(state: &MetalState, buffer_handle: BufferHandle, output: &mut [u8]) -> Result<()> {
    let buffer = state.buffers.get(&buffer_handle).context("Invalid buffer handle")?;
    if !buffer.is_grant_readback {
        anyhow::bail!("read_readback_buffer requires a grant readback buffer");
    }
    if output.len() as u64 > buffer.size {
        anyhow::bail!("read_readback_buffer would exceed buffer bounds");
    }
    unsafe {
        let ptr = buffer.buffer.contents() as *const u8;
        std::ptr::copy_nonoverlapping(ptr, output.as_mut_ptr(), output.len());
    }
    Ok(())
}
