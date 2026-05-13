//! Buffer management logic.

use super::super::{BufferHandle, DeviceHandle};
use super::types::{BufferState, MetalState, ResourceRegistry, ARGUMENT_BUFFER_SIZE};
use crate::backend::DataAccess;
use crate::types::BufferFlags;
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::MTLResourceOptions;

/// Create a buffer with the given size and access pattern.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    size: u64,
    access: DataAccess,
    element_stride: Option<u32>,
    flags: BufferFlags,
) -> Result<BufferHandle> {
    let cpu_readable = flags.contains(BufferFlags::CPU_READABLE);
    let is_storage = access == DataAccess::Scattered;
    if cpu_readable && !is_storage {
        anyhow::bail!(
            "BufferFlags::CPU_READABLE is only valid for DataAccess::Scattered (storage) buffers"
        );
    }
    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    // Allocate buffer from heap allocator with Shared storage (CPU-accessible).
    let options =
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache;

    let buffer = logical_device
        .heap_allocator
        .allocate(size, options)
        .context("Metal buffer heap allocation failed — all heaps exhausted")?;

    // Register in bindless registry based on access pattern.
    // arg_buffer_index is the LOCAL shader slot (0-63 for both Scattered and Broadcast).
    // For encoding into the flat argument buffer, Broadcast buffers need the global index.
    let arg_buffer_index = match access {
        DataAccess::Broadcast => logical_device
            .resource_registry
            .register_uniform_buffer(handle),
        DataAccess::Scattered => logical_device
            .resource_registry
            .register_storage_buffer(handle),
    };
    let encoding_index = match access {
        DataAccess::Broadcast => ResourceRegistry::uniform_global_index(arg_buffer_index),
        DataAccess::Scattered => arg_buffer_index,
    };
    tracing::debug!(
        "Allocated buffer {} from heap at bindless index {}",
        handle,
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
            size,
            arg_buffer_index,
            flags,
            element_stride,
            last_gpu_use: 0,
        },
    );

    Ok(handle)
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

    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    let arg_buffer_index = logical_device
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
            arg_buffer_index,
            flags: parent_flags,
            element_stride,
            last_gpu_use: 0,
        },
    );

    Ok(handle)
}

/// Destroy a buffer, unregistering it from the bindless registry.
///
/// Slot recycling is gated on GPU idleness: if any previously-submitted
/// compute command buffer is still running, the slot parks in the registry's
/// pending list and only becomes reusable after the next `wait_fence()`
/// succeeds. This prevents the CPU from overwriting an argument-buffer
/// descriptor that an in-flight shader is about to dereference (descriptor
/// aliasing = random wrong-buffer reads and MTLCommandBufferError::Internal).
pub(super) fn destroy(state: &mut MetalState, buffer_handle: BufferHandle) {
    let gpu_idle = super::gpu_is_idle(state);
    if let Some(buffer) = state.buffers.remove(&buffer_handle) {
        if let Some(device) = state.devices.get_mut(&buffer.device_handle) {
            let barrier = device.timeline_scheduled_max;
            device
                .resource_registry
                .unregister_buffer(buffer_handle, if gpu_idle { None } else { Some(barrier) });
            device.deletion_queue.queue(
                barrier,
                super::types::PendingDeletion::Buffer {
                    buffer: buffer.buffer,
                },
            );
        }
    }
}

/// Write data to a buffer at the specified offset.
///
/// When the GPU has finished all work that references this buffer
/// (`signaled_value >= last_gpu_use`), the write is a plain CPU memcpy —
/// Metal guarantees visibility to the next committed command buffer for
/// `StorageModeShared` resources. When the buffer is still in flight, a
/// staging buffer + blit copy is committed to the queue to serialize the
/// write with respect to the in-flight read.
pub(super) fn write(
    state: &MetalState,
    buffer_handle: BufferHandle,
    offset: u64,
    data: &[u8],
) -> Result<()> {
    let buffer = state
        .buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    if offset + data.len() as u64 > buffer.size {
        anyhow::bail!("Write would exceed buffer bounds");
    }

    if data.is_empty() {
        return Ok(());
    }

    unsafe {
        let ptr = buffer.buffer.contents().add(offset as usize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
    }

    // Fast path: if the GPU has completed all submissions that ever read this
    // buffer, the memcpy above is sufficient — Apple's StorageModeShared
    // coherence guarantee ensures visibility at the next CB commit boundary.
    let device_handle = buffer.device_handle;
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let signaled = logical_device.timeline_event.as_ref().signaled_value();
    if signaled >= buffer.last_gpu_use {
        return Ok(());
    }

    // Slow path: buffer is still in flight. Use a staging blit to serialize
    // the write into the queue so the GPU observes correct data.
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
    state
        .buffers
        .get(&buffer_handle)
        .map(|b| b.size)
        .unwrap_or(0)
}

/// Get the bindless index for a buffer.
pub(super) fn bindless_index(state: &MetalState, buffer_handle: BufferHandle) -> Option<u32> {
    state
        .buffers
        .get(&buffer_handle)
        .map(|b| b.arg_buffer_index)
}

/// Read buffer contents back to CPU memory.
/// Metal buffers use StorageModeShared so contents() is always valid.
pub(super) fn read_to_cpu(
    state: &MetalState,
    _device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    output: &mut [u8],
) -> Result<()> {
    let buffer = state
        .buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    let len = output.len() as u64;
    if len > buffer.size {
        anyhow::bail!("Read would exceed buffer bounds");
    }

    unsafe {
        let ptr = buffer.buffer.contents() as *const u8;
        std::ptr::copy_nonoverlapping(ptr, output.as_mut_ptr(), output.len());
    }

    Ok(())
}

/// Fill buffer region with zeros.
///
/// When the GPU has finished all work that references this buffer
/// (`signaled_value >= last_gpu_use`), a CPU memset is sufficient.
/// Otherwise a queue-ordered `fillBuffer` blit is committed so the next
/// command buffer on the same queue observes zeros.
pub(super) fn clear(
    state: &MetalState,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    offset: u64,
    size: u64,
) -> Result<()> {
    let buffer = state
        .buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    let clear_size = super::super::shared::resolve_clear_size(buffer.size, offset, size);

    if offset + clear_size > buffer.size {
        anyhow::bail!("Clear would exceed buffer bounds");
    }

    if clear_size == 0 {
        return Ok(());
    }

    unsafe {
        let ptr = (buffer.buffer.contents() as *mut u8).add(offset as usize);
        std::ptr::write_bytes(ptr, 0, clear_size as usize);
    }

    // Fast path: buffer is not in flight, memset is visible at next CB commit.
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let signaled = logical_device.timeline_event.as_ref().signaled_value();
    if signaled >= buffer.last_gpu_use {
        return Ok(());
    }

    // Slow path: buffer still in flight — issue a queue-ordered fill.
    let command_buffer = logical_device.command_queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    let range = mtl::NSRange::new(offset, clear_size);
    blit.fill_buffer(&buffer.buffer, range, 0);
    blit.end_encoding();
    command_buffer.commit();

    Ok(())
}
