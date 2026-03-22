//! Buffer management logic.

use super::super::{BufferHandle, DeviceHandle};
use super::types::{BufferState, MetalState, ResourceRegistry, ARGUMENT_BUFFER_SIZE};
use crate::backend::DataAccess;
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::MTLResourceOptions;

/// Create a buffer with the given size and access pattern.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    size: u64,
    access: DataAccess,
    _element_stride: Option<u32>,
) -> Result<BufferHandle> {
    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    // Allocate buffer from heap with Shared storage (CPU-accessible).
    let options =
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache;

    let buffer = logical_device
        .buffer_heap
        .new_buffer(size, options)
        .context("Metal buffer heap is full — increase heap size")?;

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

    // Write the buffer's GPU address directly into the argument buffer.
    // We bypass ArgumentEncoder because it can defer writes internally,
    // causing a subsequent render pass to read a stale (zero) pointer.
    let encoded_length = logical_device.argument_encoder.encoded_length();
    let offset = (encoding_index as u64) * encoded_length;
    if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
        let gpu_addr = buffer.gpu_address();
        unsafe {
            let dst = (logical_device.argument_buffer.contents() as *mut u8).add(offset as usize);
            std::ptr::write_unaligned(dst as *mut u64, gpu_addr);
        }
        tracing::trace!(
            "Encoded buffer {} at arg buffer offset {} (slot {}, gpu_addr=0x{:x})",
            handle,
            offset,
            arg_buffer_index,
            gpu_addr,
        );
    }

    logical_device.heap_buffer_count += 1;

    state.buffers.insert(
        handle,
        BufferState {
            device_handle,
            buffer,
            size,
            arg_buffer_index,
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
        let gpu_addr = parent_mtl_buffer.gpu_address() + offset;
        unsafe {
            let dst =
                (logical_device.argument_buffer.contents() as *mut u8).add(ab_offset as usize);
            std::ptr::write_unaligned(dst as *mut u64, gpu_addr);
        }
    }

    state.buffers.insert(
        handle,
        BufferState {
            device_handle,
            buffer: parent_mtl_buffer,
            size,
            arg_buffer_index,
        },
    );

    Ok(handle)
}

/// Destroy a buffer, unregistering it from the bindless registry.
pub(super) fn destroy(state: &mut MetalState, buffer_handle: BufferHandle) {
    if let Some(buffer) = state.buffers.remove(&buffer_handle) {
        if let Some(device) = state.devices.get_mut(&buffer.device_handle) {
            device.resource_registry.unregister_buffer(buffer_handle);
        }
    }
}

/// Write data to a buffer at the specified offset.
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

    unsafe {
        let ptr = buffer.buffer.contents().add(offset as usize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
    }

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
/// Metal buffers use StorageModeShared so we can memset via contents().
pub(super) fn clear(
    state: &MetalState,
    _device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    offset: u64,
    size: u64,
) -> Result<()> {
    let buffer = state
        .buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    let clear_size = if size == 0 {
        buffer.size.saturating_sub(offset)
    } else {
        size
    };

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

    Ok(())
}
