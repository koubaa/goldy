//! Buffer management logic.

use super::super::{BufferHandle, DeviceHandle};
use super::types::{BufferState, MetalState, ARGUMENT_BUFFER_SIZE};
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
    let arg_buffer_index = match access {
        DataAccess::Broadcast => logical_device
            .resource_registry
            .register_uniform_buffer(handle),
        DataAccess::Scattered => logical_device
            .resource_registry
            .register_storage_buffer(handle),
    };
    tracing::debug!(
        "Allocated buffer {} from heap at bindless index {}",
        handle,
        arg_buffer_index
    );

    // Encode buffer into argument buffer using ArgumentEncoder
    let encoded_length = logical_device.argument_encoder.encoded_length();
    let offset = (arg_buffer_index as u64) * encoded_length;
    if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
        logical_device
            .argument_encoder
            .set_argument_buffer(&logical_device.argument_buffer, offset);
        logical_device.argument_encoder.set_buffer(0, &buffer, 0);
        tracing::trace!(
            "Encoded buffer {} at arg buffer offset {} (slot {})",
            handle,
            offset,
            arg_buffer_index
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
