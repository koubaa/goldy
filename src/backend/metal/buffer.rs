//! Buffer management logic.

use super::super::{BufferHandle, DeviceHandle};
use super::types::{BufferState, MetalState, ResourceRegistry, ARGUMENT_BUFFER_SIZE};
use crate::backend::DataAccess;
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::MTLResourceOptions;
use std::collections::HashMap;

/// Create a buffer with the given size and access pattern.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    size: u64,
    access: DataAccess,
    element_stride: Option<u32>,
) -> Result<BufferHandle> {
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

    state.buffers.insert(
        handle,
        BufferState {
            device_handle,
            buffer,
            size,
            arg_buffer_index,
            access,
            element_stride,
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
            access: DataAccess::Scattered,
            element_stride,
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
            device
                .resource_registry
                .unregister_buffer(buffer_handle, !gpu_idle);
        }
    }
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

    let device_handle = buffer.device_handle;
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

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

/// Effective structured-buffer element stride for `GOLDY_VALIDATE_BUFFER_STRIDES` checks.
pub(super) fn element_stride_for_bindless_handle_map(
    buffers: &HashMap<BufferHandle, BufferState>,
    handle: crate::types::BindlessHandle,
) -> Option<u32> {
    use crate::types::{BindlessCategory, DataAccess};
    let want_access = match handle.category() {
        BindlessCategory::Scattered => DataAccess::Scattered,
        BindlessCategory::Broadcast => DataAccess::Broadcast,
        _ => return None,
    };
    let idx = handle.index();
    for b in buffers.values() {
        if b.access != want_access || b.arg_buffer_index != idx {
            continue;
        }
        if b.access == DataAccess::Scattered {
            return Some(b.element_stride.unwrap_or(4));
        }
        return b.element_stride;
    }
    None
}

/// See [`element_stride_for_bindless_handle_map`].
pub(super) fn element_stride_for_bindless_handle(
    state: &MetalState,
    handle: crate::types::BindlessHandle,
) -> Option<u32> {
    element_stride_for_bindless_handle_map(&state.buffers, handle)
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

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    let range = mtl::NSRange::new(offset, clear_size);
    blit.fill_buffer(&buffer.buffer, range, 0);
    blit.end_encoding();
    command_buffer.commit();

    Ok(())
}
