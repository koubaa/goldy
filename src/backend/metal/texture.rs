//! Texture management logic.

use super::super::{DeviceHandle, GpuCommand, TextureHandle};
use super::types::{MetalState, ResourceRegistry, TextureState, ARGUMENT_BUFFER_SIZE};
use super::utils::format_to_mtl;
use crate::types::{TextureFlags, TextureFormat, TextureKind};
use ::metal as mtl;
use anyhow::{bail, Context, Result};
use mtl::{MTLOrigin, MTLSize, MTLStorageMode, MTLTextureUsage, TextureDescriptor};
use std::sync::Arc;
use std::time::Duration;

const UPLOAD_WAIT_TIMEOUT: Duration = Duration::from_secs(60);

fn submit_texture_upload_sync(state: &mut MetalState, device_handle: DeviceHandle, command: GpuCommand) -> Result<()> {
    let ctx = super::context::create(state, device_handle)?;
    let result = (|| {
        let signal = super::compute::submit(state, ctx, std::slice::from_ref(&command), None)?;
        if !super::context::wait_until_device_seq_at_least(state, device_handle, signal, UPLOAD_WAIT_TIMEOUT) {
            anyhow::bail!("Timed out waiting for texture upload to complete");
        }
        Ok(())
    })();
    super::context::destroy(state, ctx);
    result
}

/// Texture heap allocation with drain-and-retry self-regulation.
///
/// When the texture heap is saturated (overflow cap reached), performs a non-blocking
/// drain-and-retry first, then if still full, waits for the oldest in-flight command
/// buffer to complete, drains again, and retries. Mirrors the buffer heap self-regulation
/// in `allocate_mtl_storage_buffer`.
fn allocate_mtl_texture(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    descriptor: &mtl::TextureDescriptorRef,
) -> Result<mtl::Texture> {
    {
        let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
        // Attempt 1: fast path.
        if let Some(tex) = logical_device.texture_heap.lock().unwrap().allocate(descriptor) {
            return Ok(tex);
        }
    }

    // Attempt 2 (non-blocking): drain signaled work, compact empty overflow heaps.
    {
        let _tz = crate::tracy_zone!("mtl.texture_heap_allocator.drain_reclaim");
        let retired = super::context::device_retired(state, device_handle);
        let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
        let completed = super::context::snapshot_context_completed_values(state, device_handle);
        logical_device.process_deletion_queue_up_to(retired, Some(&completed));
        logical_device.texture_heap.lock().unwrap().compact_overflow();
    }
    {
        let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
        if let Some(tex) = logical_device.texture_heap.lock().unwrap().allocate(descriptor) {
            return Ok(tex);
        }
    }

    // Attempt 3 (blocking): wait on the oldest in-flight CB to reclaim one frame's
    // worth of archive — a runtime boundary action, invisible to the caller.
    let oldest_cb = super::context::oldest_in_flight_cb(state, device_handle);

    if let Some(cb) = oldest_cb {
        let _tz = crate::tracy_zone!("mtl.texture_heap_allocator.wait_reclaim");
        tracing::debug!(
            "Metal texture heap saturated — waiting for oldest in-flight command buffer \
             to reclaim archive",
        );
        cb.wait_until_completed();
        let retired = super::context::device_retired(state, device_handle);
        let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
        let completed = super::context::snapshot_context_completed_values(state, device_handle);
        logical_device.process_deletion_queue_up_to(retired, Some(&completed));
        let mut th = logical_device.texture_heap.lock().unwrap();
        th.compact_overflow();
        if let Some(tex) = th.allocate(descriptor) {
            return Ok(tex);
        }
    }

    crate::signal::push_sync_signal(crate::signal::Signal::Oversubscribed {
        reason: crate::signal::OversubscribedReason::TextureHeap,
        size_hint: descriptor.width() * descriptor.height(),
    });
    bail!("Metal texture heap is full — all overflow heaps exhausted");
}

/// Create a texture.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: TextureKind,
    flags: TextureFlags,
) -> Result<TextureHandle> {
    let handle = state.next_texture_handle;
    state.next_texture_handle += 1;

    let descriptor = TextureDescriptor::new();
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_pixel_format(format_to_mtl(format));

    let mut mtl_usage = mtl::MTLTextureUsage::Unknown;
    match access {
        TextureKind::Interpolated => {
            mtl_usage |= MTLTextureUsage::ShaderRead;
        }
        TextureKind::Direct => {
            // Storage images (RWTexture2D / DirectSpatial) need both read and
            // write usage bits. Metal's `texture2d<T, access::read_write>` —
            // which Slang emits for DirectSpatial — requires ShaderRead even
            // when the shader only writes, and filter passes (filter_pass.slang)
            // do read from src/dst via integer loads. Without ShaderRead the
            // GPU faults on the first dispatch that reads the texture, which
            // then cascades into kIOGPUCommandBufferCallbackErrorSubmissionsIgnored
            // on every subsequent frame.
            mtl_usage |= MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead;
        }
        TextureKind::DirectInterpolated => {
            // Dual-access: writable as a storage image (UAV) and readable via
            // hardware sampling (SRV). Needs ShaderRead | ShaderWrite.
            mtl_usage |= MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead;
        }
    }
    if flags.contains(TextureFlags::RENDER_TARGET) {
        mtl_usage |= MTLTextureUsage::RenderTarget;
    }
    // COPY_SRC: Metal's blit encoder can copy any StorageModeShared texture without
    // a special usage bit, but we mirror the Vulkan TRANSFER_SRC convention so that
    // internal assertions stay consistent across backends.
    if flags.contains(TextureFlags::COPY_SRC) {
        mtl_usage |= MTLTextureUsage::ShaderRead;
    }
    descriptor.set_usage(mtl_usage);
    descriptor.set_storage_mode(MTLStorageMode::Shared);

    let texture = allocate_mtl_texture(state, device_handle, &descriptor)?;

    let logical_device = state.devices.get_mut(&device_handle).context("Invalid device handle")?;

    let is_storage_image = matches!(access, TextureKind::Direct | TextureKind::DirectInterpolated);
    let (arg_buffer_index, encoding_index) = if is_storage_image {
        let local = logical_device
            .descriptors
            .lock()
            .unwrap()
            .resource_registry
            .register_storage_image(handle);
        (local, ResourceRegistry::storage_image_global_index(local))
    } else {
        let local = logical_device
            .descriptors
            .lock()
            .unwrap()
            .resource_registry
            .register_texture(handle);
        (local, ResourceRegistry::texture_global_index(local))
    };

    // For DirectInterpolated, additionally register in the sampled-texture pool.
    let sampled_arg_buffer_index = if matches!(access, TextureKind::DirectInterpolated) {
        let local = logical_device
            .descriptors
            .lock()
            .unwrap()
            .resource_registry
            .register_texture(handle);
        let global = ResourceRegistry::texture_global_index(local);
        let enc = &logical_device.texture_encoder;
        let encoded_length = enc.encoded_length();
        let offset = (global as u64) * encoded_length;
        if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
            enc.set_argument_buffer(&logical_device.argument_buffer, offset);
            enc.set_texture(0, &texture);
        }
        Some(local)
    } else {
        None
    };
    tracing::debug!(
        "Allocated texture {} from heap at bindless local={} global={} storage_image={}",
        handle,
        arg_buffer_index,
        encoding_index,
        is_storage_image,
    );

    // Use the appropriate encoder: storage images need ReadWrite access in the
    // argument buffer descriptor, sampled textures only need ReadOnly.
    let encoder = if is_storage_image {
        &logical_device.storage_image_encoder
    } else {
        &logical_device.texture_encoder
    };
    let encoded_length = encoder.encoded_length();
    let offset = (encoding_index as u64) * encoded_length;
    if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
        encoder.set_argument_buffer(&logical_device.argument_buffer, offset);
        encoder.set_texture(0, &texture);
        tracing::trace!(
            "Encoded texture {} at arg buffer offset {} (global slot {}, storage_image={})",
            handle,
            offset,
            encoding_index,
            is_storage_image,
        );
    }

    state.textures.insert(
        handle,
        TextureState {
            device_handle,
            width,
            height,
            format,
            texture,
            arg_buffer_index,
            sampled_arg_buffer_index,
            is_storage_image,
            slot_owned_externally: false,
            is_heap_allocated: true,
        },
    );

    tracing::debug!("Created texture {} ({}x{}, {:?})", handle, width, height, format);
    Ok(handle)
}

/// Write data to a texture (synchronous: staging buffer + blit, then wait).
pub(super) fn write(
    state: &mut MetalState,
    texture_handle: TextureHandle,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<()> {
    let device_handle = state
        .textures
        .get(&texture_handle)
        .context("Invalid texture handle")?
        .device_handle;
    let texture = state.textures.get(&texture_handle).context("Invalid texture handle")?;
    let bytes_per_pixel = texture.format.bytes_per_pixel();
    let expected = (width as usize) * (height as usize) * (bytes_per_pixel as usize);
    anyhow::ensure!(
        data.len() == expected,
        "WriteTexture: expected {} bytes for {}x{}, got {}",
        expected,
        width,
        height,
        data.len()
    );
    anyhow::ensure!(
        width == texture.width && height == texture.height,
        "WriteTexture: dimension mismatch"
    );

    submit_texture_upload_sync(
        state,
        device_handle,
        GpuCommand::WriteTexture {
            texture: texture_handle,
            data: Arc::from(data),
            width,
            height,
        },
    )?;

    tracing::debug!(
        "Wrote {}x{} texture data ({} bytes, sync blit upload)",
        width,
        height,
        data.len()
    );
    Ok(())
}

/// Write data to a subregion of a texture (synchronous: staging buffer + blit, then wait).
pub(super) fn write_region(
    state: &mut MetalState,
    texture_handle: TextureHandle,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<()> {
    let device_handle = state
        .textures
        .get(&texture_handle)
        .context("Invalid texture handle")?
        .device_handle;
    let texture = state.textures.get(&texture_handle).context("Invalid texture handle")?;
    let bytes_per_pixel = texture.format.bytes_per_pixel();
    let expected = (width as usize) * (height as usize) * (bytes_per_pixel as usize);
    anyhow::ensure!(
        data.len() == expected,
        "WriteTextureRegion: expected {} bytes, got {}",
        expected,
        data.len()
    );
    anyhow::ensure!(
        x + width <= texture.width && y + height <= texture.height,
        "WriteTextureRegion: region out of bounds"
    );

    submit_texture_upload_sync(
        state,
        device_handle,
        GpuCommand::WriteTextureRegion {
            texture: texture_handle,
            x,
            y,
            width,
            height,
            data: Arc::from(data),
        },
    )?;

    tracing::debug!(
        "Wrote {}x{} region at ({},{}) ({} bytes, sync blit upload)",
        width,
        height,
        x,
        y,
        data.len()
    );
    Ok(())
}

/// Read texture contents to CPU memory.
/// The texture must have been created with TextureFlags::COPY_SRC.
pub(super) fn read_to_cpu(state: &MetalState, texture_handle: TextureHandle, output: &mut [u8]) -> Result<()> {
    let texture = state.textures.get(&texture_handle).context("Invalid texture handle")?;

    let logical_device = state
        .devices
        .get(&texture.device_handle)
        .context("Device no longer valid")?;

    let width = texture.width;
    let height = texture.height;
    let bytes_per_pixel = texture.format.bytes_per_pixel();
    let bytes_per_row = width * bytes_per_pixel;
    let expected_size = (bytes_per_row * height) as usize;

    if output.len() < expected_size {
        anyhow::bail!(
            "Output buffer too small: need {} bytes, got {}",
            expected_size,
            output.len()
        );
    }

    let staging_buffer = logical_device
        .device
        .new_buffer(expected_size as u64, mtl::MTLResourceOptions::StorageModeShared);

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let blit_encoder = command_buffer.new_blit_command_encoder();

    blit_encoder.copy_from_texture_to_buffer(
        &texture.texture,
        0,
        0,
        MTLOrigin { x: 0, y: 0, z: 0 },
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        &staging_buffer,
        0,
        bytes_per_row as u64,
        (bytes_per_row * height) as u64,
        mtl::MTLBlitOption::empty(),
    );

    blit_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let ptr = staging_buffer.contents();
        std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), expected_size);
    }

    Ok(())
}

/// Destroy a texture and return its bindless slot to the registry's free list.
///
/// If `slot_owned_externally` is set (e.g. a swapchain drawable whose slot is
/// owned by `SurfaceState`), the slot is NOT released here — the owner manages
/// it across frames.
///
/// Slot reclaim is gated on `slot_last_seen` epochs stamped at submit time.
pub(super) fn destroy(state: &mut MetalState, texture_handle: TextureHandle) {
    let gpu_idle = super::gpu_is_idle(state);
    if let Some(texture) = state.textures.remove(&texture_handle) {
        let device_handle = texture.device_handle;
        let ctx_h = super::context::context_handle_for_thread(state, device_handle);
        let base_barrier = super::context::reclamation_barrier(state, device_handle, gpu_idle);
        let key = if texture.is_storage_image {
            super::types::MetalSlotKey::StorageImage(texture.arg_buffer_index)
        } else {
            super::types::MetalSlotKey::Texture(texture.arg_buffer_index)
        };
        let barrier = if let Some(device) = state.devices.get(&device_handle) {
            let mut registry = device.descriptors.lock().unwrap();
            registry.unregister_texture(texture_handle);
            let mut barrier = base_barrier;
            if !texture.slot_owned_externally {
                if let Some(map) = registry.slot_last_seen.get(&key) {
                    barrier = barrier.max(map.values().copied().max().unwrap_or(0));
                }
                registry.reclaim_texture_slot(key);
            }
            barrier
        } else {
            base_barrier
        };
        let deletion = super::types::PendingDeletion::Texture {
            texture: texture.texture,
        };
        if let Some(h) = ctx_h {
            if let Some(sc_arc) = state.contexts.get(&h) {
                sc_arc.lock().unwrap().deletion_queue.queue(barrier, deletion);
                return;
            }
        }
        if let Some(device) = state.devices.get(&device_handle) {
            device.deletion_queue.lock().unwrap().queue(barrier, deletion);
        }
    }
}

/// Get the bindless index for a texture.
pub(super) fn bindless_index(state: &MetalState, texture_handle: TextureHandle) -> Option<u32> {
    state.textures.get(&texture_handle).map(|t| t.arg_buffer_index)
}

/// Return the LOCAL sampled-texture-pool index for a `DirectInterpolated` texture.
/// Returns `None` for textures that don't have a secondary sampled-SRV slot.
pub(super) fn bindless_sampled_index(state: &MetalState, texture_handle: TextureHandle) -> Option<u32> {
    state
        .textures
        .get(&texture_handle)
        .and_then(|t| t.sampled_arg_buffer_index)
}
