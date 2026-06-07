//! Texture management logic.

use super::super::{DeviceHandle, TextureHandle};
use super::types::{MetalState, ResourceRegistry, TextureState, ARGUMENT_BUFFER_SIZE};
use super::utils::format_to_mtl;
use crate::types::{TextureFlags, TextureFormat, TextureKind};
use ::metal as mtl;
use anyhow::{bail, Context, Result};
use mtl::{MTLOrigin, MTLRegion, MTLSize, MTLStorageMode, MTLTextureUsage, TextureDescriptor};

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
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;
        // Attempt 1: fast path.
        if let Some(tex) = logical_device.texture_heap.allocate(descriptor) {
            return Ok(tex);
        }
    }

    // Attempt 2 (non-blocking): drain signaled work, compact empty overflow heaps.
    {
        let _tz = crate::tracy_zone!("mtl.texture_heap_allocator.drain_reclaim");
        let retired = super::context::device_retired(state, device_handle);
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;
        logical_device.process_deletion_queue_up_to(retired);
        logical_device.texture_heap.compact_overflow();
    }
    {
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;
        if let Some(tex) = logical_device.texture_heap.allocate(descriptor) {
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
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;
        logical_device.process_deletion_queue_up_to(retired);
        logical_device.texture_heap.compact_overflow();
        if let Some(tex) = logical_device.texture_heap.allocate(descriptor) {
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
    descriptor.set_usage(mtl_usage);
    descriptor.set_storage_mode(MTLStorageMode::Shared);

    let texture = allocate_mtl_texture(state, device_handle, &descriptor)?;

    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    let is_storage_image = matches!(
        access,
        TextureKind::Direct | TextureKind::DirectInterpolated
    );
    let (arg_buffer_index, encoding_index) = if is_storage_image {
        let local = logical_device
            .ledger
            .lock()
            .unwrap()
            .resource_registry
            .register_storage_image(handle);
        (local, ResourceRegistry::storage_image_global_index(local))
    } else {
        let local = logical_device
            .ledger
            .lock()
            .unwrap()
            .resource_registry
            .register_texture(handle);
        (local, ResourceRegistry::texture_global_index(local))
    };

    // For DirectInterpolated, additionally register in the sampled-texture pool.
    let sampled_arg_buffer_index = if matches!(access, TextureKind::DirectInterpolated) {
        let local = logical_device
            .ledger
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

    tracing::debug!(
        "Created texture {} ({}x{}, {:?})",
        handle,
        width,
        height,
        format
    );
    Ok(handle)
}

/// Write data to a texture.
pub(super) fn write(
    state: &MetalState,
    texture_handle: TextureHandle,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<()> {
    let texture = state
        .textures
        .get(&texture_handle)
        .context("Invalid texture handle")?;

    let bytes_per_pixel = texture.format.bytes_per_pixel();
    let bytes_per_row = width * bytes_per_pixel;

    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };

    texture
        .texture
        .replace_region(region, 0, data.as_ptr() as *const _, bytes_per_row as u64);

    tracing::debug!(
        "Wrote {}x{} texture data ({} bytes)",
        width,
        height,
        data.len()
    );
    Ok(())
}

/// Write data to a subregion of a texture.
pub(super) fn write_region(
    state: &MetalState,
    texture_handle: TextureHandle,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<()> {
    let texture = state
        .textures
        .get(&texture_handle)
        .context("Invalid texture handle")?;

    let bytes_per_pixel = texture.format.bytes_per_pixel();
    let bytes_per_row = width * bytes_per_pixel;

    let region = MTLRegion {
        origin: MTLOrigin {
            x: x as u64,
            y: y as u64,
            z: 0,
        },
        size: MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };

    texture
        .texture
        .replace_region(region, 0, data.as_ptr() as *const _, bytes_per_row as u64);

    tracing::debug!(
        "Wrote {}x{} region at ({},{}) ({} bytes)",
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
pub(super) fn read_to_cpu(
    state: &MetalState,
    texture_handle: TextureHandle,
    output: &mut [u8],
) -> Result<()> {
    let texture = state
        .textures
        .get(&texture_handle)
        .context("Invalid texture handle")?;

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

    let staging_buffer = logical_device.device.new_buffer(
        expected_size as u64,
        mtl::MTLResourceOptions::StorageModeShared,
    );

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
/// As with buffer destroy, slot reuse is gated on GPU idleness: if any
/// in-flight command buffer might still reference this descriptor, the slot
/// parks on the pending list and is promoted by the next `wait_fence()`. This
/// is what keeps the glyph atlas (and other sampled textures) from flickering
/// when the renderer churns textures between frames.
pub(super) fn destroy(state: &mut MetalState, texture_handle: TextureHandle) {
    let gpu_idle = super::gpu_is_idle(state);
    if let Some(texture) = state.textures.remove(&texture_handle) {
        let device_handle = texture.device_handle;
        // Use the same reclamation_barrier logic as buffer::destroy so that
        // in-reclamation destroys get the tighter epoch rather than the wider
        // timeline_scheduled_max.
        let barrier = super::context::reclamation_barrier(state, device_handle, gpu_idle);
        let slot_barrier = if gpu_idle { None } else { Some(barrier) };
        if let Some(device) = state.devices.get_mut(&device_handle) {
            let mut ledger = device.ledger.lock().unwrap();
            ledger.resource_registry.unregister_texture(texture_handle);
            if !texture.slot_owned_externally {
                if texture.is_storage_image {
                    ledger
                        .resource_registry
                        .release_storage_image_slot(texture.arg_buffer_index, slot_barrier);
                } else {
                    ledger
                        .resource_registry
                        .release_texture_slot(texture.arg_buffer_index, slot_barrier);
                }
            }
        }
        let deletion = super::types::PendingDeletion::Texture {
            texture: texture.texture,
        };
        // Hot path: route to the owning context's per-context deletion queue.
        // Falls back to device-level queue (async GC safety net) when no
        // reclamation context is installed on the current thread.
        let ctx_h = super::context::context_handle_for_thread(state, device_handle);
        if let Some(h) = ctx_h {
            if let Some(sc) = state.contexts.get_mut(&h) {
                sc.deletion_queue.queue(barrier, deletion);
                return;
            }
        }
        if let Some(device) = state.devices.get_mut(&device_handle) {
            device.deletion_queue.queue(barrier, deletion);
        }
    }
}

/// Get the bindless index for a texture.
pub(super) fn bindless_index(state: &MetalState, texture_handle: TextureHandle) -> Option<u32> {
    state
        .textures
        .get(&texture_handle)
        .map(|t| t.arg_buffer_index)
}

/// Return the LOCAL sampled-texture-pool index for a `DirectInterpolated` texture.
/// Returns `None` for textures that don't have a secondary sampled-SRV slot.
pub(super) fn bindless_sampled_index(
    state: &MetalState,
    texture_handle: TextureHandle,
) -> Option<u32> {
    state
        .textures
        .get(&texture_handle)
        .and_then(|t| t.sampled_arg_buffer_index)
}
