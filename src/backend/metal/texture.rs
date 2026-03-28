//! Texture management logic.

use super::super::{DeviceHandle, TextureHandle};
use super::types::{MetalState, ResourceRegistry, TextureState, ARGUMENT_BUFFER_SIZE};
use super::utils::format_to_mtl;
use crate::types::{SpatialAccess, TextureFlags, TextureFormat};
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{MTLOrigin, MTLRegion, MTLSize, MTLStorageMode, MTLTextureUsage, TextureDescriptor};

/// Create a texture.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: SpatialAccess,
    flags: TextureFlags,
) -> Result<TextureHandle> {
    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    let handle = state.next_texture_handle;
    state.next_texture_handle += 1;

    let descriptor = TextureDescriptor::new();
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_pixel_format(format_to_mtl(format));

    let mut mtl_usage = mtl::MTLTextureUsage::Unknown;
    match access {
        SpatialAccess::Interpolated => {
            mtl_usage |= MTLTextureUsage::ShaderRead;
        }
        SpatialAccess::Direct => {
            mtl_usage |= MTLTextureUsage::ShaderWrite;
        }
    }
    if flags.contains(TextureFlags::RENDER_TARGET) {
        mtl_usage |= MTLTextureUsage::RenderTarget;
    }
    descriptor.set_usage(mtl_usage);
    descriptor.set_storage_mode(MTLStorageMode::Shared);

    let texture = logical_device
        .texture_heap
        .allocate(&descriptor)
        .context("Metal texture heap is full — all overflow heaps exhausted")?;

    let is_storage_image = matches!(access, SpatialAccess::Direct);
    let (arg_buffer_index, encoding_index) = if is_storage_image {
        let local = logical_device
            .resource_registry
            .register_storage_image(handle);
        (local, ResourceRegistry::storage_image_global_index(local))
    } else {
        let local = logical_device.resource_registry.register_texture(handle);
        (local, ResourceRegistry::texture_global_index(local))
    };
    tracing::debug!(
        "Allocated texture {} from heap at bindless local={} global={} storage_image={}",
        handle,
        arg_buffer_index,
        encoding_index,
        is_storage_image,
    );

    let encoded_length = logical_device.texture_encoder.encoded_length();
    let offset = (encoding_index as u64) * encoded_length;
    if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
        logical_device
            .texture_encoder
            .set_argument_buffer(&logical_device.argument_buffer, offset);
        logical_device.texture_encoder.set_texture(0, &texture);
        tracing::trace!(
            "Encoded texture {} at arg buffer offset {} (global slot {})",
            handle,
            offset,
            encoding_index
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

/// Destroy a texture.
pub(super) fn destroy(state: &mut MetalState, texture_handle: TextureHandle) {
    if let Some(texture) = state.textures.remove(&texture_handle) {
        if let Some(device) = state.devices.get_mut(&texture.device_handle) {
            device.resource_registry.unregister_texture(texture_handle);
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
