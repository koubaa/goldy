//! Texture management logic.

use super::super::{DeviceHandle, TextureHandle};
use super::types::{MetalState, TextureState, ARGUMENT_BUFFER_SIZE};
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
        .new_texture(&descriptor)
        .context("Metal texture heap is full — increase heap size")?;

    let arg_buffer_index = logical_device.resource_registry.register_texture(handle);
    tracing::debug!(
        "Allocated texture {} from heap at bindless index {}",
        handle,
        arg_buffer_index
    );

    let encoded_length = logical_device.texture_encoder.encoded_length();
    let offset = (arg_buffer_index as u64) * encoded_length;
    if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
        logical_device
            .texture_encoder
            .set_argument_buffer(&logical_device.argument_buffer, offset);
        logical_device.texture_encoder.set_texture(0, &texture);
        tracing::trace!(
            "Encoded texture {} at arg buffer offset {} (slot {})",
            handle,
            offset,
            arg_buffer_index
        );
    }

    logical_device.heap_texture_count += 1;

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
