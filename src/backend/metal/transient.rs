//! Transient task-graph heap: aliasable buffers/textures in one [`mtl::Heap`].

use super::super::{
    BufferHandle, DeviceHandle, TextureHandle, TransientHeapAlignments, TransientHeapHandle,
};
use super::types::{
    BufferState, LogicalDevice, MetalState, ResourceRegistry, TextureState, TransientHeapTracking,
    ARGUMENT_BUFFER_SIZE,
};
use super::utils::format_to_mtl;
use crate::types::{SpatialAccess, TextureFlags, TextureFormat};
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{
    HeapDescriptor, MTLCPUCacheMode, MTLHazardTrackingMode, MTLHeapType, MTLResourceOptions,
    MTLStorageMode, MTLTextureUsage, TextureDescriptor,
};

pub(super) fn use_transient_heaps_for_compute(
    logical_device: &LogicalDevice,
    encoder: &mtl::ComputeCommandEncoderRef,
) {
    for t in logical_device.transient_heaps.values() {
        encoder.use_heap(&t.heap);
    }
}

pub(super) fn use_transient_heaps_for_render(
    logical_device: &LogicalDevice,
    encoder: &mtl::RenderCommandEncoderRef,
    stages: mtl::MTLRenderStages,
) {
    for t in logical_device.transient_heaps.values() {
        encoder.use_heap_at(&t.heap, stages);
    }
}

pub(super) fn transient_heap_alignment_hints(
    _state: &MetalState,
    _device: DeviceHandle,
) -> TransientHeapAlignments {
    TransientHeapAlignments {
        buffer_base_align: 256,
        texture_base_align: 4096,
        buffer_image_granularity: 4096,
    }
}

pub(super) fn transient_texture_heap_footprint(
    state: &MetalState,
    device: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: SpatialAccess,
    flags: TextureFlags,
) -> Result<(u64, u64)> {
    let logical = state
        .devices
        .get(&device)
        .context("Invalid device handle")?;
    let descriptor = transient_texture_descriptor(width, height, format, access, flags);
    let sa = logical.device.heap_texture_size_and_align(&descriptor);
    Ok((sa.align, sa.size))
}

fn transient_texture_descriptor(
    width: u32,
    height: u32,
    format: TextureFormat,
    access: SpatialAccess,
    flags: TextureFlags,
) -> TextureDescriptor {
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
            mtl_usage |= MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead;
        }
    }
    if flags.contains(TextureFlags::RENDER_TARGET) {
        mtl_usage |= MTLTextureUsage::RenderTarget;
    }
    descriptor.set_usage(mtl_usage);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor
}

pub(super) fn create_transient_heap(
    state: &mut MetalState,
    device: DeviceHandle,
    size: u64,
) -> Result<Option<TransientHeapHandle>> {
    if size == 0 {
        return Ok(None);
    }
    let h = state.next_transient_heap_handle;
    state.next_transient_heap_handle += 1;
    let ld = state
        .devices
        .get_mut(&device)
        .context("Invalid device handle")?;
    let mtl_dev = ld.device.clone();
    let desc = HeapDescriptor::new();
    desc.set_size(size);
    desc.set_storage_mode(MTLStorageMode::Shared);
    desc.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);
    desc.set_heap_type(MTLHeapType::Automatic);
    desc.set_hazard_tracking_mode(MTLHazardTrackingMode::Untracked);
    let heap = mtl_dev.new_heap(&desc);
    ld.transient_heaps.insert(
        h,
        TransientHeapTracking {
            heap,
            placed_buffers: Vec::new(),
            placed_textures: Vec::new(),
        },
    );
    tracing::debug!("Metal transient heap {h} size={size}");
    Ok(Some(h))
}

pub(super) fn place_buffer_in_transient_heap(
    state: &mut MetalState,
    device: DeviceHandle,
    heap_h: TransientHeapHandle,
    offset: u64,
    size: u64,
) -> Result<BufferHandle> {
    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    let (buffer, arg_buffer_index) = {
        let ld = state
            .devices
            .get_mut(&device)
            .context("Invalid device handle")?;
        let tracking = ld
            .transient_heaps
            .get_mut(&heap_h)
            .with_context(|| format!("invalid transient heap {heap_h}"))?;
        let options =
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache;
        let buffer = tracking
            .heap
            .new_buffer_with_offset(size, options, offset)
            .context("Metal transient heap buffer placement failed")?;

        let arg_buffer_index = ld.resource_registry.register_storage_buffer(handle);
        let encoding_index = arg_buffer_index;
        let encoded_length = ld.argument_encoder.encoded_length();
        let ab_off = (encoding_index as u64) * encoded_length;
        if ab_off + encoded_length <= ARGUMENT_BUFFER_SIZE {
            ld.argument_encoder
                .set_argument_buffer(&ld.argument_buffer, ab_off);
            ld.argument_encoder.set_buffer(0, &buffer, 0);
        }
        tracking.placed_buffers.push(handle);
        (buffer, arg_buffer_index)
    };

    state.buffers.insert(
        handle,
        BufferState {
            device_handle: device,
            buffer,
            size,
            arg_buffer_index,
            flags: crate::types::BufferFlags::empty(),
        },
    );
    Ok(handle)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn place_texture_in_transient_heap(
    state: &mut MetalState,
    device: DeviceHandle,
    heap_h: TransientHeapHandle,
    offset: u64,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: SpatialAccess,
    flags: TextureFlags,
) -> Result<TextureHandle> {
    let handle = state.next_texture_handle;
    state.next_texture_handle += 1;

    let (texture, arg_buffer_index, is_storage_image) = {
        let ld = state
            .devices
            .get_mut(&device)
            .context("Invalid device handle")?;
        let tracking = ld
            .transient_heaps
            .get_mut(&heap_h)
            .with_context(|| format!("invalid transient heap {heap_h}"))?;

        let descriptor = transient_texture_descriptor(width, height, format, access, flags);
        let texture = tracking
            .heap
            .new_texture_with_offset(&descriptor, offset)
            .context("Metal transient heap texture placement failed")?;

        let is_storage_image = matches!(access, SpatialAccess::Direct);
        let (arg_buffer_index, encoding_index) = if is_storage_image {
            let local = ld.resource_registry.register_storage_image(handle);
            (local, ResourceRegistry::storage_image_global_index(local))
        } else {
            let local = ld.resource_registry.register_texture(handle);
            (local, ResourceRegistry::texture_global_index(local))
        };

        let encoder = if is_storage_image {
            &ld.storage_image_encoder
        } else {
            &ld.texture_encoder
        };
        let encoded_length = encoder.encoded_length();
        let ab_off = (encoding_index as u64) * encoded_length;
        if ab_off + encoded_length <= ARGUMENT_BUFFER_SIZE {
            encoder.set_argument_buffer(&ld.argument_buffer, ab_off);
            encoder.set_texture(0, &texture);
        }
        tracking.placed_textures.push(handle);
        (texture, arg_buffer_index, is_storage_image)
    };

    state.textures.insert(
        handle,
        TextureState {
            device_handle: device,
            width,
            height,
            format,
            texture,
            arg_buffer_index,
            is_storage_image,
            slot_owned_externally: false,
        },
    );
    Ok(handle)
}

pub(super) fn destroy_transient_heap(
    state: &mut MetalState,
    device: DeviceHandle,
    heap_h: TransientHeapHandle,
) -> Result<()> {
    let mut tracking = {
        let ld = state
            .devices
            .get_mut(&device)
            .context("Invalid device handle")?;
        ld.transient_heaps
            .remove(&heap_h)
            .with_context(|| format!("invalid transient heap {heap_h}"))?
    };

    for bh in tracking.placed_buffers.drain(..) {
        super::buffer::destroy(state, bh);
    }
    for th in tracking.placed_textures.drain(..) {
        super::texture::destroy(state, th);
    }
    drop(tracking.heap);
    Ok(())
}

pub(super) fn destroy_transient_heaps_for_device(state: &mut MetalState, device: DeviceHandle) {
    let ids: Vec<_> = if let Some(ld) = state.devices.get(&device) {
        ld.transient_heaps.keys().copied().collect()
    } else {
        return;
    };
    for h in ids {
        let _ = destroy_transient_heap(state, device, h);
    }
}
