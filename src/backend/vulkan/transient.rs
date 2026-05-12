//! One [`vk::DeviceMemory`] block with sub-allocations bound via Vulkan `vkBindBufferMemory` /
//! `vkBindImageMemory` at byte offsets.

use super::texture;
use super::types::{self, TransientHeapEntry};
use super::utils::{find_memory_type, format_to_vk};
use super::{BufferHandle, DeviceHandle, TextureHandle, TransientHeapHandle, VulkanState};
use crate::types::{SpatialAccess, TextureFlags, TextureFormat};
use anyhow::{Context, Result};
use ash::vk;

fn pick_heap_memory_type(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
) -> Result<u32> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(4096)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .context("transient heap probe buffer")?;
    let buf_req = unsafe { device.get_buffer_memory_requirements(buffer) };
    unsafe { device.destroy_buffer(buffer, None) };

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width: 64,
            height: 64,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image =
        unsafe { device.create_image(&image_info, None) }.context("transient heap probe image")?;
    let img_req = unsafe { device.get_image_memory_requirements(image) };
    unsafe { device.destroy_image(image, None) };

    let mask = buf_req.memory_type_bits & img_req.memory_type_bits;
    anyhow::ensure!(
        mask != 0,
        "no Vulkan memory type for both buffer and image in transient heap"
    );

    find_memory_type(
        instance,
        physical_device,
        mask,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .context("transient heap memory type")
}

pub(super) fn transient_heap_alignment_hints(
    state: &VulkanState,
    device: DeviceHandle,
) -> crate::backend::TransientHeapAlignments {
    let Some(ld) = state.devices.get(&device) else {
        return crate::backend::TransientHeapAlignments::default();
    };
    let limits = unsafe {
        state
            .instance
            .get_physical_device_properties(ld.physical_device)
    }
    .limits;
    crate::backend::TransientHeapAlignments {
        buffer_base_align: 256u64.max(limits.min_storage_buffer_offset_alignment as u64),
        texture_base_align: 4096,
        buffer_image_granularity: limits.buffer_image_granularity.max(1) as u64,
    }
}

pub(super) fn transient_texture_heap_footprint(
    state: &VulkanState,
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
    let mut vk_usage = vk::ImageUsageFlags::TRANSFER_DST;
    match access {
        SpatialAccess::Interpolated => {
            vk_usage |= vk::ImageUsageFlags::SAMPLED;
        }
        SpatialAccess::Direct => {
            vk_usage |= vk::ImageUsageFlags::STORAGE;
        }
    }
    if flags.contains(TextureFlags::RENDER_TARGET) {
        vk_usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if flags.contains(TextureFlags::COPY_SRC) {
        vk_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format_to_vk(format))
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { logical.device.create_image(&image_info, None) }
        .context("transient footprint image")?;
    let mem_reqs = unsafe { logical.device.get_image_memory_requirements(image) };
    unsafe { logical.device.destroy_image(image, None) };
    Ok((mem_reqs.alignment as u64, mem_reqs.size))
}

pub(super) fn create_transient_heap(
    state: &mut VulkanState,
    device: DeviceHandle,
    size: u64,
) -> Result<Option<TransientHeapHandle>> {
    if size == 0 {
        return Ok(None);
    }
    let ld = state
        .devices
        .get(&device)
        .context("Invalid device handle")?;
    let mt = pick_heap_memory_type(&state.instance, &ld.device, ld.physical_device)?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(mt);
    let memory = unsafe { ld.device.allocate_memory(&alloc, None) }
        .context("vkAllocateMemory transient heap")?;
    let h = state.next_transient_heap_handle;
    state.next_transient_heap_handle += 1;
    state.transient_heaps.insert(
        h,
        TransientHeapEntry {
            device_handle: device,
            memory,
            size,
            buffers: Vec::new(),
            textures: Vec::new(),
        },
    );
    Ok(Some(h))
}

pub(super) fn place_buffer_in_transient_heap(
    state: &mut VulkanState,
    device: DeviceHandle,
    heap: TransientHeapHandle,
    offset: u64,
    size: u64,
) -> Result<BufferHandle> {
    let shared_mem = {
        let e = state
            .transient_heaps
            .get(&heap)
            .with_context(|| format!("invalid transient heap {heap}"))?;
        anyhow::ensure!(e.device_handle == device);
        e.memory
    };

    let logical = state
        .devices
        .get(&device)
        .context("Invalid device handle")?;
    let bindless_descriptor_set = logical.bindless_descriptor_set;

    let mut vk_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::VERTEX_BUFFER
        | vk::BufferUsageFlags::INDEX_BUFFER;
    if size >= 12 {
        vk_usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }

    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { logical.device.create_buffer(&buffer_info, None) }
        .context("transient placed buffer")?;
    let mem_reqs = unsafe { logical.device.get_buffer_memory_requirements(buffer) };
    anyhow::ensure!(
        offset.is_multiple_of(mem_reqs.alignment),
        "transient buffer offset {} not aligned to {}",
        offset,
        mem_reqs.alignment
    );
    unsafe {
        logical
            .device
            .bind_buffer_memory(buffer, shared_mem, offset)
    }
    .context("vkBindBufferMemory transient")?;

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    let bindless_index = {
        let logical_device = state.devices.get_mut(&device).unwrap();
        let index = logical_device
            .resource_registry
            .register_buffer(handle, true);
        if let Some(descriptor_set) = bindless_descriptor_set {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(0)
                .range(size);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(types::bindless_bindings::STORAGE_BUFFERS)
                .dst_array_element(index)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));
            unsafe {
                logical_device
                    .device
                    .update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }
        }
        Some(index)
    };

    state
        .transient_heaps
        .get_mut(&heap)
        .expect("heap must exist")
        .buffers
        .push(handle);
    state.buffers.insert(
        handle,
        types::BufferState {
            device_handle: device,
            buffer,
            memory: shared_mem,
            size,
            bindless_index,
            is_storage: true,
            element_stride: None,
            staging_buffer: None,
            staging_memory: None,
            is_view: false,
            host_mapped: None,
            flags: crate::types::BufferFlags::empty(),
            transient_heap_suballoc: true,
            view_byte_offset: None,
        },
    );
    Ok(handle)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn place_texture_in_transient_heap(
    state: &mut VulkanState,
    device: DeviceHandle,
    heap: TransientHeapHandle,
    offset: u64,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: SpatialAccess,
    flags: TextureFlags,
) -> Result<TextureHandle> {
    let shared_mem = {
        let e = state
            .transient_heaps
            .get(&heap)
            .with_context(|| format!("invalid transient heap {heap}"))?;
        anyhow::ensure!(e.device_handle == device);
        e.memory
    };

    let logical = state
        .devices
        .get(&device)
        .context("Invalid device handle")?;
    let bindless_descriptor_set = logical.bindless_descriptor_set;

    let mut vk_usage = vk::ImageUsageFlags::TRANSFER_DST;
    match access {
        SpatialAccess::Interpolated => {
            vk_usage |= vk::ImageUsageFlags::SAMPLED;
        }
        SpatialAccess::Direct => {
            vk_usage |= vk::ImageUsageFlags::STORAGE;
        }
    }
    if flags.contains(TextureFlags::RENDER_TARGET) {
        vk_usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if flags.contains(TextureFlags::COPY_SRC) {
        vk_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
    }

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format_to_vk(format))
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { logical.device.create_image(&image_info, None) }
        .context("transient placed image")?;
    let mem_reqs = unsafe { logical.device.get_image_memory_requirements(image) };
    anyhow::ensure!(
        offset.is_multiple_of(mem_reqs.alignment),
        "transient texture offset {} not aligned to {}",
        offset,
        mem_reqs.alignment
    );
    unsafe { logical.device.bind_image_memory(image, shared_mem, offset) }
        .context("vkBindImageMemory transient")?;

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format_to_vk(format))
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { logical.device.create_image_view(&view_info, None) }
        .context("transient image view")?;

    let handle = state.next_texture_handle;
    state.next_texture_handle += 1;
    let is_storage_image = matches!(access, SpatialAccess::Direct);

    let bindless_index = {
        let logical_device = state.devices.get_mut(&device).unwrap();
        let index = logical_device
            .resource_registry
            .register_texture(handle, is_storage_image);
        if let Some(descriptor_set) = bindless_descriptor_set {
            let (binding, descriptor_type, image_layout) = if is_storage_image {
                (
                    types::bindless_bindings::STORAGE_IMAGES,
                    vk::DescriptorType::STORAGE_IMAGE,
                    vk::ImageLayout::GENERAL,
                )
            } else {
                (
                    types::bindless_bindings::SAMPLED_IMAGES,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
            };
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(image_layout);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding)
                .dst_array_element(index)
                .descriptor_type(descriptor_type)
                .image_info(std::slice::from_ref(&image_info));
            unsafe {
                logical_device
                    .device
                    .update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }
        }
        Some(index)
    };

    let initial_layout = if is_storage_image {
        let ld_ref = state.devices.get(&device).unwrap();
        texture::transition_image_layout(
            ld_ref,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        )?;
        vk::ImageLayout::GENERAL
    } else {
        vk::ImageLayout::UNDEFINED
    };

    state
        .transient_heaps
        .get_mut(&heap)
        .expect("heap must exist")
        .textures
        .push(handle);
    state.textures.insert(
        handle,
        types::TextureState {
            device_handle: device,
            width,
            height,
            format,
            image,
            memory: shared_mem,
            view,
            staging_buffer: None,
            staging_memory: None,
            bindless_index,
            current_layout: initial_layout,
            transient_heap_suballoc: true,
        },
    );
    Ok(handle)
}

pub(super) fn destroy_transient_heap(
    state: &mut VulkanState,
    device: DeviceHandle,
    heap: TransientHeapHandle,
) -> Result<()> {
    let mut entry = state
        .transient_heaps
        .remove(&heap)
        .with_context(|| format!("invalid transient heap {heap}"))?;
    anyhow::ensure!(entry.device_handle == device);
    for b in entry.buffers.drain(..) {
        super::buffer::destroy(&mut state.devices, &mut state.buffers, b);
    }
    for t in entry.textures.drain(..) {
        super::texture::destroy(&mut state.devices, &mut state.textures, t);
    }
    let vk_dev = &state
        .devices
        .get(&device)
        .context("Invalid device handle")?
        .device;
    unsafe {
        vk_dev.free_memory(entry.memory, None);
    }
    Ok(())
}

pub(super) fn destroy_all_for_device(state: &mut VulkanState, device: DeviceHandle) {
    let ids: Vec<_> = state
        .transient_heaps
        .iter()
        .filter(|(_, e)| e.device_handle == device)
        .map(|(&k, _)| k)
        .collect();
    for h in ids {
        let _ = destroy_transient_heap(state, device, h);
    }
}
