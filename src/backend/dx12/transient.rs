//! Native transient heaps: [`ID3D12Heap`] + [`ID3D12Device::CreatePlacedResource`].

use super::texture;
use super::types::{BufferState, Dx12State, TextureState, TransientHeapEntry};
use super::utils::format_to_dxgi;
use super::{BufferHandle, DeviceHandle, TextureHandle, TransientHeapHandle};
use crate::types::{SpatialAccess, TextureFlags, TextureFormat};
use anyhow::{Context, Result};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

pub(super) fn transient_heap_alignment_hints(
    _state: &Dx12State,
    _device: DeviceHandle,
) -> crate::backend::TransientHeapAlignments {
    crate::backend::TransientHeapAlignments {
        buffer_base_align: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
        texture_base_align: 65536,
        buffer_image_granularity: 512,
    }
}

pub(super) fn transient_texture_heap_footprint(
    state: &Dx12State,
    device: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: SpatialAccess,
    _flags: TextureFlags,
) -> Result<(u64, u64)> {
    let ld = state
        .devices
        .get(&device)
        .context("Invalid device handle")?;
    let is_storage = matches!(access, SpatialAccess::Direct);
    let resource_flags = if is_storage {
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_FLAG_NONE
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format_to_dxgi(format),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: resource_flags,
    };
    let info = unsafe { ld.device.GetResourceAllocationInfo(0, &[desc]) };
    Ok((info.Alignment, info.SizeInBytes))
}

pub(super) fn create_transient_heap(
    state: &mut Dx12State,
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
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size,
        Properties: heap_props,
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
    };
    let mut heap_opt: Option<ID3D12Heap> = None;
    unsafe { ld.device.CreateHeap(&desc, &mut heap_opt) }.context("CreateHeap transient")?;
    let heap = heap_opt.context("CreateHeap returned null")?;

    let h = state.next_transient_heap_handle;
    state.next_transient_heap_handle += 1;
    state.transient_heaps.insert(
        h,
        TransientHeapEntry {
            device_handle: device,
            heap,
            buffers: Vec::new(),
            textures: Vec::new(),
        },
    );
    Ok(Some(h))
}

pub(super) fn place_buffer_in_transient_heap(
    state: &mut Dx12State,
    device: DeviceHandle,
    heap_h: TransientHeapHandle,
    offset: u64,
    size: u64,
) -> Result<BufferHandle> {
    let heap = {
        let e = state
            .transient_heaps
            .get(&heap_h)
            .with_context(|| format!("invalid transient heap {heap_h}"))?;
        anyhow::ensure!(e.device_handle == device);
        e.heap.clone()
    };

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    let resource_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };

    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        state
            .devices
            .get(&device)
            .context("Invalid device handle")?
            .device
            .CreatePlacedResource(
                &heap,
                offset,
                &resource_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
    }
    .context("CreatePlacedResource buffer")?;
    let resource = resource.context("CreatePlacedResource buffer null")?;

    let ld = state
        .devices
        .get_mut(&device)
        .context("Invalid device handle")?;
    let stride = 4u32;
    let num_elements = (size as u32) / stride;
    let uav_offset = ld.resource_registry.register_buffer_uav(handle);
    let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: 0,
                NumElements: num_elements,
                StructureByteStride: stride,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_NONE,
            },
        },
    };
    let uav_cpu_handle = unsafe {
        let mut cpu_handle = ld.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
        cpu_handle.ptr += (uav_offset * ld.cbv_srv_uav_descriptor_size) as usize;
        cpu_handle
    };
    unsafe {
        ld.device
            .CreateUnorderedAccessView(&resource, None, Some(&uav_desc), uav_cpu_handle);
    }
    let srv_offset = ld.resource_registry.register_buffer_srv(handle);
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: 0,
                NumElements: num_elements,
                StructureByteStride: stride,
                Flags: D3D12_BUFFER_SRV_FLAG_NONE,
            },
        },
    };
    let srv_cpu_handle = unsafe {
        let mut cpu_handle = ld.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
        cpu_handle.ptr += (srv_offset * ld.cbv_srv_uav_descriptor_size) as usize;
        cpu_handle
    };
    unsafe {
        ld.device
            .CreateShaderResourceView(&resource, Some(&srv_desc), srv_cpu_handle);
    }

    state
        .transient_heaps
        .get_mut(&heap_h)
        .unwrap()
        .buffers
        .push(handle);
    state.buffers.insert(
        handle,
        BufferState {
            device_handle: device,
            resource,
            size,
            allocation_size: size,
            bindless_offset: Some(uav_offset),
            bindless_srv_offset: Some(srv_offset),
            is_storage: true,
            upload_buffer: None,
            element_stride: Some(stride),
            is_view: false,
            coherent_readback: None,
            coherent_readback_mapped: None,
            flags: crate::types::BufferFlags::empty(),
            transient_placed: true,
            parent_for_view: None,
            view_byte_offset: None,
            is_reserved: false,
            tile_byte_size: 0,
            reserved_tiles: Vec::new(),
        },
    );
    Ok(handle)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn place_texture_in_transient_heap(
    state: &mut Dx12State,
    device: DeviceHandle,
    heap_h: TransientHeapHandle,
    offset: u64,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: SpatialAccess,
    flags: TextureFlags,
) -> Result<TextureHandle> {
    let heap = {
        let e = state
            .transient_heaps
            .get(&heap_h)
            .with_context(|| format!("invalid transient heap {heap_h}"))?;
        anyhow::ensure!(e.device_handle == device);
        e.heap.clone()
    };

    let is_storage = matches!(access, SpatialAccess::Direct);
    let resource_flags = if is_storage {
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_FLAG_NONE
    };
    let resource_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format_to_dxgi(format),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: resource_flags,
    };

    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        state
            .devices
            .get(&device)
            .context("Invalid device handle")?
            .device
            .CreatePlacedResource(
                &heap,
                offset,
                &resource_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
    }
    .context("CreatePlacedResource texture")?;
    let resource = resource.context("CreatePlacedResource texture null")?;

    let handle = state.next_texture_handle;
    state.next_texture_handle += 1;

    let ld = state
        .devices
        .get_mut(&device)
        .context("Invalid device handle")?;
    let srv_offset = ld.resource_registry.register_texture(handle);
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format_to_dxgi(format),
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    let srv_cpu_handle = unsafe {
        let mut cpu_handle = ld.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
        cpu_handle.ptr += (srv_offset * ld.cbv_srv_uav_descriptor_size) as usize;
        cpu_handle
    };
    unsafe {
        ld.device
            .CreateShaderResourceView(&resource, Some(&srv_desc), srv_cpu_handle);
    }

    let bindless_offset = if is_storage {
        let uav_offset = ld.resource_registry.register_texture_uav(handle);
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format_to_dxgi(format),
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        let uav_cpu_handle = unsafe {
            let mut cpu_handle = ld.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
            cpu_handle.ptr += (uav_offset * ld.cbv_srv_uav_descriptor_size) as usize;
            cpu_handle
        };
        unsafe {
            ld.device.CreateUnorderedAccessView(
                Some(&resource),
                None,
                Some(&uav_desc),
                uav_cpu_handle,
            );
        }
        Some(uav_offset)
    } else {
        Some(srv_offset)
    };

    let _ = flags;
    let last_layout = if is_storage {
        texture::init_storage_texture_uav_layout(state, device, &resource)?
    } else {
        D3D12_BARRIER_LAYOUT_COMMON
    };

    state
        .transient_heaps
        .get_mut(&heap_h)
        .unwrap()
        .textures
        .push(handle);
    state.textures.insert(
        handle,
        TextureState {
            device_handle: device,
            width,
            height,
            format,
            resource,
            srv_offset,
            bindless_offset,
            last_layout,
            is_storage,
            transient_placed: true,
        },
    );
    Ok(handle)
}

pub(super) fn destroy_transient_heap(
    state: &mut Dx12State,
    device: DeviceHandle,
    heap_h: TransientHeapHandle,
) -> Result<()> {
    let mut entry = state
        .transient_heaps
        .remove(&heap_h)
        .with_context(|| format!("invalid transient heap {heap_h}"))?;
    anyhow::ensure!(entry.device_handle == device);
    for b in entry.buffers.drain(..) {
        super::buffer::destroy(state, b);
    }
    for t in entry.textures.drain(..) {
        super::texture::destroy(state, t);
    }
    drop(entry.heap);
    Ok(())
}

pub(super) fn destroy_all_for_device(state: &mut Dx12State, device: DeviceHandle) {
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
