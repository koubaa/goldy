//! Texture management operations.

use super::types::{Dx12State, TextureState};
use super::utils::{format_to_dxgi, wait_for_fence};
use super::{DeviceHandle, TextureHandle};
use crate::types::{SpatialAccess, TextureFlags, TextureFormat};
use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

/// Create a texture.
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    _access: SpatialAccess,
    _flags: TextureFlags,
) -> Result<TextureHandle> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let heap_properties = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
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
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };

    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        logical_device.device.CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &resource_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut resource,
        )
    }
    .context("Failed to create texture")?;
    let resource = resource.context("CreateCommittedResource returned null")?;

    // Get texture handle first (needed for registry)
    let handle = state.next_texture_handle;
    state.next_texture_handle += 1;

    // Create SRV - use unified resource registry to avoid descriptor heap collisions
    // (textures and buffers share the same CBV/SRV/UAV heap)
    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;
    let srv_offset = logical_device.resource_registry.register_texture(handle);

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
        let mut cpu_handle = logical_device
            .cbv_srv_uav_heap
            .GetCPUDescriptorHandleForHeapStart();
        cpu_handle.ptr += (srv_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
        cpu_handle
    };
    unsafe {
        logical_device
            .device
            .CreateShaderResourceView(&resource, Some(&srv_desc), srv_cpu_handle);
    }

    state.textures.insert(
        handle,
        TextureState {
            device_handle,
            width,
            height,
            format,
            resource,
            srv_offset,
            bindless_offset: Some(srv_offset), // SRV offset is the bindless offset
        },
    );

    tracing::debug!("Created texture {}x{} (handle={})", width, height, handle);
    Ok(handle)
}

/// Write data to a texture.
pub(super) fn write(
    state: &mut Dx12State,
    texture_handle: TextureHandle,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<()> {
    let texture = state
        .textures
        .get(&texture_handle)
        .context("Invalid texture handle")?;

    if texture.width != width || texture.height != height {
        anyhow::bail!("Texture dimensions mismatch");
    }

    let device_handle = texture.device_handle;
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Calculate row pitch (must be 256-byte aligned for D3D12)
    let row_pitch = ((width * texture.format.bytes_per_pixel() + 255) & !255) as u64;
    let staging_size = row_pitch * height as u64;

    // Create staging buffer
    let upload_heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };

    let buffer_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: staging_size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };

    let mut staging: Option<ID3D12Resource> = None;
    unsafe {
        logical_device.device.CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut staging,
        )
    }
    .context("Failed to create staging buffer")?;
    let staging = staging.context("CreateCommittedResource returned null")?;

    // Map and copy data
    let mut mapped_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let read_range = D3D12_RANGE { Begin: 0, End: 0 };
    unsafe { staging.Map(0, Some(&read_range), Some(&mut mapped_ptr)) }
        .context("Failed to map staging buffer")?;

    let texture_format = texture.format;
    let bytes_per_row = (width * texture_format.bytes_per_pixel()) as usize;
    for row in 0..height {
        let src_offset = (row as usize) * bytes_per_row;
        let dst_offset = (row as u64 * row_pitch) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().add(src_offset),
                (mapped_ptr as *mut u8).add(dst_offset),
                bytes_per_row,
            );
        }
    }

    let written_range = D3D12_RANGE {
        Begin: 0,
        End: staging_size as usize,
    };
    unsafe { staging.Unmap(0, Some(&written_range)) };

    // Execute copy command
    unsafe { logical_device.command_allocator.Reset() }
        .context("Failed to reset command allocator")?;

    let command_list: ID3D12GraphicsCommandList = unsafe {
        logical_device.device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &logical_device.command_allocator,
            None,
        )
    }
    .context("Failed to create command list")?;

    let texture = state.textures.get(&texture_handle).unwrap();

    let src_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&staging) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: format_to_dxgi(texture.format),
                    Width: width,
                    Height: height,
                    Depth: 1,
                    RowPitch: row_pitch as u32,
                },
            },
        },
    };

    let dst_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&texture.resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };

    unsafe {
        command_list.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None);

        // Transition to shader resource
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(&texture.resource),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                    StateAfter: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                }),
            },
        };
        command_list.ResourceBarrier(&[barrier]);
        command_list.Close()
    }
    .context("Failed to close command list")?;

    let cmd_list: ID3D12CommandList = command_list.cast().context("Failed to cast command list")?;

    let logical_device = state.devices.get(&device_handle).unwrap();
    unsafe {
        logical_device
            .command_queue
            .ExecuteCommandLists(&[Some(cmd_list)]);
    }

    // Wait for completion
    let fence_value = logical_device.fence_value;
    unsafe {
        logical_device
            .command_queue
            .Signal(&logical_device.fence, fence_value)
    }
    .context("Failed to signal fence")?;
    wait_for_fence(&logical_device.fence, fence_value)?;

    tracing::debug!("Wrote {}x{} texture data", width, height);
    Ok(())
}

/// Destroy a texture.
pub(super) fn destroy(state: &mut Dx12State, texture_handle: TextureHandle) {
    state.textures.remove(&texture_handle);
}

/// Get the bindless index for a texture.
pub(super) fn bindless_index(state: &Dx12State, texture_handle: TextureHandle) -> Option<u32> {
    state
        .textures
        .get(&texture_handle)
        .and_then(|t| t.bindless_offset)
}
