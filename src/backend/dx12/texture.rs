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
    access: SpatialAccess,
    flags: TextureFlags,
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

    let initial_state = if is_storage {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_STATE_COPY_DEST
    };

    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        logical_device.device.CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &resource_desc,
            initial_state,
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

    // For storage images (Direct access): create UAV so compute shaders can write via RWTexture2D.
    // bindless_offset must point to UAV for goldy_dyn_direct_spatial.
    let bindless_offset = if is_storage {
        let uav_offset = logical_device
            .resource_registry
            .register_texture_uav(handle);
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
            let mut cpu_handle = logical_device
                .cbv_srv_uav_heap
                .GetCPUDescriptorHandleForHeapStart();
            cpu_handle.ptr += (uav_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
            cpu_handle
        };
        unsafe {
            logical_device.device.CreateUnorderedAccessView(
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

    state.textures.insert(
        handle,
        TextureState {
            device_handle,
            width,
            height,
            format,
            resource,
            srv_offset,
            bindless_offset,
            current_state: initial_state,
        },
    );

    let _ = flags; // reserved for future use
    tracing::debug!(
        "Created texture {}x{} (handle={}, storage={})",
        width,
        height,
        handle,
        is_storage
    );
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

    // Use GetCopyableFootprints with the actual texture's resource desc
    let resource_desc = unsafe { texture.resource.GetDesc() };
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut num_rows: u32 = 0;
    let mut row_size: u64 = 0;
    let mut total_bytes: u64 = 0;
    unsafe {
        logical_device.device.GetCopyableFootprints(
            &resource_desc,
            0,
            1,
            0,
            Some(&mut footprint),
            Some(&mut num_rows),
            Some(&mut row_size),
            Some(&mut total_bytes),
        );
    }
    let staging_size = total_bytes;

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
        .context("Failed to map staging buffer (texture with_data)")?;

    let texture_format = texture.format;
    let bytes_per_row = (width * texture_format.bytes_per_pixel()) as usize;
    let row_pitch_bytes = footprint.Footprint.RowPitch as usize;
    for row in 0..height {
        let src_offset = (row as usize) * bytes_per_row;
        let dst_offset = (footprint.Offset + row as u64 * row_pitch_bytes as u64) as usize;
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
    let state_before = texture.current_state;

    // Transition to COPY_DEST only if not already in that state
    // (non-storage textures are created in COPY_DEST; redundant transitions are invalid)
    if state_before != D3D12_RESOURCE_STATE_COPY_DEST {
        let barrier_to_copy = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(&texture.resource) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: state_before,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
                }),
            },
        };
        unsafe { command_list.ResourceBarrier(&[barrier_to_copy]) };
    }

    let src_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&staging) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: footprint.Offset,
                Footprint: footprint.Footprint,
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

        // Transition to shader resource state for sampling
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(&texture.resource),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                    StateAfter: D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                }),
            },
        };
        command_list.ResourceBarrier(&[barrier]);
        command_list.Close()
    }
    .context("Failed to close command list")?;

    if let Some(tex) = state.textures.get_mut(&texture_handle) {
        tex.current_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }

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

    if let Some(dev) = state.devices.get_mut(&device_handle) {
        dev.fence_value += 1;
    }

    tracing::debug!("Wrote {}x{} texture data", width, height);
    Ok(())
}

/// Write data to a subregion of a texture.
pub(super) fn write_region(
    state: &mut Dx12State,
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

    if x + width > texture.width || y + height > texture.height {
        anyhow::bail!(
            "Region out of bounds: {}x{} at ({},{}) exceeds {}x{} texture",
            width,
            height,
            x,
            y,
            texture.width,
            texture.height
        );
    }

    let expected_size = (width * height * texture.format.bytes_per_pixel()) as usize;
    if data.len() != expected_size {
        anyhow::bail!(
            "Data size mismatch: expected {}, got {}",
            expected_size,
            data.len()
        );
    }

    let device_handle = texture.device_handle;
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Use footprint for the UPLOAD REGION (width x height), not the full texture.
    // The buffer layout must match GetCopyableFootprints for the region we're uploading.
    let region_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format_to_dxgi(texture.format),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut _num_rows: u32 = 0;
    let mut _row_size: u64 = 0;
    let mut total_bytes: u64 = 0;
    unsafe {
        logical_device.device.GetCopyableFootprints(
            &region_desc,
            0,
            1,
            0,
            Some(&mut footprint),
            Some(&mut _num_rows),
            Some(&mut _row_size),
            Some(&mut total_bytes),
        );
    }
    let staging_size = total_bytes;

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

    let texture_format = texture.format;
    let bytes_per_row = (width * texture_format.bytes_per_pixel()) as usize;
    let mut mapped_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let read_range = D3D12_RANGE { Begin: 0, End: 0 };
    unsafe { staging.Map(0, Some(&read_range), Some(&mut mapped_ptr)) }
        .context("Failed to map staging buffer (texture write_region)")?;

    let row_pitch_bytes = footprint.Footprint.RowPitch as usize;
    for row in 0..height {
        let src_offset = (row as usize) * bytes_per_row;
        let dst_offset = (footprint.Offset + row as u64 * row_pitch_bytes as u64) as usize;
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
    let state_before = texture.current_state;

    if state_before != D3D12_RESOURCE_STATE_COPY_DEST {
        let barrier_to_copy = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(&texture.resource) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: state_before,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
                }),
            },
        };
        unsafe { command_list.ResourceBarrier(&[barrier_to_copy]) };
    }

    let src_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&staging) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: footprint.Offset,
                Footprint: footprint.Footprint,
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
        command_list.CopyTextureRegion(&dst_location, x, y, 0, &src_location, None);
    }

    let barrier_to_shader = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(&texture.resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                StateAfter: D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            }),
        },
    };
    unsafe {
        command_list.ResourceBarrier(&[barrier_to_shader]);
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

    let fence_value = logical_device.fence_value;
    unsafe {
        logical_device
            .command_queue
            .Signal(&logical_device.fence, fence_value)
    }
    .context("Failed to signal fence")?;
    wait_for_fence(&logical_device.fence, fence_value)?;

    if let Some(dev) = state.devices.get_mut(&device_handle) {
        dev.fence_value += 1;
    }

    if let Some(tex) = state.textures.get_mut(&texture_handle) {
        tex.current_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }

    tracing::debug!("Wrote {}x{} region at ({},{})", width, height, x, y);
    Ok(())
}

/// Read texture contents to CPU memory.
/// The texture must have been created with TextureFlags::COPY_SRC.
pub(super) fn read_to_cpu(
    state: &mut Dx12State,
    texture_handle: TextureHandle,
    output: &mut [u8],
) -> Result<()> {
    use windows::Win32::Graphics::Direct3D12::*;

    let texture = state
        .textures
        .get(&texture_handle)
        .context("Invalid texture handle")?;

    let device_handle = texture.device_handle;
    let width = texture.width;
    let height = texture.height;
    let format = texture.format;
    let expected_size = (width * height * format.bytes_per_pixel()) as usize;

    if output.len() < expected_size {
        anyhow::bail!(
            "Output buffer too small: {} < {}",
            output.len(),
            expected_size
        );
    }

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Use the texture's actual resource desc for correct layout
    let res_desc = unsafe { texture.resource.GetDesc() };
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut _num_rows: u32 = 0;
    let mut _row_size: u64 = 0;
    let mut total_bytes: u64 = 0;
    unsafe {
        logical_device.device.GetCopyableFootprints(
            &res_desc,
            0,
            1,
            0,
            Some(&mut footprint),
            Some(&mut _num_rows),
            Some(&mut _row_size),
            Some(&mut total_bytes),
        );
    }
    let staging_size = total_bytes;

    let heap_properties = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };

    let resource_desc = D3D12_RESOURCE_DESC {
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

    // Check for device removal before allocating (TDR during prior compute work)
    let removed_reason = unsafe { logical_device.device.GetDeviceRemovedReason() };
    if removed_reason.is_err() {
        anyhow::bail!(
            "Device removed before texture readback: {:?}",
            removed_reason
        );
    }

    let mut staging_buffer: Option<ID3D12Resource> = None;
    unsafe {
        logical_device.device.CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &resource_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut staging_buffer,
        )
    }
    .context("Failed to create staging buffer (texture read_to_cpu)")?;
    let staging_buffer = staging_buffer.context("CreateCommittedResource returned null")?;

    // Reset allocator before reuse (required after prior compute/render work that used it)
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

    // Newly created list is already in recording state; Reset is for reusing a closed list

    let state_before = texture.current_state;

    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(&texture.resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: state_before,
                StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
            }),
        },
    };
    unsafe { command_list.ResourceBarrier(&[barrier]) };

    let src_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&texture.resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };

    let dst_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&staging_buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: footprint.Offset,
                Footprint: footprint.Footprint,
            },
        },
    };

    unsafe { command_list.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None) };

    let barrier_back = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(&texture.resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_COPY_SOURCE,
                StateAfter: state_before,
            }),
        },
    };
    unsafe { command_list.ResourceBarrier(&[barrier_back]) };

    unsafe { command_list.Close() }.context("Failed to close command list")?;

    let cmd_list: ID3D12CommandList = command_list.cast().context("Failed to cast command list")?;
    unsafe {
        logical_device
            .command_queue
            .ExecuteCommandLists(&[Some(cmd_list)]);
    }

    let fence_value = logical_device.fence_value;
    unsafe {
        logical_device
            .command_queue
            .Signal(&logical_device.fence, fence_value)
    }
    .context("Failed to signal fence")?;
    wait_for_fence(&logical_device.fence, fence_value)?;

    if let Some(dev) = state.devices.get_mut(&device_handle) {
        dev.fence_value += 1;
        // Check for device removal (compute passes may have caused TDR)
        let removed = unsafe { dev.device.GetDeviceRemovedReason() };
        if removed.is_err() {
            anyhow::bail!("Device removed before texture readback map: {:?}", removed);
        }
    }

    let mut mapped_data: *mut u8 = std::ptr::null_mut();
    unsafe {
        staging_buffer.Map(
            0,
            None,
            Some(&mut mapped_data as *mut *mut u8 as *mut *mut _),
        )
    }
    .context("Failed to map staging buffer (texture read_to_cpu)")?;

    let bytes_per_row = (width * format.bytes_per_pixel()) as usize;
    let row_pitch_bytes = footprint.Footprint.RowPitch as usize;
    for row in 0..height as usize {
        let src_offset = (footprint.Offset + row as u64 * row_pitch_bytes as u64) as usize;
        let dst_offset = row * bytes_per_row;
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped_data.add(src_offset),
                output.as_mut_ptr().add(dst_offset),
                bytes_per_row,
            );
        }
    }

    unsafe { staging_buffer.Unmap(0, None) };

    Ok(())
}

/// Destroy a texture.
pub(super) fn destroy(state: &mut Dx12State, texture_handle: TextureHandle) {
    if let Some(tex) = state.textures.get(&texture_handle) {
        let device_handle = tex.device_handle;
        if let Some(dev) = state.devices.get_mut(&device_handle) {
            dev.resource_registry.unregister_texture(texture_handle);
        }
    }
    state.textures.remove(&texture_handle);
}

/// Get the bindless index for a texture.
pub(super) fn bindless_index(state: &Dx12State, texture_handle: TextureHandle) -> Option<u32> {
    state
        .textures
        .get(&texture_handle)
        .and_then(|t| t.bindless_offset)
}
