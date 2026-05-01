//! Texture management operations.

use super::barriers;
use super::types::{Dx12State, TextureState};
use super::utils::{format_to_dxgi, wait_for_fence};
use super::{DeviceHandle, TextureHandle};
use crate::types::{SpatialAccess, TextureFlags, TextureFormat};
use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

/// A staged texture copy awaiting batch submission.
pub(super) struct PendingTextureCopy {
    pub staging_resource: ID3D12Resource,
    pub footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    pub texture_handle: TextureHandle,
    pub dst_x: u32,
    pub dst_y: u32,
    /// Texture layout at the time this copy was enqueued.
    pub layout_before: D3D12_BARRIER_LAYOUT,
}

/// Key for the texture resource cache: (width, height, format, is_storage).
pub(super) type TextureCacheKey = (u32, u32, TextureFormat, bool);

const MAX_TEXTURE_CACHE_PER_KEY: usize = 8;

/// A cached `ID3D12Resource` ready for reuse, avoiding `CreateCommittedResource`.
pub(super) struct CachedTextureResource {
    pub resource: ID3D12Resource,
    pub last_layout: D3D12_BARRIER_LAYOUT,
}

/// [`D3D12_BARRIER_ACCESS`] that matches `layout` for texture barriers.
fn access_for_layout(layout: D3D12_BARRIER_LAYOUT) -> D3D12_BARRIER_ACCESS {
    if layout == D3D12_BARRIER_LAYOUT_COMMON {
        D3D12_BARRIER_ACCESS_COMMON
    } else if layout == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE
        || layout == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE
    {
        D3D12_BARRIER_ACCESS_SHADER_RESOURCE
    } else if layout == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS
        || layout == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS
    {
        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS
    } else if layout == D3D12_BARRIER_LAYOUT_COPY_DEST {
        D3D12_BARRIER_ACCESS_COPY_DEST
    } else if layout == D3D12_BARRIER_LAYOUT_COPY_SOURCE {
        D3D12_BARRIER_ACCESS_COPY_SOURCE
    } else {
        D3D12_BARRIER_ACCESS_COMMON
    }
}

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
    let is_storage = matches!(access, SpatialAccess::Direct);

    // Try to reuse a cached resource to avoid expensive CreateCommittedResource
    let cache_key: TextureCacheKey = (width, height, format, is_storage);
    let cached = state
        .texture_cache
        .get_mut(&cache_key)
        .and_then(|v| v.pop());

    let (resource, last_layout) = if let Some(c) = cached {
        (c.resource, c.last_layout)
    } else {
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

        let initial_state = D3D12_RESOURCE_STATE_COMMON;

        let mut resource: Option<ID3D12Resource> = None;
        let hr = unsafe {
            logical_device.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                initial_state,
                None,
                &mut resource,
            )
        };
        hr.context("Failed to create texture")?;
        let resource = resource.context("CreateCommittedResource returned null")?;

        (resource, D3D12_BARRIER_LAYOUT_COMMON)
    };

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

    let last_layout = if is_storage && last_layout != D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS {
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle for storage texture initial barrier")?;

        unsafe { logical_device.command_allocator.Reset() }
            .context("Failed to reset command allocator for texture init barrier")?;
        let init_cmd: ID3D12GraphicsCommandList = unsafe {
            logical_device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &logical_device.command_allocator,
                None,
            )
        }
        .context("Failed to create init barrier command list")?;
        let init_cmd7: ID3D12GraphicsCommandList7 =
            init_cmd.cast().context("ID3D12GraphicsCommandList7")?;

        let b = barriers::texture_barrier_full(
            &resource,
            D3D12_BARRIER_SYNC_NONE,
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_NO_ACCESS,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            last_layout,
            D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS,
        );
        unsafe {
            barriers::barrier_textures(&init_cmd7, &[b]);
            init_cmd.Close()
        }
        .context("Failed to close init barrier command list")?;

        let cmd_list: ID3D12CommandList = init_cmd
            .cast()
            .context("Failed to cast init command list")?;
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
        .context("Failed to signal fence for init barrier")?;
        super::utils::wait_for_fence(&logical_device.fence, fence_value)?;

        if let Some(dev) = state.devices.get_mut(&device_handle) {
            dev.fence_value += 1;
        }

        D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS
    } else if is_storage {
        last_layout
    } else {
        last_layout
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
            last_layout,
            is_storage,
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
///
/// Stages the upload into [`PendingTextureCopy`]; actual GPU work is deferred
/// until [`flush_pending_copies`] (called automatically before compute submit
/// and texture readback).
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

    let staging = create_upload_staging(&logical_device.device, staging_size)?;

    let mut mapped_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let read_range = D3D12_RANGE { Begin: 0, End: 0 };
    unsafe { staging.Map(0, Some(&read_range), Some(&mut mapped_ptr)) }
        .context("Failed to map staging buffer (texture write)")?;

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

    let layout_before = texture.last_layout;

    state.pending_texture_copies.push(PendingTextureCopy {
        staging_resource: staging,
        footprint,
        texture_handle,
        dst_x: 0,
        dst_y: 0,
        layout_before,
    });

    tracing::debug!("Staged {}x{} texture write (pending flush)", width, height);
    Ok(())
}

/// Write data to a subregion of a texture.
///
/// Stages the upload into [`PendingTextureCopy`]; actual GPU work is deferred
/// until [`flush_pending_copies`].
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

    let staging = create_upload_staging(&logical_device.device, staging_size)?;

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

    let layout_before = texture.last_layout;

    state.pending_texture_copies.push(PendingTextureCopy {
        staging_resource: staging,
        footprint,
        texture_handle,
        dst_x: x,
        dst_y: y,
        layout_before,
    });

    tracing::debug!(
        "Staged {}x{} texture region at ({},{}) (pending flush)",
        width,
        height,
        x,
        y,
    );
    Ok(())
}

/// Allocate a UPLOAD-heap staging buffer of `size` bytes.
fn create_upload_staging(device: &ID3D12Device10, size: u64) -> Result<ID3D12Resource> {
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
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut resource,
        )
    }
    .context("Failed to create upload staging buffer")?;
    resource.context("CreateCommittedResource returned null")
}

/// Flush all pending texture copies in a single command list submission.
///
/// Groups copies by texture so each texture gets one barrier pair
/// (old layout → COPY_DEST, then COPY_DEST → SHADER_RESOURCE) around
/// all of its pending copies.
pub(super) fn flush_pending_copies(state: &mut Dx12State) -> Result<()> {
    if state.pending_texture_copies.is_empty() {
        return Ok(());
    }

    let copies = std::mem::take(&mut state.pending_texture_copies);
    let count = copies.len();

    let first_tex = state
        .textures
        .get(&copies[0].texture_handle)
        .context("flush_pending_copies: invalid texture handle")?;
    let device_handle = first_tex.device_handle;

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("flush_pending_copies: invalid device handle")?;

    unsafe { logical_device.command_allocator.Reset() }
        .context("flush_pending_copies: failed to reset command allocator")?;

    let command_list: ID3D12GraphicsCommandList = unsafe {
        logical_device.device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &logical_device.command_allocator,
            None,
        )
    }
    .context("flush_pending_copies: failed to create command list")?;
    let command_list7: ID3D12GraphicsCommandList7 =
        command_list.cast().context("ID3D12GraphicsCommandList7")?;

    // Group copies by texture handle (preserving order within each texture).
    let mut texture_order: Vec<TextureHandle> = Vec::new();
    let mut groups: std::collections::HashMap<TextureHandle, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, copy) in copies.iter().enumerate() {
        groups
            .entry(copy.texture_handle)
            .or_insert_with(|| {
                texture_order.push(copy.texture_handle);
                Vec::new()
            })
            .push(i);
    }

    let after_layout = D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE;

    for tex_handle in &texture_order {
        let indices = &groups[tex_handle];
        let texture = state
            .textures
            .get(tex_handle)
            .context("flush_pending_copies: texture disappeared")?;

        // Use the layout the texture was in when the FIRST copy was enqueued.
        let layout_before = copies[indices[0]].layout_before;

        let mut b_to_copy = [barriers::texture_barrier_full(
            &texture.resource,
            D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_SYNC_COPY,
            access_for_layout(layout_before),
            D3D12_BARRIER_ACCESS_COPY_DEST,
            layout_before,
            D3D12_BARRIER_LAYOUT_COPY_DEST,
        )];
        unsafe { barriers::barrier_textures(&command_list7, &b_to_copy) };
        unsafe { barriers::drop_texture_barriers(&mut b_to_copy) };

        for &idx in indices {
            let copy = &copies[idx];

            let src_location = D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { std::mem::transmute_copy(&copy.staging_resource) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: copy.footprint.Offset,
                        Footprint: copy.footprint.Footprint,
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
                command_list.CopyTextureRegion(
                    &dst_location,
                    copy.dst_x,
                    copy.dst_y,
                    0,
                    &src_location,
                    None,
                );
            }
        }

        let mut b_to_shader = [barriers::texture_barrier_full(
            &texture.resource,
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_ACCESS_COPY_DEST,
            D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
            D3D12_BARRIER_LAYOUT_COPY_DEST,
            after_layout,
        )];
        unsafe { barriers::barrier_textures(&command_list7, &b_to_shader) };
        unsafe { barriers::drop_texture_barriers(&mut b_to_shader) };

        if let Some(tex) = state.textures.get_mut(tex_handle) {
            tex.last_layout = after_layout;
        }
    }

    unsafe { command_list.Close() }.context("flush_pending_copies: failed to close command list")?;

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
    .context("flush_pending_copies: failed to signal fence")?;
    wait_for_fence(&logical_device.fence, fence_value)?;

    if let Some(dev) = state.devices.get_mut(&device_handle) {
        dev.fence_value += 1;
    }

    // Staging resources dropped here (GPU is done via fence wait).
    drop(copies);

    tracing::debug!(
        "Flushed {} pending texture copies in one submission",
        count
    );
    Ok(())
}

/// Read texture contents to CPU memory.
/// The texture must have been created with TextureFlags::COPY_SRC.
pub(super) fn read_to_cpu(
    state: &mut Dx12State,
    texture_handle: TextureHandle,
    output: &mut [u8],
) -> Result<()> {
    // Flush any pending texture writes so the data is on the GPU.
    flush_pending_copies(state)?;

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
    let command_list7: ID3D12GraphicsCommandList7 =
        command_list.cast().context("ID3D12GraphicsCommandList7")?;

    // Newly created list is already in recording state; Reset is for reusing a closed list

    let layout_before = texture.last_layout;

    let b_to_src = barriers::texture_barrier_full(
        &texture.resource,
        D3D12_BARRIER_SYNC_ALL,
        D3D12_BARRIER_SYNC_COPY,
        access_for_layout(layout_before),
        D3D12_BARRIER_ACCESS_COPY_SOURCE,
        layout_before,
        D3D12_BARRIER_LAYOUT_COPY_SOURCE,
    );
    unsafe { barriers::barrier_textures(&command_list7, &[b_to_src]) };

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

    let b_back = barriers::texture_barrier_full(
        &texture.resource,
        D3D12_BARRIER_SYNC_COPY,
        D3D12_BARRIER_SYNC_ALL,
        D3D12_BARRIER_ACCESS_COPY_SOURCE,
        access_for_layout(layout_before),
        D3D12_BARRIER_LAYOUT_COPY_SOURCE,
        layout_before,
    );
    unsafe {
        command_list.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None);
        barriers::barrier_textures(&command_list7, &[b_back]);
    }

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
        let is_storage = tex.is_storage;
        let cache_key: TextureCacheKey = (tex.width, tex.height, tex.format, is_storage);
        let cache_entry = CachedTextureResource {
            resource: tex.resource.clone(),
            last_layout: tex.last_layout,
        };

        if let Some(dev) = state.devices.get_mut(&device_handle) {
            dev.resource_registry.unregister_texture(texture_handle);
        }

        let cache = state.texture_cache.entry(cache_key).or_default();
        if cache.len() < MAX_TEXTURE_CACHE_PER_KEY {
            cache.push(cache_entry);
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
