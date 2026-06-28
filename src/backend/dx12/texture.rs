//! Texture management operations.

use super::barriers;
use super::types::{Dx12State, PendingDeletion, TextureState};
use super::utils::{execute_command_lists_and_signal_device, format_to_dxgi, wait_for_fence};
use super::{BufferHandle, DeviceHandle, TextureHandle};
use crate::types::{TextureFlags, TextureFormat, TextureKind};
use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

/// Staged texture upload (CPU-filled staging + copy footprint) before GPU copy.
pub(super) struct StagedTextureUpload {
    pub staging_entry: super::staging::TextureStagingEntry,
    pub footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    pub texture_handle: TextureHandle,
    pub dst_x: u32,
    pub dst_y: u32,
    /// Texture layout at the time this copy was enqueued.
    pub layout_before: D3D12_BARRIER_LAYOUT,
}

pub(super) struct TextureUploadRegion<'a> {
    pub texture_handle: TextureHandle,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
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

/// Initial enhanced barrier for a storage 2D texture in COMMON → UAV layout (shared by committed + placed creates).
pub(super) fn init_storage_texture_uav_layout(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    resource: &ID3D12Resource,
) -> Result<D3D12_BARRIER_LAYOUT> {
    let last_layout = D3D12_BARRIER_LAYOUT_COMMON;
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
    let init_cmd7: ID3D12GraphicsCommandList7 = init_cmd.cast().context("ID3D12GraphicsCommandList7")?;

    let b = barriers::texture_barrier_full(
        resource,
        D3D12_BARRIER_SYNC_NONE,
        D3D12_BARRIER_SYNC_NONE,
        D3D12_BARRIER_ACCESS_NO_ACCESS,
        D3D12_BARRIER_ACCESS_NO_ACCESS,
        last_layout,
        D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS,
    );
    unsafe {
        barriers::barrier_textures(&init_cmd7, &[b]);
        init_cmd.Close()
    }
    .context("Failed to close init barrier command list")?;

    let cmd_list: ID3D12CommandList = init_cmd.cast().context("Failed to cast init command list")?;
    let fence_value = execute_command_lists_and_signal_device(logical_device, &[Some(cmd_list)])?;
    super::utils::wait_for_fence(&logical_device.fence, fence_value)?;

    Ok(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS)
}

/// Create a texture.
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: TextureKind,
    flags: TextureFlags,
) -> Result<TextureHandle> {
    let is_storage = matches!(access, TextureKind::Direct | TextureKind::DirectInterpolated);
    let is_dual_access = matches!(access, TextureKind::DirectInterpolated);

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

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
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
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
    if hr.is_err() {
        crate::signal::push_sync_signal(crate::signal::Signal::Oversubscribed {
            reason: crate::signal::OversubscribedReason::TextureHeap,
            size_hint: (width as u64) * (height as u64) * 4,
        });
    }
    hr.context("Failed to create texture")?;
    let resource = resource.context("CreateCommittedResource returned null")?;

    // Get texture handle first (needed for registry)
    let handle = state.textures.write().unwrap().alloc_handle();

    // Create SRV - use unified resource registry to avoid descriptor heap collisions
    // (textures and buffers share the same CBV/SRV/UAV heap)
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
    let srv_offset = logical_device
        .descriptors
        .lock()
        .unwrap()
        .resource_registry
        .register_texture(handle);

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
        let mut cpu_handle = logical_device.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
        cpu_handle.ptr += (srv_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
        cpu_handle
    };
    unsafe {
        logical_device
            .device
            .CreateShaderResourceView(&resource, Some(&srv_desc), srv_cpu_handle);
    }

    // For storage images (Direct access): create UAV so compute shaders can write via RWTexture2D.
    // bindless_offset must point to UAV for goldy_direct_spatial.
    let bindless_offset = if is_storage {
        let uav_offset = logical_device
            .descriptors
            .lock()
            .unwrap()
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
            let mut cpu_handle = logical_device.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
            cpu_handle.ptr += (uav_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
            cpu_handle
        };
        unsafe {
            logical_device
                .device
                .CreateUnorderedAccessView(Some(&resource), None, Some(&uav_desc), uav_cpu_handle);
        }
        Some(uav_offset)
    } else {
        Some(srv_offset)
    };

    // For DirectInterpolated, additionally register a sampled SRV slot.
    let sampled_bindless_offset = if is_dual_access {
        let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
        let srv2_offset = logical_device
            .descriptors
            .lock()
            .unwrap()
            .resource_registry
            .register_texture_srv(handle);
        let srv2_cpu_handle = unsafe {
            let mut cpu_handle = logical_device.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
            cpu_handle.ptr += (srv2_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
            cpu_handle
        };
        let srv2_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
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
        unsafe {
            logical_device
                .device
                .CreateShaderResourceView(&resource, Some(&srv2_desc), srv2_cpu_handle);
        }
        Some(srv2_offset)
    } else {
        None
    };

    let last_layout = if is_storage {
        init_storage_texture_uav_layout(state, device_handle, &resource)?
    } else {
        D3D12_BARRIER_LAYOUT_COMMON
    };

    state.textures.write().unwrap().entries.insert(
        handle,
        TextureState {
            device_handle,
            width,
            height,
            format,
            resource,
            srv_offset,
            bindless_offset,
            sampled_bindless_offset,
            last_layout,
            is_storage,
            transient_placed: false,
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

/// Build a staged full-texture upload (caller runs GPU copy via task graph or
/// [`execute_staged_uploads_sync`]).
pub(super) fn stage_texture_upload_full(
    devices: &std::collections::HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    textures: &std::collections::HashMap<TextureHandle, super::types::TextureState>,
    pool: &mut super::staging::TextureStagingPool,
    texture_handle: TextureHandle,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<StagedTextureUpload> {
    let texture = textures.get(&texture_handle).context("Invalid texture handle")?;

    if texture.width != width || texture.height != height {
        anyhow::bail!("Texture dimensions mismatch");
    }

    let device_handle = texture.device_handle;
    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

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

    let staging_entry = pool.acquire(logical_device, staging_size)?;

    let mapped_ptr = staging_entry.mapped_ptr();
    let texture_format = texture.format;
    let bytes_per_row = (width * texture_format.bytes_per_pixel()) as usize;
    let row_pitch_bytes = footprint.Footprint.RowPitch as usize;
    for row in 0..height {
        let src_offset = (row as usize) * bytes_per_row;
        let dst_offset = (footprint.Offset + row as u64 * row_pitch_bytes as u64) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(src_offset), mapped_ptr.add(dst_offset), bytes_per_row);
        }
    }

    let layout_before = texture.last_layout;

    Ok(StagedTextureUpload {
        staging_entry,
        footprint,
        texture_handle,
        dst_x: 0,
        dst_y: 0,
        layout_before,
    })
}

/// Build a staged subregion texture upload.
pub(super) fn stage_texture_upload_region(
    devices: &std::collections::HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    textures: &std::collections::HashMap<TextureHandle, super::types::TextureState>,
    pool: &mut super::staging::TextureStagingPool,
    region: TextureUploadRegion<'_>,
) -> Result<StagedTextureUpload> {
    let TextureUploadRegion {
        texture_handle,
        x,
        y,
        width,
        height,
        data,
    } = region;
    let texture = textures.get(&texture_handle).context("Invalid texture handle")?;

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
        anyhow::bail!("Data size mismatch: expected {}, got {}", expected_size, data.len());
    }

    let device_handle = texture.device_handle;
    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

    let region_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format_to_dxgi(texture.format),
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
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

    let staging_entry = pool.acquire(logical_device, staging_size)?;

    let texture_format = texture.format;
    let bytes_per_row = (width * texture_format.bytes_per_pixel()) as usize;
    let mapped_ptr = staging_entry.mapped_ptr();

    let row_pitch_bytes = footprint.Footprint.RowPitch as usize;
    for row in 0..height {
        let src_offset = (row as usize) * bytes_per_row;
        let dst_offset = (footprint.Offset + row as u64 * row_pitch_bytes as u64) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(src_offset), mapped_ptr.add(dst_offset), bytes_per_row);
        }
    }

    let layout_before = texture.last_layout;

    Ok(StagedTextureUpload {
        staging_entry,
        footprint,
        texture_handle,
        dst_x: x,
        dst_y: y,
        layout_before,
    })
}

/// Stage a [`GpuCommand::CopyBufferToTexture`] upload from a CPU-writable source buffer.
#[allow(clippy::too_many_arguments)]
pub(super) fn stage_copy_buffer_to_texture_upload(
    devices: &std::collections::HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    textures: &std::collections::HashMap<TextureHandle, super::types::TextureState>,
    buffers: &std::collections::HashMap<super::super::BufferHandle, super::types::BufferState>,
    pool: &mut super::staging::TextureStagingPool,
    src: super::super::BufferHandle,
    src_offset: u64,
    texture_handle: TextureHandle,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<StagedTextureUpload> {
    let texture = textures.get(&texture_handle).context("Invalid texture handle")?;
    let bpp = texture.format.bytes_per_pixel();
    let flat_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|h| h.checked_mul(bpp as usize))
        .context("CopyBufferToTexture: flat byte size overflow")?;
    let data = super::buffer::cpu_writable_flat_slice(buffers, src, src_offset, flat_len)?;
    if x == 0 && y == 0 && width == texture.width && height == texture.height {
        stage_texture_upload_full(devices, textures, pool, texture_handle, data, width, height)
    } else {
        stage_texture_upload_region(
            devices,
            textures,
            pool,
            TextureUploadRegion {
                texture_handle,
                x,
                y,
                width,
                height,
                data,
            },
        )
    }
}

/// Record one staged texture upload on an open command list (task graph / compute submit).
pub(super) fn record_staged_texture_upload(
    command_list: &ID3D12GraphicsCommandList,
    command_list7: &ID3D12GraphicsCommandList7,
    textures: &mut std::collections::HashMap<TextureHandle, TextureState>,
    upload: &StagedTextureUpload,
) -> Result<()> {
    let after_layout = D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE;
    let texture = textures
        .get(&upload.texture_handle)
        .context("record_staged_texture_upload: invalid texture")?;
    let layout_before = upload.layout_before;

    let mut b_to_copy = [barriers::texture_barrier_full(
        &texture.resource,
        D3D12_BARRIER_SYNC_ALL,
        D3D12_BARRIER_SYNC_COPY,
        access_for_layout(layout_before),
        D3D12_BARRIER_ACCESS_COPY_DEST,
        layout_before,
        D3D12_BARRIER_LAYOUT_COPY_DEST,
    )];
    unsafe { barriers::barrier_textures(command_list7, &b_to_copy) };
    unsafe { barriers::drop_texture_barriers(&mut b_to_copy) };

    let src_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&upload.staging_entry.resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: upload.footprint.Offset,
                Footprint: upload.footprint.Footprint,
            },
        },
    };

    let dst_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&texture.resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    };

    unsafe {
        command_list.CopyTextureRegion(&dst_location, upload.dst_x, upload.dst_y, 0, &src_location, None);
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
    unsafe { barriers::barrier_textures(command_list7, &b_to_shader) };
    unsafe { barriers::drop_texture_barriers(&mut b_to_shader) };

    if let Some(tex) = textures.get_mut(&upload.texture_handle) {
        tex.last_layout = after_layout;
    }
    Ok(())
}

/// Write data to a texture (synchronous: submits immediately and waits).
pub(super) fn write(
    state: &mut Dx12State,
    texture_handle: TextureHandle,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<()> {
    let staged = {
        let mut pool = super::staging::TextureStagingPool::new();
        stage_texture_upload_full(
            &state.devices,
            &state.textures.read().unwrap().entries,
            &mut pool,
            texture_handle,
            data,
            width,
            height,
        )?
    };
    execute_staged_uploads_sync(state, vec![staged])?;
    tracing::debug!("Wrote {}x{} texture (sync upload)", width, height);
    Ok(())
}

/// Write data to a subregion of a texture (synchronous).
pub(super) fn write_region(
    state: &mut Dx12State,
    texture_handle: TextureHandle,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<()> {
    let staged = {
        let mut pool = super::staging::TextureStagingPool::new();
        stage_texture_upload_region(
            &state.devices,
            &state.textures.read().unwrap().entries,
            &mut pool,
            TextureUploadRegion {
                texture_handle,
                x,
                y,
                width,
                height,
                data,
            },
        )?
    };
    execute_staged_uploads_sync(state, vec![staged])?;
    tracing::debug!(
        "Wrote {}x{} texture region at ({},{}) (sync upload)",
        width,
        height,
        x,
        y,
    );
    Ok(())
}

/// Execute staged texture uploads on a dedicated command list and wait (sync path).
pub(super) fn execute_staged_uploads_sync(state: &mut Dx12State, uploads: Vec<StagedTextureUpload>) -> Result<()> {
    if uploads.is_empty() {
        return Ok(());
    }

    let copies = uploads;
    let count = copies.len();

    let device_handle = {
        let textures_read = state.textures.read().unwrap();
        textures_read
            .entries
            .get(&copies[0].texture_handle)
            .context("flush_pending_copies: invalid texture handle")?
            .device_handle
    };

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
    let command_list7: ID3D12GraphicsCommandList7 = command_list.cast().context("ID3D12GraphicsCommandList7")?;

    // Group copies by texture handle (preserving order within each texture).
    let mut texture_order: Vec<TextureHandle> = Vec::new();
    let mut groups: std::collections::HashMap<TextureHandle, Vec<usize>> = std::collections::HashMap::new();
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
        let layout_before = copies[indices[0]].layout_before;
        let resource = {
            let textures_read = state.textures.read().unwrap();
            textures_read
                .entries
                .get(tex_handle)
                .context("flush_pending_copies: texture disappeared")?
                .resource
                .clone()
        };

        let mut b_to_copy = [barriers::texture_barrier_full(
            &resource,
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
                pResource: unsafe { std::mem::transmute_copy(&copy.staging_entry.resource) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: copy.footprint.Offset,
                        Footprint: copy.footprint.Footprint,
                    },
                },
            };

            let dst_location = D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { std::mem::transmute_copy(&resource) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };

            unsafe {
                command_list.CopyTextureRegion(&dst_location, copy.dst_x, copy.dst_y, 0, &src_location, None);
            }
        }

        let mut b_to_shader = [barriers::texture_barrier_full(
            &resource,
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_ACCESS_COPY_DEST,
            D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
            D3D12_BARRIER_LAYOUT_COPY_DEST,
            after_layout,
        )];
        unsafe { barriers::barrier_textures(&command_list7, &b_to_shader) };
        unsafe { barriers::drop_texture_barriers(&mut b_to_shader) };

        {
            let mut textures_write = state.textures.write().unwrap();
            if let Some(tex) = textures_write.entries.get_mut(tex_handle) {
                tex.last_layout = after_layout;
            }
        }
    }

    unsafe { command_list.Close() }.context("flush_pending_copies: failed to close command list")?;

    let cmd_list: ID3D12CommandList = command_list.cast().context("Failed to cast command list")?;
    let logical_device = state.devices.get(&device_handle).unwrap();
    let fence_value = execute_command_lists_and_signal_device(logical_device, &[Some(cmd_list)])?;
    wait_for_fence(&logical_device.fence, fence_value)?;

    // Release and reclaim the staging entries back to the pool immediately.
    // The GPU is idle for this submission (we just waited), so we can safely destroy them.
    for copy in copies {
        unsafe { copy.staging_entry.destroy() };
    }

    tracing::debug!("Flushed {} pending texture copies in one submission", count);
    Ok(())
}

/// Query grant-readback staging layout for a 2D texture allocation.
pub(super) fn query_texture_copy_footprint(
    state: &Dx12State,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Result<crate::backend::TextureCopyFootprint> {
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
    let dxgi_format = format_to_dxgi(format);
    let res_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: dxgi_format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total_bytes: u64 = 0;
    unsafe {
        logical_device.device.GetCopyableFootprints(
            &res_desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total_bytes),
        );
    }
    let logical_bytes = (width as u64) * (height as u64) * (format.bytes_per_pixel() as u64);
    Ok(crate::backend::TextureCopyFootprint {
        width,
        height,
        format,
        logical_bytes,
        staging_bytes: total_bytes,
        row_pitch: footprint.Footprint.RowPitch,
        footprint_offset: footprint.Offset,
    })
}

/// Record copy from a texture into a grant-readback staging buffer.
pub(super) fn record_copy_texture_to_readback(
    command_list: &ID3D12GraphicsCommandList,
    command_list7: &ID3D12GraphicsCommandList7,
    textures: &mut std::collections::HashMap<TextureHandle, TextureState>,
    buffers: &std::collections::HashMap<BufferHandle, super::types::BufferState>,
    src: TextureHandle,
    dst: BufferHandle,
    layout: crate::backend::TextureCopyFootprint,
) -> Result<()> {
    // Extract everything we need from the immutable borrow before any mutable access.
    let (src_resource, dst_resource, layout_before, is_storage, dxgi_format) = {
        let texture = textures
            .get(&src)
            .context("CopyTextureToReadback: src texture not found")?;
        let dst_buf = buffers
            .get(&dst)
            .context("CopyTextureToReadback: dst buffer not found")?;
        (
            texture.resource.clone(),
            dst_buf.resource.clone(),
            texture.last_layout,
            texture.is_storage,
            format_to_dxgi(layout.format),
        )
    };

    let b_to_src = barriers::texture_barrier_full(
        &src_resource,
        D3D12_BARRIER_SYNC_ALL,
        D3D12_BARRIER_SYNC_COPY,
        access_for_layout(layout_before),
        D3D12_BARRIER_ACCESS_COPY_SOURCE,
        layout_before,
        D3D12_BARRIER_LAYOUT_COPY_SOURCE,
    );
    unsafe { barriers::barrier_textures(command_list7, &[b_to_src]) };

    let src_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&src_resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    };
    let dst_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&dst_resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: layout.footprint_offset,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: dxgi_format,
                    Width: layout.width,
                    Height: layout.height,
                    Depth: 1,
                    RowPitch: layout.row_pitch,
                },
            },
        },
    };
    unsafe { command_list.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None) };

    let post_access = if is_storage {
        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS
    } else {
        D3D12_BARRIER_ACCESS_SHADER_RESOURCE
    };
    let post_layout = if is_storage {
        D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS
    } else {
        D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE
    };
    let b_back = barriers::texture_barrier_full(
        &src_resource,
        D3D12_BARRIER_SYNC_COPY,
        D3D12_BARRIER_SYNC_ALL,
        D3D12_BARRIER_ACCESS_COPY_SOURCE,
        post_access,
        D3D12_BARRIER_LAYOUT_COPY_SOURCE,
        post_layout,
    );
    unsafe { barriers::barrier_textures(command_list7, &[b_back]) };

    // Update tracked layout so subsequent copies (retained resubmits) use the correct
    // layout_before. Without this, the pre-copy barrier would repeat the same transition
    // correctly only by accident (round-trip textures happen to match), but any code path
    // that transitions the texture to a layout other than post_layout between copies
    // would compute a wrong layout_before on the next call.
    if let Some(tex) = textures.get_mut(&src) {
        tex.last_layout = post_layout;
    }

    Ok(())
}

/// Read texture contents to CPU memory.
/// The texture must have been created with TextureFlags::COPY_SRC.
pub(super) fn read_to_cpu(state: &mut Dx12State, texture_handle: TextureHandle, output: &mut [u8]) -> Result<()> {
    use windows::Win32::Graphics::Direct3D12::*;

    let textures_read = state.textures.read().unwrap();
    let texture = textures_read
        .entries
        .get(&texture_handle)
        .context("Invalid texture handle")?;

    let device_handle = texture.device_handle;
    let width = texture.width;
    let height = texture.height;
    let format = texture.format;
    let expected_size = (width * height * format.bytes_per_pixel()) as usize;

    if output.len() < expected_size {
        anyhow::bail!("Output buffer too small: {} < {}", output.len(), expected_size);
    }

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

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
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };

    // Check for device removal before allocating (TDR during prior compute work)
    let removed_reason = unsafe { logical_device.device.GetDeviceRemovedReason() };
    if removed_reason.is_err() {
        anyhow::bail!("Device removed before texture readback: {:?}", removed_reason);
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
    unsafe { logical_device.command_allocator.Reset() }.context("Failed to reset command allocator")?;

    let command_list: ID3D12GraphicsCommandList = unsafe {
        logical_device.device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &logical_device.command_allocator,
            None,
        )
    }
    .context("Failed to create command list")?;
    let command_list7: ID3D12GraphicsCommandList7 = command_list.cast().context("ID3D12GraphicsCommandList7")?;

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
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
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
    let fence_value = execute_command_lists_and_signal_device(logical_device, &[Some(cmd_list)])?;
    wait_for_fence(&logical_device.fence, fence_value)?;

    if let Some(dev) = state.devices.get(&device_handle) {
        // Check for device removal (compute passes may have caused TDR)
        let removed = unsafe { dev.device.GetDeviceRemovedReason() };
        if removed.is_err() {
            anyhow::bail!("Device removed before texture readback map: {:?}", removed);
        }
    }

    let mut mapped_data: *mut u8 = std::ptr::null_mut();
    unsafe { staging_buffer.Map(0, None, Some(&mut mapped_data as *mut *mut u8 as *mut *mut _)) }
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

/// Destroy a texture, queueing the D3D12 resource and bindless descriptor slots
/// for deferred deletion after in-flight GPU work completes.
pub(super) fn destroy(state: &mut Dx12State, texture_handle: TextureHandle) {
    if let Some(tex) = state.textures.write().unwrap().entries.remove(&texture_handle) {
        if let Some(dev) = state.devices.get(&tex.device_handle) {
            if tex.transient_placed {
                dev.descriptors.lock().unwrap().reclaim_texture_slots(texture_handle);
                return;
            }
            let last_fence = dev
                .timeline_next
                .load(std::sync::atomic::Ordering::Relaxed)
                .saturating_sub(1);
            dev.deletion_queue.lock().unwrap().queue(
                last_fence,
                PendingDeletion::Texture {
                    texture_handle,
                    resource: tex.resource,
                },
            );
        }
    }
}

/// Get the bindless index for a texture.
pub(super) fn bindless_index(state: &Dx12State, texture_handle: TextureHandle) -> Option<u32> {
    state
        .textures
        .read()
        .unwrap()
        .entries
        .get(&texture_handle)
        .and_then(|t| t.bindless_offset)
}

/// For `TextureKind::DirectInterpolated` textures, return the sampled-texture (SRV) slot.
pub(super) fn bindless_sampled_index(state: &Dx12State, texture_handle: TextureHandle) -> Option<u32> {
    state
        .textures
        .read()
        .unwrap()
        .entries
        .get(&texture_handle)
        .and_then(|t| t.sampled_bindless_offset)
}
