//! Render target management logic.
//!
//! Handles creation, destruction, rendering, and readback of off-screen render targets.

use super::types::RenderTargetState;
use super::utils::{depth_format_to_dxgi, format_to_dxgi};
use super::{render_commands, DeviceHandle, Dx12State, RenderTargetHandle};
use crate::backend::RenderCommand;
use crate::types::{Color, DepthFormat, TextureFormat};
use anyhow::{Context, Result};
use windows::{
    core::Interface,
    Win32::{
        Foundation::RECT,
        Graphics::{Direct3D12::*, Dxgi::Common::*},
    },
};

/// Create a render target without depth buffer.
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Result<RenderTargetHandle> {
    create_with_depth(state, device_handle, width, height, format, None)
}

/// Create a render target with optional depth buffer.
#[allow(clippy::too_many_lines)]
pub(super) fn create_with_depth(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    color_format: TextureFormat,
    depth_format: Option<DepthFormat>,
) -> Result<RenderTargetHandle> {
    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    // Create color render target texture
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
        Format: format_to_dxgi(color_format),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    };

    let clear_value = D3D12_CLEAR_VALUE {
        Format: format_to_dxgi(color_format),
        Anonymous: D3D12_CLEAR_VALUE_0 {
            Color: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let mut texture: Option<ID3D12Resource> = None;
    unsafe {
        logical_device.device.CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &resource_desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some(&clear_value),
            &mut texture,
        )
    }
    .context("Failed to create render target texture")?;
    let texture = texture.context("CreateCommittedResource returned null")?;

    // Create RTV
    let rtv_offset = state.next_rtv_offset;
    state.next_rtv_offset += 1;

    let rtv_handle = unsafe {
        let mut handle = logical_device.rtv_heap.GetCPUDescriptorHandleForHeapStart();
        handle.ptr += (rtv_offset * logical_device.rtv_descriptor_size) as usize;
        handle
    };
    unsafe {
        logical_device
            .device
            .CreateRenderTargetView(&texture, None, rtv_handle);
    }

    // Create depth buffer if requested
    let (depth_texture, dsv_offset) = if let Some(df) = depth_format {
        let depth_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: depth_format_to_dxgi(df),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        };

        let depth_clear = D3D12_CLEAR_VALUE {
            Format: depth_format_to_dxgi(df),
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                    Depth: 1.0,
                    Stencil: 0,
                },
            },
        };

        let mut depth_tex: Option<ID3D12Resource> = None;
        unsafe {
            logical_device.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &depth_desc,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                Some(&depth_clear),
                &mut depth_tex,
            )
        }
        .context("Failed to create depth buffer")?;
        let depth_tex = depth_tex.context("CreateCommittedResource returned null for depth")?;

        let dsv_off = state.next_dsv_offset;
        state.next_dsv_offset += 1;

        let dsv_handle = unsafe {
            let mut handle = logical_device.dsv_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (dsv_off * logical_device.dsv_descriptor_size) as usize;
            handle
        };
        unsafe {
            logical_device
                .device
                .CreateDepthStencilView(&depth_tex, None, dsv_handle);
        }

        (Some(depth_tex), Some(dsv_off))
    } else {
        (None, None)
    };

    // Create command list for this render target
    let command_list: ID3D12GraphicsCommandList = unsafe {
        logical_device.device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &logical_device.command_allocator,
            None,
        )
    }
    .context("Failed to create command list")?;

    // Close the command list initially
    unsafe { command_list.Close() }.ok();

    let handle = state.next_render_target_handle;
    state.next_render_target_handle += 1;

    state.render_targets.insert(
        handle,
        RenderTargetState {
            device_handle,
            width,
            height,
            format: color_format,
            texture,
            rtv_offset,
            depth_format,
            depth_texture,
            dsv_offset,
            staging_buffer: None,
            command_list,
            has_rendered: false,
        },
    );

    tracing::debug!(
        "Created render target {}x{} (handle={}, depth={:?})",
        width,
        height,
        handle,
        depth_format
    );
    Ok(handle)
}

/// Destroy a render target.
pub(super) fn destroy(state: &mut Dx12State, target: RenderTargetHandle) {
    state.render_targets.remove(&target);
}

/// Render commands to a render target.
#[allow(clippy::too_many_lines)]
pub(super) fn render(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    target: RenderTargetHandle,
    commands: &[RenderCommand],
) -> Result<()> {
    // Get device first
    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    // Reset command allocator and command list
    unsafe { logical_device.command_allocator.Reset() }
        .context("Failed to reset command allocator")?;

    let render_target = state
        .render_targets
        .get(&target)
        .context("Invalid render target handle")?;

    if render_target.device_handle != device_handle {
        anyhow::bail!("Render target belongs to a different device");
    }

    let cmd = &render_target.command_list;
    let width = render_target.width;
    let height = render_target.height;

    unsafe { cmd.Reset(&logical_device.command_allocator, None) }
        .context("Failed to reset command list")?;

    // Get RTV handle
    let rtv_handle = unsafe {
        let mut handle = logical_device.rtv_heap.GetCPUDescriptorHandleForHeapStart();
        handle.ptr += (render_target.rtv_offset * logical_device.rtv_descriptor_size) as usize;
        handle
    };

    // Find clear color and depth
    let clear_color = commands
        .iter()
        .find_map(|c| match c {
            RenderCommand::Clear(color) => Some(*color),
            _ => None,
        })
        .unwrap_or(Color::BLACK);
    let clear_depth = commands
        .iter()
        .find_map(|c| match c {
            RenderCommand::ClearDepth(d) => Some(*d),
            _ => None,
        })
        .unwrap_or(1.0);

    // Clear color and set render target (with optional depth/stencil)
    unsafe {
        cmd.ClearRenderTargetView(
            rtv_handle,
            &[clear_color.r, clear_color.g, clear_color.b, clear_color.a],
            None,
        );
    }

    if let Some(dsv_off) = render_target.dsv_offset {
        let dsv_handle = unsafe {
            let mut handle = logical_device.dsv_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (dsv_off * logical_device.dsv_descriptor_size) as usize;
            handle
        };
        unsafe {
            cmd.ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, clear_depth, 0, None);
            cmd.OMSetRenderTargets(1, Some(&rtv_handle), false, Some(&dsv_handle));
        }
    } else {
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&rtv_handle), false, None);
        }
    }

    // Set viewport and scissor
    let viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: width as f32,
        Height: height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let scissor = RECT {
        left: 0,
        top: 0,
        right: width as i32,
        bottom: height as i32,
    };
    unsafe {
        cmd.RSSetViewports(&[viewport]);
        cmd.RSSetScissorRects(&[scissor]);
    }

    // Bind descriptor heaps for bindless rendering
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    unsafe {
        cmd.SetDescriptorHeaps(&[
            Some(logical_device.cbv_srv_uav_heap.clone()),
            Some(logical_device.sampler_heap.clone()),
        ]);
    }

    // Execute render commands
    render_commands::record(cmd, commands, device_handle, state);

    // Transition to copy source for potential readback
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(&render_target.texture) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
            }),
        },
    };
    unsafe { cmd.ResourceBarrier(&[barrier]) };

    // Close and execute
    unsafe { cmd.Close() }.context("Failed to close command list")?;

    let cmd_list: ID3D12CommandList = cmd.cast().context("Failed to cast command list")?;
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

    // Increment fence value for next operation
    if let Some(dev) = state.devices.get_mut(&device_handle) {
        dev.fence_value += 1;
    }

    // Mark as rendered
    if let Some(rt) = state.render_targets.get_mut(&target) {
        rt.has_rendered = true;
    }

    Ok(())
}

/// Read render target contents to CPU memory.
#[allow(clippy::too_many_lines)]
pub(super) fn read_to_cpu(
    state: &mut Dx12State,
    target: RenderTargetHandle,
    output: &mut [u8],
) -> Result<()> {
    let render_target = state
        .render_targets
        .get(&target)
        .context("Invalid render target handle")?;

    if !render_target.has_rendered {
        anyhow::bail!("Cannot read from render target that hasn't been rendered to");
    }

    let width = render_target.width;
    let height = render_target.height;
    let format = render_target.format;
    let expected_size = (width * height * format.bytes_per_pixel()) as usize;

    if output.len() < expected_size {
        anyhow::bail!(
            "Output buffer too small: {} < {}",
            output.len(),
            expected_size
        );
    }

    let device_handle = render_target.device_handle;
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Ensure staging buffer exists
    let needs_staging = render_target.staging_buffer.is_none();
    if needs_staging {
        let row_pitch = ((width * format.bytes_per_pixel() + 255) & !255) as u64; // 256-byte aligned
        let staging_size = row_pitch * height as u64;

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
        .context("Failed to create staging buffer")?;
        let staging_buffer = staging_buffer.context("CreateCommittedResource returned null")?;

        // Store staging buffer in render target
        if let Some(rt) = state.render_targets.get_mut(&target) {
            rt.staging_buffer = Some(staging_buffer);
        }
    }

    // Get render target again (borrow checker)
    let render_target = state
        .render_targets
        .get(&target)
        .context("Invalid render target handle")?;

    let staging_buffer = render_target
        .staging_buffer
        .as_ref()
        .context("Staging buffer not available")?;

    // Copy texture to staging buffer
    let cmd = &render_target.command_list;
    unsafe { cmd.Reset(&logical_device.command_allocator, None) }
        .context("Failed to reset command list")?;

    let row_pitch = ((width * format.bytes_per_pixel() + 255) & !255) as u64;
    let src_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&render_target.texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };

    let dst_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(staging_buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: format_to_dxgi(format),
                    Width: width,
                    Height: height,
                    Depth: 1,
                    RowPitch: row_pitch as u32,
                },
            },
        },
    };

    unsafe { cmd.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None) };
    unsafe { cmd.Close() }.context("Failed to close command list")?;

    let cmd_list: ID3D12CommandList = cmd.cast().context("Failed to cast command list")?;
    unsafe {
        logical_device
            .command_queue
            .ExecuteCommandLists(&[Some(cmd_list)]);
    }

    // Wait for copy to complete
    let fence_value = logical_device.fence_value;
    unsafe {
        logical_device
            .command_queue
            .Signal(&logical_device.fence, fence_value)
    }
    .context("Failed to signal fence")?;
    wait_for_fence(&logical_device.fence, fence_value)?;

    // Increment fence value
    if let Some(dev) = state.devices.get_mut(&device_handle) {
        dev.fence_value += 1;
    }

    // Map and copy data
    let mut mapped_data: *mut u8 = std::ptr::null_mut();
    unsafe {
        staging_buffer.Map(
            0,
            None,
            Some(&mut mapped_data as *mut *mut u8 as *mut *mut _),
        )
    }
    .context("Failed to map staging buffer")?;

    // Copy row by row if there's padding
    let bytes_per_row = width * format.bytes_per_pixel();
    if row_pitch == bytes_per_row as u64 {
        // No padding, copy entire buffer
        unsafe {
            std::ptr::copy_nonoverlapping(mapped_data, output.as_mut_ptr(), expected_size);
        }
    } else {
        // Copy row by row
        for y in 0..height {
            let src_offset = (y as u64 * row_pitch) as usize;
            let dst_offset = (y * bytes_per_row) as usize;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped_data.add(src_offset),
                    output.as_mut_ptr().add(dst_offset),
                    bytes_per_row as usize,
                );
            }
        }
    }

    unsafe { staging_buffer.Unmap(0, None) };

    Ok(())
}

/// Helper to wait for a fence value.
fn wait_for_fence(fence: &ID3D12Fence, value: u64) -> Result<()> {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::{CreateEventA, WaitForSingleObject, INFINITE};

    let event =
        unsafe { CreateEventA(None, false, false, None) }.context("Failed to create event")?;
    unsafe { fence.SetEventOnCompletion(value, event) }
        .context("Failed to set event on completion")?;
    unsafe { WaitForSingleObject(event, INFINITE) };
    unsafe { CloseHandle(event) }.ok();
    Ok(())
}
