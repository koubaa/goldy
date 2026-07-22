//! Render target management logic.
//!
//! Handles creation, destruction, rendering, and readback of off-screen render targets.

use super::barriers;
use super::types::RenderTargetState;
use super::utils::{depth_format_to_dxgi, execute_command_lists_and_signal_device, format_to_dxgi, wait_for_fence};
use super::{render_commands, DeviceHandle, Dx12State, RenderTargetHandle};
use crate::backend::ContextHandle;
use crate::backend::RenderCommand;
use crate::types::{DepthFormat, TargetLoad, TextureFormat};
use anyhow::{Context, Result};
use windows::{
    core::Interface,
    Win32::{
        Foundation::RECT,
        Graphics::{Direct3D12::*, Dxgi::Common::*},
    },
};

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
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

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
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
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
            D3D12_RESOURCE_STATE_COMMON,
            Some(&clear_value),
            &mut texture,
        )
    }
    .context("Failed to create render target texture")?;
    let texture = texture.context("CreateCommittedResource returned null")?;

    // Create RTV
    let rtv_offset = state.free_rtv_offsets.pop().unwrap_or_else(|| {
        let off = state.next_rtv_offset;
        state.next_rtv_offset += 1;
        off
    });

    let rtv_handle = unsafe {
        let mut handle = logical_device.rtv_heap.GetCPUDescriptorHandleForHeapStart();
        handle.ptr += (rtv_offset * logical_device.rtv_descriptor_size) as usize;
        handle
    };
    unsafe {
        logical_device.device.CreateRenderTargetView(&texture, None, rtv_handle);
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
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        };

        let depth_clear = D3D12_CLEAR_VALUE {
            Format: depth_format_to_dxgi(df),
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 },
            },
        };

        let mut depth_tex: Option<ID3D12Resource> = None;
        unsafe {
            logical_device.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &depth_desc,
                D3D12_RESOURCE_STATE_COMMON,
                Some(&depth_clear),
                &mut depth_tex,
            )
        }
        .context("Failed to create depth buffer")?;
        let depth_tex = depth_tex.context("CreateCommittedResource returned null for depth")?;

        let dsv_off = state.free_dsv_offsets.pop().unwrap_or_else(|| {
            let off = state.next_dsv_offset;
            state.next_dsv_offset += 1;
            off
        });

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
    let command_list: ID3D12GraphicsCommandList7 = command_list.cast().context("ID3D12GraphicsCommandList7")?;

    // Close the command list initially
    unsafe { command_list.Close() }.ok();

    let handle = state.render_targets.write().unwrap().alloc_handle();

    state.render_targets.write().unwrap().entries.insert(
        handle,
        RenderTargetState {
            device_handle,
            width,
            height,
            texture,
            rtv_offset,
            depth_format,
            depth_texture,
            dsv_offset,
            command_list,
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
/// Records COMMON -> RENDER_TARGET barriers, clear, viewport/scissor, descriptor heap
/// binding, draw commands, and RENDER_TARGET -> COPY_SOURCE barrier into `cmd_list`.
/// Does NOT close/execute/signal.
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
pub(super) fn record_render_pass_to_list_with_record(
    record: &super::submit_session::Dx12RecordState<'_>,
    device_handle: DeviceHandle,
    recording_ctx: Option<ContextHandle>,
    target: RenderTargetHandle,
    color_load: TargetLoad,
    commands: &[RenderCommand],
    cmd_list: &ID3D12GraphicsCommandList7,
    frame_table_prologue_already_recorded: bool,
) -> Result<(bool, Option<u32>)> {
    let logical_device = record.devices.get(&device_handle).context("Invalid device handle")?;

    let render_targets_read = record.render_targets.read().unwrap();
    let render_target = render_targets_read
        .entries
        .get(&target)
        .context("Invalid render target handle")?;

    if render_target.device_handle != device_handle {
        anyhow::bail!("Render target belongs to a different device");
    }

    let cmd_gfx: &ID3D12GraphicsCommandList = unsafe { std::mem::transmute(cmd_list) };
    let width = render_target.width;
    let height = render_target.height;

    let rtv_handle = unsafe {
        let mut handle = logical_device.rtv_heap.GetCPUDescriptorHandleForHeapStart();
        handle.ptr += (render_target.rtv_offset * logical_device.rtv_descriptor_size) as usize;
        handle
    };

    let clear_depth = commands
        .iter()
        .find_map(|c| match c {
            RenderCommand::ClearDepth(d) => Some(*d),
            _ => None,
        })
        .unwrap_or(1.0);

    // COMMON -> render target layout for color + optional depth.
    let color_tex = barriers::texture_barrier_full(
        &render_target.texture,
        D3D12_BARRIER_SYNC_NONE,
        D3D12_BARRIER_SYNC_RENDER_TARGET,
        D3D12_BARRIER_ACCESS_NO_ACCESS,
        D3D12_BARRIER_ACCESS_RENDER_TARGET,
        D3D12_BARRIER_LAYOUT_COMMON,
        D3D12_BARRIER_LAYOUT_RENDER_TARGET,
    );
    let mut initial_tex_barriers = vec![color_tex];
    if let Some(ref depth_res) = render_target.depth_texture {
        initial_tex_barriers.push(barriers::texture_barrier_full(
            depth_res,
            D3D12_BARRIER_SYNC_NONE,
            D3D12_BARRIER_SYNC_DEPTH_STENCIL,
            D3D12_BARRIER_ACCESS_NO_ACCESS,
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,
            D3D12_BARRIER_LAYOUT_COMMON,
            D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
        ));
    }
    unsafe {
        barriers::barrier_textures(cmd_list, &initial_tex_barriers);
    }

    match color_load {
        TargetLoad::Clear(clear_color) => unsafe {
            cmd_gfx.ClearRenderTargetView(
                rtv_handle,
                &[clear_color.r, clear_color.g, clear_color.b, clear_color.a],
                None,
            );
        },
        TargetLoad::Discard => unsafe {
            cmd_gfx.DiscardResource(&render_target.texture, None);
        },
        TargetLoad::Load => {}
    }

    if let Some(dsv_off) = render_target.dsv_offset {
        let dsv_handle = unsafe {
            let mut handle = logical_device.dsv_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (dsv_off * logical_device.dsv_descriptor_size) as usize;
            handle
        };
        unsafe {
            cmd_gfx.ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, clear_depth, 0, None);
            cmd_gfx.OMSetRenderTargets(1, Some(&rtv_handle), false, Some(&dsv_handle));
        }
    } else {
        unsafe {
            cmd_gfx.OMSetRenderTargets(1, Some(&rtv_handle), false, None);
        }
    }

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
        cmd_gfx.RSSetViewports(&[viewport]);
        cmd_gfx.RSSetScissorRects(&[scissor]);
    }

    unsafe {
        cmd_gfx.SetDescriptorHeaps(&[
            Some(logical_device.cbv_srv_uav_heap.clone()),
            Some(logical_device.sampler_heap.clone()),
        ]);
    }

    let (staging_data, lowered, has_bindings) = super::frame_table::prepare_render_commands(record, commands)?;
    let mut prologue_row = None;
    if has_bindings {
        if frame_table_prologue_already_recorded {
            super::frame_table::sync_table_row_to_device(record, device_handle, cmd_list, &staging_data)?;
        } else if let Some(ctx) = recording_ctx {
            prologue_row = Some(super::frame_table::record_prologue(
                record.contexts,
                logical_device,
                ctx,
                &record.frame_table,
                &record.buffers.read().unwrap().entries,
                cmd_list,
                &staging_data,
            )?);
        } else {
            prologue_row = Some(super::frame_table::record_prologue_legacy(
                &record.frame_table,
                &logical_device.fence,
                &record.buffers.read().unwrap().entries,
                cmd_list,
                &staging_data,
            )?);
        }
    }

    render_commands::record(cmd_list, &lowered, device_handle, record)?;

    // RENDER_TARGET -> COPY_SOURCE for potential readback
    let to_copy = barriers::texture_barrier_full(
        &render_target.texture,
        D3D12_BARRIER_SYNC_RENDER_TARGET,
        D3D12_BARRIER_SYNC_COPY,
        D3D12_BARRIER_ACCESS_RENDER_TARGET,
        D3D12_BARRIER_ACCESS_COPY_SOURCE,
        D3D12_BARRIER_LAYOUT_RENDER_TARGET,
        D3D12_BARRIER_LAYOUT_COPY_SOURCE,
    );
    unsafe { barriers::barrier_textures(cmd_list, &[to_copy]) };

    Ok((has_bindings, prologue_row))
}

/// Record an offscreen render pass into an existing command list without closing/executing.
#[allow(clippy::too_many_lines)]
pub(super) fn record_render_pass_to_list(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    target: RenderTargetHandle,
    color_load: TargetLoad,
    commands: &[RenderCommand],
    cmd_list: &ID3D12GraphicsCommandList7,
    frame_table_prologue_already_recorded: bool,
) -> Result<(bool, Option<u32>)> {
    let record = super::submit_session::record_state_for_legacy_render(state, device_handle)?;
    record_render_pass_to_list_with_record(
        &record,
        device_handle,
        None,
        target,
        color_load,
        commands,
        cmd_list,
        frame_table_prologue_already_recorded,
    )
}

/// Render commands to a render target.
#[allow(clippy::too_many_lines)]
pub(super) fn render(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    target: RenderTargetHandle,
    color_load: TargetLoad,
    commands: &[RenderCommand],
) -> Result<()> {
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    // Reset the single shared command allocator. This is safe only because we do
    // a blocking CPU wait at the end of every render_to_target call (see below);
    // that guarantees no GPU work recorded with this allocator is still in-flight.
    // The compute/submit_graph paths avoid this stall by using a pool of allocators
    // (ComputeAllocatorSlot) where each slot is reset only after its fence retires.
    // Upgrading render_to_target to a pool would eliminate the wait but is not yet done.
    unsafe { logical_device.command_allocator.Reset() }.context("Failed to reset command allocator")?;

    let cmd = {
        let render_targets_read = state.render_targets.read().unwrap();
        let render_target = render_targets_read
            .entries
            .get(&target)
            .context("Invalid render target handle")?;

        if render_target.device_handle != device_handle {
            anyhow::bail!("Render target belongs to a different device");
        }

        render_target.command_list.clone()
    };
    let cmd_gfx: &ID3D12GraphicsCommandList = unsafe { std::mem::transmute(&cmd) };

    unsafe { cmd_gfx.Reset(&logical_device.command_allocator, None) }.context("Failed to reset command list")?;

    let ft = super::frame_table::ensure_legacy_frame_table(state, device_handle)?;
    let mut row_guard = super::frame_table::RowReservation::new(&ft);
    let (_, prologue_row) =
        record_render_pass_to_list(state, device_handle, target, color_load, commands, &cmd, false)?;
    if let Some(row) = prologue_row {
        row_guard.set(row);
    }

    let cmd_gfx: &ID3D12GraphicsCommandList = unsafe { std::mem::transmute(&cmd) };
    unsafe { cmd_gfx.Close() }.context("Failed to close command list")?;

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    let cmd_list: ID3D12CommandList = cmd_gfx.cast().context("Failed to cast command list")?;
    let fence_value = execute_command_lists_and_signal_device(logical_device, &[Some(cmd_list)])?;
    // Record before wait: if wait fails, the row must still track in-flight GPU use.
    row_guard.commit(fence_value);
    // Blocking wait — required so the shared command_allocator can be safely
    // Reset() before the next render_to_target call (see comment above Reset()).
    wait_for_fence(&logical_device.fence, fence_value)?;

    Ok(())
}
