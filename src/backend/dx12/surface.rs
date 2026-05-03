//! Surface (swapchain) management logic.
//!
//! Handles window surface creation, presentation, and resize.

use super::barriers;
use super::render_commands;
use super::texture;
use super::types::{FrameSync, LogicalDevice, SendSyncHandle, SurfaceState, MAX_FRAMES_IN_FLIGHT};
use super::utils::{depth_format_to_dxgi, dxgi_to_format};
use super::{DeviceHandle, Dx12State, SurfaceHandle, SwapchainImageHandle, TextureHandle};
use crate::backend::RenderCommand;
use crate::types::{Color, DepthFormat, SpatialAccess, TextureFlags, TextureFormat};
use anyhow::{Context, Result};
use raw_window_handle::RawWindowHandle;
use windows::{
    core::Interface,
    Win32::{
        Foundation::{CloseHandle, HWND, RECT},
        Graphics::{
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
        System::Threading::{CreateEventA, WaitForSingleObject, INFINITE},
        UI::WindowsAndMessaging::GetClientRect,
    },
};

/// Create a surface from a window handle.
/// When `depth_format` is `Some`, a depth buffer is created for 3D rendering.
#[allow(clippy::too_many_lines)]
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    window: &dyn raw_window_handle::HasWindowHandle,
    _display: &dyn raw_window_handle::HasDisplayHandle,
    depth_format: Option<DepthFormat>,
) -> Result<SurfaceHandle> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let window_handle = window
        .window_handle()
        .map_err(|e| anyhow::anyhow!("Failed to get window handle: {:?}", e))?;

    let hwnd = match window_handle.as_raw() {
        RawWindowHandle::Win32(h) => HWND(h.hwnd.get() as *mut std::ffi::c_void),
        _ => anyhow::bail!("Expected Win32 window handle"),
    };

    // Get window dimensions
    let mut rect = RECT::default();
    unsafe { GetClientRect(hwnd, &mut rect) }.context("Failed to get window rect")?;

    let width = (rect.right - rect.left) as u32;
    let height = (rect.bottom - rect.top) as u32;

    // Create swapchain (render-target usage only).
    // DXGI/D3D12 rejects `DXGI_USAGE_UNORDERED_ACCESS` on flip-model swapchain buffers;
    // `CreateSwapChainForHwnd` fails (e.g. HRESULT 0x887A698F). Compute-to-surface must
    // use an intermediate UAV texture + copy, not a UAV on the swapchain image.
    let swap_chain_flags = {
        let mut flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        if state.allow_tearing {
            flags = DXGI_SWAP_CHAIN_FLAG(flags.0 | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0);
        }
        flags
    };
    let swap_chain_desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: MAX_FRAMES_IN_FLIGHT as u32,
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
        Flags: swap_chain_flags.0 as u32,
    };

    let swapchain: IDXGISwapChain1 = unsafe {
        state.factory.CreateSwapChainForHwnd(
            &logical_device.command_queue,
            hwnd,
            &swap_chain_desc,
            None,
            None,
        )
    }
    .context("Failed to create swapchain")?;

    let swapchain: IDXGISwapChain3 = swapchain
        .cast()
        .context("Failed to cast swapchain to IDXGISwapChain3")?;

    // Set up the frame-latency waitable object so that acquire() blocks on DXGI
    // readiness rather than stalling the CPU with an explicit fence inside present().
    let frame_latency_waitable: Option<SendSyncHandle> = {
        let sc2: IDXGISwapChain2 = swapchain
            .cast()
            .context("Failed to cast swapchain to IDXGISwapChain2")?;
        unsafe { sc2.SetMaximumFrameLatency(MAX_FRAMES_IN_FLIGHT as u32) }
            .context("Failed to set swapchain maximum frame latency")?;
        let handle = unsafe { sc2.GetFrameLatencyWaitableObject() };
        if handle.0.is_null() {
            tracing::warn!("GetFrameLatencyWaitableObject returned null; waitable disabled");
            None
        } else {
            Some(SendSyncHandle(handle))
        }
    };

    // Get swapchain buffers and create RTVs
    let mut render_targets = Vec::new();
    let mut rtv_offsets = Vec::new();

    for i in 0..MAX_FRAMES_IN_FLIGHT {
        let buffer: ID3D12Resource =
            unsafe { swapchain.GetBuffer(i as u32) }.context("Failed to get swapchain buffer")?;

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
                .CreateRenderTargetView(&buffer, None, rtv_handle);
        }

        render_targets.push(buffer);
        rtv_offsets.push(rtv_offset);
    }

    // Create depth buffer if requested
    let (depth_texture, dsv_offset) = if let Some(df) = depth_format {
        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let depth_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width.max(1) as u64,
            Height: height.max(1),
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
                D3D12_RESOURCE_STATE_COMMON,
                Some(&depth_clear),
                &mut depth_tex,
            )
        }
        .context("Failed to create surface depth buffer")?;
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

    // Create per-frame sync resources
    let mut frame_sync = Vec::new();
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        let command_allocator: ID3D12CommandAllocator = unsafe {
            logical_device
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .context("Failed to create command allocator")?;

        let command_list: ID3D12GraphicsCommandList = unsafe {
            logical_device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &command_allocator,
                None,
            )
        }
        .context("Failed to create command list")?;
        let command_list: ID3D12GraphicsCommandList7 =
            command_list.cast().context("ID3D12GraphicsCommandList7")?;

        unsafe { command_list.Close() }.ok();

        frame_sync.push(FrameSync {
            command_list,
            command_allocator,
            fence_value: 0,
            render_pass_submitted: false,
        });
    }

    let handle = state.next_surface_handle;
    state.next_surface_handle += 1;

    state.surfaces.insert(
        handle,
        SurfaceState {
            device_handle,
            swapchain,
            render_targets,
            rtv_offsets,
            width,
            height,
            format: DXGI_FORMAT_B8G8R8A8_UNORM,
            depth_format,
            depth_texture,
            dsv_offset,
            current_frame: 0,
            current_image_index: None,
            frame_sync,
            current_texture_handle: None,
        compute_scratch_textures: vec![None; MAX_FRAMES_IN_FLIGHT],
        present_mode: crate::types::PresentMode::Fifo,
        frame_latency_waitable,
    },
    );

    tracing::info!("Created surface {}x{}", width, height);
    Ok(handle)
}

/// Destroy a surface.
pub(super) fn destroy(state: &mut Dx12State, surface_handle: SurfaceHandle) {
    let scratch_handles: Vec<TextureHandle> = state
        .surfaces
        .get(&surface_handle)
        .map(|s| {
            s.compute_scratch_textures
                .iter()
                .filter_map(|x| *x)
                .collect()
        })
        .unwrap_or_default();

    if state
        .surfaces
        .get(&surface_handle)
        .and_then(|s| s.current_texture_handle)
        .is_some()
    {
        tracing::warn!(
            "Surface destroyed with an acquired frame; scratch textures freed separately"
        );
    }

    for h in scratch_handles {
        texture::destroy(state, h);
    }

    if let Some(surface_state) = state.surfaces.remove(&surface_handle) {
        if let Some(logical_device) = state.devices.get(&surface_state.device_handle) {
            let _ = wait_for_gpu(logical_device);
        }
        if let Some(SendSyncHandle(waitable)) = surface_state.frame_latency_waitable {
            unsafe { CloseHandle(waitable) }.ok();
        }
    }
}

/// Acquire the next swapchain image.
///
/// Binds a per-buffer **compute scratch** texture (UAV-capable) for `frame.texture()`.
/// Swapchain back buffers cannot be UAVs on D3D12; `present` copies scratch → back buffer
/// when no graphics `surface_render` ran this frame.
pub(super) fn acquire(
    state: &mut Dx12State,
    surface_handle: SurfaceHandle,
) -> Result<SwapchainImageHandle> {
    if state
        .surfaces
        .get(&surface_handle)
        .and_then(|s| s.current_texture_handle)
        .is_some()
    {
        tracing::warn!("Previous surface frame was not presented; continuing acquire");
    }

    // Extract values needed before the pre-acquire blocking waits.
    let (cf, device_handle, waitable, prev_fence) = {
        let surface = state
            .surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        let cf = surface.current_frame;
        (
            cf,
            surface.device_handle,
            surface.frame_latency_waitable,
            surface.frame_sync[cf].fence_value,
        )
    };

    // Fix 2: Block until DXGI signals it is ready to accept a new frame.
    // This caps CPU-ahead frames to MAX_FRAMES_IN_FLIGHT and replaces the
    // ad-hoc fence stall that previously occurred inside present().
    if let Some(SendSyncHandle(waitable_handle)) = waitable {
        unsafe { WaitForSingleObject(waitable_handle, INFINITE) };
    }

    // Fix 1: Ensure this slot's command allocator and scratch texture are no
    // longer in use by the GPU before we reset and reuse them.  With
    // MAX_FRAMES_IN_FLIGHT=2 this is almost always a near-zero stall because
    // roughly two CPU frames have elapsed since the slot was last submitted.
    if prev_fence > 0 {
        {
            let logical_device = state
                .devices
                .get(&device_handle)
                .context("Surface's device is invalid")?;
            wait_for_fence(&logical_device.fence, prev_fence)?;
        }
        if let Some(surf) = state.surfaces.get_mut(&surface_handle) {
            surf.frame_sync[cf].fence_value = 0;
        }
    }

    // Process deferred deletions now that the fence wait has completed.
    if let Some(device) = state.devices.get_mut(&device_handle) {
        device.deletion_queue.process(&device.fence);
    }

    let surface = state
        .surfaces
        .get_mut(&surface_handle)
        .context("Invalid surface handle")?;

    surface.frame_sync[cf].render_pass_submitted = false;

    let image_index = unsafe { surface.swapchain.GetCurrentBackBufferIndex() };
    surface.current_image_index = Some(image_index);

    let width = surface.width;
    let height = surface.height;
    let dxgi_format = surface.format;
    let goldy_format = dxgi_to_format(dxgi_format).unwrap_or(TextureFormat::Bgra8Unorm);
    let idx = image_index as usize;

    let tex_handle = ensure_compute_scratch_texture(
        state,
        surface_handle,
        idx,
        width,
        height,
        goldy_format,
        device_handle,
    )?;

    let surface = state.surfaces.get_mut(&surface_handle).unwrap();
    surface.current_texture_handle = Some(tex_handle);

    Ok(image_index as SwapchainImageHandle)
}

/// Get the texture handle for the currently acquired surface frame.
pub(super) fn frame_texture(
    state: &Dx12State,
    surface_handle: SurfaceHandle,
) -> Option<super::TextureHandle> {
    state
        .surfaces
        .get(&surface_handle)
        .and_then(|s| s.current_texture_handle)
}

/// Render commands to a surface.
#[allow(clippy::too_many_lines)]
pub(super) fn render(
    state: &mut Dx12State,
    surface_handle: SurfaceHandle,
    _image: SwapchainImageHandle,
    commands: &[RenderCommand],
) -> Result<()> {
    let surface = state
        .surfaces
        .get(&surface_handle)
        .context("Invalid surface handle")?;

    let image_index = surface
        .current_image_index
        .context("No image acquired - call surface_acquire first")?;

    let device_handle = surface.device_handle;
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Surface's device is invalid")?;

    let current_frame = surface.current_frame;
    let frame = &surface.frame_sync[current_frame];
    let cmd = &frame.command_list;
    let cmd_gfx: &ID3D12GraphicsCommandList = unsafe { std::mem::transmute(cmd) };
    let width = surface.width;
    let height = surface.height;
    let render_target = &surface.render_targets[image_index as usize];
    let rtv_offset = surface.rtv_offsets[image_index as usize];
    let depth_resource = surface.depth_texture.clone();

    // Reset command allocator and list
    unsafe { frame.command_allocator.Reset() }.context("Failed to reset command allocator")?;
    unsafe { cmd_gfx.Reset(&frame.command_allocator, None) }
        .context("Failed to reset command list")?;

    // PRESENT -> RENDER_TARGET (enhanced barrier, per MS DirectX-Graphics-Samples).
    // SYNC_NONE + NO_ACCESS: no preceding work on this resource in this command list.
    let to_rt = barriers::texture_barrier_full(
        render_target,
        D3D12_BARRIER_SYNC_NONE,
        D3D12_BARRIER_SYNC_RENDER_TARGET,
        D3D12_BARRIER_ACCESS_NO_ACCESS,
        D3D12_BARRIER_ACCESS_RENDER_TARGET,
        D3D12_BARRIER_LAYOUT_PRESENT,
        D3D12_BARRIER_LAYOUT_RENDER_TARGET,
    );
    let mut start_barriers = vec![to_rt];
    if let Some(ref depth_res) = surface.depth_texture {
        start_barriers.push(barriers::texture_barrier_full(
            depth_res,
            D3D12_BARRIER_SYNC_NONE,
            D3D12_BARRIER_SYNC_DEPTH_STENCIL,
            D3D12_BARRIER_ACCESS_NO_ACCESS,
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,
            D3D12_BARRIER_LAYOUT_COMMON,
            D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
        ));
    }
    unsafe { barriers::barrier_textures(cmd, &start_barriers) };
    unsafe { barriers::drop_texture_barriers(&mut start_barriers) };

    // Get RTV handle
    let rtv_handle = unsafe {
        let mut handle = logical_device.rtv_heap.GetCPUDescriptorHandleForHeapStart();
        handle.ptr += (rtv_offset * logical_device.rtv_descriptor_size) as usize;
        handle
    };

    // Find clear color and clear depth
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

    unsafe {
        cmd_gfx.ClearRenderTargetView(
            rtv_handle,
            &[clear_color.r, clear_color.g, clear_color.b, clear_color.a],
            None,
        );
    }

    // Set render target(s) and optionally depth/stencil
    if let (Some(dsv_off), Some(_df)) = (surface.dsv_offset, surface.depth_format) {
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
        cmd_gfx.RSSetViewports(&[viewport]);
        cmd_gfx.RSSetScissorRects(&[scissor]);
    }

    // Bind descriptor heaps for bindless rendering
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    unsafe {
        cmd_gfx.SetDescriptorHeaps(&[
            Some(logical_device.cbv_srv_uav_heap.clone()),
            Some(logical_device.sampler_heap.clone()),
        ]);
    }

    // Execute render commands
    render_commands::record(cmd, commands, device_handle, state)?;

    // RENDER_TARGET -> PRESENT (enhanced barrier, per MS DirectX-Graphics-Samples).
    // SYNC_NONE + NO_ACCESS: no subsequent work on this resource in this command list.
    let mut end_barriers = vec![barriers::texture_barrier_full(
        render_target,
        D3D12_BARRIER_SYNC_RENDER_TARGET,
        D3D12_BARRIER_SYNC_NONE,
        D3D12_BARRIER_ACCESS_RENDER_TARGET,
        D3D12_BARRIER_ACCESS_NO_ACCESS,
        D3D12_BARRIER_LAYOUT_RENDER_TARGET,
        D3D12_BARRIER_LAYOUT_PRESENT,
    )];
    if let Some(ref depth_res) = depth_resource {
        end_barriers.push(barriers::texture_barrier_full(
            depth_res,
            D3D12_BARRIER_SYNC_DEPTH_STENCIL,
            D3D12_BARRIER_SYNC_NONE,
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,
            D3D12_BARRIER_ACCESS_NO_ACCESS,
            D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
            D3D12_BARRIER_LAYOUT_COMMON,
        ));
    }
    unsafe { barriers::barrier_textures(cmd, &end_barriers) };
    unsafe { barriers::drop_texture_barriers(&mut end_barriers) };

    // Close and execute
    unsafe { cmd_gfx.Close() }.context("Failed to close command list")?;

    let cmd_list: ID3D12CommandList = cmd_gfx.cast().context("Failed to cast command list")?;
    unsafe {
        logical_device
            .command_queue
            .ExecuteCommandLists(&[Some(cmd_list)]);
    }

    // Signal fence for this frame
    let fence_value = logical_device.fence_value;
    unsafe {
        logical_device
            .command_queue
            .Signal(&logical_device.fence, fence_value)
    }
    .context("Failed to signal fence")?;

    // Update fence value for next operation
    if let Some(dev) = state.devices.get_mut(&device_handle) {
        dev.fence_value += 1;
    }

    if let Some(surf) = state.surfaces.get_mut(&surface_handle) {
        surf.frame_sync[current_frame].fence_value = fence_value;
        surf.frame_sync[current_frame].render_pass_submitted = true;
    }

    Ok(())
}

/// Present a rendered surface.
///
/// Graphics path: waits for the last graphics submit then presents.
/// Compute-only path: copies the scratch UAV texture to the swapchain back buffer, then presents.
pub(super) fn present(
    state: &mut Dx12State,
    surface_handle: SurfaceHandle,
    _image: SwapchainImageHandle,
) -> Result<()> {
    let (
        device_handle,
        _image_index,
        current_frame,
        scratch_handle,
        backbuffer,
        render_pass_submitted,
    ) = {
        let surface = state
            .surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        let image_index = surface
            .current_image_index
            .context("No image to present - call surface_acquire first")?
            as usize;
        let scratch_handle = surface
            .current_texture_handle
            .context("No acquired surface texture")?;
        let expected = surface
            .compute_scratch_textures
            .get(image_index)
            .copied()
            .flatten();
        if expected != Some(scratch_handle) {
            anyhow::bail!("DX12 surface present: scratch texture handle mismatch");
        }
        let render_pass_submitted = surface.frame_sync[surface.current_frame].render_pass_submitted;
        (
            surface.device_handle,
            image_index,
            surface.current_frame,
            scratch_handle,
            surface.render_targets[image_index].clone(),
            render_pass_submitted,
        )
    };

    if let Some(s) = state.surfaces.get_mut(&surface_handle) {
        s.current_texture_handle = None;
    }

    if render_pass_submitted {
        // fence_value was stored in frame_sync[current_frame].fence_value by
        // surface::render; acquire() waits for it on the next cycle through this slot.
    } else {
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;
        let scratch_res = state
            .textures
            .get(&scratch_handle)
            .context("Scratch texture not found")?
            .resource
            .clone();

        let (cmd7, cmd_alloc) = {
            let surface = state.surfaces.get(&surface_handle).unwrap();
            let frame = &surface.frame_sync[current_frame];
            (frame.command_list.clone(), frame.command_allocator.clone())
        };

        unsafe { cmd_alloc.Reset() }
            .context("Failed to reset command allocator for present copy")?;
        let cmd_gfx: &ID3D12GraphicsCommandList = unsafe { std::mem::transmute(&cmd7) };
        unsafe { cmd_gfx.Reset(&cmd_alloc, None) }
            .context("Failed to reset command list for present copy")?;

        // Flip-discard swapchain: the acquired buffer's contents are undefined until written.
        // `D3D12_BARRIER_LAYOUT_PRESENT` and `COMMON` share the same numeric value in windows-rs
        // (both 0), so treat the destination as UNDEFINED before COPY_DEST.
        unsafe { cmd_gfx.DiscardResource(&backbuffer, None) };

        let mut prep_barriers = vec![
            barriers::texture_barrier_full(
                &scratch_res,
                D3D12_BARRIER_SYNC_ALL,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS,
                D3D12_BARRIER_LAYOUT_COPY_SOURCE,
            ),
            barriers::texture_barrier_full(
                &backbuffer,
                D3D12_BARRIER_SYNC_NONE,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_NO_ACCESS,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_LAYOUT_UNDEFINED,
                D3D12_BARRIER_LAYOUT_COPY_DEST,
            ),
        ];
        unsafe { barriers::barrier_textures(&cmd7, &prep_barriers) };
        unsafe { barriers::drop_texture_barriers(&mut prep_barriers) };

        unsafe { cmd_gfx.CopyResource(&backbuffer, &scratch_res) };

        let mut post_barriers = vec![
            barriers::texture_barrier_full(
                &backbuffer,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_SYNC_NONE,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_ACCESS_NO_ACCESS,
                D3D12_BARRIER_LAYOUT_COPY_DEST,
                D3D12_BARRIER_LAYOUT_PRESENT,
            ),
            barriers::texture_barrier_full(
                &scratch_res,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_LAYOUT_COPY_SOURCE,
                D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS,
            ),
        ];
        unsafe { barriers::barrier_textures(&cmd7, &post_barriers) };
        unsafe { barriers::drop_texture_barriers(&mut post_barriers) };

        unsafe { cmd_gfx.Close() }.context("Failed to close present copy command list")?;

        let cmd_list: ID3D12CommandList = cmd_gfx.cast().context("Failed to cast command list")?;
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
        .context("Failed to signal fence after present copy")?;

        if let Some(dev) = state.devices.get_mut(&device_handle) {
            dev.fence_value += 1;
        }

        // Record the fence so acquire() can wait for it before reusing this slot.
        if let Some(surf) = state.surfaces.get_mut(&surface_handle) {
            surf.frame_sync[current_frame].fence_value = fence_value;
        }

        if let Some(tex) = state.textures.get_mut(&scratch_handle) {
            tex.last_layout = D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS;
        }
    }

    // Present
    let surface = state.surfaces.get_mut(&surface_handle).unwrap();
    let (sync_interval, present_flags) = present_args(surface.present_mode, state.allow_tearing);
    let hr = unsafe { surface.swapchain.Present(sync_interval, present_flags) };
    if hr.is_err() {
        anyhow::bail!("Present failed with HRESULT: {:?}", hr);
    }

    // Advance frame
    surface.current_image_index = None;
    surface.current_frame = (surface.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

    Ok(())
}

/// Resize a surface.
#[allow(clippy::too_many_lines)]
pub(super) fn resize(
    state: &mut Dx12State,
    surface_handle: SurfaceHandle,
    width: u32,
    height: u32,
) -> Result<()> {
    // Get device handle and surface format first
    let (device_handle, surface_format) = {
        let surface = state
            .surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        (surface.device_handle, surface.format)
    };

    // Wait for GPU
    {
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;
        let _ = wait_for_gpu(logical_device);
    }

    let scratch_destroy: Vec<TextureHandle> = {
        let surface = state.surfaces.get_mut(&surface_handle).unwrap();
        surface
            .compute_scratch_textures
            .iter_mut()
            .filter_map(|slot| slot.take())
            .collect()
    };
    for h in scratch_destroy {
        texture::destroy(state, h);
    }

    // Release old render targets, depth buffer, and resize swapchain
    let depth_format = {
        let surface = state.surfaces.get_mut(&surface_handle).unwrap();
        surface.render_targets.clear();
        surface.rtv_offsets.clear();
        surface.depth_texture = None;
        surface.dsv_offset = None;
        let df = surface.depth_format;

        let resize_flags = {
            let mut flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
            if state.allow_tearing {
                flags = DXGI_SWAP_CHAIN_FLAG(flags.0 | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0);
            }
            flags
        };
        // Resize swapchain
        unsafe {
            surface.swapchain.ResizeBuffers(
                MAX_FRAMES_IN_FLIGHT as u32,
                width,
                height,
                surface_format,
                resize_flags,
            )
        }
        .context("Failed to resize swapchain")?;

        surface.width = width;
        surface.height = height;
        df
    };

    // Get device info for creating RTVs
    let (rtv_heap, rtv_descriptor_size, device) = {
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;
        (
            logical_device.rtv_heap.clone(),
            logical_device.rtv_descriptor_size,
            logical_device.device.clone(),
        )
    };

    // Recreate render targets
    for i in 0..MAX_FRAMES_IN_FLIGHT {
        let surface = state.surfaces.get(&surface_handle).unwrap();
        let buffer: ID3D12Resource = unsafe { surface.swapchain.GetBuffer(i as u32) }
            .context("Failed to get swapchain buffer")?;

        let rtv_offset = state.next_rtv_offset;
        state.next_rtv_offset += 1;

        let rtv_handle = unsafe {
            let mut handle = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (rtv_offset * rtv_descriptor_size) as usize;
            handle
        };

        unsafe {
            device.CreateRenderTargetView(&buffer, None, rtv_handle);
        }

        let surface = state.surfaces.get_mut(&surface_handle).unwrap();
        surface.render_targets.push(buffer);
        surface.rtv_offsets.push(rtv_offset);
    }

    // Recreate depth buffer if the surface had one
    if let Some(df) = depth_format {
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Surface's device is invalid")?;

        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let w = width.max(1);
        let h = height.max(1);
        let depth_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: w as u64,
            Height: h,
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
                D3D12_RESOURCE_STATE_COMMON,
                Some(&depth_clear),
                &mut depth_tex,
            )
        }
        .context("Failed to create surface depth buffer on resize")?;
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

        let surface = state.surfaces.get_mut(&surface_handle).unwrap();
        surface.depth_texture = Some(depth_tex);
        surface.dsv_offset = Some(dsv_off);
    }

    let surface = state.surfaces.get_mut(&surface_handle).unwrap();
    surface.current_frame = 0;
    surface.current_image_index = None;
    surface.current_texture_handle = None;
    surface.compute_scratch_textures = vec![None; MAX_FRAMES_IN_FLIGHT];

    tracing::debug!("Resized surface to {}x{}", width, height);
    Ok(())
}

/// Ensure a UAV scratch texture exists for swapchain buffer index `idx`.
fn ensure_compute_scratch_texture(
    state: &mut Dx12State,
    surface_handle: SurfaceHandle,
    idx: usize,
    width: u32,
    height: u32,
    format: TextureFormat,
    device_handle: DeviceHandle,
) -> Result<TextureHandle> {
    let reuse = {
        let surface = state
            .surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;
        let slot = surface.compute_scratch_textures.get(idx).copied().flatten();
        if let Some(h) = slot {
            if let Some(tex) = state.textures.get(&h) {
                if tex.width == width && tex.height == height && tex.format == format {
                    return Ok(h);
                }
            }
            Some(h)
        } else {
            None
        }
    };

    if let Some(old) = reuse {
        texture::destroy(state, old);
        let surface = state.surfaces.get_mut(&surface_handle).unwrap();
        surface.compute_scratch_textures[idx] = None;
    }

    let h = texture::create(
        state,
        device_handle,
        width,
        height,
        format,
        SpatialAccess::Direct,
        TextureFlags::empty(),
    )?;
    let surface = state.surfaces.get_mut(&surface_handle).unwrap();
    surface.compute_scratch_textures[idx] = Some(h);
    Ok(h)
}

/// Get surface dimensions.
pub(super) fn size(state: &Dx12State, surface_handle: SurfaceHandle) -> (u32, u32) {
    state
        .surfaces
        .get(&surface_handle)
        .map(|s| (s.width, s.height))
        .unwrap_or((0, 0))
}

/// Get surface format.
pub(super) fn format(state: &Dx12State, surface_handle: SurfaceHandle) -> TextureFormat {
    state
        .surfaces
        .get(&surface_handle)
        .and_then(|s| dxgi_to_format(s.format))
        .unwrap_or(TextureFormat::Bgra8Unorm)
}

// Helper functions

fn wait_for_fence(fence: &ID3D12Fence, value: u64) -> Result<()> {
    let event =
        unsafe { CreateEventA(None, false, false, None) }.context("Failed to create event")?;
    unsafe { fence.SetEventOnCompletion(value, event) }
        .context("Failed to set event on completion")?;
    unsafe { WaitForSingleObject(event, INFINITE) };
    unsafe { CloseHandle(event) }.ok();
    Ok(())
}

fn wait_for_gpu(device: &LogicalDevice) -> Result<()> {
    let fence_value = device.fence_value;
    unsafe { device.command_queue.Signal(&device.fence, fence_value) }
        .context("Failed to signal fence")?;
    wait_for_fence(&device.fence, fence_value)
}

/// Map a `PresentMode` to DXGI `Present()` arguments (SyncInterval, Flags).
fn present_args(mode: crate::types::PresentMode, allow_tearing: bool) -> (u32, DXGI_PRESENT) {
    use crate::types::PresentMode;
    match mode {
        PresentMode::Fifo | PresentMode::Mailbox | PresentMode::Auto => (1, DXGI_PRESENT(0)),
        PresentMode::Immediate => {
            if allow_tearing {
                (0, DXGI_PRESENT_ALLOW_TEARING)
            } else {
                (0, DXGI_PRESENT(0))
            }
        }
    }
}

/// Set the present mode on a surface (takes effect on the next Present call).
pub(super) fn set_present_mode(
    state: &mut Dx12State,
    surface_handle: SurfaceHandle,
    mode: crate::types::PresentMode,
) -> Result<()> {
    let surface = state
        .surfaces
        .get_mut(&surface_handle)
        .context("Invalid surface handle")?;
    surface.present_mode = mode;
    tracing::debug!(?mode, "DX12 present mode set");
    Ok(())
}

/// Get the current present mode of a surface.
pub(super) fn get_present_mode(
    state: &Dx12State,
    surface_handle: SurfaceHandle,
) -> crate::types::PresentMode {
    state
        .surfaces
        .get(&surface_handle)
        .map(|s| s.present_mode)
        .unwrap_or_default()
}
