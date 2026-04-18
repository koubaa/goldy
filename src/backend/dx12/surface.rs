//! Surface (swapchain) management logic.
//!
//! Handles window surface creation, presentation, and resize.

use super::barriers;
use super::render_commands;
use super::types::{FrameSync, LogicalDevice, SurfaceState, TextureState, MAX_FRAMES_IN_FLIGHT};
use super::utils::{depth_format_to_dxgi, dxgi_to_format, format_to_dxgi};
use super::{DeviceHandle, Dx12State, SurfaceHandle, SwapchainImageHandle, TextureHandle};
use crate::backend::RenderCommand;
use crate::types::{Color, DepthFormat, TextureFormat};
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

    // Create swapchain
    // ALLOW_UNORDERED_ACCESS lets compute shaders write directly to back buffers
    // via UAV descriptors (the compute-to-surface path).
    // Not all adapters support this — if creation fails, consider removing this flag
    // and falling back to a separate render target + copy.
    let swap_chain_desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_UNORDERED_ACCESS,
        BufferCount: MAX_FRAMES_IN_FLIGHT as u32,
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
        Flags: 0,
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
        },
    );

    tracing::info!("Created surface {}x{}", width, height);
    Ok(handle)
}

/// Destroy a surface.
pub(super) fn destroy(state: &mut Dx12State, surface_handle: SurfaceHandle) {
    // Clean up any outstanding surface texture
    if let Some(tex_handle) = state
        .surfaces
        .get(&surface_handle)
        .and_then(|s| s.current_texture_handle)
    {
        unregister_surface_texture(state, tex_handle);
    }

    if let Some(surface_state) = state.surfaces.remove(&surface_handle) {
        if let Some(logical_device) = state.devices.get(&surface_state.device_handle) {
            let _ = wait_for_gpu(logical_device);
        }
    }
}

/// Acquire the next swapchain image.
///
/// After acquiring, the back buffer is registered in the bindless descriptor
/// heap as a UAV. The resulting `TextureHandle` is available via
/// `frame_texture()` until `present()` is called.
pub(super) fn acquire(
    state: &mut Dx12State,
    surface_handle: SurfaceHandle,
) -> Result<SwapchainImageHandle> {
    // Clean up any previously acquired surface texture that wasn't presented
    if let Some(tex_handle) = state
        .surfaces
        .get(&surface_handle)
        .and_then(|s| s.current_texture_handle)
    {
        tracing::warn!("Previous surface texture was not presented; cleaning up");
        unregister_surface_texture(state, tex_handle);
    }

    let surface = state
        .surfaces
        .get_mut(&surface_handle)
        .context("Invalid surface handle")?;

    let image_index = unsafe { surface.swapchain.GetCurrentBackBufferIndex() };
    surface.current_image_index = Some(image_index);

    let device_handle = surface.device_handle;
    let resource = surface.render_targets[image_index as usize].clone();
    let width = surface.width;
    let height = surface.height;
    let dxgi_format = surface.format;
    let goldy_format = dxgi_to_format(dxgi_format).unwrap_or(TextureFormat::Bgra8Unorm);

    let tex_handle = register_surface_texture(
        state,
        device_handle,
        resource,
        dxgi_format,
        goldy_format,
        width,
        height,
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

    Ok(())
}

/// Present a rendered surface.
///
/// Unregisters the transient surface texture from the bindless descriptor heap,
/// then presents the swapchain image.
pub(super) fn present(
    state: &mut Dx12State,
    surface_handle: SurfaceHandle,
    _image: SwapchainImageHandle,
) -> Result<()> {
    // Unregister the transient surface texture before presenting
    if let Some(tex_handle) = state
        .surfaces
        .get(&surface_handle)
        .and_then(|s| s.current_texture_handle)
    {
        unregister_surface_texture(state, tex_handle);
    }

    if let Some(surface) = state.surfaces.get_mut(&surface_handle) {
        surface.current_texture_handle = None;
    }

    let surface = state
        .surfaces
        .get(&surface_handle)
        .context("Invalid surface handle")?;

    let device_handle = surface.device_handle;

    // Wait for render to complete before presenting
    {
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;

        let fence_value = logical_device.fence_value.saturating_sub(1);
        wait_for_fence(&logical_device.fence, fence_value)?;
    }

    // Present
    let surface = state.surfaces.get_mut(&surface_handle).unwrap();
    let hr = unsafe { surface.swapchain.Present(1, DXGI_PRESENT(0)) };
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

    // Release old render targets, depth buffer, and resize swapchain
    let depth_format = {
        let surface = state.surfaces.get_mut(&surface_handle).unwrap();
        surface.render_targets.clear();
        surface.rtv_offsets.clear();
        surface.depth_texture = None;
        surface.dsv_offset = None;
        let df = surface.depth_format;

        // Resize swapchain
        unsafe {
            surface.swapchain.ResizeBuffers(
                MAX_FRAMES_IN_FLIGHT as u32,
                width,
                height,
                surface_format,
                DXGI_SWAP_CHAIN_FLAG(0),
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

    tracing::debug!("Resized surface to {}x{}", width, height);
    Ok(())
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

// ---------------------------------------------------------------------------
// Transient surface texture registration
// ---------------------------------------------------------------------------
// These functions register/unregister back-buffer resources in the global
// bindless descriptor heap so that compute shaders can write to them via
// RWTexture2D. The ID3D12Resource is COM-cloned (refcount bump) on acquire
// and released (refcount decrement) on present — the swapchain retains its
// own reference throughout.

/// Register a back-buffer resource as a transient storage texture.
///
/// Creates a UAV descriptor in the CBV/SRV/UAV heap and inserts a `TextureState`.
/// Returns a `TextureHandle` that the caller stores in
/// `SurfaceState::current_texture_handle`.
#[allow(clippy::too_many_arguments)]
fn register_surface_texture(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    resource: ID3D12Resource,
    dxgi_format: DXGI_FORMAT,
    goldy_format: TextureFormat,
    width: u32,
    height: u32,
) -> Result<TextureHandle> {
    let handle = state.next_texture_handle;
    state.next_texture_handle += 1;

    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Device no longer valid")?;

    // Register a UAV descriptor so compute shaders can write to this texture.
    // SRV is also registered so the texture can be read back if needed.
    let srv_offset = logical_device.resource_registry.register_texture(handle);
    let uav_offset = logical_device
        .resource_registry
        .register_texture_uav(handle);

    // Create SRV descriptor
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: dxgi_format,
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
        let mut h = logical_device
            .cbv_srv_uav_heap
            .GetCPUDescriptorHandleForHeapStart();
        h.ptr += (srv_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
        h
    };
    unsafe {
        logical_device
            .device
            .CreateShaderResourceView(&resource, Some(&srv_desc), srv_cpu_handle);
    }

    // Create UAV descriptor for compute write access
    let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: dxgi_format,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV {
                MipSlice: 0,
                PlaneSlice: 0,
            },
        },
    };

    let uav_cpu_handle = unsafe {
        let mut h = logical_device
            .cbv_srv_uav_heap
            .GetCPUDescriptorHandleForHeapStart();
        h.ptr += (uav_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
        h
    };
    unsafe {
        logical_device.device.CreateUnorderedAccessView(
            Some(&resource),
            None,
            Some(&uav_desc),
            uav_cpu_handle,
        );
    }

    state.textures.insert(
        handle,
        TextureState {
            device_handle,
            width,
            height,
            format: goldy_format,
            resource,
            srv_offset,
            bindless_offset: Some(uav_offset),
            last_layout: D3D12_BARRIER_LAYOUT_PRESENT,
        },
    );

    tracing::debug!(
        "Registered surface texture {} ({}x{}, srv={}, uav={})",
        handle,
        width,
        height,
        srv_offset,
        uav_offset,
    );

    Ok(handle)
}

/// Unregister a transient surface texture.
///
/// Removes the `TextureState` and unregisters from the resource registry.
/// The `ID3D12Resource` COM refcount decrements when the `TextureState` is dropped,
/// but the swapchain retains its own reference, so the back buffer stays alive.
fn unregister_surface_texture(state: &mut Dx12State, tex_handle: TextureHandle) {
    if let Some(tex_state) = state.textures.remove(&tex_handle) {
        if let Some(dev) = state.devices.get_mut(&tex_state.device_handle) {
            dev.resource_registry.unregister_texture(tex_handle);
        }
        tracing::debug!("Unregistered surface texture {}", tex_handle);
    }
}
