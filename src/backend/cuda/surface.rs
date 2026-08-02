//! CUDA + DX12 surface / swapchain / present.
//!
//! Swapchain backbuffers stay BGRA8 (non-shareable). Per-image float4 shared scratch
//! textures are CUDA-writable; present blits scratch → BGRA UAV → backbuffer, waiting
//! on the shared D3D12 fence that CUDA signals at submit completion.

use super::dx12_companion::{cuda_signal_fence, cuda_wait_fence, Dx12Companion, MAX_FRAMES};
use super::dx12_interop::{
    record_present_copy, PresentBlitPipeline, SharedScratchTexture, SURFACE_COMPUTE_FORMAT,
    SWAPCHAIN_DXGI_FORMAT,
};
use super::{CudaBackend, CudaDevice};
use crate::backend::{
    ContextHandle, DeviceHandle, FrameToken, PresentFinishState, PresentGpuWork, SurfaceHandle,
    SwapchainImageHandle, TextureHandle,
};
use anyhow::{bail, Context as _, Result};
use raw_window_handle::RawWindowHandle;
use std::sync::Arc;
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HWND, RECT, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{WaitForSingleObject, INFINITE};
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;

/// `HANDLE` is a raw pointer; DXGI waitables are process-local and only touched
/// from Goldy's backend lock / present path.
#[derive(Clone, Copy)]
struct SendSyncHandle(HANDLE);
// SAFETY: see comment on type.
unsafe impl Send for SendSyncHandle {}
unsafe impl Sync for SendSyncHandle {}

pub(super) struct CudaSurfaceState {
    pub device: DeviceHandle,
    pub swapchain: IDXGISwapChain3,
    pub backbuffers: Vec<ID3D12Resource>,
    pub rtv_heap: ID3D12DescriptorHeap,
    pub width: u32,
    pub height: u32,
    pub present_mode: crate::types::PresentMode,
    pub frame_latency_waitable: Option<SendSyncHandle>,
    /// Rotating present-slot index (0..MAX_FRAMES).
    pub current_frame: usize,
    pub current_image_index: Option<u32>,
    pub current_texture_handle: Option<TextureHandle>,
    /// Per-swapchain-image shared scratch (indexed by backbuffer index).
    pub scratch: Vec<Option<ScratchSlot>>,
    /// Fence value that guards each present-slot's command allocator reuse.
    pub slot_fence: [u64; MAX_FRAMES],
    /// After create/resize, DXGI backbuffers are in COMMON until the first present copy.
    pub backbuffer_in_common: [bool; MAX_FRAMES],
    pub blit: PresentBlitPipeline,
}

pub(super) struct ScratchSlot {
    pub shared: SharedScratchTexture,
    pub texture_handle: TextureHandle,
    /// Last CUDA→DX12 handoff fence value that wrote this scratch (0 = never).
    pub cuda_complete: u64,
    /// Fence value after DX12 finished present-copy (CUDA may wait before reuse).
    pub dx12_complete: u64,
}

pub(super) fn create_surface(
    backend: &mut CudaBackend,
    device: DeviceHandle,
    window: &dyn raw_window_handle::HasWindowHandle,
    _display: &dyn raw_window_handle::HasDisplayHandle,
    depth_format: Option<crate::types::DepthFormat>,
) -> Result<SurfaceHandle> {
    if depth_format.is_some() {
        bail!("CUDA/DX12: depth buffers on surfaces are not supported in the first presentation slice");
    }
    let companion = companion_ref(backend, device)?;

    let window_handle = window
        .window_handle()
        .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e:?}"))?;
    let hwnd = match window_handle.as_raw() {
        RawWindowHandle::Win32(h) => HWND(h.hwnd.get() as *mut std::ffi::c_void),
        _ => bail!("CUDA/DX12: expected Win32 window handle"),
    };

    let mut rect = RECT::default();
    unsafe { GetClientRect(hwnd, &mut rect) }.context("GetClientRect failed")?;
    let width = (rect.right - rect.left).max(1) as u32;
    let height = (rect.bottom - rect.top).max(1) as u32;

    let allow_tearing = companion.allow_tearing;
    let mut swap_flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    if allow_tearing {
        swap_flags = DXGI_SWAP_CHAIN_FLAG(swap_flags.0 | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0);
    }
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: SWAPCHAIN_DXGI_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: MAX_FRAMES as u32,
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
        Flags: swap_flags.0 as u32,
    };

    let swapchain1: IDXGISwapChain1 = unsafe {
        companion
            .factory
            .CreateSwapChainForHwnd(&companion.queue, hwnd, &desc, None, None)
    }
    .context("CUDA/DX12: CreateSwapChainForHwnd failed")?;
    let swapchain: IDXGISwapChain3 = swapchain1
        .cast()
        .context("CUDA/DX12: cast to IDXGISwapChain3")?;

    let frame_latency_waitable = {
        unsafe { swapchain.SetMaximumFrameLatency(MAX_FRAMES as u32) }.ok();
        let h = unsafe { swapchain.GetFrameLatencyWaitableObject() };
        if h.is_invalid() {
            None
        } else {
            Some(SendSyncHandle(h))
        }
    };

    let rtv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: MAX_FRAMES as u32,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    let rtv_heap: ID3D12DescriptorHeap = unsafe { companion.device.CreateDescriptorHeap(&rtv_heap_desc) }
        .context("CUDA/DX12: CreateDescriptorHeap(RTV)")?;
    let rtv_size =
        unsafe { companion.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) };
    let mut backbuffers = Vec::with_capacity(MAX_FRAMES);
    for i in 0..MAX_FRAMES {
        let buf: ID3D12Resource = unsafe { swapchain.GetBuffer(i as u32) }
            .context("CUDA/DX12: GetBuffer failed")?;
        let handle = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: handle.ptr + i * rtv_size as usize,
        };
        unsafe { companion.device.CreateRenderTargetView(&buf, None, cpu) };
        backbuffers.push(buf);
    }

    let blit = PresentBlitPipeline::create(&companion.device)?;

    let handle = backend.next_surface;
    backend.next_surface += 1;
    backend.surfaces.insert(
        handle,
        CudaSurfaceState {
            device,
            swapchain,
            backbuffers,
            rtv_heap,
            width,
            height,
            present_mode: crate::types::PresentMode::Fifo,
            frame_latency_waitable,
            current_frame: 0,
            current_image_index: None,
            current_texture_handle: None,
            scratch: (0..MAX_FRAMES).map(|_| None).collect(),
            slot_fence: [0; MAX_FRAMES],
            backbuffer_in_common: [true; MAX_FRAMES],
            blit,
        },
    );
    tracing::info!("CUDA/DX12: created surface {width}x{height} (compute format {SURFACE_COMPUTE_FORMAT:?})");
    Ok(handle)
}

pub(super) fn destroy_surface(backend: &mut CudaBackend, surface: SurfaceHandle) {
    let Some(mut state) = backend.surfaces.remove(&surface) else {
        return;
    };
    if let Some(gpu) = backend.devices.get(&state.device) {
        if let Some(companion) = gpu.dx12.as_ref() {
            let _ = companion.wait_idle();
        }
    }
    // Drop CUDA tex/surf objects before imported external memory (strict interop order).
    let tex_handles: Vec<_> = state
        .scratch
        .iter()
        .filter_map(|s| s.as_ref().map(|x| x.texture_handle))
        .collect();
    for h in &tex_handles {
        if let Some(resource) = backend.textures.remove(h) {
            if let Some(slot) = resource.storage_slot {
                backend.texture_slots.remove(&slot);
            }
            if let Some(slot) = resource.sampled_slot {
                backend.texture_slots.remove(&slot);
            }
            drop(resource);
        }
    }
    state.scratch.clear();
    if let Some(SendSyncHandle(waitable)) = state.frame_latency_waitable.take() {
        unsafe {
            let _ = CloseHandle(waitable);
        }
    }
}

pub(super) fn surface_size(backend: &CudaBackend, surface: SurfaceHandle) -> (u32, u32) {
    backend
        .surfaces
        .get(&surface)
        .map(|s| (s.width, s.height))
        .unwrap_or((0, 0))
}

pub(super) fn surface_format(_backend: &CudaBackend, _surface: SurfaceHandle) -> crate::types::TextureFormat {
    SURFACE_COMPUTE_FORMAT
}

pub(super) fn surface_set_present_mode(
    backend: &mut CudaBackend,
    surface: SurfaceHandle,
    mode: crate::types::PresentMode,
) -> Result<()> {
    let state = backend
        .surfaces
        .get_mut(&surface)
        .context("CUDA/DX12: invalid surface")?;
    state.present_mode = mode;
    Ok(())
}

pub(super) fn surface_resize(
    backend: &mut CudaBackend,
    surface: SurfaceHandle,
    width: u32,
    height: u32,
) -> Result<()> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    let device = backend
        .surfaces
        .get(&surface)
        .context("CUDA/DX12: invalid surface")?
        .device;

    // Take scratch slots, wait idle, then drop CUDA views before external memory.
    let old_slots: Vec<ScratchSlot> = {
        let state = backend.surfaces.get_mut(&surface).unwrap();
        state.scratch.iter_mut().filter_map(|s| s.take()).collect()
    };
    {
        let companion = companion_ref(backend, device)?;
        companion.wait_idle()?;
    }
    for slot in old_slots {
        let h = slot.texture_handle;
        if let Some(resource) = backend.textures.remove(&h) {
            if let Some(sid) = resource.storage_slot {
                backend.texture_slots.remove(&sid);
            }
            drop(resource);
        }
        drop(slot);
    }
    // Retained CUDA graphs may embed destroyed surf objects from old scratch.
    let stale_keys: Vec<(ContextHandle, u64)> = backend.retained.keys().copied().collect();
    for (ctx, key) in stale_keys {
        if backend.retained.remove(&(ctx, key)).is_some() {
            backend.enqueue_evict_retained(ctx, key);
        }
    }

    let state = backend.surfaces.get_mut(&surface).unwrap();
    state.backbuffers.clear();
    let allow_tearing = companion_ref(backend, device)?.allow_tearing;
    let mut resize_flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    if allow_tearing {
        resize_flags = DXGI_SWAP_CHAIN_FLAG(resize_flags.0 | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0);
    }
    let swapchain = backend.surfaces.get(&surface).unwrap().swapchain.clone();
    unsafe {
        swapchain.ResizeBuffers(
            MAX_FRAMES as u32,
            width,
            height,
            SWAPCHAIN_DXGI_FORMAT,
            resize_flags,
        )
    }
    .context("CUDA/DX12: ResizeBuffers failed")?;

    let (device_com, rtv_size) = {
        let companion = companion_ref(backend, device)?;
        let rtv_size = unsafe {
            companion
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
        };
        (companion.device.clone(), rtv_size)
    };
    let state = backend.surfaces.get_mut(&surface).unwrap();
    for i in 0..MAX_FRAMES {
        let buf: ID3D12Resource = unsafe { state.swapchain.GetBuffer(i as u32) }
            .context("CUDA/DX12: GetBuffer after resize")?;
        let base = unsafe { state.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + i * rtv_size as usize,
        };
        unsafe { device_com.CreateRenderTargetView(&buf, None, cpu) };
        state.backbuffers.push(buf);
    }
    state.width = width;
    state.height = height;
    state.current_texture_handle = None;
    state.current_image_index = None;
    state.slot_fence = [0; MAX_FRAMES];
    state.backbuffer_in_common = [true; MAX_FRAMES];
    Ok(())
}

pub(super) fn begin_frame(
    backend: &mut CudaBackend,
    surface: SurfaceHandle,
    ctx: ContextHandle,
) -> Result<(FrameToken, TextureHandle)> {
    let device = backend
        .surfaces
        .get(&surface)
        .context("CUDA/DX12: invalid surface")?
        .device;

    let (present_slot, waitable, prev_fence) = {
        let state = backend.surfaces.get(&surface).unwrap();
        let present_slot = state.current_frame;
        (
            present_slot,
            state.frame_latency_waitable,
            state.slot_fence[present_slot],
        )
    };

    if let Some(SendSyncHandle(waitable)) = waitable {
        unsafe { WaitForSingleObject(waitable, INFINITE) };
    }
    if prev_fence > 0 {
        companion_ref(backend, device)?.cpu_wait(prev_fence)?;
        backend.surfaces.get_mut(&surface).unwrap().slot_fence[present_slot] = 0;
    }

    let image_index = {
        let state = backend.surfaces.get_mut(&surface).unwrap();
        let idx = unsafe { state.swapchain.GetCurrentBackBufferIndex() };
        state.current_image_index = Some(idx);
        idx as usize
    };

    let tex_handle = ensure_scratch(backend, surface, image_index)?;

    // CUDA must wait until DX12 finished the previous present-copy on this scratch.
    if let Some(slot) = backend
        .surfaces
        .get(&surface)
        .and_then(|s| s.scratch.get(image_index))
        .and_then(|s| s.as_ref())
    {
        if slot.dx12_complete > 0 {
            let stream = Arc::clone(&backend.context(ctx)?.stream);
            let companion = companion_ref(backend, device)?;
            cuda_wait_fence(
                &companion.cuda_ctx,
                companion.cuda_semaphore,
                stream.cu_stream(),
                slot.dx12_complete,
            )?;
        }
    }

    {
        let state = backend.surfaces.get_mut(&surface).unwrap();
        state.current_texture_handle = Some(tex_handle);
        state.current_frame = (present_slot + 1) % MAX_FRAMES;
    }

    if let Some(context) = backend.contexts.get(&ctx) {
        context.signal_queue.push(crate::signal::Signal::SwapchainAcquired {
            image_index: image_index as u32,
        });
    }

    Ok((
        FrameToken {
            surface,
            image: image_index as SwapchainImageHandle,
            context: ctx,
            frame_slot: image_index as u32,
            present_slot: present_slot as u32,
        },
        tex_handle,
    ))
}

pub(super) fn submit_frame(
    backend: &mut CudaBackend,
    frame: &FrameToken,
) -> Result<crate::timeline::TimelineValue> {
    // Scheme present submits compute via the normal submit path before present.
    // report the context's last submitted timeline value (0 if none yet).
    let context = backend.context(frame.context)?;
    let completed = context.completed.load(std::sync::atomic::Ordering::Acquire);
    // Prefer the latest ledger entry for this context if higher.
    let ledger_max = {
        let guard = context.event_ledger.lock().unwrap();
        guard
            .iter()
            .filter(|(_, e)| e.context == frame.context)
            .map(|(tv, _)| *tv)
            .max()
            .unwrap_or(0)
    };
    Ok(completed.max(ledger_max))
}

pub(super) fn take_present_gpu_work(
    backend: &mut CudaBackend,
    frame: FrameToken,
    submit_tv: crate::timeline::TimelineValue,
) -> Result<Box<dyn PresentGpuWork>> {
    let surface_handle = frame.surface;
    let image_index = frame.image as usize;
    let present_slot = frame.present_slot as usize;

    if let Some(s) = backend.surfaces.get_mut(&surface_handle) {
        s.current_texture_handle = None;
    }

    let device = backend
        .surfaces
        .get(&surface_handle)
        .context("invalid surface")?
        .device;
    let companion = Arc::clone(
        backend
            .devices
            .get(&device)
            .context("invalid device")?
            .dx12
            .as_ref()
            .context("CUDA device missing DX12 companion")?,
    );

    // Signal the shared fence from CUDA once submit_tv's completion event is done.
    let cuda_complete = if submit_tv > 0 {
        let context = backend.context(frame.context)?;
        let event = {
            let guard = context.event_ledger.lock().unwrap();
            guard.get(&submit_tv).map(|e| Arc::clone(&e.event))
        };
        let stream = Arc::clone(&context.stream);
        let signal_value = companion.next_fence_value();
        if let Some(event) = event {
            stream
                .wait(&event)
                .context("CUDA/DX12: stream wait on submit completion")?;
        }
        cuda_signal_fence(
            &companion.cuda_ctx,
            companion.cuda_semaphore,
            stream.cu_stream(),
            signal_value,
        )?;
        signal_value
    } else {
        0
    };

    let state = backend
        .surfaces
        .get_mut(&surface_handle)
        .context("invalid surface")?;
    if let Some(slot) = state.scratch.get_mut(image_index).and_then(|s| s.as_mut()) {
        slot.cuda_complete = cuda_complete;
    }

    let scratch = state
        .scratch
        .get(image_index)
        .and_then(|s| s.as_ref())
        .context("no scratch for present")?;
    let scratch_handle = scratch.texture_handle;
    let scratch_res = scratch.shared.d3d12_resource.clone();
    let blit_target = scratch.shared.blit_target.clone();
    let width = scratch.shared.width;
    let height = scratch.shared.height;
    let backbuffer = state.backbuffers[image_index].clone();
    let backbuffer_from_common = state.backbuffer_in_common[image_index];
    let swapchain = state.swapchain.clone();
    let present_mode = state.present_mode;
    let allow_tearing = companion.allow_tearing;
    let existing_fence = state.slot_fence[present_slot];

    // Ensure blit descriptors are written for this slot.
    companion
        .as_ref(); // keep alive
    let blit = &state.blit;
    blit.write_descriptors(
        &companion.device,
        image_index,
        &scratch_res,
        &blit_target,
    );

    // Clone command slot resources.
    let allocator = companion.present_slots[present_slot].allocator.clone();
    let list = companion.present_slots[present_slot].list.clone();
    // Rebuild PresentBlitPipeline handles for the work item (COM clone).
    let blit_pipe = PresentBlitPipeline {
        root_signature: state.blit.root_signature.clone(),
        pso: state.blit.pso.clone(),
        srv_uav_heap: state.blit.srv_uav_heap.clone(),
        descriptor_size: state.blit.descriptor_size,
    };

    Ok(Box::new(CudaDx12PresentGpuWork {
        frame,
        image_index,
        present_slot,
        scratch_handle,
        cuda_complete,
        companion,
        scratch_res,
        blit_target,
        backbuffer,
        backbuffer_from_common,
        allocator,
        list,
        blit: blit_pipe,
        swapchain,
        present_mode,
        allow_tearing,
        existing_fence,
        width,
        height,
    }))
}

pub(super) fn finish_present(
    backend: &mut CudaBackend,
    finish: PresentFinishState,
    _submit_tv: crate::timeline::TimelineValue,
) -> Result<crate::timeline::TimelineValue> {
    let surface = finish.frame.surface;
    let present_slot = finish.frame.present_slot as usize;
    let image_index = finish.frame.image as usize;
    let ctx = finish.frame.context;

    if let Some(state) = backend.surfaces.get_mut(&surface) {
        if finish.return_fence > 0 {
            state.slot_fence[present_slot] = finish.return_fence;
            if let Some(slot) = state.scratch.get_mut(image_index).and_then(|s| s.as_mut()) {
                slot.dx12_complete = finish.return_fence;
            }
        }
        if finish.present_ok {
            state.backbuffer_in_common[image_index] = false;
        }
        if let Some(sc) = backend.contexts.get(&ctx) {
            sc.signal_queue
                .push(crate::signal::Signal::SwapchainReturned { image_index: image_index as u32 });
        }
    }
    Ok(finish.present_timeline)
}

struct CudaDx12PresentGpuWork {
    frame: FrameToken,
    image_index: usize,
    #[allow(dead_code)]
    present_slot: usize,
    scratch_handle: TextureHandle,
    cuda_complete: u64,
    companion: Arc<Dx12Companion>,
    scratch_res: ID3D12Resource,
    blit_target: ID3D12Resource,
    backbuffer: ID3D12Resource,
    backbuffer_from_common: bool,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    blit: PresentBlitPipeline,
    swapchain: IDXGISwapChain3,
    present_mode: crate::types::PresentMode,
    allow_tearing: bool,
    existing_fence: u64,
    width: u32,
    height: u32,
}

impl PresentGpuWork for CudaDx12PresentGpuWork {
    fn run(self: Box<Self>) -> Result<PresentFinishState> {
        if self.cuda_complete > 0 {
            self.companion.wait_queue(self.cuda_complete)?;
        }

        if self.existing_fence > 0 {
            self.companion.cpu_wait(self.existing_fence)?;
        }

        unsafe { self.allocator.Reset() }.context("reset present allocator")?;
        unsafe { self.list.Reset(&self.allocator, None) }.context("reset present list")?;

        record_present_copy(
            &self.list,
            &self.blit,
            self.image_index,
            &self.scratch_res,
            &self.blit_target,
            &self.backbuffer,
            self.backbuffer_from_common,
            self.width,
            self.height,
        )?;
        unsafe { self.list.Close() }.context("close present list")?;

        let cmd: ID3D12CommandList = self.list.cast().context("cast present list")?;
        let return_fence = self.companion.next_fence_value();
        self.companion
            .execute_and_signal(&[Some(cmd)], return_fence)?;

        let (sync_interval, flags) = present_args(self.present_mode, self.allow_tearing);
        let hr = unsafe { self.swapchain.Present(sync_interval, flags) };
        if hr.is_err() {
            bail!("CUDA/DX12: Present failed: {hr:?}");
        }

        Ok(PresentFinishState {
            frame: self.frame,
            return_fence,
            scratch_texture: Some(self.scratch_handle),
            scratch_layout_updated: true,
            present_timeline: return_fence,
            copy_timeline: Some(return_fence),
            frame_compute_timeline: None,
            signal_timeline: None,
            render_pass_submitted: false,
            present_ok: true,
        })
    }
}

fn ensure_scratch(
    backend: &mut CudaBackend,
    surface: SurfaceHandle,
    image_index: usize,
) -> Result<TextureHandle> {
    let (device, width, height, reuse) = {
        let state = backend.surfaces.get(&surface).context("invalid surface")?;
        let reuse = state
            .scratch
            .get(image_index)
            .and_then(|s| s.as_ref())
            .map(|s| {
                (
                    s.texture_handle,
                    s.shared.width == state.width && s.shared.height == state.height,
                )
            });
        (state.device, state.width, state.height, reuse)
    };

    if let Some((handle, true)) = reuse {
        return Ok(handle);
    }
    if let Some((old, false)) = reuse {
        if let Some(slot) = backend
            .surfaces
            .get_mut(&surface)
            .unwrap()
            .scratch
            .get_mut(image_index)
            .and_then(|s| s.take())
        {
            // Drop CUDA views before external memory.
            if let Some(resource) = backend.textures.remove(&old) {
                if let Some(slot_id) = resource.storage_slot {
                    backend.texture_slots.remove(&slot_id);
                }
                drop(resource);
            }
            drop(slot);
        }
    }

    let storage_slot = backend.alloc_registry_slot();
    let companion = companion_ref(backend, device)?;
    let cuda_ctx = Arc::clone(&backend.device(device)?.ctx);
    let shared = SharedScratchTexture::create(companion, &cuda_ctx, width, height, storage_slot)?;

    // Write blit descriptors for this image index.
    {
        let state = backend.surfaces.get(&surface).unwrap();
        state.blit.write_descriptors(
            &companion.device,
            image_index,
            &shared.d3d12_resource,
            &shared.blit_target,
        );
    }

    let texture_handle = backend.next_texture;
    backend.next_texture += 1;
    backend
        .texture_slots
        .insert(storage_slot, texture_handle);
    backend
        .textures
        .insert(texture_handle, Arc::clone(&shared.cuda_texture));

    backend
        .surfaces
        .get_mut(&surface)
        .unwrap()
        .scratch[image_index] = Some(ScratchSlot {
        shared,
        texture_handle,
        cuda_complete: 0,
        dx12_complete: 0,
    });
    Ok(texture_handle)
}

fn companion_ref(backend: &CudaBackend, device: DeviceHandle) -> Result<&Dx12Companion> {
    let gpu = backend.devices.get(&device).context("CUDA: invalid device")?;
    gpu.dx12
        .as_deref()
        .context("CUDA: DX12 companion not available (requires cuda+graphics+dx12 on Windows)")
}

fn present_args(mode: crate::types::PresentMode, allow_tearing: bool) -> (u32, DXGI_PRESENT) {
    match mode {
        crate::types::PresentMode::Fifo
        | crate::types::PresentMode::Mailbox
        | crate::types::PresentMode::Auto => (1, DXGI_PRESENT(0)),
        crate::types::PresentMode::Immediate => {
            if allow_tearing {
                (0, DXGI_PRESENT_ALLOW_TEARING)
            } else {
                (1, DXGI_PRESENT(0))
            }
        }
    }
}

/// Ensure a CUDA device has a DX12 companion (called from `create_device`).
pub(super) fn attach_companion(gpu: &mut CudaDevice) -> Result<()> {
    if gpu.dx12.is_some() {
        return Ok(());
    }
    let companion = Dx12Companion::create(&gpu.ctx)?;
    tracing::info!(
        "CUDA/DX12 companion ready (dxgi_adapter={}, luid={:02x?})",
        companion.dxgi_adapter_id,
        companion.adapter_luid.bytes
    );
    gpu.dx12 = Some(Arc::new(companion));
    Ok(())
}
