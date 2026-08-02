//! CUDA + DX12 surface / swapchain / present.
//!
//! Swapchain backbuffers stay BGRA8 (non-shareable). Per-image float4 shared scratch
//! textures are CUDA-writable; present blits scratch → BGRA UAV → backbuffer, waiting
//! on the shared D3D12 fence that CUDA signals at submit completion.
//!
//! Present completion is published on the **Goldy/CUDA timeline** (event ledger), not
//! the companion DX12 fence counter. `Frame::present` returns that Goldy value so
//! `Context::wait_until` observes the same namespace as compute submits.

use super::dx12_companion::{cuda_signal_fence, cuda_wait_fence, Dx12Companion, MAX_FRAMES};
use super::dx12_interop::{
    record_present_copy, PresentBlitPipeline, SharedScratchTexture, SURFACE_COMPUTE_FORMAT,
    SWAPCHAIN_DXGI_FORMAT,
};
use super::timeline::{self, EventLedger, LedgerEntry};
use super::{CudaBackend, CudaDevice, CudaSubmitContext};
use crate::backend::submission_worker::{self, PendingSubmit};
use crate::backend::{
    ContextHandle, DeviceHandle, FrameToken, PresentFinishState, PresentGpuWork, SurfaceHandle,
    SwapchainImageHandle, TextureHandle,
};
use anyhow::{bail, Context as _, Result};
use cudarc::driver::{sys, CudaContext, CudaEvent, CudaStream};
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
    let device = state.device;
    // CUDA may still be writing imported scratch; drain CUDA + DX12 before destroying
    // tex/surf objects and external memory.
    if let Err(e) = wait_device_idle_for_surface(backend, device) {
        tracing::error!("CUDA/DX12: destroy_surface idle wait failed: {e:#}");
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

    // Take scratch slots, wait CUDA+DX12 idle, then drop CUDA views before external memory.
    let old_slots: Vec<ScratchSlot> = {
        let state = backend.surfaces.get_mut(&surface).unwrap();
        state.scratch.iter_mut().filter_map(|s| s.take()).collect()
    };
    wait_device_idle_for_surface(backend, device)?;
    for slot in old_slots {
        let h = slot.texture_handle;
        if let Some(resource) = backend.textures.remove(&h) {
            if let Some(sid) = resource.storage_slot {
                backend.texture_slots.remove(&sid);
            }
            if let Some(sid) = resource.sampled_slot {
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

    let context = Arc::clone(backend.context(frame.context)?);
    let (worker, next_timeline, event_ledger, cuda_ctx) = {
        let gpu = backend.device(device)?;
        (
            Arc::clone(&gpu.submission_worker),
            Arc::clone(&gpu.next_timeline),
            Arc::clone(&gpu.event_ledger),
            Arc::clone(&gpu.ctx),
        )
    };

    // Require a real completion event for the compute submit — never present without waiting.
    let submit_event = if submit_tv > 0 {
        let guard = event_ledger.lock().unwrap();
        let entry = guard.get(&submit_tv).with_context(|| {
            format!(
                "CUDA/DX12: present submit_tv {submit_tv} has no completion event on context {}",
                frame.context
            )
        })?;
        if entry.context != frame.context {
            bail!(
                "CUDA/DX12: present submit_tv {submit_tv} belongs to context {}, not {}",
                entry.context,
                frame.context
            );
        }
        Some(Arc::clone(&entry.event))
    } else {
        None
    };

    // Goldy timeline value for present/copy completion (same namespace as compute).
    let present_tv = submission_worker::allocate_timeline_value(&next_timeline);
    let present_event = Arc::new(
        cuda_ctx
            .new_event(None)
            .context("CUDA/DX12: create present completion event failed")?,
    );
    event_ledger.lock().unwrap().insert(
        present_tv,
        LedgerEntry {
            context: frame.context,
            event: Arc::clone(&present_event),
            recorded: false,
        },
    );

    // Enqueue CUDA wait(submit) + signal(fence) on the submission worker so later
    // context submits cannot insert between the wait and the handoff signal.
    let cuda_complete = if let Some(submit_event) = submit_event {
        let signal_value = companion.next_fence_value();
        let handoff = CudaPresentHandoff {
            stream: Arc::clone(&context.stream),
            submit_event,
            cuda_ctx: Arc::clone(&companion.cuda_ctx),
            cuda_semaphore: companion.cuda_semaphore,
            signal_value,
        };
        worker
            .enqueue(0, Box::new(handoff))
            .context("CUDA/DX12: enqueue present handoff failed")?;
        // PresentGpuWork::wait_queue needs the CUDA→DX12 fence signal; flush so the handoff
        // cannot still be queued behind unrelated worker jobs when DXGI work starts.
        worker
            .flush()
            .context("CUDA/DX12: flush present handoff before GPU work")?;
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

    companion
        .as_ref(); // keep alive
    let blit = &state.blit;
    blit.write_descriptors(
        &companion.device,
        image_index,
        &scratch_res,
        &blit_target,
    );

    let allocator = companion.present_slots[present_slot].allocator.clone();
    let list = companion.present_slots[present_slot].list.clone();
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
        present_tv,
        present_event,
        event_ledger,
        context,
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
    if !finish.present_ok {
        bail!(
            "CUDA/DX12: Present failed after copy submit (return_fence {} recorded for reuse)",
            finish.return_fence
        );
    }
    Ok(finish.present_timeline)
}

/// Worker job: wait for compute completion, then signal the shared DX12 fence.
struct CudaPresentHandoff {
    stream: Arc<CudaStream>,
    submit_event: Arc<CudaEvent>,
    cuda_ctx: Arc<CudaContext>,
    cuda_semaphore: sys::CUexternalSemaphore,
    signal_value: u64,
}

// SAFETY: executed only on the goldy-submit worker; the semaphore is owned by
// `Dx12Companion` for the lifetime of the device and is not freed while jobs run.
unsafe impl Send for CudaPresentHandoff {}

impl PendingSubmit for CudaPresentHandoff {
    fn execute(self: Box<Self>) -> Result<()> {
        self.stream
            .wait(&self.submit_event)
            .context("CUDA/DX12: stream wait on submit completion")?;
        cuda_signal_fence(
            &self.cuda_ctx,
            self.cuda_semaphore,
            self.stream.cu_stream(),
            self.signal_value,
        )?;
        self.stream
            .synchronize()
            .context("CUDA/DX12: synchronize handoff fence signal")?;
        Ok(())
    }
}

struct CudaDx12PresentGpuWork {
    frame: FrameToken,
    image_index: usize,
    #[allow(dead_code)]
    present_slot: usize,
    scratch_handle: TextureHandle,
    cuda_complete: u64,
    present_tv: crate::timeline::TimelineValue,
    present_event: Arc<CudaEvent>,
    event_ledger: EventLedger,
    context: Arc<CudaSubmitContext>,
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

        // Wait for the copy to retire before DXGI Present (flip model requires rendering done).
        self.companion
            .cpu_wait(return_fence)
            .context("CUDA/DX12: wait for present-copy fence")?;

        let (sync_interval, flags) = present_args(self.present_mode, self.allow_tearing);
        let hr = unsafe { self.swapchain.Present(sync_interval, flags) };
        // Present may fail after the copy is already submitted. Always retire
        // `return_fence` so allocator / scratch reuse stays guarded via finish_present.
        let present_ok = hr.is_ok();
        if !present_ok {
            tracing::error!("CUDA/DX12: Present failed: {hr:?} (retiring copy fence {return_fence})");
        }

        // Publish present/copy completion on the Goldy timeline (not the DX12 fence counter).
        // Record on the dedicated present stream (avoids racing the submission worker's context stream).
        cuda_wait_fence(
            &self.companion.cuda_ctx,
            self.companion.cuda_semaphore,
            self.companion.present_stream.cu_stream(),
            return_fence,
        )?;
        self.present_event
            .record(&self.companion.present_stream)
            .context("CUDA/DX12: record present completion event")?;
        self.companion
            .present_stream
            .synchronize()
            .context("CUDA/DX12: synchronize present stream")?;
        timeline::mark_recorded(&self.event_ledger, self.present_tv);
        timeline::poll_retire_events(
            &self.event_ledger,
            &self.context.completed,
            self.context.handle,
            &self.context.device_retired,
            &self.context.signal_queue,
            &self.context.last_emitted,
        );

        Ok(PresentFinishState {
            frame: self.frame,
            return_fence,
            scratch_texture: Some(self.scratch_handle),
            scratch_layout_updated: true,
            present_timeline: self.present_tv,
            copy_timeline: Some(self.present_tv),
            frame_compute_timeline: None,
            signal_timeline: None,
            render_pass_submitted: false,
            present_ok,
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
        // Size mismatch: drain GPU before destroying the imported scratch.
        wait_device_idle_for_surface(backend, device)?;
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
                if let Some(slot_id) = resource.sampled_slot {
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

/// Drain CUDA submission + all device streams and the DX12 companion before
/// destroying imported scratch (tex/surf objects and external memory).
fn wait_device_idle_for_surface(backend: &mut CudaBackend, device: DeviceHandle) -> Result<()> {
    let (worker, alloc_stream, present_stream, companion) = {
        let gpu = backend.device(device)?;
        (
            Arc::clone(&gpu.submission_worker),
            Arc::clone(&gpu.alloc_stream),
            gpu.dx12.as_ref().map(|c| Arc::clone(&c.present_stream)),
            gpu.dx12.as_ref().map(Arc::clone),
        )
    };
    worker
        .flush()
        .context("CUDA/DX12: flush submission worker before surface teardown")?;
    for context in backend.contexts.values().filter(|c| c.device == device) {
        context
            .stream
            .synchronize()
            .context("CUDA/DX12: context stream synchronize before surface teardown")?;
    }
    if let Some(stream) = present_stream {
        stream
            .synchronize()
            .context("CUDA/DX12: present stream synchronize before surface teardown")?;
    }
    alloc_stream
        .synchronize()
        .context("CUDA/DX12: alloc stream synchronize before surface teardown")?;
    if let Some(companion) = companion {
        companion
            .wait_idle()
            .context("CUDA/DX12: companion wait_idle before surface teardown")?;
    }
    Ok(())
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

#[cfg(test)]
mod present_tests {
    use super::*;
    use crate::backend::{GpuBackend, GpuBackendPresentSplit};
    use raw_window_handle::{
        DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
        RawWindowHandle, Win32WindowHandle, WindowHandle, WindowsDisplayHandle,
    };
    use std::num::NonZeroIsize;
    use windows::core::w;
    use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DestroyWindow, CW_USEDEFAULT, WS_OVERLAPPEDWINDOW,
    };

    struct TestHwnd {
        hwnd: HWND,
    }

    impl TestHwnd {
        fn create() -> Option<Self> {
            // Use a built-in class so we do not need Win32_Graphics_Gdi (RegisterClassW).
            let hwnd = unsafe {
                CreateWindowExW(
                    Default::default(),
                    w!("STATIC"),
                    w!("goldy cuda surface unit"),
                    WS_OVERLAPPEDWINDOW,
                    CW_USEDEFAULT,
                    CW_USEDEFAULT,
                    64,
                    64,
                    None,
                    None,
                    None,
                    None,
                )
            }
            .ok()?;
            Some(Self { hwnd })
        }
    }

    impl Drop for TestHwnd {
        fn drop(&mut self) {
            unsafe {
                let _ = DestroyWindow(self.hwnd);
            }
        }
    }

    impl HasWindowHandle for TestHwnd {
        fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
            let handle = Win32WindowHandle::new(
                NonZeroIsize::new(self.hwnd.0 as isize).ok_or(HandleError::Unavailable)?,
            );
            Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
        }
    }

    impl HasDisplayHandle for TestHwnd {
        fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
            Ok(unsafe {
                DisplayHandle::borrow_raw(RawDisplayHandle::Windows(WindowsDisplayHandle::new()))
            })
        }
    }

    fn try_backend_with_surface() -> Option<(CudaBackend, DeviceHandle, ContextHandle, SurfaceHandle, TestHwnd)> {
        let mut backend = CudaBackend::new().ok()?;
        if backend.enumerate_adapters().is_empty() {
            return None;
        }
        let device = backend.create_device(0).ok()?;
        let ctx = backend.create_context(device).ok()?;
        let window = TestHwnd::create()?;
        let surface = backend
            .create_surface(device, &window, &window, None)
            .ok()?;
        Some((backend, device, ctx, surface, window))
    }

    #[test]
    fn present_errors_when_submit_tv_missing_from_ledger() {
        let Some((mut backend, _device, ctx, surface, _window)) = try_backend_with_surface() else {
            eprintln!("skip: CUDA/DX12 surface unavailable");
            return;
        };
        let (token, _tex) = backend.begin_frame(surface, ctx).expect("begin_frame");
        match backend.take_present_gpu_work(token, 0xDEAD_BEEF) {
            Ok(_) => panic!("missing ledger entry must fail"),
            Err(err) => {
                let msg = format!("{err:#}");
                assert!(
                    msg.contains("no completion event"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn present_returns_goldy_timeline_value() {
        let Some((mut backend, _device, ctx, surface, _window)) = try_backend_with_surface() else {
            eprintln!("skip: CUDA/DX12 surface unavailable");
            return;
        };
        let (token, _tex) = backend.begin_frame(surface, ctx).expect("begin_frame");
        // submit_tv == 0: no compute wait, but present still publishes a Goldy TV.
        let work = backend
            .take_present_gpu_work(token, 0)
            .expect("take_present_gpu_work");
        let finish = work.run().expect("present run");
        let present_tv = backend
            .finish_present(finish, 0)
            .expect("finish_present");
        assert!(present_tv > 0, "present must allocate a Goldy timeline value");
        // Must be waitable in the CUDA event ledger (not a raw DX12 fence counter).
        let event = timeline::lookup_event(
            &backend.context(ctx).unwrap().event_ledger,
            ctx,
            present_tv,
        );
        assert!(
            event.is_some(),
            "present_tv {present_tv} missing from CUDA event ledger"
        );
        assert!(
            event.unwrap().is_complete(),
            "present_tv {present_tv} event should be complete after PresentGpuWork::run"
        );
        backend
            .wait_until(ctx, present_tv)
            .expect("wait_until present_tv on Goldy timeline");
    }
}
