//! CUDA + DX12 surface / swapchain / present.
//!
//! Swapchain backbuffers stay BGRA8 (non-shareable). Per-image float4 shared scratch
//! textures are CUDA-writable; present blits color → BGRA UAV → backbuffer. Compute
//! writes wait on the shared D3D12 fence CUDA signals at submit completion. When
//! `CopyRenderTarget` targets scratch (`bind_render_target`), present blits the DX12
//! raster RT directly and skips the CUDA array round-trip.
//!
//! Present completion is published on the **Goldy/CUDA timeline** (event ledger).
//! After DX12 Execute/Signal, present waits that fence on the dedicated
//! `present_stream` and records a CUDA event there; `Frame::present` returns that
//! Goldy value so `Context::wait_until` / submit sync observe the same namespace as
//! compute. Downstream CUDA work uses stream waits on that event (not submission-
//! stream `cuWaitExternalSemaphoresAsync` on the companion fence).
//!
//! Shared-fence signal ordering: `cuSignalExternalSemaphoresAsync(V)` must be
//! *submitted to CUDA* before D3D12 `Queue.Signal(W)` for any `W > V` on the same
//! fence — a GPU `Wait(V)` between them is not enough and yields
//! `CUDA_ERROR_INVALID_VALUE`. Scratch presents therefore join the submission worker
//! for `submit_tv` inside [`PresentGpuWork::run`] (so the tail `SignalExternalFence`
//! has been issued) before the present copy's return fence is signaled — not under
//! the backend lock in `take_present_gpu_work`.

use super::dx12_companion::PresentCommandSlot;
use super::dx12_companion::{cuda_signal_fence, cuda_wait_fence, Dx12Companion, MAX_FRAMES};
use super::dx12_interop::{
    record_present_copy, PresentBlitPipeline, PresentColorSrcState, SharedScratchTexture, SURFACE_COMPUTE_FORMAT,
    SWAPCHAIN_DXGI_FORMAT,
};
use super::timeline::{self, EventLedger, LedgerCompletion, LedgerEntry};
use super::{CudaBackend, CudaDevice, CudaSubmitContext};
use crate::backend::submission_worker::{self, PendingSubmit};
use crate::backend::{
    ContextHandle, DeviceHandle, FrameToken, PresentFinishState, PresentGpuWork, SurfaceHandle, SwapchainImageHandle,
    TextureHandle,
};
use anyhow::{bail, Context as _, Result};
use cudarc::driver::{sys, CudaContext, CudaEvent, CudaStream};
use raw_window_handle::RawWindowHandle;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, RECT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{WaitForSingleObject, INFINITE};
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;

/// `HANDLE` is a raw pointer; DXGI waitables are process-local and only touched
/// from Goldy's backend lock / present path.
#[derive(Clone, Copy)]
pub(super) struct SendSyncHandle(HANDLE);
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
    /// Requested depth format (DX12-only depth resource; not CUDA-imported).
    pub depth_format: Option<crate::types::DepthFormat>,
    pub depth_texture: Option<ID3D12Resource>,
    pub dsv_offset: Option<u32>,
    pub frame_latency_waitable: Option<SendSyncHandle>,
    /// Rotating present-slot index (0..MAX_FRAMES).
    pub current_frame: usize,
    pub current_image_index: Option<u32>,
    pub current_texture_handle: Option<TextureHandle>,
    /// Per-swapchain-image shared scratch (indexed by backbuffer index).
    pub scratch: Vec<Option<ScratchSlot>>,
    /// Fence value that guards each present-slot's command allocator reuse.
    pub slot_fence: [u64; MAX_FRAMES],
    /// Per-surface present allocator/list pool. Must not share companion.present_slots
    /// across windows — destroying one surface would Reset lists still used by others.
    pub present_cmd_slots: Arc<Vec<PresentCommandSlot>>,
    /// After create/resize, DXGI backbuffers are in COMMON until the first present copy.
    pub backbuffer_in_common: [bool; MAX_FRAMES],
    pub blit: PresentBlitPipeline,
    present_generation: u64,
    present_cache: Arc<Mutex<[PresentListCache; MAX_FRAMES]>>,
    /// Scratch retired from the live slots; destroyed/pooled once `retire_at` completes.
    pending_scratch: Vec<PendingScratchDrop>,
    /// Recently-retired scratch reusable across oscillating resizes (keyed by size).
    scratch_pool: Vec<SharedScratchTexture>,
}

#[derive(Clone, Copy, Default)]
struct PresentListCache {
    generation: u64,
    /// Companion `present_slots[i].generation` when this surface last recorded the
    /// global list. Other surfaces bump that counter on re-record — reuse must miss.
    slot_generation: u64,
    color_src_ptr: usize,
    color_src_state: Option<PresentColorSrcState>,
    color_src_format: Option<crate::types::TextureFormat>,
    blit_target_ptr: usize,
    backbuffer_ptr: usize,
    image_index: usize,
    width: u32,
    height: u32,
    backbuffer_from_common: bool,
    recorded: bool,
}

struct PendingScratchDrop {
    /// Companion fence value that must complete before CUDA/D3D12 teardown is safe.
    retire_at: u64,
    slot: ScratchSlot,
}

/// Soft cap on pooled scratch textures per surface (oscillating interactive resize).
const SCRATCH_POOL_CAP: usize = 4;

pub(super) struct ScratchSlot {
    pub shared: SharedScratchTexture,
    pub texture_handle: TextureHandle,
    /// Fence value after DX12 finished present-copy (CUDA may wait before reuse).
    pub dx12_complete: u64,
    pub present_source: Option<PresentSource>,
    /// Deferred until a submission worker first writes this imported scratch.
    pub pending_scratch_reuse_fence: u64,
}

pub(super) enum PresentSource {
    /// CUDA wrote imported scratch; present waits on companion fence `cuda_complete`.
    CudaScratch { cuda_complete: u64 },
    /// DX12 raster RT; present blits this resource. Same-queue submission order
    /// after `render_to_target` is enough — no extra queue Wait on `fence`.
    Dx12Raster {
        resource: ID3D12Resource,
        fence: u64,
        format: crate::types::TextureFormat,
    },
}

fn present_source_fence(source: &Option<PresentSource>) -> u64 {
    match source {
        Some(PresentSource::CudaScratch { cuda_complete }) => *cuda_complete,
        Some(PresentSource::Dx12Raster { fence, .. }) => *fence,
        None => 0,
    }
}

pub(super) fn create_surface(
    backend: &mut CudaBackend,
    device: DeviceHandle,
    window: &dyn raw_window_handle::HasWindowHandle,
    _display: &dyn raw_window_handle::HasDisplayHandle,
    depth_format: Option<crate::types::DepthFormat>,
) -> Result<SurfaceHandle> {
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
    let swapchain: IDXGISwapChain3 = swapchain1.cast().context("CUDA/DX12: cast to IDXGISwapChain3")?;

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
    let rtv_size = unsafe {
        companion
            .device
            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
    };
    let mut backbuffers = Vec::with_capacity(MAX_FRAMES);
    for i in 0..MAX_FRAMES {
        let buf: ID3D12Resource = unsafe { swapchain.GetBuffer(i as u32) }.context("CUDA/DX12: GetBuffer failed")?;
        let handle = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: handle.ptr + i * rtv_size as usize,
        };
        unsafe { companion.device.CreateRenderTargetView(&buf, None, cpu) };
        backbuffers.push(buf);
    }

    let (depth_texture, dsv_offset) = if let Some(df) = depth_format {
        let (tex, offset) = companion.create_depth_texture(width, height, df)?;
        (Some(tex), Some(offset))
    } else {
        (None, None)
    };

    let blit = PresentBlitPipeline::create(&companion.device)?;
    let present_cmd_slots = Arc::new(companion.create_present_command_slots()?);

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
            depth_format,
            depth_texture,
            dsv_offset,
            frame_latency_waitable,
            current_frame: 0,
            current_image_index: None,
            current_texture_handle: None,
            scratch: (0..MAX_FRAMES).map(|_| None).collect(),
            slot_fence: [0; MAX_FRAMES],
            present_cmd_slots,
            backbuffer_in_common: [true; MAX_FRAMES],
            blit,
            present_generation: 1,
            present_cache: Arc::new(Mutex::new([PresentListCache::default(); MAX_FRAMES])),
            pending_scratch: Vec::new(),
            scratch_pool: Vec::new(),
        },
    );
    tracing::info!(
        "CUDA/DX12: created surface {width}x{height} (compute format {SURFACE_COMPUTE_FORMAT:?}, depth={depth_format:?})"
    );
    Ok(handle)
}

pub(super) fn destroy_surface(backend: &mut CudaBackend, surface: SurfaceHandle) {
    let Some(mut state) = backend.surfaces.remove(&surface) else {
        return;
    };
    let device = state.device;
    // Free DSV slot before dropping depth texture.
    if let Some(dsv) = state.dsv_offset.take() {
        if let Some(gpu) = backend.devices.get(&device) {
            if let Some(companion) = gpu.dx12.as_ref() {
                companion.free_dsv_offset(dsv);
            }
        }
    }
    state.depth_texture = None;
    // CUDA may still be writing imported scratch; drain CUDA + DX12 before destroying
    // tex/surf objects and external memory.
    let mut live_slots: Vec<ScratchSlot> = state.scratch.iter_mut().filter_map(|s| s.take()).collect();
    live_slots.extend(state.pending_scratch.drain(..).map(|p| p.slot));
    state.scratch_pool.clear();

    // Flush worker + drain companion FIRST. Present fences can sit behind DX12
    // Queue.Wait(cuda_fence); invalidate's cpu_wait would hang forever if CUDA's
    // SignalExternalFence is still queued on the submission worker.
    if let Err(e) = wait_device_idle_for_surface(backend, device) {
        tracing::error!(
            "CUDA/DX12: destroy_surface idle wait failed ({e:#}); leaking {} scratch slot(s)",
            live_slots.len()
        );
        for slot in live_slots {
            if let Some(resource) = backend.textures.remove(&slot.texture_handle) {
                if let Some(sid) = resource.storage_slot {
                    backend.texture_slots.remove(&sid);
                }
                if let Some(sid) = resource.sampled_slot {
                    backend.texture_slots.remove(&sid);
                }
                std::mem::forget(resource);
            }
            std::mem::forget(slot);
        }
        if let Some(SendSyncHandle(waitable)) = state.frame_latency_waitable.take() {
            unsafe {
                let _ = CloseHandle(waitable);
            }
        }
        return;
    }

    // Per-surface present lists may still reference this surface's backbuffers.
    // Invalidate only *this* surface's pool — never companion.present_slots (shared).
    if let Some(gpu) = backend.devices.get(&device) {
        if let Some(companion) = gpu.dx12.as_ref() {
            if let Err(e) = companion.invalidate_command_slots(&state.present_cmd_slots) {
                tracing::error!("CUDA/DX12: invalidate surface present slots on destroy failed: {e:#}");
            }
        }
    }

    for slot in live_slots {
        unregister_scratch_texture(backend, slot.texture_handle);
        drop(slot);
    }
    if let Some(SendSyncHandle(waitable)) = state.frame_latency_waitable.take() {
        unsafe {
            let _ = CloseHandle(waitable);
        }
    }
}

fn unregister_scratch_texture(backend: &mut CudaBackend, handle: TextureHandle) {
    if let Some(resource) = backend.textures.remove(&handle) {
        if let Some(sid) = resource.storage_slot {
            backend.texture_slots.remove(&sid);
        }
        if let Some(sid) = resource.sampled_slot {
            backend.texture_slots.remove(&sid);
        }
        drop(resource);
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

pub(super) fn surface_resize(backend: &mut CudaBackend, surface: SurfaceHandle, width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    let device = backend
        .surfaces
        .get(&surface)
        .context("CUDA/DX12: invalid surface")?
        .device;
    {
        let state = backend.surfaces.get(&surface).unwrap();
        if state.width == width && state.height == height {
            return Ok(());
        }
    }

    let resize_t0 = std::time::Instant::now();
    backend.graph_stats.surface_resizes.fetch_add(1, Ordering::Relaxed);

    // Snapshot surface fence high-water before taking scratch slots so the idle
    // drain can wait on present-source / dx12_complete / slot_fence values.
    let fence_high = {
        let state = backend.surfaces.get(&surface).unwrap();
        let mut high = 0u64;
        for v in &state.slot_fence {
            high = high.max(*v);
        }
        for slot in state.scratch.iter().flatten() {
            high = high
                .max(present_source_fence(&slot.present_source))
                .max(slot.dx12_complete);
        }
        for pending in &state.pending_scratch {
            high = high.max(pending.retire_at);
        }
        high
    };

    // Take live scratch out of the swapchain slots. Teardown is deferred until the
    // companion fence retires so ResizeBuffers is not blocked on cuDestroyExternalMemory.
    let old_slots: Vec<ScratchSlot> = {
        let state = backend.surfaces.get_mut(&surface).unwrap();
        state.scratch.iter_mut().filter_map(|s| s.take()).collect()
    };
    let destroyed_handles: Vec<TextureHandle> = old_slots.iter().map(|s| s.texture_handle).collect();
    let destroyed_slots: Vec<u32> = destroyed_handles
        .iter()
        .filter_map(|h| {
            backend.textures.get(h).and_then(|tex| {
                let mut slots = Vec::new();
                if let Some(s) = tex.storage_slot {
                    slots.push(s);
                }
                if let Some(s) = tex.sampled_slot {
                    slots.push(s);
                }
                (!slots.is_empty()).then_some(slots)
            })
        })
        .flatten()
        .collect();

    let idle_t0 = std::time::Instant::now();
    wait_surface_resources_idle(backend, device, fence_high)?;
    let idle_ns = idle_t0.elapsed().as_nanos() as u64;
    backend
        .graph_stats
        .surface_resize_idle_ns
        .fetch_add(idle_ns, Ordering::Relaxed);

    // Evict retained entries that reference the retiring scratch, unregister handles,
    // then park the imported textures for deferred drop/pool (not on the resize hot path).
    let teardown_t0 = std::time::Instant::now();
    let evicted = evict_retained_touching_scratch(backend, &destroyed_handles, &destroyed_slots);
    backend
        .graph_stats
        .surface_resize_evictions
        .fetch_add(evicted, Ordering::Relaxed);
    for slot in old_slots {
        let retire_at = fence_high
            .max(present_source_fence(&slot.present_source))
            .max(slot.dx12_complete);
        unregister_scratch_texture(backend, slot.texture_handle);
        backend
            .surfaces
            .get_mut(&surface)
            .unwrap()
            .pending_scratch
            .push(PendingScratchDrop { retire_at, slot });
    }
    let teardown_ns = teardown_t0.elapsed().as_nanos() as u64;
    backend
        .graph_stats
        .surface_resize_teardown_ns
        .fetch_add(teardown_ns, Ordering::Relaxed);

    {
        let companion = companion_ref(backend, device)?;
        let slots = Arc::clone(&backend.surfaces.get(&surface).unwrap().present_cmd_slots);
        companion.invalidate_command_slots(&slots)?;
    }
    let state = backend.surfaces.get_mut(&surface).unwrap();
    state.backbuffers.clear();
    let allow_tearing = companion_ref(backend, device)?.allow_tearing;
    let mut resize_flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    if allow_tearing {
        resize_flags = DXGI_SWAP_CHAIN_FLAG(resize_flags.0 | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0);
    }
    let swapchain = backend.surfaces.get(&surface).unwrap().swapchain.clone();
    unsafe { swapchain.ResizeBuffers(MAX_FRAMES as u32, width, height, SWAPCHAIN_DXGI_FORMAT, resize_flags) }
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
    {
        let state = backend.surfaces.get_mut(&surface).unwrap();
        for i in 0..MAX_FRAMES {
            let buf: ID3D12Resource =
                unsafe { state.swapchain.GetBuffer(i as u32) }.context("CUDA/DX12: GetBuffer after resize")?;
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
        state.present_generation = state.present_generation.wrapping_add(1);
        *state.present_cache.lock().unwrap() = [PresentListCache::default(); MAX_FRAMES];
    }

    // Recreate DX12-only depth at the new size (free old DSV first).
    let (depth_format, old_dsv) = {
        let state = backend.surfaces.get_mut(&surface).unwrap();
        let df = state.depth_format;
        let old = state.dsv_offset.take();
        state.depth_texture = None;
        (df, old)
    };
    if let Some(dsv) = old_dsv {
        companion_ref(backend, device)?.free_dsv_offset(dsv);
    }
    if let Some(df) = depth_format {
        let (tex, offset) = companion_ref(backend, device)?.create_depth_texture(width, height, df)?;
        let state = backend.surfaces.get_mut(&surface).unwrap();
        state.depth_texture = Some(tex);
        state.dsv_offset = Some(offset);
    }

    // Intentionally do not drain pending scratch here: cuDestroyExternalMemory /
    // pool insertion runs on the next acquire (`ensure_scratch`) so ResizeBuffers
    // returns without paying import teardown on the interactive resize path.

    tracing::info!(
        width,
        height,
        idle_ns,
        teardown_ns,
        evicted,
        total_ns = resize_t0.elapsed().as_nanos() as u64,
        "CUDA/DX12: surface_resize complete"
    );
    Ok(())
}

/// Evict retained partitions that still reference destroyed surface scratch.
///
/// Present writes use imported D3D12 scratch and are not CUDA-graph-capturable, so
/// graph islands normally have no scratch pins. Stream-replay segments and Ops
/// entries may still embed destroyed texture handles or registry slot indices — those
/// must go. Unrelated retained compute graphs are left intact across resize.
fn evict_retained_touching_scratch(
    backend: &mut CudaBackend,
    destroyed_handles: &[TextureHandle],
    destroyed_slots: &[u32],
) -> u64 {
    use crate::backend::{GpuCommand, GraphCommand};
    use std::collections::HashSet;

    if destroyed_handles.is_empty() && destroyed_slots.is_empty() {
        return 0;
    }
    let handles: HashSet<TextureHandle> = destroyed_handles.iter().copied().collect();
    let slots: HashSet<u32> = destroyed_slots.iter().copied().collect();
    let destroyed_ptrs: HashSet<*const super::texture::CudaTextureResource> = destroyed_handles
        .iter()
        .filter_map(|h| backend.textures.get(h).map(|t| Arc::as_ptr(t)))
        .collect();

    let touches = |entry: &super::RetainedEntry| -> bool {
        match entry {
            super::RetainedEntry::Segmented {
                segments,
                scratch_images,
                ..
            } => {
                let scratch_hit = scratch_images.iter().any(|(surface, image)| {
                    backend
                        .surfaces
                        .get(surface)
                        .and_then(|state| state.scratch.get(*image))
                        .and_then(|slot| slot.as_ref())
                        .is_some_and(|slot| handles.contains(&slot.texture_handle))
                });
                if scratch_hit {
                    return true;
                }
                let stream_ops = super::pending_submit::collect_stream_ops(segments);
                let (_buffers, _modules, textures) = super::pending_submit::collect_pins(&stream_ops);
                textures
                    .iter()
                    .any(|texture| destroyed_ptrs.contains(&Arc::as_ptr(texture)))
            }
            super::RetainedEntry::Ops(ops) => {
                let (buffers, modules, textures) = super::pending_submit::collect_pins(ops);
                let _ = (buffers, modules, &slots);
                textures
                    .iter()
                    .any(|texture| destroyed_ptrs.contains(&Arc::as_ptr(texture)))
            }
            super::RetainedEntry::Render(commands) => commands.iter().any(|cmd| match cmd {
                GraphCommand::Compute(command) => match command {
                    GpuCommand::WriteTexture { texture, .. }
                    | GpuCommand::WriteTextureRegion { texture, .. }
                    | GpuCommand::CopyRenderTarget { dst: texture, .. }
                    | GpuCommand::CopyBufferToTexture { dst: texture, .. }
                    | GpuCommand::CopyTextureToReadback { src: texture, .. } => handles.contains(texture),
                    GpuCommand::CopyTexture { src, dst } => handles.contains(src) || handles.contains(dst),
                    GpuCommand::BindResourcesRaw { indices, .. } => indices.iter().any(|i| slots.contains(i)),
                    GpuCommand::FrameTableStaging { data } => data.iter().any(|word| slots.contains(word)),
                    GpuCommand::ResourceBarrier { textures, .. } => {
                        textures.iter().any(|(handle, _)| handles.contains(handle))
                    }
                    _ => false,
                },
                GraphCommand::Render { .. } => false,
            }),
            super::RetainedEntry::PresentRenderTarget { .. } => false,
        }
    };

    let mut stale: Vec<(ContextHandle, u64)> = backend
        .retained
        .iter()
        .filter_map(|(&(ctx, key), entry)| touches(entry).then_some((ctx, key)))
        .collect();

    // Belt-and-suspenders: if a captured graph somehow pinned imported scratch, evict it.
    if !destroyed_ptrs.is_empty() {
        for (&(ctx, key), entry) in backend.retained.iter() {
            if !matches!(entry, super::RetainedEntry::Segmented { .. }) {
                continue;
            }
            let Some(context) = backend.contexts.get(&ctx) else {
                continue;
            };
            let Some(device) = backend.devices.get(&context.device) else {
                continue;
            };
            let Ok(registry) = device.graph_registry.lock() else {
                continue;
            };
            if destroyed_ptrs
                .iter()
                .any(|ptr| registry.program_holds_texture_ptr(ctx, key, *ptr))
            {
                stale.push((ctx, key));
            }
        }
    }

    stale.sort_unstable();
    stale.dedup();
    let mut n = 0u64;
    for (ctx, key) in stale {
        if backend.retained.remove(&(ctx, key)).is_some() {
            backend.enqueue_evict_retained(ctx, key);
            n += 1;
        }
    }
    n
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
    // Per-surface present_cmd_slots: wait only this surface's slot fence (plus DXGI
    // waitable above). Do not touch companion.present_slots — those are shared and
    // caused multi-window freezes/teardown hangs when one surface destroyed others' lists.
    let companion = companion_ref(backend, device)?;
    let slot_prev = backend.surfaces.get(&surface).unwrap().present_cmd_slots[present_slot]
        .fence_value
        .load(Ordering::Acquire);
    let wait_fence = prev_fence.max(slot_prev);
    if wait_fence > 0 {
        companion.cpu_wait(wait_fence)?;
        backend.surfaces.get_mut(&surface).unwrap().slot_fence[present_slot] = 0;
    }

    let image_index = {
        let state = backend.surfaces.get_mut(&surface).unwrap();
        let idx = unsafe { state.swapchain.GetCurrentBackBufferIndex() };
        state.current_image_index = Some(idx);
        idx as usize
    };

    let tex_handle = ensure_scratch(backend, surface, image_index)?;

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

pub(super) fn submit_frame(backend: &mut CudaBackend, frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
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

    let device = backend.surfaces.get(&surface_handle).context("invalid surface")?.device;
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
    let (worker, next_timeline, event_ledger, event_pool) = {
        let gpu = backend.device(device)?;
        (
            Arc::clone(&gpu.submission_worker),
            Arc::clone(&gpu.next_timeline),
            Arc::clone(&gpu.event_ledger),
            Arc::clone(&gpu.event_pool),
        )
    };

    let present_source = backend
        .surfaces
        .get_mut(&surface_handle)
        .and_then(|state| state.scratch.get_mut(image_index))
        .and_then(|slot| slot.as_mut())
        .context("no scratch for present")?
        .present_source
        .take();
    let direct_cuda_complete = match &present_source {
        Some(PresentSource::CudaScratch { cuda_complete }) => *cuda_complete,
        _ => 0,
    };
    let needs_fallback_handoff = !matches!(&present_source, Some(PresentSource::Dx12Raster { .. }))
        && direct_cuda_complete == 0
        && submit_tv > 0;

    // DX12-raster presents are already guarded by their raster fence and do not
    // depend on a CUDA submit-ledger event. CUDA scratch normally carries the
    // tail signal in PresentSource; this lookup only supports legacy/fallback paths.
    let submit_event = if needs_fallback_handoff {
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
        match &entry.completion {
            LedgerCompletion::CudaEvent(event) => Some(Arc::clone(event)),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            LedgerCompletion::Dx12Fence { .. } => {
                bail!("CUDA/DX12: present submit_tv {submit_tv} is a DX12 fence entry, not a CUDA event")
            }
        }
    } else {
        None
    };

    // Goldy timeline value for present/copy completion (same namespace as compute).
    //
    // CUDA scratch present publishes a CUDA event: after DX12 Execute/Signal,
    // PresentGpuWork waits the companion fence on `present_stream`
    // (`cuWaitExternalSemaphoresAsync`, Signal already issued) and records the
    // event there. Downstream CUDA submits then `stream.wait(event)` — device-side
    // CUDA→CUDA ordering — instead of demoting a Dx12Fence ledger entry onto the
    // submission stream (which deposits sticky NOT_SUPPORTED on this WDDM stack).
    //
    // Dx12Raster (direct raster) stays on the companion fence ledger: no CUDA body
    // to bridge and no per-frame present-completion event allocation in steady state.
    // Fence values stay allocated at Signal time in `run` (not here).
    let present_tv = submission_worker::allocate_timeline_value(&next_timeline);
    let dx12_raster_direct = matches!(&present_source, Some(PresentSource::Dx12Raster { .. }));
    let present_completion = if dx12_raster_direct {
        // Fence value is bound at Signal time in PresentGpuWork::run — allocating here
        // races other companion fence users and collapses present-slot wait depth.
        event_ledger.lock().unwrap().insert(
            present_tv,
            LedgerEntry {
                context: frame.context,
                completion: LedgerCompletion::Dx12Fence {
                    companion: Arc::clone(&companion),
                    value: 0,
                },
                recorded: false,
            },
        );
        PresentCompletion::Dx12Fence
    } else {
        let present_event = event_pool
            .acquire()
            .context("CUDA/DX12: create present completion event failed")?;
        event_ledger.lock().unwrap().insert(
            present_tv,
            LedgerEntry {
                context: frame.context,
                completion: LedgerCompletion::CudaEvent(Arc::clone(&present_event)),
                recorded: false,
            },
        );
        backend
            .graph_stats
            .present_completion_events
            .fetch_add(1, Ordering::Relaxed);
        backend.graph_stats.completion_events.fetch_add(1, Ordering::Relaxed);
        PresentCompletion::CudaEvent(present_event)
    };

    // New CUDA scratch submits signal in their own tail. Keep a temporary handoff
    // only for callers that supplied a submit timeline without a scratch-tail signal.
    //
    // Steady-state scratch: do **not** CPU-join the worker here (under the backend
    // lock). `PresentGpuWork::run` waits for `submit_tv` before Queue.Signal of a
    // higher companion fence — see module docs on SignalExternalFence ordering.
    let join_submit_tv = if direct_cuda_complete > 0 && submit_tv > 0 {
        Some(submit_tv)
    } else {
        None
    };
    let cuda_complete = if direct_cuda_complete > 0 {
        direct_cuda_complete
    } else if let Some(submit_event) = submit_event {
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
        backend.graph_stats.present_handoffs.fetch_add(1, Ordering::Relaxed);
        // PresentGpuWork::wait_queue needs the CUDA→DX12 fence signal; flush so the handoff
        // cannot still be queued behind unrelated worker jobs when DXGI work starts.
        worker
            .flush()
            .context("CUDA/DX12: flush present handoff before GPU work")?;
        backend.graph_stats.worker_flushes.fetch_add(1, Ordering::Relaxed);
        signal_value
    } else {
        0
    };

    let state = backend.surfaces.get_mut(&surface_handle).context("invalid surface")?;
    let scratch = state
        .scratch
        .get_mut(image_index)
        .and_then(|s| s.as_mut())
        .context("no scratch for present")?;
    let scratch_handle = scratch.texture_handle;
    let blit_target = scratch.shared.blit_target.clone();
    let width = scratch.shared.width;
    let height = scratch.shared.height;
    // Prefer a stashed DX12 raster RT over CUDA-written imported scratch.
    let (color_src, color_src_state, color_src_format) = match present_source {
        Some(PresentSource::Dx12Raster {
            resource,
            fence: _,
            format,
        }) => (resource, PresentColorSrcState::Common, format),
        Some(PresentSource::CudaScratch { .. }) | None => (
            scratch.shared.d3d12_resource.clone(),
            PresentColorSrcState::UnorderedAccess,
            SURFACE_COMPUTE_FORMAT,
        ),
    };
    let backbuffer = state.backbuffers[image_index].clone();
    let backbuffer_from_common = state.backbuffer_in_common[image_index];
    let swapchain = state.swapchain.clone();
    let present_mode = state.present_mode;
    let allow_tearing = companion.allow_tearing;
    let present_cache = Arc::clone(&state.present_cache);
    let present_cmd_slots = Arc::clone(&state.present_cmd_slots);
    let slot_generation = present_cmd_slots[present_slot].generation.load(Ordering::Acquire);
    let cache_entry = PresentListCache {
        generation: state.present_generation,
        slot_generation,
        color_src_ptr: color_src.as_raw() as usize,
        color_src_state: Some(color_src_state),
        color_src_format: Some(color_src_format),
        blit_target_ptr: blit_target.as_raw() as usize,
        backbuffer_ptr: backbuffer.as_raw() as usize,
        image_index,
        width,
        height,
        backbuffer_from_common,
        recorded: true,
    };
    let reuse_list = {
        let cache = present_cache.lock().unwrap();
        let prior = cache[present_slot];
        prior.recorded
            && prior.generation == cache_entry.generation
            && prior.slot_generation == slot_generation
            && prior.slot_generation != 0
            && prior.color_src_ptr == cache_entry.color_src_ptr
            && prior.color_src_state == cache_entry.color_src_state
            && prior.color_src_format == cache_entry.color_src_format
            && prior.blit_target_ptr == cache_entry.blit_target_ptr
            && prior.backbuffer_ptr == cache_entry.backbuffer_ptr
            && prior.image_index == cache_entry.image_index
            && prior.width == cache_entry.width
            && prior.height == cache_entry.height
            && prior.backbuffer_from_common == cache_entry.backbuffer_from_common
    };
    companion.as_ref(); // keep alive
    let blit = &state.blit;
    if !reuse_list {
        blit.write_descriptors(
            &companion.device,
            image_index,
            &color_src,
            &blit_target,
            color_src_format,
        )?;
    }

    let allocator = present_cmd_slots[present_slot].allocator.clone();
    let list = present_cmd_slots[present_slot].list.clone();
    let blit_pipe = PresentBlitPipeline {
        root_signature: state.blit.root_signature.clone(),
        pso_float: state.blit.pso_float.clone(),
        pso_unorm8: state.blit.pso_unorm8.clone(),
        srv_uav_heap: state.blit.srv_uav_heap.clone(),
        descriptor_size: state.blit.descriptor_size,
    };

    Ok(Box::new(CudaDx12PresentGpuWork {
        frame,
        image_index,
        present_slot,
        scratch_handle,
        cuda_complete,
        join_submit_tv,
        worker: Arc::clone(&worker),
        next_timeline: Arc::clone(&next_timeline),
        present_tv,
        present_completion,
        event_ledger,
        stats: Arc::clone(&backend.graph_stats),
        context,
        companion,
        present_cmd_slots,
        color_src,
        color_src_state,
        color_src_format,
        blit_target,
        backbuffer,
        backbuffer_from_common,
        allocator,
        list,
        blit: blit_pipe,
        swapchain,
        present_mode,
        allow_tearing,
        width,
        height,
        present_cache,
        cache_entry,
        reuse_list,
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
                slot.pending_scratch_reuse_fence = finish.return_fence;
            }
        }
        if finish.present_ok {
            state.backbuffer_in_common[image_index] = false;
        }
        if let Some(sc) = backend.contexts.get(&ctx) {
            sc.signal_queue.push(crate::signal::Signal::SwapchainReturned {
                image_index: image_index as u32,
            });
        }
    }
    if !finish.present_ok {
        bail!(
            "CUDA/DX12: Present failed after copy submit (return_fence {} recorded for reuse; \
             see prior 'CUDA/DX12: Present failed' log for HRESULT)",
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
        // Do not CPU-sync the stream: DX12 `wait_queue(signal_value)` orders the
        // present copy after this CUDA→fence signal on the GPU timeline.
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
    /// When set, join the submission worker for this timeline before companion `Signal(W)`.
    join_submit_tv: Option<crate::timeline::TimelineValue>,
    worker: Arc<crate::backend::submission_worker::SubmissionWorker>,
    next_timeline: Arc<std::sync::atomic::AtomicU64>,
    present_tv: crate::timeline::TimelineValue,
    present_completion: PresentCompletion,
    event_ledger: EventLedger,
    stats: Arc<super::retained_graph::CudaGraphStats>,
    context: Arc<CudaSubmitContext>,
    companion: Arc<Dx12Companion>,
    present_cmd_slots: Arc<Vec<PresentCommandSlot>>,
    color_src: ID3D12Resource,
    color_src_state: PresentColorSrcState,
    color_src_format: crate::types::TextureFormat,
    blit_target: ID3D12Resource,
    backbuffer: ID3D12Resource,
    backbuffer_from_common: bool,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    blit: PresentBlitPipeline,
    swapchain: IDXGISwapChain3,
    present_mode: crate::types::PresentMode,
    allow_tearing: bool,
    width: u32,
    height: u32,
    present_cache: Arc<Mutex<[PresentListCache; MAX_FRAMES]>>,
    cache_entry: PresentListCache,
    reuse_list: bool,
}

enum PresentCompletion {
    /// Present completion bridged to CUDA via `present_stream` after DX12 Signal.
    CudaEvent(Arc<CudaEvent>),
    /// Raster-direct: companion fence allocated at Execute/Signal, not in take_present.
    Dx12Fence,
}

impl PresentGpuWork for CudaDx12PresentGpuWork {
    fn run(mut self: Box<Self>) -> Result<PresentFinishState> {
        // Cross-domain only: CUDA→DX12 external fence. Raster→present on the same
        // DIRECT queue is already ordered by submission; do not wait_queue(dx12_src_fence).
        //
        // Join the submission worker so `SignalExternalFence(cuda_complete)` has been
        // issued to CUDA before we Queue.Signal a higher present return fence (module
        // docs). Done here — off the backend lock — not in `take_present_gpu_work`.
        if let Some(submit_tv) = self.join_submit_tv {
            self.worker
                .wait_submitted_if_scheduled(
                    submit_tv,
                    submission_worker::submission_horizon(&self.next_timeline),
                )
                .context("CUDA/DX12: wait for scratch SignalExternalFence before present")?;
        }
        if self.cuda_complete > 0 {
            self.companion.wait_queue(self.cuda_complete)?;
        }

        if !self.reuse_list {
            // begin_frame already CPU-waited the global + per-surface present-slot fences.
            unsafe { self.allocator.Reset() }.context("reset present allocator")?;
            unsafe { self.list.Reset(&self.allocator, None) }.context("reset present list")?;
            // Invalidate reuse caches for this surface slot only.
            let new_gen = self.present_cmd_slots[self.present_slot]
                .generation
                .fetch_add(1, Ordering::AcqRel)
                + 1;
            self.cache_entry.slot_generation = new_gen;

            record_present_copy(
                &self.list,
                &self.blit,
                self.image_index,
                &self.color_src,
                self.color_src_state,
                self.color_src_format,
                &self.blit_target,
                &self.backbuffer,
                self.backbuffer_from_common,
                self.width,
                self.height,
            )?;
            unsafe { self.list.Close() }.context("close present list")?;
            self.stats.present_list_records.fetch_add(1, Ordering::Relaxed);
        }

        let cmd: ID3D12CommandList = self.list.cast().context("cast present list")?;
        let return_fence = self.companion.next_fence_value();
        if matches!(self.present_completion, PresentCompletion::Dx12Fence) {
            timeline::bind_dx12_fence_value(&self.event_ledger, self.present_tv, return_fence);
        }
        self.companion.execute_and_signal(&[Some(cmd)], return_fence)?;
        self.present_cmd_slots[self.present_slot]
            .fence_value
            .store(return_fence, Ordering::Release);
        if !self.reuse_list {
            self.present_cache.lock().unwrap()[self.present_slot] = self.cache_entry;
        }

        // Flip-model Present is ordered after the copy on the DX12 queue; no CPU wait.
        let (sync_interval, flags) = present_args(self.present_mode, self.allow_tearing);
        let hr = unsafe { self.swapchain.Present(sync_interval, flags) };
        // Present may fail after the copy is already submitted. Always retire
        // `return_fence` so allocator / scratch reuse stays guarded via finish_present.
        let present_ok = hr.is_ok();
        if !present_ok {
            tracing::error!(
                "CUDA/DX12: Present failed: {hr:?} sync_interval={sync_interval} flags={flags:?} \
                 (retiring copy fence {return_fence})"
            );
        }

        // Publish present/copy completion on the Goldy timeline.
        match self.present_completion {
            PresentCompletion::CudaEvent(ref present_event) => {
                // Bridge DX12 Signal → CUDA event on the dedicated present stream (Signal
                // already issued — satisfies CUDA external-semaphore wait-before-signal rule).
                // Do not wait the fence on the submission stream; that path poisons later
                // cuStreamSynchronize on WDDM+D3D12.
                cuda_wait_fence(
                    &self.companion.cuda_ctx,
                    self.companion.cuda_semaphore,
                    self.companion.present_stream.cu_stream(),
                    return_fence,
                )?;
                present_event
                    .record(&self.companion.present_stream)
                    .context("CUDA/DX12: record present completion event")?;
                // Leave present_stream async — ledger polling retires the timeline later.
            }
            PresentCompletion::Dx12Fence => {
                // DX12 fence ledger: copy completion is already on the companion fence.
            }
        }
        timeline::mark_recorded(&self.event_ledger, self.present_tv);
        self.context.poll_retire_events();

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

fn ensure_scratch(backend: &mut CudaBackend, surface: SurfaceHandle, image_index: usize) -> Result<TextureHandle> {
    let has_pending = backend
        .surfaces
        .get(&surface)
        .map(|state| !state.pending_scratch.is_empty())
        .unwrap_or(false);
    if has_pending {
        drain_pending_scratch(backend, surface)?;
    }

    let (device, width, height, reuse) = {
        let state = backend.surfaces.get(&surface).context("invalid surface")?;
        let reuse = state.scratch.get(image_index).and_then(|s| s.as_ref()).map(|s| {
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
        // Size mismatch should be rare after resize clears slots; park for deferred drop.
        if let Some(slot) = backend
            .surfaces
            .get_mut(&surface)
            .unwrap()
            .scratch
            .get_mut(image_index)
            .and_then(|s| s.take())
        {
            let retire_at = present_source_fence(&slot.present_source).max(slot.dx12_complete);
            unregister_scratch_texture(backend, old);
            backend
                .surfaces
                .get_mut(&surface)
                .unwrap()
                .pending_scratch
                .push(PendingScratchDrop { retire_at, slot });
            if retire_at > 0 {
                wait_surface_resources_idle(backend, device, retire_at)?;
            }
            drain_pending_scratch(backend, surface)?;
        }
    }

    let storage_slot = backend.alloc_registry_slot();
    let cuda_ctx = Arc::clone(&backend.device(device)?.ctx);

    let shared = if let Some(mut pooled) = take_pooled_scratch(backend, surface, width, height) {
        pooled.retarget_storage_slot(&cuda_ctx, storage_slot)?;
        pooled
    } else {
        let companion = companion_ref(backend, device)?;
        SharedScratchTexture::create(companion, &cuda_ctx, width, height, storage_slot)?
    };

    // Write blit descriptors for this image index.
    {
        let companion = companion_ref(backend, device)?;
        let state = backend.surfaces.get(&surface).unwrap();
        state.blit.write_descriptors(
            &companion.device,
            image_index,
            &shared.d3d12_resource,
            &shared.blit_target,
            SURFACE_COMPUTE_FORMAT,
        )?;
    }

    let texture_handle = backend.next_texture;
    backend.next_texture += 1;
    backend.texture_slots.insert(storage_slot, texture_handle);
    backend
        .textures
        .insert(texture_handle, Arc::clone(&shared.cuda_texture));

    backend.surfaces.get_mut(&surface).unwrap().scratch[image_index] = Some(ScratchSlot {
        shared,
        texture_handle,
        dx12_complete: 0,
        present_source: None,
        pending_scratch_reuse_fence: 0,
    });
    {
        let state = backend.surfaces.get_mut(&surface).unwrap();
        state.present_generation = state.present_generation.wrapping_add(1);
        *state.present_cache.lock().unwrap() = [PresentListCache::default(); MAX_FRAMES];
    }
    Ok(texture_handle)
}

fn take_pooled_scratch(
    backend: &mut CudaBackend,
    surface: SurfaceHandle,
    width: u32,
    height: u32,
) -> Option<SharedScratchTexture> {
    let state = backend.surfaces.get_mut(&surface)?;
    let idx = state
        .scratch_pool
        .iter()
        .position(|s| s.width == width && s.height == height)?;
    Some(state.scratch_pool.swap_remove(idx))
}

/// Recycle fence-retired scratch into the size pool, or drop when the pool is full.
fn drain_pending_scratch(backend: &mut CudaBackend, surface: SurfaceHandle) -> Result<()> {
    let device = backend
        .surfaces
        .get(&surface)
        .context("CUDA/DX12: invalid surface")?
        .device;
    let completed = {
        let companion = companion_ref(backend, device)?;
        unsafe { companion.fence.GetCompletedValue() }
    };
    let state = backend.surfaces.get_mut(&surface).unwrap();
    let mut still_pending = Vec::new();
    let mut retired = Vec::new();
    for pending in state.pending_scratch.drain(..) {
        if pending.retire_at == 0 || pending.retire_at <= completed {
            retired.push(pending.slot);
        } else {
            still_pending.push(pending);
        }
    }
    state.pending_scratch = still_pending;
    for slot in retired {
        let shared = slot.shared;
        if state.scratch_pool.len() < SCRATCH_POOL_CAP {
            state.scratch_pool.push(shared);
        } else {
            // Prefer pooling the newly retired size; drop an arbitrary older entry.
            let _ = state.scratch_pool.pop();
            state.scratch_pool.push(shared);
        }
    }
    Ok(())
}

fn companion_ref(backend: &CudaBackend, device: DeviceHandle) -> Result<&Dx12Companion> {
    let gpu = backend.devices.get(&device).context("CUDA: invalid device")?;
    gpu.dx12
        .as_deref()
        .context("CUDA: DX12 companion not available (requires cuda+graphics+dx12 on Windows)")
}

/// Locate the live surface scratch slot that owns `texture` (present drawable).
pub(super) fn scratch_slot_for_texture(
    backend: &CudaBackend,
    texture: TextureHandle,
) -> Option<(SurfaceHandle, usize)> {
    for (surface, state) in &backend.surfaces {
        for (image_index, slot) in state.scratch.iter().enumerate() {
            if slot.as_ref().is_some_and(|s| s.texture_handle == texture) {
                return Some((*surface, image_index));
            }
        }
    }
    None
}

/// True when any recorded CUDA completion event on this device is still outstanding.
fn cuda_device_has_pending_work(backend: &CudaBackend, device: DeviceHandle) -> bool {
    for context in backend.contexts.values().filter(|c| c.device == device) {
        let Ok(guard) = context.event_ledger.lock() else {
            return true;
        };
        for entry in guard.values() {
            if entry.context == context.handle && entry.recorded && !entry.is_complete() {
                return true;
            }
        }
    }
    false
}

/// Drain GPU work that may still reference this surface's imported scratch / present slots.
///
/// Skips CUDA stream synchronizes when the event ledger shows no in-flight work (common
/// between frames). Skips the DX12 CPU wait when `fence_high` has already retired.
fn wait_surface_resources_idle(backend: &mut CudaBackend, device: DeviceHandle, mut fence_high: u64) -> Result<()> {
    let (worker, present_stream, companion, cuda_ctx) = {
        let gpu = backend.device(device)?;
        let companion = gpu.dx12.as_ref().map(Arc::clone);
        if let Some(c) = companion.as_ref() {
            fence_high = fence_high.max(c.companion_fence_high_water());
        }
        (
            Arc::clone(&gpu.submission_worker),
            companion.as_ref().map(|c| Arc::clone(&c.present_stream)),
            companion,
            Arc::clone(&gpu.ctx),
        )
    };
    worker
        .flush()
        .context("CUDA/DX12: flush submission worker before surface teardown")?;
    backend.graph_stats.worker_flushes.fetch_add(1, Ordering::Relaxed);

    // Retire DX12 first so any CUDA WaitExternalFence (scratch reuse) can complete
    // before cuStreamSynchronize.
    if let Some(companion) = companion.as_ref() {
        if fence_high > 0 {
            let completed = unsafe { companion.fence.GetCompletedValue() };
            if completed < fence_high {
                companion
                    .cpu_wait(fence_high)
                    .context("CUDA/DX12: companion fence wait before surface teardown")?;
            }
        }
    }

    // See wait_device_idle_for_surface: drain cudarc sticky error_state after DX12 idle.
    if let Err(e) = cuda_ctx.check_err() {
        tracing::debug!("CUDA/DX12: cleared sticky context error before surface resource sync: {e:?}");
    }

    if cuda_device_has_pending_work(backend, device) {
        cuda_ctx
            .bind_to_thread()
            .context("CUDA/DX12: bind context before surface resource sync")?;
        for context in backend.contexts.values().filter(|c| c.device == device) {
            super::cuda_context_stream_sync_after_interop(
                &cuda_ctx,
                &context.stream,
                "context stream synchronize before surface resource sync",
            )?;
        }
        if let Some(stream) = present_stream {
            super::cuda_context_stream_sync_after_interop(
                &cuda_ctx,
                &stream,
                "present stream synchronize before surface resource sync",
            )?;
        }
    }
    Ok(())
}

/// Full CUDA + DX12 drain used when destroying a surface or replacing a size-mismatched
/// scratch outside `surface_resize` (no per-surface fence high-water available yet).
fn wait_device_idle_for_surface(backend: &mut CudaBackend, device: DeviceHandle) -> Result<()> {
    let (worker, alloc_stream, present_stream, companion, cuda_ctx) = {
        let gpu = backend.device(device)?;
        (
            Arc::clone(&gpu.submission_worker),
            Arc::clone(&gpu.alloc_stream),
            gpu.dx12.as_ref().map(|c| Arc::clone(&c.present_stream)),
            gpu.dx12.as_ref().map(Arc::clone),
            Arc::clone(&gpu.ctx),
        )
    };
    // Unblock CUDA WaitExternalFence on the submission/present streams before joining
    // the worker. Otherwise flush waits forever on a Wait whose Signal sits behind a
    // stalled DX12 queue or was never issued (multi-window teardown).
    if let Some(companion) = companion.as_ref() {
        let unblock = companion.next_fence_value();
        companion
            .signal_queue(unblock)
            .context("CUDA/DX12: signal fence to unblock CUDA waits before teardown flush")?;
    }
    worker
        .flush()
        .context("CUDA/DX12: flush submission worker before surface teardown")?;
    backend.graph_stats.worker_flushes.fetch_add(1, Ordering::Relaxed);

    // DX12 before CUDA stream sync: streams may still be waiting on companion fence
    // values (scratch reuse WaitExternalFence). Waiting DX12 first unblocks them.
    if let Some(companion) = companion.as_ref() {
        companion
            .wait_idle()
            .context("CUDA/DX12: companion wait_idle before surface teardown")?;
    }

    // cudarc records failures from Drop (e.g. destroying a stream after a raced /
    // ignored context-destroy sync) into CudaContext::error_state. bind_to_thread
    // returns that sticky error via check_err before touching the driver. Drain it
    // only after DX12 is idle so subsequent stream syncs reflect real CUDA state.
    if let Err(e) = cuda_ctx.check_err() {
        tracing::debug!("CUDA/DX12: cleared sticky context error before surface teardown sync: {e:?}");
    }

    cuda_ctx
        .bind_to_thread()
        .context("CUDA/DX12: bind context before surface teardown sync")?;
    for context in backend.contexts.values().filter(|c| c.device == device) {
        super::cuda_context_stream_sync_after_interop(
            &cuda_ctx,
            &context.stream,
            "context stream synchronize before surface teardown",
        )?;
    }
    if let Some(stream) = present_stream {
        super::cuda_context_stream_sync_after_interop(
            &cuda_ctx,
            &stream,
            "present stream synchronize before surface teardown",
        )?;
    }
    super::cuda_context_stream_sync_after_interop(
        &cuda_ctx,
        &alloc_stream,
        "alloc stream synchronize before surface teardown",
    )?;
    Ok(())
}

fn present_args(mode: crate::types::PresentMode, allow_tearing: bool) -> (u32, DXGI_PRESENT) {
    match mode {
        crate::types::PresentMode::Fifo | crate::types::PresentMode::Mailbox | crate::types::PresentMode::Auto => {
            (1, DXGI_PRESENT(0))
        }
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
        DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
        Win32WindowHandle, WindowHandle, WindowsDisplayHandle,
    };
    use std::num::NonZeroIsize;
    use windows::core::w;
    use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
    use windows::Win32::UI::WindowsAndMessaging::{CreateWindowExW, DestroyWindow, CW_USEDEFAULT, WS_OVERLAPPEDWINDOW};

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
            let handle =
                Win32WindowHandle::new(NonZeroIsize::new(self.hwnd.0 as isize).ok_or(HandleError::Unavailable)?);
            Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
        }
    }

    impl HasDisplayHandle for TestHwnd {
        fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
            Ok(unsafe { DisplayHandle::borrow_raw(RawDisplayHandle::Windows(WindowsDisplayHandle::new())) })
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
        let surface = backend.create_surface(device, &window, &window, None).ok()?;
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
                assert!(msg.contains("no completion event"), "unexpected error: {msg}");
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
        let work = backend.take_present_gpu_work(token, 0).expect("take_present_gpu_work");
        let finish = work.run().expect("present run");
        let present_tv = backend.finish_present(finish, 0).expect("finish_present");
        assert!(present_tv > 0, "present must allocate a Goldy timeline value");
        // Must be waitable on the Goldy timeline (CUDA event or DX12 fence ledger entry).
        let completion = timeline::lookup_completion(&backend.context(ctx).unwrap().event_ledger, ctx, present_tv);
        assert!(
            completion.is_some(),
            "present_tv {present_tv} missing from CUDA event ledger"
        );
        backend
            .wait_until(ctx, present_tv)
            .expect("wait_until present_tv on Goldy timeline");
        match completion.unwrap() {
            LedgerCompletion::CudaEvent(event) => assert!(
                event.is_complete(),
                "present_tv {present_tv} event should complete after wait_until"
            ),
            #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
            LedgerCompletion::Dx12Fence { companion, value } => assert!(
                unsafe { companion.fence.GetCompletedValue() } >= value,
                "present_tv {present_tv} DX12 fence should complete after wait_until"
            ),
        }
    }
}
