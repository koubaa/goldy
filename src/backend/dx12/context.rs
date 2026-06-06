//! Per-context submission stream lifecycle (DX12).

use super::types::{Dx12State, Dx12SubmissionContext};
use super::{ContextHandle, DeviceHandle};
use anyhow::{Context as _, Result};
use windows::Win32::Graphics::Direct3D12::*;

/// Latest device-global seq retired on `device` (max over live context fences, device sync fence, floored).
pub(super) fn device_retired(state: &Dx12State, device: DeviceHandle) -> u64 {
    let floor = state
        .devices
        .get(&device)
        .map(|d| d.retired_floor)
        .unwrap_or(0);
    let max_ctx = state
        .contexts
        .values()
        .filter(|c| c.device == device)
        .map(|c| unsafe { c.fence.GetCompletedValue() })
        .max()
        .unwrap_or(0);
    let device_sync = state
        .devices
        .get(&device)
        .map(|d| unsafe { d.fence.GetCompletedValue() })
        .unwrap_or(0);
    floor.max(max_ctx).max(device_sync)
}

pub(super) fn create(state: &mut Dx12State, device: DeviceHandle) -> Result<ContextHandle> {
    let ld = state
        .devices
        .get(&device)
        .context("Invalid device handle")?;

    let fence: ID3D12Fence = unsafe { ld.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .context("Failed to create per-context DX12 fence")?;

    let compute_initial_allocator: ID3D12CommandAllocator = unsafe {
        ld.device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
    }
    .context("Failed to create per-context compute command allocator")?;
    let compute_allocator_pool = vec![super::types::ComputeAllocatorSlot {
        allocator: compute_initial_allocator,
        fence_value: 0,
        command_list: None,
        retained: false,
    }];

    let signal_queue = std::sync::Arc::new(crate::signal::SignalQueue::new());
    let fence_shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let fence_for_poll = fence.clone();
    let signal_queue_poll = std::sync::Arc::clone(&signal_queue);
    let shutdown_poll = std::sync::Arc::clone(&fence_shutdown);
    let fence_thread = Some(crate::backend::signal_fence::spawn_fence_poller(
        crate::backend::signal_fence::FencePollerState {
            shutdown: shutdown_poll,
            signal_queue: signal_queue_poll,
            last_emitted_epoch: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            gpu_completed: std::sync::Arc::new(move || unsafe {
                fence_for_poll.GetCompletedValue()
            }),
        },
    ));

    let id = state.next_context_id;
    state.next_context_id = state.next_context_id.saturating_add(1);
    state.contexts.insert(
        id,
        Dx12SubmissionContext {
            device,
            fence,
            last_submitted_seq: 0,
            signal_queue,
            fence_shutdown,
            fence_thread,
            compute_allocator_pool,
            retained_graph: None,
            staging_belt: super::staging::StagingBelt::new(
                super::staging::DEFAULT_STAGING_CHUNK_SIZE,
            ),
            texture_staging_pool: super::staging::TextureStagingPool::new(),
        },
    );
    Ok(id)
}

pub(super) fn destroy(state: &mut Dx12State, ctx: ContextHandle) {
    let Some(mut sc) = state.contexts.remove(&ctx) else {
        return;
    };
    let device = sc.device;

    // Drain in-flight GPU work before releasing command allocators / retained CLs.
    if sc.last_submitted_seq > 0 {
        let _ = super::utils::wait_for_fence(&sc.fence, sc.last_submitted_seq);
    }

    let completed = unsafe { sc.fence.GetCompletedValue() };
    if let Some(ld) = state.devices.get_mut(&device) {
        ld.retired_floor = ld.retired_floor.max(completed);
    }

    crate::backend::signal_fence::join_fence_poller(&sc.fence_shutdown, sc.fence_thread.take());

    if let Some(old) = sc.retained_graph.take() {
        if let Some(slot) = sc.compute_allocator_pool.get_mut(old.slot_idx) {
            slot.retained = false;
        }
    }

    // GPU is idle for this context (waited above); destroy per-context staging resources.
    unsafe { sc.staging_belt.destroy_all() };
    unsafe { sc.texture_staging_pool.destroy_all() };
}

pub(super) fn context_device(state: &Dx12State, ctx: ContextHandle) -> DeviceHandle {
    state
        .contexts
        .get(&ctx)
        .expect("invalid context handle")
        .device
}
