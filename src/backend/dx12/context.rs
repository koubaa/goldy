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
        .map(|d| d.retired_floor.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0);
    // Use context_fences for a lock-free scan over per-context fence completion values.
    let fences = state.context_fences.read().unwrap();
    let max_ctx = fences
        .values()
        .filter(|(dev, _)| *dev == device)
        .map(|(_, fence)| unsafe { fence.GetCompletedValue() })
        .max()
        .unwrap_or(0);
    drop(fences);
    let device_sync = state
        .devices
        .get(&device)
        .map(|d| unsafe { d.fence.GetCompletedValue() })
        .unwrap_or(0);
    floor.max(max_ctx).max(device_sync)
}

pub(super) fn create(state: &mut Dx12State, device: DeviceHandle) -> Result<ContextHandle> {
    let ld = state.devices.get(&device).context("Invalid device handle")?;

    let fence: ID3D12Fence = unsafe { ld.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .context("Failed to create per-context DX12 fence")?;

    let compute_initial_allocator: ID3D12CommandAllocator =
        unsafe { ld.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
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
            gpu_completed: std::sync::Arc::new(move || unsafe { fence_for_poll.GetCompletedValue() }),
        },
    ));

    let id = state.next_context_id;
    state.next_context_id = state.next_context_id.saturating_add(1);

    // Register the fence in the lock-free index before inserting the context so that
    // any concurrent drain_ready_slot_reclamations sees a consistent view.
    state
        .context_fences
        .write()
        .unwrap()
        .insert(id, (device, fence.clone()));
    state.contexts.write().unwrap().insert(
        id,
        std::sync::Arc::new(std::sync::Mutex::new(Dx12SubmissionContext {
            device,
            fence,
            last_submitted_seq: 0,
            signal_queue,
            fence_shutdown,
            fence_thread,
            compute_allocator_pool,
            retained_graphs: std::collections::HashMap::new(),
            staging_belt: super::staging::StagingBelt::new(super::staging::DEFAULT_STAGING_CHUNK_SIZE),
            texture_staging_pool: super::staging::TextureStagingPool::new(),
            deletion_queue: super::types::DeletionQueue::new(),
        })),
    );
    Ok(id)
}

pub(super) fn destroy(state: &mut Dx12State, ctx: ContextHandle) {
    // Snapshot in-flight work before remove. Async pending submits hold `Arc<Mutex<sc>>`;
    // flush the worker so those refs drop before `try_unwrap` below.
    let drain = {
        let contexts = state.contexts.read().unwrap();
        let Some(sc_arc) = contexts.get(&ctx) else {
            return;
        };
        let sc = sc_arc.lock().expect("context Mutex poisoned");
        (sc.device, sc.last_submitted_seq, sc.fence.clone())
    };
    let (device, last_submitted_seq, ctx_fence) = drain;
    if let Some(ld) = state.devices.get(&device) {
        match ld.submission_worker.flush() {
            Ok(()) if last_submitted_seq > 0 => {
                if let Err(e) = ld.submission_worker.wait_submitted(last_submitted_seq) {
                    tracing::warn!("context {ctx} destroy: wait_submitted failed: {e:#}");
                } else {
                    let submitted = ld
                        .submission_worker
                        .submitted_epoch()
                        .load(std::sync::atomic::Ordering::Acquire);
                    let fence_wait = last_submitted_seq.min(submitted);
                    if fence_wait > 0 {
                        let _ = super::utils::wait_for_fence(&ctx_fence, fence_wait);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("context {ctx} destroy: submission worker flush failed: {e:#}");
            }
            Ok(()) => {}
        }
    }

    // Remove from both maps; the fence index must not outlive the context.
    let Some(sc_arc) = state.contexts.write().unwrap().remove(&ctx) else {
        return;
    };
    state.context_fences.write().unwrap().remove(&ctx);

    // Cloned per-context handles (`ContextTimelineReader`, `ContextDeferredDeletionFlush`, …)
    // must be dropped by [`crate::Context`] before this runs; see `ContextInner::drop`.
    let sc_mutex = std::sync::Arc::try_unwrap(sc_arc)
        .unwrap_or_else(|arc| {
            panic!(
                "context {ctx} Arc still has {} extra owners at destroy",
                std::sync::Arc::strong_count(&arc).saturating_sub(1)
            )
        });
    let mut sc = sc_mutex.into_inner().expect("context Mutex poisoned");

    let completed = unsafe { sc.fence.GetCompletedValue() };
    if let Some(ld) = state.devices.get(&device) {
        ld.retired_floor
            .fetch_max(completed, std::sync::atomic::Ordering::Relaxed);
    }

    crate::backend::signal_fence::join_fence_poller(&sc.fence_shutdown, sc.fence_thread.take());

    for old in sc.retained_graphs.drain().map(|(_, g)| g) {
        if let Some(row) = old.frame_table_row {
            if let Some(ft) = state.frame_tables.read().unwrap().get(&device) {
                super::frame_table::unpin_row(ft, row);
            }
        }
        if let Some(slot) = sc.compute_allocator_pool.get_mut(old.slot_idx) {
            slot.retained = false;
        }
    }

    // GPU is idle for this context (waited above); destroy per-context staging resources.
    unsafe { sc.staging_belt.destroy_all() };
    unsafe { sc.texture_staging_pool.destroy_all() };

    // Drain any remaining per-context pending deletions now that the GPU is idle.
    let batch = sc.deletion_queue.drain_everything();
    if !batch.is_empty() {
        if let Some(ld) = state.devices.get(&device) {
            let descriptors_arc = std::sync::Arc::clone(&ld.descriptors);
            let mut registry = descriptors_arc.lock().unwrap();
            for resource in batch {
                super::types::destroy_pending_deletion(ld, &mut registry, resource);
            }
        }
    }
}

pub(super) fn context_device(state: &Dx12State, ctx: ContextHandle) -> DeviceHandle {
    state
        .contexts
        .read()
        .unwrap()
        .get(&ctx)
        .expect("invalid context handle")
        .lock()
        .expect("context Mutex poisoned")
        .device
}
