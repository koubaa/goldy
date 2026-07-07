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
    let retired = floor.max(max_ctx).max(device_sync);
    if retired == u64::MAX {
        if let Some(ld) = state.devices.get(&device) {
            super::diagnostic::first_touch_device_removed(
                &ld.device,
                &state.device_removed,
                "dx12::context::device_retired",
                0,
                retired,
            );
        }
    }
    retired
}

pub(super) fn create(state: &mut Dx12State, device: DeviceHandle) -> Result<ContextHandle> {
    let ld = state.devices.get(&device).context("Invalid device handle")?.clone();

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

    let frame_table = super::frame_table::init_context(state, device, &ld)?;

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
            frame_table,
            reclamation_context: None,
        })),
    );
    if super::api_log::enabled() {
        super::api_log::log_context_create(device, id, is_warp);
    }
    Ok(id)
}

/// Context handles live on `device`, filtered via the lock-free `context_fences` map
/// (keyed by `DeviceHandle` without requiring any `sc_arc` mutex acquisition).
///
/// Critical for avoiding cross-test/cross-device lock contention: `state.contexts` is a
/// single global map shared by every `Device` in the process, so scanning it and locking
/// every `sc_arc` just to filter by device would serialize buffer destroys against
/// unrelated, concurrently-running tests' in-flight GPU waits (see issue: `cargo test`
/// runs tests in parallel by default, each with its own `Device`/contexts).
fn contexts_on_device(state: &Dx12State, device: DeviceHandle) -> Vec<ContextHandle> {
    let result: Vec<ContextHandle> = state
        .context_fences
        .read()
        .unwrap()
        .iter()
        .filter(|(_, (dev, _))| *dev == device)
        .map(|(h, _)| *h)
        .collect();
    result
}

/// Returns the `ContextHandle` whose reclamation context is installed on the current thread.
pub(super) fn context_handle_for_thread(state: &Dx12State, device: DeviceHandle) -> Option<ContextHandle> {
    let thread = std::thread::current().id();
    let contexts = state.contexts.read().unwrap();
    contexts_on_device(state, device).into_iter().find_map(|h| {
        let sc_arc = contexts.get(&h)?;
        let sc = sc_arc.lock().expect("context Mutex poisoned");
        if let Some((t, _)) = sc.reclamation_context {
            if t == thread {
                return Some(h);
            }
        }
        None
    })
}

/// When exactly one live context exists on `device`, return it for destroy attribution.
pub(super) fn sole_context_on_device(state: &Dx12State, device: DeviceHandle) -> Option<ContextHandle> {
    let mut on_device = contexts_on_device(state, device);
    if on_device.len() == 1 {
        on_device.pop()
    } else {
        None
    }
}

/// Context to receive a user-initiated buffer destroy (thread reclamation scope, else sole context).
pub(super) fn destroy_attribution_context(state: &Dx12State, device: DeviceHandle) -> Option<ContextHandle> {
    context_handle_for_thread(state, device).or_else(|| sole_context_on_device(state, device))
}

/// Fence barrier for deferred buffer destruction on `device`.
pub(super) fn reclamation_barrier(state: &Dx12State, device: DeviceHandle) -> u64 {
    let thread = std::thread::current().id();
    let candidates = contexts_on_device(state, device);
    if !candidates.is_empty() {
        let contexts = state.contexts.read().unwrap();
        for h in candidates {
            if let Some(sc_arc) = contexts.get(&h) {
                let sc = sc_arc.lock().expect("context Mutex poisoned");
                if let Some((t, epoch)) = sc.reclamation_context {
                    if t == thread {
                        return epoch;
                    }
                }
            }
        }
    }
    state
        .devices
        .get(&device)
        .map(|d| {
            let timeline_next = d.timeline_next.load(std::sync::atomic::Ordering::Relaxed);
            timeline_next.saturating_sub(1)
        })
        .unwrap_or(0)
}

pub(super) fn destroy(state: &mut Dx12State, ctx: ContextHandle) {
    // Remove from both maps first; the fence index must not outlive the context.
    let Some(sc_arc) = state.contexts.write().unwrap().remove(&ctx) else {
        return;
    };
    state.context_fences.write().unwrap().remove(&ctx);

    // Cloned per-context handles (`ContextTimelineReader`, `ContextDeferredDeletionFlush`, …)
    // must be dropped by [`crate::Context`] before this runs; see `ContextInner::drop`.
    let sc_mutex = std::sync::Arc::try_unwrap(sc_arc)
        .unwrap_or_else(|_| panic!("context {ctx} Arc still has extra owners at destroy"));
    let mut sc = sc_mutex.into_inner().expect("context Mutex poisoned");

    let device = sc.device;

    // Drain in-flight GPU work before releasing command allocators / retained CLs.
    if sc.last_submitted_seq > 0 {
        if super::api_log::enabled() {
            super::api_log::log_fence_wait_cpu("context_destroy", sc.last_submitted_seq);
        }
        let _ = super::utils::wait_for_fence(&sc.fence, sc.last_submitted_seq);
    }

    let completed = unsafe { sc.fence.GetCompletedValue() };
    if let Some(ld) = state.devices.get(&device) {
        ld.retired_floor
            .fetch_max(completed, std::sync::atomic::Ordering::Relaxed);
    }

    crate::backend::signal_fence::join_fence_poller(&sc.fence_shutdown, sc.fence_thread.take());

    for old in sc.retained_graphs.drain().map(|(_, g)| g) {
        if let Some(row) = old.frame_table_row {
            super::frame_table::unpin_row(&sc.frame_table, row);
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

    super::frame_table::destroy_context(state, device, &sc.frame_table);

    if super::api_log::enabled() {
        super::api_log::log_context_destroy(device, ctx);
    }
}

/// Drain per-context deferred deletions retired up to `completed` on the render/wait thread.
pub(super) fn drain_context_deletion_queue_up_to(
    ld: &super::types::LogicalDevice,
    sc: &mut super::types::Dx12SubmissionContext,
    completed: u64,
) {
    let batch = sc.deletion_queue.drain_up_to_completed(completed);
    if batch.is_empty() {
        return;
    }
    let mut registry = ld.descriptors.lock().unwrap();
    for resource in batch {
        super::types::destroy_pending_deletion(ld, &mut registry, resource);
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
