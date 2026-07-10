//! Per-context submission stream lifecycle (DX12).

use super::types::{Dx12State, Dx12SubmissionContext};
use super::{ContextHandle, DeviceHandle};
use crate::backend::ContextDestroyHandle;
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
        .filter(|(dev, _, _)| *dev == device)
        .map(|(_, fence, _)| unsafe { fence.GetCompletedValue() })
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
    let is_warp = ld.adapter_id == super::WARP_ADAPTER_ID;

    let fence: ID3D12Fence = unsafe { ld.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .context("Failed to create per-context DX12 fence")?;

    // Own COMPUTE queue per context (see `Dx12SubmissionContext::command_queue`): a
    // GPU-side cross-context `Wait` enqueued here only ever stalls this context.
    // Graphics and present use `LogicalDevice::command_queue` (DIRECT).
    let queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
        Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
        Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
        NodeMask: 0,
    };
    let command_queue: ID3D12CommandQueue = unsafe { ld.device.CreateCommandQueue(&queue_desc) }
        .context("Failed to create per-context DX12 compute command queue")?;
    let queue_lock = std::sync::Arc::new(std::sync::Mutex::new(()));

    let compute_initial_allocator: ID3D12CommandAllocator =
        unsafe { ld.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE) }
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
    let last_submitted_seq = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

    // Register the fence in the lock-free index before inserting the context so that
    // any concurrent drain_ready_slot_reclamations sees a consistent view.
    state
        .context_fences
        .write()
        .unwrap()
        .insert(id, (device, fence.clone(), std::sync::Arc::clone(&last_submitted_seq)));
    state.contexts.write().unwrap().insert(
        id,
        std::sync::Arc::new(std::sync::Mutex::new(Dx12SubmissionContext {
            device,
            fence,
            command_queue,
            queue_lock,
            last_submitted_seq,
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
        .filter(|(_, (dev, _, _))| *dev == device)
        .map(|(h, _)| *h)
        .collect();
    result
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

/// Context to receive a user-initiated buffer/texture destroy when attribution is unambiguous.
pub(super) fn destroy_attribution_context(state: &Dx12State, device: DeviceHandle) -> Option<ContextHandle> {
    sole_context_on_device(state, device)
}

/// Deferred-deletion requirement for a destroy attributed to a single context.
///
/// Reads `last_submitted_seq` from the lock-free `context_fences` index (no `sc` mutex).
pub(super) fn reclamation_barrier_for_context(state: &Dx12State, ctx: ContextHandle) -> u64 {
    state
        .context_fences
        .read()
        .unwrap()
        .get(&ctx)
        .map(|(_, _, seq)| seq.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0)
}

/// Per-context requirement snapshot when destroy could not be attributed to one context.
///
/// Covers GPU uses that never update `slot_last_seen` (e.g. `CopyBuffer` / grant readback):
/// every live context on the device must retire its last submit before the resource is freed.
pub(super) fn reclamation_requirements_all_contexts(
    state: &Dx12State,
    device: DeviceHandle,
) -> Vec<(ContextHandle, u64)> {
    state
        .context_fences
        .read()
        .unwrap()
        .iter()
        .filter(|(_, (dev, _, _))| *dev == device)
        .map(|(h, (_, _, seq))| (*h, seq.load(std::sync::atomic::Ordering::Relaxed)))
        .collect()
}

/// Full base requirement snapshot for a buffer/texture destroy (before merging `slot_last_seen`).
pub(super) fn reclamation_requirements(
    state: &Dx12State,
    device: DeviceHandle,
    ctx_h: Option<ContextHandle>,
) -> Vec<(ContextHandle, u64)> {
    match ctx_h {
        Some(ctx) => vec![(ctx, reclamation_barrier_for_context(state, ctx))],
        None => reclamation_requirements_all_contexts(state, device),
    }
}

/// Returns the `ContextHandle` whose reclamation context is installed on the current thread.
#[allow(dead_code, reason = "reserved for thread-pinned reclamation scope (Metal parity)")]
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

/// Block until every live context on `device` has drained its own per-context command
/// queue up to its last submitted value.
///
/// Each context now owns its own `ID3D12CommandQueue` (see `Dx12SubmissionContext`), so
/// a device-level wait that only touches `LogicalDevice::command_queue`/`fence` no longer
/// observes in-flight per-context GPU work. Callers that need a true "everything on this
/// device has retired" barrier (device-wide idle wait, device teardown) must drain every
/// context's own fence explicitly first.
pub(super) fn wait_for_all_contexts_on_device(state: &Dx12State, device: DeviceHandle) {
    let fences = state.context_fences.read().unwrap();
    for ctx in contexts_on_device(state, device) {
        let Some((_, fence, seq)) = fences.get(&ctx) else {
            continue;
        };
        let seq = seq.load(std::sync::atomic::Ordering::Relaxed);
        if seq > 0 {
            let _ = super::utils::wait_for_fence(fence, seq);
        }
    }
}

pub(super) struct Dx12ContextDestroyWork {
    ctx: ContextHandle,
    sc: super::types::Dx12SubmissionContext,
    device: DeviceHandle,
    ld: super::types::SharedLogicalDevice,
    buffers: super::types::SharedBufferTable,
    context_fences:
        std::sync::Arc<std::sync::RwLock<std::collections::HashMap<ContextHandle, super::types::ContextFenceEntry>>>,
}

impl ContextDestroyHandle for Dx12ContextDestroyWork {
    fn wait(&self) -> Result<()> {
        let last_submitted = self.sc.last_submitted_seq.load(std::sync::atomic::Ordering::Relaxed);
        if last_submitted > 0 {
            super::utils::wait_for_fence(&self.sc.fence, last_submitted)?;
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        finish_destroy(self);
        Ok(())
    }
}

/// Remove `ctx` from live lookup tables.
pub(super) fn detach_for_destroy(state: &Dx12State, ctx: ContextHandle) -> Option<Dx12ContextDestroyWork> {
    let sc_arc = state.contexts.write().unwrap().remove(&ctx)?;

    // Cloned per-context handles (`ContextDeferredDeletionFlush`, `ContextReclamationScope`, …)
    // must be dropped by [`crate::Context`] before this runs; see `ContextInner::drop`.
    let sc_mutex = std::sync::Arc::try_unwrap(sc_arc)
        .unwrap_or_else(|_| panic!("context {ctx} Arc still has extra owners at destroy"));
    let sc = sc_mutex.into_inner().expect("context Mutex poisoned");

    let device = sc.device;
    let ld = state.devices.get(&device)?.clone();

    Some(Dx12ContextDestroyWork {
        ctx,
        sc,
        device,
        ld,
        buffers: std::sync::Arc::clone(&state.buffers),
        context_fences: std::sync::Arc::clone(&state.context_fences),
    })
}

fn finish_destroy(work: Box<Dx12ContextDestroyWork>) {
    let Dx12ContextDestroyWork {
        ctx,
        mut sc,
        device,
        ld,
        buffers,
        context_fences,
    } = *work;

    let completed_after_wait = unsafe { sc.fence.GetCompletedValue() };
    if completed_after_wait == u64::MAX && super::api_log::enabled() {
        let hresult: i32 = match unsafe { ld.device.GetDeviceRemovedReason() } {
            Ok(()) => 0,
            Err(e) => e.code().0,
        };
        super::api_log::log_device_removed(device, hresult);
    }

    // Only now is it safe to drop the fence out of the shared lookup table: any deferred
    // deletion elsewhere with a requirement on `ctx` uses `is_none_or` to treat a *missing*
    // context as "already retired" (see `slot_requirements_met`). Removing this earlier (before
    // the GPU-drain wait above) let concurrent drains on other threads see `ctx` as gone and
    // release resources this context's still in-flight work could still be touching.
    context_fences.write().unwrap().remove(&ctx);

    // Do not drain the device-global deletion queue here. Context destroy only waits on
    // this context's own submitted work; device-owned reclamation is serviced at
    // boundary_crossed / flush_deferred_deletions / timeline waits (see runtime §7).
    // Eagerly sweeping device deletions on context death conflates "handle gone" with
    // "physically reclaimed" and can stall / DeviceLost under multi-context sharing.

    let completed = unsafe { sc.fence.GetCompletedValue() };
    ld.retired_floor
        .fetch_max(completed, std::sync::atomic::Ordering::Relaxed);

    crate::backend::signal_fence::join_fence_poller(&sc.fence_shutdown, sc.fence_thread.take());

    {
        let mut registry = ld.descriptors.lock().unwrap();
        for (_, old) in sc.retained_graphs.drain() {
            registry.unpin_retained_slots(old.used_slots.clone());
            if let Some(row) = old.frame_table_row {
                super::frame_table::unpin_row(&sc.frame_table, row);
            }
            if old.on_device_queue {
                if let Some(slot) = ld.device_direct_pool.lock().unwrap().get_mut(old.slot_idx) {
                    slot.retained = false;
                }
            } else if let Some(slot) = sc.compute_allocator_pool.get_mut(old.slot_idx) {
                slot.retained = false;
            }
        }
    }

    // GPU is idle for this context (waited before detach); destroy per-context staging resources.
    unsafe { sc.staging_belt.destroy_all() };
    unsafe { sc.texture_staging_pool.destroy_all() };

    // Drain any remaining per-context pending deletions now that the GPU is idle.
    let batch = sc.deletion_queue.drain_everything();
    if !batch.is_empty() {
        let mut registry = ld.descriptors.lock().unwrap();
        for resource in batch {
            super::types::destroy_pending_deletion(&ld, &mut registry, resource, Vec::new());
        }
    }

    super::frame_table::destroy_context_resources(&buffers, &ld, &sc.frame_table);

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
        // Per-context queue entries are already gated on this context's fence; no
        // cross-context requirement set is attached at enqueue time.
        super::types::destroy_pending_deletion(ld, &mut registry, resource, Vec::new());
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
