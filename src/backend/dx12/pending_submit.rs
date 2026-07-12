//! Async GPU submission work enqueued on the per-device submission worker.

use super::host_wait::HostWait;
use super::types::{ContextFenceEntry, LogicalDevice, SharedBufferTable, SharedLogicalDevice, SharedSubmissionContext};
use super::ContextHandle;
use crate::backend::submission_worker::PendingSubmit;
use crate::backend::{DeferredHostWrite, SubmitSync};
use crate::timeline::TimelineValue;
use anyhow::{Context as _, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandList, ID3D12CommandQueue, ID3D12Fence};

pub(super) fn resolve_queue_waits(
    _ld: &LogicalDevice,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, ContextFenceEntry>>>,
    sync: Option<&SubmitSync>,
) -> Result<Vec<(ID3D12Fence, u64)>> {
    let Some(s) = sync else {
        return Ok(Vec::new());
    };
    let mut waits = Vec::with_capacity(s.waits.len());
    for epoch in &s.waits {
        let fences = context_fences.read().unwrap();
        let (_, producer_fence, _) = fences
            .get(&epoch.context)
            .with_context(|| format!("cross-submit wait: unknown producer context {:?}", epoch.context))?;
        waits.push((producer_fence.clone(), epoch.value));
    }
    Ok(waits)
}

pub(super) fn resolve_host_observed_waits(
    context_fences: &Arc<RwLock<HashMap<ContextHandle, ContextFenceEntry>>>,
    sync: Option<&SubmitSync>,
) -> Result<Vec<HostWait>> {
    let Some(s) = sync else {
        return Ok(Vec::new());
    };
    let mut waits = Vec::with_capacity(s.host_observed_waits.len());
    let fences = context_fences.read().unwrap();
    for epoch in &s.host_observed_waits {
        let (_, producer_fence, _) = fences
            .get(&epoch.context)
            .with_context(|| format!("host-observed wait: unknown context {:?}", epoch.context))?;
        waits.push(HostWait::Fence {
            fence: producer_fence.clone(),
            value: epoch.value,
        });
    }
    Ok(waits)
}

fn apply_deferred_host_writes(buffers: &SharedBufferTable, deferred_writes: &[DeferredHostWrite]) -> Result<()> {
    for w in deferred_writes {
        let buffers_read = buffers.read().unwrap();
        let buffer = buffers_read
            .entries
            .get(&w.buffer)
            .with_context(|| format!("deferred host write: invalid buffer handle {}", w.buffer))?;
        if let Some(base) = buffer.cpu_writable_upload_mapped {
            unsafe {
                std::ptr::copy_nonoverlapping(w.data.as_ptr(), (base as *mut u8).add(w.offset as usize), w.data.len());
            }
        } else {
            anyhow::bail!(
                "deferred host write requires CPU-writable mapped buffer (handle={})",
                w.buffer
            );
        }
    }
    Ok(())
}

fn apply_host_sidecar_before_gpu(
    host_observed_waits: &[HostWait],
    buffers: &SharedBufferTable,
    deferred_writes: &[DeferredHostWrite],
) -> Result<()> {
    let _tz = crate::tracy_zone!("goldy.dx12.pending_submit.apply_host_sidecar_before_gpu");
    for wait in host_observed_waits {
        wait.wait()?;
    }
    apply_deferred_host_writes(buffers, deferred_writes)
}

pub(super) struct Dx12ComputePendingSubmit {
    logical_device: SharedLogicalDevice,
    sc: SharedSubmissionContext,
    queue: ID3D12CommandQueue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    ctx_fence: ID3D12Fence,
    slot_idx: usize,
    command_lists: Vec<Option<ID3D12CommandList>>,
    queue_waits: Vec<(ID3D12Fence, u64)>,
    host_observed_waits: Vec<HostWait>,
    deferred_host_writes: Vec<DeferredHostWrite>,
    buffers: SharedBufferTable,
    fence_value: TimelineValue,
}

impl PendingSubmit for Dx12ComputePendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.compute");
        apply_host_sidecar_before_gpu(&self.host_observed_waits, &self.buffers, &self.deferred_host_writes)?;
        {
            let _tz = crate::tracy_zone!("dx12.submit_worker.pre_reset_slots.before");
            let mut sc = self.sc.lock().unwrap();
            super::compute::pre_reset_retired_compute_slots(&self.logical_device, &mut sc, &self.ctx_fence)?;
        }
        super::utils::execute_preallocated_context_submit(
            &self.logical_device,
            &self.queue,
            &self.queue_lock,
            &self.ctx_fence,
            &self.command_lists,
            &self.queue_waits,
            self.fence_value,
        )?;
        {
            let _tz = crate::tracy_zone!("dx12.submit_worker.pre_reset_slots.after");
            let mut sc = self.sc.lock().unwrap();
            super::compute::finish_compute_slot_submit(&self.logical_device, &mut sc, &self.ctx_fence, self.slot_idx)?;
        }
        Ok(())
    }
}

pub(super) struct Dx12RetainedResubmitPending {
    logical_device: SharedLogicalDevice,
    queue: ID3D12CommandQueue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    queue_waits: Vec<(ID3D12Fence, u64)>,
    host_observed_waits: Vec<HostWait>,
    deferred_host_writes: Vec<DeferredHostWrite>,
    buffers: SharedBufferTable,
    fence_value: TimelineValue,
}

impl PendingSubmit for Dx12RetainedResubmitPending {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.retained_resubmit");
        apply_host_sidecar_before_gpu(&self.host_observed_waits, &self.buffers, &self.deferred_host_writes)?;
        {
            let ctx_completed = unsafe { self.ctx_fence.GetCompletedValue() };
            let device_completed = unsafe { self.logical_device.fence.GetCompletedValue() };
            if ctx_completed == u64::MAX || device_completed == u64::MAX {
                let completed = ctx_completed.max(device_completed);
                super::diagnostic::first_touch_device_removed(
                    &self.logical_device.device,
                    &self.logical_device.device_removed,
                    "dx12::Dx12RetainedResubmitPending::execute",
                    self.fence_value,
                    completed,
                );
                anyhow::bail!("GPU device removed before retained resubmit tv={}", self.fence_value);
            }
        }
        super::utils::execute_preallocated_context_submit(
            &self.logical_device,
            &self.queue,
            &self.queue_lock,
            &self.ctx_fence,
            &self.command_lists,
            &self.queue_waits,
            self.fence_value,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue_compute_submit(
    logical_device: &SharedLogicalDevice,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, ContextFenceEntry>>>,
    buffers: &SharedBufferTable,
    sc: SharedSubmissionContext,
    queue: ID3D12CommandQueue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    ctx_fence: ID3D12Fence,
    slot_idx: usize,
    command_lists: Vec<Option<ID3D12CommandList>>,
    sync: Option<&SubmitSync>,
    fence_value: TimelineValue,
) -> Result<()> {
    logical_device.submission_worker.check_error()?;
    let queue_waits = resolve_queue_waits(logical_device, context_fences, sync)?;
    let host_observed_waits = resolve_host_observed_waits(context_fences, sync)?;
    let deferred_host_writes = sync.map(|s| s.deferred_host_writes.clone()).unwrap_or_default();
    logical_device.submission_worker.enqueue(
        fence_value,
        Box::new(Dx12ComputePendingSubmit {
            logical_device: Arc::clone(logical_device),
            sc,
            queue,
            queue_lock,
            ctx_fence,
            slot_idx,
            command_lists,
            queue_waits,
            host_observed_waits,
            deferred_host_writes,
            buffers: Arc::clone(buffers),
            fence_value,
        }),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue_retained_resubmit(
    logical_device: &SharedLogicalDevice,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, ContextFenceEntry>>>,
    buffers: &SharedBufferTable,
    queue: ID3D12CommandQueue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    sync: Option<&SubmitSync>,
    fence_value: TimelineValue,
) -> Result<()> {
    logical_device.submission_worker.check_error()?;
    let queue_waits = resolve_queue_waits(logical_device, context_fences, sync)?;
    let host_observed_waits = resolve_host_observed_waits(context_fences, sync)?;
    let deferred_host_writes = sync.map(|s| s.deferred_host_writes.clone()).unwrap_or_default();
    logical_device.submission_worker.enqueue(
        fence_value,
        Box::new(Dx12RetainedResubmitPending {
            logical_device: Arc::clone(logical_device),
            queue,
            queue_lock,
            ctx_fence,
            command_lists,
            queue_waits,
            host_observed_waits,
            deferred_host_writes,
            buffers: Arc::clone(buffers),
            fence_value,
        }),
    )
}
