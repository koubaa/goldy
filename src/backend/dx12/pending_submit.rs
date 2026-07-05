//! Async GPU submission work enqueued on the per-device submission worker.

use super::host_wait::HostWait;
use super::types::{LogicalDevice, SharedBufferTable, SharedLogicalDevice, SharedSubmissionContext};
use super::{ContextHandle, DeviceHandle};
use crate::backend::dx12::tv::{CtxTv, DeviceTv};
use crate::backend::submission_worker::{PendingSubmit, SubmissionWorker};
use crate::backend::{DeferredHostWrite, SubmitSync};
use anyhow::{Context as _, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandList, ID3D12CommandQueue, ID3D12Fence};

pub(super) fn resolve_queue_waits(
    logical_device: &LogicalDevice,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
    sync: Option<&SubmitSync>,
) -> Result<Vec<(ID3D12Fence, u64)>> {
    let Some(s) = sync else {
        return Ok(Vec::new());
    };
    let mut waits = Vec::with_capacity(s.waits.len() + s.device_queue_waits.len());
    for epoch in &s.waits {
        let fences = context_fences.read().unwrap();
        let (_, producer_fence) = fences
            .get(&epoch.context)
            .with_context(|| format!("cross-submit wait: unknown producer context {:?}", epoch.context))?;
        waits.push((producer_fence.clone(), epoch.value));
    }
    if !s.device_queue_waits.is_empty() {
        let device_fence = logical_device.fence.clone();
        for tv in &s.device_queue_waits {
            waits.push((device_fence.clone(), *tv));
        }
    }
    Ok(waits)
}

pub(super) fn resolve_host_observed_waits(
    context_fences: &Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
    sync: Option<&SubmitSync>,
) -> Result<Vec<HostWait>> {
    let Some(s) = sync else {
        return Ok(Vec::new());
    };
    let mut waits = Vec::with_capacity(s.host_observed_waits.len());
    let fences = context_fences.read().unwrap();
    for epoch in &s.host_observed_waits {
        let (_, producer_fence) = fences
            .get(&epoch.context)
            .with_context(|| format!("host-observed wait: unknown context {:?}", epoch.context))?;
        waits.push(HostWait::Fence {
            fence: producer_fence.clone(),
            value: epoch.value,
        });
    }
    Ok(waits)
}

fn validate_deferred_host_writes(buffers: &SharedBufferTable, deferred_writes: &[DeferredHostWrite]) -> Result<()> {
    if deferred_writes.is_empty() {
        return Ok(());
    }
    let buffers_read = buffers.read().unwrap();
    for w in deferred_writes {
        let buffer = buffers_read
            .entries
            .get(&w.buffer)
            .with_context(|| format!("deferred host write: invalid buffer handle {}", w.buffer))?;
        if buffer.cpu_writable_upload_mapped.is_none() {
            anyhow::bail!(
                "deferred host write requires CPU-writable mapped buffer (handle={})",
                w.buffer
            );
        }
    }
    Ok(())
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
    command_queue: ID3D12CommandQueue,
    queue_lock: Arc<Mutex<()>>,
    sc: SharedSubmissionContext,
    ctx_fence: ID3D12Fence,
    slot_idx: usize,
    command_lists: Vec<Option<ID3D12CommandList>>,
    queue_waits: Vec<(ID3D12Fence, u64)>,
    host_observed_waits: Vec<HostWait>,
    deferred_host_writes: Vec<DeferredHostWrite>,
    buffers: SharedBufferTable,
    fence_value: CtxTv,
}

impl PendingSubmit for Dx12ComputePendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.compute");
        apply_host_sidecar_before_gpu(&self.host_observed_waits, &self.buffers, &self.deferred_host_writes)?;
        let command_lists = self.command_lists;
        {
            let _tz = crate::tracy_zone!("dx12.submit_worker.pre_reset_slots.before");
            let mut sc = self.sc.lock().unwrap();
            super::compute::pre_reset_retired_compute_slots(&self.logical_device, &mut sc, &self.ctx_fence, None)?;
        }
        {
            let ctx_completed = unsafe { self.ctx_fence.GetCompletedValue() };
            let device_completed = unsafe { self.logical_device.fence.GetCompletedValue() };
            if ctx_completed == u64::MAX || device_completed == u64::MAX {
                let completed = ctx_completed.max(device_completed);
                super::diagnostic::first_touch_device_removed(
                    &self.logical_device.device,
                    &self.logical_device.device_removed,
                    "dx12::Dx12ComputePendingSubmit::execute",
                    self.fence_value.raw(),
                    completed,
                );
                anyhow::bail!("GPU device removed before compute submit tv={}", self.fence_value.raw());
            }
        }
        super::utils::execute_preallocated_context_submit(
            &self.command_queue,
            &self.queue_lock,
            &self.ctx_fence,
            &command_lists,
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
    command_queue: ID3D12CommandQueue,
    queue_lock: Arc<Mutex<()>>,
    sc: SharedSubmissionContext,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    queue_waits: Vec<(ID3D12Fence, u64)>,
    host_observed_waits: Vec<HostWait>,
    deferred_host_writes: Vec<DeferredHostWrite>,
    buffers: SharedBufferTable,
    fence_value: CtxTv,
}

impl PendingSubmit for Dx12RetainedResubmitPending {
    fn execute(self: Box<Self>) -> Result<()> {
        // Hold the submission context until this job completes (retained graphs reference it).
        let _sc = &self.sc;
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.retained_resubmit");
        apply_host_sidecar_before_gpu(&self.host_observed_waits, &self.buffers, &self.deferred_host_writes)?;
        let command_lists = self.command_lists;
        {
            let ctx_completed = unsafe { self.ctx_fence.GetCompletedValue() };
            let device_completed = unsafe { self.logical_device.fence.GetCompletedValue() };
            if ctx_completed == u64::MAX || device_completed == u64::MAX {
                let completed = ctx_completed.max(device_completed);
                super::diagnostic::first_touch_device_removed(
                    &self.logical_device.device,
                    &self.logical_device.device_removed,
                    "dx12::Dx12RetainedResubmitPending::execute",
                    self.fence_value.raw(),
                    completed,
                );
                anyhow::bail!("GPU device removed before retained resubmit tv={}", self.fence_value.raw());
            }
        }
        let result = super::utils::execute_preallocated_context_submit(
            &self.command_queue,
            &self.queue_lock,
            &self.ctx_fence,
            &command_lists,
            &self.queue_waits,
            self.fence_value,
        );
        result
    }
}

/// Copy blit (optional) + queue signal enqueued at scheme submit. WSI present runs at
/// grant consumption via [`Dx12ScheduledPresentJob`].
pub(super) struct Dx12PresentCopyPendingSubmit {
    logical_device: SharedLogicalDevice,
    command_lists: Vec<Option<ID3D12CommandList>>,
    copy_tv: DeviceTv,
    /// When zero, signals `present_only_tv` instead of executing command lists.
    present_only_tv: DeviceTv,
}

impl PendingSubmit for Dx12PresentCopyPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.scheduled_present_copy");
        if !self.command_lists.is_empty() {
            super::utils::signal_preallocated_device(&self.logical_device, &self.command_lists, self.copy_tv)?;
        } else if self.present_only_tv.raw() > 0 {
            super::utils::with_queue_lock(&self.logical_device.queue_lock, || {
                unsafe {
                    self.logical_device
                        .command_queue
                        .Signal(&self.logical_device.fence, self.present_only_tv.raw())
                }
                .context("Failed to signal device fence for scheduled present copy epoch")
            })?;
        }
        Ok(())
    }
}

/// Ensures present lifecycle hooks fire on every consume attempt, including error paths.
struct PresentLifecycleGuard {
    on_began: Option<Box<dyn FnOnce() + Send>>,
    on_completed: Option<Box<dyn FnOnce() + Send>>,
    began_published: bool,
}

impl PresentLifecycleGuard {
    fn new(hooks: crate::backend::PresentConsumeHooks) -> Self {
        Self {
            on_began: hooks.on_present_began,
            on_completed: hooks.on_present_completed,
            began_published: false,
        }
    }

    fn signal_began(&mut self) {
        if !self.began_published {
            if let Some(on_began) = self.on_began.take() {
                on_began();
            }
            self.began_published = true;
        }
    }
}

impl Drop for PresentLifecycleGuard {
    fn drop(&mut self) {
        if !self.began_published {
            if let Some(on_began) = self.on_began.take() {
                on_began();
            }
        }
        if let Some(on_completed) = self.on_completed.take() {
            on_completed();
        }
    }
}

/// WSI present deferred to [`PresentGrant::consume`] on TID_PRESENT.
pub(super) struct Dx12ScheduledPresentJob {
    logical_device: SharedLogicalDevice,
    copy_tv: DeviceTv,
    issue_token: Option<std::sync::Arc<crate::backend::submission_worker::SubmissionJobToken>>,
    sc: Option<SharedSubmissionContext>,
    /// Device-fence value to wait on before advancing the context fence.
    sync_tv: DeviceTv,
    /// Context-fence value to signal after `sync_tv` retires on the device queue.
    ctx_sync_tv: CtxTv,
    swapchain: windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
    present_mode: crate::types::PresentMode,
    allow_tearing: bool,
    finish: crate::backend::PresentFinishState,
    pending_finishes: std::sync::Arc<std::sync::Mutex<Vec<crate::backend::PresentFinishState>>>,
    device_removed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl crate::backend::ScheduledPresentJob for Dx12ScheduledPresentJob {
    fn copy_tv(&self) -> crate::timeline::TimelineValue {
        self.copy_tv.to_public()
    }

    fn run(self: Box<Self>, hooks: crate::backend::PresentConsumeHooks) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.dx12.scheduled_present_job.run");
        let mut lifecycle = PresentLifecycleGuard::new(hooks);
        let Dx12ScheduledPresentJob {
            logical_device,
            copy_tv,
            issue_token,
            sc,
            sync_tv,
            ctx_sync_tv,
            swapchain,
            present_mode,
            allow_tearing,
            finish,
            pending_finishes,
            device_removed,
        } = *self;

        if let Some(token) = issue_token {
            if !token.is_done() {
                token
                    .wait()
                    .context("scheduled present job: copy submit issue wait failed")?;
            }
        } else {
            logical_device
                .submission_worker
                .wait_submitted(copy_tv.raw())
                .context("scheduled present job: copy submit epoch wait failed")?;
        }
        logical_device.submission_worker.check_error()?;
        lifecycle.signal_began();

        {
            let _tz = crate::tracy_zone!("dx12.present.swapchain_present");
            let (sync_interval, present_flags) = super::surface::present_args(present_mode, allow_tearing);
            let hr = unsafe { swapchain.Present(sync_interval, present_flags) };
            if hr.is_err() {
                return Err(super::utils::map_d3d12_hresult_failure(
                    &logical_device.device,
                    &device_removed,
                    hr,
                    "Present failed",
                ));
            }
        }

        if let Some(sc) = sc {
            if sync_tv != DeviceTv::ZERO && ctx_sync_tv != CtxTv::ZERO {
                let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.ctx_fence_sync");
                super::utils::sync_context_fence_after_device_retire(
                    &sc,
                    &logical_device.fence,
                    sync_tv,
                    ctx_sync_tv,
                )?;
            }
        }

        pending_finishes.lock().unwrap().push(finish);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue_scheduled_present(
    logical_device: &SharedLogicalDevice,
    command_lists: Vec<Option<ID3D12CommandList>>,
    copy_tv: DeviceTv,
    sc: Option<SharedSubmissionContext>,
    sync_tv: DeviceTv,
    ctx_sync_tv: CtxTv,
    swapchain: windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
    present_mode: crate::types::PresentMode,
    allow_tearing: bool,
    finish: crate::backend::PresentFinishState,
    pending_finishes: std::sync::Arc<std::sync::Mutex<Vec<crate::backend::PresentFinishState>>>,
    device_removed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    // When `copy_tv == 0` (skip-copy / render-pass-direct), the caller must pre-allocate
    // the present-only queue signal so eager bookkeeping and enqueue share one timeline value.
    preallocated_present_tv: Option<DeviceTv>,
    issue_token: Option<std::sync::Arc<crate::backend::submission_worker::SubmissionJobToken>>,
) -> Result<crate::backend::SchedulePresentOnWorkerResult> {
    logical_device.submission_worker.check_error()?;
    let enqueue_tv = if copy_tv.raw() > 0 {
        copy_tv
    } else {
        preallocated_present_tv.unwrap_or_else(|| logical_device.allocate_device_tv())
    };
    let copy_epoch = if copy_tv.raw() > 0 { copy_tv } else { enqueue_tv };
    logical_device.submission_worker.enqueue_with_token(
        enqueue_tv.raw(),
        Box::new(Dx12PresentCopyPendingSubmit {
            logical_device: Arc::clone(logical_device),
            command_lists,
            copy_tv,
            present_only_tv: if copy_tv.raw() > 0 {
                DeviceTv::ZERO
            } else {
                enqueue_tv
            },
        }),
        issue_token.clone(),
    )?;
    let consume_job: Box<dyn crate::backend::ScheduledPresentJob> = Box::new(Dx12ScheduledPresentJob {
        logical_device: Arc::clone(logical_device),
        copy_tv: copy_epoch,
        issue_token,
        sc,
        sync_tv,
        ctx_sync_tv,
        swapchain,
        present_mode,
        allow_tearing,
        finish,
        pending_finishes,
        device_removed,
    });
    Ok(crate::backend::SchedulePresentOnWorkerResult {
        present_tv: enqueue_tv.to_public(),
        consume_job: Some(consume_job),
    })
}

pub(super) fn enqueue_present_copy(
    logical_device: &SharedLogicalDevice,
    command_lists: Vec<Option<ID3D12CommandList>>,
    tv: DeviceTv,
) -> Result<()> {
    logical_device.submission_worker.check_error()?;
    logical_device.submission_worker.enqueue(
        tv.raw(),
        Box::new(Dx12PresentCopyPendingSubmit {
            logical_device: Arc::clone(logical_device),
            command_lists,
            copy_tv: tv,
            present_only_tv: DeviceTv::ZERO,
        }),
    )
}

pub(super) fn enqueue_compute_submit(
    logical_device: &SharedLogicalDevice,
    command_queue: ID3D12CommandQueue,
    queue_lock: Arc<Mutex<()>>,
    submission_worker: Arc<SubmissionWorker>,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
    buffers: &SharedBufferTable,
    sc: SharedSubmissionContext,
    ctx_fence: ID3D12Fence,
    slot_idx: usize,
    command_lists: Vec<Option<ID3D12CommandList>>,
    sync: Option<&SubmitSync>,
    fence_value: CtxTv,
) -> Result<()> {
    if let Err(e) = submission_worker.check_error() {
        return Err(e);
    }
    let queue_waits = resolve_queue_waits(logical_device, context_fences, sync)?;
    let host_observed_waits = resolve_host_observed_waits(context_fences, sync)?;
    let deferred_host_writes = sync.map(|s| s.deferred_host_writes.clone()).unwrap_or_default();
    validate_deferred_host_writes(buffers, &deferred_host_writes)?;
    submission_worker.enqueue(
        fence_value.raw(),
        Box::new(Dx12ComputePendingSubmit {
            logical_device: Arc::clone(logical_device),
            command_queue,
            queue_lock,
            sc,
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

pub(super) fn enqueue_retained_resubmit(
    logical_device: &SharedLogicalDevice,
    command_queue: ID3D12CommandQueue,
    queue_lock: Arc<Mutex<()>>,
    submission_worker: Arc<SubmissionWorker>,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
    buffers: &SharedBufferTable,
    sc: SharedSubmissionContext,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    sync: Option<&SubmitSync>,
    fence_value: CtxTv,
) -> Result<()> {
    submission_worker.check_error()?;
    let queue_waits = resolve_queue_waits(logical_device, context_fences, sync)?;
    let host_observed_waits = resolve_host_observed_waits(context_fences, sync)?;
    let deferred_host_writes = sync.map(|s| s.deferred_host_writes.clone()).unwrap_or_default();
    validate_deferred_host_writes(buffers, &deferred_host_writes)?;
    submission_worker.enqueue(
        fence_value.raw(),
        Box::new(Dx12RetainedResubmitPending {
            logical_device: Arc::clone(logical_device),
            command_queue,
            queue_lock,
            sc,
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
