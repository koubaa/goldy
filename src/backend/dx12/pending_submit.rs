//! Async GPU submission work enqueued on the per-device submission worker.

use super::types::{LogicalDevice, SharedLogicalDevice, SharedSubmissionContext};
use super::{ContextHandle, DeviceHandle};
use crate::backend::submission_worker::PendingSubmit;
use crate::backend::SubmitSync;
use crate::timeline::TimelineValue;
use anyhow::{Context as _, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandList, ID3D12Fence};

pub(super) fn resolve_queue_waits(
    _ld: &LogicalDevice,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
    sync: Option<&SubmitSync>,
) -> Result<Vec<(ID3D12Fence, u64)>> {
    let Some(s) = sync else {
        return Ok(Vec::new());
    };
    let mut waits = Vec::with_capacity(s.waits.len());
    for epoch in &s.waits {
        let fences = context_fences.read().unwrap();
        let (_, producer_fence) = fences
            .get(&epoch.context)
            .with_context(|| format!("cross-submit wait: unknown producer context {:?}", epoch.context))?;
        waits.push((producer_fence.clone(), epoch.value));
    }
    Ok(waits)
}

pub(super) struct Dx12ComputePendingSubmit {
    logical_device: SharedLogicalDevice,
    sc: SharedSubmissionContext,
    ctx_fence: ID3D12Fence,
    slot_idx: usize,
    command_lists: Vec<Option<ID3D12CommandList>>,
    queue_waits: Vec<(ID3D12Fence, u64)>,
    fence_value: TimelineValue,
}

impl PendingSubmit for Dx12ComputePendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.compute");
        {
            let _tz = crate::tracy_zone!("dx12.submit_worker.pre_reset_slots.before");
            let mut sc = self.sc.lock().unwrap();
            super::compute::pre_reset_retired_compute_slots(&self.logical_device, &mut sc, &self.ctx_fence)?;
        }
        super::utils::execute_preallocated_context_submit(
            &self.logical_device,
            &self.ctx_fence,
            &self.command_lists,
            &self.queue_waits,
            self.fence_value,
        )?;
        {
            let _tz = crate::tracy_zone!("dx12.submit_worker.pre_reset_slots.after");
            let mut sc = self.sc.lock().unwrap();
            super::compute::finish_compute_slot_submit(
                &self.logical_device,
                &mut sc,
                &self.ctx_fence,
                self.slot_idx,
            )?;
        }
        Ok(())
    }
}

pub(super) struct Dx12RetainedResubmitPending {
    logical_device: SharedLogicalDevice,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    queue_waits: Vec<(ID3D12Fence, u64)>,
    fence_value: TimelineValue,
}

impl PendingSubmit for Dx12RetainedResubmitPending {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.retained_resubmit");
        super::utils::execute_preallocated_context_submit(
            &self.logical_device,
            &self.ctx_fence,
            &self.command_lists,
            &self.queue_waits,
            self.fence_value,
        )
    }
}

struct Dx12PresentCopyPendingSubmit {
    logical_device: SharedLogicalDevice,
    command_lists: Vec<Option<ID3D12CommandList>>,
    tv: u64,
}

impl PendingSubmit for Dx12PresentCopyPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.present_copy");
        super::utils::signal_preallocated_device(&self.logical_device, &self.command_lists, self.tv)
    }
}

/// Full present job (optional scratch→backbuffer copy + DXGI Present) enqueued at scheme submit.
pub(super) struct Dx12ScheduledPresentPendingSubmit {
    logical_device: SharedLogicalDevice,
    command_lists: Vec<Option<ID3D12CommandList>>,
    copy_tv: u64,
    enqueue_tv: u64,
    ctx_fence: Option<ID3D12Fence>,
    sync_tv: u64,
    swapchain: windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
    present_mode: crate::types::PresentMode,
    allow_tearing: bool,
    finish: crate::backend::PresentFinishState,
    pending_finishes: std::sync::Arc<std::sync::Mutex<Vec<crate::backend::PresentFinishState>>>,
}

impl PendingSubmit for Dx12ScheduledPresentPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.scheduled_present");
        if !self.command_lists.is_empty() {
            super::utils::signal_preallocated_device(&self.logical_device, &self.command_lists, self.copy_tv)?;
        } else {
            super::utils::with_queue_lock(&self.logical_device, || {
                unsafe {
                    self.logical_device
                        .command_queue
                        .Signal(&self.logical_device.fence, self.enqueue_tv)
                }
                .context("Failed to signal device fence for scheduled present")
            })?;
        }

        {
            let _tz = crate::tracy_zone!("dx12.present.swapchain_present");
            // SAFETY: flip-model `Present` with `DXGI_SWAP_EFFECT_FLIP_DISCARD` is valid from any
            // thread; the submission worker serializes queue access via `queue_lock`.
            let (sync_interval, present_flags) = super::surface::present_args(self.present_mode, self.allow_tearing);
            let hr = unsafe { self.swapchain.Present(sync_interval, present_flags) };
            if hr.is_err() {
                anyhow::bail!("Present failed with HRESULT: {:?}", hr);
            }
        }

        if let Some(ctx_fence) = self.ctx_fence {
            if self.sync_tv > 0 {
                let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.ctx_fence_sync");
                super::utils::sync_context_fence_after_device_retire(
                    &self.logical_device,
                    &self.logical_device.fence,
                    &ctx_fence,
                    self.sync_tv,
                )?;
            }
        }

        self.pending_finishes.lock().unwrap().push(self.finish);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue_scheduled_present(
    logical_device: &SharedLogicalDevice,
    command_lists: Vec<Option<ID3D12CommandList>>,
    copy_tv: u64,
    ctx_fence: Option<ID3D12Fence>,
    sync_tv: u64,
    swapchain: windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
    present_mode: crate::types::PresentMode,
    allow_tearing: bool,
    finish: crate::backend::PresentFinishState,
    pending_finishes: std::sync::Arc<std::sync::Mutex<Vec<crate::backend::PresentFinishState>>>,
) -> Result<u64> {
    logical_device.submission_worker.check_error()?;
    let enqueue_tv = if copy_tv > 0 {
        copy_tv
    } else {
        crate::backend::submission_worker::allocate_timeline_value(&logical_device.timeline_next)
    };
    logical_device.submission_worker.enqueue(
        enqueue_tv,
        Box::new(Dx12ScheduledPresentPendingSubmit {
            logical_device: Arc::clone(logical_device),
            command_lists,
            copy_tv,
            enqueue_tv,
            ctx_fence,
            sync_tv,
            swapchain,
            present_mode,
            allow_tearing,
            finish,
            pending_finishes,
        }),
    )?;
    Ok(enqueue_tv)
}

pub(super) fn enqueue_present_copy(
    logical_device: &SharedLogicalDevice,
    command_lists: Vec<Option<ID3D12CommandList>>,
    tv: u64,
) -> Result<()> {
    logical_device.submission_worker.check_error()?;
    logical_device.submission_worker.enqueue(
        tv,
        Box::new(Dx12PresentCopyPendingSubmit {
            logical_device: Arc::clone(logical_device),
            command_lists,
            tv,
        }),
    )
}

pub(super) fn enqueue_compute_submit(
    logical_device: &SharedLogicalDevice,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
    sc: SharedSubmissionContext,
    ctx_fence: ID3D12Fence,
    slot_idx: usize,
    command_lists: Vec<Option<ID3D12CommandList>>,
    sync: Option<&SubmitSync>,
    fence_value: TimelineValue,
) -> Result<()> {
    logical_device.submission_worker.check_error()?;
    let queue_waits = resolve_queue_waits(logical_device, context_fences, sync)?;
    logical_device.submission_worker.enqueue(
        fence_value,
        Box::new(Dx12ComputePendingSubmit {
            logical_device: Arc::clone(logical_device),
            sc,
            ctx_fence,
            slot_idx,
            command_lists,
            queue_waits,
            fence_value,
        }),
    )
}

pub(super) fn enqueue_retained_resubmit(
    logical_device: &SharedLogicalDevice,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    sync: Option<&SubmitSync>,
    fence_value: TimelineValue,
) -> Result<()> {
    logical_device.submission_worker.check_error()?;
    let queue_waits = resolve_queue_waits(logical_device, context_fences, sync)?;
    logical_device.submission_worker.enqueue(
        fence_value,
        Box::new(Dx12RetainedResubmitPending {
            logical_device: Arc::clone(logical_device),
            ctx_fence,
            command_lists,
            queue_waits,
            fence_value,
        }),
    )
}
