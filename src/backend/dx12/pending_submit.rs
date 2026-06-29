//! Async GPU submission work enqueued on the per-device submission worker.

use super::compute::Dx12GpuProfileResources;
use super::staging::TextureStagingEntry;
use super::types::{self, LogicalDevice, SharedLogicalDevice, SharedSubmissionContext};
use super::{ContextHandle, DeviceHandle};
use crate::backend::submission_worker::PendingSubmit;
use crate::backend::SubmitSync;
use crate::timeline::TimelineValue;
use anyhow::{Context as _, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandList, ID3D12Fence};

pub(super) fn resolve_queue_waits(
    ld: &LogicalDevice,
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

pub(super) fn dx12_post_signal_cleanup(
    logical_device: &LogicalDevice,
    sc: &SharedSubmissionContext,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
    ctx_fence: &ID3D12Fence,
    fence_value: TimelineValue,
    gpu_profile: Option<Dx12GpuProfileResources>,
    staged_texture_entries: Vec<TextureStagingEntry>,
) -> Result<()> {
    if let Some(prof) = gpu_profile {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.gpu_profile_readback");
        if let Err(e) = super::compute::dx12_finish_gpu_profile(
            ctx_fence,
            &logical_device.command_queue,
            fence_value,
            prof,
        ) {
            tracing::warn!("GOLDY_GPU_PROFILE: DX12 readback failed: {e}");
        }
    }

    let ctx_completed = unsafe { ctx_fence.GetCompletedValue() };
    let ctx_del_batch = {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.deletion_drain");
        sc.lock().unwrap().deletion_queue.drain_up_to_completed(ctx_completed)
    };
    {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.slot_reclamation");
        let descriptors_arc = Arc::clone(&logical_device.descriptors);
        let mut registry = descriptors_arc.lock().unwrap();
        for resource in ctx_del_batch {
            types::destroy_pending_deletion(logical_device, &mut registry, resource);
        }
        let fences = context_fences.read().unwrap();
        registry.drain_ready_slot_reclamations(&fences);
    }

    {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.staging_finish");
        sc.lock().unwrap().staging_belt.finish(fence_value);
    }

    if !staged_texture_entries.is_empty() {
        sc.lock()
            .unwrap()
            .texture_staging_pool
            .release(fence_value, staged_texture_entries);
    }

    Ok(())
}

pub(super) struct Dx12ComputePendingSubmit {
    logical_device: SharedLogicalDevice,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    queue_waits: Vec<(ID3D12Fence, u64)>,
    sc: SharedSubmissionContext,
    context_fences: Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
    fence_value: TimelineValue,
    gpu_profile: Option<Dx12GpuProfileResources>,
    staged_texture_entries: Vec<TextureStagingEntry>,
}

impl PendingSubmit for Dx12ComputePendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.compute");
        super::utils::execute_preallocated_context_submit(
            &self.logical_device,
            &self.ctx_fence,
            &self.command_lists,
            &self.queue_waits,
            self.fence_value,
        )?;
        dx12_post_signal_cleanup(
            &self.logical_device,
            &self.sc,
            &self.context_fences,
            &self.ctx_fence,
            self.fence_value,
            self.gpu_profile,
            self.staged_texture_entries,
        )
    }
}

pub(super) struct Dx12RetainedResubmitPending {
    logical_device: SharedLogicalDevice,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    queue_waits: Vec<(ID3D12Fence, u64)>,
    sc: SharedSubmissionContext,
    context_fences: Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
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
        )?;
        // Read completed value immediately after Signal without blocking: the GPU has not
        // finished this submission yet, so we only drain deletions retired by prior work.
        // Waiting for `fence_value` here would stall the submission worker on every resubmit.
        let ctx_completed = unsafe { self.ctx_fence.GetCompletedValue() };
        let retained_del_batch = {
            let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.deletion_drain");
            self.sc
                .lock()
                .unwrap()
                .deletion_queue
                .drain_up_to_completed(ctx_completed)
        };
        {
            let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.slot_reclamation");
            let dev = &self.logical_device;
            let descriptors_arc = Arc::clone(&dev.descriptors);
            let mut registry = descriptors_arc.lock().unwrap();
            for resource in retained_del_batch {
                types::destroy_pending_deletion(dev, &mut registry, resource);
            }
            let fences = self.context_fences.read().unwrap();
            registry.drain_ready_slot_reclamations(&fences);
        }
        Ok(())
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
            let (sync_interval, present_flags) =
                super::surface::present_args(self.present_mode, self.allow_tearing);
            let hr = unsafe { self.swapchain.Present(sync_interval, present_flags) };
            if hr.is_err() {
                anyhow::bail!("Present failed with HRESULT: {:?}", hr);
            }
        }

        self.pending_finishes.lock().unwrap().push(self.finish);
        Ok(())
    }
}

pub(super) fn enqueue_scheduled_present(
    logical_device: &SharedLogicalDevice,
    command_lists: Vec<Option<ID3D12CommandList>>,
    copy_tv: u64,
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
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    sync: Option<&SubmitSync>,
    sc: SharedSubmissionContext,
    fence_value: TimelineValue,
    gpu_profile: Option<Dx12GpuProfileResources>,
    staged_texture_entries: Vec<TextureStagingEntry>,
) -> Result<()> {
    logical_device.submission_worker.check_error()?;
    let queue_waits = resolve_queue_waits(logical_device, context_fences, sync)?;
    logical_device.submission_worker.enqueue(
        fence_value,
        Box::new(Dx12ComputePendingSubmit {
            logical_device: Arc::clone(logical_device),
            ctx_fence,
            command_lists,
            queue_waits,
            sc,
            context_fences: Arc::clone(context_fences),
            fence_value,
            gpu_profile,
            staged_texture_entries,
        }),
    )
}

pub(super) fn enqueue_retained_resubmit(
    logical_device: &SharedLogicalDevice,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, (DeviceHandle, ID3D12Fence)>>>,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    sync: Option<&SubmitSync>,
    sc: SharedSubmissionContext,
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
            sc,
            context_fences: Arc::clone(context_fences),
            fence_value,
        }),
    )
}
