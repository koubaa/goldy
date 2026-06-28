//! Async GPU submission work enqueued on the per-device submission worker.

use super::compute::Dx12GpuProfileResources;
use super::staging::TextureStagingEntry;
use super::types::{self, ContextFenceEntry, LogicalDevice, SharedLogicalDevice, SharedSubmissionContext};
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

pub(super) fn dx12_post_signal_cleanup(
    logical_device: &LogicalDevice,
    sc: &SharedSubmissionContext,
    context_fences: &Arc<RwLock<HashMap<ContextHandle, ContextFenceEntry>>>,
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
            types::destroy_pending_deletion(logical_device, &mut registry, resource, Vec::new());
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
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    queue_waits: Vec<(ID3D12Fence, u64)>,
    sc: SharedSubmissionContext,
    context_fences: Arc<RwLock<HashMap<ContextHandle, ContextFenceEntry>>>,
    fence_value: TimelineValue,
    gpu_profile: Option<Dx12GpuProfileResources>,
    staged_texture_entries: Vec<TextureStagingEntry>,
}

impl PendingSubmit for Dx12ComputePendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.compute");
        super::utils::execute_preallocated_context_submit(
            &self.logical_device,
            &self.queue,
            &self.queue_lock,
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
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    ctx_fence: ID3D12Fence,
    command_lists: Vec<Option<ID3D12CommandList>>,
    queue_waits: Vec<(ID3D12Fence, u64)>,
    sc: SharedSubmissionContext,
    context_fences: Arc<RwLock<HashMap<ContextHandle, ContextFenceEntry>>>,
    fence_value: TimelineValue,
}

impl PendingSubmit for Dx12RetainedResubmitPending {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.dx12.retained_resubmit");
        super::utils::execute_preallocated_context_submit(
            &self.logical_device,
            &self.queue,
            &self.queue_lock,
            &self.ctx_fence,
            &self.command_lists,
            &self.queue_waits,
            self.fence_value,
        )?;
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
                types::destroy_pending_deletion(dev, &mut registry, resource, Vec::new());
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
    context_fences: &Arc<RwLock<HashMap<ContextHandle, ContextFenceEntry>>>,
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    queue_lock: Arc<std::sync::Mutex<()>>,
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
            queue,
            queue_lock,
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
    context_fences: &Arc<RwLock<HashMap<ContextHandle, ContextFenceEntry>>>,
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    queue_lock: Arc<std::sync::Mutex<()>>,
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
            queue,
            queue_lock,
            ctx_fence,
            command_lists,
            queue_waits,
            sc,
            context_fences: Arc::clone(context_fences),
            fence_value,
        }),
    )
}
