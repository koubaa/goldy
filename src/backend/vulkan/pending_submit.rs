//! Async GPU submission work enqueued on the per-device submission worker.

use super::types::{self, SharedBufferTable, SharedContextMap, SharedLogicalDevice, SharedSubmissionContext};
use super::{ContextHandle, DeviceHandle};
use crate::backend::submission_worker::PendingSubmit;
use crate::backend::{DeferredHostWrite, SubmitSync};
use crate::timeline::{Epoch, TimelineValue};
use anyhow::{Context as _, Result};
use ash::vk;
use std::sync::Arc;

pub(super) fn resolve_cross_submit_waits(
    contexts: &SharedContextMap,
    sync: Option<&SubmitSync>,
) -> Result<Vec<(vk::Semaphore, u64)>> {
    let Some(s) = sync else {
        return Ok(Vec::new());
    };
    let mut waits = Vec::with_capacity(s.waits.len());
    for epoch in &s.waits {
        let sem = contexts
            .read()
            .unwrap()
            .get(&epoch.context)
            .with_context(|| format!("cross-submit wait: invalid producer context {:?}", epoch.context))?
            .lock()
            .unwrap()
            .timeline_semaphore;
        waits.push((sem, epoch.value));
    }
    Ok(waits)
}

fn wait_host_observed_epochs(ld: &types::LogicalDevice, contexts: &SharedContextMap, epochs: &[Epoch]) -> Result<()> {
    if epochs.is_empty() {
        return Ok(());
    }
    let _tz = crate::tracy_zone!("goldy.vk.pending_submit.wait_host_observed_epochs");
    for epoch in epochs {
        let sem = contexts
            .read()
            .unwrap()
            .get(&epoch.context)
            .with_context(|| format!("host-observed wait: invalid producer context {:?}", epoch.context))?
            .lock()
            .unwrap()
            .timeline_semaphore;
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&sem))
            .values(std::slice::from_ref(&epoch.value));
        unsafe { ld.device.wait_semaphores(&wait, u64::MAX) }.context("host-observed vkWaitSemaphores")?;
    }
    Ok(())
}

fn validate_deferred_host_writes(buffers: &SharedBufferTable, deferred_writes: &[DeferredHostWrite]) -> Result<()> {
    if deferred_writes.is_empty() {
        return Ok(());
    }
    let buffers_read = buffers.read().unwrap();
    for w in deferred_writes {
        buffers_read
            .entries
            .get(&w.buffer)
            .with_context(|| format!("deferred host write: invalid buffer handle {}", w.buffer))?;
    }
    Ok(())
}

fn apply_deferred_host_writes(buffers: &SharedBufferTable, deferred_writes: &[DeferredHostWrite]) -> Result<()> {
    let _tz = crate::tracy_zone!("goldy.vk.pending_submit.apply_deferred_host_writes");
    for w in deferred_writes {
        let buffers_read = buffers.read().unwrap();
        let buffer = buffers_read
            .entries
            .get(&w.buffer)
            .with_context(|| format!("deferred host write: invalid buffer handle {}", w.buffer))?;
        if let Some(base) = buffer.host_mapped {
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
    ld: &types::LogicalDevice,
    contexts: &SharedContextMap,
    host_observed_waits: &[Epoch],
    buffers: &SharedBufferTable,
    deferred_writes: &[DeferredHostWrite],
) -> Result<()> {
    let _tz = crate::tracy_zone!("goldy.vk.pending_submit.apply_host_sidecar_before_gpu");
    wait_host_observed_epochs(ld, contexts, host_observed_waits)?;
    apply_deferred_host_writes(buffers, deferred_writes)?;
    Ok(())
}

pub(super) fn vulkan_post_signal_cleanup(
    ld: &types::LogicalDevice,
    contexts: &SharedContextMap,
    device_handle: DeviceHandle,
    sc: &SharedSubmissionContext,
    completed_hint: u64,
) {
    let ctx_batch = sc.lock().unwrap().deletion_queue.drain_up_to(completed_hint);
    if ctx_batch.is_empty() {
        let descriptors_arc = Arc::clone(&ld.descriptors);
        let mut registry = descriptors_arc.lock().unwrap();
        let completed_values = types::snapshot_context_completed_values(&ld.device, contexts, device_handle);
        registry.drain_ready_slot_reclamations(&completed_values);
        return;
    }
    let descriptors_arc = Arc::clone(&ld.descriptors);
    let mut registry = descriptors_arc.lock().unwrap();
    for r in ctx_batch {
        types::destroy_pending_deletion(ld, &mut registry, r);
    }
    let completed_values = types::snapshot_context_completed_values(&ld.device, contexts, device_handle);
    registry.drain_ready_slot_reclamations(&completed_values);
}

/// Per-context deletion drain + slot reclamation on the render/wait thread.
pub(super) fn vulkan_drain_context_deletion_up_to(
    ld: &types::LogicalDevice,
    contexts: &SharedContextMap,
    device_handle: DeviceHandle,
    sc: &SharedSubmissionContext,
    completed: u64,
) {
    let _tz = crate::tracy_zone!("goldy.submit.vk.deletion_drain");
    vulkan_post_signal_cleanup(ld, contexts, device_handle, sc, completed);
}

/// Read back deferred GPU profile results once `completed` covers each submit TV.
pub(super) fn vulkan_drain_pending_gpu_profiles_up_to(
    ld: &types::LogicalDevice,
    sc: &mut types::SubmissionContext,
    completed: u64,
) {
    if sc.pending_gpu_profiles.is_empty() {
        return;
    }
    let _tz = crate::tracy_zone!("goldy.gpu_profile_readback");
    let (ready, pending): (Vec<_>, Vec<_>) = sc.pending_gpu_profiles.drain(..).partition(|(tv, _)| *tv <= completed);
    sc.pending_gpu_profiles = pending;
    for (tv, prof) in ready {
        if let Err(e) = unsafe { super::compute::vulkan_readback_gpu_profile(&ld.device, tv, prof.prof) } {
            tracing::warn!("GOLDY_GPU_PROFILE: Vulkan readback failed: {e}");
        }
        let _ = (prof.ctx, prof.cmd);
    }
}

pub(super) fn vulkan_finish_staging_after_enqueue(
    sc: &SharedSubmissionContext,
    signal_value: TimelineValue,
    staging_belt_finish: bool,
    texture_staging_entries: Vec<super::staging::TextureStagingEntry>,
) {
    if !staging_belt_finish && texture_staging_entries.is_empty() {
        return;
    }
    let _tz = crate::tracy_zone!("goldy.submit.vk.staging_finish");
    let mut sc_guard = sc.lock().unwrap();
    if staging_belt_finish {
        sc_guard.staging_belt.finish(signal_value);
    }
    if !texture_staging_entries.is_empty() {
        sc_guard
            .texture_staging_pool
            .release(signal_value, texture_staging_entries);
    }
}

pub(super) struct VulkanQueueSubmitPending {
    ld: SharedLogicalDevice,
    contexts: SharedContextMap,
    frame_table: super::types::SharedFrameTableDevice,
    buffers: SharedBufferTable,
    timeline_sem: vk::Semaphore,
    signal_value: TimelineValue,
    cmd: Option<vk::CommandBuffer>,
    wait_semaphores: Vec<(vk::Semaphore, u64)>,
    host_observed_waits: Vec<Epoch>,
    deferred_host_writes: Vec<DeferredHostWrite>,
}

pub(super) struct VulkanGpuProfileWork {
    pub ctx: ContextHandle,
    pub cmd: vk::CommandBuffer,
    pub prof: super::compute::VulkanGpuProfilePool,
}

impl PendingSubmit for VulkanQueueSubmitPending {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.vk.queue_submit");
        apply_host_sidecar_before_gpu(
            &self.ld,
            &self.contexts,
            &self.host_observed_waits,
            &self.buffers,
            &self.deferred_host_writes,
        )?;
        // Per-context frame-table slots are bound once at context init; no
        // per-submit rebinding needed.
        let queue_lock = Arc::clone(&self.ld.queue_lock);
        let wait_infos: Vec<vk::SemaphoreSubmitInfo> = self
            .wait_semaphores
            .iter()
            .map(|(sem, val)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*sem)
                    .value(*val)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline_sem)
            .value(self.signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let queue_submit_result = {
            let _queue_guard = queue_lock.lock().unwrap();
            let _submit = crate::tracy_zone!("goldy.submit_worker.vk.queue_submit2");
            match (self.cmd, wait_infos.is_empty()) {
                (Some(cmd), true) => {
                    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
                    let submit_info2 = vk::SubmitInfo2::default()
                        .command_buffer_infos(std::slice::from_ref(&cmd_info))
                        .signal_semaphore_infos(std::slice::from_ref(&signal_info));
                    unsafe {
                        self.ld.device.queue_submit2(
                            self.ld.queue,
                            std::slice::from_ref(&submit_info2),
                            vk::Fence::null(),
                        )
                    }
                }
                (Some(cmd), false) => {
                    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
                    let submit_info2 = vk::SubmitInfo2::default()
                        .command_buffer_infos(std::slice::from_ref(&cmd_info))
                        .wait_semaphore_infos(&wait_infos)
                        .signal_semaphore_infos(std::slice::from_ref(&signal_info));
                    unsafe {
                        self.ld.device.queue_submit2(
                            self.ld.queue,
                            std::slice::from_ref(&submit_info2),
                            vk::Fence::null(),
                        )
                    }
                }
                (None, true) => {
                    let submit_info2 =
                        vk::SubmitInfo2::default().signal_semaphore_infos(std::slice::from_ref(&signal_info));
                    unsafe {
                        self.ld.device.queue_submit2(
                            self.ld.queue,
                            std::slice::from_ref(&submit_info2),
                            vk::Fence::null(),
                        )
                    }
                }
                (None, false) => {
                    let submit_info2 = vk::SubmitInfo2::default()
                        .wait_semaphore_infos(&wait_infos)
                        .signal_semaphore_infos(std::slice::from_ref(&signal_info));
                    unsafe {
                        self.ld.device.queue_submit2(
                            self.ld.queue,
                            std::slice::from_ref(&submit_info2),
                            vk::Fence::null(),
                        )
                    }
                }
            }
        };
        queue_submit_result.context("Failed queue_submit2 on submission worker")?;
        Ok(())
    }
}

pub(super) fn enqueue_vulkan_submit(
    ld: &SharedLogicalDevice,
    contexts: &SharedContextMap,
    sync: Option<&SubmitSync>,
    frame_table: super::types::SharedFrameTableDevice,
    buffers: &SharedBufferTable,
    timeline_sem: vk::Semaphore,
    signal_value: TimelineValue,
    cmd: Option<vk::CommandBuffer>,
) -> Result<()> {
    ld.submission_worker.check_error()?;
    let wait_semaphores = resolve_cross_submit_waits(contexts, sync)?;
    let host_observed_waits = sync.map(|s| s.host_observed_waits.clone()).unwrap_or_default();
    let deferred_host_writes = sync.map(|s| s.deferred_host_writes.clone()).unwrap_or_default();
    validate_deferred_host_writes(buffers, &deferred_host_writes)?;
    ld.submission_worker.enqueue(
        signal_value,
        Box::new(VulkanQueueSubmitPending {
            ld: Arc::clone(ld),
            contexts: Arc::clone(contexts),
            frame_table,
            buffers: Arc::clone(buffers),
            timeline_sem,
            signal_value,
            cmd,
            wait_semaphores,
            host_observed_waits,
            deferred_host_writes,
        }),
    )
}

struct VulkanPresentCopyPendingSubmit {
    ld: SharedLogicalDevice,
    copy_cb: vk::CommandBuffer,
    timeline_sem: vk::Semaphore,
    frame_compute_timeline_value: u64,
    image_available_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    signal_timeline_value: TimelineValue,
}

impl PendingSubmit for VulkanPresentCopyPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.vk.present_copy");
        let queue_lock = Arc::clone(&self.ld.queue_lock);
        let _queue_guard = queue_lock.lock().unwrap();
        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(self.copy_cb);
        let wait_compute_done = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline_sem)
            .value(self.frame_compute_timeline_value)
            .stage_mask(vk::PipelineStageFlags2::TRANSFER);
        let wait_acq = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.image_available_sem)
            .value(0)
            .stage_mask(vk::PipelineStageFlags2::TRANSFER);
        let waits = [wait_compute_done, wait_acq];
        let sig_render_finished = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.render_finished_sem)
            .value(0)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let sig_timeline = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline_sem)
            .value(self.signal_timeline_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let signals = [sig_render_finished, sig_timeline];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(&signals);
        let _submit = crate::tracy_zone!("goldy.submit_worker.vk.queue_submit2");
        unsafe {
            self.ld
                .device
                .queue_submit2(self.ld.queue, std::slice::from_ref(&submit), vk::Fence::null())
        }
        .context("Failed queue_submit2 on submission worker (present copy)")?;
        Ok(())
    }
}

pub(super) fn enqueue_vulkan_present_copy(
    ld: &SharedLogicalDevice,
    copy_cb: vk::CommandBuffer,
    timeline_sem: vk::Semaphore,
    frame_compute_timeline_value: u64,
    image_available_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    signal_timeline_value: TimelineValue,
) -> Result<()> {
    ld.submission_worker.check_error()?;
    ld.submission_worker.enqueue(
        signal_timeline_value,
        Box::new(VulkanPresentCopyPendingSubmit {
            ld: Arc::clone(ld),
            copy_cb,
            timeline_sem,
            frame_compute_timeline_value,
            image_available_sem,
            render_finished_sem,
            signal_timeline_value,
        }),
    )
}

/// Full present job (optional scratch→swapchain copy + WSI present) enqueued at scheme submit.
struct VulkanScheduledPresentPendingSubmit {
    ld: SharedLogicalDevice,
    instance: ash::Instance,
    copy_cb: vk::CommandBuffer,
    timeline_sem: vk::Semaphore,
    image_available_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    copy_tv: u64,
    enqueue_tv: u64,
    render_pass_submitted: bool,
    swapchain: vk::SwapchainKHR,
    image_index: u32,
    finish: crate::backend::PresentFinishState,
    pending_finishes: std::sync::Arc<std::sync::Mutex<Vec<crate::backend::PresentFinishState>>>,
    device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    swapchain_out_of_date: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl PendingSubmit for VulkanScheduledPresentPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.vk.scheduled_present");
        let queue_lock = Arc::clone(&self.ld.queue_lock);
        let _queue_guard = queue_lock.lock().unwrap();

        if !self.render_pass_submitted {
            let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(self.copy_cb);
            let wait_acq = vk::SemaphoreSubmitInfo::default()
                .semaphore(self.image_available_sem)
                .value(0)
                .stage_mask(vk::PipelineStageFlags2::TRANSFER);
            let waits = [wait_acq];
            let sig_render_finished = vk::SemaphoreSubmitInfo::default()
                .semaphore(self.render_finished_sem)
                .value(0)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
            let sig_timeline = vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline_sem)
                .value(self.copy_tv)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
            let signals = [sig_render_finished, sig_timeline];
            let submit = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&waits)
                .command_buffer_infos(std::slice::from_ref(&cmd_info))
                .signal_semaphore_infos(&signals);
            let _submit = crate::tracy_zone!("goldy.submit_worker.vk.queue_submit2");
            unsafe {
                self.ld
                    .device
                    .queue_submit2(self.ld.queue, std::slice::from_ref(&submit), vk::Fence::null())
            }
            .context("Failed queue_submit2 on submission worker (scheduled present copy)")?;
        } else {
            let sig_timeline = vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline_sem)
                .value(self.enqueue_tv)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
            let submit = vk::SubmitInfo2::default().signal_semaphore_infos(std::slice::from_ref(&sig_timeline));
            unsafe {
                self.ld
                    .device
                    .queue_submit2(self.ld.queue, std::slice::from_ref(&submit), vk::Fence::null())
            }
            .context("Failed queue_submit2 on submission worker (scheduled present signal)")?;
        }

        {
            let _pz = crate::tracy_zone!("vk.present.queue_present");
            let swapchain_loader = ash::khr::swapchain::Device::new(&self.instance, &self.ld.device);
            let swapchains = [self.swapchain];
            let image_indices = [self.image_index];
            let wait_semaphores = [self.render_finished_sem];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            let result = unsafe { swapchain_loader.queue_present(self.ld.queue, &present_info) };
            let present_ok = matches!(result, Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR));
            if let Err(e) = &result {
                let expected_during_resize =
                    matches!(*e, vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR);
                if expected_during_resize {
                    if *e == vk::Result::ERROR_OUT_OF_DATE_KHR {
                        self.swapchain_out_of_date
                            .store(true, std::sync::atomic::Ordering::Release);
                    }
                    tracing::debug!(
                        image_index = self.image_index,
                        result = ?e,
                        "scheduled queue_present: swapchain out of date (will rebuild)"
                    );
                } else {
                    tracing::warn!(
                        image_index = self.image_index,
                        result = ?e,
                        "scheduled queue_present failed"
                    );
                    if *e == vk::Result::ERROR_DEVICE_LOST {
                        self.device_lost.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    anyhow::bail!("Failed to present: {:?}", e);
                }
            }
            if !present_ok {
                anyhow::bail!("Failed to present");
            }
        }

        self.pending_finishes.lock().unwrap().push(self.finish);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue_scheduled_present(
    ld: &SharedLogicalDevice,
    instance: ash::Instance,
    copy_cb: vk::CommandBuffer,
    timeline_sem: vk::Semaphore,
    image_available_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    copy_tv: u64,
    render_pass_submitted: bool,
    swapchain: vk::SwapchainKHR,
    image_index: u32,
    finish: crate::backend::PresentFinishState,
    pending_finishes: std::sync::Arc<std::sync::Mutex<Vec<crate::backend::PresentFinishState>>>,
    device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    swapchain_out_of_date: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<u64> {
    ld.submission_worker.check_error()?;
    let enqueue_tv = if copy_tv > 0 {
        copy_tv
    } else {
        crate::backend::submission_worker::allocate_timeline_value(&ld.timeline_next)
    };
    ld.submission_worker.enqueue(
        enqueue_tv,
        Box::new(VulkanScheduledPresentPendingSubmit {
            ld: Arc::clone(ld),
            instance,
            copy_cb,
            timeline_sem,
            image_available_sem,
            render_finished_sem,
            copy_tv,
            enqueue_tv,
            render_pass_submitted,
            swapchain,
            image_index,
            finish,
            pending_finishes,
            device_lost,
            swapchain_out_of_date,
        }),
    )?;
    Ok(enqueue_tv)
}
