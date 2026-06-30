//! Async GPU submission work enqueued on the per-device submission worker.

use super::types::{self, SharedContextMap, SharedLogicalDevice, SharedSubmissionContext};
use super::{ContextHandle, DeviceHandle};
use crate::backend::submission_worker::PendingSubmit;
use crate::backend::SubmitSync;
use crate::timeline::TimelineValue;
use anyhow::{Context as _, Result};
use ash::vk;
use std::sync::Arc;

fn apply_cpu_epoch_waits(
    ld: &SharedLogicalDevice,
    contexts: &SharedContextMap,
    sync: Option<&SubmitSync>,
) -> Result<()> {
    let Some(s) = sync else {
        return Ok(());
    };
    if s.cpu_waits.is_empty() {
        return Ok(());
    }
    for epoch in &s.cpu_waits {
        let sem = contexts
            .read()
            .unwrap()
            .get(&epoch.context)
            .with_context(|| format!("cross-submit cpu wait: invalid context {:?}", epoch.context))?
            .lock()
            .unwrap()
            .timeline_semaphore;
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&sem))
            .values(std::slice::from_ref(&epoch.value));
        unsafe { ld.device.wait_semaphores(&wait, u64::MAX) }
            .context("cross-submit cpu wait on timeline semaphore")?;
    }
    Ok(())
}

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
    queue: vk::Queue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    timeline_sem: vk::Semaphore,
    signal_value: TimelineValue,
    signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfo<'static>>,
    cmd: Option<vk::CommandBuffer>,
    wait_semaphores: Vec<(vk::Semaphore, u64)>,
    gpu_profile: Option<VulkanGpuProfileWork>,
    device_lost: Arc<std::sync::atomic::AtomicBool>,
}

pub(super) struct VulkanGpuProfileWork {
    pub ctx: ContextHandle,
    pub cmd: vk::CommandBuffer,
    pub prof: super::compute::VulkanGpuProfilePool,
}

impl PendingSubmit for VulkanQueueSubmitPending {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.vk.queue_submit");
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
        let queue_submit_result = {
            let _queue_guard = self.queue_lock.lock().unwrap();
            let _submit = crate::tracy_zone!("goldy.submit_worker.vk.queue_submit2");
            match (self.cmd, wait_infos.is_empty()) {
                (Some(cmd), true) => {
                    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
                    let submit_info2 = vk::SubmitInfo2::default()
                        .command_buffer_infos(std::slice::from_ref(&cmd_info))
                        .signal_semaphore_infos(&self.signal_semaphore_infos);
                    unsafe {
                        self.ld.device.queue_submit2(
                            self.queue,
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
                        .signal_semaphore_infos(&self.signal_semaphore_infos);
                    unsafe {
                        self.ld.device.queue_submit2(
                            self.queue,
                            std::slice::from_ref(&submit_info2),
                            vk::Fence::null(),
                        )
                    }
                }
                (None, true) => {
                    let submit_info2 =
                        vk::SubmitInfo2::default().signal_semaphore_infos(&self.signal_semaphore_infos);
                    unsafe {
                        self.ld.device.queue_submit2(
                            self.queue,
                            std::slice::from_ref(&submit_info2),
                            vk::Fence::null(),
                        )
                    }
                }
                (None, false) => {
                    let submit_info2 = vk::SubmitInfo2::default()
                        .wait_semaphore_infos(&wait_infos)
                        .signal_semaphore_infos(&self.signal_semaphore_infos);
                    unsafe {
                        self.ld.device.queue_submit2(
                            self.queue,
                            std::slice::from_ref(&submit_info2),
                            vk::Fence::null(),
                        )
                    }
                }
            }
        };
        queue_submit_result.context("Failed queue_submit2 on submission worker")?;

        if let Some(prof) = self.gpu_profile {
            let _tz = crate::tracy_zone!("goldy.submit_worker.vk.gpu_profile_readback");
            unsafe {
                super::compute::vulkan_finish_gpu_profile_pending(
                    prof.ctx,
                    &self.ld.device,
                    self.timeline_sem,
                    self.signal_value,
                    prof.cmd,
                    prof.prof,
                    &self.device_lost,
                )?;
            }
        }
        Ok(())
    }
}

pub(super) fn enqueue_vulkan_submit(
    ld: &SharedLogicalDevice,
    contexts: &SharedContextMap,
    queue: vk::Queue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    timeline_sem: vk::Semaphore,
    signal_value: TimelineValue,
    signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfo<'static>>,
    cmd: Option<vk::CommandBuffer>,
    sync: Option<&SubmitSync>,
    gpu_profile: Option<VulkanGpuProfileWork>,
    device_lost: Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    ld.submission_worker.check_error()?;
    apply_cpu_epoch_waits(ld, contexts, sync)?;
    let wait_semaphores = resolve_cross_submit_waits(contexts, sync)?;
    ld.submission_worker.enqueue(
        signal_value,
        Box::new(VulkanQueueSubmitPending {
            ld: Arc::clone(ld),
            queue,
            queue_lock,
            timeline_sem,
            signal_value,
            signal_semaphore_infos,
            cmd,
            wait_semaphores,
            gpu_profile,
            device_lost,
        }),
    )
}

struct VulkanPresentCopyPendingSubmit {
    ld: SharedLogicalDevice,
    queue: vk::Queue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    copy_cb: vk::CommandBuffer,
    timeline_sem: vk::Semaphore,
    frame_compute_timeline_value: u64,
    image_available_sem: vk::Semaphore,
    signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfo<'static>>,
}

impl PendingSubmit for VulkanPresentCopyPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.vk.present_copy");
        let _queue_guard = self.queue_lock.lock().unwrap();
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
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(&self.signal_semaphore_infos);
        let _submit = crate::tracy_zone!("goldy.submit_worker.vk.queue_submit2");
        unsafe {
            self.ld.device.queue_submit2(
                self.queue,
                std::slice::from_ref(&submit),
                vk::Fence::null(),
            )
        }
        .context("Failed queue_submit2 on submission worker (present copy)")?;
        Ok(())
    }
}

pub(super) fn enqueue_vulkan_present_copy(
    ld: &SharedLogicalDevice,
    queue: vk::Queue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    copy_cb: vk::CommandBuffer,
    timeline_sem: vk::Semaphore,
    frame_compute_timeline_value: u64,
    image_available_sem: vk::Semaphore,
    _render_finished_sem: vk::Semaphore,
    signal_timeline_value: TimelineValue,
    signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfo<'static>>,
) -> Result<()> {
    ld.submission_worker.check_error()?;
    ld.submission_worker.enqueue(
        signal_timeline_value,
        Box::new(VulkanPresentCopyPendingSubmit {
            ld: Arc::clone(ld),
            queue,
            queue_lock,
            copy_cb,
            timeline_sem,
            frame_compute_timeline_value,
            image_available_sem,
            signal_semaphore_infos,
        }),
    )
}
