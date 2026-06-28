//! Async GPU submission work enqueued on the per-device submission worker.

use super::types::{self, SharedContextMap, SharedLogicalDevice, SharedSubmissionContext};
use super::{ContextHandle, DeviceHandle};
use crate::backend::submission_worker::PendingSubmit;
use crate::backend::SubmitSync;
use crate::timeline::TimelineValue;
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
            .with_context(|| format!("cross-submit wait: invalid context {:?}", epoch.context))?
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

pub(super) struct VulkanQueueSubmitPending {
    ld: SharedLogicalDevice,
    queue: vk::Queue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    timeline_sem: vk::Semaphore,
    signal_value: TimelineValue,
    signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfo<'static>>,
    cmd: Option<vk::CommandBuffer>,
    wait_semaphores: Vec<(vk::Semaphore, u64)>,
    sc: SharedSubmissionContext,
    contexts: SharedContextMap,
    device_handle: DeviceHandle,
    staging_belt_finish: bool,
    texture_staging_entries: Vec<super::staging::TextureStagingEntry>,
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

        {
            let _tz = crate::tracy_zone!("goldy.submit_worker.vk.post_signal_cleanup");
            if self.staging_belt_finish {
                self.sc.lock().unwrap().staging_belt.finish(self.signal_value);
            }
            if !self.texture_staging_entries.is_empty() {
                self.sc
                    .lock()
                    .unwrap()
                    .texture_staging_pool
                    .release(self.signal_value, self.texture_staging_entries);
            }
            if let Some(prof) = self.gpu_profile {
                let sem = self.sc.lock().unwrap().timeline_semaphore;
                unsafe {
                    super::compute::vulkan_finish_gpu_profile_pending(
                        prof.ctx,
                        &self.ld.device,
                        sem,
                        self.signal_value,
                        prof.cmd,
                        prof.prof,
                        &self.device_lost,
                    )?;
                }
            }

            let completed = unsafe {
                self.ld
                    .device
                    .get_semaphore_counter_value(self.timeline_sem)
                    .unwrap_or(self.signal_value)
            };
            vulkan_post_signal_cleanup(&self.ld, &self.contexts, self.device_handle, &self.sc, completed);
        }
        Ok(())
    }
}

pub(super) fn enqueue_vulkan_submit(
    ld: &SharedLogicalDevice,
    contexts: &SharedContextMap,
    device_handle: DeviceHandle,
    sc: &SharedSubmissionContext,
    queue: vk::Queue,
    queue_lock: Arc<std::sync::Mutex<()>>,
    timeline_sem: vk::Semaphore,
    signal_value: TimelineValue,
    signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfo<'static>>,
    cmd: Option<vk::CommandBuffer>,
    sync: Option<&SubmitSync>,
    staging_belt_finish: bool,
    texture_staging_entries: Vec<super::staging::TextureStagingEntry>,
    gpu_profile: Option<VulkanGpuProfileWork>,
    device_lost: Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    ld.submission_worker.check_error()?;
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
            sc: Arc::clone(sc),
            contexts: Arc::clone(contexts),
            device_handle,
            staging_belt_finish,
            texture_staging_entries,
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
    render_finished_sem: vk::Semaphore,
    signal_timeline_value: TimelineValue,
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
    render_finished_sem: vk::Semaphore,
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
            render_finished_sem,
            signal_timeline_value,
            signal_semaphore_infos,
        }),
    )
}
