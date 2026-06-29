//! Lock-split present path for Vulkan (copy submit + WSI present without global backend lock).

use super::types::{SharedLogicalDevice, VulkanState};
use super::SurfaceHandle;
use anyhow::{Context, Result};
use ash::{khr, vk};

pub(super) fn prepare_present_work(
    state: &mut VulkanState,
    frame: crate::backend::FrameToken,
    frame_compute_timeline_value: u64,
) -> Result<Box<dyn crate::backend::PresentGpuWork>> {
    Ok(Box::new(
        prepare_present_plan(state, frame, frame_compute_timeline_value)?.into_gpu_work(),
    ))
}

struct VulkanPresentPlan {
    frame: crate::backend::FrameToken,
    surface_handle: SurfaceHandle,
    image_index: u32,
    present_slot: usize,
    frame_compute_timeline_value: u64,
    render_pass_submitted: bool,
    instance: ash::Instance,
    logical_device: SharedLogicalDevice,
    render_finished_sem: vk::Semaphore,
    image_available_sem: vk::Semaphore,
    swapchain: vk::SwapchainKHR,
    timeline_sem: vk::Semaphore,
    copy_cb: vk::CommandBuffer,
    scratch_image: vk::Image,
    swapchain_image: vk::Image,
    width: u32,
    height: u32,
}

impl VulkanPresentPlan {
    fn into_gpu_work(self) -> VulkanPresentGpuWork {
        VulkanPresentGpuWork {
            frame: self.frame,
            surface_handle: self.surface_handle,
            image_index: self.image_index,
            present_slot: self.present_slot,
            frame_compute_timeline_value: self.frame_compute_timeline_value,
            render_pass_submitted: self.render_pass_submitted,
            instance: self.instance,
            logical_device: self.logical_device,
            render_finished_sem: self.render_finished_sem,
            image_available_sem: self.image_available_sem,
            swapchain: self.swapchain,
            timeline_sem: self.timeline_sem,
            copy_cb: self.copy_cb,
            scratch_image: self.scratch_image,
            swapchain_image: self.swapchain_image,
            width: self.width,
            height: self.height,
        }
    }

    fn record_present_copy(&self) -> Result<u64> {
        if self.render_pass_submitted {
            return Ok(0);
        }

        let _rcz = crate::tracy_zone!("vk.present.record_copy_cb");
        unsafe {
            self.logical_device
                .device
                .reset_command_buffer(self.copy_cb, vk::CommandBufferResetFlags::empty())
                .context("Failed to reset copy command buffer")?;
            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.logical_device
                .device
                .begin_command_buffer(self.copy_cb, &begin_info)
                .context("Failed to begin copy command buffer")?;

            let scratch_to_src = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(self.scratch_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let swapchain_to_dst = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(self.swapchain_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let pre_barriers = [scratch_to_src, swapchain_to_dst];
            let dep_pre = vk::DependencyInfo::default().image_memory_barriers(&pre_barriers);
            self.logical_device.device.cmd_pipeline_barrier2(self.copy_cb, &dep_pre);

            let region = vk::ImageCopy::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_offset(vk::Offset3D::default())
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_offset(vk::Offset3D::default())
                .extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                });
            self.logical_device.device.cmd_copy_image(
                self.copy_cb,
                self.scratch_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );

            let scratch_back = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(self.scratch_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let swapchain_to_present = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(self.swapchain_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let post_barriers = [scratch_back, swapchain_to_present];
            let dep_post = vk::DependencyInfo::default().image_memory_barriers(&post_barriers);
            self.logical_device
                .device
                .cmd_pipeline_barrier2(self.copy_cb, &dep_post);

            self.logical_device
                .device
                .end_command_buffer(self.copy_cb)
                .context("Failed to end copy command buffer")?;
        }

        // Pre-allocated on the render thread; enqueued on the FIFO worker after compute partitions.
        // GPU ordering vs prior compute relies on single-queue FIFO submission (no explicit Wait).
        let copy_tv =
            crate::backend::submission_worker::allocate_timeline_value(&self.logical_device.timeline_next);
        Ok(copy_tv)
    }

    fn build_finish_state(&self, return_fence: u64, scratch_layout_updated: bool) -> crate::backend::PresentFinishState {
        let present_timeline = if scratch_layout_updated {
            return_fence
        } else {
            self.frame_compute_timeline_value
        };
        crate::backend::PresentFinishState {
            frame: self.frame,
            return_fence,
            scratch_texture: None,
            scratch_layout_updated,
            present_timeline,
            copy_timeline: if scratch_layout_updated {
                Some(return_fence)
            } else {
                None
            },
            frame_compute_timeline: if self.render_pass_submitted {
                None
            } else {
                Some(self.frame_compute_timeline_value)
            },
            signal_timeline: if scratch_layout_updated {
                Some(return_fence)
            } else {
                None
            },
            render_pass_submitted: self.render_pass_submitted,
            present_ok: true,
        }
    }
}

fn prepare_present_plan(
    state: &mut VulkanState,
    frame: crate::backend::FrameToken,
    frame_compute_timeline_value: u64,
) -> Result<VulkanPresentPlan> {
    let surface_handle = frame.surface;
    let image_index = frame.image as u32;
    let present_slot = frame.present_slot as usize;

    if let Some(s) = state.surfaces.get_mut(&surface_handle) {
        s.current_texture_handle = None;
    }

    let s = state.surfaces.get(&surface_handle).context("Invalid surface handle")?;
    let render_pass_submitted = s.frame_sync[present_slot].render_pass_submitted;
    let device_handle = s.device_handle;
    let render_finished_sem = s.frame_sync[present_slot].render_finished_semaphore;
    let image_available_sem = s.frame_sync[present_slot].image_available_semaphore;
    let swapchain = s.swapchain;

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Surface's device is invalid")?
        .clone();

    let timeline_sem = state
        .contexts
        .read()
        .unwrap()
        .get(&frame.context)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
        .timeline_semaphore;

    let (copy_cb, scratch_image, swapchain_image, width, height) = if render_pass_submitted {
        (vk::CommandBuffer::null(), vk::Image::null(), vk::Image::null(), 0, 0)
    } else {
        let s = state.surfaces.get(&surface_handle).unwrap();
        let scratch = s.scratch_texture_slots[present_slot]
            .as_ref()
            .expect("scratch texture slot not initialized before present");
        (
            s.frame_sync[present_slot].copy_command_buffer,
            scratch.image,
            s.swapchain_images[image_index as usize],
            s.width,
            s.height,
        )
    };

    Ok(VulkanPresentPlan {
        frame,
        surface_handle,
        image_index,
        present_slot,
        frame_compute_timeline_value,
        render_pass_submitted,
        instance: state.instance.clone(),
        logical_device,
        render_finished_sem,
        image_available_sem,
        swapchain,
        timeline_sem,
        copy_cb,
        scratch_image,
        swapchain_image,
        width,
        height,
    })
}

/// Enqueue present copy + WSI present on the submission worker during scheme submit.
pub(super) fn schedule_present_on_submission_worker(
    state: &mut VulkanState,
    frame: crate::backend::FrameToken,
    submit_tv: u64,
) -> Result<u64> {
    let plan = prepare_present_plan(state, frame, submit_tv)?;
    let pending_finishes = std::sync::Arc::clone(&state.pending_present_finishes);
    let swapchain_out_of_date = std::sync::Arc::clone(
        &state
            .surfaces
            .get(&plan.surface_handle)
            .context("Invalid surface handle")?
            .swapchain_out_of_date,
    );
    let copy_tv = plan.record_present_copy()?;
    let return_fence = if copy_tv > 0 {
        copy_tv
    } else {
        plan.frame_compute_timeline_value
    };
    let scratch_layout_updated = copy_tv > 0;
    let finish = plan.build_finish_state(return_fence, scratch_layout_updated);
    let present_tv = super::pending_submit::enqueue_scheduled_present(
        &plan.logical_device,
        plan.instance,
        plan.copy_cb,
        plan.timeline_sem,
        plan.image_available_sem,
        plan.render_finished_sem,
        copy_tv,
        plan.render_pass_submitted,
        plan.swapchain,
        plan.image_index,
        finish,
        pending_finishes,
        std::sync::Arc::clone(&state.device_lost),
        swapchain_out_of_date,
    )?;
    Ok(present_tv)
}

/// Drain worker-scheduled presents and wait on the graphics queue before swapchain recreation.
pub(super) fn prepare_surface_for_resize(
    state: &mut VulkanState,
    surface_handle: SurfaceHandle,
) -> Result<()> {
    let device_handle = state
        .surfaces
        .get(&surface_handle)
        .with_context(|| format!("Invalid surface handle {:?}", surface_handle))?
        .device_handle;
    let ld = state
        .devices
        .get(&device_handle)
        .with_context(|| format!("Surface's device {:?} is invalid", device_handle))?
        .clone();

    ld.submission_worker.flush()?;
    ld.submission_worker.check_error()?;

    if let Some(surface) = state.surfaces.get_mut(&surface_handle) {
        surface.resize_prepare_applied.clear();
    }

    loop {
        let finish = {
            let mut pending = state.pending_present_finishes.lock().unwrap();
            pending
                .iter()
                .position(|f| f.frame.surface == surface_handle)
                .map(|idx| pending.remove(idx))
        };
        let Some(finish) = finish else {
            break;
        };
        let key = (finish.frame.image, finish.frame.present_slot);
        if finish.return_fence > 0 {
            let timeline_sem = state
                .contexts
                .read()
                .unwrap()
                .get(&finish.frame.context)
                .with_context(|| format!("Invalid context handle {:?}", finish.frame.context))?
                .lock()
                .unwrap()
                .timeline_semaphore;
            let wait = vk::SemaphoreWaitInfo::default()
                .semaphores(std::slice::from_ref(&timeline_sem))
                .values(std::slice::from_ref(&finish.return_fence));
            if let Err(e) = unsafe { ld.device.wait_semaphores(&wait, u64::MAX) } {
                if e == vk::Result::ERROR_DEVICE_LOST {
                    state
                        .device_lost
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    anyhow::bail!("device lost waiting for present before resize: {:?}", e);
                }
                tracing::warn!(?e, "wait_semaphores before resize failed; continuing");
            }
        }
        if let Err(e) = finish_present(state, finish) {
            tracing::warn!(?e, "finish_present before resize failed; continuing");
        } else if let Some(surface) = state.surfaces.get_mut(&surface_handle) {
            surface.resize_prepare_applied.insert(key);
        }
    }

    if let Err(e) = ld.synchronized_queue_wait_idle() {
        if e == vk::Result::ERROR_DEVICE_LOST {
            state
                .device_lost
                .store(true, std::sync::atomic::Ordering::Relaxed);
            anyhow::bail!("device lost before resize: {:?}", e);
        }
        tracing::warn!(?e, "queue_wait_idle before resize failed; continuing with swapchain recreation");
    }
    Ok(())
}

fn synthesize_scheduled_present_finish(
    state: &VulkanState,
    frame: crate::backend::FrameToken,
    present_tv: u64,
) -> Result<crate::backend::PresentFinishState> {
    let surface = state
        .surfaces
        .get(&frame.surface)
        .with_context(|| format!("Invalid surface handle {:?}", frame.surface))?;
    let present_slot = frame.present_slot as usize;
    let render_pass_submitted = surface.frame_sync[present_slot].render_pass_submitted;
    let return_fence = if render_pass_submitted {
        surface.frame_sync[present_slot]
            .copy_timeline_value
            .or(surface.frame_sync[present_slot].frame_timeline_value)
            .unwrap_or(present_tv)
    } else {
        present_tv
    };
    let scratch_layout_updated = !render_pass_submitted;
    let present_timeline = if scratch_layout_updated {
        return_fence
    } else {
        surface.frame_sync[present_slot]
            .frame_timeline_value
            .unwrap_or(return_fence)
    };
    Ok(crate::backend::PresentFinishState {
        frame,
        return_fence,
        scratch_texture: None,
        scratch_layout_updated,
        present_timeline,
        copy_timeline: if scratch_layout_updated {
            Some(return_fence)
        } else {
            None
        },
        frame_compute_timeline: if render_pass_submitted {
            None
        } else {
            surface.frame_sync[present_slot].frame_timeline_value
        },
        signal_timeline: if scratch_layout_updated {
            Some(return_fence)
        } else {
            None
        },
        render_pass_submitted,
        present_ok: true,
    })
}

pub(super) fn take_scheduled_present_blocking_wait(
    state: &VulkanState,
    frame: crate::backend::FrameToken,
    present_tv: u64,
) -> Result<Option<Box<dyn crate::backend::ScheduledPresentBlockingWait>>> {
    let device_handle = state
        .surfaces
        .get(&frame.surface)
        .context("Invalid surface handle")?
        .device_handle;
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Surface's device is invalid")?
        .clone();
    let pending_finishes = std::sync::Arc::clone(&state.pending_present_finishes);
    let timeline_sem = state
        .contexts
        .read()
        .unwrap()
        .get(&frame.context)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
        .timeline_semaphore;
    Ok(Some(Box::new(VulkanScheduledPresentWait {
        logical_device,
        pending_finishes,
        frame,
        present_tv,
        timeline_sem,
        device_lost: std::sync::Arc::clone(&state.device_lost),
    })))
}

struct VulkanScheduledPresentWait {
    logical_device: SharedLogicalDevice,
    pending_finishes: std::sync::Arc<std::sync::Mutex<Vec<crate::backend::PresentFinishState>>>,
    frame: crate::backend::FrameToken,
    present_tv: u64,
    timeline_sem: vk::Semaphore,
    device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl crate::backend::ScheduledPresentBlockingWait for VulkanScheduledPresentWait {
    fn run(self: Box<Self>) -> Result<crate::backend::ScheduledPresentWaitOutcome> {
        let _tz = crate::tracy_zone!("goldy.vk.scheduled_present.wait");
        self.logical_device.submission_worker.wait_submitted(self.present_tv)?;
        self.logical_device.submission_worker.check_error()?;

        let finish = {
            let mut pending = self.pending_finishes.lock().unwrap();
            pending
                .iter()
                .position(|f| {
                    f.frame.surface == self.frame.surface
                        && f.frame.image == self.frame.image
                        && f.frame.present_slot == self.frame.present_slot
                })
                .map(|idx| pending.remove(idx))
        };
        let return_fence = finish.as_ref().map(|f| f.return_fence).unwrap_or(0);
        if return_fence > 0 {
            let wait = vk::SemaphoreWaitInfo::default()
                .semaphores(std::slice::from_ref(&self.timeline_sem))
                .values(std::slice::from_ref(&return_fence));
            if let Err(e) = unsafe { self.logical_device.device.wait_semaphores(&wait, u64::MAX) } {
                if e == vk::Result::ERROR_DEVICE_LOST {
                    self.device_lost
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
                anyhow::bail!("wait_semaphores after scheduled present: {:?}", e);
            }
        }
        Ok(crate::backend::ScheduledPresentWaitOutcome {
            frame: self.frame,
            present_tv: self.present_tv,
            finish,
            return_fence,
        })
    }
}

pub(super) fn apply_scheduled_present_bookkeeping(
    state: &mut VulkanState,
    outcome: crate::backend::ScheduledPresentWaitOutcome,
) -> Result<()> {
    if let Some(surface) = state.surfaces.get_mut(&outcome.frame.surface) {
        let key = (outcome.frame.image, outcome.frame.present_slot);
        if surface.resize_prepare_applied.remove(&key) {
            return Ok(());
        }
    }
    let finish = match outcome.finish {
        Some(finish) => finish,
        None => {
            tracing::warn!(
                target: "goldy::vulkan",
                surface = outcome.frame.surface,
                image = outcome.frame.image,
                present_slot = outcome.frame.present_slot,
                present_tv = outcome.present_tv,
                "scheduled present finish missing from pending queue; synthesizing surface bookkeeping"
            );
            synthesize_scheduled_present_finish(state, outcome.frame, outcome.present_tv)?
        }
    };
    finish_present(state, finish)?;
    Ok(())
}

pub(super) fn finish_present(state: &mut VulkanState, finish: crate::backend::PresentFinishState) -> Result<u64> {
    let surface_handle = finish.frame.surface;
    let present_slot = finish.frame.present_slot as usize;
    let ctx = finish.frame.context;
    let image_index = finish.frame.image as u32;

    if let Some(signal_timeline) = finish.signal_timeline {
        if let Some(sc_arc) = state.contexts.read().unwrap().get(&ctx) {
            sc_arc.lock().unwrap().last_submitted_seq = signal_timeline;
        }
    }

    if let Some(frame_compute) = finish.frame_compute_timeline {
        if let Some(surface_state_mut) = state.surfaces.get_mut(&surface_handle) {
            surface_state_mut.frame_sync[present_slot].frame_timeline_value = Some(frame_compute);
            surface_state_mut.frame_sync[present_slot].last_compute_timeline_value = frame_compute;
        }
    }

    if let Some(copy_timeline) = finish.copy_timeline {
        if let Some(surface_state_mut) = state.surfaces.get_mut(&surface_handle) {
            surface_state_mut.frame_sync[present_slot].copy_timeline_value = Some(copy_timeline);
        }
    }

    let copy_tv = finish.copy_timeline.or_else(|| {
        state
            .surfaces
            .get(&surface_handle)
            .and_then(|s| s.frame_sync[present_slot].copy_timeline_value)
    });

    if let Some(tv) = copy_tv {
        if let Some(surface_state) = state.surfaces.get_mut(&surface_handle) {
            surface_state.pending_swapchain_returns.push((image_index, tv));
        }
    } else if let Some(surface_state) = state.surfaces.get_mut(&surface_handle) {
        surface_state.pending_acquire_count = surface_state.pending_acquire_count.saturating_sub(1);
        if let Some(sc_arc) = state.contexts.read().unwrap().get(&ctx) {
            sc_arc
                .lock()
                .unwrap()
                .signal_queue
                .push(crate::signal::Signal::SwapchainReturned { image_index });
        }
    }

    if finish.present_ok {
        let easement_tv = finish.copy_timeline.or_else(|| {
            state
                .surfaces
                .get_mut(&surface_handle)
                .and_then(|s| s.frame_sync[present_slot].frame_timeline_value.take())
        });
        easement_tv.context("present: easement timeline value missing (internal error)")
    } else {
        Err(anyhow::anyhow!("Failed to present"))
    }
}

struct VulkanPresentGpuWork {
    frame: crate::backend::FrameToken,
    surface_handle: SurfaceHandle,
    image_index: u32,
    present_slot: usize,
    frame_compute_timeline_value: u64,
    render_pass_submitted: bool,
    instance: ash::Instance,
    logical_device: SharedLogicalDevice,
    render_finished_sem: vk::Semaphore,
    image_available_sem: vk::Semaphore,
    swapchain: vk::SwapchainKHR,
    timeline_sem: vk::Semaphore,
    copy_cb: vk::CommandBuffer,
    scratch_image: vk::Image,
    swapchain_image: vk::Image,
    width: u32,
    height: u32,
}

impl crate::backend::PresentGpuWork for VulkanPresentGpuWork {
    fn run(self: Box<Self>) -> Result<crate::backend::PresentFinishState> {
        let _tz = crate::tracy_zone!("vk.surface.present");
        let mut copy_timeline = None;
        let mut signal_timeline = None;

        if !self.render_pass_submitted {
            let plan = VulkanPresentPlan {
                frame: self.frame,
                surface_handle: self.surface_handle,
                image_index: self.image_index,
                present_slot: self.present_slot,
                frame_compute_timeline_value: self.frame_compute_timeline_value,
                render_pass_submitted: self.render_pass_submitted,
                instance: self.instance.clone(),
                logical_device: self.logical_device.clone(),
                render_finished_sem: self.render_finished_sem,
                image_available_sem: self.image_available_sem,
                swapchain: self.swapchain,
                timeline_sem: self.timeline_sem,
                copy_cb: self.copy_cb,
                scratch_image: self.scratch_image,
                swapchain_image: self.swapchain_image,
                width: self.width,
                height: self.height,
            };
            let copy_tv = plan.record_present_copy()?;
            signal_timeline = Some(copy_tv);
            copy_timeline = Some(copy_tv);
            super::pending_submit::enqueue_vulkan_present_copy(
                &self.logical_device,
                self.copy_cb,
                self.timeline_sem,
                self.frame_compute_timeline_value,
                self.image_available_sem,
                self.render_finished_sem,
                copy_tv,
            )?;
            self.logical_device.submission_worker.wait_submitted(copy_tv)?;
            self.logical_device.submission_worker.check_error()?;
        }

        let swapchain_loader = khr::swapchain::Device::new(&self.instance, &self.logical_device.device);
        let swapchains = [self.swapchain];
        let image_indices = [self.image_index];
        let wait_semaphores = [self.render_finished_sem];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let queue_lock = std::sync::Arc::clone(&self.logical_device.queue_lock);
        let result = {
            let _queue_guard = queue_lock.lock().unwrap();
            let _pz = crate::tracy_zone!("vk.present.queue_present");
            unsafe { swapchain_loader.queue_present(self.logical_device.queue, &present_info) }
        };

        let present_ok = matches!(result, Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR));
        if let Err(e) = &result {
            let expected_during_resize = matches!(*e, vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR);
            if expected_during_resize {
                tracing::debug!(
                    self.surface_handle,
                    present_slot = self.present_slot,
                    image_index = self.image_index,
                    result = ?e,
                    "queue_present: swapchain out of date (will rebuild)"
                );
            } else {
                tracing::warn!(
                    self.surface_handle,
                    present_slot = self.present_slot,
                    image_index = self.image_index,
                    result = ?e,
                    "queue_present failed"
                );
            }
        }

        let present_timeline = copy_timeline.unwrap_or(self.frame_compute_timeline_value);

        Ok(crate::backend::PresentFinishState {
            frame: self.frame,
            return_fence: copy_timeline.unwrap_or(0),
            scratch_texture: None,
            scratch_layout_updated: false,
            present_timeline,
            copy_timeline,
            frame_compute_timeline: if self.render_pass_submitted {
                None
            } else {
                Some(self.frame_compute_timeline_value)
            },
            signal_timeline,
            render_pass_submitted: self.render_pass_submitted,
            present_ok,
        })
    }
}
