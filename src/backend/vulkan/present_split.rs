//! Lock-split present path for Vulkan (copy submit + WSI present without global backend lock).

use super::types::{SharedLogicalDevice, VulkanState};
use super::SurfaceHandle;
use anyhow::{Context, Result};
use ash::{khr, vk};

pub(super) fn prepare_present_work(
    state: &VulkanState,
    frame: crate::backend::FrameToken,
    frame_compute_timeline_value: u64,
) -> Result<Box<dyn crate::backend::PresentGpuWork>> {
    let surface_handle = frame.surface;
    let image_index = frame.image as u32;
    let present_slot = frame.present_slot as usize;

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

    let compute_timeline_sem = state
        .contexts
        .read()
        .unwrap()
        .get(&frame.context)
        .context("Invalid context handle")?
        .lock()
        .unwrap()
        .timeline_semaphore;

    let owner_timeline_sem = super::context::owner_timeline_semaphore(state, device_handle)
        .context("device-owner context missing for present")?;

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

    Ok(Box::new(VulkanPresentGpuWork {
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
        compute_timeline_sem,
        owner_timeline_sem,
        copy_cb,
        scratch_image,
        swapchain_image,
        width,
        height,
    }))
}

pub(super) fn finish_present(state: &mut VulkanState, finish: crate::backend::PresentFinishState) -> Result<u64> {
    let surface_handle = finish.frame.surface;
    let present_slot = finish.frame.present_slot as usize;
    let ctx = finish.frame.context;
    let image_index = finish.frame.image as u32;

    if let Some(signal_timeline) = finish.signal_timeline {
        let device_handle = state
            .surfaces
            .get(&surface_handle)
            .map(|s| s.device_handle)
            .context("Invalid surface handle")?;
        if let Some(owner) = state.device_owner_handles.get(&device_handle) {
            if let Some(owner_arc) = state.contexts.read().unwrap().get(owner) {
                owner_arc.lock().unwrap().last_submitted_seq = signal_timeline;
            }
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
        if !finish.render_pass_submitted {
            if let Some(scratch) = state
                .surfaces
                .get(&surface_handle)
                .and_then(|s| s.scratch_texture_slots.get(present_slot))
                .and_then(|slot| slot.as_ref())
            {
                if let Some(tex) = state.textures.read().unwrap().entries.get(&scratch.texture_handle) {
                    tex.set_image_layout(vk::ImageLayout::GENERAL);
                }
            }
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
    compute_timeline_sem: vk::Semaphore,
    owner_timeline_sem: vk::Semaphore,
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
        if !self.render_pass_submitted {
            let copy_signal_timeline = {
                let _queue_guard = queue_lock.lock().unwrap();
                let signal_timeline_value = super::context::reserve_device_owner_timeline_locked(&self.logical_device);
                signal_timeline = Some(signal_timeline_value);
                copy_timeline = Some(signal_timeline_value);
                signal_timeline_value
            };
            let sig_render_finished = vk::SemaphoreSubmitInfo::default()
                .semaphore(self.render_finished_sem)
                .value(0)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
            let sig_timeline = vk::SemaphoreSubmitInfo::default()
                .semaphore(self.owner_timeline_sem)
                .value(copy_signal_timeline)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
            let signals = vec![sig_render_finished, sig_timeline];
            super::pending_submit::enqueue_vulkan_present_copy(
                &self.logical_device,
                self.logical_device.queue,
                queue_lock.clone(),
                self.copy_cb,
                self.compute_timeline_sem,
                self.frame_compute_timeline_value,
                self.image_available_sem,
                self.render_finished_sem,
                copy_signal_timeline,
                signals,
            )?;
            self.logical_device
                .submission_worker
                .wait_submitted(copy_signal_timeline)?;
            self.logical_device.submission_worker.check_error()?;
        }
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
            scratch_layout_updated: !self.render_pass_submitted,
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
