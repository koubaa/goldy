//! Async GPU submission work enqueued on the per-device submission worker.
//!
//! Uses `cocoa::base::id` for drawable retain/release; see `surface.rs` for the
//! objc2 migration note.
#![allow(deprecated)]

use super::super::SurfaceHandle;
use super::types::{SharedLogicalDevice, SharedMetalSubmissionContext, TimelineWaiter};
use crate::backend::submission_worker::{allocate_timeline_value, PendingSubmit};
use crate::timeline::TimelineValue;
use ::metal as mtl;
use anyhow::Result;
use block::ConcreteBlock;
use cocoa::base::id;
use objc::{msg_send, sel, sel_impl};
use std::sync::atomic::Ordering;
use std::time::Instant;

pub(super) fn preallocate_device_timeline(ld: &SharedLogicalDevice) -> TimelineValue {
    let v = allocate_timeline_value(&ld.timeline_next);
    ld.timeline_scheduled_max.fetch_max(v, Ordering::Relaxed);
    v
}

fn track_in_flight_cb(
    sc_arc: &SharedMetalSubmissionContext,
    signal_value: TimelineValue,
    command_buffer: &mtl::CommandBuffer,
) {
    let mut sc = sc_arc.lock().unwrap();
    sc.in_flight_command_buffers
        .push_back((signal_value, command_buffer.to_owned()));
    super::drain_completed_cbs(&mut sc);
}

fn untrack_in_flight_cb(sc_arc: &SharedMetalSubmissionContext, signal_value: TimelineValue) {
    let mut sc = sc_arc.lock().unwrap();
    if sc
        .in_flight_command_buffers
        .back()
        .is_some_and(|(tv, _)| *tv == signal_value)
    {
        sc.in_flight_command_buffers.pop_back();
    }
}

struct MetalCommitPendingSubmit {
    logical_device: SharedLogicalDevice,
    command_buffer: mtl::CommandBuffer,
    signal_value: TimelineValue,
    timeline_event: mtl::SharedEvent,
    waiter: TimelineWaiter,
    log_kind: &'static str,
    api_log_commit: bool,
    compute_commit_instant: Option<Instant>,
}

impl PendingSubmit for MetalCommitPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.mtl.commit", self.log_kind);
        let command_buffer_ref = self.command_buffer.as_ref();
        let _queue_guard = self.logical_device.queue_lock.lock().unwrap();
        if self.api_log_commit && super::api_log::enabled() {
            super::api_log::log_commit(self.signal_value);
        }
        let signal_value = self.signal_value;
        let waiter = self.waiter.clone();
        let log_kind = self.log_kind;
        let commit_instant = self.compute_commit_instant;
        let handler = ConcreteBlock::new(move |cb: &mtl::CommandBufferRef| {
            let status = cb.status();
            if status != mtl::MTLCommandBufferStatus::Completed {
                let description = super::compute::read_command_buffer_error_description(cb);
                tracing::error!(
                    "GPU command buffer ({log_kind}, timeline={signal_value}) finished with status={status:?}: {description}"
                );
            }
            if let Some(start) = commit_instant {
                let cpu_lifetime = start.elapsed();
                let (gpu_start, gpu_end): (f64, f64) = unsafe {
                    (
                        msg_send![cb, GPUStartTime],
                        msg_send![cb, GPUEndTime],
                    )
                };
                let gpu_ms = (gpu_end - gpu_start) * 1000.0;
                tracing::debug!(
                    "[mtl.cb_done] kind={log_kind} signal_value={signal_value} commit_to_complete={cpu_lifetime:?} gpu_exec={gpu_ms:.3}ms"
                );
                if crate::gpu_profiler::gpu_profile_enabled() {
                    crate::gpu_profiler::log_cb_timing("metal", signal_value, gpu_ms);
                }
            } else if crate::gpu_profiler::gpu_profile_enabled() {
                let gpu_start: f64 = unsafe { msg_send![cb, GPUStartTime] };
                let gpu_end: f64 = unsafe { msg_send![cb, GPUEndTime] };
                let ms = (gpu_end - gpu_start) * 1000.0;
                crate::gpu_profiler::log_cb_timing("metal", signal_value, ms);
            }
            waiter.signal(signal_value);
        })
        .copy();
        command_buffer_ref.add_completed_handler(&handler);
        command_buffer_ref.encode_signal_event(self.timeline_event.as_ref(), signal_value);
        tracing::debug!(
            "[mtl.cb_commit] kind={} signal_value={signal_value} queue=command_queue",
            self.log_kind
        );
        {
            let _commit = crate::tracy_zone!("goldy.submit_worker.mtl.execute_and_signal");
            command_buffer_ref.commit();
        }

        Ok(())
    }
}

pub(super) fn enqueue_metal_commit(
    ld: &SharedLogicalDevice,
    command_buffer: mtl::CommandBuffer,
    signal_value: TimelineValue,
    timeline_event: mtl::SharedEvent,
    waiter: TimelineWaiter,
    sc_arc: Option<SharedMetalSubmissionContext>,
    log_kind: &'static str,
    api_log_commit: bool,
    compute_commit_instant: Option<Instant>,
) -> Result<()> {
    ld.submission_worker.check_error()?;
    if let Some(ref sc_arc) = sc_arc {
        track_in_flight_cb(sc_arc, signal_value, &command_buffer);
    }
    match ld.submission_worker.enqueue(
        signal_value,
        Box::new(MetalCommitPendingSubmit {
            logical_device: std::sync::Arc::clone(ld),
            command_buffer,
            signal_value,
            timeline_event,
            waiter,
            log_kind,
            api_log_commit,
            compute_commit_instant,
        }),
    ) {
        Ok(()) => Ok(()),
        Err(e) => {
            if let Some(ref sc_arc) = sc_arc {
                untrack_in_flight_cb(sc_arc, signal_value);
            }
            Err(e)
        }
    }
}

struct MetalPresentPendingSubmit {
    logical_device: SharedLogicalDevice,
    command_buffer: mtl::CommandBuffer,
    signal_value: TimelineValue,
    timeline_event: mtl::SharedEvent,
    waiter: TimelineWaiter,
    surface: SurfaceHandle,
    drawable_ptr: usize,
    return_image: Option<u32>,
    signal_queue_present: std::sync::Arc<crate::signal::SignalQueue>,
    return_pending: std::sync::Arc<std::sync::Mutex<Vec<(SurfaceHandle, u32)>>>,
}

impl PendingSubmit for MetalPresentPendingSubmit {
    fn execute(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.submit_worker.mtl.present");

        let command_buffer = self.command_buffer.as_ref();
        let _queue_guard = self.logical_device.queue_lock.lock().unwrap();
        let signal_value = self.signal_value;
        let drawable_ptr = self.drawable_ptr as id;
        let drawable_ref: &mtl::DrawableRef = unsafe { &*(drawable_ptr as *const mtl::DrawableRef) };

        command_buffer.encode_signal_event(self.timeline_event.as_ref(), signal_value);
        let return_image = self.return_image;
        let surface = self.surface;
        let signal_queue_present = std::sync::Arc::clone(&self.signal_queue_present);
        let return_pending = std::sync::Arc::clone(&self.return_pending);
        let waiter = self.waiter.clone();
        let handler = ConcreteBlock::new(move |_cb: &mtl::CommandBufferRef| {
            waiter.signal(signal_value);
            if let Some(idx) = return_image {
                signal_queue_present.push(crate::signal::Signal::SwapchainReturned { image_index: idx });
                if let Ok(mut pending) = return_pending.lock() {
                    pending.push((surface, idx));
                }
            }
        })
        .copy();
        command_buffer.add_completed_handler(&handler);
        command_buffer.present_drawable(drawable_ref);
        if super::api_log::enabled() {
            super::api_log::log_present_drawable(signal_value);
        }
        {
            let _commit = crate::tracy_zone!("goldy.submit_worker.mtl.execute_and_signal");
            command_buffer.commit();
        }

        unsafe {
            let (): () = msg_send![drawable_ptr, release];
        }

        Ok(())
    }
}

pub(super) fn enqueue_metal_present(
    ld: &SharedLogicalDevice,
    command_buffer: mtl::CommandBuffer,
    signal_value: TimelineValue,
    timeline_event: mtl::SharedEvent,
    waiter: TimelineWaiter,
    sc_arc: SharedMetalSubmissionContext,
    surface: SurfaceHandle,
    drawable_ptr: usize,
    return_image: Option<u32>,
    signal_queue_present: std::sync::Arc<crate::signal::SignalQueue>,
    return_pending: std::sync::Arc<std::sync::Mutex<Vec<(SurfaceHandle, u32)>>>,
) -> Result<()> {
    ld.timeline_scheduled_max.fetch_max(signal_value, Ordering::Relaxed);
    ld.submission_worker.check_error()?;
    track_in_flight_cb(&sc_arc, signal_value, &command_buffer);
    match ld.submission_worker.enqueue(
        signal_value,
        Box::new(MetalPresentPendingSubmit {
            logical_device: std::sync::Arc::clone(ld),
            command_buffer,
            signal_value,
            timeline_event,
            waiter,
            surface,
            drawable_ptr,
            return_image,
            signal_queue_present,
            return_pending,
        }),
    ) {
        Ok(()) => Ok(()),
        Err(e) => {
            untrack_in_flight_cb(&sc_arc, signal_value);
            Err(e)
        }
    }
}
