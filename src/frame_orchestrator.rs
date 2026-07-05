//! Pipelined frame lifecycle: ring of in-flight frames, depth cap, and deferred retirement.
//!
//! [`FrameOrchestrator`] centralizes the pattern of pushing per-frame cleanup bundles keyed by
//! [`TimelineValue`], including swapchain paths where the epoch arrives only after
//! [`crate::surface::Frame::present`].

use crate::context::Context;
use crate::device::Device;
use crate::error::GoldyError;
use crate::surface::{Frame, Surface};
use crate::task_graph::TaskGraph;
use crate::timeline::TimelineValue;
use crate::tracy_frame_mark;
use crate::tracy_zone;
use anyhow::anyhow;
use std::collections::VecDeque;

/// Token returned from [`FrameOrchestrator::begin_frame`]; must be passed to
/// [`FrameOrchestrator::flush`], [`FrameOrchestrator::end_frame_standalone`],
/// [`FrameOrchestrator::end_frame_for_present`], or
/// [`FrameOrchestrator::end_frame_for_surface`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameHandle(pub(crate) u64);

/// A completed frame slot ready for application-specific cleanup.
#[derive(Debug)]
pub struct RetiredFrame<T> {
    /// GPU timeline epoch used for pool / allocator retirement (may be `0` if unknown).
    pub timeline: TimelineValue,
    /// User payload supplied when the frame was closed.
    pub data: T,
}

struct FrameSlot<T> {
    timeline: Option<TimelineValue>,
    /// `true` when `timeline` is a device-queue epoch (present/copy); otherwise context submit.
    timeline_is_device: bool,
    data: T,
}

/// Owns an in-flight ring of per-frame cleanup slots and enforces a maximum pipelining depth.
///
/// Typical use:
/// 1. [`Self::begin_frame`] (with retire closure)  
/// 2. Record dispatch into one or more [`TaskGraph`]s; call [`Self::flush`] for mid-frame submits.  
/// 3. [`Self::end_frame_standalone`], [`Self::end_frame_for_present`], or [`Self::end_frame_for_surface`].  
/// 4. For swapchain frames, [`Self::note_presented`] after [`Frame::present`].
pub struct FrameOrchestrator<T> {
    context: Context,
    max_depth: usize,
    ring: VecDeque<FrameSlot<T>>,
    /// Monotonic id for the next [`FrameHandle`].
    next_id: u64,
    /// `Some` between [`Self::begin_frame`] and a matching end-frame call.
    open: Option<FrameHandle>,
    /// Retention fingerprint stored by the most recent successful [`Self::submit_with_retention`]
    /// call.  `None` until the first retain-eligible submit.  Used when `max_depth == 1` to
    /// attempt zero-cost resubmission on subsequent frames.
    last_retention_key: Option<u64>,
    /// When true, [`Self::begin_frame`] retires full ring slots for depth backpressure
    /// without blocking on their stamped timelines. Callers must enforce per-parcel reuse
    /// ordering elsewhere (ekrano remediation step 3b).
    skip_ring_gpu_wait: bool,
}

impl<T> FrameOrchestrator<T> {
    /// Create an orchestrator. `max_depth` bounds how many frames may be in flight before the
    /// next [`Self::begin_frame`] blocks or forces the oldest slot to retire.
    pub fn new(context: &Context, max_depth: usize) -> Self {
        Self {
            context: context.clone(),
            max_depth: max_depth.max(1),
            ring: VecDeque::new(),
            next_id: 1,
            open: None,
            last_retention_key: None,
            skip_ring_gpu_wait: false,
        }
    }

    /// Skip the GPU wait in [`Self::begin_frame`] when the ring is at capacity.
    ///
    /// Depth backpressure and retirement still run; only the coarse
    /// `wait_until(stamped_timeline)` is omitted so per-parcel reuse gates can
    /// serialize at take time.
    pub fn with_skip_ring_gpu_wait(mut self, skip: bool) -> Self {
        self.skip_ring_gpu_wait = skip;
        self
    }

    /// `true` when depth is 1 and command-buffer retention may be used.
    #[inline]
    pub fn retains_command_buffers(&self) -> bool {
        self.max_depth == 1
    }

    /// Maximum number of in-flight frame slots (configured at construction).
    #[inline]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Current number of slots waiting on the GPU or a swapchain present timeline.
    #[inline]
    pub fn pending_frames(&self) -> usize {
        self.ring.len()
    }

    /// `true` if [`Self::begin_frame`] was called and the frame was not yet ended.
    #[inline]
    pub fn has_open_frame(&self) -> bool {
        self.open.is_some()
    }

    /// Discard the currently open frame without pushing a cleanup slot.
    ///
    /// Call this when a `run_frame` error makes it impossible to call `end_frame_standalone` or
    /// `end_frame_for_surface`. Leaves the ring intact so subsequent frames can begin normally.
    pub fn abort_frame(&mut self, handle: FrameHandle) {
        if self.open == Some(handle) {
            self.open = None;
        }
    }

    /// Non-blocking drain of slots whose GPU timeline has completed, plus mandatory pops when the
    /// ring is deeper than [`Self::max_depth`]. Same ordering rules as a manual cleanup deque.
    pub fn reclaim<E, F>(&mut self, mut retire: F) -> Result<(), GoldyError>
    where
        E: std::fmt::Display,
        F: FnMut(&Device, RetiredFrame<T>) -> Result<(), E>,
    {
        self.drain_ring_with_retire(&mut retire)
    }

    /// Begin recording a new frame: reclaims completed slots, then returns a handle if there is
    /// capacity (possibly after blocking on the oldest in-flight work).
    ///
    /// # Errors
    ///
    /// Returns [`GoldyError`] if a frame is already open (missing end-frame call), or if
    /// `retire` returns an error.
    pub fn begin_frame<E, F>(&mut self, mut retire: F) -> Result<FrameHandle, GoldyError>
    where
        E: std::fmt::Display,
        F: FnMut(&Device, RetiredFrame<T>) -> Result<(), E>,
    {
        let _tz = tracy_zone!("orchestrator.begin_frame");
        if self.open.is_some() {
            return Err(GoldyError::Backend(anyhow!(
                "FrameOrchestrator::begin_frame: a frame is already open"
            )));
        }
        self.drain_ring_with_retire(&mut retire)?;
        let h = FrameHandle(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.open = Some(h);
        Ok(h)
    }

    /// Mid-frame submit: dispatches the current graph and starts a fresh one.
    ///
    /// When [`Self::retains_command_buffers`] is true the graph is submitted via
    /// [`crate::Context::submit_pipelined_and_retain`] and on subsequent frames zero-cost
    /// resubmission is attempted first via [`crate::Context::try_resubmit_retained`].
    /// All other strategies use [`crate::Context::submit_pipelined`] unconditionally.
    ///
    /// This creates a real command-buffer boundary on all backends, including windowed/
    /// surface frames where the fine pass and present are submitted later via
    /// [`Self::end_frame_for_surface`].  Because Metal (and Vulkan/DX12) execute command
    /// buffers on the same queue in submission order, the fine command buffer automatically
    /// waits for the coarse one to complete — no explicit fence is required.
    pub fn flush(
        &mut self,
        handle: FrameHandle,
        graph: &mut TaskGraph,
        last_timeline: &mut Option<TimelineValue>,
    ) -> Result<(), GoldyError> {
        let _tz = tracy_zone!("orchestrator.flush");
        self.expect_open(handle)?;
        if graph.is_empty() {
            return Ok(());
        }
        let tv = self.submit_with_retention(graph)?;
        *last_timeline = Some(tv);
        // TODO(retained-graph): clear()+rebuild each flush — see `TaskGraph::clear` docs.
        graph.clear();
        Ok(())
    }

    /// End a standalone (headless / render-to-texture) frame: submit remaining work, push the
    /// cleanup slot with a known timeline, clear the open handle.
    ///
    /// When [`Self::retains_command_buffers`] is true the same retention logic as [`Self::flush`] applies:
    /// zero-cost resubmission is attempted first, falling back to a retain-and-record submit.
    ///
    /// If `graph` is empty, `fallback_timeline` is used (e.g. the value from previous
    /// [`Self::flush`] calls). If that is `None`, [`Context::gpu_progress`](crate::Context::gpu_progress) is used.
    pub fn end_frame_standalone(
        &mut self,
        handle: FrameHandle,
        graph: &mut TaskGraph,
        fallback_timeline: Option<TimelineValue>,
        cleanup: T,
    ) -> Result<TimelineValue, GoldyError> {
        let _tz = tracy_zone!("orchestrator.end_frame_standalone");
        self.expect_open(handle)?;
        let tv = if graph.is_empty() {
            fallback_timeline.unwrap_or_else(|| self.context.gpu_progress())
        } else {
            self.submit_with_retention(graph)?
        };
        self.ring.push_back(FrameSlot {
            timeline: Some(tv),
            timeline_is_device: false,
            data: cleanup,
        });
        self.open = None;
        tracy_frame_mark!();
        Ok(tv)
    }

    /// End a frame whose scanout is deferred to [`crate::surface::Frame::present`] or
    /// [`crate::Grant::consume`] on a [`crate::PresentGrant`].
    ///
    /// Same retirement semantics as [`Self::end_frame_for_surface`]: pushes a ring slot
    /// whose timeline is filled later via [`Self::note_presented`], and does **not** emit a
    /// Tracy frame mark (the mark belongs at present time, like the TaskGraph surface path).
    pub fn end_frame_for_present(
        &mut self,
        handle: FrameHandle,
        submit_timeline: TimelineValue,
        cleanup: T,
    ) -> Result<TimelineValue, GoldyError> {
        let _tz = tracy_zone!("orchestrator.end_frame_for_present");
        self.expect_open(handle)?;
        self.ring.push_back(FrameSlot {
            timeline: None,
            timeline_is_device: false,
            data: cleanup,
        });
        self.open = None;
        Ok(submit_timeline)
    }

    /// Submit `graph`, using command-buffer retention when [`Self::retains_command_buffers`].
    ///
    /// - **depth == 1**: tries [`crate::Context::try_resubmit_retained`] first
    ///   (zero CPU recording cost on a hit); on a miss (first frame or fingerprint change)
    ///   falls back to [`crate::Context::submit_pipelined_and_retain`] so the next frame can hit.
    ///   The last successful retention key is stored in `self.last_retention_key`.
    /// - **All other strategies**: always [`crate::Context::submit_pipelined`].  Retention would be
    ///   unsafe at pipeline depth > 1 because the same CB can still be in-flight from the
    ///   previous frame when a new submission begins.
    fn submit_with_retention(&mut self, graph: &mut TaskGraph) -> Result<TimelineValue, GoldyError> {
        if !self.retains_command_buffers() {
            return self.context.submit_pipelined(graph);
        }

        // Compute the full content fingerprint — covers pipeline, dispatch dims, push constants.
        let fp = graph.compute_retention_fingerprint();

        // Attempt zero-cost resubmission when the fingerprint matches the stored CB.
        if self.last_retention_key == Some(fp) {
            match self.context.try_resubmit_retained(fp)? {
                Some(tv) => {
                    tracing::trace!(
                        target: "goldy::retention",
                        key = fp,
                        "retention hit — resubmitted without re-recording"
                    );
                    return Ok(tv);
                }
                None => {
                    // Key matched but CB was evicted (e.g. a previous fallback path);
                    // fall through to record-and-retain below.
                    tracing::trace!(
                        target: "goldy::retention",
                        key = fp,
                        "retention miss — CB evicted, re-recording"
                    );
                }
            }
        } else {
            tracing::trace!(
                target: "goldy::retention",
                old_key = ?self.last_retention_key,
                new_key = fp,
                "retention fingerprint changed — re-recording"
            );
        }

        // Record the command buffer and store it for the next frame.
        let tv = self.context.submit_pipelined_and_retain(graph)?;
        self.last_retention_key = Some(fp);
        Ok(tv)
    }

    /// End a surface frame: submit the graph via [`Surface::submit_graph`]
    /// (deferred acquire), push a ring slot whose timeline is filled later by
    /// [`Self::note_presented`], and return the acquired [`Frame`] for the
    /// caller to present.
    ///
    /// When the graph is empty, falls back to [`Surface::begin`] so the caller
    /// still receives a `Frame` (useful for cleared-only frames).
    pub fn end_frame_for_surface(
        &mut self,
        handle: FrameHandle,
        graph: &mut TaskGraph,
        surface: &Surface,
        cleanup: T,
    ) -> Result<Frame, GoldyError> {
        let _tz = tracy_zone!("orchestrator.end_frame_for_surface");
        self.expect_open(handle)?;
        let frame = if graph.is_empty() {
            surface.begin().map_err(GoldyError::from)?
        } else {
            surface.submit_graph(graph).map_err(GoldyError::from)?
        };
        self.ring.push_back(FrameSlot {
            timeline: None,
            timeline_is_device: false,
            data: cleanup,
        });
        self.open = None;
        Ok(frame)
    }

    /// End a surface frame using a frame that was acquired before graph recording began.
    ///
    /// This keeps the same retirement semantics as [`Self::end_frame_for_surface`] while allowing
    /// callers to overlap WSI image acquisition with CPU graph construction and early GPU work.
    pub fn end_frame_for_acquired_surface(
        &mut self,
        handle: FrameHandle,
        graph: &mut TaskGraph,
        surface: &Surface,
        frame: Frame,
        cleanup: T,
    ) -> Result<Frame, GoldyError> {
        let _tz = tracy_zone!("orchestrator.end_frame_for_acquired_surface");
        self.expect_open(handle)?;
        let frame = if graph.is_empty() {
            frame
        } else {
            surface.submit_graph_to_frame(graph, frame).map_err(GoldyError::from)?
        };
        self.ring.push_back(FrameSlot {
            timeline: None,
            timeline_is_device: false,
            data: cleanup,
        });
        self.open = None;
        Ok(frame)
    }

    /// After [`Frame::present`], stamp the most recent surface slot with the returned timeline.
    ///
    /// Present timelines retire on the device queue, not the owning context's compute fence.
    pub fn note_presented(&mut self, tv: TimelineValue) {
        if let Some(back) = self.ring.back_mut() {
            if back.timeline.is_none() {
                back.timeline = Some(tv);
                back.timeline_is_device = true;
            }
        }
    }

    /// Block until every pending slot has retired and invoke `retire` for each.
    ///
    /// Slots whose timeline is still unknown (`None`, i.e. surface path before
    /// [`Self::note_presented`]) use [`crate::Context::high_water_timeline`] as a completion fence —
    /// callers draining mid-presentation should prefer [`Self::reclaim`] / presenting first.
    pub fn drain_all<E, F>(&mut self, mut retire: F) -> Result<(), GoldyError>
    where
        E: std::fmt::Display,
        F: FnMut(&Device, RetiredFrame<T>) -> Result<(), E>,
    {
        while let Some(slot) = self.ring.pop_front() {
            let timeline = match slot.timeline {
                Some(t) => t,
                None => self.context.high_water_timeline().max(self.context.gpu_progress()),
            };
            if slot.timeline_is_device {
                let device = self.context.device();
                if device.timeline_retired() < timeline {
                    device
                        .wait_until_retired(timeline)
                        .map_err(|e| GoldyError::Backend(anyhow!("{e}")))?;
                }
            } else if self.context.gpu_progress() < timeline {
                self.context.wait_until(timeline)?;
            }
            retire(
                self.context.device(),
                RetiredFrame {
                    timeline,
                    data: slot.data,
                },
            )
            .map_err(|e| GoldyError::Backend(anyhow!("{e}")))?;
        }
        Ok(())
    }

    fn expect_open(&self, handle: FrameHandle) -> Result<(), GoldyError> {
        match self.open {
            Some(h) if h == handle => Ok(()),
            Some(_) => Err(GoldyError::Backend(anyhow!(
                "FrameOrchestrator: wrong FrameHandle for this frame"
            ))),
            None => Err(GoldyError::Backend(anyhow!(
                "FrameOrchestrator: no open frame (call begin_frame first)"
            ))),
        }
    }

    fn drain_ring_with_retire<E, F>(&mut self, retire: &mut F) -> Result<(), GoldyError>
    where
        E: std::fmt::Display,
        F: FnMut(&Device, RetiredFrame<T>) -> Result<(), E>,
    {
        let _tz = tracy_zone!("orchestrator.drain_ring");
        let device = self.context.device();
        let mut ctx_progress = {
            let _pg = tracy_zone!("orchestrator.drain_ring.gpu_progress");
            self.context.gpu_progress()
        };
        let mut device_progress = device.timeline_retired();
        while let Some(front) = self.ring.front() {
            let done = match front.timeline {
                Some(tv) if front.timeline_is_device => device_progress >= tv,
                Some(tv) => ctx_progress >= tv,
                None => false,
            };
            let must_wait = !done && self.ring.len() >= self.max_depth;
            if done || must_wait {
                let slot = self.ring.pop_front().unwrap();
                if let Some(tv) = slot.timeline {
                    if must_wait && !self.skip_ring_gpu_wait {
                        if slot.timeline_is_device {
                            if device_progress < tv {
                                let _wz = tracy_zone!("orchestrator.wait_gpu");
                                device
                                    .wait_until_retired(tv)
                                    .map_err(|e| GoldyError::Backend(anyhow!("{e}")))?;
                                device_progress = device.timeline_retired().max(tv);
                            }
                        } else if ctx_progress < tv {
                            let _wz = tracy_zone!("orchestrator.wait_gpu");
                            self.context.wait_until(tv)?;
                            ctx_progress = self.context.gpu_progress().max(tv);
                        }
                    }
                }
                let timeline = slot.timeline.unwrap_or(0);
                {
                    let _rz = tracy_zone!("orchestrator.retire_cb");
                    retire(
                        self.context.device(),
                        RetiredFrame {
                            timeline,
                            data: slot.data,
                        },
                    )
                    .map_err(|e| GoldyError::Backend(anyhow!("{e}")))?;
                }
            } else {
                break;
            }
        }
        Ok(())
    }
}
