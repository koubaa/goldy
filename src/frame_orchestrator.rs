//! Pipelined frame lifecycle: ring of in-flight frames, depth cap, and deferred retirement.
//!
//! [`FrameOrchestrator`] centralizes the pattern of pushing per-frame cleanup bundles keyed by
//! [`TimelineValue`], including swapchain paths where the epoch arrives only after
//! [`crate::surface::Frame::present`].

use crate::context::Context;
use crate::device::Device;
use crate::error::GoldyError;
use crate::timeline::TimelineValue;
use crate::tracy_frame_mark;
use crate::tracy_zone;
use anyhow::anyhow;
use std::collections::VecDeque;

/// Token returned from [`FrameOrchestrator::begin_frame`]; must be passed to
/// [`FrameOrchestrator::end_frame_standalone`], [`FrameOrchestrator::end_frame_for_present`],
/// [`FrameOrchestrator::end_frame_externally_ordered`], or [`FrameOrchestrator::abort_frame`].
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
    data: T,
}

/// Owns an in-flight ring of per-frame cleanup slots and enforces a maximum pipelining depth.
///
/// Typical use:
/// 1. [`Self::begin_frame`] (with retire closure)
/// 2. Record and submit work via [`crate::Scheme`]
/// 3. [`Self::end_frame_standalone`], [`Self::end_frame_for_present`], or
///    [`Self::end_frame_externally_ordered`]
/// 4. For swapchain frames, [`Self::note_presented`] after present.
pub struct FrameOrchestrator<T> {
    context: Context,
    max_depth: usize,
    ring: VecDeque<FrameSlot<T>>,
    /// Monotonic id for the next [`FrameHandle`].
    next_id: u64,
    /// `Some` between [`Self::begin_frame`] and a matching end-frame call.
    open: Option<FrameHandle>,
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
        }
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
    /// Call this when a `run_frame` error makes it impossible to call an end-frame
    /// method. Leaves the ring intact so subsequent frames can begin normally.
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

    /// End a standalone (headless / render-to-texture) frame whose GPU work was already
    /// submitted (e.g. via [`crate::Scheme::submit`]).
    ///
    /// Pushes a retirement-ring slot stamped with `timeline`, clears the open handle, and
    /// emits a Tracy frame mark.
    pub fn end_frame_standalone(
        &mut self,
        handle: FrameHandle,
        timeline: TimelineValue,
        cleanup: T,
    ) -> Result<TimelineValue, GoldyError> {
        let _tz = tracy_zone!("orchestrator.end_frame_standalone");
        self.expect_open(handle)?;
        self.ring.push_back(FrameSlot {
            timeline: Some(timeline),
            data: cleanup,
        });
        self.open = None;
        tracy_frame_mark!();
        Ok(timeline)
    }

    /// End a frame whose scanout is deferred to [`crate::surface::Frame::present`] or
    /// [`crate::Claim::consume`].
    ///
    /// Same retirement semantics as a surface frame closed before present: pushes a ring slot
    /// whose timeline is filled later via [`Self::note_presented`], and does **not** emit a
    /// Tracy frame mark (the mark belongs at present time).
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
            data: cleanup,
        });
        self.open = None;
        Ok(submit_timeline)
    }

    /// Close an open frame without creating a retirement-ring slot.
    ///
    /// Use when cross-frame resource ordering is enforced externally (scheme reuse epochs,
    /// deferred host writes, present-easement ledger) so `begin_frame` must not wait on a
    /// coarse frame timeline. The open handle is cleared; no Tracy frame mark is emitted.
    pub fn end_frame_externally_ordered(&mut self, handle: FrameHandle) -> Result<(), GoldyError> {
        let _tz = tracy_zone!("orchestrator.end_frame_externally_ordered");
        self.expect_open(handle)?;
        self.open = None;
        Ok(())
    }

    /// After [`crate::Frame::present`], stamp the most recent surface slot with the returned timeline.
    pub fn note_presented(&mut self, tv: TimelineValue) {
        if let Some(back) = self.ring.back_mut() {
            if back.timeline.is_none() {
                back.timeline = Some(tv);
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
            if self.context.gpu_progress() < timeline {
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
        let mut progress = {
            let _pg = tracy_zone!("orchestrator.drain_ring.gpu_progress");
            self.context.gpu_progress()
        };
        while let Some(front) = self.ring.front() {
            let done = match front.timeline {
                Some(tv) => progress >= tv,
                None => false,
            };
            let must_wait = !done && self.ring.len() >= self.max_depth;
            if done || must_wait {
                let slot = self.ring.pop_front().unwrap();
                if let Some(tv) = slot.timeline {
                    if progress < tv && must_wait {
                        let _wz = tracy_zone!("orchestrator.wait_gpu");
                        self.context.wait_until(tv)?;
                        progress = tv;
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
