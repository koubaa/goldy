//! Pipelined frame lifecycle: ring of in-flight frames, depth cap, and deferred retirement.
//!
//! [`FrameOrchestrator`] centralizes the pattern of pushing per-frame cleanup bundles keyed by
//! [`TimelineValue`], including swapchain paths where the epoch arrives only after
//! [`crate::surface::Frame::present`].

use crate::device::Device;
use crate::error::GoldyError;
use crate::surface::Frame;
use crate::task_graph::TaskGraph;
use crate::timeline::TimelineValue;
use crate::tracy_frame_mark;
use crate::tracy_zone;
use anyhow::anyhow;
use std::collections::VecDeque;

/// Semantic hint for how the frame pipeline should be scheduled.
///
/// The application chooses based on its tolerance for input lag vs. desire for
/// throughput. Goldy selects the optimal internal strategy (pipelining depth,
/// buffer allocation, CB retention) accordingly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameStrategy {
    /// Minimize latency. Frames are fully synchronous: the CPU blocks until the
    /// GPU finishes each frame before starting the next. CPU recording overhead
    /// is eliminated via command buffer retention when the binding structure is
    /// stable across frames.
    ///
    /// Best for: interactive applications, games, live previews.
    LowLatency,

    /// Balanced throughput. Two frames overlap: the CPU records frame N+1 while
    /// the GPU executes frame N. VRAM stays bounded via graph coloring across
    /// staggered pipeline stages.
    ///
    /// Best for: applications that can tolerate one frame of input lag.
    Balanced,

    /// Maximum throughput. Deeply pipelined: the CPU is always recording ahead
    /// of the GPU. Latency is highest but FPS ceiling is maximized.
    ///
    /// Best for: offline rendering, benchmarks, batch processing.
    MaxThroughput {
        /// Maximum number of frames allowed in-flight. Defaults to 3 if `None`.
        max_frames_in_flight: Option<u32>,
    },
}

impl FrameStrategy {
    /// Derive the pipelining depth from the strategy.
    pub fn depth(&self) -> usize {
        match self {
            FrameStrategy::LowLatency => 1,
            FrameStrategy::Balanced => 2,
            FrameStrategy::MaxThroughput {
                max_frames_in_flight,
            } => max_frames_in_flight.unwrap_or(3) as usize,
        }
    }

    /// Whether transient-buffer graph coloring should be used for VRAM reuse.
    ///
    /// At depth=1 (LowLatency) all buffers are persistent — graph coloring adds
    /// overhead with no benefit and prevents stable bindless indices needed for CB
    /// retention. At depth>1 graph coloring aliases physical memory across
    /// staggered pipeline stages.
    pub fn use_graph_coloring(&self) -> bool {
        self.depth() > 1
    }
}

/// Token returned from [`FrameOrchestrator::begin_frame`]; must be passed to
/// [`FrameOrchestrator::flush`], [`FrameOrchestrator::end_frame_standalone`], or
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
    data: T,
}

/// Owns an in-flight ring of per-frame cleanup slots and enforces a maximum pipelining depth.
///
/// Typical use:
/// 1. [`Self::begin_frame`] (with retire closure)  
/// 2. Record dispatch into one or more [`TaskGraph`]s; call [`Self::flush`] for mid-frame submits.  
/// 3. [`Self::end_frame_standalone`] or [`Self::end_frame_for_surface`].  
/// 4. For swapchain frames, [`Self::note_presented`] after [`Frame::present`].
pub struct FrameOrchestrator<T> {
    device: Device,
    strategy: FrameStrategy,
    max_depth: usize,
    ring: VecDeque<FrameSlot<T>>,
    /// Monotonic id for the next [`FrameHandle`].
    next_id: u64,
    /// `Some` between [`Self::begin_frame`] and a matching end-frame call.
    open: Option<FrameHandle>,
}

impl<T> FrameOrchestrator<T> {
    /// Create an orchestrator. `max_depth` bounds how many frames may be in flight before the
    /// next [`Self::begin_frame`] blocks or forces the oldest slot to retire (same role as
    /// `max_regions` on pooled transient allocators).
    pub fn new(device: &Device, max_depth: usize) -> Self {
        let strategy = match max_depth {
            0 | 1 => FrameStrategy::LowLatency,
            2 => FrameStrategy::Balanced,
            n => FrameStrategy::MaxThroughput {
                max_frames_in_flight: Some(n as u32),
            },
        };
        Self {
            device: device.clone(),
            strategy,
            max_depth: max_depth.max(1),
            ring: VecDeque::new(),
            next_id: 1,
            open: None,
        }
    }

    /// Create an orchestrator from a [`FrameStrategy`], which derives the
    /// pipelining depth automatically.
    pub fn with_strategy(device: &Device, strategy: FrameStrategy) -> Self {
        let max_depth = strategy.depth();
        Self {
            device: device.clone(),
            strategy,
            max_depth,
            ring: VecDeque::new(),
            next_id: 1,
            open: None,
        }
    }

    /// The frame scheduling strategy configured at construction.
    #[inline]
    pub fn strategy(&self) -> FrameStrategy {
        self.strategy
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
    /// Always uses [`Device::submit_pipelined`] so the GPU can begin executing the submitted
    /// work (e.g. coarse rasterization) while the CPU continues recording subsequent work
    /// (e.g. fine rasterization) into a new [`TaskGraph`].  This creates a real command-buffer
    /// boundary on all backends, including windowed/surface frames where the fine pass and
    /// present are submitted later via [`Self::end_frame_for_surface`].
    ///
    /// Because Metal (and Vulkan/DX12) execute command buffers on the same queue in submission
    /// order, the fine command buffer automatically waits for the coarse one to complete —
    /// no explicit fence is required.
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
        let tv = self.device.submit_pipelined(graph)?;
        *last_timeline = Some(tv);
        graph.clear();
        Ok(())
    }

    /// End a standalone (headless / render-to-texture) frame: submit remaining work, push the
    /// cleanup slot with a known timeline, clear the open handle.
    ///
    /// If `graph` is empty, `fallback_timeline` is used (e.g. the value from previous
    /// [`Self::flush`] calls). If that is `None`, [`Device::gpu_progress`] is used.
    pub fn end_frame_standalone(
        &mut self,
        handle: FrameHandle,
        mut graph: TaskGraph,
        fallback_timeline: Option<TimelineValue>,
        cleanup: T,
    ) -> Result<TimelineValue, GoldyError> {
        let _tz = tracy_zone!("orchestrator.end_frame_standalone");
        self.expect_open(handle)?;
        let tv = if graph.is_empty() {
            fallback_timeline.unwrap_or_else(|| self.device.gpu_progress())
        } else {
            self.device.submit_pipelined(&mut graph)?
        };
        self.ring.push_back(FrameSlot {
            timeline: Some(tv),
            data: cleanup,
        });
        self.open = None;
        tracy_frame_mark!();
        Ok(tv)
    }

    /// End a surface frame: optionally submits compute, then pushes a slot whose timeline is
    /// filled later via [`Self::note_presented`].
    pub fn end_frame_for_surface(
        &mut self,
        handle: FrameHandle,
        graph: &mut TaskGraph,
        frame: &Frame,
        cleanup: T,
    ) -> Result<(), GoldyError> {
        let _tz = tracy_zone!("orchestrator.end_frame_for_surface");
        self.expect_open(handle)?;
        if !graph.is_empty() {
            frame.submit_compute(graph).map_err(GoldyError::from)?;
        }
        self.ring.push_back(FrameSlot {
            timeline: None,
            data: cleanup,
        });
        self.open = None;
        Ok(())
    }

    /// After [`Frame::present`], stamp the most recent surface slot with the returned timeline.
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
    /// [`Self::note_presented`]) use [`Device::high_water_timeline`] as a completion fence —
    /// callers draining mid-presentation should prefer [`Self::reclaim`] / presenting first.
    pub fn drain_all<E, F>(&mut self, mut retire: F) -> Result<(), GoldyError>
    where
        E: std::fmt::Display,
        F: FnMut(&Device, RetiredFrame<T>) -> Result<(), E>,
    {
        while let Some(slot) = self.ring.pop_front() {
            let timeline = match slot.timeline {
                Some(t) => t,
                None => self
                    .device
                    .high_water_timeline()
                    .max(self.device.gpu_progress()),
            };
            if self.device.gpu_progress() < timeline {
                self.device.wait_until(timeline)?;
            }
            retire(
                &self.device,
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
        let mut progress = self.device.gpu_progress();
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
                        self.device.wait_until(tv)?;
                        progress = tv;
                    }
                }
                let timeline = slot.timeline.unwrap_or(0);
                {
                    let _rz = tracy_zone!("orchestrator.retire_cb");
                    retire(
                        &self.device,
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
