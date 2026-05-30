//! Pluggable transient-memory allocation strategies for pipelined rendering.
//!
//! A *transient* allocation is a sub-allocation from a GPU buffer whose lifetime is bounded by
//! the lifetime of the GPU command buffers that read it. Rendering pipelines typically need
//! to make many such allocations per frame (scratch buffers, per-pass storage, vertex
//! streams) and then reuse the memory in the next frame once the GPU is done with it.
//!
//! The naive approach — a single bump allocator that is `reset()`-ed at frame boundaries —
//! forces the CPU to wait for the previous frame's GPU work before it can begin recording the
//! next frame. Pipelined approaches need more sophisticated reuse strategies.
//!
//! This module defines a [`TransientAllocator`] trait so clients can pick (and switch between)
//! strategies without changing call sites. Strategies map onto the same kinds of patterns the
//! CPU memory-management world has spent decades exploring:
//!
//! * [`BumpResetAllocator`] — single bump pool, blocking reset between frames. Equivalent to
//!   a per-thread arena with synchronous reset. Minimal memory; serializes CPU and GPU.
//! * [`EpochRegionsAllocator`] — many bump regions tagged with [`TimelineValue`] epochs;
//!   reclamation is asynchronous via [`Device::gpu_progress`]. Closest analog: epoch-based
//!   reclamation (EBR) used by lock-free data structures and region-based garbage collectors
//!   like G1. Non-blocking hot path; pipeline depth adapts to workload.
//! * [`HeapTransientAllocator`] — single monolithic buffer with a real free list (best-fit
//!   with coalescing). Freed sub-allocations return to the free list once their GPU epoch
//!   retires, enabling mid-pipeline memory reuse. Peak memory ≈ peak live working set rather
//!   than sum of everything allocated.
//!
//! Selection at construction time uses [`TransientAllocatorStrategy`]. Consumers that want
//! a runtime switch (e.g. via environment variable) should read the variable themselves and
//! call [`TransientAllocatorStrategy::parse`] + [`TransientAllocatorStrategy::create`].
//!
//! # Lifecycle
//!
//! Each frame follows the same three-step pattern regardless of strategy:
//!
//! 1. [`TransientAllocator::begin_frame`] — opportunistic reclamation + capacity check
//! 2. [`TransientAllocator::alloc`] — repeated bump allocations
//! 3. [`TransientAllocator::end_frame`] — record the frame's epoch so the strategy can
//!    reclaim later
//!
//! **Important:** `end_frame` may be called *after* the next `begin_frame` when the epoch
//! is not known until after surface presentation. [`EpochRegionsAllocator`] handles this
//! correctly by tracking "pending" regions whose epoch has not yet been supplied.
//!
//! Strategies are free to over- or under-implement individual steps as long as semantics
//! are preserved. For example, `BumpResetAllocator::end_frame` only stores the epoch for the
//! next `begin_frame`'s wait; `EpochRegionsAllocator::end_frame` tags every region used.

use crate::buffer::{lcm, BufferPool, BufferView};
use crate::device::Device;
use crate::timeline::TimelineValue;
use crate::types::BufferFlags;
use anyhow::Result;
use std::collections::{BTreeMap, VecDeque};

// -----------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------

/// Configuration for a [`TransientAllocator`].
///
/// Strategies may ignore fields that don't apply (e.g. `min_region_size` is unused by
/// [`BumpResetAllocator`]). Default values are tuned for typical 2D-rendering workloads.
#[derive(Debug, Clone)]
pub struct TransientAllocatorConfig {
    /// Initial backing-storage size, in bytes. Allocators may grow beyond this on demand.
    pub initial_size: u64,
    /// Minimum region size for region-based strategies (e.g. [`EpochRegionsAllocator`]). Ignored
    /// by non-regional strategies. Default 4 MiB; smaller values trade region overhead for
    /// finer-grained reclamation.
    pub min_region_size: u64,
    /// Maximum number of regions before [`EpochRegionsAllocator`] falls back to waiting on the
    /// oldest in-flight region. Acts as a pipeline-depth cap. Ignored by other strategies.
    pub max_regions: usize,
    /// Sub-allocation alignment in bytes. Must be a power of two. 256 satisfies
    /// `minStorageBufferOffsetAlignment` on all supported backends.
    pub alignment: u64,
    /// Buffer flags applied to backing storage. [`BufferFlags::GPU_ONLY`] is typical for
    /// transient compute scratch where CPU access is never needed.
    pub flags: BufferFlags,
}

impl Default for TransientAllocatorConfig {
    fn default() -> Self {
        Self {
            initial_size: 64 * 1024,
            min_region_size: 4 * 1024 * 1024,
            max_regions: 3,
            alignment: 256,
            flags: BufferFlags::GPU_ONLY,
        }
    }
}

// -----------------------------------------------------------------------
// Trait
// -----------------------------------------------------------------------

/// A pluggable strategy for sub-allocating short-lived GPU buffers across rendering frames.
///
/// Implementations carve [`BufferView`]s out of one or more backing [`Buffer`](crate::buffer::Buffer)s and must
/// guarantee that memory returned from [`Self::alloc`] is safe for GPU consumption until at
/// least the corresponding [`Self::end_frame`]'s epoch has been reached on the device timeline.
///
/// See the module-level docs for the per-frame lifecycle and strategy comparison.
pub trait TransientAllocator: Send {
    /// Called once at the start of each frame, before any [`Self::alloc`] calls.
    ///
    /// Strategies use this to perform opportunistic reclamation, grow capacity if needed,
    /// and prepare the next frame's allocation state. Implementations *may* block here (e.g.
    /// [`BumpResetAllocator`] waits for the previous frame's GPU work) but a non-blocking
    /// implementation is preferred for steady-state throughput.
    ///
    /// `hint_size` is an optional pre-sizing hint — pass `0` if unknown.
    fn begin_frame(&mut self, device: &Device, hint_size: u64) -> Result<()>;

    /// Allocate `size` bytes for use within the current frame.
    ///
    /// Each allocation is aligned to the configured alignment (and `element_stride` if
    /// provided). The returned [`BufferView`] carries its own bindless descriptor so it can be
    /// bound to shaders independently.
    ///
    /// Implementations should be lock-free on the steady-state hot path. Allocation failure
    /// (capacity exhaustion that cannot be remedied by growth or reclamation) returns `Err`.
    fn alloc(
        &mut self,
        device: &Device,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferView>;

    /// Called once per frame after submitting all GPU work, with the timeline value the GPU
    /// will signal when that work completes.
    ///
    /// Strategies use this epoch to track when in-flight regions become safe to reclaim.
    /// For surface-presented frames, pass the [`TimelineValue`] returned by `Frame::present`.
    /// For headless rendering, pass the value returned by `Device::submit`.
    ///
    /// `device` is used to call [`Device::defer_release`] for epoch-based resource cleanup.
    fn end_frame(&mut self, device: &Device, epoch: TimelineValue);

    /// Total bytes of GPU memory currently held across all backing storage.
    fn capacity(&self) -> u64;

    /// Strategy identifier for diagnostics and tracing.
    fn name(&self) -> &'static str;

    /// Bytes consumed by allocations in the current frame so far. Default `0`.
    fn used_this_frame(&self) -> u64 {
        0
    }

    /// Hint that bytes at and above `offset` in the most-recently-active region are unused
    /// for the rest of this frame. Strategies may forward to [`Buffer::hint_unused_above`](crate::buffer::Buffer::hint_unused_above)
    /// to release physical pages. Default is a no-op.
    fn hint_unused_above(&mut self, _offset: u64) {}

    /// Return a sub-allocation's byte range to the allocator for reuse.
    ///
    /// `offset` and `size` identify the byte range within the backing buffer.
    /// `epoch` is the timeline value of the last GPU dispatch that reads this buffer. The
    /// allocator must not reuse the byte range until `device.gpu_progress() >= epoch`.
    /// If `epoch` is `None`, the range is immediately available (caller guarantees no GPU use).
    ///
    /// The caller retains ownership of the `BufferView` (and its bindless slot) for deferred
    /// cleanup — only the byte range is returned to the allocator.
    ///
    /// The default implementation is a no-op — bump-style allocators that reclaim entire
    /// regions at once simply ignore per-view frees.
    fn free(&mut self, _offset: u64, _size: u64, _epoch: Option<TimelineValue>) {}

    /// Release backing capacity that exceeds the observed peak working set.
    ///
    /// Implementations compact their free lists / watermarks and, where possible, downsize
    /// the backing buffer or hint the backend that trailing pages are unused. Safe to call
    /// at any time — live allocations are preserved.
    ///
    /// The default is a no-op; strategies that grow dynamically should override.
    fn shrink_to_fit(&mut self) {}

    /// Optional emergency reset — drops all backing storage and forgets any in-flight epochs.
    /// Callers are responsible for ensuring no GPU work still references this allocator's
    /// allocations.
    fn clear(&mut self) {}
}

// -----------------------------------------------------------------------
// Strategy enum + env-var switch
// -----------------------------------------------------------------------

/// Selectable allocation strategies. Use [`Self::create`] to instantiate.
///
/// The default is [`Self::EpochRegions`] — the pipelined strategy that allows CPU and GPU
/// to overlap. Use [`Self::BumpReset`] as a diagnostic baseline or when minimum memory
/// footprint is more important than throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransientAllocatorStrategy {
    /// Single bump pool that is `reset()`-ed each frame after waiting for the previous
    /// frame's GPU work to complete. Lowest memory usage; serializes CPU and GPU.
    BumpReset,
    /// Multiple bump regions tagged with [`TimelineValue`] epochs. Reclamation is
    /// asynchronous: a region becomes reusable as soon as its epoch is ≤ `gpu_progress()`,
    /// no waiting required. Pipeline depth adapts to workload up to `max_regions`.
    EpochRegions,
    /// Single monolithic buffer with a real free list. Mid-pipeline `free()` calls return
    /// sub-allocations to a deferred-free queue keyed by GPU timeline; once retired, ranges
    /// merge back into the free list for reuse within the same or subsequent frames.
    /// Peak memory ≈ peak live working set, not sum of all allocations.
    #[default]
    Heap,
}

impl TransientAllocatorStrategy {
    /// Parse a strategy name. Case-insensitive. Returns `None` for unrecognised values.
    ///
    /// Recognised values: `bump`, `bump_reset`, `epoch`, `epoch_regions`, `regions`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "bump" | "bump_reset" | "bumpreset" => Some(Self::BumpReset),
            "epoch" | "regions" | "epoch_regions" | "epochregions" => Some(Self::EpochRegions),
            "heap" | "freelist" | "heap_freelist" => Some(Self::Heap),
            _ => None,
        }
    }

    /// Construct a fresh allocator of this strategy.
    pub fn create(
        self,
        device: &Device,
        config: TransientAllocatorConfig,
    ) -> Result<Box<dyn TransientAllocator>> {
        match self {
            Self::BumpReset => Ok(Box::new(BumpResetAllocator::new(device, config)?)),
            Self::EpochRegions => Ok(Box::new(EpochRegionsAllocator::new(device, config)?)),
            Self::Heap => Ok(Box::new(HeapTransientAllocator::new(device, config)?)),
        }
    }
}

// -----------------------------------------------------------------------
// BumpReset strategy
// -----------------------------------------------------------------------

/// Single-pool bump allocator that resets between frames.
///
/// `begin_frame` blocks on the previous frame's epoch (if any) before resetting the bump
/// pointer when `gpu_progress() < last_epoch`. This is the simplest correct strategy and
/// provides a useful baseline / fallback for debugging, but it serializes the CPU and the
/// GPU when the prior frame is still in flight — the CPU cannot begin recording frame N+1
/// until frame N's GPU work is done.
///
/// Equivalent to a per-thread arena allocator with synchronous reset.
pub struct BumpResetAllocator {
    pool: BufferPool,
    last_epoch: Option<TimelineValue>,
}

impl BumpResetAllocator {
    /// Create a new allocator. Allocates initial backing immediately.
    pub fn new(device: &Device, config: TransientAllocatorConfig) -> Result<Self> {
        let initial = config.initial_size.max(config.alignment);
        let pool = BufferPool::with_alignment_capacity_hint_and_flags(
            device,
            initial,
            initial,
            config.alignment,
            config.flags,
        )?;
        Ok(Self {
            pool,
            last_epoch: None,
        })
    }
}

impl TransientAllocator for BumpResetAllocator {
    fn begin_frame(&mut self, device: &Device, hint_size: u64) -> Result<()> {
        if let Some(tv) = self.last_epoch {
            // gpu_progress() and wait_until() are two separate lock acquisitions.
            // The GPU may retire tv between the check and the wait — that's harmless
            // (wait_until returns immediately when the semaphore/fence has already
            // fired). The opposite direction (progress reads stale-low on a poller
            // backend) causes an unnecessary but correct wait of at most ~1 ms.
            if device.gpu_progress() < tv {
                device.wait_until(tv)?;
            }
        }

        // Grow to fit the hint if known. resize() may fail (OOM); if it does we
        // propagate the error *before* reset(), preserving the prior frame's data
        // so a retry with a smaller hint is possible.
        if hint_size > self.pool.capacity() {
            self.pool.resize(hint_size)?;
        }
        self.pool.reset();
        self.last_epoch = None;
        Ok(())
    }

    fn alloc(
        &mut self,
        _device: &Device,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferView> {
        let used = self.pool.used();
        if used.saturating_add(size) > self.pool.capacity() {
            let target = used
                .saturating_add(size)
                .saturating_mul(2)
                .max(self.pool.capacity().saturating_mul(2));
            self.pool.resize(target)?;
        }
        self.pool.alloc_bytes(size, element_stride)
    }

    fn end_frame(&mut self, _device: &Device, epoch: TimelineValue) {
        self.last_epoch = Some(epoch);
    }

    fn capacity(&self) -> u64 {
        self.pool.capacity()
    }

    fn used_this_frame(&self) -> u64 {
        self.pool.used()
    }

    fn name(&self) -> &'static str {
        "bump_reset"
    }

    fn hint_unused_above(&mut self, offset: u64) {
        self.pool.hint_unused_above(offset);
    }

    fn clear(&mut self) {
        self.pool.reset();
        self.last_epoch = None;
    }
}

// -----------------------------------------------------------------------
// EpochRegions strategy
// -----------------------------------------------------------------------

/// State of a region inside [`EpochRegionsAllocator`]. Designed so reclamation is a single
/// comparison against [`Device::gpu_progress`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegionState {
    /// Backing exists and is ready for bump allocation.
    Empty,
    /// Currently being bump-allocated from. (Multiple regions may be active simultaneously if
    /// a single frame's allocation spilled across regions.)
    Active,
    /// Frame has ended but the epoch is not yet known (surface-presentation path where
    /// the timeline value arrives via a deferred `end_frame` after `Frame::present`).
    /// The allocator treats these conservatively: they cannot be reclaimed until an epoch
    /// is supplied, but they do not count as Active so the next `begin_frame` can proceed.
    Pending,
    /// Written by an earlier frame; reusable once `gpu_progress >= epoch`.
    DeferredReclaim { epoch: TimelineValue },
}

struct Region {
    pool: BufferPool,
    state: RegionState,
}

/// Region-pool bump allocator with [`TimelineValue`]-based reclamation.
///
/// At any point in time, the regions are partitioned into:
///
/// * **Empty** — ready for use; bump pointer is at 0.
/// * **Active** — being bump-allocated from this frame.
/// * **Pending** — frame finished but epoch not yet known (surface path). Promoted to
///   `DeferredReclaim` when [`TransientAllocator::end_frame`] supplies the epoch.
/// * **DeferredReclaim** — written by an earlier frame; reusable once the GPU has
///   finished that frame's work (`epoch <= gpu_progress()`).
///
/// `begin_frame` moves Active regions to Pending (if `end_frame` wasn't called yet) and
/// opportunistically recycles `DeferredReclaim` regions whose epochs have passed. `alloc`
/// spills to a fresh region when the active one fills. `end_frame` assigns the epoch to
/// all Pending regions and retires any remaining Active ones.
///
/// Closest CPU analog: an N-arena epoch-based-reclamation allocator. This is the same
/// pattern that drives the Linux kernel's RCU, JVM's ZGC region recycling, and crossbeam's
/// `Collector`.
pub struct EpochRegionsAllocator {
    config: TransientAllocatorConfig,
    regions: Vec<Region>,
    /// Indices of regions in `Active` state, in the order they were activated this frame.
    /// On `end_frame`, every index here is moved to `DeferredReclaim`.
    active: VecDeque<usize>,
}

impl EpochRegionsAllocator {
    /// Create a new allocator. Allocates one initial region immediately.
    pub fn new(device: &Device, config: TransientAllocatorConfig) -> Result<Self> {
        let region = Self::make_region(device, &config, config.min_region_size)?;
        Ok(Self {
            config,
            regions: vec![region],
            active: VecDeque::new(),
        })
    }

    fn make_region(
        device: &Device,
        config: &TransientAllocatorConfig,
        min_size: u64,
    ) -> Result<Region> {
        let size = config.min_region_size.max(min_size).max(config.alignment);
        // Capacity hint equals the region size — each region only needs to handle
        // its own share of per-frame demand, not the entire frame.
        let pool = BufferPool::with_alignment_capacity_hint_and_flags(
            device,
            size,
            size,
            config.alignment,
            config.flags,
        )?;
        Ok(Region {
            pool,
            state: RegionState::Empty,
        })
    }

    /// Reset `DeferredReclaim` regions whose epoch has retired.
    fn reclaim_retired_regions(&mut self, device: &Device) {
        let progress = device.gpu_progress();
        for r in &mut self.regions {
            if let RegionState::DeferredReclaim { epoch } = r.state {
                if epoch <= progress {
                    r.pool.reset();
                    r.state = RegionState::Empty;
                }
            }
        }
    }

    /// Find an empty region with at least `min_size` bytes of capacity. Prefers regions that
    /// fit `min_size` without growth.
    fn find_empty(&self, min_size: u64) -> Option<usize> {
        // First pass: exact fit (no growth needed).
        for (i, r) in self.regions.iter().enumerate() {
            if matches!(r.state, RegionState::Empty) && r.pool.capacity() >= min_size {
                return Some(i);
            }
        }
        // Second pass: any empty region (we'll grow it).
        self.regions
            .iter()
            .position(|r| matches!(r.state, RegionState::Empty))
    }

    /// Index of the `DeferredReclaim` region with the smallest epoch (for the safety-valve wait).
    fn oldest_deferred(&self) -> Option<(usize, TimelineValue)> {
        self.regions
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r.state {
                RegionState::DeferredReclaim { epoch } => Some((i, epoch)),
                _ => None,
            })
            .min_by_key(|(_, e)| *e)
    }

    /// Activate region `idx`. Caller guarantees state is currently Empty.
    fn activate(&mut self, idx: usize) {
        debug_assert_eq!(self.regions[idx].state, RegionState::Empty);
        self.regions[idx].state = RegionState::Active;
        self.active.push_back(idx);
    }

    /// Acquire a region to allocate from, growing or waiting as needed. Used both at the
    /// start of a frame and when an active region runs out mid-frame.
    fn acquire_region(&mut self, device: &Device, min_size: u64) -> Result<usize> {
        self.reclaim_retired_regions(device);

        if let Some(idx) = self.find_empty(min_size) {
            // Grow the region if it was empty but undersized.
            if self.regions[idx].pool.capacity() < min_size {
                self.regions[idx].pool.resize(min_size)?;
            }
            self.activate(idx);
            return Ok(idx);
        }

        // No empty region. Try growing the pool.
        if self.regions.len() < self.config.max_regions {
            let region = Self::make_region(device, &self.config, min_size)?;
            self.regions.push(region);
            let idx = self.regions.len() - 1;
            self.activate(idx);
            return Ok(idx);
        }

        // Pipeline-depth cap hit. Safety valve: wait on the oldest DeferredReclaim.
        if let Some((idx, epoch)) = self.oldest_deferred() {
            device.wait_until(epoch)?;
            self.reclaim_retired_regions(device);
            // Force-reset as fallback if progress still lags the signal epoch.
            if matches!(self.regions[idx].state, RegionState::DeferredReclaim { .. }) {
                self.regions[idx].pool.reset();
                self.regions[idx].state = RegionState::Empty;
            }
            if self.regions[idx].pool.capacity() < min_size {
                self.regions[idx].pool.resize(min_size)?;
            }
            self.activate(idx);
            return Ok(idx);
        }

        // No DeferredReclaim regions — but there may be Pending regions (deferred end_frame,
        // epoch not yet known). Reset them to Empty so allocation can proceed.
        for r in &mut self.regions {
            if r.state == RegionState::Pending {
                r.pool.reset();
                r.state = RegionState::Empty;
            }
        }

        if let Some(idx) = self.find_empty(min_size) {
            if self.regions[idx].pool.capacity() < min_size {
                self.regions[idx].pool.resize(min_size)?;
            }
            self.activate(idx);
            return Ok(idx);
        }

        anyhow::bail!(
            "EpochRegionsAllocator: all {} regions Active, cannot proceed",
            self.regions.len()
        )
    }

    /// Test-only inspection.
    #[doc(hidden)]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Test-only inspection. Returns the count of `DeferredReclaim` regions.
    #[doc(hidden)]
    pub fn retired_count(&self) -> usize {
        self.regions
            .iter()
            .filter(|r| matches!(r.state, RegionState::DeferredReclaim { .. }))
            .count()
    }

    /// Test-only inspection.
    #[doc(hidden)]
    pub fn empty_count(&self) -> usize {
        self.regions
            .iter()
            .filter(|r| matches!(r.state, RegionState::Empty))
            .count()
    }

    /// Test-only inspection.
    #[doc(hidden)]
    pub fn pending_count(&self) -> usize {
        self.regions
            .iter()
            .filter(|r| matches!(r.state, RegionState::Pending))
            .count()
    }

    /// Test-only inspection.
    #[doc(hidden)]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}

impl TransientAllocator for EpochRegionsAllocator {
    fn begin_frame(&mut self, device: &Device, hint_size: u64) -> Result<()> {
        // If the previous frame's end_frame hasn't been called yet (surface path),
        // move any leftover Active regions to Pending so they're out of the way.
        // end_frame will assign the epoch when it arrives.
        if !self.active.is_empty() {
            for &idx in &self.active {
                self.regions[idx].state = RegionState::Pending;
            }
            self.active.clear();
        }

        self.reclaim_retired_regions(device);

        // Pre-acquire one region so the first alloc is hot-path lock-free. If hint_size is 0
        // (caller has no estimate), use min_region_size so we don't allocate a tiny region.
        let min_size = hint_size.max(self.config.min_region_size);
        let _ = self.acquire_region(device, min_size)?;
        Ok(())
    }

    fn alloc(
        &mut self,
        device: &Device,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferView> {
        // Hot path: try the most-recently-activated region.
        if let Some(&idx) = self.active.back() {
            if self.regions[idx].pool.would_fit(size, element_stride) {
                return self.regions[idx].pool.alloc_bytes(size, element_stride);
            }
        }

        // Spill: current active region is full. Acquire another.
        let _ = self.acquire_region(device, size)?;
        let idx = *self.active.back().expect("acquire_region pushed an active");
        self.regions[idx].pool.alloc_bytes(size, element_stride)
    }

    fn end_frame(&mut self, _device: &Device, epoch: TimelineValue) {
        // Collect indices to retire from the active set and any Pending regions.
        let mut retiring: Vec<usize> = Vec::with_capacity(self.active.len());
        for &idx in &self.active {
            retiring.push(idx);
        }
        self.active.clear();

        // Also promote Pending regions (surface-path deferred end_frame).
        for (idx, r) in self.regions.iter_mut().enumerate() {
            if r.state == RegionState::Pending {
                retiring.push(idx);
            }
        }

        if retiring.is_empty() {
            return;
        }

        for &idx in &retiring {
            self.regions[idx].state = RegionState::DeferredReclaim { epoch };
        }
    }

    fn capacity(&self) -> u64 {
        self.regions.iter().map(|r| r.pool.capacity()).sum()
    }

    fn used_this_frame(&self) -> u64 {
        self.active
            .iter()
            .map(|&i| self.regions[i].pool.used())
            .sum()
    }

    fn name(&self) -> &'static str {
        "epoch_regions"
    }

    fn hint_unused_above(&mut self, offset: u64) {
        // Forward to the most-recently-active region. The offset is relative to the start of
        // that region, matching the bump-pointer semantics of the caller.
        if let Some(&idx) = self.active.back() {
            self.regions[idx].pool.hint_unused_above(offset);
        }
    }

    fn clear(&mut self) {
        for r in &mut self.regions {
            r.pool.reset();
            r.state = RegionState::Empty;
        }
        self.active.clear();
    }
}

// -----------------------------------------------------------------------
// Heap strategy (free-list inside a monolithic buffer)
// -----------------------------------------------------------------------

/// A pending free: byte range + the GPU epoch that must retire before reuse.
struct DeferredFree {
    offset: u64,
    size: u64,
    epoch: Option<TimelineValue>,
}

/// Best-fit free-list allocator inside a single monolithic [`BufferPool`].
///
/// Unlike the bump-only strategies, `HeapTransientAllocator` supports mid-frame
/// [`TransientAllocator::free`] calls. Freed ranges enter a deferred-free queue
/// keyed by GPU timeline value; once the GPU retires past that epoch, the range
/// is merged back into a coalescing free list (`BTreeMap<offset, size>`).
///
/// Allocation uses best-fit search over the free list, falling back to bumping the
/// high-water mark, and finally growing the backing buffer. This gives peak memory
/// proportional to the peak *live* working set rather than the sum of all allocations.
pub struct HeapTransientAllocator {
    pool: BufferPool,
    alignment: u64,
    /// Bump pointer (high-water mark). Ranges below this that aren't in the free list
    /// are considered in-use.
    watermark: u64,
    /// Coalesced free ranges available for immediate reuse: `offset -> size`.
    free_list: BTreeMap<u64, u64>,
    /// Ranges freed via `free()` awaiting GPU retirement before reuse.
    /// `None` epoch means the epoch is not yet known (assigned in `end_frame`).
    deferred: Vec<DeferredFree>,
    /// Bytes currently live (allocated minus freed-and-retired).
    live_bytes: u64,
    /// Peak `live_bytes` observed across the lifetime of this allocator (diagnostic).
    peak_live_bytes: u64,
    /// Frame counter — used to defer `shrink_to_fit` until the pipeline has warmed up.
    frame_count: u64,
}

impl HeapTransientAllocator {
    pub fn new(device: &Device, config: TransientAllocatorConfig) -> Result<Self> {
        let initial = config.initial_size.max(config.alignment);
        let pool = BufferPool::with_alignment_capacity_hint_and_flags(
            device,
            initial,
            initial,
            config.alignment,
            config.flags,
        )?;
        Ok(Self {
            pool,
            alignment: config.alignment,
            watermark: 0,
            free_list: BTreeMap::new(),
            deferred: Vec::new(),
            live_bytes: 0,
            peak_live_bytes: 0,
            frame_count: 0,
        })
    }

    /// Return deferred frees whose epoch has retired to the coalescing free list.
    fn reclaim_retired_frees(&mut self, device: &Device) {
        let progress = device.gpu_progress();
        let mut i = 0;
        while i < self.deferred.len() {
            if self.deferred[i]
                .epoch
                .is_some_and(|epoch| epoch <= progress)
            {
                let d = self.deferred.swap_remove(i);
                self.insert_free(d.offset, d.size);
            } else {
                i += 1;
            }
        }
    }

    /// Pull the watermark back when the highest free range extends up to (or past) it.
    /// Prevents the watermark from creeping forward indefinitely due to fragmentation.
    fn compact_watermark(&mut self) {
        while let Some((&off, &size)) = self.free_list.iter().next_back() {
            if off + size >= self.watermark {
                self.free_list.remove(&off);
                self.watermark = off;
            } else {
                break;
            }
        }
    }

    /// Insert a range into the free list, coalescing with adjacent neighbours.
    fn insert_free(&mut self, offset: u64, size: u64) {
        let mut merged_off = offset;
        let mut merged_size = size;

        // Coalesce with the preceding range if it ends exactly at our start.
        if let Some((&prev_off, &prev_size)) = self.free_list.range(..offset).next_back() {
            if prev_off + prev_size == offset {
                merged_off = prev_off;
                merged_size += prev_size;
                self.free_list.remove(&prev_off);
            }
        }

        // Coalesce with the following range if it starts exactly at our end.
        let end = merged_off + merged_size;
        if let Some((&next_off, &next_size)) = self.free_list.range(end..).next() {
            if next_off == end {
                merged_size += next_size;
                self.free_list.remove(&next_off);
            }
        }

        self.free_list.insert(merged_off, merged_size);
    }

    /// Best-fit search: find the smallest free range that can satisfy `aligned_size` with
    /// the given `alloc_align`. Returns `(offset_in_range, aligned_start, range_key)`.
    fn best_fit(&self, size: u64, alloc_align: u64) -> Option<(u64, u64)> {
        let mut best: Option<(u64, u64, u64)> = None; // (aligned_start, range_off, range_size)
        for (&off, &rng_size) in &self.free_list {
            let aligned_start = off.div_ceil(alloc_align) * alloc_align;
            let end = aligned_start + size;
            if end <= off + rng_size {
                let waste = rng_size - size;
                if best.is_none() || waste < best.unwrap().2 {
                    best = Some((aligned_start, off, waste));
                }
            }
        }
        best.map(|(aligned_start, range_off, _)| (aligned_start, range_off))
    }

    /// Allocate from the free list or bump the watermark.
    fn alloc_inner(&mut self, size: u64, alloc_align: u64) -> Option<u64> {
        // Try free list first (best fit).
        if let Some((aligned_start, range_off)) = self.best_fit(size, alloc_align) {
            let range_size = self.free_list.remove(&range_off).unwrap();
            let range_end = range_off + range_size;
            let alloc_end = aligned_start + size;

            // Return any leftover prefix (between range start and aligned alloc start).
            if aligned_start > range_off {
                self.free_list.insert(range_off, aligned_start - range_off);
            }
            // Return any leftover suffix.
            if alloc_end < range_end {
                self.free_list.insert(alloc_end, range_end - alloc_end);
            }
            return Some(aligned_start);
        }

        // Bump the watermark.
        let aligned_wm = self.watermark.div_ceil(alloc_align) * alloc_align;
        let end = aligned_wm + size;
        if end <= self.pool.capacity() {
            self.watermark = end;
            return Some(aligned_wm);
        }
        None
    }

    fn grow(&mut self, min_capacity: u64) -> Result<()> {
        let current = self.pool.capacity();
        // Grow to at least min_capacity with 25% headroom. The heap recycles freed ranges,
        // so once the pipeline is full the capacity stabilises at the peak concurrent set.
        // After warmup, `shrink_to_fit` trims any excess.
        let target = min_capacity.max(current + current / 4);
        self.pool.resize(target)
    }

    /// Peak live bytes observed (diagnostic).
    #[doc(hidden)]
    pub fn peak_live_bytes(&self) -> u64 {
        self.peak_live_bytes
    }
}

/// Frames to wait before the first auto-shrink. Lets the pipeline fill and the
/// peak working set stabilise before we trim.
const HEAP_WARMUP_FRAMES: u64 = 8;

impl TransientAllocator for HeapTransientAllocator {
    fn begin_frame(&mut self, device: &Device, _hint_size: u64) -> Result<()> {
        self.frame_count += 1;
        self.reclaim_retired_frees(device);
        self.compact_watermark();
        // After warmup, auto-shrink if we over-grew during the first few frames.
        if self.frame_count == HEAP_WARMUP_FRAMES {
            self.shrink_to_fit();
        }
        Ok(())
    }

    fn alloc(
        &mut self,
        _device: &Device,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferView> {
        let stride = element_stride.unwrap_or(4) as u64;
        let alloc_align = lcm(self.alignment, stride);

        // Try allocation from free list or bump.
        if let Some(offset) = self.alloc_inner(size, alloc_align) {
            self.live_bytes += size;
            if self.live_bytes > self.peak_live_bytes {
                self.peak_live_bytes = self.live_bytes;
            }
            return self
                .pool
                .backing_buffer()
                .create_view(offset, size, element_stride);
        }

        // Grow the pool and retry. Deferred frees are only drained in begin_frame
        // when we know the GPU has advanced past their epochs.
        let aligned_wm = self.watermark.div_ceil(alloc_align) * alloc_align;
        let needed = aligned_wm.saturating_add(size);
        self.grow(needed)?;
        if let Some(offset) = self.alloc_inner(size, alloc_align) {
            self.live_bytes += size;
            if self.live_bytes > self.peak_live_bytes {
                self.peak_live_bytes = self.live_bytes;
            }
            return self
                .pool
                .backing_buffer()
                .create_view(offset, size, element_stride);
        }

        anyhow::bail!(
            "HeapTransientAllocator: failed to allocate {} bytes (capacity={}, watermark={}, free_ranges={})",
            size,
            self.pool.capacity(),
            self.watermark,
            self.free_list.len()
        )
    }

    fn free(&mut self, offset: u64, size: u64, epoch: Option<TimelineValue>) {
        self.live_bytes = self.live_bytes.saturating_sub(size);
        self.deferred.push(DeferredFree {
            offset,
            size,
            epoch,
        });
    }

    fn end_frame(&mut self, _device: &Device, epoch: TimelineValue) {
        // Stamp any mid-frame frees (epoch=None) with this frame's timeline so
        // they become eligible for reuse only after the GPU retires this frame.
        for d in &mut self.deferred {
            if d.epoch.is_none() {
                d.epoch = Some(epoch);
            }
        }
    }

    fn capacity(&self) -> u64 {
        self.pool.capacity()
    }

    fn used_this_frame(&self) -> u64 {
        self.live_bytes
    }

    fn name(&self) -> &'static str {
        "heap"
    }

    fn shrink_to_fit(&mut self) {
        self.compact_watermark();
        let target = self.watermark.max(self.peak_live_bytes);
        if target > 0 && self.pool.capacity() > target * 2 {
            let new_cap = target + target / 4; // 25 % headroom
            if new_cap < self.pool.capacity() {
                tracing::info!(
                    capacity = self.pool.capacity(),
                    watermark = self.watermark,
                    peak_live = self.peak_live_bytes,
                    new_cap,
                    "HeapTransientAllocator: shrink_to_fit"
                );
                self.pool.hint_unused_above(new_cap);
            }
        }
    }

    fn clear(&mut self) {
        self.watermark = 0;
        self.free_list.clear();
        self.deferred.clear();
        self.live_bytes = 0;
        self.frame_count = 0;
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_parse_recognises_canonical_names() {
        assert_eq!(
            TransientAllocatorStrategy::parse("bump"),
            Some(TransientAllocatorStrategy::BumpReset)
        );
        assert_eq!(
            TransientAllocatorStrategy::parse("bump_reset"),
            Some(TransientAllocatorStrategy::BumpReset)
        );
        assert_eq!(
            TransientAllocatorStrategy::parse("BumpReset"),
            Some(TransientAllocatorStrategy::BumpReset)
        );
        assert_eq!(
            TransientAllocatorStrategy::parse("epoch"),
            Some(TransientAllocatorStrategy::EpochRegions)
        );
        assert_eq!(
            TransientAllocatorStrategy::parse("regions"),
            Some(TransientAllocatorStrategy::EpochRegions)
        );
        assert_eq!(
            TransientAllocatorStrategy::parse("epoch_regions"),
            Some(TransientAllocatorStrategy::EpochRegions)
        );
    }

    #[test]
    fn strategy_parse_rejects_unknown_names() {
        assert_eq!(TransientAllocatorStrategy::parse(""), None);
        assert_eq!(TransientAllocatorStrategy::parse("nope"), None);
        assert_eq!(TransientAllocatorStrategy::parse("default"), None);
    }

    #[test]
    fn strategy_default_is_heap() {
        assert_eq!(
            TransientAllocatorStrategy::default(),
            TransientAllocatorStrategy::Heap
        );
    }

    #[test]
    fn config_default_is_sensible() {
        let c = TransientAllocatorConfig::default();
        assert!(c.alignment.is_power_of_two());
        assert!(c.initial_size > 0);
        assert!(c.min_region_size > 0);
        assert!(c.max_regions > 0);
    }

    // ---------------------------------------------------------------
    // HeapTransientAllocator tests
    // ---------------------------------------------------------------

    use crate::backend::mock::MockBackend;
    use crate::device::Device;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn heap_config() -> TransientAllocatorConfig {
        TransientAllocatorConfig {
            initial_size: 64 * 1024,
            min_region_size: 64 * 1024,
            max_regions: 3,
            alignment: 256,
            flags: crate::types::BufferFlags::empty(),
        }
    }

    #[test]
    fn heap_alloc_and_free_reuses_range() {
        let device = test_device();
        let mut alloc = HeapTransientAllocator::new(&device, heap_config()).unwrap();

        alloc.begin_frame(&device, 0).unwrap();

        let v1 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        let v1_off = v1.offset();
        let v1_size = v1.size();
        assert_eq!(v1_size, 1024);

        // Free with epoch=0 — retired immediately since mock GPU progress starts at 0.
        alloc.free(v1_off, v1_size, Some(0));

        alloc.begin_frame(&device, 0).unwrap();

        let v2 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        assert_eq!(v2.offset(), v1_off, "freed range should be reused");
    }

    #[test]
    fn heap_deferred_free_respects_epoch() {
        let device = test_device();
        let mut alloc = HeapTransientAllocator::new(&device, heap_config()).unwrap();

        alloc.begin_frame(&device, 0).unwrap();

        let v1 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        let v1_off = v1.offset();
        let v1_size = v1.size();

        // Free with epoch=5. GPU progress is still 0, so it shouldn't be reused.
        alloc.free(v1_off, v1_size, Some(5));

        // Begin frame: gpu_progress=0, epoch=5 not retired.
        alloc.begin_frame(&device, 0).unwrap();

        let v2 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        // v2 should NOT reuse v1's range because epoch hasn't retired.
        assert_ne!(v2.offset(), v1_off, "epoch not retired — should not reuse");

        // Now advance GPU timeline past epoch=5.
        device.wait_until(5).unwrap();

        alloc.begin_frame(&device, 0).unwrap();

        let v3 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        // v3 should reuse the first range (v1_off) which is now retired.
        assert_eq!(
            v3.offset(),
            v1_off,
            "epoch retired — should reuse freed range"
        );
    }

    #[test]
    fn heap_coalescing_merges_adjacent_frees() {
        let device = test_device();
        let mut alloc = HeapTransientAllocator::new(&device, heap_config()).unwrap();

        alloc.begin_frame(&device, 0).unwrap();

        let v1 = alloc.alloc(&device, 256, Some(4)).unwrap();
        let v2 = alloc.alloc(&device, 256, Some(4)).unwrap();
        let v3 = alloc.alloc(&device, 256, Some(4)).unwrap();
        let (v1_off, v1_sz) = (v1.offset(), v1.size());
        let (v2_off, v2_sz) = (v2.offset(), v2.size());
        let (v3_off, v3_sz) = (v3.offset(), v3.size());

        // Free all three with epoch=0 (immediately retired).
        alloc.free(v1_off, v1_sz, Some(0));
        alloc.free(v2_off, v2_sz, Some(0));
        alloc.free(v3_off, v3_sz, Some(0));

        // Drain deferred frees.
        alloc.begin_frame(&device, 0).unwrap();

        // Now we should be able to allocate a contiguous 768-byte region
        // from the coalesced free block.
        let big = alloc.alloc(&device, 768, Some(4)).unwrap();
        assert_eq!(
            big.offset(),
            v1_off,
            "coalesced range should start at v1's offset"
        );
    }

    #[test]
    fn heap_none_epoch_not_drained_until_stamped() {
        let device = test_device();
        let mut alloc = HeapTransientAllocator::new(&device, heap_config()).unwrap();

        alloc.begin_frame(&device, 0).unwrap();

        let v1 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        let v1_off = v1.offset();

        // Free with epoch=None (mid-pipeline free, pending stamp).
        alloc.free(v1_off, 1024, None);

        // drain_retired should NOT recycle it — epoch is None.
        alloc.begin_frame(&device, 0).unwrap();
        let v2 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        assert_ne!(
            v2.offset(),
            v1_off,
            "None-epoch range must not be reused before stamp"
        );

        // Stamp with epoch=1 via end_frame.
        alloc.end_frame(&device, 1);

        // Still not available — GPU progress is 0, epoch is 1.
        alloc.begin_frame(&device, 0).unwrap();
        let v3 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        assert_ne!(
            v3.offset(),
            v1_off,
            "range should not be reused before epoch retires"
        );

        // Advance past epoch=1.
        device.wait_until(1).unwrap();
        alloc.begin_frame(&device, 0).unwrap();
        let v4 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        assert_eq!(
            v4.offset(),
            v1_off,
            "range should be reused after epoch retires"
        );
    }

    #[test]
    fn heap_peak_live_bytes_tracks_maximum() {
        let device = test_device();
        let mut alloc = HeapTransientAllocator::new(&device, heap_config()).unwrap();

        alloc.begin_frame(&device, 0).unwrap();

        let v1 = alloc.alloc(&device, 4096, Some(4)).unwrap();
        let v2 = alloc.alloc(&device, 8192, Some(4)).unwrap();
        assert_eq!(alloc.peak_live_bytes(), 4096 + 8192);

        alloc.free(v1.offset(), v1.size(), None);
        assert_eq!(
            alloc.peak_live_bytes(),
            4096 + 8192,
            "peak should not decrease on free"
        );

        alloc.free(v2.offset(), v2.size(), None);
        assert_eq!(alloc.used_this_frame(), 0);
        assert_eq!(alloc.peak_live_bytes(), 4096 + 8192);
    }

    #[test]
    fn heap_strategy_parse_and_default() {
        assert_eq!(
            TransientAllocatorStrategy::parse("heap"),
            Some(TransientAllocatorStrategy::Heap)
        );
        assert_eq!(
            TransientAllocatorStrategy::parse("freelist"),
            Some(TransientAllocatorStrategy::Heap)
        );
        assert_eq!(
            TransientAllocatorStrategy::default(),
            TransientAllocatorStrategy::Heap
        );
    }

    #[test]
    fn heap_grows_when_needed() {
        let device = test_device();
        let config = TransientAllocatorConfig {
            initial_size: 1024,
            ..heap_config()
        };
        let mut alloc = HeapTransientAllocator::new(&device, config).unwrap();

        alloc.begin_frame(&device, 0).unwrap();

        // Allocate more than initial_size — should grow.
        let v1 = alloc.alloc(&device, 2048, Some(4)).unwrap();
        assert!(alloc.capacity() >= 2048);
        assert_eq!(v1.size(), 2048);
    }

    // -----------------------------------------------------------------------
    // Tests for defer_release-driven reclamation (#150)
    // -----------------------------------------------------------------------

    fn small_config() -> TransientAllocatorConfig {
        TransientAllocatorConfig {
            initial_size: 4 * 1024,
            min_region_size: 4 * 1024,
            max_regions: 3,
            alignment: 256,
            flags: crate::types::BufferFlags::empty(),
        }
    }

    #[test]
    fn epoch_regions_recycles_after_epoch() {
        // After end_frame, regions should be in DeferredReclaim state.
        // After wait + begin_frame, they should be Empty.
        let device = test_device();
        let mut a = EpochRegionsAllocator::new(&device, small_config()).expect("create");

        a.begin_frame(&device, 0).expect("begin 1");
        let _v = a.alloc(&device, 1024, Some(4)).expect("alloc");
        let mut graph = crate::task_graph::TaskGraph::new();
        let tv = device.submit(&mut graph).expect("submit");

        a.end_frame(&device, tv);
        assert!(
            a.retired_count() >= 1,
            "active region should be DeferredReclaim"
        );
        assert_eq!(
            a.empty_count(),
            0,
            "no Empty regions immediately after end_frame"
        );

        device.wait_until(tv).expect("wait");
        a.begin_frame(&device, 0).expect("begin 2");
        assert_eq!(
            a.retired_count(),
            0,
            "DeferredReclaim should be gone after wait + begin_frame"
        );
    }

    #[test]
    fn heap_transient_recycles_after_epoch() {
        // After end_frame, freed ranges return to the free list once gpu_progress >= epoch.
        let device = test_device();
        let mut alloc = HeapTransientAllocator::new(&device, heap_config()).unwrap();

        alloc.begin_frame(&device, 0).unwrap();
        let v1 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        let offset = v1.offset();
        let mut graph = crate::task_graph::TaskGraph::new();
        let tv = device.submit(&mut graph).expect("submit");

        alloc.free(offset, 1024, Some(tv));
        alloc.end_frame(&device, tv);

        device.wait_until(tv).expect("wait");
        alloc.begin_frame(&device, 0).unwrap();
        let v2 = alloc
            .alloc(&device, 1024, Some(4))
            .expect("alloc after reclaim");
        assert_eq!(
            v2.offset(),
            offset,
            "freed range should be reused after epoch retires"
        );
    }

    #[test]
    fn bump_reset_recycles_after_epoch() {
        // After end_frame records the epoch, begin_frame resets once gpu_progress >= epoch.
        let device = test_device();
        let mut a = BumpResetAllocator::new(&device, small_config()).expect("create");

        a.begin_frame(&device, 0).expect("begin 1");
        let _v = a.alloc(&device, 512, Some(4)).expect("alloc");
        let mut graph = crate::task_graph::TaskGraph::new();
        let tv = device.submit(&mut graph).expect("submit");

        a.end_frame(&device, tv);
        device.wait_until(tv).expect("wait");
        a.begin_frame(&device, 0).expect("begin 2 should not block");
    }
}
