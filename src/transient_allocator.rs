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
//!
//! Future strategies (not yet implemented) might include:
//!
//! * `PerNameRecycle` — per-`(name, size_class)` `Buffer` pool, modeled after Vello/wgpu's
//!   internal resource pool. Easier to reason about across backends but loses the
//!   "single virtual address range" property.
//! * `BackendNative` — delegate to a backend-specific primitive (Metal placement heap with
//!   `makeAliasable`, Vulkan sparse rebind, DX12 tiled `UpdateTileMappings`).
//! * `DebugSequential` — fresh `Buffer` per allocation, no reuse. Catches use-after-free at
//!   the cost of memory; useful for validation.
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

use crate::buffer::{BufferPool, BufferView};
use crate::device::Device;
use crate::timeline::TimelineValue;
use crate::types::BufferFlags;
use anyhow::Result;
use std::collections::VecDeque;

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
    /// Hint for peak demand. Used by backends that support capacity reservation (e.g. Metal
    /// placement heaps) to pre-reserve virtual address range and avoid mid-frame reallocations.
    pub expected_max: u64,
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
            expected_max: 16 * 1024 * 1024,
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
/// Implementations carve [`BufferView`]s out of one or more backing [`Buffer`]s and must
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
    fn end_frame(&mut self, epoch: TimelineValue);

    /// Total bytes of GPU memory currently held across all backing storage.
    fn capacity(&self) -> u64;

    /// Strategy identifier for diagnostics and tracing.
    fn name(&self) -> &'static str;

    /// Bytes consumed by allocations in the current frame so far. Default `0`.
    fn used_this_frame(&self) -> u64 {
        0
    }

    /// Hint that bytes at and above `offset` in the most-recently-active region are unused
    /// for the rest of this frame. Strategies may forward to [`Buffer::hint_unused_above`]
    /// to release physical pages. Default is a no-op.
    fn hint_unused_above(&mut self, _offset: u64) {}

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
    #[default]
    EpochRegions,
}

impl TransientAllocatorStrategy {
    /// Parse a strategy name. Case-insensitive. Returns `None` for unrecognised values.
    ///
    /// Recognised values: `bump`, `bump_reset`, `epoch`, `epoch_regions`, `regions`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "bump" | "bump_reset" | "bumpreset" => Some(Self::BumpReset),
            "epoch" | "regions" | "epoch_regions" | "epochregions" => Some(Self::EpochRegions),
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
        }
    }
}

// -----------------------------------------------------------------------
// BumpReset strategy
// -----------------------------------------------------------------------

/// Single-pool bump allocator that resets between frames.
///
/// `begin_frame` blocks on the previous frame's epoch (if any) before resetting the bump
/// pointer. This is the simplest correct strategy and provides a useful baseline / fallback
/// for debugging, but it serializes the CPU and the GPU — the CPU cannot begin recording
/// frame N+1 until frame N's GPU work is done.
///
/// Equivalent to a per-thread arena allocator with synchronous reset.
pub struct BumpResetAllocator {
    pool: BufferPool,
    last_epoch: Option<TimelineValue>,
    expected_max: u64,
}

impl BumpResetAllocator {
    /// Create a new allocator. Allocates initial backing immediately.
    pub fn new(device: &Device, config: TransientAllocatorConfig) -> Result<Self> {
        let pool = BufferPool::with_alignment_capacity_hint_and_flags(
            device,
            config.initial_size.max(config.alignment),
            config.expected_max.max(config.initial_size),
            config.alignment,
            config.flags,
        )?;
        Ok(Self {
            pool,
            last_epoch: None,
            expected_max: config.expected_max,
        })
    }
}

impl TransientAllocator for BumpResetAllocator {
    fn begin_frame(&mut self, device: &Device, hint_size: u64) -> Result<()> {
        // Wait for the previous frame's GPU work to finish before reusing the pool. This is
        // the *fundamental* synchronization cost of the single-pool strategy: a tighter
        // pipeline requires either a ring of pools or epoch-tagged regions (see
        // EpochRegionsAllocator).
        if let Some(tv) = self.last_epoch {
            if device.gpu_progress() < tv {
                device.wait_until(tv)?;
            }
        }

        // Grow to fit the hint if known.
        if hint_size > self.pool.capacity() {
            self.pool.resize(hint_size)?;
        }
        self.pool.reset();
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
                .max(self.pool.capacity().saturating_mul(2))
                .max(self.expected_max);
            self.pool.resize(target)?;
        }
        self.pool.alloc_bytes(size, element_stride)
    }

    fn end_frame(&mut self, epoch: TimelineValue) {
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
    /// Has been written by a frame whose GPU work has not yet completed. Cannot be reused
    /// until `device.gpu_progress() >= epoch`.
    Retired { epoch: TimelineValue },
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
///   Retired when [`TransientAllocator::end_frame`] supplies the epoch.
/// * **Retired** — written by an earlier frame; reusable once the GPU has finished that
///   frame's work.
///
/// `begin_frame` moves Active regions to Pending (if `end_frame` wasn't called yet) and
/// opportunistically promotes Retired regions whose epochs have passed. `alloc` spills to
/// a fresh region when the active one fills. `end_frame` assigns the epoch to all Pending
/// regions and retires any remaining Active ones.
///
/// Closest CPU analog: an N-arena epoch-based-reclamation allocator. This is the same
/// pattern that drives the Linux kernel's RCU, JVM's ZGC region recycling, and crossbeam's
/// `Collector`.
pub struct EpochRegionsAllocator {
    config: TransientAllocatorConfig,
    regions: Vec<Region>,
    /// Indices of regions in `Active` state, in the order they were activated this frame.
    /// On `end_frame`, every index here is moved to `Retired`.
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
        // Capacity hint is per-region, not the global expected_max. Each region only
        // needs to handle its own share of per-frame demand, not the entire frame.
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

    /// Promote any retired region whose epoch has passed back to Empty (resetting its bump
    /// pointer). Cheap — one comparison per region.
    fn reclaim_completed(&mut self, progress: TimelineValue) {
        for r in &mut self.regions {
            if let RegionState::Retired { epoch } = r.state {
                if progress >= epoch {
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

    /// Index of the retired region with the smallest epoch, if any.
    fn oldest_retired(&self) -> Option<(usize, TimelineValue)> {
        self.regions
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r.state {
                RegionState::Retired { epoch } => Some((i, epoch)),
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
        // Lazy reclaim — cheap; off the steady-state hot path nothing here will succeed.
        let progress = device.gpu_progress();
        self.reclaim_completed(progress);

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

        // Pipeline-depth cap hit. Safety valve: wait on the oldest retiree.
        if let Some((idx, epoch)) = self.oldest_retired() {
            device.wait_until(epoch)?;
            self.regions[idx].pool.reset();
            if self.regions[idx].pool.capacity() < min_size {
                self.regions[idx].pool.resize(min_size)?;
            }
            self.regions[idx].state = RegionState::Empty;
            self.activate(idx);
            return Ok(idx);
        }

        // No Retired regions — but there may be Pending regions (deferred end_frame).
        // Force a GPU drain and convert Pending → Empty.
        let progress = device.gpu_progress();
        for r in &mut self.regions {
            if r.state == RegionState::Pending {
                r.pool.reset();
                r.state = RegionState::Empty;
            }
        }
        // Reclaim anything that completed during the drain.
        self.reclaim_completed(progress);

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

    /// Test-only inspection.
    #[doc(hidden)]
    pub fn retired_count(&self) -> usize {
        self.regions
            .iter()
            .filter(|r| matches!(r.state, RegionState::Retired { .. }))
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

        // Promote completed retirees. Non-blocking.
        let progress = device.gpu_progress();
        self.reclaim_completed(progress);

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

    fn end_frame(&mut self, epoch: TimelineValue) {
        // Retire any Active regions from the current frame.
        for &idx in &self.active {
            self.regions[idx].state = RegionState::Retired { epoch };
        }
        self.active.clear();

        // Also assign the epoch to any Pending regions from a previous frame whose
        // end_frame was deferred (surface-presentation path).
        for r in &mut self.regions {
            if r.state == RegionState::Pending {
                r.state = RegionState::Retired { epoch };
            }
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
    fn strategy_default_is_epoch_regions() {
        assert_eq!(
            TransientAllocatorStrategy::default(),
            TransientAllocatorStrategy::EpochRegions
        );
    }

    #[test]
    fn config_default_is_sensible() {
        let c = TransientAllocatorConfig::default();
        assert!(c.alignment.is_power_of_two());
        assert!(c.initial_size > 0);
        assert!(c.min_region_size > 0);
        assert!(c.max_regions > 0);
        assert!(c.expected_max >= c.initial_size);
    }
}
