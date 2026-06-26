//! Metal backend internal types.
//!
//! This module contains all the state structs used by the Metal backend.
//!
//! ## Bindless Architecture
//!
//! The Metal backend uses Argument Buffers Tier 2 for bindless resource access:
//! - Resources are allocated from MTLHeap for efficient residency management
//! - A global argument buffer stores GPU resource IDs (gpuResourceID)
//! - At render time, useHeap() declares residency for all resources
//! - Shaders access resources by index into the argument buffer

use super::super::{
    BufferHandle, ComputePipelineHandle, DeviceHandle, PipelineHandle, RenderTargetHandle, SamplerHandle, ShaderHandle,
    SurfaceHandle, TextureHandle,
};
use crate::backend::BufferKind;
use crate::timeline::TimelineValue;
use crate::types::{DepthFormat, TextureFormat};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
// Use explicit crate path to avoid collision with our module name
use ::metal as mtl;
use mtl::{
    ArgumentEncoder, Buffer as MTLBuffer, CommandQueue, ComputePipelineState as MTLComputePipelineState,
    DepthStencilState as MTLDepthStencilState, Device as MTLDevice, Heap, Library, MTLPrimitiveType,
    MTLResourceOptions, RenderPipelineState, SamplerState, SharedEvent, Texture as MTLTexture,
};

/// Maximum size of the argument buffer.
/// 5 categories × 4096 slots × 8 bytes per resource ID = 163840 bytes.
/// Categories: storageBuffers(0..4K), uniformBuffers(4K..8K), textures(8K..12K),
///             storageImages(12K..16K), samplers(16K..20K).
pub const ARGUMENT_BUFFER_SIZE: u64 = 20 * 1024 * 8; // 5 × MAX_RESOURCES_PER_CATEGORY × 8

/// Buffer slot for resource binding indices in shaders.
/// Slang assigns uniform entry-point params to [[buffer(1)]] (gGoldy ParameterBlock takes [[buffer(0)]]).
pub const RESOURCE_SLOT_BUFFER: u64 = 1;

/// Starting Metal buffer index for vertex attributes.
/// Slots 0 and 1 are reserved for the argument buffer (gGoldy) and resource slots.
/// Vertex data must use higher indices to avoid collisions.
pub const VERTEX_BUFFER_START_SLOT: u64 = 2;

// PushLayout and its constants live in the shared module so all three backends
// use one definition. Re-export them here so internal code keeps using the
// same unqualified names as before.
pub use super::super::shared::PushLayout;
// TOTAL_PUSH_BYTES is available at crate::backend::shared::TOTAL_PUSH_BYTES if needed.

/// Minimum primary heap size (64 MB).
const MIN_HEAP_SIZE: u64 = 64 * 1024 * 1024;

/// Minimum overflow heap size (16 MB).
const MIN_OVERFLOW_HEAP_SIZE: u64 = 16 * 1024 * 1024;

/// Maximum number of overflow heaps per allocator. Prevents runaway heap
/// creation when frames are submitted faster than the GPU retires resources
/// (e.g. vsync off). Beyond this limit, allocations return `None` and the
/// caller must wait for GPU progress or reduce pipelining depth.
const MAX_OVERFLOW_HEAPS: usize = 16;

/// Upper bound on any single heap allocation. Metal can nominally create
/// heaps up to the device's `maxBufferLength` (tens of GB on Apple Silicon),
/// but in practice a request that large always reflects an upstream bug:
/// a stale bump counter fed into `RenderConfig::with_bump_estimates`, a
/// `u32` wraparound, or similar. Creating the heap anyway does not usually
/// succeed, and when it does it bakes a pathological memory footprint into
/// a process that will then be OOM-killed. Failing fast here instead turns
/// the symptom (hours-long hangs, crash spirals logging tens of thousands
/// of 12 GB allocation attempts) into a single clean error the caller can
/// handle. Raise this if a legitimate workload needs it.
pub(super) const MAX_HEAP_SIZE: u64 = 1024 * 1024 * 1024; // 1 GB

/// Multi-heap allocator for Metal buffer allocations.
///
/// Uses a long-lived primary heap that is right-sized between frames, plus
/// ephemeral overflow heaps created on demand when the primary fills up.
/// Fragmentation is not a concern within a frame because ekrano's allocation
/// pattern is strictly monotonic (all frees are deferred until after recording).
pub(crate) struct HeapAllocator {
    device: MTLDevice,
    primary: Heap,
    overflow: Vec<Heap>,
    high_water_mark: u64,
    primary_size: u64,
    buffer_count: u32,
}

impl HeapAllocator {
    pub fn new(device: MTLDevice, primary: Heap, primary_size: u64) -> Self {
        Self {
            device,
            primary,
            overflow: Vec::new(),
            high_water_mark: 0,
            primary_size,
            buffer_count: 0,
        }
    }

    /// Allocate a buffer from the heap hierarchy.
    ///
    /// Tries primary, then recent overflow heaps, then creates a new overflow.
    /// Overflow heap creation is capped to prevent runaway growth when the GPU
    /// hasn't yet freed resources from retired command buffers.
    /// Returns `None` if the requested size is larger than [`MAX_HEAP_SIZE`]
    /// (see rationale there — we'd rather fail this one allocation than let
    /// a corrupt size request wedge the whole process).
    pub fn allocate(&mut self, size: u64, options: MTLResourceOptions) -> Option<MTLBuffer> {
        if size > MAX_HEAP_SIZE {
            tracing::error!(
                "Refusing buffer allocation of {}MB (cap={}MB); this usually indicates \
                 a stale bump counter or similar upstream corruption",
                size / 1024 / 1024,
                MAX_HEAP_SIZE / 1024 / 1024,
            );
            return None;
        }
        if let Some(buf) = self.primary.new_buffer(size, options) {
            self.buffer_count += 1;
            self.update_high_water_mark();
            return Some(buf);
        }

        // Primary is full — log the spill point so heap growth is visible.
        if tracing::enabled!(target: "goldy::diag::alloc", tracing::Level::INFO) {
            tracing::info!(
                target: "goldy::diag::alloc",
                size_mb = size / (1024 * 1024),
                primary_used_mb = self.primary.used_size() / (1024 * 1024),
                primary_total_mb = self.primary_size / (1024 * 1024),
                overflow_count = self.overflow.len(),
                "heap.primary_full"
            );
        }

        // Try all overflow heaps (newest first). The cap at MAX_OVERFLOW_HEAPS
        // keeps the search bounded. Older heaps may have space from freed buffers.
        for (idx, heap) in self.overflow.iter().rev().enumerate() {
            if let Some(buf) = heap.new_buffer(size, options) {
                if tracing::enabled!(target: "goldy::diag::alloc", tracing::Level::INFO) {
                    tracing::info!(
                        target: "goldy::diag::alloc",
                        size_mb = size / (1024 * 1024),
                        heap_idx = self.overflow.len() - 1 - idx,
                        overflow_count = self.overflow.len(),
                        "heap.alloc_from_overflow"
                    );
                }
                self.buffer_count += 1;
                self.update_high_water_mark();
                return Some(buf);
            }
        }

        // Cap overflow heaps to prevent runaway growth when frames queue faster
        // than the GPU retires them (vsync off). The caller should handle None
        // by waiting for GPU progress or reducing pipelining depth.
        if self.overflow.len() >= MAX_OVERFLOW_HEAPS {
            return None;
        }

        // Clamp the overflow heap size too: `size * 2` on a ~1 GB request is
        // already at the cap, and we do not want to chase the size in a loop.
        let overflow_size = (size * 2).clamp(MIN_OVERFLOW_HEAP_SIZE, MAX_HEAP_SIZE);
        let new_heap = self.create_heap(overflow_size);
        tracing::info!(
            target: "goldy::diag::alloc",
            "Created overflow buffer heap (size={}MB, overflow_count={}, request_mb={}, primary_used_mb={}/{}MB, hwm_mb={})",
            overflow_size / 1024 / 1024,
            self.overflow.len() + 1,
            size / (1024 * 1024),
            self.primary.used_size() / (1024 * 1024),
            self.primary_size / (1024 * 1024),
            self.high_water_mark / (1024 * 1024),
        );
        let buf = new_heap.new_buffer(size, options);
        self.overflow.push(new_heap);

        if buf.is_some() {
            self.buffer_count += 1;
            self.update_high_water_mark();
        }
        buf
    }

    pub fn has_buffers(&self) -> bool {
        self.buffer_count > 0
    }

    /// Number of live buffers allocated from this heap hierarchy.
    pub fn buffer_count(&self) -> u32 {
        self.buffer_count
    }

    /// Number of overflow heaps currently alive.
    pub fn overflow_count(&self) -> usize {
        self.overflow.len()
    }

    /// Peak total bytes used since last reset.
    pub fn high_water_mark(&self) -> u64 {
        self.high_water_mark
    }

    /// Size of the primary heap in bytes.
    pub fn primary_size(&self) -> u64 {
        self.primary_size
    }

    /// Declare all buffer heaps resident for a compute encoder.
    pub fn use_heaps_for_compute(&self, encoder: &mtl::ComputeCommandEncoderRef) {
        if !self.has_buffers() {
            return;
        }
        encoder.use_heap(&self.primary);
        for heap in &self.overflow {
            encoder.use_heap(heap);
        }
    }

    /// Declare all buffer heaps resident for a render encoder at the given stages.
    pub fn use_heaps_for_render(&self, encoder: &mtl::RenderCommandEncoderRef, stages: mtl::MTLRenderStages) {
        if !self.has_buffers() {
            return;
        }
        encoder.use_heap_at(&self.primary, stages);
        for heap in &self.overflow {
            encoder.use_heap_at(heap, stages);
        }
    }

    /// Right-size the primary heap after a frame completes.
    ///
    /// If overflow heaps were used, the primary is replaced with a larger one
    /// sized to 1.5x the peak usage (rounded up to a power of two). Overflow
    /// heaps are dropped since all buffers have been freed by this point.
    pub fn reset_for_frame(&mut self) {
        if !self.overflow.is_empty() {
            let recommended_max = self.device.recommended_max_working_set_size();
            let new_size = (self.high_water_mark * 3 / 2)
                .next_power_of_two()
                .max(MIN_HEAP_SIZE)
                .min(recommended_max / 2);

            if new_size > self.primary_size {
                let new_primary = self.create_heap(new_size);
                tracing::info!(
                    target: "goldy::diag::alloc",
                    "Resized primary buffer heap: {}MB -> {}MB (high_water_mark={}MB)",
                    self.primary_size / 1024 / 1024,
                    new_size / 1024 / 1024,
                    self.high_water_mark / 1024 / 1024,
                );
                self.primary = new_primary;
                self.primary_size = new_size;
            }

            let overflow_count = self.overflow.len();
            self.overflow.clear();
            if tracing::enabled!(target: "goldy::diag::alloc", tracing::Level::INFO) {
                tracing::info!(
                    target: "goldy::diag::alloc",
                    overflow_cleared = overflow_count,
                    primary_size_mb = self.primary_size / (1024 * 1024),
                    "heap.reset_for_frame"
                );
            } else {
                tracing::debug!("Cleared {} overflow buffer heaps", overflow_count);
            }
        }
        self.high_water_mark = 0;
    }

    /// Drop overflow heaps that are completely empty (all buffers freed).
    /// Lighter than `reset_for_frame` (no GPU idle required) — safe to call
    /// after frame cleanup has dropped retired buffers.
    pub fn compact_overflow(&mut self) {
        let before = self.overflow.len();
        self.overflow.retain(|heap| heap.used_size() > 0);
        let dropped = before - self.overflow.len();
        if dropped > 0 {
            if tracing::enabled!(target: "goldy::diag::alloc", tracing::Level::INFO) {
                tracing::info!(
                    target: "goldy::diag::alloc",
                    freed = dropped,
                    overflow_remaining = self.overflow.len(),
                    "heap.compact"
                );
            } else {
                tracing::debug!(
                    "Compacted {} empty overflow buffer heaps ({} remaining)",
                    dropped,
                    self.overflow.len()
                );
            }
        }
    }

    /// Ensure the primary heap is right-sized for `min_capacity` bytes.
    /// Grows the heap if it's too small. Also shrinks if it's more than 4x
    /// the requested capacity (avoids holding a 1 GB heap when 64 MB suffices).
    /// Call *after* all old buffers have been freed and the GPU is idle, but
    /// *before* allocating the next large buffer.
    pub fn ensure_primary_capacity(&mut self, min_capacity: u64) {
        let min_capacity = min_capacity.min(MAX_HEAP_SIZE);
        let recommended_max = self.device.recommended_max_working_set_size();
        let target = min_capacity
            .next_power_of_two()
            .max(MIN_HEAP_SIZE)
            .min(recommended_max / 2);

        let too_small = self.primary_size < min_capacity;
        let too_large = self.primary_size > target.saturating_mul(4);

        if too_small || too_large {
            let new_primary = self.create_heap(target);
            tracing::info!(
                target: "goldy::diag::alloc",
                "{} primary buffer heap: {}MB -> {}MB (requested={}MB)",
                if too_small { "Grew" } else { "Shrank" },
                self.primary_size / 1024 / 1024,
                target / 1024 / 1024,
                min_capacity / 1024 / 1024,
            );
            self.primary = new_primary;
            self.primary_size = target;
        }
    }

    fn update_high_water_mark(&mut self) {
        let mut total = self.primary.used_size();
        for heap in &self.overflow {
            total += heap.used_size();
        }
        self.high_water_mark = self.high_water_mark.max(total);
    }

    fn create_heap(&self, size: u64) -> Heap {
        let desc = mtl::HeapDescriptor::new();
        desc.set_size(size);
        desc.set_storage_mode(mtl::MTLStorageMode::Shared);
        desc.set_cpu_cache_mode(mtl::MTLCPUCacheMode::DefaultCache);
        desc.set_heap_type(mtl::MTLHeapType::Automatic);
        desc.set_hazard_tracking_mode(mtl::MTLHazardTrackingMode::Tracked);
        self.device.new_heap(&desc)
    }
}

/// Multi-heap allocator for Metal texture allocations.
///
/// Mirrors `HeapAllocator` for buffers: a long-lived primary heap plus
/// ephemeral overflow heaps created on demand when the primary fills up.
pub(crate) struct TextureHeapAllocator {
    device: MTLDevice,
    primary: Heap,
    overflow: Vec<Heap>,
    /// Recorded for telemetry / future heap resize decisions.
    /// Not read in production paths; suppress dead_code since
    /// it exists for observability, not active logic.
    #[allow(dead_code)]
    primary_size: u64,
    texture_count: u32,
}

impl TextureHeapAllocator {
    pub fn new(device: MTLDevice, primary: Heap, primary_size: u64) -> Self {
        Self {
            device,
            primary,
            overflow: Vec::new(),
            primary_size,
            texture_count: 0,
        }
    }

    /// Allocate a texture from the heap hierarchy.
    ///
    /// Tries primary, then all overflow heaps (newest first), then creates a
    /// new overflow. Overflow heap creation is capped to prevent runaway growth.
    pub fn allocate(&mut self, descriptor: &mtl::TextureDescriptorRef) -> Option<MTLTexture> {
        if let Some(tex) = self.primary.new_texture(descriptor) {
            self.texture_count += 1;
            return Some(tex);
        }

        for heap in self.overflow.iter().rev() {
            if let Some(tex) = heap.new_texture(descriptor) {
                self.texture_count += 1;
                return Some(tex);
            }
        }

        if self.overflow.len() >= MAX_OVERFLOW_HEAPS {
            return None;
        }

        let alloc_size = self.device.heap_texture_size_and_align(descriptor).size;
        let overflow_size = (alloc_size * 2).max(MIN_OVERFLOW_HEAP_SIZE);
        let new_heap = self.create_heap(overflow_size);
        tracing::info!(
            target: "goldy::diag::alloc",
            "Created overflow texture heap (size={}MB, overflow_count={})",
            overflow_size / 1024 / 1024,
            self.overflow.len() + 1
        );
        let tex = new_heap.new_texture(descriptor);
        self.overflow.push(new_heap);

        if tex.is_some() {
            self.texture_count += 1;
        }
        tex
    }

    /// Drop overflow heaps that are completely empty (all textures freed).
    /// Called during frame cleanup when the GPU has retired old work.
    pub fn compact_overflow(&mut self) {
        let before = self.overflow.len();
        self.overflow.retain(|heap| heap.used_size() > 0);
        let dropped = before - self.overflow.len();
        if dropped > 0 {
            tracing::debug!(
                "Compacted {} empty overflow texture heaps ({} remaining)",
                dropped,
                self.overflow.len()
            );
        }
    }

    pub fn has_textures(&self) -> bool {
        self.texture_count > 0
    }

    /// Number of live textures allocated from this heap hierarchy.
    pub fn texture_count(&self) -> u32 {
        self.texture_count
    }

    /// Number of overflow heaps currently alive.
    pub fn overflow_count(&self) -> usize {
        self.overflow.len()
    }

    /// Declare all texture heaps resident for a compute encoder.
    pub fn use_heaps_for_compute(&self, encoder: &mtl::ComputeCommandEncoderRef) {
        if !self.has_textures() {
            return;
        }
        encoder.use_heap(&self.primary);
        for heap in &self.overflow {
            encoder.use_heap(heap);
        }
    }

    /// Declare all texture heaps resident for a render encoder at the given stages.
    pub fn use_heaps_for_render(&self, encoder: &mtl::RenderCommandEncoderRef, stages: mtl::MTLRenderStages) {
        if !self.has_textures() {
            return;
        }
        encoder.use_heap_at(&self.primary, stages);
        for heap in &self.overflow {
            encoder.use_heap_at(heap, stages);
        }
    }

    fn create_heap(&self, size: u64) -> Heap {
        let desc = mtl::HeapDescriptor::new();
        desc.set_size(size);
        desc.set_storage_mode(mtl::MTLStorageMode::Shared);
        desc.set_cpu_cache_mode(mtl::MTLCPUCacheMode::DefaultCache);
        desc.set_heap_type(mtl::MTLHeapType::Automatic);
        desc.set_hazard_tracking_mode(mtl::MTLHazardTrackingMode::Tracked);
        self.device.new_heap(&desc)
    }
}

/// GPU resource retained until [`TimelineValue`] completion on a context [`SharedEvent`].
///
/// Variant payloads are held **only for their `Drop` impl** — once the GPU signals the
/// timeline value, `DeletionQueue::drain` removes the variant so the MTL object is released.
/// Nothing reads the inner fields; `#[allow(dead_code)]` is intentional.
#[allow(dead_code)]
pub(crate) enum PendingDeletion {
    Buffer { buffer: MTLBuffer },
    Texture { texture: MTLTexture },
    Sampler { sampler: SamplerState },
}

pub(crate) struct DeletionQueue {
    inner: super::super::shared::DeferredQueue<TimelineValue, PendingDeletion>,
}

impl DeletionQueue {
    pub fn new() -> Self {
        Self {
            inner: super::super::shared::DeferredQueue::new(),
        }
    }

    pub fn queue(&mut self, barrier: TimelineValue, resource: PendingDeletion) {
        self.inner.push(barrier, resource);
    }

    /// Destroy resources whose `barrier` has been signaled by the GPU (`signaled_value >= barrier`).
    /// The variants are dropped here which releases the MTL objects.
    pub fn process_up_to(&mut self, signaled: TimelineValue) {
        drop(self.inner.drain_up_to(signaled));
    }

    pub fn flush_all(&mut self) {
        drop(self.inner.flush_all().collect::<Vec<_>>());
    }

    pub(crate) fn pending_len(&self) -> usize {
        self.inner.len()
    }
}

/// CPU-side waiter for GPU timeline completion, driven by `notifyListener:atValue:block:`.
///
/// The inner `Mutex<u64>` tracks the highest timeline value confirmed complete by the GPU.
/// The `Condvar` is signaled from the Metal shared-event listener callback so that
/// `wait_until` can sleep with zero polling overhead.
#[derive(Clone)]
pub(crate) struct TimelineWaiter {
    inner: Arc<(Mutex<u64>, Condvar)>,
    signal_queue: Option<Arc<crate::signal::SignalQueue>>,
    last_emitted: Arc<AtomicU64>,
}

impl TimelineWaiter {
    pub fn new_with_signals(signal_queue: Arc<crate::signal::SignalQueue>) -> Self {
        Self {
            inner: Arc::new((Mutex::new(0), Condvar::new())),
            signal_queue: Some(signal_queue),
            last_emitted: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Called from command-buffer completion handlers when the GPU signals a timeline value.
    pub fn signal(&self, value: u64) {
        if let Some(queue) = &self.signal_queue {
            let mut last = self.last_emitted.load(Ordering::Acquire);
            while last < value {
                last += 1;
                queue.push(crate::signal::Signal::BoundaryCrossed { epoch: last });
                self.last_emitted.store(last, Ordering::Release);
            }
        }
        let (lock, cvar) = &*self.inner;
        let mut signaled = lock.lock().unwrap();
        if value > *signaled {
            *signaled = value;
        }
        cvar.notify_all();
    }

    /// Highest timeline value confirmed complete by completion handlers on this context.
    pub fn completed_value(&self) -> u64 {
        *self.inner.0.lock().unwrap()
    }

    /// Block until the signaled value reaches at least `target`, or timeout.
    /// Returns `Ok(true)` if reached, `Ok(false)` on timeout.
    pub fn wait_until(&self, target: u64, timeout: std::time::Duration) -> bool {
        let (lock, cvar) = &*self.inner;
        let mut signaled = lock.lock().unwrap();
        if *signaled >= target {
            return true;
        }
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return *signaled >= target;
            }
            let (guard, result) = cvar.wait_timeout(signaled, remaining).unwrap();
            signaled = guard;
            if *signaled >= target {
                return true;
            }
            if result.timed_out() {
                return *signaled >= target;
            }
        }
    }
}

/// Per-context async submission stream (timeline shared event, in-flight CBs, signals).
pub(crate) struct MetalSubmissionContext {
    pub device: super::DeviceHandle,
    pub timeline_event: SharedEvent,
    pub timeline_waiter: TimelineWaiter,
    pub signal_queue: std::sync::Arc<crate::signal::SignalQueue>,
    /// Last device-global seq value submitted on this context.
    pub last_submitted_seq: u64,
    pub in_flight_command_buffers: VecDeque<(crate::timeline::TimelineValue, mtl::CommandBuffer)>,
    /// Thread-scoped reclamation epoch from [`GpuBackend::set_reclamation_context`].
    pub reclamation_context: Option<(std::thread::ThreadId, u64)>,
    /// Swapchain returns posted from completion handlers; drained on `poll_signals`.
    pub pending_swapchain_returns: Arc<Mutex<Vec<PendingSwapchainReturn>>>,
    /// Most recently committed timeline on this context (WriteBuffer fast-path gate).
    pub last_committed_timeline: Option<crate::timeline::TimelineValue>,
    /// Per-context staging belt for `WriteBuffer` uploads (bump-allocated shared chunks).
    pub staging_belt: super::staging::StagingBelt,
    /// Per-context staging entries for `WriteTexture` / `WriteTextureRegion` uploads.
    pub texture_staging_pool: super::staging::TextureStagingPool,
    /// Per-context deferred GPU resource teardown, drained on this context's own
    /// completed timeline value (not the device-wide `device_retired` horizon).
    /// Resources destroyed while a reclamation context is installed on the current
    /// thread are routed here so they reclaim without blocking on any other context.
    /// See issue #190.
    pub deletion_queue: DeletionQueue,
    /// Retained graph commands keyed by fingerprint, one entry per live [`Scheme`].
    ///
    /// Metal cannot re-execute a committed command buffer, so each entry stores the
    /// original `GraphCommand` slice; resubmit re-records from it. The `Arc` makes
    /// resubmit clones allocation-free. Multiple schemes can share one context without
    /// evicting each other because each fingerprint is a distinct map key.
    pub retained_graphs: std::collections::HashMap<u64, std::sync::Arc<[super::super::GraphCommand]>>,
}

/// A logical Metal device with associated resources.
///
/// Requires Argument Buffers Tier 2 (Apple Silicon, Intel 2017+, AMD 2015+).
pub(crate) struct LogicalDevice {
    pub device: MTLDevice,
    /// The single command queue shared by all contexts on this device.
    ///
    /// Metal guarantees that command buffers committed to the same queue execute
    /// in FIFO order.  The slot-reclamation logic in `unregister_buffer` and
    /// `drain_pending_slots_up_to` relies on this: it defers recycling a
    /// descriptor slot until `device_retired` (max signaled timeline across all
    /// contexts) passes `timeline_scheduled_max`, which is only sound when every
    /// submission that could reference the slot is ordered behind every other
    /// submission on the same queue.
    ///
    /// **If this ever changes to per-context `MTLCommandQueue`s**, that ordering
    /// guarantee disappears.  Submissions from different contexts could then race,
    /// and a slot freed once *one* context retires could still be in use by
    /// another.  The fix would be the same per-resource per-context retirement
    /// tracking used by the DX12/Vulkan backends (see `slot_last_seen` and
    /// `pending_slot_reclamations` there).
    pub command_queue: CommandQueue,

    // Bindless infrastructure (always present — Tier 2 required)
    /// Multi-heap allocator for buffer allocations (grows on demand)
    pub heap_allocator: Mutex<HeapAllocator>,
    /// Multi-heap allocator for texture allocations (grows on demand)
    pub texture_heap: Mutex<TextureHeapAllocator>,
    /// Global argument buffer containing resource IDs
    pub argument_buffer: MTLBuffer,
    /// Encoder for writing buffers to the argument buffer
    pub argument_encoder: ArgumentEncoder,
    /// Encoder for writing sampled textures (ReadOnly) to the argument buffer
    pub texture_encoder: ArgumentEncoder,
    /// Encoder for writing storage images (ReadWrite) to the argument buffer
    pub storage_image_encoder: ArgumentEncoder,
    /// Encoder for writing samplers to the argument buffer (MTLDataType::Sampler).
    /// Its `encoded_length()` is the authoritative per-slot stride for the sampler
    /// category; never hardcode 8 when encoding sampler offsets.
    pub sampler_encoder: ArgumentEncoder,
    /// Frame-table selector + device table (arg slots 0–1) and N-frame ring guard.
    pub frame_table: Mutex<super::frame_table::MetalFrameTable>,
    /// Registry tracking resource indices in the argument buffer
    pub descriptors: Arc<Mutex<DescriptorRegistry>>,
    /// Device-global submission sequence (contexts signal their own shared events).
    pub timeline_next: Arc<AtomicU64>,
    /// Highest device-global seq scheduled on the GPU queue (used for idle / flush).
    pub timeline_scheduled_max: AtomicU64,
    /// Minimum completed horizon after a context is destroyed (never lowers `device_retired`).
    pub retired_floor: AtomicU64,
    /// Deferred GPU resource teardown until device timeline reaches the queued barrier.
    /// Wrapped in `Mutex` so `LogicalDevice` can be Arc-shared and `process_deletion_queue_up_to`
    /// can take `&self` — matching the pattern used by the Vulkan and DX12 backends (phase 5).
    pub deletion_queue: Mutex<DeletionQueue>,
    /// Serialises `command_buffer.commit()` across concurrent submits on this device.
    ///
    /// Metal's single `MTLCommandQueue` is thread-safe for `newCommandBuffer()` but
    /// `commit()` must be issued in timeline order.  Holding this lock only for the
    /// commit call (not all of MetalState) allows submit prep to overlap while still
    /// serialising the queue-enqueue moment.  Cloned out of `LogicalDevice` before
    /// recording begins so the global backend lock can be dropped before commit.
    pub queue_lock: Arc<Mutex<()>>,
}

impl LogicalDevice {
    /// Drop deferred resources whose barrier is `<= completed` (device-global retirement horizon).
    pub(crate) fn process_deletion_queue_up_to(&self, completed: u64) {
        self.deletion_queue.lock().unwrap().process_up_to(completed);
        self.descriptors
            .lock()
            .unwrap()
            .resource_registry
            .drain_pending_slots_up_to(completed);
    }
}

/// Maximum resources per access pattern category (must match GOLDY_MAX_RESOURCES in shaders).
///
/// Raised from 64 to 4096 in issue #125. The previous 64-slot limit only
/// supported ~2 frames in flight with typical per-frame buffer counts; 4096
/// comfortably handles hundreds of in-flight frames. The argument buffer
/// cost is 5 × 4096 × 8 B ≈ 160 KB — negligible vs Metal Tier 2's 500K
/// limit. Vulkan and DX12 already use 16 384.
pub const MAX_RESOURCES_PER_CATEGORY: u32 = 4096;

use super::super::shared::SlotAllocator;

/// Registry for tracking bindless resource indices
///
/// The layout matches GoldyBindlessResources in bindless_resources.slang.
/// Each category occupies [`MAX_RESOURCES_PER_CATEGORY`] slots:
/// - storageBuffers at indices `0..N`   (Scattered access)
/// - uniformBuffers at indices `N..2N`  (Broadcast access)
/// - textures at indices `2N..3N`       (Interpolated / Texture2D)
/// - storageImages at indices `3N..4N`  (Direct / RWTexture2D)
/// - samplers at indices `4N..5N`       (Filter config)
///
/// where `N = MAX_RESOURCES_PER_CATEGORY`.
///
/// Each category uses a [`SlotAllocator`] that recycles freed LOCAL indices
/// before minting new ones. The global argument-buffer encoding offset
/// is computed by adding the category's base (`*_global_index` helpers).
/// Metal's two-phase recycle (slots parked in `pending_free_*_slots` until the
/// GPU timeline advances past the barrier) is layered on top: pending slots are
/// promoted to the `SlotAllocator` free list by `drain_pending_slots_up_to`.
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    /// Slot allocators per resource category (all return LOCAL 0-based indices).
    storage_buffer: SlotAllocator,
    uniform_buffer: SlotAllocator,
    texture: SlotAllocator,
    storage_image: SlotAllocator,
    sampler: SlotAllocator,
    /// Slots released while at least one GPU command buffer was in-flight.
    ///
    /// Metal's argument buffer is **just CPU-writable memory**: the descriptor
    /// at slot N is a device pointer that the shader dereferences at dispatch
    /// time. If we recycle slot N (overwrite its descriptor) while an in-flight
    /// command buffer still has dispatches that will read slot N, the GPU
    /// reads the *new* descriptor — pointing at a different buffer than the
    /// shader was originally compiled to expect. The result is descriptor
    /// aliasing: random garbage in some dispatches, and (eventually) an
    /// `MTLCommandBufferError::Internal` when the shader dereferences a
    /// pointer that happens to fall outside the resource's residency set.
    ///
    /// To prevent this, `destroy_*` checks whether the GPU is idle (all
    /// entries in `compute_fence_pool` are `Completed`). If so, slots go
    /// straight to the free list above. Otherwise they park here until
    /// `wait_fence()` succeeds, at which point `drain_pending_slots()`
    /// promotes them to the free list.
    /// Each entry is `(local_index, barrier)` where `barrier` is the
    /// `timeline_scheduled_max` recorded at destroy time. The slot becomes
    /// safe to recycle once `timeline_event.signaled_value() >= barrier`.
    pending_free_storage_buffer_slots: Vec<(u32, TimelineValue)>,
    pending_free_uniform_buffer_slots: Vec<(u32, TimelineValue)>,
    pending_free_texture_slots: Vec<(u32, TimelineValue)>,
    pending_free_storage_image_slots: Vec<(u32, TimelineValue)>,
    /// (local_index, access) for each live buffer handle. The access is
    /// needed at `unregister_buffer()` time to know which free list the slot
    /// should be returned to.
    pub buffer_indices: HashMap<BufferHandle, (u32, BufferKind)>,
    pub texture_indices: HashMap<TextureHandle, u32>,
    pub sampler_indices: HashMap<SamplerHandle, u32>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a storage buffer (Scattered access) - LOCAL indices 0..MAX_RESOURCES_PER_CATEGORY.
    ///
    /// Reuses a freed slot if available (see `unregister_buffer`). Without
    /// this reuse, long-running apps that churn transient buffers every frame
    /// (e.g. pool views) exhaust the argument-buffer encoder in seconds.
    ///
    /// # Panics on overflow
    ///
    /// If the local index would exceed [`MAX_RESOURCES_PER_CATEGORY`], the
    /// returned slot would silently bleed into the next category's
    /// argument-buffer region (uniform buffers at 64-127). The shader's
    /// `goldy_buf_ro<T>(slot)` would then read undefined / zero bytes
    /// from a wrong heap entry — observed in ekrano as binning's
    /// `config.lines_size == 0` and a spurious `STAGE_FLATTEN` overflow.
    /// Reserve low storage-buffer slots (frame-table selector + device table).
    pub fn ensure_storage_start(&mut self, min: u32) {
        self.storage_buffer.ensure_minimum_next(min);
    }

    /// Failing fast surfaces the leak instead of producing corrupt frames.
    pub fn register_storage_buffer(&mut self, handle: BufferHandle) -> u32 {
        assert!(
            self.storage_buffer.next_fresh() < MAX_RESOURCES_PER_CATEGORY || self.storage_buffer.free_count() > 0,
            "storage-buffer bindless slots exhausted ({MAX_RESOURCES_PER_CATEGORY} max). \
             next_index={} free={} pending_free={} live_indices={} \
             (Scattered={}, Broadcast={}). \
             Likely a per-frame leak in bind_map; check that all transient buffers \
             (config_buf, scene_buf, indirect_buf, etc.) are explicitly freed via \
             `recording.free_buffer(...)` and that `run_recording` evicts them at \
             the end of the frame.",
            self.storage_buffer.next_fresh(),
            self.storage_buffer.free_count(),
            self.pending_free_storage_buffer_slots.len(),
            self.buffer_indices.len(),
            self.buffer_indices
                .values()
                .filter(|(_, a)| *a == BufferKind::Scattered)
                .count(),
            self.buffer_indices
                .values()
                .filter(|(_, a)| *a == BufferKind::Broadcast)
                .count(),
        );
        let local_index = self.storage_buffer.alloc();
        self.buffer_indices.insert(handle, (local_index, BufferKind::Scattered));
        local_index
    }

    /// Register a uniform buffer (Broadcast access) — returns a LOCAL index.
    /// The global argument-buffer encoding offset is `uniform_global_index(local)`.
    ///
    /// Reuses a freed slot if available (see `unregister_buffer`).
    ///
    /// # Panics on overflow
    ///
    /// See [`Self::register_storage_buffer`] for the analogous rationale —
    /// a uniform-slot overflow would alias into the texture-index region
    /// and cause silent shader-side garbage reads.
    pub fn register_uniform_buffer(&mut self, handle: BufferHandle) -> u32 {
        assert!(
            self.uniform_buffer.next_fresh() < MAX_RESOURCES_PER_CATEGORY || self.uniform_buffer.free_count() > 0,
            "uniform-buffer bindless slots exhausted ({MAX_RESOURCES_PER_CATEGORY} max). \
             Likely a per-frame leak in bind_map for Broadcast buffers."
        );
        let local_index = self.uniform_buffer.alloc();
        self.buffer_indices.insert(handle, (local_index, BufferKind::Broadcast));
        local_index
    }

    /// Returns the global argument buffer index for a uniform buffer
    /// (local + MAX_RESOURCES_PER_CATEGORY), needed for encoding offsets.
    pub fn uniform_global_index(local_index: u32) -> u32 {
        local_index + MAX_RESOURCES_PER_CATEGORY
    }

    /// Register a sampled texture (Interpolated / Texture2D) — returns a LOCAL index.
    /// Use `texture_global_index()` to get the argument buffer encoding offset.
    ///
    /// Reuses a freed slot if available (see `release_texture_slot`).
    pub fn register_texture(&mut self, handle: TextureHandle) -> u32 {
        let local_index = self.texture.alloc();
        self.texture_indices.insert(handle, local_index);
        local_index
    }

    /// Return a sampled-texture LOCAL index to the free list so it can be
    /// reused by a subsequent `register_texture`.
    ///
    /// Pass `Some(barrier)` when any GPU command buffer is still in-flight
    /// that might still reference this slot's descriptor; the slot parks in
    /// the pending list until `drain_pending_slots_up_to(signaled)` promotes
    /// it. Pass `None` when the GPU is known idle.
    pub fn release_texture_slot(&mut self, local_index: u32, barrier: Option<TimelineValue>) {
        match barrier {
            Some(b) => self.pending_free_texture_slots.push((local_index, b)),
            None => self.texture.free(local_index),
        }
    }

    /// Returns the global argument buffer index for a sampled texture.
    pub fn texture_global_index(local_index: u32) -> u32 {
        local_index + 2 * MAX_RESOURCES_PER_CATEGORY
    }

    /// Register a storage image (Direct / RWTexture2D) — returns a LOCAL index.
    /// Use `storage_image_global_index()` to get the argument buffer encoding offset.
    ///
    /// Reuses a freed slot if available (see `release_storage_image_slot`).
    pub fn register_storage_image(&mut self, handle: TextureHandle) -> u32 {
        let local_index = self.storage_image.alloc();
        self.texture_indices.insert(handle, local_index);
        local_index
    }

    /// Reserve a storage-image LOCAL index without binding it to a TextureHandle.
    ///
    /// Used for transient bindless slots that outlive any single `TextureHandle`
    /// but belong to a long-lived owner (e.g. a `Surface` that re-encodes a
    /// fresh drawable into the same slot every frame). The owner must release
    /// the slot via [`Self::release_storage_image_slot`] when destroyed.
    pub fn reserve_storage_image_slot(&mut self) -> u32 {
        self.storage_image.alloc()
    }

    /// Associate a TextureHandle with a previously-reserved storage-image
    /// LOCAL index so `texture_bindless_index()` / `Texture::bindless_index()`
    /// resolves to the right slot. Does not bump any counters.
    pub fn bind_storage_image_slot(&mut self, handle: TextureHandle, local_index: u32) {
        self.texture_indices.insert(handle, local_index);
    }

    /// Return a storage-image LOCAL index to the free list so it can be
    /// reused by a subsequent `register_storage_image` / `reserve_storage_image_slot`.
    ///
    /// See [`Self::release_texture_slot`] for the `barrier` semantics.
    pub fn release_storage_image_slot(&mut self, local_index: u32, barrier: Option<TimelineValue>) {
        match barrier {
            Some(b) => self.pending_free_storage_image_slots.push((local_index, b)),
            None => self.storage_image.free(local_index),
        }
    }

    /// Returns the global argument buffer index for a storage image.
    pub fn storage_image_global_index(local_index: u32) -> u32 {
        local_index + 3 * MAX_RESOURCES_PER_CATEGORY
    }

    /// Register a sampler — returns a LOCAL index for resource slots.
    /// Use `sampler_global_index()` to get the argument buffer encoding offset.
    pub fn register_sampler(&mut self, handle: SamplerHandle) -> u32 {
        let local_index = self.sampler.alloc();
        self.sampler_indices.insert(handle, local_index);
        local_index
    }

    /// Returns the global argument buffer index for a sampler,
    /// needed for encoding offsets.
    pub fn sampler_global_index(local_index: u32) -> u32 {
        local_index + 4 * MAX_RESOURCES_PER_CATEGORY
    }

    /// Remove a buffer handle from the registry and return its LOCAL slot
    /// to the appropriate free list so subsequent `register_*_buffer` calls
    /// reuse it. Without this, per-frame buffer churn exhausts the
    /// argument-buffer window and the shader starts reading
    /// into corrupt / out-of-range descriptors.
    ///
    /// Pass `Some(barrier)` when any GPU command buffer that may still reference
    /// this slot's descriptor is still in-flight; the slot parks in the pending
    /// list and is promoted by `drain_pending_slots_up_to(signaled)` once
    /// `signaled >= barrier`. Pass `None` when the GPU is known idle.
    pub fn unregister_buffer(&mut self, handle: BufferHandle, barrier: Option<TimelineValue>) {
        if let Some((local_index, access)) = self.buffer_indices.remove(&handle) {
            match (access, barrier) {
                (BufferKind::Scattered, None) => {
                    self.storage_buffer.free(local_index);
                }
                (BufferKind::Scattered, Some(b)) => {
                    self.pending_free_storage_buffer_slots.push((local_index, b));
                }
                (BufferKind::Broadcast, None) => {
                    self.uniform_buffer.free(local_index);
                }
                (BufferKind::Broadcast, Some(b)) => {
                    self.pending_free_uniform_buffer_slots.push((local_index, b));
                }
            }
        }
    }

    pub fn unregister_texture(&mut self, handle: TextureHandle) {
        self.texture_indices.remove(&handle);
    }

    pub fn unregister_sampler(&mut self, handle: SamplerHandle) {
        self.sampler_indices.remove(&handle);
    }

    /// Promote pending slots whose GPU barrier has been signaled.
    ///
    /// Only entries where `barrier <= signaled` are moved to the [`SlotAllocator`]
    /// free list; entries still waiting for a higher timeline value stay pending.
    /// This is the per-frame call path — invoked on every `acquire_frame` /
    /// `present` so slots are recycled as soon as the GPU catches up.
    pub fn drain_pending_slots_up_to(&mut self, signaled: TimelineValue) {
        macro_rules! drain_to_allocator {
            ($pending:expr, $alloc:expr) => {{
                let mut i = 0;
                while i < $pending.len() {
                    let (slot, barrier) = $pending[i];
                    if barrier <= signaled {
                        $pending.swap_remove(i);
                        $alloc.free(slot);
                    } else {
                        i += 1;
                    }
                }
            }};
        }
        drain_to_allocator!(self.pending_free_storage_buffer_slots, self.storage_buffer);
        drain_to_allocator!(self.pending_free_uniform_buffer_slots, self.uniform_buffer);
        drain_to_allocator!(self.pending_free_texture_slots, self.texture);
        drain_to_allocator!(self.pending_free_storage_image_slots, self.storage_image);
    }

    /// Promote all pending slots unconditionally.
    ///
    /// Only safe to call after `wait_all_in_flight` has confirmed that no
    /// GPU command buffers are in-flight. For the per-frame path use
    /// `drain_pending_slots_up_to(signaled)` instead.
    pub fn drain_pending_slots(&mut self) {
        self.drain_pending_slots_up_to(TimelineValue::MAX);
    }

    /// Number of available (allocatable) slots in the given category.
    ///
    /// Includes both recycled free-list entries and not-yet-minted slots up to
    /// [`MAX_RESOURCES_PER_CATEGORY`]. Pending-free slots (awaiting GPU timeline)
    /// are counted as occupied.
    pub fn available_slots(&self, category: crate::types::ResourceCategory) -> u32 {
        let allocator = match category {
            crate::types::ResourceCategory::Scattered => &self.storage_buffer,
            crate::types::ResourceCategory::Broadcast => &self.uniform_buffer,
            crate::types::ResourceCategory::Texture => &self.texture,
            crate::types::ResourceCategory::StorageImage => &self.storage_image,
            crate::types::ResourceCategory::Sampler => &self.sampler,
        };
        MAX_RESOURCES_PER_CATEGORY.saturating_sub(allocator.live_count())
    }

    /// Number of buffer slots currently waiting for GPU idle before reuse.
    /// Exposed for tests; not part of the public API.
    #[cfg(test)]
    pub fn pending_buffer_slot_count(&self) -> usize {
        self.pending_free_storage_buffer_slots.len() + self.pending_free_uniform_buffer_slots.len()
    }

    /// Number of free storage-buffer slots ready for immediate reuse.
    /// Exposed for tests to verify recycling without touching internals.
    #[cfg(test)]
    pub fn free_storage_buffer_count(&self) -> usize {
        self.storage_buffer.free_count()
    }

    /// Number of free uniform-buffer slots ready for immediate reuse.
    #[cfg(test)]
    pub fn free_uniform_buffer_count(&self) -> usize {
        self.uniform_buffer.free_count()
    }
}

/// Device-shared descriptor registry.
///
/// Wraps `ResourceRegistry` (the bindless slot allocator + pending-free lists)
/// behind an `Arc<Mutex<>>` so that submit paths can acquire it independently
/// of the global backend mutex. The critical section is intentionally minimal:
/// slot alloc/free only. MTL object encoding and GPU submission stay outside.
///
/// Metal does not need `slot_last_seen` / `pending_slot_reclamations` tracking
/// (those exist on Vulkan/DX12 for multi-referencer slots). Under the
/// no-cross-context-dependency axiom each slot has a single owning context, so
/// per-context own-clock reclaim is sufficient; the conservative
/// `device_retired` GC drain is kept as a safety net (see issue #190).
pub(crate) struct DescriptorRegistry {
    pub resource_registry: ResourceRegistry,
}

impl DescriptorRegistry {
    pub(crate) fn new() -> Self {
        Self {
            resource_registry: ResourceRegistry::new(),
        }
    }
}

/// GPU buffer state.
#[derive(Clone)]
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub buffer: MTLBuffer,
    /// Logical byte size (API-visible).
    pub size: u64,
    /// Backing allocation size (`MTLBuffer.length()`).
    pub allocation_size: u64,
    /// `true` when created via [`MTLDevice::new_buffer`] (jumbo) rather than a heap.
    pub is_device_allocated: bool,
    /// Index in the global argument buffer (always present — heap required).
    pub arg_buffer_index: u32,
    pub flags: crate::types::BufferFlags,
    /// Structured-buffer element stride from buffer creation (for stride validation).
    pub element_stride: Option<u32>,
    /// For buffer views: parent [`BufferHandle`]. `None` for root buffers.
    pub parent_for_view: Option<BufferHandle>,
    /// Access pattern at creation (for argument-buffer re-encoding on resize).
    pub access: BufferKind,
    /// Byte offset into parent for views; [`None`] for root buffers.
    pub view_byte_offset: Option<u64>,
    /// Grant-read staging buffer (shared storage, CPU-readable; no shader binding).
    pub is_grant_readback: bool,
    pub texture_copy_footprint: Option<crate::backend::TextureCopyFootprint>,
}

/// Shader module state with cached compiled stages.
pub(crate) struct ShaderState {
    pub device_handle: DeviceHandle,
    pub slang_source: String,
    /// Search paths for Slang module resolution
    pub search_paths: Vec<String>,
    /// Extra preprocessor defines (e.g. msaa, msaa8)
    pub defines: Vec<(String, String)>,
    /// Per-shader Slang optimization level
    pub optimization_level: crate::types::OptimizationLevel,
    /// Compiled vertex shader library
    pub vertex_library: Option<Library>,
    /// Compiled fragment shader library
    pub fragment_library: Option<Library>,
    /// Compiled compute shader library
    pub compute_library: Option<Library>,
    /// Reflection data for bindless rendering (ParameterBlock layouts)
    pub reflection: Option<crate::slang::ShaderReflection>,
    /// Pending struct layout validation on first stage compile; cleared after success.
    pub layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
}

/// Graphics pipeline state.
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: RenderPipelineState,
    pub depth_stencil: Option<MTLDepthStencilState>,
    pub primitive_type: MTLPrimitiveType,
    /// Per push-constant slot expected element stride (bytes) from reflection.
    pub binding_element_strides: Vec<Option<u32>>,
    /// Human-readable identifier for debugging.
    pub shader_debug_name: String,
}

/// Compute pipeline state.
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: MTLComputePipelineState,
    /// Thread group size from [numthreads(x, y, z)] attribute
    pub workgroup_size: [u32; 3],
    /// Per push-constant slot category expectations from shader analysis.
    pub push_constant_categories: Vec<Option<crate::types::ResourceCategory>>,
    /// Per push-constant slot expected element stride (bytes) from reflection.
    pub binding_element_strides: Vec<Option<u32>>,
    /// Human-readable identifier for debugging.
    pub shader_debug_name: String,
}

/// GPU render target state with optional staging for CPU readback.
pub(crate) struct RenderTargetState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// GPU render target texture
    pub texture: MTLTexture,
    /// Depth buffer (optional)
    pub depth_texture: Option<MTLTexture>,
    /// Track if we've rendered (for readback validation)
    pub has_rendered: bool,
}

/// GPU texture state.
pub(crate) struct TextureState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub texture: MTLTexture,
    /// LOCAL index in the texture category this
    /// texture was registered in (`storageImages[]` when `is_storage_image`,
    /// otherwise `textures[]`).
    pub arg_buffer_index: u32,
    /// For `TextureKind::DirectInterpolated` textures, the LOCAL index in the
    /// sampled-texture pool (separate from the storage-image `arg_buffer_index`).
    pub sampled_arg_buffer_index: Option<u32>,
    /// Which bindless region the `arg_buffer_index` belongs to; needed at
    /// destroy time to release the slot back to the correct free list.
    pub is_storage_image: bool,
    /// When true, the slot is owned by a long-lived entity (e.g. a `Surface`
    /// that re-encodes its drawable each frame) and should NOT be released
    /// when this `TextureState` is dropped. The owner manages slot lifetime.
    pub slot_owned_externally: bool,
    /// `true` when allocated from a Goldy-owned `MTLHeap` (texture_heap or
    /// transient heap). Heap-resident textures are already covered by
    /// `use_heap` and don't need individual `use_resource` calls.
    #[allow(dead_code)]
    pub is_heap_allocated: bool,
}

/// GPU sampler state.
pub(crate) struct SamplerState_ {
    pub device_handle: DeviceHandle,
    /// Held so the GPU sampler stays resident in memory while its resource ID is
    /// encoded in the argument buffer. The field is never read after construction;
    /// `#[allow(dead_code)]` is intentional — dropping it early would invalidate
    /// the GPU-side binding.
    #[allow(dead_code)]
    pub sampler: SamplerState,
    /// Index in the global argument buffer (always present).
    pub arg_buffer_index: u32,
}

/// Maximum number of frames that can be in-flight at once.
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// Present-completion bookkeeping drained by [`ContextPollReader::poll_signals`].
pub(crate) struct PendingSwapchainReturn {
    pub image_index: u32,
    pub pending_acquire_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

/// Surface (CAMetalLayer) state for window presentation.
pub(crate) struct SurfaceState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// Depth buffer (optional)
    pub depth_format: Option<DepthFormat>,
    pub depth_texture: Option<MTLTexture>,
    /// Current frame index (0..MAX_FRAMES_IN_FLIGHT)
    pub current_frame: usize,
    /// The CAMetalLayer (stored as raw pointer for objc interop)
    pub layer: *mut std::ffi::c_void,
    /// Per-slot acquired CAMetalDrawables (set during acquire, cleared on present).
    pub drawable_slots: [Option<*mut std::ffi::c_void>; MAX_FRAMES_IN_FLIGHT],
    /// Texture handle for each slot's drawable (registered for bindless access).
    pub drawable_texture_handles: [Option<TextureHandle>; MAX_FRAMES_IN_FLIGHT],
    /// Texture handle for the most recently acquired frame (for `frame_texture()`).
    pub current_texture_handle: Option<TextureHandle>,
    /// Triple-buffered storage-image LOCAL indices reserved at surface create.
    /// Each frame uses `bindless_storage_slots[current_frame]` so the CPU never
    /// re-encodes a slot that the GPU is still reading from a previous frame.
    /// Released back to the device's `ResourceRegistry` free list on surface destroy.
    pub bindless_storage_slots: [u32; MAX_FRAMES_IN_FLIGHT],
    /// Current present mode
    pub present_mode: crate::types::PresentMode,
    /// Frame-scoped GPU commands ([`crate::backend::GpuBackend::record_gpu_work`]).
    pub frame_pending_gpu_commands: Vec<crate::backend::GpuCommand>,
    pub pending_acquire_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
    pub last_acquired_image_index: Option<u32>,
    /// Prepared acquire state between [`surface::take_surface_acquire_work`] and
    /// [`surface::finish_surface_acquire_from_drawable`] (split `nextDrawable` path).
    pub pending_acquire: Option<super::surface::MetalAcquirePending>,
}

// SAFETY: `SurfaceState` contains raw pointers to a `CALayer` and `CAMetalDrawable`.
// These Metal objects are reference-counted by Objective-C and are themselves thread-safe
// for retain/release. All mutation (acquire, present, destroy) is serialised through the
// `MetalBackend` mutex (`backend.lock()` in every GpuBackend method), so no two threads
// can access these pointers concurrently. Callers must uphold this invariant — do not
// share `SurfaceState` directly across threads without the backend lock.
unsafe impl Send for SurfaceState {}
unsafe impl Sync for SurfaceState {}

/// Physical Metal GPU adapter (the `MTLDevice` is both adapter and device substrate).
pub(crate) struct MetalAdapterInfo {
    pub device: mtl::Device,
    pub adapter_id: u32,
}

/// `Arc`-wrapped logical device — cloned out of `MetalState` before the global backend
/// lock is dropped so submit paths can hold per-device state independently (phase 5).
pub(crate) type SharedLogicalDevice = Arc<LogicalDevice>;

/// `Arc<Mutex>`-wrapped per-context state — allows submit paths to lock only the
/// submitting context rather than all of `MetalState` (phase 5).
pub(crate) type SharedMetalSubmissionContext = Arc<Mutex<MetalSubmissionContext>>;

/// Consolidated Metal backend state.
/// Holds all resources and state for the Metal backend.
pub(super) struct MetalState {
    /// Physical adapters discovered at backend init.
    pub adapters: Vec<MetalAdapterInfo>,
    /// Set once any GPU wait has timed out (the GPU has wedged in a compute
    /// shader without the driver watchdog noticing). Subsequent waits fail
    /// fast instead of burning the full timeout budget per frame, letting
    /// the app cascade errors quickly and exit cleanly rather than appearing
    /// frozen for tens of seconds while each frame times out.
    pub device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub devices: std::collections::HashMap<DeviceHandle, SharedLogicalDevice>,
    pub next_device_handle: DeviceHandle,
    pub contexts: std::collections::HashMap<super::ContextHandle, SharedMetalSubmissionContext>,
    pub next_context_id: super::ContextHandle,
    pub buffers: std::collections::HashMap<BufferHandle, BufferState>,
    pub next_buffer_handle: BufferHandle,
    pub shaders: std::collections::HashMap<ShaderHandle, ShaderState>,
    pub next_shader_handle: ShaderHandle,
    pub pipelines: std::collections::HashMap<PipelineHandle, PipelineState>,
    pub next_pipeline_handle: PipelineHandle,
    pub compute_pipelines: std::collections::HashMap<ComputePipelineHandle, ComputePipelineState>,
    pub next_compute_pipeline_handle: ComputePipelineHandle,
    pub render_targets: std::collections::HashMap<RenderTargetHandle, RenderTargetState>,
    pub next_render_target_handle: RenderTargetHandle,
    pub surfaces: std::collections::HashMap<SurfaceHandle, SurfaceState>,
    pub next_surface_handle: SurfaceHandle,
    pub textures: std::collections::HashMap<TextureHandle, TextureState>,
    pub next_texture_handle: TextureHandle,
    pub samplers: std::collections::HashMap<SamplerHandle, SamplerState_>,
    pub next_sampler_handle: SamplerHandle,
    /// `None` after release via [`crate::device::Device::release_idle_shader_compiler`].
    /// Re-created automatically on demand when a shader must be lazily compiled.
    pub slang_compiler: Option<crate::slang::SlangCompiler>,
}

impl MetalState {
    #[inline]
    pub(super) fn slang_compiler_mut_or_init(&mut self) -> anyhow::Result<&mut crate::slang::SlangCompiler> {
        use anyhow::Context;
        if self.slang_compiler.is_none() {
            self.slang_compiler = Some(crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?);
        }
        Ok(self.slang_compiler.as_mut().expect("just set"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the ekrano "black screen after ~540 frames" bug.
    ///
    /// Before the free-list fix, `register_storage_buffer` bumped a monotonic
    /// counter with every call and `unregister_buffer` merely deleted the
    /// HashMap entry — the LOCAL slot was leaked. Per-frame pool-view churn
    /// blew through the 64-slot storage-buffer window in ~9 s and subsequent
    /// slots bled into the uniform/texture/storage-image categories, which
    /// corrupted the surface drawable and wedged the GPU.
    ///
    /// This test would have failed: after 64 register+unregister cycles, the
    /// 65th registration would have returned slot 64 instead of recycling 0.
    #[test]
    fn storage_buffer_slots_are_reused_after_unregister() {
        let mut reg = ResourceRegistry::new();
        let mut all_indices = Vec::new();

        // Churn 4x the per-category capacity; without slot reuse this would
        // return 0, 1, 2, ..., 4*64 - 1 and blow through into the uniform
        // region of the argument buffer.
        for handle in 0..(MAX_RESOURCES_PER_CATEGORY as u64 * 4) {
            let idx = reg.register_storage_buffer(handle);
            all_indices.push(idx);
            // defer=false: simulate the "GPU idle when destroy fires" case,
            // e.g. ekrano's end-of-frame deferred_free_buffers after flush.
            reg.unregister_buffer(handle, None);
        }

        assert!(
            all_indices.iter().all(|&i| i < MAX_RESOURCES_PER_CATEGORY),
            "storage buffer slots escaped the 0..{} window: {:?}",
            MAX_RESOURCES_PER_CATEGORY,
            all_indices
                .iter()
                .filter(|&&i| i >= MAX_RESOURCES_PER_CATEGORY)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn uniform_buffer_slots_are_reused_after_unregister() {
        let mut reg = ResourceRegistry::new();
        let mut all_indices = Vec::new();
        for handle in 0..(MAX_RESOURCES_PER_CATEGORY as u64 * 4) {
            let idx = reg.register_uniform_buffer(handle);
            all_indices.push(idx);
            reg.unregister_buffer(handle, None);
        }
        assert!(
            all_indices.iter().all(|&i| i < MAX_RESOURCES_PER_CATEGORY),
            "uniform buffer slots escaped the 0..{} window: {:?}",
            MAX_RESOURCES_PER_CATEGORY,
            all_indices
                .iter()
                .filter(|&&i| i >= MAX_RESOURCES_PER_CATEGORY)
                .collect::<Vec<_>>()
        );
    }

    /// Ensure a Broadcast handle freed via `unregister_buffer` goes to the
    /// uniform free list (not the storage list), so a subsequent Scattered
    /// registration can't accidentally re-hand out a uniform-local index
    /// that isn't actually free in the storage category.
    #[test]
    fn unregister_routes_slot_to_correct_category() {
        let mut reg = ResourceRegistry::new();
        let h_uni: BufferHandle = 10;
        let h_sto: BufferHandle = 20;

        let _ = reg.register_uniform_buffer(h_uni);
        let _ = reg.register_storage_buffer(h_sto);
        reg.unregister_buffer(h_uni, None);
        reg.unregister_buffer(h_sto, None);

        assert_eq!(
            reg.free_uniform_buffer_count(),
            1,
            "uniform free list should have reclaimed the uniform slot"
        );
        assert_eq!(
            reg.free_storage_buffer_count(),
            1,
            "storage free list should have reclaimed the storage slot"
        );
    }

    /// Slots recycled via the free list must hand back the most recently
    /// freed index first (LIFO) — primarily to make behavior deterministic
    /// for tests, and to keep hot slots warm.
    #[test]
    fn freed_storage_buffer_slot_is_reused_lifo() {
        let mut reg = ResourceRegistry::new();
        let h0: BufferHandle = 1;
        let h1: BufferHandle = 2;

        let i0 = reg.register_storage_buffer(h0);
        let i1 = reg.register_storage_buffer(h1);
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);

        reg.unregister_buffer(h0, None);
        reg.unregister_buffer(h1, None);

        let h2: BufferHandle = 3;
        let i2 = reg.register_storage_buffer(h2);
        assert_eq!(i2, 1, "expected LIFO reuse of freed slot");
    }

    /// Deferred release: slots freed while `defer=true` must not be reused by
    /// the next `register_*` call — they stay in the pending list until
    /// `drain_pending_slots()` promotes them. This is the core mechanism
    /// protecting against descriptor-aliasing with in-flight GPU work.
    #[test]
    fn deferred_buffer_slot_is_not_reused_until_drain() {
        let mut reg = ResourceRegistry::new();
        let h0: BufferHandle = 1;

        let i0 = reg.register_storage_buffer(h0);
        assert_eq!(i0, 0);

        // GPU is "busy" → park on pending.
        reg.unregister_buffer(h0, Some(1));
        assert_eq!(reg.pending_buffer_slot_count(), 1, "expected slot to land in pending");

        // Until drain, register must NOT re-hand out slot 0.
        let h1: BufferHandle = 2;
        let i1 = reg.register_storage_buffer(h1);
        assert_eq!(i1, 1, "register_storage_buffer must not recycle a still-pending slot");

        // After drain, pending→free, and the next register picks it up.
        reg.drain_pending_slots();
        assert_eq!(reg.pending_buffer_slot_count(), 0);

        reg.unregister_buffer(h1, None);
        let h2: BufferHandle = 3;
        let i2 = reg.register_storage_buffer(h2);
        assert_eq!(i2, 1, "LIFO pick from free list after drain");
    }

    /// Drain must route pending slots back to the right category (storage vs
    /// uniform). A bug here would let a pending uniform slot show up as a
    /// "free" storage slot and vice versa.
    #[test]
    fn drain_pending_routes_to_correct_category() {
        let mut reg = ResourceRegistry::new();
        let h_sto: BufferHandle = 10;
        let h_uni: BufferHandle = 20;
        let _ = reg.register_storage_buffer(h_sto);
        let _ = reg.register_uniform_buffer(h_uni);

        reg.unregister_buffer(h_sto, Some(1));
        reg.unregister_buffer(h_uni, Some(1));
        reg.drain_pending_slots();

        assert_eq!(reg.free_storage_buffer_count(), 1);
        assert_eq!(reg.free_uniform_buffer_count(), 1);
        assert_eq!(reg.pending_buffer_slot_count(), 0);
    }

    /// Texture slots must honor the same deferred-release pattern. In the
    /// pre-fix behaviour, a font-atlas texture slot could be recycled while
    /// the previous frame's compute shader was still reading it, producing
    /// the "glitchy stats-overlay text" symptom.
    #[test]
    fn deferred_texture_slot_is_not_reused_until_drain() {
        let mut reg = ResourceRegistry::new();
        let h0: TextureHandle = 100;
        let i0 = reg.register_texture(h0);
        reg.release_texture_slot(i0, Some(1));

        let h1: TextureHandle = 101;
        let i1 = reg.register_texture(h1);
        assert_ne!(i1, i0, "texture slot must not be recycled while still pending");

        reg.drain_pending_slots();
        reg.release_texture_slot(i1, None);
        let h2: TextureHandle = 102;
        let i2 = reg.register_texture(h2);
        // Either the just-released i1 or the drained i0 is acceptable; the
        // guarantee we need is that it's one of the freed slots (not a fresh
        // monotonic bump).
        assert!(
            i2 == i0 || i2 == i1,
            "expected texture slot reuse after drain, got {i2}"
        );
    }
}
