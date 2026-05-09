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
    BufferHandle, ComputePipelineHandle, DeviceHandle, PipelineHandle, RenderTargetHandle,
    SamplerHandle, ShaderHandle, SurfaceHandle, TextureHandle,
};
use crate::backend::DataAccess;
use crate::timeline::TimelineValue;
use crate::types::{DepthFormat, TextureFormat};
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
// Use explicit crate path to avoid collision with our module name
use ::metal as mtl;
use mtl::{
    ArgumentEncoder, Buffer as MTLBuffer, CommandQueue,
    ComputePipelineState as MTLComputePipelineState, DepthStencilState as MTLDepthStencilState,
    Device as MTLDevice, Heap, Library, MTLPrimitiveType, MTLResourceOptions, RenderPipelineState,
    SamplerState, SharedEvent, Texture as MTLTexture,
};

/// Maximum size of the argument buffer (supports up to 16K resources)
pub const ARGUMENT_BUFFER_SIZE: u64 = 16 * 1024 * 8; // 8 bytes per resource ID

/// Buffer slot for resource binding indices in shaders.
/// Slang assigns uniform entry-point params to [[buffer(1)]] (gGoldy ParameterBlock takes [[buffer(0)]]).
pub const RESOURCE_SLOT_BUFFER: u64 = 1;

/// Starting Metal buffer index for vertex attributes.
/// Slots 0 and 1 are reserved for the argument buffer (gGoldy) and resource slots.
/// Vertex data must use higher indices to avoid collisions.
pub const VERTEX_BUFFER_START_SLOT: u64 = 2;

/// Maximum number of bindless resource indices in region A.
pub const MAX_BINDLESS_SLOTS: usize = 16;
/// Maximum number of u32 user parameters in region B.
pub const MAX_USER_SLOTS: usize = 8;
/// Total set_bytes size in bytes.
pub const TOTAL_PUSH_BYTES: usize = 128;

/// Packed 128-byte buffer layout passed via Metal `set_bytes`.
///
/// ```text
/// Bytes  0–31:  16 × u16  bindless resource indices  (region A)
/// Bytes 32–63:  8  × u32  user parameters            (region B)
/// Bytes 64–127: 64 × u8   reserved / future           (region C)
/// ```
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PushLayout {
    pub bindless: [u16; MAX_BINDLESS_SLOTS],
    pub user: [u32; MAX_USER_SLOTS],
    pub _reserved: [u32; 16],
}

const _: () = assert!(std::mem::size_of::<PushLayout>() == TOTAL_PUSH_BYTES);

impl PushLayout {
    /// Reinterpret the layout as a raw byte slice suitable for Metal's `set_bytes`.
    ///
    /// # Safety
    /// `PushLayout` is `#[repr(C)]` with a static size assertion (`TOTAL_PUSH_BYTES`),
    /// so the cast is well-defined with no padding surprises.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const _ as *const u8, TOTAL_PUSH_BYTES) }
    }
}

/// Minimum primary heap size (64 MB).
const MIN_HEAP_SIZE: u64 = 64 * 1024 * 1024;

/// Minimum overflow heap size (16 MB).
const MIN_OVERFLOW_HEAP_SIZE: u64 = 16 * 1024 * 1024;

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
const MAX_HEAP_SIZE: u64 = 1024 * 1024 * 1024; // 1 GB

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
    /// Tries primary, then the last overflow heap, then creates a new overflow.
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

        if let Some(last) = self.overflow.last() {
            if let Some(buf) = last.new_buffer(size, options) {
                self.buffer_count += 1;
                self.update_high_water_mark();
                return Some(buf);
            }
        }

        // Clamp the overflow heap size too: `size * 2` on a ~1 GB request is
        // already at the cap, and we do not want to chase the size in a loop.
        let overflow_size = (size * 2).clamp(MIN_OVERFLOW_HEAP_SIZE, MAX_HEAP_SIZE);
        let new_heap = self.create_heap(overflow_size);
        tracing::info!(
            "Created overflow buffer heap (size={}MB, overflow_count={})",
            overflow_size / 1024 / 1024,
            self.overflow.len() + 1
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
    pub fn use_heaps_for_render(
        &self,
        encoder: &mtl::RenderCommandEncoderRef,
        stages: mtl::MTLRenderStages,
    ) {
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
            tracing::debug!("Cleared {} overflow buffer heaps", overflow_count);
        }
        self.high_water_mark = 0;
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
    /// Tries primary, then the last overflow heap, then creates a new overflow.
    pub fn allocate(&mut self, descriptor: &mtl::TextureDescriptorRef) -> Option<MTLTexture> {
        if let Some(tex) = self.primary.new_texture(descriptor) {
            self.texture_count += 1;
            return Some(tex);
        }

        if let Some(last) = self.overflow.last() {
            if let Some(tex) = last.new_texture(descriptor) {
                self.texture_count += 1;
                return Some(tex);
            }
        }

        let alloc_size = self.device.heap_texture_size_and_align(descriptor).size;
        let overflow_size = (alloc_size * 2).max(MIN_OVERFLOW_HEAP_SIZE);
        let new_heap = self.create_heap(overflow_size);
        tracing::info!(
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

    pub fn has_textures(&self) -> bool {
        self.texture_count > 0
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
    pub fn use_heaps_for_render(
        &self,
        encoder: &mtl::RenderCommandEncoderRef,
        stages: mtl::MTLRenderStages,
    ) {
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

/// GPU resource retained until [`TimelineValue`] completion on [`LogicalDevice::timeline_event`].
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
    pending: Vec<(TimelineValue, PendingDeletion)>,
}

impl DeletionQueue {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    pub fn queue(&mut self, barrier: TimelineValue, resource: PendingDeletion) {
        self.pending.push((barrier, resource));
    }

    /// Destroy resources whose `barrier` has been signaled by the GPU (`signaled_value >= barrier`).
    pub fn process_up_to(&mut self, signaled: TimelineValue) {
        let (done, rest): (Vec<_>, Vec<_>) =
            self.pending.drain(..).partition(|(b, _)| *b <= signaled);
        self.pending = rest;
        drop(done);
    }

    pub fn flush_all(&mut self) {
        self.pending.clear();
    }

    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
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
}

impl TimelineWaiter {
    pub fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(0), Condvar::new())),
        }
    }

    /// Called from the `notifyListener` block when the GPU signals a timeline value.
    pub fn signal(&self, value: u64) {
        let (lock, cvar) = &*self.inner;
        let mut signaled = lock.lock().unwrap();
        if value > *signaled {
            *signaled = value;
        }
        cvar.notify_all();
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

/// A logical Metal device with associated resources.
///
/// Requires Argument Buffers Tier 2 (Apple Silicon, Intel 2017+, AMD 2015+).
pub(crate) struct LogicalDevice {
    pub device: MTLDevice,
    pub command_queue: CommandQueue,

    // Bindless infrastructure (always present — Tier 2 required)
    /// Multi-heap allocator for buffer allocations (grows on demand)
    pub heap_allocator: HeapAllocator,
    /// Multi-heap allocator for texture allocations (grows on demand)
    pub texture_heap: TextureHeapAllocator,
    /// Global argument buffer containing resource IDs
    pub argument_buffer: MTLBuffer,
    /// Encoder for writing buffers to the argument buffer
    pub argument_encoder: ArgumentEncoder,
    /// Encoder for writing sampled textures (ReadOnly) to the argument buffer
    pub texture_encoder: ArgumentEncoder,
    /// Encoder for writing storage images (ReadWrite) to the argument buffer
    pub storage_image_encoder: ArgumentEncoder,
    /// Registry tracking resource indices in the argument buffer
    pub resource_registry: ResourceRegistry,
    /// GPU timeline shared between standalone submits and surface presents.
    pub timeline_event: SharedEvent,
    /// Event-driven waiter for GPU timeline completion (replaces poll loop).
    pub timeline_waiter: TimelineWaiter,
    /// Next [`TimelineValue`] assigned on submission (`timeline_next` follows successful commits).
    pub timeline_next: u64,
    /// Highest timeline value scheduled on the GPU queue for this device (used for idle / flush).
    pub timeline_scheduled_max: u64,
    /// Deferred GPU resource teardown until [`SharedEvent`] reaches the queued barrier.
    pub deletion_queue: DeletionQueue,
}

impl LogicalDevice {
    pub(crate) fn process_deletion_queue_up_to_signaled(&mut self) {
        let signaled = self.timeline_event.as_ref().signaled_value();
        self.deletion_queue.process_up_to(signaled);
    }
}

/// Maximum resources per access pattern category (must match GOLDY_MAX_RESOURCES in shaders)
pub const MAX_RESOURCES_PER_CATEGORY: u32 = 64;

/// Registry for tracking bindless resource indices
///
/// The layout matches GoldyBindlessResources in bindless_resources.slang:
/// - storageBuffers[64] at indices 0-63   (Scattered access)
/// - uniformBuffers[64] at indices 64-127 (Broadcast access)
/// - textures[64] at indices 128-191      (Interpolated / Texture2D)
/// - storageImages[64] at indices 192-255 (Direct / RWTexture2D)
/// - samplers[64] at indices 256-319      (Filter config)
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    next_storage_buffer_index: u32,
    next_uniform_buffer_index: u32,
    next_texture_index: u32,
    next_storage_image_index: u32,
    next_sampler_index: u32,
    /// Free list of previously-released storage-image LOCAL indices.
    ///
    /// Populated by `release_storage_image_slot()` (used when a Surface is
    /// destroyed). `register_storage_image()` / `reserve_storage_image_slot()`
    /// pop from this list before bumping `next_storage_image_index`, so
    /// transient bindless slots (per-frame swapchain drawables) don't leak
    /// across the 64-slot storage-image window.
    free_storage_image_slots: Vec<u32>,
    /// Symmetric free list for sampled (Interpolated / `Texture2D`) slots.
    ///
    /// Populated by `release_texture_slot()` when a regular texture is
    /// destroyed; `register_texture()` consults it before bumping
    /// `next_texture_index`. Prevents slot exhaustion when textures are
    /// created and destroyed repeatedly at runtime.
    free_texture_slots: Vec<u32>,
    /// Free list of previously-released storage-buffer LOCAL indices.
    ///
    /// Without this list, `register_storage_buffer()` monotonically bumps
    /// `next_storage_buffer_index` forever. In workloads that allocate
    /// transient pool views every frame (e.g. ekrano's per-frame
    /// `BufferPool::alloc_bytes` calls, each of which creates a buffer view),
    /// the counter grows past the 64-slot storage-buffer window within
    /// seconds, then past 128 it writes descriptors into the uniform-buffer
    /// region of the argument buffer, past 192 into the storage-image region
    /// (corrupting the surface drawable!), and eventually past the whole
    /// argument buffer — at which point `set_argument_buffer` silently skips
    /// the encode (bounds check) and the shader reads a zero pointer,
    /// producing empty frames or a wedged GPU.
    free_storage_buffer_slots: Vec<u32>,
    /// Free list of previously-released uniform-buffer LOCAL indices.
    /// Same rationale as `free_storage_buffer_slots`.
    free_uniform_buffer_slots: Vec<u32>,
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
    pending_free_storage_buffer_slots: Vec<u32>,
    pending_free_uniform_buffer_slots: Vec<u32>,
    pending_free_texture_slots: Vec<u32>,
    pending_free_storage_image_slots: Vec<u32>,
    /// (local_index, access) for each live buffer handle. The access is
    /// needed at `unregister_buffer()` time to know which free list the slot
    /// should be returned to.
    pub buffer_indices: HashMap<BufferHandle, (u32, DataAccess)>,
    pub texture_indices: HashMap<TextureHandle, u32>,
    pub sampler_indices: HashMap<SamplerHandle, u32>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            // Storage buffers (Scattered) at indices 0-63, bytes 0-511
            next_storage_buffer_index: 0,
            // Uniform buffers (Broadcast) at indices 64-127, bytes 512-1023
            next_uniform_buffer_index: MAX_RESOURCES_PER_CATEGORY,
            // Textures (Interpolated) at indices 128-191, bytes 1024-1535
            next_texture_index: 2 * MAX_RESOURCES_PER_CATEGORY,
            // Storage images (Direct) at indices 192-255, bytes 1536-2047
            next_storage_image_index: 3 * MAX_RESOURCES_PER_CATEGORY,
            // Samplers at indices 256-319, bytes 2048-2559
            next_sampler_index: 4 * MAX_RESOURCES_PER_CATEGORY,
            free_storage_image_slots: Vec::new(),
            free_texture_slots: Vec::new(),
            free_storage_buffer_slots: Vec::new(),
            free_uniform_buffer_slots: Vec::new(),
            pending_free_storage_buffer_slots: Vec::new(),
            pending_free_uniform_buffer_slots: Vec::new(),
            pending_free_texture_slots: Vec::new(),
            pending_free_storage_image_slots: Vec::new(),
            buffer_indices: HashMap::new(),
            texture_indices: HashMap::new(),
            sampler_indices: HashMap::new(),
        }
    }

    /// Register a storage buffer (Scattered access) - indices 0-63.
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
    /// Failing fast surfaces the leak instead of producing corrupt frames.
    pub fn register_storage_buffer(&mut self, handle: BufferHandle) -> u32 {
        let local_index = if let Some(free) = self.free_storage_buffer_slots.pop() {
            free
        } else {
            let index = self.next_storage_buffer_index;
            assert!(
                index < MAX_RESOURCES_PER_CATEGORY,
                "storage-buffer bindless slots exhausted ({MAX_RESOURCES_PER_CATEGORY} max). \
                 next_index={} free={} pending_free={} live_indices={} \
                 (Scattered={}, Broadcast={}). \
                 Likely a per-frame leak in bind_map; check that all transient buffers \
                 (config_buf, scene_buf, indirect_buf, etc.) are explicitly freed via \
                 `recording.free_buffer(...)` and that `run_recording` evicts them at \
                 the end of the frame.",
                self.next_storage_buffer_index,
                self.free_storage_buffer_slots.len(),
                self.pending_free_storage_buffer_slots.len(),
                self.buffer_indices.len(),
                self.buffer_indices
                    .values()
                    .filter(|(_, a)| *a == DataAccess::Scattered)
                    .count(),
                self.buffer_indices
                    .values()
                    .filter(|(_, a)| *a == DataAccess::Broadcast)
                    .count(),
            );
            self.next_storage_buffer_index += 1;
            index
        };
        self.buffer_indices
            .insert(handle, (local_index, DataAccess::Scattered));
        local_index
    }

    /// Register a uniform buffer (Broadcast access) - local indices 0-63 (shader slot),
    /// global indices 64-127 (argument buffer encoding offset).
    /// Returns the LOCAL index so resource slots pass 0-63 to the shader
    /// (which indexes into uniformBuffers[0..63]).
    ///
    /// Reuses a freed slot if available (see `unregister_buffer`).
    ///
    /// # Panics on overflow
    ///
    /// See [`Self::register_storage_buffer`] for the analogous rationale —
    /// a uniform-slot overflow would alias into the texture-index region
    /// and cause silent shader-side garbage reads.
    pub fn register_uniform_buffer(&mut self, handle: BufferHandle) -> u32 {
        let local_index = if let Some(free) = self.free_uniform_buffer_slots.pop() {
            free
        } else {
            let global_index = self.next_uniform_buffer_index;
            let local = global_index - MAX_RESOURCES_PER_CATEGORY;
            assert!(
                local < MAX_RESOURCES_PER_CATEGORY,
                "uniform-buffer bindless slots exhausted ({MAX_RESOURCES_PER_CATEGORY} max). \
                 Likely a per-frame leak in bind_map for Broadcast buffers."
            );
            self.next_uniform_buffer_index += 1;
            local
        };
        self.buffer_indices
            .insert(handle, (local_index, DataAccess::Broadcast));
        local_index
    }

    /// Returns the global argument buffer index for a uniform buffer
    /// (local + MAX_RESOURCES_PER_CATEGORY), needed for encoding offsets.
    pub fn uniform_global_index(local_index: u32) -> u32 {
        local_index + MAX_RESOURCES_PER_CATEGORY
    }

    /// Register a sampled texture (Interpolated / Texture2D) — returns the LOCAL index (0-63).
    /// Use `texture_global_index()` to get the argument buffer encoding offset.
    ///
    /// Reuses a freed slot if available (see `release_texture_slot`).
    pub fn register_texture(&mut self, handle: TextureHandle) -> u32 {
        let local_index = if let Some(free) = self.free_texture_slots.pop() {
            free
        } else {
            let global_index = self.next_texture_index;
            let local = global_index - 2 * MAX_RESOURCES_PER_CATEGORY;
            self.next_texture_index += 1;
            local
        };
        self.texture_indices.insert(handle, local_index);
        local_index
    }

    /// Return a sampled-texture LOCAL index to the free list so it can be
    /// reused by a subsequent `register_texture`.
    ///
    /// If `defer` is true, the slot is parked in the pending list until the
    /// next successful `drain_pending_slots()`. Callers should pass `true`
    /// whenever any GPU command buffer is still in-flight that might still
    /// reference this slot's descriptor.
    pub fn release_texture_slot(&mut self, local_index: u32, defer: bool) {
        if defer {
            self.pending_free_texture_slots.push(local_index);
        } else {
            self.free_texture_slots.push(local_index);
        }
    }

    /// Returns the global argument buffer index for a sampled texture.
    pub fn texture_global_index(local_index: u32) -> u32 {
        local_index + 2 * MAX_RESOURCES_PER_CATEGORY
    }

    /// Register a storage image (Direct / RWTexture2D) — returns the LOCAL index (0-63).
    /// Use `storage_image_global_index()` to get the argument buffer encoding offset.
    ///
    /// Reuses a freed slot if available (see `release_storage_image_slot`).
    pub fn register_storage_image(&mut self, handle: TextureHandle) -> u32 {
        let local_index = self.allocate_storage_image_local();
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
        self.allocate_storage_image_local()
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
    /// See [`Self::release_texture_slot`] for the `defer` semantics.
    pub fn release_storage_image_slot(&mut self, local_index: u32, defer: bool) {
        if defer {
            self.pending_free_storage_image_slots.push(local_index);
        } else {
            self.free_storage_image_slots.push(local_index);
        }
    }

    /// Pop a free slot if any, otherwise bump the monotonic counter.
    fn allocate_storage_image_local(&mut self) -> u32 {
        if let Some(local) = self.free_storage_image_slots.pop() {
            return local;
        }
        let global_index = self.next_storage_image_index;
        let local_index = global_index - 3 * MAX_RESOURCES_PER_CATEGORY;
        self.next_storage_image_index += 1;
        local_index
    }

    /// Returns the global argument buffer index for a storage image.
    pub fn storage_image_global_index(local_index: u32) -> u32 {
        local_index + 3 * MAX_RESOURCES_PER_CATEGORY
    }

    /// Register a sampler — returns the LOCAL index (0-63) for resource slots.
    /// Use `sampler_global_index()` to get the argument buffer encoding offset.
    pub fn register_sampler(&mut self, handle: SamplerHandle) -> u32 {
        let global_index = self.next_sampler_index;
        let local_index = global_index - 4 * MAX_RESOURCES_PER_CATEGORY;
        self.next_sampler_index += 1;
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
    /// reuse it. Without this, per-frame buffer churn exhausts the 64-slot
    /// argument-buffer window in ~9 seconds and the shader starts reading
    /// into corrupt / out-of-range descriptors.
    ///
    /// If `defer` is true, the slot is parked in the pending list rather
    /// than the free list. See the `pending_free_*` fields for rationale.
    pub fn unregister_buffer(&mut self, handle: BufferHandle, defer: bool) {
        if let Some((local_index, access)) = self.buffer_indices.remove(&handle) {
            match (access, defer) {
                (DataAccess::Scattered, false) => {
                    self.free_storage_buffer_slots.push(local_index);
                }
                (DataAccess::Scattered, true) => {
                    self.pending_free_storage_buffer_slots.push(local_index);
                }
                (DataAccess::Broadcast, false) => {
                    self.free_uniform_buffer_slots.push(local_index);
                }
                (DataAccess::Broadcast, true) => {
                    self.pending_free_uniform_buffer_slots.push(local_index);
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

    /// Promote every pending slot to its corresponding free list.
    ///
    /// Called by the Metal backend after a successful `wait_fence()` (or any
    /// other synchronization that establishes "all previously-submitted GPU
    /// work has completed"). At that point the argument buffer descriptors
    /// the pending slots used to hold are no longer reachable from any
    /// in-flight command buffer, so the slots are safe to overwrite on the
    /// next `register_*` call.
    pub fn drain_pending_slots(&mut self) {
        self.free_storage_buffer_slots
            .append(&mut self.pending_free_storage_buffer_slots);
        self.free_uniform_buffer_slots
            .append(&mut self.pending_free_uniform_buffer_slots);
        self.free_texture_slots
            .append(&mut self.pending_free_texture_slots);
        self.free_storage_image_slots
            .append(&mut self.pending_free_storage_image_slots);
    }

    /// Number of buffer slots currently waiting for GPU idle before reuse.
    /// Exposed for tests; not part of the public API.
    #[cfg(test)]
    pub fn pending_buffer_slot_count(&self) -> usize {
        self.pending_free_storage_buffer_slots.len() + self.pending_free_uniform_buffer_slots.len()
    }
}

/// GPU buffer state.
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub buffer: MTLBuffer,
    pub size: u64,
    /// Index in the global argument buffer (always present — heap required).
    pub arg_buffer_index: u32,
    pub flags: crate::types::BufferFlags,
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
    /// Per push-constant slot category expectations from shader analysis.
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
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
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
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
    /// LOCAL index (0..MAX_RESOURCES_PER_CATEGORY) in the texture category this
    /// texture was registered in (`storageImages[]` when `is_storage_image`,
    /// otherwise `textures[]`).
    pub arg_buffer_index: u32,
    /// Which bindless region the `arg_buffer_index` belongs to; needed at
    /// destroy time to release the slot back to the correct free list.
    pub is_storage_image: bool,
    /// When true, the slot is owned by a long-lived entity (e.g. a `Surface`
    /// that re-encodes its drawable each frame) and should NOT be released
    /// when this `TextureState` is dropped. The owner manages slot lifetime.
    pub slot_owned_externally: bool,
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
    /// The currently acquired CAMetalDrawable (set during acquire, cleared on present)
    pub current_drawable: Option<*mut std::ffi::c_void>,
    /// Texture handle for the current drawable's texture (registered for bindless access)
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
}

// SAFETY: `SurfaceState` contains raw pointers to a `CALayer` and `CAMetalDrawable`.
// These Metal objects are reference-counted by Objective-C and are themselves thread-safe
// for retain/release. All mutation (acquire, present, destroy) is serialised through the
// `MetalBackend` mutex (`backend.lock()` in every GpuBackend method), so no two threads
// can access these pointers concurrently. Callers must uphold this invariant — do not
// share `SurfaceState` directly across threads without the backend lock.
unsafe impl Send for SurfaceState {}
unsafe impl Sync for SurfaceState {}

/// Consolidated Metal backend state.
/// Holds all resources and state for the Metal backend.
pub(super) struct MetalState {
    /// Set once any GPU wait has timed out (the GPU has wedged in a compute
    /// shader without the driver watchdog noticing). Subsequent waits fail
    /// fast instead of burning the full timeout budget per frame, letting
    /// the app cascade errors quickly and exit cleanly rather than appearing
    /// frozen for tens of seconds while each frame times out.
    pub device_lost: std::sync::atomic::AtomicBool,
    pub devices: std::collections::HashMap<DeviceHandle, LogicalDevice>,
    pub next_device_handle: DeviceHandle,
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
    pub slang_compiler: crate::slang::SlangCompiler,
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
            reg.unregister_buffer(handle, false);
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
            reg.unregister_buffer(handle, false);
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
        reg.unregister_buffer(h_uni, false);
        reg.unregister_buffer(h_sto, false);

        assert_eq!(
            reg.free_uniform_buffer_slots.len(),
            1,
            "uniform free list should have reclaimed the uniform slot"
        );
        assert_eq!(
            reg.free_storage_buffer_slots.len(),
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

        reg.unregister_buffer(h0, false);
        reg.unregister_buffer(h1, false);

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
        reg.unregister_buffer(h0, true);
        assert_eq!(
            reg.pending_buffer_slot_count(),
            1,
            "expected slot to land in pending"
        );

        // Until drain, register must NOT re-hand out slot 0.
        let h1: BufferHandle = 2;
        let i1 = reg.register_storage_buffer(h1);
        assert_eq!(
            i1, 1,
            "register_storage_buffer must not recycle a still-pending slot"
        );

        // After drain, pending→free, and the next register picks it up.
        reg.drain_pending_slots();
        assert_eq!(reg.pending_buffer_slot_count(), 0);

        reg.unregister_buffer(h1, false);
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

        reg.unregister_buffer(h_sto, true);
        reg.unregister_buffer(h_uni, true);
        reg.drain_pending_slots();

        assert_eq!(reg.free_storage_buffer_slots.len(), 1);
        assert_eq!(reg.free_uniform_buffer_slots.len(), 1);
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
        reg.release_texture_slot(i0, true);

        let h1: TextureHandle = 101;
        let i1 = reg.register_texture(h1);
        assert_ne!(
            i1, i0,
            "texture slot must not be recycled while still pending"
        );

        reg.drain_pending_slots();
        reg.release_texture_slot(i1, false);
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
