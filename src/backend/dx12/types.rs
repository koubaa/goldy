//! DX12 backend internal types.
//!
//! This module contains all the state structs used by the DX12 backend.
//!
//! ## Bindless Architecture
//!
//! The DX12 backend uses shader-visible descriptor heaps for bindless resource access:
//! - A large CBV/SRV/UAV heap contains all buffer and texture descriptors
//! - A large sampler heap contains all sampler descriptors
//! - Resources are registered at creation time and assigned heap offsets
//! - Shaders access resources by indexing into the descriptor heaps

use super::super::{
    AccelerationStructureHandle, BufferHandle, ComputePipelineHandle, ContextHandle, DeviceHandle, PipelineHandle,
    RayTracingPipelineHandle, RenderTargetHandle, SamplerHandle, ShaderHandle, SurfaceHandle, TextureHandle,
};
use crate::timeline::SmallContextMap;
use crate::types::{DepthFormat, SamplerDesc, TextureFormat};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, RwLock,
};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::{Direct3D12, Dxgi};

/// Newtype around [`HANDLE`] that is `Send + Sync`.
///
/// Win32 kernel-object handles (e.g. the DXGI frame-latency waitable) are
/// valid to use from any thread; the raw-pointer representation is purely an
/// ABI artifact and is never dereferenced by user code.
#[derive(Clone, Copy)]
pub(crate) struct SendSyncHandle(pub HANDLE);
// Safety: Win32 kernel handles identify kernel objects by integer value and
// carry no thread-affinity; `WaitForSingleObject` / `CloseHandle` are safe to
// call from any thread on the same handle.
unsafe impl Send for SendSyncHandle {}
unsafe impl Sync for SendSyncHandle {}

/// Maximum number of descriptors in the CBV/SRV/UAV heap for bindless rendering
#[allow(dead_code)]
pub const MAX_BINDLESS_CBV_SRV_UAV: u32 = 16384;

/// Maximum number of descriptors in the sampler heap for bindless rendering
#[allow(dead_code)]
pub const MAX_BINDLESS_SAMPLERS: u32 = 2048;

// PushLayout and its constants live in the shared module so all three backends
// use one definition. Re-export them here so internal code keeps using the
// same unqualified names as before.
pub use super::super::shared::{PushLayout, TOTAL_PUSH_BYTES};

/// Registry for tracking bindless resource descriptor heap offsets.
///
/// IMPORTANT: All CBV, SRV, and UAV descriptors share the same heap (cbv_srv_uav_heap),
/// so we use a unified offset counter to avoid collisions.
///
/// Each `register_*` call pops a slot from the appropriate free list before minting a
/// new one, preventing monotonic counter exhaustion when transient resources (e.g.
/// per-frame pool views or swapchain-back-buffer UAVs) are created and destroyed every
/// frame. Without this recycling the `next_cbv_srv_uav_offset` would hit
/// `MAX_BINDLESS_CBV_SRV_UAV` (16 384) and subsequent descriptor writes would go
/// out-of-bounds, corrupting the heap and causing GPU hangs / device loss.
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    /// Unified CBV/SRV/UAV slot allocator. All buffer and texture descriptors
    /// share the same heap, so a single allocator prevents offset collisions.
    cbv_srv_uav: super::super::shared::SlotAllocator,
    /// Separate sampler heap allocator.
    sampler: super::super::shared::SlotAllocator,
    /// Maps buffer handle to its primary descriptor offset (UAV for storage, CBV for uniform)
    pub buffer_offsets: HashMap<BufferHandle, u32>,
    /// Maps buffer handle to its secondary SRV offset (for storage buffers that need read access)
    pub buffer_srv_offsets: HashMap<BufferHandle, u32>,
    pub texture_offsets: HashMap<TextureHandle, u32>,
    /// Maps texture handle to UAV offset (for storage textures / TextureKind::Direct)
    pub texture_uav_offsets: HashMap<TextureHandle, u32>,
    pub sampler_offsets: HashMap<SamplerHandle, u32>,
    pub accel_offsets: HashMap<AccelerationStructureHandle, u32>,
}

#[allow(dead_code)]
impl ResourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_buffer_cbv(&mut self, handle: BufferHandle) -> u32 {
        let offset = self.cbv_srv_uav.alloc();
        self.buffer_offsets.insert(handle, offset);
        offset
    }

    pub fn register_buffer_srv(&mut self, handle: BufferHandle) -> u32 {
        let offset = self.cbv_srv_uav.alloc();
        self.buffer_srv_offsets.insert(handle, offset);
        offset
    }

    pub fn register_buffer_uav(&mut self, handle: BufferHandle) -> u32 {
        let offset = self.cbv_srv_uav.alloc();
        self.buffer_offsets.insert(handle, offset);
        offset
    }

    pub fn register_texture(&mut self, handle: TextureHandle) -> u32 {
        let offset = self.cbv_srv_uav.alloc();
        self.texture_offsets.insert(handle, offset);
        offset
    }

    /// Register a UAV descriptor for a texture (e.g. storage image / TextureKind::Direct).
    pub fn register_texture_uav(&mut self, handle: TextureHandle) -> u32 {
        let offset = self.cbv_srv_uav.alloc();
        self.texture_uav_offsets.insert(handle, offset);
        offset
    }

    /// Register a secondary SRV descriptor for a texture (used by `DirectInterpolated`).
    /// Unlike `register_texture`, this one doesn't overwrite the primary SRV slot; the
    /// slot is returned directly and the caller stores it in `sampled_bindless_offset`.
    pub fn register_texture_srv(&mut self, _handle: TextureHandle) -> u32 {
        self.cbv_srv_uav.alloc()
    }

    pub fn register_sampler(&mut self, handle: SamplerHandle) -> u32 {
        let offset = self.sampler.alloc();
        self.sampler_offsets.insert(handle, offset);
        offset
    }

    pub fn register_accel(&mut self, handle: AccelerationStructureHandle) -> u32 {
        let offset = self.cbv_srv_uav.alloc();
        self.accel_offsets.insert(handle, offset);
        offset
    }

    pub fn extract_accel_slots(&mut self, handle: AccelerationStructureHandle) -> Vec<u32> {
        self.accel_offsets.remove(&handle).into_iter().collect()
    }

    /// Reserve the low bindless indices for runtime protocol (frame table).
    pub fn ensure_cbv_start(&mut self, start: u32) {
        self.cbv_srv_uav.ensure_minimum_next(start);
    }

    /// Get the SRV offset for a buffer (for read-only access to storage buffers)
    pub fn get_buffer_srv_offset(&self, handle: BufferHandle) -> Option<u32> {
        self.buffer_srv_offsets.get(&handle).copied()
    }

    /// Allocate a raw descriptor slot not tied to any resource handle.
    /// Used for permanent device-lifetime slots that need a reserved CBV/SRV/UAV index.
    pub fn alloc_cbv_srv_uav_slot(&mut self) -> u32 {
        self.cbv_srv_uav.alloc()
    }

    /// Remove a buffer's handle mappings and return its raw descriptor slot(s)
    /// without recycling them (caller must [`LogicalDevice::queue_slot_reclamation`]).
    pub fn extract_buffer_slots(&mut self, handle: BufferHandle) -> Vec<u32> {
        let mut slots = Vec::new();
        if let Some(offset) = self.buffer_offsets.remove(&handle) {
            slots.push(offset);
        }
        if let Some(offset) = self.buffer_srv_offsets.remove(&handle) {
            slots.push(offset);
        }
        slots
    }

    /// Remove a texture's handle mappings and return its raw descriptor slot(s)
    /// without recycling them (caller must [`LogicalDevice::queue_slot_reclamation`]).
    pub fn extract_texture_slots(&mut self, handle: TextureHandle) -> Vec<u32> {
        let mut slots = Vec::new();
        if let Some(offset) = self.texture_offsets.remove(&handle) {
            slots.push(offset);
        }
        if let Some(offset) = self.texture_uav_offsets.remove(&handle) {
            slots.push(offset);
        }
        slots
    }

    /// Return a CBV/SRV/UAV slot to the free list (immediate reclaim).
    pub fn free_cbv_srv_uav_slot(&mut self, slot: u32) {
        self.cbv_srv_uav.free(slot);
    }

    /// Remove a sampler's handle mapping and return its slot without recycling.
    pub fn extract_sampler_slots(&mut self, handle: SamplerHandle) -> Vec<DeferredSlot> {
        if let Some(offset) = self.sampler_offsets.remove(&handle) {
            vec![DeferredSlot::Sampler(offset)]
        } else {
            Vec::new()
        }
    }

    /// Return a sampler slot to the free list (immediate reclaim).
    pub fn free_sampler_slot(&mut self, slot: u32) {
        self.sampler.free(slot);
    }

    /// Free a [`DeferredSlot`] — dispatches to the correct heap allocator.
    pub fn free_deferred_slot(&mut self, slot: DeferredSlot) {
        match slot {
            DeferredSlot::CbvSrvUav(s) => self.cbv_srv_uav.free(s),
            DeferredSlot::Sampler(s) => self.sampler.free(s),
        }
    }

    pub fn unregister_sampler(&mut self, handle: SamplerHandle) {
        if let Some(offset) = self.sampler_offsets.remove(&handle) {
            self.sampler.free(offset);
        }
    }

    /// Number of available (allocatable) slots in the given category.
    ///
    /// DX12 uses a unified CBV/SRV/UAV heap for all non-sampler categories,
    /// so Scattered/Broadcast/Texture/StorageImage all report against the
    /// same pool. Sampler has its own heap.
    pub fn available_slots(&self, category: crate::types::ResourceCategory) -> u32 {
        match category {
            crate::types::ResourceCategory::Sampler => MAX_BINDLESS_SAMPLERS.saturating_sub(self.sampler.live_count()),
            _ => MAX_BINDLESS_CBV_SRV_UAV.saturating_sub(self.cbv_srv_uav.live_count()),
        }
    }

    /// Maximum slots for the given category.
    pub fn max_slots(category: crate::types::ResourceCategory) -> u32 {
        match category {
            crate::types::ResourceCategory::Sampler => MAX_BINDLESS_SAMPLERS,
            _ => MAX_BINDLESS_CBV_SRV_UAV,
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod registry_tests {
    use super::*;

    fn free_buffer_slots(reg: &mut ResourceRegistry, handle: BufferHandle) {
        for slot in reg.extract_buffer_slots(handle) {
            reg.free_cbv_srv_uav_slot(slot);
        }
    }

    fn free_texture_slots(reg: &mut ResourceRegistry, handle: TextureHandle) {
        for slot in reg.extract_texture_slots(handle) {
            reg.free_cbv_srv_uav_slot(slot);
        }
    }

    /// Simulate the per-frame create/destroy churn that transient pool consumers generate for transient
    /// pool-view buffers. The counter must stay bounded — well below MAX_BINDLESS_CBV_SRV_UAV
    /// — even after far more iterations than the heap limit.
    #[test]
    fn buffer_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as BufferHandle;
            reg.register_buffer_uav(handle);
            free_buffer_slots(&mut reg, handle);
        }
        assert_eq!(
            reg.cbv_srv_uav.next_fresh(),
            1,
            "UAV counter grew; slot recycling not working"
        );
        assert_eq!(reg.cbv_srv_uav.free_count(), 1);
    }

    /// Storage buffers register two slots (UAV primary + SRV secondary).
    /// Both must be returned to the free list on unregister.
    #[test]
    fn storage_buffer_dual_slot_recycled() {
        let mut reg = ResourceRegistry::new();
        for i in 0..1_000u64 {
            let handle = i as BufferHandle;
            reg.register_buffer_uav(handle);
            reg.register_buffer_srv(handle);
            free_buffer_slots(&mut reg, handle);
        }
        assert_eq!(
            reg.cbv_srv_uav.next_fresh(),
            2,
            "counter should only have advanced twice (one UAV + one SRV slot ever minted)"
        );
        assert_eq!(reg.cbv_srv_uav.free_count(), 2, "both slots must be in the free list");
    }

    /// Textures with both SRV and UAV views (storage textures) must recycle both slots.
    #[test]
    fn texture_dual_slot_recycled() {
        let mut reg = ResourceRegistry::new();
        for i in 0..1_000u64 {
            let handle = i as TextureHandle;
            reg.register_texture(handle);
            reg.register_texture_uav(handle);
            free_texture_slots(&mut reg, handle);
        }
        assert_eq!(reg.cbv_srv_uav.next_fresh(), 2);
        assert_eq!(reg.cbv_srv_uav.free_count(), 2);
    }

    /// Sampler slots are in a separate heap; verify they recycle independently.
    #[test]
    fn sampler_slots_recycled() {
        let mut reg = ResourceRegistry::new();
        for i in 0..5_000u64 {
            let handle = i as SamplerHandle;
            reg.register_sampler(handle);
            reg.unregister_sampler(handle);
        }
        assert_eq!(reg.sampler.next_fresh(), 1);
        assert_eq!(reg.sampler.free_count(), 1);
    }

    /// Simultaneously-live resources must receive distinct slots.
    #[test]
    fn live_resources_get_distinct_slots() {
        let mut reg = ResourceRegistry::new();
        const N: u64 = 64;
        let mut slots: Vec<u32> = (0..N).map(|i| reg.register_buffer_uav(i as BufferHandle)).collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), N as usize, "duplicate slots assigned to live resources");
    }

    /// Slots freed by destroyed resources must be reused before the counter advances,
    /// keeping the high-water mark at or below the number of concurrently-live resources.
    #[test]
    fn high_water_mark_bounded_by_live_count() {
        let mut reg = ResourceRegistry::new();
        const LIVE: u64 = 8;
        const ROUNDS: u64 = 10_000;

        for i in 0..LIVE {
            reg.register_buffer_uav(i as BufferHandle);
        }
        for i in LIVE..LIVE + ROUNDS {
            free_buffer_slots(&mut reg, (i - LIVE) as BufferHandle);
            reg.register_buffer_uav(i as BufferHandle);
        }
        assert!(
            reg.cbv_srv_uav.next_fresh() <= LIVE as u32,
            "counter ({}) exceeded live count ({LIVE}); slot recycling broken",
            reg.cbv_srv_uav.next_fresh()
        );
    }

    /// Per-slot retirement: slot stays off the free list until referencing contexts retire.
    #[test]
    fn slot_deferred_until_context_retires() {
        use crate::backend::ContextHandle;
        let mut reg = ResourceRegistry::new();
        let handle = 1u64 as BufferHandle;
        let slot = reg.register_buffer_uav(handle);
        const CTX_A: ContextHandle = 10;
        const SEQ: u64 = 5;

        let slots = reg.extract_buffer_slots(handle);
        assert_eq!(slots, vec![slot]);

        let mut pending = vec![PendingSlotReclamation {
            slot: DeferredSlot::CbvSrvUav(slot),
            requirements: vec![(CTX_A, SEQ)],
        }];

        assert_eq!(reg.cbv_srv_uav.free_count(), 0, "slot must not be freed yet");

        let mut retired = HashMap::from([(CTX_A, 4u64)]);
        let drain_pending = |retired: &HashMap<ContextHandle, u64>,
                             reg: &mut ResourceRegistry,
                             pending: &mut Vec<PendingSlotReclamation>| {
            let mut i = 0;
            while i < pending.len() {
                let ready = pending[i]
                    .requirements
                    .iter()
                    .all(|(ctx, seq)| retired.get(ctx).copied().unwrap_or(0) >= *seq);
                if ready {
                    let entry = pending.swap_remove(i);
                    reg.free_deferred_slot(entry.slot);
                } else {
                    i += 1;
                }
            }
        };

        drain_pending(&retired, &mut reg, &mut pending);
        assert_eq!(reg.cbv_srv_uav.free_count(), 0, "still in flight at seq 4");

        retired.insert(CTX_A, SEQ);
        drain_pending(&retired, &mut reg, &mut pending);
        assert_eq!(reg.cbv_srv_uav.free_count(), 1, "slot freed after context retires");
    }

    /// Slot used by two contexts waits for both to retire.
    #[test]
    fn slot_waits_for_all_referencing_contexts() {
        use crate::backend::ContextHandle;
        let mut reg = ResourceRegistry::new();
        let handle = 2u64 as BufferHandle;
        let slot = reg.register_buffer_uav(handle);
        const CTX_A: ContextHandle = 1;
        const CTX_B: ContextHandle = 2;

        let mut pending = vec![PendingSlotReclamation {
            slot: DeferredSlot::CbvSrvUav(slot),
            requirements: vec![(CTX_A, 3), (CTX_B, 7)],
        }];
        reg.extract_buffer_slots(handle);

        let mut retired = HashMap::from([(CTX_A, 0u64), (CTX_B, 0u64)]);

        let drain_pending = |retired: &HashMap<ContextHandle, u64>,
                             reg: &mut ResourceRegistry,
                             pending: &mut Vec<PendingSlotReclamation>| {
            let mut i = 0;
            while i < pending.len() {
                let ready = pending[i]
                    .requirements
                    .iter()
                    .all(|(ctx, seq)| retired.get(ctx).copied().unwrap_or(0) >= *seq);
                if ready {
                    let entry = pending.swap_remove(i);
                    reg.free_deferred_slot(entry.slot);
                } else {
                    i += 1;
                }
            }
        };

        drain_pending(&retired, &mut reg, &mut pending);
        assert_eq!(reg.cbv_srv_uav.free_count(), 0);

        retired.insert(CTX_A, 3);
        drain_pending(&retired, &mut reg, &mut pending);
        assert_eq!(reg.cbv_srv_uav.free_count(), 0, "CTX_B still in flight");

        retired.insert(CTX_B, 7);
        drain_pending(&retired, &mut reg, &mut pending);
        assert_eq!(reg.cbv_srv_uav.free_count(), 1);
    }

    /// Retained-graph pin blocks slot free even when GPU requirements are met.
    #[test]
    fn retained_pin_blocks_slot_free_until_unpin() {
        let mut dr = DescriptorRegistry::new();
        let handle = 3u64 as BufferHandle;
        let slot = dr.resource_registry.register_buffer_uav(handle);
        let deferred = DeferredSlot::CbvSrvUav(slot);

        dr.pin_retained_slots([deferred]);
        dr.queue_slot_reclamation(deferred);
        dr.resource_registry.extract_buffer_slots(handle);

        let fences: HashMap<ContextHandle, ContextFenceEntry> = HashMap::new();
        dr.drain_ready_slot_reclamations(&fences);
        assert_eq!(
            dr.resource_registry.cbv_srv_uav.free_count(),
            0,
            "pinned slot must not free"
        );
        assert_eq!(dr.retained_user_count(deferred), 1);

        dr.unpin_retained_slots([deferred]);
        dr.drain_ready_slot_reclamations(&fences);
        assert_eq!(
            dr.resource_registry.cbv_srv_uav.free_count(),
            1,
            "unpin then drain frees"
        );
    }

    /// Two retained graphs sharing a slot require two unpins.
    #[test]
    fn retained_pin_shared_slot_needs_two_unpins() {
        let mut dr = DescriptorRegistry::new();
        let slot = dr.resource_registry.register_buffer_uav(4);
        let deferred = DeferredSlot::CbvSrvUav(slot);

        dr.pin_retained_slots([deferred]);
        dr.pin_retained_slots([deferred]);
        dr.queue_slot_reclamation(deferred);
        dr.resource_registry.buffer_offsets.remove(&4);

        let fences: HashMap<ContextHandle, ContextFenceEntry> = HashMap::new();
        dr.drain_ready_slot_reclamations(&fences);
        assert_eq!(dr.resource_registry.cbv_srv_uav.free_count(), 0);

        dr.unpin_retained_slots([deferred]);
        dr.drain_ready_slot_reclamations(&fences);
        assert_eq!(dr.resource_registry.cbv_srv_uav.free_count(), 0, "one pin remains");

        dr.unpin_retained_slots([deferred]);
        dr.drain_ready_slot_reclamations(&fences);
        assert_eq!(dr.resource_registry.cbv_srv_uav.free_count(), 1);
    }

    /// Replacing a retained graph: unpin old-only slots, new slots stay pinned.
    #[test]
    fn retained_pin_replace_frees_old_only_slots() {
        let mut dr = DescriptorRegistry::new();
        let slot_old = dr.resource_registry.register_buffer_uav(5);
        let slot_new = dr.resource_registry.register_buffer_uav(6);
        let old_deferred = DeferredSlot::CbvSrvUav(slot_old);
        let new_deferred = DeferredSlot::CbvSrvUav(slot_new);

        dr.pin_retained_slots([old_deferred]);
        dr.pin_retained_slots([new_deferred]);
        dr.unpin_retained_slots([old_deferred]);

        dr.queue_slot_reclamation(old_deferred);
        dr.queue_slot_reclamation(new_deferred);
        dr.resource_registry.buffer_offsets.remove(&5);
        dr.resource_registry.buffer_offsets.remove(&6);

        let fences: HashMap<ContextHandle, ContextFenceEntry> = HashMap::new();
        dr.drain_ready_slot_reclamations(&fences);
        assert_eq!(dr.resource_registry.cbv_srv_uav.free_count(), 1, "old slot freed");
        assert_eq!(dr.retained_user_count(new_deferred), 1);
        assert_eq!(dr.retained_user_count(old_deferred), 0);
    }

    /// Retained-graph pin blocks physical GPU release readiness (same gate as slot reclaim).
    #[test]
    fn retained_pin_blocks_gpu_release_readiness() {
        let mut dr = DescriptorRegistry::new();
        let slot = dr.resource_registry.register_buffer_uav(7);
        let deferred = DeferredSlot::CbvSrvUav(slot);

        dr.pin_retained_slots([deferred]);
        assert!(!dr.retained_pins_clear(&[deferred]));

        dr.unpin_retained_slots([deferred]);
        assert!(dr.retained_pins_clear(&[deferred]));
    }
}

/// Information about a physical DXGI adapter.
/// Named DxgiAdapterInfo to avoid conflict with super::AdapterInfo.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct DxgiAdapterInfo {
    pub adapter: Dxgi::IDXGIAdapter1,
    pub desc: Dxgi::DXGI_ADAPTER_DESC1,
    pub adapter_id: u32,
    /// From `D3D12_FEATURE_DATA_D3D12_OPTIONS::TiledResourcesTier` at enumeration.
    pub supports_reserved_buffers: bool,
    /// DXR tier 1.1 (`D3D12_RAYTRACING_TIER_1_1`) — inline ray query.
    pub ray_query: bool,
    /// DXR tier 1.0+ — ray-tracing pipelines.
    pub ray_tracing_pipelines: bool,
    /// `D3D12_MESH_SHADER_TIER_1`.
    pub mesh_shaders: bool,
    /// Amplification shaders ship with mesh-shader tier 1.
    pub amplification_shaders: bool,
}

/// A slot in the compute command allocator pool.
/// An allocator can only be reset after its associated GPU work has completed.
#[allow(dead_code)]
pub(crate) struct ComputeAllocatorSlot {
    pub allocator: Direct3D12::ID3D12CommandAllocator,
    /// Fence value when this slot was last used (for reuse detection)
    pub fence_value: u64,
    /// Reusable command list (created on first use, then reset with allocator).
    pub command_list: Option<Direct3D12::ID3D12GraphicsCommandList>,
    /// When `true`, this slot holds a retained command list that must not be reset
    /// until the caller explicitly releases it via `evict_retained`.
    pub retained: bool,
    /// Allocator and command list were reset on the submission worker and are ready to record.
    pub pre_reset: bool,
    /// Render thread holds this slot open for recording until the matching submit executes.
    pub in_recording: bool,
}

/// Per-context async submission stream (fence, poller, compute allocator pool).
pub(crate) struct Dx12SubmissionContext {
    pub device: super::DeviceHandle,
    pub fence: Direct3D12::ID3D12Fence,
    /// This context's own COMPUTE command queue.
    ///
    /// Each context gets its own `ID3D12CommandQueue` (`COMPUTE`) so that an operation on one
    /// context can never block an operation on another: a GPU-side cross-context `Wait`
    /// enqueued on this queue only stalls *this* queue, never a sibling context's queue.
    /// Graphics and present use [`LogicalDevice::command_queue`] (DIRECT).
    pub command_queue: Direct3D12::ID3D12CommandQueue,
    /// Serializes `Wait`/`ExecuteCommandLists`/`Signal` on [`Self::command_queue`] — the queue
    /// is externally synchronized in D3D12 and this context's queue may still be submitted to
    /// from multiple threads concurrently (e.g. several worker threads resubmitting different
    /// retained graphs on the same context). Scoped per-context (unlike the old device-global
    /// `LogicalDevice::queue_lock`) so contention here never involves other contexts.
    pub queue_lock: Arc<Mutex<()>>,
    /// Last device-global seq value submitted on this context.
    pub last_submitted_seq: Arc<AtomicU64>,
    pub signal_queue: std::sync::Arc<crate::signal::SignalQueue>,
    pub fence_shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub fence_thread: Option<std::thread::JoinHandle<()>>,
    /// Pool of command allocators for non-blocking compute submission on this context.
    pub compute_allocator_pool: Vec<ComputeAllocatorSlot>,
    /// Round-robin start index for compute allocator pool reuse.
    pub allocator_recycle_hint: usize,
    /// Retained command lists keyed by scheme fingerprint for zero-recording-cost re-submission.
    pub retained_graphs: HashMap<u64, RetainedGraph>,
    /// Upload belt for `GpuCommand::WriteBuffer` on this context.
    pub staging_belt: super::staging::StagingBelt,
    /// Pool that recycles texture-upload staging buffers across frames on this context.
    pub texture_staging_pool: super::staging::TextureStagingPool,
    /// Per-context deferred deletion queue — resources whose lifetime is bounded by this
    /// context's timeline (dispatch-batch arg buffers). Drained by this context's own
    /// completed fence value.
    pub deletion_queue: DeletionQueue,
    /// Thread-scoped reclamation epoch from [`ContextReclamationScope::set_epoch`].
    pub reclamation_context: Option<(std::thread::ThreadId, u64)>,
    /// Per-context frame-table GPU resources and ring state (bindless slots 0/1).
    pub frame_table: SharedContextFrameTable,
    /// GPU profile readbacks deferred until the context timeline retires each submit TV.
    pub pending_gpu_profiles: Vec<(u64, super::compute::Dx12GpuProfileResources)>,
}

/// A retained (closed but not reset) command list available for re-execution.
///
/// DX12 allows re-executing a closed command list via `ExecuteCommandLists`
/// without calling `Reset`, as long as the backing allocator is not reset first.
/// This enables zero-recording-cost re-submission for static scenes — the DX12
/// realization of a clean [`crate::Scheme`] resubmit.
///
/// # Not DX12 Work Graphs
///
/// Despite the name, this is **not** a D3D12 Work Graph (`DispatchGraph` /
/// `ID3D12WorkGraph*`). Correspondence for readers of both models:
///
/// | Mechanism | Role here |
/// |---|---|
/// | **Retained CL** (this type) | Host-recorded DAG replay — closest to CUDA Graph capture/launch. Cuts CPU record cost when the scheme is clean. |
/// | **`ExecuteIndirect`** | GPU fills dispatch (or batched-direct) args under a fixed PSO — used for `DispatchDim::Indirect` / same-pipeline batches. Does **not** choose child nodes or switch shaders. |
/// | **DX12 Work Graphs** | GPU emits records to child nodes; hardware schedules producer→consumer fan-out across PSOs. Unused by Goldy today. |
///
/// A scheme (dispatches + precedences) could in principle lower onto Work
/// Graphs as a future backend transformation; the current path is command-list
/// retention + barriers (+ optional `ExecuteIndirect` for grid size only).
pub(crate) struct RetainedGraph {
    /// The closed (but not reset) command list.  Cloned from the pool slot so both
    /// the pool and this struct hold a reference-counted pointer to the same COM object.
    pub command_list: Direct3D12::ID3D12GraphicsCommandList,
    /// Index into the backing allocator pool — [`Dx12SubmissionContext::compute_allocator_pool`]
    /// when [`Self::on_device_queue`] is false, else [`LogicalDevice::device_direct_pool`].
    pub slot_idx: usize,
    /// When true, the list was recorded for the device DIRECT queue and must be
    /// re-executed there (not on the context COMPUTE queue).
    pub on_device_queue: bool,
    /// Bindless heap indices baked into this command list (for slot retirement on resubmit).
    pub used_slots: Arc<[DeferredSlot]>,
    /// Snapshot of staging at record time (prologue copy offsets are baked into the CB).
    pub frame_table_staging: Option<std::sync::Arc<[u32]>>,
    /// Row index baked into this CB's prologue copies; pinned until evict.
    pub frame_table_row: Option<u32>,
}

/// Resource pending deferred deletion.
/// Kept alive until the GPU frame that was in-flight at queue time completes.
#[allow(dead_code)]
pub(crate) enum PendingDeletion {
    Buffer {
        buffer_handle: BufferHandle,
        resource: Direct3D12::ID3D12Resource,
        upload_buffer: Option<Direct3D12::ID3D12Resource>,
        coherent_readback: Option<Direct3D12::ID3D12Resource>,
        /// Reserved-buffer tile map when [`super::buffer::BufferState::is_reserved`].
        reserved_tiles: Option<Vec<Option<(Direct3D12::ID3D12Heap, u64)>>>,
    },
    /// Old GPU allocations after an in-place buffer resize; logical handle and heap slots stay live.
    ReplacedBufferGpu {
        resource: Direct3D12::ID3D12Resource,
        upload_buffer: Option<Direct3D12::ID3D12Resource>,
        coherent_readback: Option<Direct3D12::ID3D12Resource>,
    },
    /// Previous reserved buffer after migration to committed storage — unmap tiles then drop.
    ReplacedReservedBufferGpu {
        resource: Direct3D12::ID3D12Resource,
        tiles: Vec<Option<(Direct3D12::ID3D12Heap, u64)>>,
        upload_buffer: Option<Direct3D12::ID3D12Resource>,
        coherent_readback: Option<Direct3D12::ID3D12Resource>,
    },
    /// A buffer view whose D3D12 resource belongs to the parent; only the
    /// bindless descriptor slots need deregistration.
    BufferView { buffer_handle: BufferHandle },
    Texture {
        texture_handle: TextureHandle,
        resource: Direct3D12::ID3D12Resource,
    },
    /// An ad-hoc GPU resource (e.g. a DispatchBatch argument buffer) that is not
    /// tracked in any resource map.  Dropping it releases the COM reference and
    /// frees the GPU memory once the fence is met.
    StandaloneResource(Direct3D12::ID3D12Resource),
}

/// Identifies which descriptor heap owns a deferred slot reclamation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DeferredSlot {
    /// Slot in the shared CBV/SRV/UAV heap (buffers and textures).
    CbvSrvUav(u32),
    /// Slot in the separate sampler heap.
    Sampler(u32),
}

/// One deferred descriptor-slot reclamation.
/// Ready when every `(context, required_seq)` pair has retired.
pub(crate) struct PendingSlotReclamation {
    pub slot: DeferredSlot,
    /// `(context_handle, min_seq_that_must_retire)`
    pub requirements: Vec<(super::ContextHandle, u64)>,
}

/// GPU buffer resources held until bindless slot requirements retire.
pub(crate) struct PendingBufferGpuRelease {
    pub requirements: Vec<(super::ContextHandle, u64)>,
    /// Bindless slots reclaimed from the destroyed buffer; physical free waits until
    /// retained-graph pins on these slots drop to zero.
    pub retained_slots: Vec<DeferredSlot>,
    pub resource: Direct3D12::ID3D12Resource,
    pub upload_buffer: Option<Direct3D12::ID3D12Resource>,
    pub coherent_readback: Option<Direct3D12::ID3D12Resource>,
    pub reserved_tiles: Option<Vec<Option<(Direct3D12::ID3D12Heap, u64)>>>,
}

/// Per-context fence + last-submitted seq shared with [`Dx12SubmissionContext`].
pub(crate) type ContextFenceEntry = (DeviceHandle, Direct3D12::ID3D12Fence, Arc<AtomicU64>);

pub(crate) type SharedContextFences = Arc<std::sync::RwLock<HashMap<super::ContextHandle, ContextFenceEntry>>>;

fn slot_requirements_met(
    requirements: &[(super::ContextHandle, u64)],
    context_fences: &HashMap<super::ContextHandle, ContextFenceEntry>,
) -> bool {
    // Missing context ⇒ already destroyed after its GPU drain wait, so requirements on it
    // are satisfied. Callers must keep the fence entry in the map until that wait + local
    // teardown finish (see `finish_destroy`); removing earlier lets concurrent drains free
    // slots still referenced by the dying context.
    requirements.iter().all(|(ctx_id, required_seq)| {
        context_fences
            .get(ctx_id)
            .is_none_or(|(_, fence, _)| unsafe { fence.GetCompletedValue() >= *required_seq })
    })
}

/// Device-global deletions with no recorded cross-context bindless requirements may still be
/// referenced by retained command lists on other live contexts. Only drain those once every
/// context fence is gone (teardown / last-context destroy).
fn device_deletion_requirements_met(
    requirements: &[(super::ContextHandle, u64)],
    context_fences: &HashMap<super::ContextHandle, ContextFenceEntry>,
) -> bool {
    if requirements.is_empty() {
        return context_fences.is_empty();
    }
    slot_requirements_met(requirements, context_fences)
}

/// Deferred deletion queue for a DX12 device.
pub(crate) struct DeletionQueue {
    inner: super::super::shared::DeferredQueue<u64, PendingDeletion>,
}

impl DeletionQueue {
    pub fn new() -> Self {
        Self {
            inner: super::super::shared::DeferredQueue::new(),
        }
    }

    pub fn queue(&mut self, fence_value: u64, resource: PendingDeletion) {
        self.inner.push(fence_value, resource);
    }

    pub(crate) fn drain_up_to_completed(&mut self, completed: u64) -> Vec<PendingDeletion> {
        self.inner.drain_up_to(completed)
    }

    pub(crate) fn drain_everything(&mut self) -> Vec<PendingDeletion> {
        self.inner.flush_all().collect()
    }

    pub(crate) fn pending_len(&self) -> usize {
        self.inner.len()
    }
}

/// Device-level deferred deletion queue for resources whose destroy could touch more than
/// one context (bindless-registry-tracked buffers/textures/views).
///
/// Per-context [`DeletionQueue`] entries are drained against that same context's own fence,
/// which stays correct even though contexts no longer share a value space. This queue's
/// entries have no single owning context (or may have been read by several), so they are
/// keyed by a `(ContextHandle, min_seq)` requirement snapshot — the same shape already used
/// for [`PendingSlotReclamation`]/[`PendingBufferGpuRelease`] — and only drained once every
/// listed context's own fence has retired its recorded value.
pub(crate) struct DeviceDeletionQueue {
    inner: super::super::shared::DeferredQueue<Vec<(super::ContextHandle, u64)>, PendingDeletion>,
}

impl DeviceDeletionQueue {
    pub fn new() -> Self {
        Self {
            inner: super::super::shared::DeferredQueue::new(),
        }
    }

    pub fn queue(&mut self, requirements: Vec<(super::ContextHandle, u64)>, resource: PendingDeletion) {
        self.inner.push(requirements, resource);
    }

    /// Drain ready entries, preserving each entry's requirement snapshot so GPU release
    /// can reuse it (instead of re-snapshotting `slot_last_seen`, which misses non-bindless
    /// uses such as `CopyBuffer`).
    pub(crate) fn drain_ready(
        &mut self,
        context_fences: &HashMap<super::ContextHandle, ContextFenceEntry>,
    ) -> Vec<(Vec<(super::ContextHandle, u64)>, PendingDeletion)> {
        self.inner
            .drain_where_with_keys(|reqs| device_deletion_requirements_met(reqs, context_fences))
    }

    pub(crate) fn drain_everything(&mut self) -> Vec<(Vec<(super::ContextHandle, u64)>, PendingDeletion)> {
        // Teardown path: return original requirement keys (unused once the device is idle).
        self.inner.drain_where_with_keys(|_| true)
    }

    pub(crate) fn pending_len(&self) -> usize {
        self.inner.len()
    }
}

/// Device-shared descriptor registry.
///
/// Contains the irreducible shared state for bindless slot allocation: the
/// `ResourceRegistry` (descriptor slot allocator), the per-context
/// `slot_last_seen` reference table, and the `pending_slot_reclamations`
/// list.  Wrapped in `Arc<Mutex<>>` on `LogicalDevice` so that future phases
/// can lock it independently of the global backend mutex.
pub(crate) struct DescriptorRegistry {
    /// Registry tracking resource offsets in descriptor heaps.
    pub resource_registry: ResourceRegistry,
    /// Maps bindless slot → per-context last-submitted seq that referenced it.
    /// Updated at every submit. Entry removed when the slot is queued for reclamation.
    pub slot_last_seen: FxHashMap<DeferredSlot, SmallContextMap<u64>>,
    /// Slots waiting for referencing contexts to retire before returning to the free list.
    pub pending_slot_reclamations: Vec<PendingSlotReclamation>,
    /// Retained command lists still baking each bindless slot (incremental refcount).
    retained_users: HashMap<DeferredSlot, u32>,
}

impl DescriptorRegistry {
    pub(crate) fn new() -> Self {
        Self {
            resource_registry: ResourceRegistry::new(),
            slot_last_seen: FxHashMap::default(),
            pending_slot_reclamations: Vec::new(),
            retained_users: HashMap::new(),
        }
    }

    /// Increment pin count for each slot baked into a newly retained command list.
    pub(crate) fn pin_retained_slots(&mut self, slots: impl IntoIterator<Item = DeferredSlot>) {
        for slot in slots {
            *self.retained_users.entry(slot).or_insert(0) += 1;
        }
    }

    /// Decrement pin count when a retained graph is evicted or a context is destroyed.
    pub(crate) fn unpin_retained_slots(&mut self, slots: impl IntoIterator<Item = DeferredSlot>) {
        for slot in slots {
            if let Some(count) = self.retained_users.get_mut(&slot) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.retained_users.remove(&slot);
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn retained_user_count(&self, slot: DeferredSlot) -> u32 {
        self.retained_users.get(&slot).copied().unwrap_or(0)
    }

    /// True when no retained command list still bakes any of `slots`.
    pub(crate) fn retained_pins_clear(&self, slots: &[DeferredSlot]) -> bool {
        slots
            .iter()
            .all(|slot| self.retained_users.get(slot).copied().unwrap_or(0) == 0)
    }

    /// Record that `ctx` submitted `seq` referencing each bindless slot in `slots`.
    pub(crate) fn record_slot_usage(
        &mut self,
        ctx: super::ContextHandle,
        seq: u64,
        slots: impl IntoIterator<Item = DeferredSlot>,
    ) {
        for slot in slots {
            self.slot_last_seen.entry(slot).or_default().mark_max(ctx, seq);
        }
    }

    /// Queue a slot for deferred reclamation once all referencing contexts retire.
    pub(crate) fn queue_slot_reclamation(&mut self, slot: DeferredSlot) {
        let requirements: Vec<_> = self
            .slot_last_seen
            .remove(&slot)
            .map(|m| m.iter().collect())
            .unwrap_or_default();
        // Always defer: empty requirements may still be referenced by retained command lists
        // on other live contexts (see `device_deletion_requirements_met`).
        self.pending_slot_reclamations
            .push(PendingSlotReclamation { slot, requirements });
    }

    /// Reclaim all descriptor slots for a destroyed buffer handle.
    ///
    /// Returns the deferred slots that were reclaimed (for gating physical GPU free).
    pub(crate) fn reclaim_buffer_slots(&mut self, handle: BufferHandle) -> Vec<DeferredSlot> {
        let slots = self.resource_registry.extract_buffer_slots(handle);
        let deferred: Vec<DeferredSlot> = slots.iter().map(|&s| DeferredSlot::CbvSrvUav(s)).collect();
        for slot in slots {
            self.queue_slot_reclamation(DeferredSlot::CbvSrvUav(slot));
        }
        deferred
    }

    /// Bindless slots currently assigned to `handle` (without reclaiming).
    pub(crate) fn buffer_slot_keys(&self, handle: BufferHandle) -> Vec<DeferredSlot> {
        let rr = &self.resource_registry;
        let mut slots = Vec::new();
        if let Some(&offset) = rr.buffer_offsets.get(&handle) {
            slots.push(DeferredSlot::CbvSrvUav(offset));
        }
        if let Some(&offset) = rr.buffer_srv_offsets.get(&handle) {
            slots.push(DeferredSlot::CbvSrvUav(offset));
        }
        slots
    }

    /// Bindless slots currently assigned to a texture handle (without reclaiming).
    pub(crate) fn texture_slot_keys(&self, handle: TextureHandle) -> Vec<DeferredSlot> {
        let rr = &self.resource_registry;
        let mut slots = Vec::new();
        if let Some(&offset) = rr.texture_offsets.get(&handle) {
            slots.push(DeferredSlot::CbvSrvUav(offset));
        }
        if let Some(&offset) = rr.texture_uav_offsets.get(&handle) {
            slots.push(DeferredSlot::CbvSrvUav(offset));
        }
        slots
    }

    /// Reclaim all descriptor slots for a destroyed texture handle.
    pub(crate) fn reclaim_texture_slots(&mut self, handle: TextureHandle) {
        let slots = self.resource_registry.extract_texture_slots(handle);
        for slot in slots {
            self.queue_slot_reclamation(DeferredSlot::CbvSrvUav(slot));
        }
    }

    /// Reclaim all descriptor slots for a destroyed sampler handle.
    pub(crate) fn reclaim_sampler_slots(&mut self, handle: SamplerHandle) {
        let slots = self.resource_registry.extract_sampler_slots(handle);
        for slot in slots {
            self.queue_slot_reclamation(slot);
        }
    }

    /// Return pending slots to the free list once every referencing context has retired.
    ///
    /// Takes the per-state `context_fences` index (not the full context map) so this
    /// method can be called while holding the descriptors lock without risking a lock-ordering
    /// deadlock with per-context `Mutex<Dx12SubmissionContext>`.
    pub(crate) fn drain_ready_slot_reclamations(&mut self, context_fences: &HashMap<ContextHandle, ContextFenceEntry>) {
        let mut i = 0;
        while i < self.pending_slot_reclamations.len() {
            let slot = self.pending_slot_reclamations[i].slot;
            let gpu_ready =
                device_deletion_requirements_met(&self.pending_slot_reclamations[i].requirements, context_fences);
            let pin_clear = self.retained_users.get(&slot).copied().unwrap_or(0) == 0;
            if gpu_ready && pin_clear {
                let entry = self.pending_slot_reclamations.swap_remove(i);
                self.resource_registry.free_deferred_slot(entry.slot);
            } else {
                i += 1;
            }
        }
    }

    /// Per-context requirements that must retire before a buffer's GPU resource can be
    /// released: `base` (the destroying context's own barrier, or a snapshot across every
    /// live context when the destroy could not be attributed to one) merged with every live
    /// `slot_last_seen` entry for this buffer's bindless slots, so cross-context shader
    /// access and non-bindless GPU uses (copies) cannot outlive the D3D12 resource.
    ///
    /// Returns per-context pairs rather than a single folded fence value: contexts do not
    /// share a value space, so maxing raw counter values from different contexts together
    /// would be meaningless.
    pub(crate) fn bindless_retirement_requirements_for_buffer(
        &self,
        handle: BufferHandle,
        base: Vec<(super::ContextHandle, u64)>,
    ) -> Vec<(super::ContextHandle, u64)> {
        let rr = &self.resource_registry;
        let mut slots = Vec::new();
        if let Some(&offset) = rr.buffer_offsets.get(&handle) {
            slots.push(offset);
        }
        if let Some(&offset) = rr.buffer_srv_offsets.get(&handle) {
            slots.push(offset);
        }
        self.merge_slot_requirements(&slots, base)
    }

    /// Same as [`Self::bindless_retirement_requirements_for_buffer`] but for a texture handle.
    pub(crate) fn bindless_retirement_requirements_for_texture(
        &self,
        handle: TextureHandle,
        base: Vec<(super::ContextHandle, u64)>,
    ) -> Vec<(super::ContextHandle, u64)> {
        let rr = &self.resource_registry;
        let mut slots = Vec::new();
        if let Some(&offset) = rr.texture_offsets.get(&handle) {
            slots.push(offset);
        }
        self.merge_slot_requirements(&slots, base)
    }

    fn merge_slot_requirements(
        &self,
        slots: &[u32],
        base: Vec<(super::ContextHandle, u64)>,
    ) -> Vec<(super::ContextHandle, u64)> {
        let mut merged: HashMap<super::ContextHandle, u64> = base.into_iter().collect();
        for &slot in slots {
            if let Some(map) = self.slot_last_seen.get(&DeferredSlot::CbvSrvUav(slot)) {
                for (ctx, seq) in map.iter() {
                    merged.entry(ctx).and_modify(|v| *v = (*v).max(seq)).or_insert(seq);
                }
            }
        }
        merged.into_iter().collect()
    }
}

/// Cached PSO blobs and disk-dirty flag for a logical device.
///
/// Wrapped in `Arc<RwLock<>>` on `LogicalDevice` so that PSO creation (which
/// is write-rare after warmup) does not need the global backend lock.
pub(crate) struct PsoCache {
    /// Cached graphics PSO blobs from `ID3D12PipelineState::GetCachedBlob`.
    pub graphics_blobs: HashMap<u64, Vec<u8>>,
    /// Cached compute PSO blobs.
    pub compute_blobs: HashMap<u64, Vec<u8>>,
    /// `true` if either blob map changed since loading from the disk file.
    pub dirty: bool,
}

impl PsoCache {
    pub(crate) fn new(graphics_blobs: HashMap<u64, Vec<u8>>, compute_blobs: HashMap<u64, Vec<u8>>) -> Self {
        Self {
            graphics_blobs,
            compute_blobs,
            dirty: false,
        }
    }
}

/// Reusable DIRECT command allocator/list for render partitions on the device queue.
pub(crate) struct DeviceDirectSlot {
    pub allocator: Direct3D12::ID3D12CommandAllocator,
    pub command_list: Direct3D12::ID3D12GraphicsCommandList,
    pub fence_value: u64,
    /// When `true`, this slot holds a retained command list that must not be reset
    /// until the caller explicitly releases it via `evict_retained`.
    pub retained: bool,
}

/// A logical D3D12 device with associated resources.
#[allow(dead_code)]
pub(crate) struct LogicalDevice {
    pub device: Direct3D12::ID3D12Device10,
    pub adapter_id: u32,
    pub command_queue: Direct3D12::ID3D12CommandQueue,
    /// Legacy single allocator for non-compute paths (e.g. render target). Compute uses the pool.
    pub command_allocator: Direct3D12::ID3D12CommandAllocator,
    pub rtv_heap: Direct3D12::ID3D12DescriptorHeap,
    pub rtv_descriptor_size: u32,
    pub dsv_heap: Direct3D12::ID3D12DescriptorHeap,
    pub dsv_descriptor_size: u32,
    pub cbv_srv_uav_heap: Direct3D12::ID3D12DescriptorHeap,
    pub cbv_srv_uav_descriptor_size: u32,
    pub sampler_heap: Direct3D12::ID3D12DescriptorHeap,
    pub sampler_descriptor_size: u32,
    /// Device fence for synchronous Signal+wait paths only (not per-submit timeline).
    pub fence: Direct3D12::ID3D12Fence,
    /// Device-global submission sequence (shared value space; contexts signal their own fences).
    pub timeline_next: Arc<AtomicU64>,
    /// Minimum completed horizon after a context is destroyed (never lowers `device_retired`).
    pub retired_floor: AtomicU64,

    // Bindless infrastructure
    /// `true` when adapter reports tiled resources tier >= 1 (buffer reserved resources).
    pub supports_reserved_buffers: bool,
    pub tile_heap_pool: Mutex<Option<super::tiles::TileHeapPool>>,
    /// Shared root signature for all bindless pipelines (graphics and compute)
    pub bindless_root_signature: Option<Direct3D12::ID3D12RootSignature>,
    /// Command signature for indirect compute dispatch (ExecuteIndirect, dispatch-only)
    pub compute_dispatch_indirect_signature: Option<Direct3D12::ID3D12CommandSignature>,
    /// Command signature for batched dispatch: sets push constants then dispatches.
    /// Each argument entry contains `[PushLayout (TOTAL_PUSH_BYTES)] [wg_x] [wg_y] [wg_z]`.
    /// Requires the shared `bindless_root_signature` to be non-None.
    pub compute_batch_dispatch_signature: Option<Direct3D12::ID3D12CommandSignature>,
    /// Device-lifetime zero-filled UPLOAD-heap buffer used as the source for
    /// `CopyBufferRegion` clears. One buffer per device; clears of any size are
    /// handled by chunking `CopyBufferRegion` calls. Using a copy instead of
    /// `ClearUnorderedAccessViewUint` avoids the shared-descriptor aliasing hazard
    /// that caused silent corruption on WARP when multiple buffers were cleared in
    /// the same wave (all clears rewrote the same single-slot descriptor heap).
    pub zero_buffer: Direct3D12::ID3D12Resource,
    /// Deferred deletion queue — resources are dropped only after every context listed in
    /// their requirement snapshot has retired. Used for buffer/texture destroys that could
    /// touch more than one context (see [`DeviceDeletionQueue`]).
    pub deletion_queue: Mutex<DeviceDeletionQueue>,
    /// Buffer GPU memory held until bindless slot `slot_last_seen` requirements retire.
    pub pending_buffer_gpu_releases: Mutex<Vec<PendingBufferGpuRelease>>,
    /// Shared with [`Dx12State::device_removed`] — first `u64::MAX` fence triggers DRED once.
    pub device_removed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Descriptor registry: `ResourceRegistry` + slot reference-tracking.
    /// `Arc` so Phase 5 can clone it out of `LogicalDevice` before dropping the
    /// global backend lock.
    pub descriptors: Arc<Mutex<DescriptorRegistry>>,
    /// Cached PSO blobs (graphics + compute) and disk-dirty flag.
    /// `RwLock` because reads dominate after warmup; `Arc` for Phase 5 cloning.
    pub pso_cache: Arc<RwLock<PsoCache>>,
    /// Serialises all `ExecuteCommandLists` + `Signal` pairs on this device's queue.
    ///
    /// D3D12 marks the command queue as externally synchronized for concurrent submits.
    /// Phase 5 lock-free submit clones this `Arc` and holds it only across the GPU
    /// enqueue, matching Vulkan's `queue_lock` and Metal's present/compute pairing.
    pub queue_lock: Arc<Mutex<()>>,
    /// Last device-queue submission seq (signals [`Self::fence`]).
    pub device_last_submitted_seq: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Pool for render-partition recording on the device DIRECT queue (compute style).
    /// Retained render lists pin a slot until `evict_retained` (same contract as the
    /// per-context compute allocator pool).
    pub device_direct_pool: std::sync::Mutex<Vec<DeviceDirectSlot>>,
    /// Frame table for legacy `render_to_target` (no submission context).
    pub legacy_frame_table: Mutex<Option<SharedContextFrameTable>>,
    /// Async FIFO worker for `ExecuteCommandLists` + `Signal` (render thread enqueues, worker runs).
    pub submission_worker: std::sync::Arc<super::super::submission_worker::SubmissionWorker>,
}

/// Shared logical device handle — cloned out of `Dx12State` before dropping the global lock.
pub(crate) type SharedLogicalDevice = Arc<LogicalDevice>;

/// Shared submission context handle — allows cloning a context reference out of `Dx12State`
/// before dropping the global backend lock, enabling fine-grained per-context locking.
pub(crate) type SharedSubmissionContext = Arc<Mutex<Dx12SubmissionContext>>;

/// Live contexts keyed by handle — `Arc<RwLock<>>` so submit sessions can scan/evict retained
/// graphs without the global backend mutex.
pub(crate) type SharedContextMap = Arc<std::sync::RwLock<HashMap<super::ContextHandle, SharedSubmissionContext>>>;

/// Per-context frame-table GPU state — cloned into submit sessions at context creation.
pub(crate) type SharedContextFrameTable = Arc<super::frame_table::ContextFrameTable>;

impl LogicalDevice {
    pub(crate) fn process_deletion_queue_up_to(&self, context_fences: &SharedContextFences) {
        let batch = {
            let fences = context_fences.read().unwrap();
            self.deletion_queue.lock().unwrap().drain_ready(&fences)
        };

        if !batch.is_empty() {
            let descriptors_arc = Arc::clone(&self.descriptors);
            let mut registry = descriptors_arc.lock().unwrap();
            for (requirements, resource) in batch {
                destroy_pending_deletion(self, &mut registry, resource, requirements);
            }
        }

        // Take ready GPU releases under the fences + descriptors locks, then drop COM refs
        // without holding them — reserved-tile teardown may Signal+wait and must not stall
        // other contexts that need `context_fences`.
        let ready = {
            let fences = context_fences.read().unwrap();
            let registry = self.descriptors.lock().unwrap();
            self.take_ready_buffer_gpu_releases(&fences, &registry)
        };
        for entry in ready {
            release_buffer_gpu_resources(self, entry);
        }
    }

    /// Collect deferred buffer GPU releases whose requirements have retired.
    /// Caller must release the `context_fences` read lock before calling
    /// [`release_buffer_gpu_resources`] (reserved-tile path may block).
    pub(crate) fn take_ready_buffer_gpu_releases(
        &self,
        context_fences: &HashMap<super::ContextHandle, ContextFenceEntry>,
        registry: &DescriptorRegistry,
    ) -> Vec<PendingBufferGpuRelease> {
        let mut pending = self.pending_buffer_gpu_releases.lock().unwrap();
        let mut ready = Vec::new();
        let mut i = 0;
        while i < pending.len() {
            let gpu_ready = device_deletion_requirements_met(&pending[i].requirements, context_fences);
            let pin_clear = registry.retained_pins_clear(&pending[i].retained_slots);
            if gpu_ready && pin_clear {
                ready.push(pending.swap_remove(i));
            } else {
                i += 1;
            }
        }
        ready
    }

    /// Drop deferred buffer GPU memory once every referencing context has retired.
    pub(crate) fn drain_pending_buffer_gpu_releases(
        &self,
        context_fences: &HashMap<super::ContextHandle, ContextFenceEntry>,
    ) {
        let ready = {
            let registry = self.descriptors.lock().unwrap();
            self.take_ready_buffer_gpu_releases(context_fences, &registry)
        };
        for entry in ready {
            release_buffer_gpu_resources(self, entry);
        }
    }

    pub(crate) fn flush_deletion_queue(&self, context_fences: &HashMap<super::ContextHandle, ContextFenceEntry>) {
        // Called only at device teardown, after wait_for_gpu ensures all GPU work has
        // completed. Slots queued here via reclaim_*_slots will have empty requirements
        // (slot_last_seen was cleared as contexts were destroyed before the device) and
        // are freed immediately in queue_slot_reclamation. Any slots that somehow still
        // have requirements are just dropped with the LogicalDevice — the whole allocator
        // is discarded at this point, so skipping drain_ready_slot_reclamations is safe.
        let batch = self.deletion_queue.lock().unwrap().drain_everything();
        if !batch.is_empty() {
            let descriptors_arc = Arc::clone(&self.descriptors);
            let mut registry = descriptors_arc.lock().unwrap();
            for (requirements, resource) in batch {
                destroy_pending_deletion(self, &mut registry, resource, requirements);
            }
        }
        self.drain_pending_buffer_gpu_releases(context_fences);
    }
}

/// Release GPU buffer memory after its retirement requirements have already been observed
/// as met. Ordinary buffers drop immediately (no new Signal). Reserved/tiled buffers still
/// need a queue flush after unmap before `Release`, or the driver can remove the device.
fn release_buffer_gpu_resources(ld: &LogicalDevice, entry: PendingBufferGpuRelease) {
    let PendingBufferGpuRelease {
        requirements: _,
        retained_slots: _,
        resource,
        upload_buffer,
        coherent_readback,
        reserved_tiles,
        ..
    } = entry;
    if let Some(tiles) = reserved_tiles {
        {
            let mut pool = ld.tile_heap_pool.lock().unwrap();
            super::tiles::teardown_reserved_mappings(&ld.command_queue, &mut pool, &resource, &tiles);
        }
        // Flush the queue so the unmap is processed before releasing the resource.
        let fv = ld.timeline_next.fetch_add(1, Ordering::Relaxed);
        let signaled = super::utils::with_queue_lock(ld, || {
            unsafe { ld.command_queue.Signal(&ld.fence, fv) }
                .map_err(|e| anyhow::anyhow!("Failed to signal device fence for reserved buffer deletion: {e:?}"))
        });
        if signaled.is_ok() {
            let _ = super::utils::wait_for_fence_on_device(&ld.fence, fv, Some(ld));
        }
    }
    drop(resource);
    drop(upload_buffer);
    drop(coherent_readback);
}

/// Process a drained pending deletion.
///
/// `queue_requirements` is the requirement set that gated the deletion-queue entry
/// (device-global: destroy/resize attribution + `slot_last_seen`; per-context: empty —
/// already gated on that context's fence). For [`PendingDeletion::Buffer`], this set is
/// propagated into [`PendingBufferGpuRelease`] so GPU free does not re-snapshot
/// `slot_last_seen` alone (which misses non-bindless uses such as `CopyBuffer`).
pub(crate) fn destroy_pending_deletion(
    ld: &LogicalDevice,
    registry: &mut DescriptorRegistry,
    resource: PendingDeletion,
    queue_requirements: Vec<(super::ContextHandle, u64)>,
) {
    match resource {
        PendingDeletion::Buffer {
            buffer_handle,
            resource,
            upload_buffer,
            coherent_readback,
            reserved_tiles,
        } => {
            // Reclaim descriptor slots now (handle is already gone from the API). Keep the
            // D3D12 resource alive until `queue_requirements` retire — do not replace that
            // set with a fresh `slot_last_seen` snapshot.
            let retained_slots = registry.reclaim_buffer_slots(buffer_handle);
            let release = PendingBufferGpuRelease {
                requirements: queue_requirements,
                retained_slots,
                resource,
                upload_buffer,
                coherent_readback,
                reserved_tiles,
            };
            ld.pending_buffer_gpu_releases.lock().unwrap().push(release);
        }
        PendingDeletion::BufferView { buffer_handle } => {
            registry.reclaim_buffer_slots(buffer_handle);
        }
        PendingDeletion::ReplacedBufferGpu {
            resource,
            upload_buffer,
            coherent_readback,
        } => {
            // Requirements already met by the deletion-queue drain; drop without a new Signal.
            let _ = queue_requirements;
            drop(resource);
            drop(upload_buffer);
            drop(coherent_readback);
        }
        PendingDeletion::ReplacedReservedBufferGpu {
            resource,
            tiles,
            upload_buffer,
            coherent_readback,
        } => {
            let _ = queue_requirements;
            {
                let mut pool = ld.tile_heap_pool.lock().unwrap();
                super::tiles::teardown_reserved_mappings(&ld.command_queue, &mut pool, &resource, &tiles);
            }
            // Flush the queue so the unmap is processed before releasing the resource.
            // Without this, the driver can crash (device removal) on Release.
            let fv = ld.timeline_next.fetch_add(1, Ordering::Relaxed);
            let signaled = super::utils::with_queue_lock(ld, || {
                unsafe { ld.command_queue.Signal(&ld.fence, fv) }
                    .map_err(|e| anyhow::anyhow!("Failed to signal device fence for buffer deletion: {e:?}"))
            });
            if signaled.is_ok() {
                let _ = super::utils::wait_for_fence(&ld.fence, fv);
            }
            drop(resource);
            drop(upload_buffer);
            drop(coherent_readback);
        }
        PendingDeletion::Texture {
            texture_handle,
            resource,
        } => {
            let _ = queue_requirements;
            registry.reclaim_texture_slots(texture_handle);
            drop(resource);
        }
        PendingDeletion::StandaloneResource(resource) => {
            let _ = queue_requirements;
            drop(resource);
        }
    }
}

/// GPU buffer state.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub resource: Direct3D12::ID3D12Resource,
    /// Logical byte size (descriptor element counts, bounds).
    pub size: u64,
    /// Committed resource width in bytes (`>= size`).
    pub allocation_size: u64,
    /// Primary descriptor heap offset for bindless access (UAV for storage, CBV for uniform)
    pub bindless_offset: Option<u32>,
    /// Secondary SRV descriptor offset for storage buffers (for read-only graphics access)
    pub bindless_srv_offset: Option<u32>,
    /// Whether this is a storage buffer (uses UAV instead of CBV/SRV)
    pub is_storage: bool,
    /// Upload buffer for DEFAULT heap resources (lazy-created on first CPU write)
    pub upload_buffer: Option<Direct3D12::ID3D12Resource>,
    /// StructuredBuffer element stride (for UAV clear rect calculations)
    pub element_stride: Option<u32>,
    /// If true, this is a view into another buffer — don't free the resource on destroy.
    pub is_view: bool,
    /// Direct3D 12: paired READBACK resource for [`crate::types::BufferFlags::CPU_READABLE`]
    /// storage buffers.
    pub coherent_readback: Option<Direct3D12::ID3D12Resource>,
    /// Persistent map of `coherent_readback` (see above).
    /// Persistent `Map` result address for the readback resource (`usize` for `Send`/`Sync`).
    pub coherent_readback_mapped: Option<usize>,
    /// Persistent map of the paired UPLOAD heap for [`crate::types::BufferFlags::CPU_WRITABLE`].
    pub cpu_writable_upload_mapped: Option<usize>,
    /// Creation-time flags.
    pub flags: crate::types::BufferFlags,
    /// Created with [`crate::backend::GpuBackend::place_buffer_in_transient_heap`] (placed resource).
    pub transient_placed: bool,
    /// Parent buffer handle when [`Self::is_view`]; [`None`] for root buffers.
    pub parent_for_view: Option<BufferHandle>,
    /// Byte offset into the parent for views; [`None`] for root buffers.
    pub view_byte_offset: Option<u64>,
    /// Built with [`ID3D12Device::CreateReservedResource`] + tile mappings.
    pub is_reserved: bool,
    pub tile_byte_size: u32,
    pub reserved_tiles: Vec<Option<(Direct3D12::ID3D12Heap, u64)>>,
    /// Grant-read staging buffer (READBACK heap, persistently mapped; no bindless slot).
    pub is_withdraw_staging: bool,
    pub texture_copy_footprint: Option<crate::backend::TextureCopyFootprint>,
}

/// Shader module state with cached compiled bytecode.
pub(crate) struct ShaderState {
    pub device_handle: DeviceHandle,
    pub slang_source: String,
    /// Search paths for Slang module resolution
    pub search_paths: Vec<String>,
    /// Extra preprocessor defines (e.g. msaa, msaa8)
    pub defines: Vec<(String, String)>,
    /// Per-shader Slang optimization level
    pub optimization_level: crate::types::OptimizationLevel,
    /// Cached compiled vertex shader bytecode
    pub vertex_bytecode: Option<Vec<u8>>,
    /// Cached compiled fragment shader bytecode
    pub fragment_bytecode: Option<Vec<u8>>,
    /// Cached compiled compute shader bytecode
    pub compute_bytecode: Option<Vec<u8>>,
    /// Cached bytecode for ray-tracing / mesh / amplification stages.
    pub extra_bytecode: HashMap<crate::slang::SlangStage, Vec<u8>>,
    /// Reflection data for bindless rendering (ParameterBlock layouts)
    pub reflection: Option<crate::slang::ShaderReflection>,
    /// Pending struct layout validation on first stage compile; cleared after success.
    pub layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
}

/// Graphics pipeline state.
#[allow(dead_code)]
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline_state: Direct3D12::ID3D12PipelineState,
    pub root_signature: Direct3D12::ID3D12RootSignature,
    /// Vertex buffer stride from vertex layout
    pub vertex_stride: u32,
    /// Primitive topology for IASetPrimitiveTopology
    pub topology: crate::types::PrimitiveTopology,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
    /// Per push-constant slot category expectations from shader analysis.
    pub push_constant_categories: Vec<Option<crate::types::ResourceCategory>>,
    /// Per push-constant slot SRV/UAV expectations from shader analysis (DX12).
    pub push_constant_slot_kinds: Vec<Option<crate::types::BindlessSlotKind>>,
    /// Per push-constant slot expected element stride (bytes) from reflection.
    pub binding_element_strides: Vec<Option<u32>>,
    /// Human-readable identifier used in category-mismatch error messages.
    pub shader_debug_name: String,
}

/// Compute pipeline state.
#[allow(dead_code)]
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline_state: Direct3D12::ID3D12PipelineState,
    pub root_signature: Direct3D12::ID3D12RootSignature,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
    /// Per push-constant slot category expectations from shader analysis.
    pub push_constant_categories: Vec<Option<crate::types::ResourceCategory>>,
    /// Per push-constant slot SRV/UAV expectations from shader analysis (DX12).
    pub push_constant_slot_kinds: Vec<Option<crate::types::BindlessSlotKind>>,
    /// Per push-constant slot expected element stride (bytes) from reflection.
    pub binding_element_strides: Vec<Option<u32>>,
    /// Human-readable identifier used in category-mismatch error messages.
    pub shader_debug_name: String,
}

/// DXR state object + GPU shader-binding table.
#[allow(dead_code)] // `sbt` / `device_handle` keep COM objects and identity; GPU uses VAs
pub(crate) struct RayTracingPipelineState {
    pub device_handle: DeviceHandle,
    pub state_object: Direct3D12::ID3D12StateObject,
    pub root_signature: Direct3D12::ID3D12RootSignature,
    pub sbt: Direct3D12::ID3D12Resource,
    pub raygen_va: u64,
    pub raygen_size: u64,
    pub miss_va: u64,
    pub miss_size: u64,
    pub miss_stride: u64,
    pub hit_va: u64,
    pub hit_size: u64,
    pub hit_stride: u64,
    pub push_constant_categories: Vec<Option<crate::types::ResourceCategory>>,
    pub push_constant_slot_kinds: Vec<Option<crate::types::BindlessSlotKind>>,
    pub binding_element_strides: Vec<Option<u32>>,
    pub shader_debug_name: String,
}

/// GPU render target state.
#[allow(dead_code)]
pub(crate) struct RenderTargetState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    /// GPU-only render target texture
    pub texture: Direct3D12::ID3D12Resource,
    /// RTV descriptor handle offset
    pub rtv_offset: u32,
    /// Depth buffer (optional)
    pub depth_format: Option<DepthFormat>,
    pub depth_texture: Option<Direct3D12::ID3D12Resource>,
    pub dsv_offset: Option<u32>,
    /// Command list for rendering
    pub command_list: Direct3D12::ID3D12GraphicsCommandList7,
}

/// GPU texture state.
#[allow(dead_code)]
pub(crate) struct TextureState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub resource: Direct3D12::ID3D12Resource,
    /// SRV descriptor offset in CBV/SRV/UAV heap
    pub srv_offset: u32,
    /// Bindless descriptor heap offset (same as srv_offset when bindless is enabled).
    /// For `DirectInterpolated` textures this is the UAV (storage-image) slot.
    pub bindless_offset: Option<u32>,
    /// For `TextureKind::DirectInterpolated` textures, the SRV (sampled-texture) slot.
    pub sampled_bindless_offset: Option<u32>,
    /// Last known layout for enhanced texture barriers (replaces legacy `current_state`).
    ///
    /// Stored on the texture, not on the submission context: layout is device-global
    /// state for the GPU resource. Recording updates this field (`textures.write()` in
    /// [`super::compute::record_gpu_command`] and texture upload/copy helpers) so the
    /// next barrier on this texture knows where to transition from.
    ///
    /// Concurrent recording does not require a per-context copy of this field. Parcels
    /// and the record-gate enforce exclusive mutation claims — a scheme that writes a
    /// texture must fully claim it before recording, and the ledger blocks a second
    /// context from recording against the same resource until the first submit retires.
    /// Disjoint textures therefore update disjoint map entries with no semantic conflict;
    /// two contexts never legitimately race on the same `last_layout`.
    ///
    /// Phase 5b-iv may still see `textures.write()` block unrelated readers on other
    /// entries (whole-map `RwLock` writer exclusion); that is a performance concern only.
    pub last_layout: Direct3D12::D3D12_BARRIER_LAYOUT,
    /// Whether this texture was created with UAV access (TextureKind::Direct).
    pub is_storage: bool,
    /// Placed resource from a transient DX12 heap (`place_texture_in_transient_heap`).
    pub transient_placed: bool,
}

/// GPU sampler state.
#[allow(dead_code)]
pub(crate) struct SamplerState {
    pub device_handle: DeviceHandle,
    /// Sampler descriptor offset in sampler heap
    pub sampler_offset: u32,
    pub desc: SamplerDesc,
    /// Bindless descriptor heap offset (same as sampler_offset when bindless is enabled)
    pub bindless_offset: Option<u32>,
}

/// BLAS or TLAS plus bindless raytracing-acceleration-structure SRV.
pub(crate) struct AccelState {
    pub device_handle: DeviceHandle,
    pub is_tlas: bool,
    pub resource: Direct3D12::ID3D12Resource,
    pub gpu_va: u64,
    pub bindless_offset: Option<u32>,
    pub scratch: Direct3D12::ID3D12Resource,
    pub scratch_va: u64,
    pub max_primitives: u32,
    pub max_vertices: u32,
    pub vertex_stride: u32,
}

/// Maximum number of frames that can be in-flight at once.
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// Per-frame synchronization resources for proper swapchain pipelining.
#[allow(dead_code)]
pub(crate) struct FrameSync {
    pub command_list: Direct3D12::ID3D12GraphicsCommandList7,
    pub command_allocator: Direct3D12::ID3D12CommandAllocator,
    pub fence_value: u64,
    /// Set after `surface::render` submits. When false, `present` copies the compute scratch texture.
    pub render_pass_submitted: bool,
}

/// Surface (swapchain) state for window presentation.
pub(crate) struct SurfaceState {
    pub device_handle: DeviceHandle,
    pub swapchain: Dxgi::IDXGISwapChain3,
    pub render_targets: Vec<Direct3D12::ID3D12Resource>,
    pub rtv_offsets: Vec<u32>,
    pub width: u32,
    pub height: u32,
    pub format: Dxgi::Common::DXGI_FORMAT,
    /// Depth buffer (optional)
    pub depth_format: Option<DepthFormat>,
    #[allow(dead_code)] // Held for ownership; dropped when surface is destroyed
    pub depth_texture: Option<Direct3D12::ID3D12Resource>,
    pub dsv_offset: Option<u32>,
    /// Current frame index (0..MAX_FRAMES_IN_FLIGHT)
    pub current_frame: usize,
    /// Currently acquired swapchain image index
    pub current_image_index: Option<u32>,
    /// Per-frame synchronization resources
    pub frame_sync: Vec<FrameSync>,
    /// Transient texture handle for the currently acquired back buffer,
    /// registered in the bindless descriptor heap as a UAV so compute shaders
    /// can write directly to the swapchain image.
    pub current_texture_handle: Option<super::TextureHandle>,
    /// Per swapchain buffer index: persistent UAV texture for compute; results are
    /// copied to the real back buffer in `present` (swapchain images cannot be UAVs).
    pub compute_scratch_textures: Vec<Option<super::TextureHandle>>,
    /// Presentation mode (vsync strategy).
    pub present_mode: crate::types::PresentMode,
    /// DXGI frame-latency waitable object handle.
    /// Acquired once at swapchain creation via `IDXGISwapChain2::GetFrameLatencyWaitableObject`;
    /// closed in `surface::destroy`.  `acquire()` calls `WaitForSingleObject` on this handle
    /// to block until DXGI is ready to accept a new frame, replacing the per-present CPU stall.
    pub frame_latency_waitable: Option<SendSyncHandle>,
    /// Compute commands recorded between `begin_frame` and `end_frame` / `present`.
    pub pending_frame_compute: Vec<crate::backend::GpuCommand>,
    pub pending_acquire_count: u32,
    pub pending_swapchain_returns: Vec<(u32, crate::timeline::TimelineValue)>,
}

/// Map plus monotonic handle allocator for a single resource kind.
///
/// Wrapped in [`Arc<RwLock<_>>`] on [`Dx12State`] so submit recording can take read
/// guards without the global backend mutex (Phase 5b-iii).
macro_rules! handle_table {
    ($table:ident, $shared:ident, $handle:ty, $value:ty) => {
        #[derive(Default)]
        pub(crate) struct $table {
            pub entries: HashMap<$handle, $value>,
            pub next_handle: $handle,
        }

        impl $table {
            pub fn new() -> Self {
                Self {
                    entries: HashMap::new(),
                    next_handle: 1,
                }
            }

            pub fn alloc_handle(&mut self) -> $handle {
                let h = self.next_handle;
                self.next_handle += 1;
                h
            }
        }

        pub(crate) type $shared = Arc<RwLock<$table>>;
    };
}

handle_table!(BufferTable, SharedBufferTable, BufferHandle, BufferState);
handle_table!(ShaderTable, SharedShaderTable, ShaderHandle, ShaderState);
handle_table!(PipelineTable, SharedPipelineTable, PipelineHandle, PipelineState);
handle_table!(
    ComputePipelineTable,
    SharedComputePipelineTable,
    ComputePipelineHandle,
    ComputePipelineState
);
handle_table!(
    RenderTargetTable,
    SharedRenderTargetTable,
    RenderTargetHandle,
    RenderTargetState
);
handle_table!(TextureTable, SharedTextureTable, TextureHandle, TextureState);
handle_table!(SamplerTable, SharedSamplerTable, SamplerHandle, SamplerState);
handle_table!(AccelTable, SharedAccelTable, AccelerationStructureHandle, AccelState);
handle_table!(
    RayTracingPipelineTable,
    SharedRayTracingPipelineTable,
    RayTracingPipelineHandle,
    RayTracingPipelineState
);

/// Consolidated DX12 backend state.
/// This holds all the resources and state for the DX12 backend.
pub(super) struct Dx12State {
    pub factory: Dxgi::IDXGIFactory4,
    /// Whether the DXGI factory/driver supports `DXGI_PRESENT_ALLOW_TEARING`
    /// (needed for tear-free immediate presentation in windowed mode).
    pub allow_tearing: bool,
    pub adapters: Vec<DxgiAdapterInfo>,
    pub devices: HashMap<DeviceHandle, SharedLogicalDevice>,
    pub next_device_handle: DeviceHandle,
    pub contexts: SharedContextMap,
    pub next_context_id: super::ContextHandle,
    /// Synthetic context handle per device for device-queue epoch stamps (compute style).
    pub device_owner_handles: HashMap<DeviceHandle, super::ContextHandle>,
    /// Fence handles for every live context, keyed by context ID.
    ///
    /// Maintained in sync with `contexts` (inserted on create, removed on destroy).
    /// Used by [`DescriptorRegistry::drain_ready_slot_reclamations`] and [`device_retired`] so
    /// those paths can query fence completion without acquiring any per-context lock,
    /// avoiding a descriptors-lock → context-lock ordering hazard.
    ///
    /// Shared via [`Arc<RwLock<>>`] so [`ContextDeferredDeletionFlush`] clones can drain slots
    /// without the global backend mutex.
    pub context_fences: std::sync::Arc<std::sync::RwLock<HashMap<ContextHandle, ContextFenceEntry>>>,
    pub buffers: SharedBufferTable,
    pub shaders: SharedShaderTable,
    pub pipelines: SharedPipelineTable,
    pub compute_pipelines: SharedComputePipelineTable,
    pub rt_pipelines: SharedRayTracingPipelineTable,
    pub render_targets: SharedRenderTargetTable,
    pub surfaces: HashMap<SurfaceHandle, SurfaceState>,
    pub next_surface_handle: SurfaceHandle,
    pub textures: SharedTextureTable,
    pub samplers: SharedSamplerTable,
    pub accels: SharedAccelTable,
    /// Next RTV descriptor offset (high-water mark; prefer free_rtv_offsets first)
    pub next_rtv_offset: u32,
    /// Recycled RTV descriptor slots available for reuse
    pub free_rtv_offsets: Vec<u32>,
    /// Next DSV descriptor offset (high-water mark; prefer free_dsv_offsets first)
    pub next_dsv_offset: u32,
    /// Recycled DSV descriptor slots available for reuse
    pub free_dsv_offsets: Vec<u32>,
    /// Per-backend Slang compiler instance
    pub slang_compiler: crate::slang::SlangCompiler,
    /// Set to `true` when a TDR / device-removal is detected (fence completed with `u64::MAX`
    /// or `GetDeviceRemovedReason` returns a non-ok HRESULT).
    /// Polled by [`GpuBackend::is_device_lost`] without holding any lock.
    pub device_removed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
