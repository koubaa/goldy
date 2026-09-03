//! Vulkan backend internal types.
//!
//! This module contains all the state structs used by the Vulkan backend.
//!
//! ## Bindless Architecture
//!
//! The Vulkan backend uses descriptor indexing for bindless resource access (requires Vulkan 1.4):
//! - A global descriptor set contains arrays of all resource types
//! - Resources are registered at creation time and assigned indices
//! - Shaders access resources by index using nonuniformEXT qualifier
//! - Update-after-bind allows descriptor updates without pipeline barriers

use super::super::{
    AccelerationStructureHandle, BufferHandle, ComputePipelineHandle, DeviceHandle, PipelineHandle,
    RayTracingPipelineHandle, RenderTargetHandle, SamplerHandle, ShaderHandle, SurfaceHandle, TextureHandle,
};
use crate::timeline::TimelineValue;
use crate::types::{DepthFormat, TextureFormat};
use ash::vk;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Maximum number of descriptors per resource type in the global bindless set
pub const MAX_BINDLESS_RESOURCES: u32 = 16384;

/// Identifies a bindless slot within one of the category-specific allocators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SlotKey {
    StorageBuffer(u32),
    UniformBuffer(u32),
    SampledTexture(u32),
    StorageImage(u32),
    Sampler(u32),
    Accel(u32),
}

/// One deferred descriptor-slot reclamation.
pub(crate) struct PendingSlotReclamation {
    pub slot: SlotKey,
    /// `(context_handle, min_seq_that_must_retire)`
    pub requirements: Vec<(super::ContextHandle, u64)>,
}

fn slot_requirements_met(
    requirements: &[(super::ContextHandle, u64)],
    completed_values: &HashMap<super::ContextHandle, u64>,
) -> bool {
    requirements
        .iter()
        .all(|(ctx_id, required_seq)| completed_values.get(ctx_id).is_none_or(|&v| v >= *required_seq))
}

/// Device-shared descriptor registry.
///
/// Contains the irreducible shared state for bindless slot allocation: the
/// `ResourceRegistry` (descriptor slot allocator), the per-context
/// `slot_last_seen` reference table, and the `pending_slot_reclamations`
/// list.  Wrapped in `Arc<Mutex<DescriptorRegistry>>` on `LogicalDevice` so that
/// submit paths can hold the descriptors lock independently of the global backend
/// mutex (Phase 4), and can clone the `Arc` before dropping the global lock
/// (Phase 5).
pub(crate) struct DescriptorRegistry {
    /// Registry tracking resource indices in the global descriptor set.
    pub resource_registry: ResourceRegistry,
    /// Maps bindless slot → per-context last-submitted seq that referenced it.
    /// Updated at every submit. Entry removed when the slot is queued for reclamation.
    pub slot_last_seen: HashMap<SlotKey, HashMap<super::ContextHandle, u64>>,
    /// Slots waiting for referencing contexts to retire before returning to free lists.
    pub pending_slot_reclamations: Vec<PendingSlotReclamation>,
    /// Retained command buffers still baking each bindless slot (incremental refcount).
    retained_users: HashMap<SlotKey, u32>,
}

impl DescriptorRegistry {
    pub(crate) fn new() -> Self {
        Self {
            resource_registry: ResourceRegistry::new(),
            slot_last_seen: HashMap::new(),
            pending_slot_reclamations: Vec::new(),
            retained_users: HashMap::new(),
        }
    }

    /// Increment pin count for each slot baked into a newly retained command buffer.
    pub(crate) fn pin_retained_slots(&mut self, slots: impl IntoIterator<Item = SlotKey>) {
        for slot in slots {
            *self.retained_users.entry(slot).or_insert(0) += 1;
        }
    }

    /// Decrement pin count when a retained CB is evicted or a context is destroyed.
    pub(crate) fn unpin_retained_slots(&mut self, slots: impl IntoIterator<Item = SlotKey>) {
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
    pub(crate) fn retained_user_count(&self, slot: SlotKey) -> u32 {
        self.retained_users.get(&slot).copied().unwrap_or(0)
    }

    /// True when no retained command buffer still bakes any of `slots`.
    pub(crate) fn retained_pins_clear(&self, slots: &[SlotKey]) -> bool {
        slots
            .iter()
            .all(|slot| self.retained_users.get(slot).copied().unwrap_or(0) == 0)
    }

    /// Record that `ctx` submitted `seq` referencing each bindless slot in `slots`.
    pub(crate) fn record_slot_usage(
        &mut self,
        ctx: super::ContextHandle,
        seq: u64,
        slots: impl IntoIterator<Item = SlotKey>,
    ) {
        for slot in slots {
            self.slot_last_seen
                .entry(slot)
                .or_default()
                .entry(ctx)
                .and_modify(|v| *v = (*v).max(seq))
                .or_insert(seq);
        }
    }

    /// Queue a slot for deferred reclamation once all referencing contexts retire.
    pub(crate) fn queue_slot_reclamation(&mut self, slot: SlotKey) {
        let requirements: Vec<_> = self
            .slot_last_seen
            .remove(&slot)
            .map(|m| m.into_iter().collect())
            .unwrap_or_default();
        // Always defer: empty requirements may still be referenced by retained command buffers.
        self.pending_slot_reclamations
            .push(PendingSlotReclamation { slot, requirements });
    }

    /// Reclaim all descriptor slots for a destroyed buffer handle.
    ///
    /// Returns the slot keys that were reclaimed (for gating physical GPU free).
    pub(crate) fn reclaim_buffer_slots(&mut self, handle: BufferHandle) -> Vec<SlotKey> {
        let slots = self.resource_registry.extract_buffer_slots(handle);
        for slot in slots.iter().copied() {
            self.queue_slot_reclamation(slot);
        }
        slots
    }

    /// Bindless slots currently assigned to `handle` (without reclaiming).
    pub(crate) fn buffer_slot_keys(&self, handle: BufferHandle) -> Vec<SlotKey> {
        self.resource_registry.buffer_slot_keys(handle)
    }

    /// Bindless slots currently assigned to a texture handle (without reclaiming).
    pub(crate) fn texture_slot_keys(&self, handle: TextureHandle) -> Vec<SlotKey> {
        self.resource_registry.texture_slot_keys(handle)
    }

    /// Reclaim all descriptor slots for a destroyed texture handle.
    pub(crate) fn reclaim_texture_slots(&mut self, handle: TextureHandle) {
        let slots = self.resource_registry.extract_texture_slots(handle);
        for slot in slots {
            self.queue_slot_reclamation(slot);
        }
    }

    /// Reclaim all descriptor slots for a destroyed sampler handle.
    pub(crate) fn reclaim_sampler_slots(&mut self, handle: SamplerHandle) {
        let slots = self.resource_registry.extract_sampler_slots(handle);
        for slot in slots {
            self.queue_slot_reclamation(slot);
        }
    }

    /// Reclaim bindless slots for a destroyed acceleration structure.
    pub(crate) fn reclaim_accel_slots(&mut self, handle: AccelerationStructureHandle) {
        let slots = self.resource_registry.extract_accel_slots(handle);
        for slot in slots {
            self.queue_slot_reclamation(slot);
        }
    }

    pub(crate) fn accel_slot_keys(&self, handle: AccelerationStructureHandle) -> Vec<SlotKey> {
        self.resource_registry.accel_slot_keys(handle)
    }

    /// Same as [`Self::bindless_retirement_requirements_for_buffer`] but for an AS handle.
    pub(crate) fn bindless_retirement_requirements_for_accel(
        &self,
        handle: AccelerationStructureHandle,
        base: Vec<(super::ContextHandle, u64)>,
    ) -> Vec<(super::ContextHandle, u64)> {
        let slots = self.resource_registry.accel_slot_keys(handle);
        self.merge_slot_requirements(&slots, base)
    }

    /// Return pending slots to the free list once every referencing context has retired.
    ///
    /// Takes a pre-snapshotted map of `(context_handle → completed_timeline_value)` rather
    /// than querying semaphores directly, so this method is safe to call while holding the
    /// descriptors lock without creating a semaphore-query → descriptors-lock ordering hazard.
    /// A missing entry means the context has been destroyed and is considered fully retired.
    pub(crate) fn drain_ready_slot_reclamations(&mut self, completed_values: &HashMap<super::ContextHandle, u64>) {
        let mut i = 0;
        while i < self.pending_slot_reclamations.len() {
            let slot = self.pending_slot_reclamations[i].slot;
            let gpu_ready = self.pending_slot_reclamations[i]
                .requirements
                .iter()
                .all(|(ctx_id, required_seq)| completed_values.get(ctx_id).is_none_or(|&v| v >= *required_seq));
            let pin_clear = self.retained_users.get(&slot).copied().unwrap_or(0) == 0;
            if gpu_ready && pin_clear {
                let entry = self.pending_slot_reclamations.swap_remove(i);
                self.resource_registry.free_slot(entry.slot);
            } else {
                i += 1;
            }
        }
    }

    /// Per-context requirements that must retire before a buffer's GPU resource can be
    /// released: `base` merged with every live `slot_last_seen` entry for this buffer's
    /// bindless slots.
    pub(crate) fn bindless_retirement_requirements_for_buffer(
        &self,
        handle: BufferHandle,
        base: Vec<(super::ContextHandle, u64)>,
    ) -> Vec<(super::ContextHandle, u64)> {
        let slots = self.resource_registry.buffer_slot_keys(handle);
        self.merge_slot_requirements(&slots, base)
    }

    /// Same as [`Self::bindless_retirement_requirements_for_buffer`] but for a texture handle.
    pub(crate) fn bindless_retirement_requirements_for_texture(
        &self,
        handle: TextureHandle,
        base: Vec<(super::ContextHandle, u64)>,
    ) -> Vec<(super::ContextHandle, u64)> {
        let slots = self.resource_registry.texture_slot_keys(handle);
        self.merge_slot_requirements(&slots, base)
    }

    fn merge_slot_requirements(
        &self,
        slots: &[SlotKey],
        base: Vec<(super::ContextHandle, u64)>,
    ) -> Vec<(super::ContextHandle, u64)> {
        let mut merged: HashMap<super::ContextHandle, u64> = base.into_iter().collect();
        for &slot in slots {
            if let Some(map) = self.slot_last_seen.get(&slot) {
                for (ctx, seq) in map.iter() {
                    merged.entry(*ctx).and_modify(|v| *v = (*v).max(*seq)).or_insert(*seq);
                }
            }
        }
        merged.into_iter().collect()
    }
}

/// Snapshot the completed timeline value for every context belonging to `for_device`.
///
/// Call this **before** locking the descriptors so `drain_ready_slot_reclamations` has the
/// retirement data it needs without touching live context state while the descriptors lock is held.
pub(super) fn snapshot_context_completed_values(
    device: &ash::Device,
    contexts: &SharedContextMap,
    for_device: super::DeviceHandle,
) -> HashMap<super::ContextHandle, u64> {
    contexts
        .read()
        .unwrap()
        .iter()
        .filter_map(|(&id, sc_arc)| {
            let sc = sc_arc.lock().unwrap();
            if sc.device != for_device {
                return None;
            }
            let v = unsafe { device.get_semaphore_counter_value(sc.timeline_semaphore).unwrap_or(0) };
            Some((id, v))
        })
        .collect()
}

/// Binding indices within the global bindless descriptor set
/// Organized by ACCESS PATTERN:
///
///   0: SCATTERED - Any thread reads/writes any address. No coherence assumptions.
///   1: BROADCAST - All threads read same address. Hardware optimizes for this.
///   2: INTERPOLATED - Hardware filtering between neighboring elements (texture units).
///   3: DIRECT_SPATIAL - 2D/3D indexing without filtering. Read/write.
///   4: FILTER_CONFIG - Not data. Configuration for interpolated access.
///
/// These map to Vulkan descriptor types, but the access pattern is what matters.
pub mod bindless_bindings {
    /// Scattered access: any thread, any address, read/write
    /// Maps to: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    /// Slang: `StructuredBuffer<T>`, `RWStructuredBuffer<T>`
    pub const SCATTERED: u32 = 0;

    /// Broadcast access: all threads same address, read-only (enables cache optimization)
    /// Maps to: VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    /// Slang: `ConstantBuffer<T>`
    pub const BROADCAST: u32 = 1;

    /// Interpolated access: hardware filtering between neighbors (texture units)
    /// Maps to: VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
    /// Slang: `Texture2D<T>` (read with sampler)
    pub const INTERPOLATED: u32 = 2;

    /// Direct spatial access: 2D/3D indexing without filtering, read/write
    /// Maps to: VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
    /// Slang: `RWTexture2D<T>`
    pub const DIRECT_SPATIAL: u32 = 3;

    /// Filter configuration: settings for interpolated access (not data)
    /// Maps to: VK_DESCRIPTOR_TYPE_SAMPLER
    /// Slang: SamplerState
    pub const FILTER_CONFIG: u32 = 4;
    /// Acceleration structures (ray query).
    /// Maps to: VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
    pub const ACCEL: u32 = 5;

    // Legacy aliases for legitibility to graphics programmers
    pub const STORAGE_BUFFERS: u32 = SCATTERED;
    pub const UNIFORM_BUFFERS: u32 = BROADCAST;
    pub const SAMPLED_IMAGES: u32 = INTERPOLATED;
    pub const STORAGE_IMAGES: u32 = DIRECT_SPATIAL;
    pub const SAMPLERS: u32 = FILTER_CONFIG;
}

// PushLayout and its constants live in the shared module so all three backends
// use one definition. Re-export them here so internal code keeps using the
// same unqualified names as before.
pub use super::super::shared::{PushLayout, TOTAL_PUSH_BYTES};

use super::super::shared::SlotAllocator;

/// Registry for tracking bindless resource indices.
///
/// Each resource type has a [`SlotAllocator`] that prefers recycled free-list
/// slots before minting new indices, keeping the live counter bounded by
/// `MAX_BINDLESS_RESOURCES` under create/destroy churn.
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    storage_buffer: SlotAllocator,
    uniform_buffer: SlotAllocator,
    sampled_texture: SlotAllocator,
    storage_image: SlotAllocator,
    sampler: SlotAllocator,
    accel: SlotAllocator,
    /// Map BufferHandle -> (bindless_index, is_storage)
    pub buffer_indices: HashMap<BufferHandle, (u32, bool)>,
    /// Map TextureHandle -> (bindless_index, is_storage_image)
    pub texture_indices: HashMap<TextureHandle, (u32, bool)>,
    pub sampler_indices: HashMap<SamplerHandle, u32>,
    pub accel_indices: HashMap<AccelerationStructureHandle, u32>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_buffer(&mut self, handle: BufferHandle, is_storage: bool) -> u32 {
        let index = if is_storage {
            self.storage_buffer.alloc()
        } else {
            self.uniform_buffer.alloc()
        };
        self.buffer_indices.insert(handle, (index, is_storage));
        index
    }

    /// Reserve storage-array indices `[0, min)` for frame-table protocol slots.
    pub fn ensure_storage_start(&mut self, min: u32) {
        self.storage_buffer.ensure_minimum_next(min);
    }

    pub fn register_texture(&mut self, handle: TextureHandle, is_storage_image: bool) -> u32 {
        let index = if is_storage_image {
            self.storage_image.alloc()
        } else {
            self.sampled_texture.alloc()
        };
        self.texture_indices.insert(handle, (index, is_storage_image));
        index
    }

    pub fn register_sampler(&mut self, handle: SamplerHandle) -> u32 {
        let index = self.sampler.alloc();
        self.sampler_indices.insert(handle, index);
        index
    }

    pub fn register_accel(&mut self, handle: AccelerationStructureHandle) -> u32 {
        let index = self.accel.alloc();
        self.accel_indices.insert(handle, index);
        index
    }

    pub fn accel_slot_keys(&self, handle: AccelerationStructureHandle) -> Vec<SlotKey> {
        self.accel_indices
            .get(&handle)
            .copied()
            .map(|index| vec![SlotKey::Accel(index)])
            .unwrap_or_default()
    }

    /// Remove an acceleration structure's handle mapping without recycling the slot.
    pub fn extract_accel_slots(&mut self, handle: AccelerationStructureHandle) -> Vec<SlotKey> {
        self.accel_indices
            .remove(&handle)
            .map(|index| vec![SlotKey::Accel(index)])
            .unwrap_or_default()
    }

    /// Bindless slot keys for `handle` without removing the registry entry.
    pub fn buffer_slot_keys(&self, handle: BufferHandle) -> Vec<SlotKey> {
        self.buffer_indices
            .get(&handle)
            .map(|&(index, is_storage)| {
                vec![if is_storage {
                    SlotKey::StorageBuffer(index)
                } else {
                    SlotKey::UniformBuffer(index)
                }]
            })
            .unwrap_or_default()
    }

    /// Bindless slot keys for `handle` without removing the registry entry.
    pub fn texture_slot_keys(&self, handle: TextureHandle) -> Vec<SlotKey> {
        self.texture_indices
            .get(&handle)
            .map(|&(index, is_storage_image)| {
                vec![if is_storage_image {
                    SlotKey::StorageImage(index)
                } else {
                    SlotKey::SampledTexture(index)
                }]
            })
            .unwrap_or_default()
    }

    /// Remove a buffer's handle mapping and return its slot key without recycling.
    pub fn extract_buffer_slots(&mut self, handle: BufferHandle) -> Vec<SlotKey> {
        let mut slots = Vec::new();
        if let Some((index, is_storage)) = self.buffer_indices.remove(&handle) {
            slots.push(if is_storage {
                SlotKey::StorageBuffer(index)
            } else {
                SlotKey::UniformBuffer(index)
            });
        }
        slots
    }

    /// Remove a texture's handle mapping and return its slot key without recycling.
    pub fn extract_texture_slots(&mut self, handle: TextureHandle) -> Vec<SlotKey> {
        let mut slots = Vec::new();
        if let Some((index, is_storage_image)) = self.texture_indices.remove(&handle) {
            slots.push(if is_storage_image {
                SlotKey::StorageImage(index)
            } else {
                SlotKey::SampledTexture(index)
            });
        }
        slots
    }

    pub fn free_slot(&mut self, key: SlotKey) {
        match key {
            SlotKey::StorageBuffer(i) => self.storage_buffer.free(i),
            SlotKey::UniformBuffer(i) => self.uniform_buffer.free(i),
            SlotKey::SampledTexture(i) => self.sampled_texture.free(i),
            SlotKey::StorageImage(i) => self.storage_image.free(i),
            SlotKey::Sampler(i) => self.sampler.free(i),
            SlotKey::Accel(i) => self.accel.free(i),
        }
    }

    /// Remove a sampler's handle mapping and return its slot key without recycling.
    pub fn extract_sampler_slots(&mut self, handle: SamplerHandle) -> Vec<SlotKey> {
        if let Some(index) = self.sampler_indices.remove(&handle) {
            vec![SlotKey::Sampler(index)]
        } else {
            Vec::new()
        }
    }

    #[cfg(test)]
    pub fn unregister_sampler(&mut self, handle: SamplerHandle) {
        if let Some(index) = self.sampler_indices.remove(&handle) {
            self.sampler.free(index);
        }
    }

    /// Number of available (allocatable) slots in the given category.
    pub fn available_slots(&self, category: crate::types::ResourceCategory) -> u32 {
        let allocator = match category {
            crate::types::ResourceCategory::Scattered => &self.storage_buffer,
            crate::types::ResourceCategory::Broadcast => &self.uniform_buffer,
            crate::types::ResourceCategory::Texture => &self.sampled_texture,
            crate::types::ResourceCategory::StorageImage => &self.storage_image,
            crate::types::ResourceCategory::Sampler => &self.sampler,
            crate::types::ResourceCategory::Accel => &self.accel,
        };
        MAX_BINDLESS_RESOURCES.saturating_sub(allocator.live_count())
    }
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    fn free_buffer_slots(reg: &mut ResourceRegistry, handle: BufferHandle) {
        for key in reg.extract_buffer_slots(handle) {
            reg.free_slot(key);
        }
    }

    fn free_texture_slots(reg: &mut ResourceRegistry, handle: TextureHandle) {
        for key in reg.extract_texture_slots(handle) {
            reg.free_slot(key);
        }
    }

    /// Simulate the per-frame create/destroy churn that transient pool consumers generate for transient
    /// pool-view storage buffers. The counter must stay bounded — well below
    /// MAX_BINDLESS_RESOURCES — even after far more iterations than the heap limit.
    #[test]
    fn storage_buffer_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as BufferHandle;
            reg.register_buffer(handle, true);
            free_buffer_slots(&mut reg, handle);
        }
        assert_eq!(
            reg.storage_buffer.next_fresh(),
            1,
            "storage buffer counter grew; slot recycling not working"
        );
        assert_eq!(reg.storage_buffer.free_count(), 1);
    }

    /// Uniform buffer (non-storage) slots must recycle independently.
    #[test]
    fn uniform_buffer_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as BufferHandle;
            reg.register_buffer(handle, false);
            free_buffer_slots(&mut reg, handle);
        }
        assert_eq!(
            reg.uniform_buffer.next_fresh(),
            1,
            "uniform buffer counter grew; slot recycling not working"
        );
    }

    /// Sampled textures (non-storage) must recycle their indices.
    #[test]
    fn sampled_texture_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as TextureHandle;
            reg.register_texture(handle, false);
            free_texture_slots(&mut reg, handle);
        }
        assert_eq!(reg.sampled_texture.next_fresh(), 1);
        assert_eq!(reg.sampled_texture.free_count(), 1);
    }

    /// Storage images (RWTexture2D) must recycle their indices.
    #[test]
    fn storage_image_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as TextureHandle;
            reg.register_texture(handle, true);
            free_texture_slots(&mut reg, handle);
        }
        assert_eq!(reg.storage_image.next_fresh(), 1);
        assert_eq!(reg.storage_image.free_count(), 1);
    }

    /// Sampler slots must recycle independently.
    #[test]
    fn sampler_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..5_000u64 {
            let handle = i as SamplerHandle;
            reg.register_sampler(handle);
            reg.unregister_sampler(handle);
        }
        assert_eq!(reg.sampler.next_fresh(), 1);
        assert_eq!(reg.sampler.free_count(), 1);
    }

    /// Simultaneously-live resources must receive distinct indices.
    #[test]
    fn live_resources_get_distinct_indices() {
        let mut reg = ResourceRegistry::new();
        const N: u64 = 64;
        let mut indices: Vec<u32> = (0..N).map(|i| reg.register_buffer(i as BufferHandle, true)).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(
            indices.len(),
            N as usize,
            "duplicate indices assigned to live resources"
        );
    }

    /// The high-water mark of the monotonic counter must never exceed the number of
    /// concurrently-live resources, i.e. freed slots are reused before fresh ones are minted.
    #[test]
    fn high_water_mark_bounded_by_live_count() {
        let mut reg = ResourceRegistry::new();
        const LIVE: u64 = 8;
        const ROUNDS: u64 = 10_000;

        for i in 0..LIVE {
            reg.register_buffer(i as BufferHandle, true);
        }
        for i in LIVE..LIVE + ROUNDS {
            free_buffer_slots(&mut reg, (i - LIVE) as BufferHandle);
            reg.register_buffer(i as BufferHandle, true);
        }
        assert!(
            reg.storage_buffer.next_fresh() <= LIVE as u32,
            "counter ({}) exceeded live count ({LIVE}); slot recycling broken",
            reg.storage_buffer.next_fresh()
        );
    }

    #[test]
    fn slot_deferred_until_context_retires() {
        use crate::backend::ContextHandle;
        let mut reg = ResourceRegistry::new();
        let handle = 1u64 as BufferHandle;
        let slot = reg.register_buffer(handle, true);
        const CTX_A: ContextHandle = 10;
        const SEQ: u64 = 5;

        let slots = reg.extract_buffer_slots(handle);
        assert_eq!(slots, vec![SlotKey::StorageBuffer(slot)]);

        let mut pending = vec![PendingSlotReclamation {
            slot: SlotKey::StorageBuffer(slot),
            requirements: vec![(CTX_A, SEQ)],
        }];

        assert_eq!(reg.storage_buffer.free_count(), 0);

        let mut retired = HashMap::from([(CTX_A, 4u64)]);
        let mut i = 0;
        while i < pending.len() {
            let ready = pending[i]
                .requirements
                .iter()
                .all(|(ctx, seq)| retired.get(ctx).copied().unwrap_or(0) >= *seq);
            if ready {
                let entry = pending.swap_remove(i);
                reg.free_slot(entry.slot);
            } else {
                i += 1;
            }
        }
        assert_eq!(reg.storage_buffer.free_count(), 0);

        retired.insert(CTX_A, SEQ);
        i = 0;
        while i < pending.len() {
            let ready = pending[i]
                .requirements
                .iter()
                .all(|(ctx, seq)| retired.get(ctx).copied().unwrap_or(0) >= *seq);
            if ready {
                let entry = pending.swap_remove(i);
                reg.free_slot(entry.slot);
            } else {
                i += 1;
            }
        }
        assert_eq!(reg.storage_buffer.free_count(), 1);
    }

    #[test]
    fn slot_waits_for_all_referencing_contexts() {
        use crate::backend::ContextHandle;
        let mut reg = ResourceRegistry::new();
        let handle = 2u64 as BufferHandle;
        let slot = reg.register_buffer(handle, true);
        const CTX_A: ContextHandle = 1;
        const CTX_B: ContextHandle = 2;

        let mut pending = vec![PendingSlotReclamation {
            slot: SlotKey::StorageBuffer(slot),
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
                    reg.free_slot(entry.slot);
                } else {
                    i += 1;
                }
            }
        };

        drain_pending(&retired, &mut reg, &mut pending);
        assert_eq!(reg.storage_buffer.free_count(), 0);

        retired.insert(CTX_A, 3);
        drain_pending(&retired, &mut reg, &mut pending);
        assert_eq!(reg.storage_buffer.free_count(), 0);

        retired.insert(CTX_B, 7);
        drain_pending(&retired, &mut reg, &mut pending);
        assert_eq!(reg.storage_buffer.free_count(), 1);
    }

    /// Retained-graph pin blocks slot free even when GPU requirements are met.
    #[test]
    fn retained_pin_blocks_slot_free_until_unpin() {
        use crate::backend::ContextHandle;
        let mut dr = DescriptorRegistry::new();
        let handle = 3u64 as BufferHandle;
        let slot = dr.resource_registry.register_buffer(handle, true);
        let slot_key = SlotKey::StorageBuffer(slot);

        dr.pin_retained_slots([slot_key]);
        dr.queue_slot_reclamation(slot_key);
        dr.resource_registry.extract_buffer_slots(handle);

        let completed: HashMap<ContextHandle, u64> = HashMap::new();
        dr.drain_ready_slot_reclamations(&completed);
        assert_eq!(dr.resource_registry.storage_buffer.free_count(), 0, "pin blocks free");
        assert_eq!(dr.retained_user_count(slot_key), 1);

        dr.unpin_retained_slots([slot_key]);
        dr.drain_ready_slot_reclamations(&completed);
        assert_eq!(
            dr.resource_registry.storage_buffer.free_count(),
            1,
            "unpin then drain frees"
        );
    }

    /// Two retained graphs sharing a slot: free only after both unpinned.
    #[test]
    fn retained_pin_shared_slot_needs_two_unpins() {
        use crate::backend::ContextHandle;
        let mut dr = DescriptorRegistry::new();
        let handle = 4u64 as BufferHandle;
        let slot = dr.resource_registry.register_buffer(handle, true);
        let slot_key = SlotKey::StorageBuffer(slot);

        dr.pin_retained_slots([slot_key]);
        dr.pin_retained_slots([slot_key]);
        dr.queue_slot_reclamation(slot_key);
        dr.resource_registry.extract_buffer_slots(handle);

        let completed: HashMap<ContextHandle, u64> = HashMap::new();
        dr.unpin_retained_slots([slot_key]);
        dr.drain_ready_slot_reclamations(&completed);
        assert_eq!(dr.resource_registry.storage_buffer.free_count(), 0, "one pin remains");

        dr.unpin_retained_slots([slot_key]);
        dr.drain_ready_slot_reclamations(&completed);
        assert_eq!(dr.resource_registry.storage_buffer.free_count(), 1);
    }

    /// Replacing a retained graph: unpin old-only slots, new slots stay pinned.
    #[test]
    fn retained_pin_replace_frees_old_only_slots() {
        use crate::backend::ContextHandle;
        let mut dr = DescriptorRegistry::new();
        let slot_old = dr.resource_registry.register_buffer(5, true);
        let slot_new = dr.resource_registry.register_buffer(6, true);
        let old_key = SlotKey::StorageBuffer(slot_old);
        let new_key = SlotKey::StorageBuffer(slot_new);

        dr.pin_retained_slots([old_key]);
        dr.pin_retained_slots([new_key]);
        dr.unpin_retained_slots([old_key]);

        dr.queue_slot_reclamation(old_key);
        dr.queue_slot_reclamation(new_key);
        dr.resource_registry.buffer_indices.remove(&5);
        dr.resource_registry.buffer_indices.remove(&6);

        let completed: HashMap<ContextHandle, u64> = HashMap::new();
        dr.drain_ready_slot_reclamations(&completed);
        assert_eq!(dr.resource_registry.storage_buffer.free_count(), 1, "old slot freed");
        assert_eq!(dr.retained_user_count(new_key), 1);
        assert_eq!(dr.retained_user_count(old_key), 0);
    }
}

/// Information about a physical Vulkan device.
pub(crate) struct PhysicalDeviceInfo {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub adapter_id: u32,
    /// From physical-device features at enumeration (immutable for this adapter).
    pub supports_sparse_buffer: bool,
    /// From [`vk::PhysicalDeviceLimits::timestamp_compute_and_graphics`].
    pub vk_timestamp_compute_and_graphics: bool,
    pub vk_timestamp_period_ns: f32,
    /// `VK_KHR_ray_query` + acceleration structures with feature bits enabled.
    pub ray_query: bool,
    /// `VK_KHR_ray_tracing_pipeline` + acceleration structures with feature bits enabled.
    pub ray_tracing_pipelines: bool,
    /// `VK_EXT_mesh_shader` `meshShader` feature.
    pub mesh_shaders: bool,
    /// `VK_EXT_mesh_shader` `taskShader` feature.
    pub amplification_shaders: bool,
}

/// Per-context async submission stream (timeline, poller, command pool).
/// Max compute-capable queues allocated for per-context assignment at device create.
pub(crate) const MAX_CONTEXT_COMPUTE_QUEUES: u32 = 8;

pub(crate) struct SubmissionContext {
    pub device: super::DeviceHandle,
    /// Synthetic device-owner context for render-partition epoch stamps (no compute queue).
    pub is_device_owner: bool,
    /// Per-context compute queue (or device graphics queue for [`Self::is_device_owner`]).
    pub queue: vk::Queue,
    /// Queue family of [`Self::queue`] (compute family for normal contexts).
    #[allow(dead_code)]
    pub queue_family: u32,
    /// Index into [`LogicalDevice::compute_queues`] free list; `None` for device owner.
    pub queue_index: Option<usize>,
    /// Serialises `vkQueueSubmit2` on this context's queue.
    pub queue_lock: std::sync::Arc<std::sync::Mutex<()>>,
    pub timeline_semaphore: vk::Semaphore,
    /// Last device-global seq value submitted on this context.
    pub last_submitted_seq: u64,
    pub signal_queue: std::sync::Arc<crate::signal::SignalQueue>,
    pub fence_shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub fence_thread: Option<std::thread::JoinHandle<()>>,
    pub command_pool: vk::CommandPool,
    pub free_cmd_buffers: Vec<vk::CommandBuffer>,
    /// Retained dispatch CBs keyed by scheme fingerprint for zero-recording-cost re-submission.
    pub retained_compute_cbs: HashMap<u64, RetainedVkCb>,
    /// Command buffers to free once this context's timeline reaches the key.
    pub timeline_cmd_buffers: std::collections::HashMap<u64, Vec<vk::CommandBuffer>>,
    /// Device-queue render CBs (from [`LogicalDevice::command_pool`]) keyed by timeline value.
    pub graphics_timeline_cmd_buffers: std::collections::HashMap<u64, Vec<vk::CommandBuffer>>,
    /// Per-context staging belt for DEVICE_LOCAL WriteBuffer uploads.
    /// Pools HOST_VISIBLE chunks across submits so no staging memory is reused
    /// before its GPU copy finishes (keyed by this context's timeline values).
    pub staging_belt: super::staging::StagingBelt,
    /// Per-context pool for texture-upload staging buffers.
    /// Eliminates per-frame vkAllocateMemory / vkFreeMemory for WriteTexture.
    pub texture_staging_pool: super::staging::TextureStagingPool,
    /// Per-context deferred deletion queue.
    ///
    /// Holds resources whose GPU lifetime is bounded exclusively by **this**
    /// context's timeline semaphore (e.g. submit-internal temporaries).  Drained
    /// on each submit using the context timeline semaphore — never via `device_retired` —
    /// so no other context's progress can block reclaim here.
    ///
    /// Dispatch-batch arg buffers and other resources whose lifetime is bounded by
    /// this context's timeline only.
    pub deletion_queue: DeletionQueue,
    /// Per-context frame-table GPU resources and ring state (bindless slots 0/1).
    pub frame_table: SharedContextFrameTable,
    /// GPU profile readbacks deferred until the context timeline retires each submit TV.
    pub pending_gpu_profiles: Vec<(u64, super::pending_submit::VulkanGpuProfileWork)>,
}

/// Which timeline semaphore must reach a device-global submission value before that
/// value is considered retired. Independent per-context compute queues cannot share
/// a single max-over-contexts completion horizon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimelineWaitTarget {
    Context(super::ContextHandle),
    DeviceOwner,
}

/// A logical Vulkan device with associated resources.
pub(crate) struct LogicalDevice {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    #[allow(dead_code)]
    pub adapter_id: u32,
    pub queue: vk::Queue,
    pub queue_family: u32,
    /// Queue family used for per-context compute submits (may equal [`Self::queue_family`]).
    pub compute_queue_family: u32,
    /// Pool of compute-capable queues assigned to submission contexts.
    ///
    /// When [`Self::compute_queues_alias_graphics`] is true, every entry is the
    /// same handle as [`Self::queue`] (logical slots on a one-queue device).
    pub compute_queues: Vec<vk::Queue>,
    /// True when the graphics family exposes only one queue: contexts share
    /// [`Self::queue`] and [`Self::queue_lock`] instead of private compute queues.
    pub compute_queues_alias_graphics: bool,
    pub free_compute_queue_indices: std::sync::Mutex<std::collections::VecDeque<usize>>,
    /// Reusable primary command buffers for device-queue render submits.
    ///
    /// Also serialises allocate/free against [`Self::command_pool`] — see
    /// [`Self::acquire_device_cmd_buffer`].
    pub free_device_cmd_buffers: std::sync::Mutex<Vec<vk::CommandBuffer>>,
    /// Queue used for [`vk::Device::queue_bind_sparse`] (often same as graphics).
    pub sparse_binding_queue: vk::Queue,
    /// Shared device-queue command pool. Externally synchronised via
    /// [`Self::free_device_cmd_buffers`] for allocate / free / recycle.
    pub command_pool: vk::CommandPool,

    /// Host-visible oversize pools use dense allocations; device-local oversize may use sparse binding.
    pub supports_sparse_buffer: bool,
    /// Sparse **buffer** binding alignment from [`vk::MemoryRequirements::alignment`] (typically 64 KiB).
    pub sparse_buffer_block_size: u64,
    #[allow(dead_code)] // captured at device init for diagnostics / future use
    pub sparse_memory_type_index: u32,
    /// Sub-allocated DEVICE_LOCAL pages for [`super::sparse::SparsePagePool`].
    /// `Mutex` so `LogicalDevice` can be `Arc`-wrapped (Phase 5a).
    pub sparse_page_pool: Mutex<Option<super::sparse::SparsePagePool>>,

    // Vulkan 1.4 core via KHR extension loaders (ash 0.38 doesn't have core 1.4 wrappers yet)
    pub map_memory2: ash::khr::map_memory2::Device,
    /// Inline ray query + acceleration structures (optional).
    pub ray_query: bool,
    pub ray_tracing_pipelines: bool,
    pub accel_khr: Option<ash::khr::acceleration_structure::Device>,
    pub rtp_khr: Option<ash::khr::ray_tracing_pipeline::Device>,
    pub mesh_shaders: bool,
    pub mesh_ext: Option<ash::ext::mesh_shader::Device>,
    pub rt_shader_group_handle_size: u32,
    pub rt_shader_group_handle_alignment: u32,
    pub rt_shader_group_base_alignment: u32,

    // Bindless infrastructure
    /// Global descriptor pool for bindless resources
    pub bindless_descriptor_pool: Option<vk::DescriptorPool>,
    /// Global descriptor set layout for bindless resources
    pub bindless_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    /// Global descriptor set containing all bindless resources
    pub bindless_descriptor_set: Option<vk::DescriptorSet>,
    /// Pipeline layout for bindless rendering (includes the global set)
    pub bindless_pipeline_layout: Option<vk::PipelineLayout>,
    /// Descriptor registry: `ResourceRegistry` + slot reference-tracking.
    /// `Arc` so Phase 5 can clone it out of `LogicalDevice` before dropping the
    /// global backend lock.
    pub descriptors: Arc<Mutex<DescriptorRegistry>>,
    /// Deferred deletion queue for bindless-tracked buffer/texture destroys that may
    /// span more than one context. Keyed by per-context requirement snapshots.
    pub deletion_queue: Mutex<DeviceDeletionQueue>,
    /// Deferred Vk buffer frees after fence requirements and retained-graph pins clear.
    pub pending_buffer_gpu_releases: Mutex<Vec<PendingBufferGpuRelease>>,
    /// Device-global submission sequence (shared value space; contexts signal their own semaphores).
    /// `Arc` allows submit paths to clone the counter out before dropping device/state borrows
    /// (required for Phase 5 lock-free submit).
    pub timeline_next: Arc<AtomicU64>,
    /// Minimum completed horizon after a context is destroyed (never lowers `device_retired`).
    /// `AtomicU64` so `LogicalDevice` can be `Arc`-wrapped; updated with `fetch_max`.
    pub retired_floor: AtomicU64,
    /// Per global timeline value: which native semaphore must reach that value.
    pub timeline_wait_targets: Mutex<BTreeMap<u64, TimelineWaitTarget>>,
    /// Cached highest contiguous retired timeline value (≥ [`Self::retired_floor`]).
    pub timeline_retired: AtomicU64,

    /// Optional driver pipeline cache persisted to disk (`~/.cache/goldy/pipeline_cache_<adapter>.bin`).
    pub pipeline_cache: vk::PipelineCache,

    /// Serialises all `vkQueueSubmit2` calls on this device's graphics/present queue.
    ///
    /// Per-context compute submits use [`SubmissionContext::queue_lock`] instead.
    pub queue_lock: Arc<Mutex<()>>,
    /// Live per-context compute queue locks (registered at context create, removed at destroy).
    /// [`Self::device_wait_idle_locked`] and [`Self::queues_wait_idle_locked`] acquire these
    /// alongside [`Self::queue_lock`] so idle calls do not race with compute submits.
    pub active_context_queue_locks: Mutex<Vec<Arc<Mutex<()>>>>,

    /// Timestamp query support (`VkPhysicalDeviceLimits::timestamp_compute_and_graphics`).
    pub vk_timestamp_compute_and_graphics: bool,
    pub vk_timestamp_period_ns: f32,

    /// Frame table for legacy `render_to_target` (no submission context).
    pub legacy_frame_table: Mutex<Option<SharedContextFrameTable>>,
    /// Async FIFO worker for `vkQueueSubmit2` (render thread enqueues, worker runs).
    pub submission_worker: Arc<super::super::submission_worker::SubmissionWorker>,
}

/// A Vulkan command buffer retained for resubmission.
pub(crate) struct RetainedVkCb {
    /// The retained `VkCommandBuffer` (in executable state when GPU has completed).
    pub command_buffer: vk::CommandBuffer,
    /// Bindless slots baked into this command buffer (for slot retirement on resubmit).
    pub used_slots: Vec<SlotKey>,
    /// Frame-table staging row pinned while this CB may re-copy from upload staging.
    pub frame_table_row: Option<u32>,
    /// Timeline value signalled by the most recent submission of this CB.
    /// Used to defer free-listing until the GPU has retired the CB.
    pub last_signal_value: u64,
    /// When true, resubmit on [`LogicalDevice::queue`] under [`LogicalDevice::queue_lock`].
    pub on_graphics_queue: bool,
}

impl LogicalDevice {
    /// Graphics + compute family indices for [`vk::SharingMode::CONCURRENT`], or
    /// `None` when both use the same family (EXCLUSIVE is fine).
    ///
    /// Cross-family resources must be CONCURRENT (or use ownership-transfer barriers).
    /// Goldy uses CONCURRENT so timeline waits alone order access between context
    /// compute queues and the device graphics/present queue.
    pub(crate) fn concurrent_queue_families(&self) -> Option<[u32; 2]> {
        if self.compute_queue_family != self.queue_family {
            Some([self.queue_family, self.compute_queue_family])
        } else {
            None
        }
    }

    pub(crate) fn register_active_compute_queue_lock(&self, lock: Arc<Mutex<()>>) {
        self.active_context_queue_locks.lock().unwrap().push(lock);
    }

    pub(crate) fn unregister_active_compute_queue_lock(&self, lock: &Arc<Mutex<()>>) {
        // Remove one matching entry (shared graphics lock may be registered zero times).
        let mut locks = self.active_context_queue_locks.lock().unwrap();
        if let Some(i) = locks.iter().position(|l| Arc::ptr_eq(l, lock)) {
            locks.swap_remove(i);
        }
    }

    /// `vkMapMemory2KHR` — core in Vulkan 1.4. Struct-based API that replaces `vkMapMemory`.
    pub unsafe fn map_memory2(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> ash::prelude::VkResult<*mut core::ffi::c_void> {
        let info = vk::MemoryMapInfoKHR::default().memory(memory).offset(offset).size(size);
        let mut ptr = core::ptr::null_mut();
        (self.map_memory2.fp().map_memory2_khr)(self.device.handle(), &info, &mut ptr).result_with_success(ptr)
    }

    /// `vkUnmapMemory2KHR` — core in Vulkan 1.4. Returns `VkResult` (unlike legacy `vkUnmapMemory`).
    pub unsafe fn unmap_memory2(&self, memory: vk::DeviceMemory) -> ash::prelude::VkResult<()> {
        let info = vk::MemoryUnmapInfoKHR::default().memory(memory);
        (self.map_memory2.fp().unmap_memory2_khr)(self.device.handle(), &info).result()
    }

    /// `vkDeviceWaitIdle` under all queue submit locks: externally synchronized like queue submits.
    pub(crate) fn device_wait_idle_locked(&self) -> ash::prelude::VkResult<()> {
        let _graphics_guard = self.queue_lock.lock().unwrap();
        let arcs = {
            let mut locks = self.active_context_queue_locks.lock().unwrap().clone();
            locks.sort_by_key(|l| Arc::as_ptr(l) as usize);
            locks
        };
        let mut compute_guards = Vec::with_capacity(arcs.len());
        for arc in &arcs {
            if !Arc::ptr_eq(arc, &self.queue_lock) {
                compute_guards.push(arc.lock().unwrap());
            }
        }
        unsafe { self.device.device_wait_idle() }
    }

    /// Idle the graphics/present queue and every compute queue in the context pool.
    ///
    /// Prefer this over graphics-only `queue_wait_idle` when tearing down surfaces or
    /// other resources that may still be referenced from per-context compute work.
    pub(crate) fn queues_wait_idle_locked(&self) -> ash::prelude::VkResult<()> {
        let _graphics_guard = self.queue_lock.lock().unwrap();
        let arcs = {
            let mut locks = self.active_context_queue_locks.lock().unwrap().clone();
            locks.sort_by_key(|l| Arc::as_ptr(l) as usize);
            locks
        };
        let mut compute_guards = Vec::with_capacity(arcs.len());
        for arc in &arcs {
            if !Arc::ptr_eq(arc, &self.queue_lock) {
                compute_guards.push(arc.lock().unwrap());
            }
        }
        unsafe {
            self.device.queue_wait_idle(self.queue)?;
            for &q in &self.compute_queues {
                self.device.queue_wait_idle(q)?;
            }
        }
        Ok(())
    }

    /// Drain async submits, then wait on all queues under queue submit locks.
    pub(crate) fn synchronized_device_wait_idle(&self) -> ash::prelude::VkResult<()> {
        let _ = self.submission_worker.flush();
        let _ = self.submission_worker.check_error();
        self.device_wait_idle_locked()
    }

    /// Drain async submits, then idle the graphics/present queue under `queue_lock`.
    pub(crate) fn synchronized_queue_wait_idle(&self) -> ash::prelude::VkResult<()> {
        let _ = self.submission_worker.flush();
        let _ = self.submission_worker.check_error();
        let _guard = self.queue_lock.lock().unwrap();
        unsafe { self.device.queue_wait_idle(self.queue) }
    }

    /// `vkQueueSubmit` under worker drain + `queue_lock` (safe vs submission worker).
    pub(crate) fn synchronized_queue_submit(
        &self,
        submit_infos: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> ash::prelude::VkResult<()> {
        let _ = self.submission_worker.flush();
        let _ = self.submission_worker.check_error();
        let _guard = self.queue_lock.lock().unwrap();
        unsafe { self.device.queue_submit(self.queue, submit_infos, fence) }
    }

    /// `vkQueueSubmit2` under worker drain + `queue_lock` (safe vs submission worker).
    pub(crate) fn synchronized_queue_submit2(
        &self,
        submit_infos: &[vk::SubmitInfo2],
        fence: vk::Fence,
    ) -> ash::prelude::VkResult<()> {
        let _ = self.submission_worker.flush();
        let _ = self.submission_worker.check_error();
        let _guard = self.queue_lock.lock().unwrap();
        unsafe { self.device.queue_submit2(self.queue, submit_infos, fence) }
    }

    /// Acquire a primary command buffer from [`Self::command_pool`].
    ///
    /// Recycles from [`Self::free_device_cmd_buffers`] when possible. Allocation and the
    /// free-list share one mutex so concurrent one-shot paths (texture/buffer uploads,
    /// layout transitions) cannot race `vkAllocateCommandBuffers` /
    /// `vkFreeCommandBuffers` on the shared device pool (Vulkan external sync).
    pub(crate) fn acquire_device_cmd_buffer(&self) -> anyhow::Result<vk::CommandBuffer> {
        let mut out = self.allocate_device_cmd_buffers(1)?;
        Ok(out.pop().unwrap())
    }

    /// Acquire `count` primary command buffers from [`Self::command_pool`].
    pub(crate) fn allocate_device_cmd_buffers(&self, count: u32) -> anyhow::Result<Vec<vk::CommandBuffer>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut free = self.free_device_cmd_buffers.lock().unwrap();
        let mut out = Vec::with_capacity(count as usize);
        while out.len() < count as usize {
            if let Some(cb) = free.pop() {
                out.push(cb);
            } else {
                break;
            }
        }
        let need = count as usize - out.len();
        if need > 0 {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(need as u32);
            let cbs = unsafe { self.device.allocate_command_buffers(&alloc_info) }
                .map_err(|e| anyhow::anyhow!("Failed to allocate device command buffers: {e:?}"))?;
            out.extend(cbs);
        }
        Ok(out)
    }

    /// Return a device-pool command buffer for reuse (pool has `RESET_COMMAND_BUFFER`).
    pub(crate) fn recycle_device_cmd_buffer(&self, cb: vk::CommandBuffer) {
        self.free_device_cmd_buffers.lock().unwrap().push(cb);
    }

    /// Return several device-pool command buffers for reuse.
    pub(crate) fn recycle_device_cmd_buffers(&self, cbs: &[vk::CommandBuffer]) {
        self.free_device_cmd_buffers.lock().unwrap().extend_from_slice(cbs);
    }

    /// Free device-pool command buffers under the same lock as allocate/recycle.
    pub(crate) fn free_device_cmd_buffers_now(&self, cbs: &[vk::CommandBuffer]) {
        if cbs.is_empty() {
            return;
        }
        let _guard = self.free_device_cmd_buffers.lock().unwrap();
        unsafe {
            self.device.free_command_buffers(self.command_pool, cbs);
        }
    }
}

/// GPU buffer state.
#[derive(Clone)]
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// Logical byte size (descriptor range, bounds checks).
    pub size: u64,
    /// Bytes reserved in the VkBuffer (`>= size` when oversize allocations are used).
    pub allocation_size: u64,
    /// Index in the global bindless descriptor set (if bindless enabled)
    pub bindless_index: Option<u32>,
    /// Whether this is a storage buffer (vs uniform buffer)
    #[allow(dead_code)]
    pub is_storage: bool,
    /// Structured-buffer / uniform element stride from buffer creation (retained for future validation).
    #[allow(dead_code)]
    pub element_stride: Option<u32>,
    /// HOST_VISIBLE staging buffer for DEVICE_LOCAL storage buffers (CPU upload/readback)
    pub staging_buffer: Option<vk::Buffer>,
    pub staging_memory: Option<vk::DeviceMemory>,
    /// If true, this is a view into another buffer — don't free the VkBuffer/memory on destroy.
    pub is_view: bool,
    /// Set when the buffer was created with [`crate::types::BufferFlags::CPU_READABLE`]:
    /// persistent host mapping of the entire buffer for CPU read/write.
    /// Opaque address of the persistent `map_memory2` region for `CPU_READABLE` storage.
    /// Stored as [`usize`] so `BufferState` is `Send`/`Sync` for `GpuBackend` (raw pointers and
    /// `NonNull` are not in this environment).
    pub host_mapped: Option<usize>,
    /// Mirror of create-time flags.
    pub flags: crate::types::BufferFlags,
    /// Sub-allocated from [`crate::backend::GpuBackend::create_transient_heap`]; `memory` is shared.
    pub transient_heap_suballoc: bool,
    /// Byte offset in parent for buffer views; [`None`] for root buffers.
    pub view_byte_offset: Option<u64>,
    /// `true` for device-local storage using sparse binding (`VkBuffer` sparse flags; no single `memory`).
    pub is_sparse: bool,
    /// Alignment from [`vk::MemoryRequirements::alignment`] when [`Self::is_sparse`]; `0` otherwise.
    pub sparse_block_size: u64,
    /// Per sparse page: physical backing from the page pool (`None` = unbound tile).
    pub sparse_pages: Vec<Option<(vk::DeviceMemory, vk::DeviceSize)>>,
    /// Grant-read staging buffer (host-visible, persistently mapped; no bindless slot).
    pub is_withdraw_staging: bool,
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
    /// Cached compiled vertex shader module
    pub vertex_module: Option<vk::ShaderModule>,
    /// Cached compiled fragment shader module
    pub fragment_module: Option<vk::ShaderModule>,
    /// Cached compiled compute shader module
    pub compute_module: Option<vk::ShaderModule>,
    /// Cached modules for ray-tracing / mesh / amplification stages.
    pub extra_modules: HashMap<crate::slang::SlangStage, vk::ShaderModule>,
    /// Reflection data for bindless rendering (ParameterBlock layouts)
    pub reflection: Option<crate::slang::ShaderReflection>,
    /// Pending struct layout validation on first stage compile; cleared after success.
    pub layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
}

/// Graphics pipeline state.
#[allow(dead_code)]
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    /// Whether this pipeline owns its layout (false when using bindless_pipeline_layout)
    pub owns_layout: bool,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
    /// Per push-constant slot category expectations from shader analysis.
    pub push_constant_categories: Vec<Option<crate::types::ResourceCategory>>,
    /// Per push-constant slot expected element stride (bytes) from reflection.
    pub binding_element_strides: Vec<Option<u32>>,
    /// Human-readable identifier for debugging.
    pub shader_debug_name: String,
    /// True when this PSO is mesh (+ optional task) rather than vertex/fragment.
    pub is_mesh: bool,
}

/// Compute pipeline state.
#[allow(dead_code)]
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    /// Whether this pipeline owns its layout (false when using bindless_pipeline_layout)
    pub owns_layout: bool,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
    /// Per push-constant slot category expectations from shader analysis.
    pub push_constant_categories: Vec<Option<crate::types::ResourceCategory>>,
    /// Per push-constant slot expected element stride (bytes) from reflection.
    pub binding_element_strides: Vec<Option<u32>>,
    /// Human-readable identifier for debugging.
    pub shader_debug_name: String,
}

/// Ray-tracing pipeline plus internally allocated shader-binding table.
pub(crate) struct RayTracingPipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub sbt_buffer: vk::Buffer,
    pub sbt_memory: vk::DeviceMemory,
    pub raygen: vk::StridedDeviceAddressRegionKHR,
    pub miss: vk::StridedDeviceAddressRegionKHR,
    pub hit: vk::StridedDeviceAddressRegionKHR,
    pub callable: vk::StridedDeviceAddressRegionKHR,
    pub push_constant_categories: Vec<Option<crate::types::ResourceCategory>>,
    pub binding_element_strides: Vec<Option<u32>>,
    pub shader_debug_name: String,
}

/// GPU render target state.
pub(crate) struct RenderTargetState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    /// GPU-only render target image
    pub image: vk::Image,
    pub image_memory: vk::DeviceMemory,
    pub image_view: vk::ImageView,
    /// Depth buffer (optional)
    pub depth_format: Option<DepthFormat>,
    pub depth_image: Option<vk::Image>,
    pub depth_memory: Option<vk::DeviceMemory>,
    pub depth_view: Option<vk::ImageView>,
    /// Command buffer for rendering
    pub command_buffer: vk::CommandBuffer,
}

/// GPU texture state.
pub(crate) struct TextureState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub format: TextureFormat,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    /// Staging buffer for texture uploads
    pub staging_buffer: Option<vk::Buffer>,
    pub staging_memory: Option<vk::DeviceMemory>,
    /// Index in the global bindless descriptor set (if bindless enabled).
    /// For `DirectInterpolated` textures this is the storage-image (UAV) slot.
    pub bindless_index: Option<u32>,
    /// For `TextureKind::DirectInterpolated` textures, the sampled-texture (SRV) slot.
    pub sampled_bindless_index: Option<u32>,
    /// Current image layout (for subregion writes / transitions)
    pub current_layout: AtomicI32,
    /// `true` when the image was created with `STORAGE` usage (`TextureKind::Direct` /
    /// `DirectInterpolated`) and *not* sampled-only. Storage images that lack
    /// `SAMPLED` usage must never be transitioned to `SHADER_READ_ONLY_OPTIMAL`
    /// (VUID-VkImageMemoryBarrier2-oldLayout-01211); their settled read layout is
    /// `GENERAL`. Pure-sampled (`Interpolated`) images settle to `SHADER_READ_ONLY_OPTIMAL`.
    pub is_storage_image: bool,
    /// Sub-allocated from a transient heap; `memory` is shared with the heap.
    pub transient_heap_suballoc: bool,
    /// Human-readable name from [`Texture::set_debug_name`], for layout diagnostics.
    pub debug_name: Mutex<Option<String>>,
}

impl TextureState {
    /// The layout this image should rest in after a transfer write makes it
    /// available to shaders. Storage images (created without `SAMPLED` usage in the
    /// pure case) must use `GENERAL`; transitioning them to `SHADER_READ_ONLY_OPTIMAL`
    /// violates VUID-VkImageMemoryBarrier2-oldLayout-01211. Sampled-only images
    /// (`Interpolated`) use `SHADER_READ_ONLY_OPTIMAL`. `DirectInterpolated` is a
    /// storage image whose sampled descriptor is also registered as `GENERAL`, so it
    /// settles to `GENERAL` as well.
    pub fn settled_shader_read_layout(&self) -> vk::ImageLayout {
        if self.is_storage_image {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        }
    }

    pub fn image_layout(&self) -> vk::ImageLayout {
        vk::ImageLayout::from_raw(self.current_layout.load(Ordering::Relaxed))
    }

    pub fn set_image_layout(&self, layout: vk::ImageLayout) {
        self.current_layout.store(layout.as_raw(), Ordering::Relaxed);
    }
}

/// GPU sampler state.
pub(crate) struct SamplerState {
    pub device_handle: DeviceHandle,
    pub sampler: vk::Sampler,
    /// Index in the global bindless descriptor set (if bindless enabled)
    pub bindless_index: Option<u32>,
}

/// GPU acceleration structure (BLAS or TLAS).
pub(crate) struct AccelState {
    pub device_handle: DeviceHandle,
    pub kind: vk::AccelerationStructureTypeKHR,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub as_handle: vk::AccelerationStructureKHR,
    pub device_address: u64,
    pub bindless_index: Option<u32>,
    pub scratch: vk::Buffer,
    pub scratch_memory: vk::DeviceMemory,
    pub max_primitives: u32,
}

/// Maximum number of frames that can be in-flight at once.
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// Per-frame synchronization resources for proper swapchain pipelining.
pub(crate) struct FrameSync {
    pub command_buffer: vk::CommandBuffer,
    /// Recorded fresh each present with ONE_TIME_SUBMIT to copy the per-slot
    /// scratch texture into the acquired swapchain image (scratch→swapchain
    /// blit path). This is WSI work, separate from runtime-owned compute
    /// submissions.
    pub copy_command_buffer: vk::CommandBuffer,
    pub image_available_semaphore: vk::Semaphore,
    /// Signaled by Submit 1 (render work) and consumed by Submit 2 (present
    /// barrier) in the *graphics* (render-pass) present path only.
    pub work_done_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub in_flight_fence: vk::Fence,
    /// True when `in_flight_fence` has been submitted to the queue and not yet waited on.
    /// Only the render (graphics) path submits this fence; the compute path does not.
    /// `acquire()` must check this before calling `wait_for_fences` to avoid hanging
    /// when the slot was last used via the compute path.
    pub fence_pending: bool,
    /// Set after `surface_render` submits the graphics command buffer. Compute-only
    /// presentation uses the scratch-texture copy path in `present` instead (see
    /// `surface::present`).
    pub render_pass_submitted: bool,
    /// Device timeline value signaled for this frame slot's final frame work.
    /// Consumed when presenting.
    pub frame_timeline_value: Option<u64>,
    /// Persistent cache of the last compute timeline value signaled for this frame slot.
    /// Unlike `frame_timeline_value`, this is not consumed when presenting.
    pub last_compute_timeline_value: u64,
    /// Timeline value signaled by the WSI copy submit.  Used by `acquire()` to
    /// ensure the scratch texture's copy-read has completed before the slot is
    /// reused for new compute writes.  Higher than `frame_timeline_value` when
    /// the scratch-texture path is active.
    pub copy_timeline_value: Option<u64>,
}

/// A device-local scratch texture used as the compute render target for one
/// frame slot.  Compute shaders write to this image each frame; at present
/// time it is copied into the acquired swapchain image, decoupling the frame's
/// render phase from WSI image availability entirely.
pub(crate) struct ScratchTextureSlot {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    /// Bindless storage-image handle.  Equal to the index in `TextureState`
    /// so compute shaders can write to this slot without knowing about WSI.
    pub texture_handle: super::TextureHandle,
}

/// Surface (swapchain) state for window presentation.
pub(crate) struct SurfaceState {
    pub device_handle: DeviceHandle,
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    /// One pre-recorded command buffer per swapchain image. Each records a
    /// single `UNDEFINED → GENERAL` barrier for that image. Always using
    /// `UNDEFINED` as `old_layout` lets the driver discard prior contents and
    /// avoids per-frame re-recording. Rebuilt on swapchain recreation.
    pub swapchain_prep_command_buffers: Vec<vk::CommandBuffer>,
    /// Pre-recorded `GENERAL → PRESENT_SRC_KHR` barrier, one per swapchain image.
    /// Used as the second submit in the compute path so the fence can fire
    /// before the present barrier executes.
    pub swapchain_compute_present_command_buffers: Vec<vk::CommandBuffer>,
    /// Pre-recorded `COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR` barrier, one per
    /// swapchain image.  Used as the second submit in the graphics (render) path.
    pub swapchain_render_present_command_buffers: Vec<vk::CommandBuffer>,
    /// Persistent bindless texture handles, one per swapchain image, registered
    /// at swapchain creation. Avoids per-frame `vkCreateImageView` /
    /// `vkUpdateDescriptorSets` / `vkDestroyImageView`. Rebuilt on resize.
    pub swapchain_texture_handles: Vec<super::TextureHandle>,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    /// Desired present mode. When changed by `set_present_mode`, `present_mode_dirty`
    /// is set so `resize()` knows to recreate the swapchain even if dimensions match.
    pub present_mode: vk::PresentModeKHR,
    /// True when `present_mode` was updated but the swapchain has not yet been
    /// recreated with the new mode.
    pub present_mode_dirty: bool,
    /// Depth buffer (when depth_format is Some)
    pub depth_format: Option<DepthFormat>,
    pub depth_image: Option<vk::Image>,
    pub depth_memory: Option<vk::DeviceMemory>,
    pub depth_view: Option<vk::ImageView>,
    /// Current frame index (0..MAX_FRAMES_IN_FLIGHT)
    pub current_frame: usize,
    /// Currently acquired swapchain image index
    pub current_image_index: Option<u32>,
    /// Per-frame synchronization resources
    pub frame_sync: Vec<FrameSync>,
    /// One scratch texture slot per frame slot (`MAX_FRAMES_IN_FLIGHT` entries).
    /// Each slot is lazily created on first acquire and reused every frame.
    /// Compute shaders write to the slot for `current_frame`; at present time
    /// that image is copied into the acquired swapchain image, completely
    /// decoupling GPU rendering from WSI image availability.
    ///
    /// NOTE: making this opt-in / configurable (e.g. for a latency-sensitive
    /// mode that trades throughput for lower frame latency) is future work.
    /// The current design is a max-throughput strategy.
    pub scratch_texture_slots: Vec<Option<ScratchTextureSlot>>,
    /// Handle of the scratch texture for the current frame slot — what compute
    /// shaders write to.  Cleared at present; the underlying slot persists.
    pub current_texture_handle: Option<super::TextureHandle>,
    /// Compute commands accumulated for the active frame during surface submit.
    pub frame_pending_gpu_commands: Vec<super::GpuCommand>,
    /// Drawables acquired or presented but not yet returned to the swapchain pool.
    pub pending_acquire_count: u32,
    /// `(image_index, timeline)` pairs waiting for GPU completion before `SwapchainReturned`.
    pub pending_swapchain_returns: Vec<(u32, crate::timeline::TimelineValue)>,
}

/// Pending buffer operations for command recording.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct PendingBuffer {
    pub buffer: BufferHandle,
    pub slot: u32,
    pub offset: u64,
}

/// GPU buffer objects held until retained-graph pins on reclaimed slots clear.
pub(crate) struct PendingBufferGpuRelease {
    pub retained_slots: Vec<SlotKey>,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub staging_buffer: Option<vk::Buffer>,
    pub staging_memory: Option<vk::DeviceMemory>,
    pub sparse_teardown: Option<SparseBufferTeardown>,
}

/// Deferred sparse teardown (after timeline barrier): unbind + pool recycle + buffer destroy.
#[allow(dead_code)]
pub(crate) struct SparseBufferTeardown {
    pub allocation_size: u64,
    pub block_size: u64,
    pub binds: Vec<(u64, vk::DeviceMemory, vk::DeviceSize)>,
}

/// Resource pending deferred deletion.
/// Resources are kept alive until the device timeline reaches the queued barrier
/// ([`DeletionQueue::queue`]) — see [`crate::timeline`].
#[allow(dead_code)]
pub(crate) enum PendingDeletion {
    Buffer {
        buffer_handle: BufferHandle,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        staging_buffer: Option<vk::Buffer>,
        staging_memory: Option<vk::DeviceMemory>,
        /// When set, unbind sparse tiles and recycle before destroying `buffer`.
        sparse_teardown: Option<SparseBufferTeardown>,
    },
    /// A buffer view whose Vk resources belong to the parent buffer; only the
    /// bindless descriptor index needs deregistration.
    BufferView {
        buffer_handle: BufferHandle,
    },
    /// Previous Vk allocation after [`super::buffer::resize`]; the logical
    /// [`BufferHandle`] stays registered — destroy GPU objects only.
    ReplacedBufferGpu {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        staging_buffer: Option<vk::Buffer>,
        staging_memory: Option<vk::DeviceMemory>,
    },
    /// Previous sparse buffer after resize / migration: unbind pages, recycle pool memory, destroy.
    ReplacedSparseBufferGpu {
        buffer: vk::Buffer,
        allocation_size: u64,
        block_size: u64,
        binds: Vec<(u64, vk::DeviceMemory, vk::DeviceSize)>,
        staging_buffer: Option<vk::Buffer>,
        staging_memory: Option<vk::DeviceMemory>,
    },
    Texture {
        texture_handle: TextureHandle,
        image: vk::Image,
        view: vk::ImageView,
        memory: vk::DeviceMemory,
        staging_buffer: Option<vk::Buffer>,
        staging_memory: Option<vk::DeviceMemory>,
    },
    Sampler {
        sampler: vk::Sampler,
    },
    Accel {
        accel_handle: AccelerationStructureHandle,
        as_handle: vk::AccelerationStructureKHR,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        scratch: vk::Buffer,
        scratch_memory: vk::DeviceMemory,
    },
}

/// Deferred deletion queue for a device.
/// Tracks resources waiting to be deleted after GPU work completes.
pub(crate) struct DeletionQueue {
    inner: super::super::shared::DeferredQueue<TimelineValue, PendingDeletion>,
}

impl DeletionQueue {
    pub fn new() -> Self {
        Self {
            inner: super::super::shared::DeferredQueue::new(),
        }
    }

    /// Queue a resource for deferred deletion once the device timeline reaches `barrier`.
    #[allow(dead_code, reason = "per-context deferred deletion API; device queue used today")]
    pub fn queue(&mut self, barrier: TimelineValue, resource: PendingDeletion) {
        self.inner.push(barrier, resource);
    }

    pub fn drain_up_to(&mut self, completed: TimelineValue) -> Vec<PendingDeletion> {
        self.inner.drain_up_to(completed)
    }

    /// Drain all pending deletions (device teardown).
    pub fn flush_all_drain(&mut self) -> Vec<PendingDeletion> {
        self.inner.flush_all().collect()
    }

    /// Number of resources currently pending deletion.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Device-level deferred deletion queue for resources whose destroy could touch more than
/// one context (bindless-registry-tracked buffers/textures/views).
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

    pub(crate) fn drain_ready(
        &mut self,
        completed_values: &HashMap<super::ContextHandle, u64>,
    ) -> Vec<PendingDeletion> {
        self.inner
            .drain_where(|reqs| slot_requirements_met(reqs, completed_values))
    }

    pub(crate) fn drain_everything(&mut self) -> Vec<PendingDeletion> {
        self.inner.flush_all().collect()
    }

    pub(crate) fn pending_len(&self) -> usize {
        self.inner.len()
    }
}

/// Deferred-delete one entry ([`SparseBufferTeardown`] needs [`LogicalDevice`] for the page pool).
///
/// `registry` must be the already-locked `DescriptorRegistry` from `ld.descriptors`; passing it separately
/// avoids holding the descriptors lock while the caller re-locks it (double-lock hazard).
pub(crate) fn destroy_pending_deletion(
    ld: &LogicalDevice,
    registry: &mut DescriptorRegistry,
    resource: PendingDeletion,
) {
    match resource {
        PendingDeletion::Buffer {
            buffer_handle,
            buffer,
            memory,
            staging_buffer,
            staging_memory,
            sparse_teardown,
        } => {
            let retained_slots = registry.reclaim_buffer_slots(buffer_handle);
            ld.pending_buffer_gpu_releases
                .lock()
                .unwrap()
                .push(PendingBufferGpuRelease {
                    retained_slots,
                    buffer,
                    memory,
                    staging_buffer,
                    staging_memory,
                    sparse_teardown,
                });
        }
        PendingDeletion::BufferView { buffer_handle } => {
            registry.reclaim_buffer_slots(buffer_handle);
        }
        PendingDeletion::Texture { texture_handle, .. } => {
            registry.reclaim_texture_slots(texture_handle);
            destroy_pending_deletion_gpu(ld, resource);
        }
        PendingDeletion::Accel { accel_handle, .. } => {
            registry.reclaim_accel_slots(accel_handle);
            destroy_pending_deletion_gpu(ld, resource);
        }
        other => destroy_pending_deletion_gpu(ld, other),
    }
}

fn destroy_pending_deletion_gpu(ld: &LogicalDevice, resource: PendingDeletion) {
    let device = &ld.device;
    let bind_queue = ld.sparse_binding_queue;
    let mut pool_guard = ld.sparse_page_pool.lock().unwrap();

    unsafe {
        match resource {
            PendingDeletion::Buffer { .. } | PendingDeletion::BufferView { .. } => {}
            PendingDeletion::ReplacedBufferGpu {
                buffer,
                memory,
                staging_buffer,
                staging_memory,
            } => {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                if let Some(buf) = staging_buffer {
                    device.destroy_buffer(buf, None);
                }
                if let Some(mem) = staging_memory {
                    device.free_memory(mem, None);
                }
            }
            PendingDeletion::ReplacedSparseBufferGpu {
                buffer,
                allocation_size: _,
                block_size,
                binds,
                staging_buffer,
                staging_memory,
            } => {
                if !binds.is_empty() {
                    let mut sparse_binds = Vec::with_capacity(binds.len());
                    for (res_off, _mem, _mem_off) in &binds {
                        sparse_binds.push(
                            vk::SparseMemoryBind::default()
                                .resource_offset(*res_off)
                                .size(block_size)
                                .memory(vk::DeviceMemory::default())
                                .memory_offset(0)
                                .flags(vk::SparseMemoryBindFlags::empty()),
                        );
                    }
                    if let Err(e) =
                        super::sparse::queue_bind_sparse_sync(device, &ld.queue_lock, bind_queue, buffer, &sparse_binds)
                    {
                        tracing::warn!(?e, "sparse unbind on replaced buffer failed");
                    }
                    for (_res_off, mem, mem_off) in &binds {
                        if let Some(pool) = pool_guard.as_mut() {
                            pool.free_page(*mem, *mem_off);
                        }
                    }
                }
                device.destroy_buffer(buffer, None);
                if let Some(buf) = staging_buffer {
                    device.destroy_buffer(buf, None);
                }
                if let Some(mem) = staging_memory {
                    device.free_memory(mem, None);
                }
            }
            PendingDeletion::Texture {
                texture_handle: _,
                image,
                view,
                memory,
                staging_buffer,
                staging_memory,
            } => {
                device.destroy_image_view(view, None);
                device.destroy_image(image, None);
                device.free_memory(memory, None);
                if let Some(buf) = staging_buffer {
                    device.destroy_buffer(buf, None);
                }
                if let Some(mem) = staging_memory {
                    device.free_memory(mem, None);
                }
            }
            PendingDeletion::Sampler { sampler } => {
                device.destroy_sampler(sampler, None);
            }
            PendingDeletion::Accel {
                accel_handle: _,
                as_handle,
                buffer,
                memory,
                scratch,
                scratch_memory,
            } => {
                if let Some(khr) = ld.accel_khr.as_ref() {
                    khr.destroy_acceleration_structure(as_handle, None);
                }
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                device.destroy_buffer(scratch, None);
                device.free_memory(scratch_memory, None);
            }
        }
    }
}

fn release_buffer_gpu_resources(ld: &LogicalDevice, entry: PendingBufferGpuRelease) {
    let PendingBufferGpuRelease {
        retained_slots: _,
        buffer,
        memory,
        staging_buffer,
        staging_memory,
        sparse_teardown,
    } = entry;
    let device = &ld.device;
    let bind_queue = ld.sparse_binding_queue;
    let mut pool_guard = ld.sparse_page_pool.lock().unwrap();

    unsafe {
        if let Some(td) = sparse_teardown {
            if !td.binds.is_empty() {
                let mut sparse_binds = Vec::with_capacity(td.binds.len());
                for (res_off, _mem, _mem_off) in &td.binds {
                    sparse_binds.push(
                        vk::SparseMemoryBind::default()
                            .resource_offset(*res_off)
                            .size(td.block_size)
                            .memory(vk::DeviceMemory::default())
                            .memory_offset(0)
                            .flags(vk::SparseMemoryBindFlags::empty()),
                    );
                }
                if let Err(e) =
                    super::sparse::queue_bind_sparse_sync(device, &ld.queue_lock, bind_queue, buffer, &sparse_binds)
                {
                    tracing::warn!(?e, "sparse unbind on buffer destroy failed");
                }
                for (_res_off, mem, mem_off) in &td.binds {
                    if let Some(pool) = pool_guard.as_mut() {
                        pool.free_page(*mem, *mem_off);
                    }
                }
            }
            device.destroy_buffer(buffer, None);
        } else {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        if let Some(buf) = staging_buffer {
            device.destroy_buffer(buf, None);
        }
        if let Some(mem) = staging_memory {
            device.free_memory(mem, None);
        }
    }
}

impl LogicalDevice {
    /// Drain ready device-level deletions without locking the descriptor registry.
    pub(crate) fn drain_deletion_queue_ready(
        &self,
        completed_values: &HashMap<super::ContextHandle, u64>,
    ) -> Vec<PendingDeletion> {
        self.deletion_queue.lock().unwrap().drain_ready(completed_values)
    }

    /// Drop deferred resources once every listed context requirement has retired.
    ///
    /// Locks the deletion queue and descriptor registry internally; takes `&self` so this
    /// can be called through an `Arc<LogicalDevice>` (Phase 5a).
    pub(crate) fn process_deletion_queue_up_to(&self, completed_values: &HashMap<super::ContextHandle, u64>) {
        let drained = self.drain_deletion_queue_ready(completed_values);
        if !drained.is_empty() {
            let descriptors_arc = Arc::clone(&self.descriptors);
            let mut registry = descriptors_arc.lock().unwrap();
            for r in drained {
                destroy_pending_deletion(self, &mut registry, r);
            }
        }
        let ready = {
            let registry = self.descriptors.lock().unwrap();
            self.take_ready_buffer_gpu_releases(&registry)
        };
        for entry in ready {
            release_buffer_gpu_resources(self, entry);
        }
    }

    pub(crate) fn take_ready_buffer_gpu_releases(&self, registry: &DescriptorRegistry) -> Vec<PendingBufferGpuRelease> {
        let mut pending = self.pending_buffer_gpu_releases.lock().unwrap();
        let mut ready = Vec::new();
        let mut i = 0;
        while i < pending.len() {
            if registry.retained_pins_clear(&pending[i].retained_slots) {
                ready.push(pending.swap_remove(i));
            } else {
                i += 1;
            }
        }
        ready
    }

    /// Snapshot live context semaphores and drain the device deletion queue.
    pub(crate) fn process_deletion_queue_for_device(
        &self,
        contexts: &SharedContextMap,
        device_handle: super::DeviceHandle,
    ) {
        let completed_values = snapshot_context_completed_values(&self.device, contexts, device_handle);
        self.process_deletion_queue_up_to(&completed_values);
    }

    /// Drain all pending deletions (device teardown).
    pub(crate) fn flush_deletion_queue(&self) {
        let batch = self.deletion_queue.lock().unwrap().drain_everything();
        if !batch.is_empty() {
            let descriptors_arc = Arc::clone(&self.descriptors);
            let mut registry = descriptors_arc.lock().unwrap();
            for r in batch {
                destroy_pending_deletion(self, &mut registry, r);
            }
        }
        let all = std::mem::take(&mut *self.pending_buffer_gpu_releases.lock().unwrap());
        for entry in all {
            release_buffer_gpu_resources(self, entry);
        }
    }
}

/// `Arc`-wrapped logical device — cloned out of `VulkanState` before dropping the
/// global backend lock so submit paths can hold per-device state without borrowing
/// all of `VulkanState` (Phase 5).
pub(crate) type SharedLogicalDevice = Arc<LogicalDevice>;

/// `Arc<Mutex>`-wrapped per-context state — allows submit paths to lock only the
/// submitting context rather than all of `VulkanState` (Phase 5).
pub(crate) type SharedSubmissionContext = Arc<Mutex<SubmissionContext>>;

/// Per-submit fence and optional command-buffer kept alive until completion.
pub(super) type ComputeFencePoolEntry = (DeviceHandle, vk::Fence, Option<vk::CommandBuffer>);

/// Per-backend non-blocking compute fence pool keyed by submission token.
pub(super) type ComputeFencePool = Mutex<HashMap<u64, ComputeFencePoolEntry>>;

/// Cloned into lock-free submit sessions (Phase 5b-iv).
pub(super) type SharedComputeFencePool = Arc<ComputeFencePool>;

/// Per-context map — read/write independently of the global backend mutex (Phase 5b-iii).
pub(crate) type SharedContextMap = Arc<RwLock<HashMap<super::ContextHandle, SharedSubmissionContext>>>;

/// Per-context frame-table GPU resources — cloned into submit sessions at context creation.
pub(crate) type SharedContextFrameTable = Arc<super::frame_table::ContextFrameTable>;

/// Map plus monotonic handle allocator for a single resource kind.
///
/// Wrapped in [`Arc<RwLock<_>>`] on [`VulkanState`] so submit recording can take read
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

/// Consolidated Vulkan backend state.
/// This holds all the resources and state for the Vulkan backend.
pub(super) struct VulkanState {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_devices: Vec<PhysicalDeviceInfo>,
    pub devices: HashMap<DeviceHandle, SharedLogicalDevice>,
    pub next_device_handle: DeviceHandle,
    pub contexts: SharedContextMap,
    pub next_context_id: super::ContextHandle,
    /// Synthetic context per device for device-queue render epoch stamps.
    pub device_owner_handles: HashMap<super::DeviceHandle, super::ContextHandle>,
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
    /// Per-backend Slang compiler instance (avoids global state issues in tests)
    pub slang_compiler: crate::slang::SlangCompiler,
    /// Per-submission fences for non-blocking compute; token -> (device, `VkFence`, `Option<VkCommandBuffer>`).
    /// The command buffer is kept alive until the fence signals (Vulkan spec: must not free a pending CB).
    pub compute_fence_pool: SharedComputeFencePool,
    /// Set to `true` when any Vulkan call returns `VK_ERROR_DEVICE_LOST`.
    /// Polled by [`GpuBackend::is_device_lost`] without holding any lock.
    pub device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// `true` when `VK_EXT_debug_utils` was loaded (i.e. validation layers are
    /// active).  Guards `set_texture_debug_name` so we never call a null fp.
    pub enable_validation: bool,
    /// Instance-level debug-utils loader (validation messenger).
    pub debug_utils: Option<ash::ext::debug_utils::Instance>,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub validation_sink: Option<std::sync::Arc<super::debug_utils::ValidationSink>>,
}
