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
    BufferHandle, ComputePipelineHandle, DeviceHandle, PipelineHandle, RenderTargetHandle,
    SamplerHandle, ShaderHandle, SurfaceHandle, TextureHandle,
};
use crate::types::{DepthFormat, SamplerDesc, TextureFormat};
use std::collections::HashMap;
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
pub use super::super::shared::{PushLayout, MAX_BINDLESS_SLOTS, TOTAL_PUSH_BYTES};

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
    /// Maps texture handle to UAV offset (for storage textures / SpatialAccess::Direct)
    pub texture_uav_offsets: HashMap<TextureHandle, u32>,
    pub sampler_offsets: HashMap<SamplerHandle, u32>,
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

    /// Register a UAV descriptor for a texture (e.g. storage image / SpatialAccess::Direct).
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

    /// Get the SRV offset for a buffer (for read-only access to storage buffers)
    pub fn get_buffer_srv_offset(&self, handle: BufferHandle) -> Option<u32> {
        self.buffer_srv_offsets.get(&handle).copied()
    }

    /// Allocate a raw descriptor slot not tied to any resource handle.
    /// Used for permanent device-lifetime slots that need a reserved CBV/SRV/UAV index.
    pub fn alloc_cbv_srv_uav_slot(&mut self) -> u32 {
        self.cbv_srv_uav.alloc()
    }

    /// Unregister a buffer, returning all of its descriptor slots to the free list.
    ///
    /// Storage buffers may occupy two slots (UAV primary + SRV secondary); both are
    /// recycled here. Without this, per-frame buffer churn exhausts the 16 384-slot
    /// heap in seconds and subsequent descriptor writes corrupt adjacent entries.
    pub fn unregister_buffer(&mut self, handle: BufferHandle) {
        if let Some(offset) = self.buffer_offsets.remove(&handle) {
            self.cbv_srv_uav.free(offset);
        }
        if let Some(offset) = self.buffer_srv_offsets.remove(&handle) {
            self.cbv_srv_uav.free(offset);
        }
    }

    /// Unregister a texture, returning all of its descriptor slots to the free list.
    ///
    /// Storage textures may occupy two slots (SRV + UAV); both are recycled here.
    pub fn unregister_texture(&mut self, handle: TextureHandle) {
        if let Some(offset) = self.texture_offsets.remove(&handle) {
            self.cbv_srv_uav.free(offset);
        }
        if let Some(offset) = self.texture_uav_offsets.remove(&handle) {
            self.cbv_srv_uav.free(offset);
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
    pub fn available_slots(&self, category: crate::types::BindlessCategory) -> u32 {
        match category {
            crate::types::BindlessCategory::Sampler => {
                MAX_BINDLESS_SAMPLERS.saturating_sub(self.sampler.live_count())
            }
            _ => MAX_BINDLESS_CBV_SRV_UAV.saturating_sub(self.cbv_srv_uav.live_count()),
        }
    }

    /// Maximum slots for the given category.
    pub fn max_slots(category: crate::types::BindlessCategory) -> u32 {
        match category {
            crate::types::BindlessCategory::Sampler => MAX_BINDLESS_SAMPLERS,
            _ => MAX_BINDLESS_CBV_SRV_UAV,
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod registry_tests {
    use super::*;

    /// Simulate the per-frame create/destroy churn that ekrano generates for transient
    /// pool-view buffers. The counter must stay bounded — well below MAX_BINDLESS_CBV_SRV_UAV
    /// — even after far more iterations than the heap limit.
    #[test]
    fn buffer_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as BufferHandle;
            reg.register_buffer_uav(handle);
            reg.unregister_buffer(handle);
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
            reg.unregister_buffer(handle);
        }
        assert_eq!(
            reg.cbv_srv_uav.next_fresh(),
            2,
            "counter should only have advanced twice (one UAV + one SRV slot ever minted)"
        );
        assert_eq!(
            reg.cbv_srv_uav.free_count(),
            2,
            "both slots must be in the free list"
        );
    }

    /// Textures with both SRV and UAV views (storage textures) must recycle both slots.
    #[test]
    fn texture_dual_slot_recycled() {
        let mut reg = ResourceRegistry::new();
        for i in 0..1_000u64 {
            let handle = i as TextureHandle;
            reg.register_texture(handle);
            reg.register_texture_uav(handle);
            reg.unregister_texture(handle);
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
        let mut slots: Vec<u32> = (0..N)
            .map(|i| reg.register_buffer_uav(i as BufferHandle))
            .collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(
            slots.len(),
            N as usize,
            "duplicate slots assigned to live resources"
        );
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
            reg.unregister_buffer((i - LIVE) as BufferHandle);
            reg.register_buffer_uav(i as BufferHandle);
        }
        assert!(
            reg.cbv_srv_uav.next_fresh() <= LIVE as u32,
            "counter ({}) exceeded live count ({LIVE}); slot recycling broken",
            reg.cbv_srv_uav.next_fresh()
        );
    }
}

/// Information about a physical DXGI adapter.
/// Named DxgiAdapterInfo to avoid conflict with super::AdapterInfo.
#[allow(dead_code)]
pub(crate) struct DxgiAdapterInfo {
    pub adapter: Dxgi::IDXGIAdapter1,
    pub desc: Dxgi::DXGI_ADAPTER_DESC1,
    pub adapter_id: u32,
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
}

/// A retained (closed but not reset) command list available for re-execution.
///
/// DX12 allows re-executing a closed command list via `ExecuteCommandLists`
/// without calling `Reset`, as long as the backing allocator is not reset first.
/// This enables zero-recording-cost re-submission for static scenes.
pub(crate) struct RetainedGraph {
    /// Scene fingerprint that uniquely identifies this command list's content.
    pub fingerprint: u64,
    /// The closed (but not reset) command list.  Cloned from the pool slot so both
    /// the pool and this struct hold a reference-counted pointer to the same COM object.
    pub command_list: Direct3D12::ID3D12GraphicsCommandList,
    /// Index into `LogicalDevice::compute_allocator_pool` for the backing allocator slot.
    pub slot_idx: usize,
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
    pub fence: Direct3D12::ID3D12Fence,
    pub fence_value: u64,

    // Bindless infrastructure
    /// `true` when adapter reports tiled resources tier >= 1 (buffer reserved resources).
    pub supports_reserved_buffers: bool,
    pub tile_heap_pool: Option<super::tiles::TileHeapPool>,
    /// Shared root signature for all bindless pipelines (graphics and compute)
    pub bindless_root_signature: Option<Direct3D12::ID3D12RootSignature>,
    /// Command signature for indirect compute dispatch (ExecuteIndirect, dispatch-only)
    pub compute_dispatch_indirect_signature: Option<Direct3D12::ID3D12CommandSignature>,
    /// Command signature for batched dispatch: sets push constants then dispatches.
    /// Each argument entry contains `[PushLayout (TOTAL_PUSH_BYTES)] [wg_x] [wg_y] [wg_z]`.
    /// Requires the shared `bindless_root_signature` to be non-None.
    pub compute_batch_dispatch_signature: Option<Direct3D12::ID3D12CommandSignature>,
    /// Registry tracking resource offsets in descriptor heaps
    pub resource_registry: ResourceRegistry,
    /// Pool of command allocators for non-blocking compute submission.
    /// Slots can be reused when fence signals completion.
    pub compute_allocator_pool: Vec<ComputeAllocatorSlot>,
    /// A single retained command list for zero-recording-cost re-submission.
    /// `None` until a `submit_graph_and_retain` call stores a closed CL here.
    pub retained_graph: Option<RetainedGraph>,
    /// Device-lifetime zero-filled UPLOAD-heap buffer used as the source for
    /// `CopyBufferRegion` clears. One buffer per device; clears of any size are
    /// handled by chunking `CopyBufferRegion` calls. Using a copy instead of
    /// `ClearUnorderedAccessViewUint` avoids the shared-descriptor aliasing hazard
    /// that caused silent corruption on WARP when multiple buffers were cleared in
    /// the same wave (all clears rewrote the same single-slot descriptor heap).
    pub zero_buffer: Direct3D12::ID3D12Resource,
    /// Deferred deletion queue — resources are dropped only after the GPU finishes
    /// the command list that was last submitted when the resource was queued.
    pub deletion_queue: DeletionQueue,
    /// Cached graphics PSO blobs from `ID3D12PipelineState::GetCachedBlob` (cold-load via `CachedPSO`).
    pub graphics_pso_blobs: HashMap<u64, Vec<u8>>,
    /// Cached compute PSO blobs.
    pub compute_pso_blobs: HashMap<u64, Vec<u8>>,
    /// `true` if either blob map changed since loading from [`super::pso_cache`] disk file.
    pub pso_disk_cache_dirty: bool,
}

impl LogicalDevice {
    pub(crate) fn process_deletion_queue(&mut self) {
        let completed = unsafe { self.fence.GetCompletedValue() };
        let batch = self.deletion_queue.drain_up_to_completed(completed);
        for resource in batch {
            destroy_pending_deletion(self, resource);
        }
    }

    pub(crate) fn flush_deletion_queue(&mut self) {
        let batch = self.deletion_queue.drain_everything();
        for resource in batch {
            destroy_pending_deletion(self, resource);
        }
    }
}

pub(crate) fn destroy_pending_deletion(ld: &mut LogicalDevice, resource: PendingDeletion) {
    match resource {
        PendingDeletion::Buffer {
            buffer_handle,
            resource,
            upload_buffer,
            coherent_readback,
            reserved_tiles,
        } => {
            ld.resource_registry.unregister_buffer(buffer_handle);
            if let Some(tiles) = reserved_tiles {
                super::tiles::teardown_reserved_mappings(
                    &ld.command_queue,
                    &mut ld.tile_heap_pool,
                    &resource,
                    &tiles,
                );
            }
            // Flush the queue so the unmap is processed before releasing the resource.
            // Without this, the driver can crash (device removal) on Release.
            let fv = ld.fence_value;
            ld.fence_value += 1;
            if unsafe { ld.command_queue.Signal(&ld.fence, fv) }.is_ok() {
                let _ = super::utils::wait_for_fence(&ld.fence, fv);
            }
            drop(resource);
            drop(upload_buffer);
            drop(coherent_readback);
        }
        PendingDeletion::BufferView { buffer_handle } => {
            ld.resource_registry.unregister_buffer(buffer_handle);
        }
        PendingDeletion::ReplacedBufferGpu {
            resource,
            upload_buffer,
            coherent_readback,
        } => {
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
            super::tiles::teardown_reserved_mappings(
                &ld.command_queue,
                &mut ld.tile_heap_pool,
                &resource,
                &tiles,
            );
            // Flush the queue so the unmap is processed before releasing the resource.
            // Without this, the driver can crash (device removal) on Release.
            let fv = ld.fence_value;
            ld.fence_value += 1;
            if unsafe { ld.command_queue.Signal(&ld.fence, fv) }.is_ok() {
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
            ld.resource_registry.unregister_texture(texture_handle);
            drop(resource);
        }
        PendingDeletion::StandaloneResource(resource) => {
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
    /// storage buffers. Copied UAV → READBACK by [`super::buffer::read_to_cpu`].
    pub coherent_readback: Option<Direct3D12::ID3D12Resource>,
    /// Persistent map of `coherent_readback` (see above).
    /// Persistent `Map` result address for the readback resource (`usize` for `Send`/`Sync`).
    pub coherent_readback_mapped: Option<usize>,
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
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
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
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
    /// Per push-constant slot expected element stride (bytes) from reflection.
    pub binding_element_strides: Vec<Option<u32>>,
    /// Human-readable identifier used in category-mismatch error messages.
    pub shader_debug_name: String,
}

/// GPU render target state with optional staging for CPU readback.
#[allow(dead_code)]
pub(crate) struct RenderTargetState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// GPU-only render target texture
    pub texture: Direct3D12::ID3D12Resource,
    /// RTV descriptor handle offset
    pub rtv_offset: u32,
    /// Depth buffer (optional)
    pub depth_format: Option<DepthFormat>,
    pub depth_texture: Option<Direct3D12::ID3D12Resource>,
    pub dsv_offset: Option<u32>,
    /// Staging buffer for CPU readback (lazy-created on first read)
    pub staging_buffer: Option<Direct3D12::ID3D12Resource>,
    /// Command list for rendering
    pub command_list: Direct3D12::ID3D12GraphicsCommandList7,
    /// Track if we've rendered (for readback validation)
    pub has_rendered: bool,
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
    /// For `SpatialAccess::DirectInterpolated` textures, the SRV (sampled-texture) slot.
    pub sampled_bindless_offset: Option<u32>,
    /// Last known layout for enhanced texture barriers (replaces legacy `current_state`).
    pub last_layout: Direct3D12::D3D12_BARRIER_LAYOUT,
    /// Whether this texture was created with UAV access (SpatialAccess::Direct).
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

/// Maximum number of frames that can be in-flight at once.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

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
}

/// Consolidated DX12 backend state.
/// This holds all the resources and state for the DX12 backend.
pub(super) struct Dx12State {
    pub factory: Dxgi::IDXGIFactory4,
    /// Whether the DXGI factory/driver supports `DXGI_PRESENT_ALLOW_TEARING`
    /// (needed for tear-free immediate presentation in windowed mode).
    pub allow_tearing: bool,
    pub adapters: Vec<DxgiAdapterInfo>,
    pub devices: HashMap<DeviceHandle, LogicalDevice>,
    pub next_device_handle: DeviceHandle,
    pub buffers: HashMap<BufferHandle, BufferState>,
    pub next_buffer_handle: BufferHandle,
    pub shaders: HashMap<ShaderHandle, ShaderState>,
    pub next_shader_handle: ShaderHandle,
    pub pipelines: HashMap<PipelineHandle, PipelineState>,
    pub next_pipeline_handle: PipelineHandle,
    pub compute_pipelines: HashMap<ComputePipelineHandle, ComputePipelineState>,
    pub next_compute_pipeline_handle: ComputePipelineHandle,
    pub render_targets: HashMap<RenderTargetHandle, RenderTargetState>,
    pub next_render_target_handle: RenderTargetHandle,
    pub surfaces: HashMap<SurfaceHandle, SurfaceState>,
    pub next_surface_handle: SurfaceHandle,
    pub textures: HashMap<TextureHandle, TextureState>,
    pub next_texture_handle: TextureHandle,
    pub samplers: HashMap<SamplerHandle, SamplerState>,
    pub next_sampler_handle: SamplerHandle,
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
    /// Per-device upload belts for `GpuCommand::WriteBuffer`.
    pub(super) staging_belts: HashMap<DeviceHandle, super::staging::StagingBelt>,
    /// Set to `true` when a TDR / device-removal is detected (fence completed with `u64::MAX`
    /// or `GetDeviceRemovedReason` returns a non-ok HRESULT).
    /// Polled by [`GpuBackend::is_device_lost`] without holding any lock.
    pub device_removed: std::sync::atomic::AtomicBool,
}
