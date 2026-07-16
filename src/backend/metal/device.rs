//! Device management logic.

use super::super::DeviceHandle;
use super::types::{
    DeletionQueue, DescriptorRegistry, HeapAllocator, LogicalDevice, MetalAdapterInfo, MetalState,
    TextureHeapAllocator, ARGUMENT_BUFFER_SIZE,
};
use crate::backend::{AdapterInfo, BackendType, DeviceType};
use ::metal as mtl;
use anyhow::{Context, Result};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

/// Initial heap size for both the buffer and texture heaps.
///
/// TODO(#heap-config): This is a hardcoded starting point. It should be tuned
/// based on `MTLDevice.recommendedMaxWorkingSetSize` or per-device heuristics.
/// For a 16 GB M-series Mac the primary heap could be bumped to 256 MB to reduce
/// the likelihood of overflow allocations. See types.rs `MAX_HEAP_SIZE` for the
/// upper bound.
const INITIAL_HEAP_SIZE: u64 = 64 * 1024 * 1024;
use mtl::{
    Device as MTLDevice, HeapDescriptor, MTLCPUCacheMode, MTLHazardTrackingMode, MTLHeapType, MTLResourceOptions,
    MTLStorageMode,
};

/// Enumerate available Metal devices/adapters.
pub(super) fn enumerate(adapters: &[MetalAdapterInfo]) -> Vec<AdapterInfo> {
    adapters
        .iter()
        .map(|entry| {
            let device = &entry.device;
            let name = device.name().to_string();
            let device_type = if device.is_low_power() {
                DeviceType::IntegratedGpu
            } else {
                DeviceType::DiscreteGpu
            };

            AdapterInfo {
                id: entry.adapter_id,
                name,
                vendor: "Apple".to_string(),
                backend: BackendType::Metal,
                device_type,
            }
        })
        .collect()
}

/// Build the public capability snapshot for a physical adapter.
pub(super) fn adapter_capabilities(_adapter_id: u32) -> crate::device::DeviceCapabilities {
    crate::device::DeviceCapabilities {
        has_zero_copy_storage_readback: true,
        buffer_resize_cost: crate::types::BufferResizeCost::Constant,
        buffer_page_size: 16 * 1024,
        buffer_decommit_supported: true,
        host_sidecar_on_submit_worker: true,
        ..crate::device::DeviceCapabilities::default()
    }
}

/// Create a logical device from an adapter ID.
pub(super) fn create(state: &mut MetalState, adapter_id: u32) -> Result<DeviceHandle> {
    let mtl_device = state
        .adapters
        .iter()
        .find(|a| a.adapter_id == adapter_id)
        .map(|a| a.device.clone())
        .or_else(|| MTLDevice::all().get(adapter_id as usize).cloned())
        .or_else(MTLDevice::system_default)
        .context("No Metal device available")?;
    let device = mtl_device;

    let command_queue = device.new_command_queue();

    // Require Argument Buffers Tier 2.
    // Supported on: Apple Silicon (all), Intel Macs 2017+, AMD GPUs 2015+.
    anyhow::ensure!(
        device.argument_buffers_support() == mtl::MTLArgumentBuffersTier::Tier2,
        "Metal Argument Buffers Tier 2 is required but not supported on this GPU. \
         Apple Silicon, Intel 2017+, and AMD 2015+ are all supported."
    );
    tracing::info!("Metal Argument Buffers Tier 2 confirmed — bindless enabled");

    // Create global argument buffer (stores resource device pointers)
    let argument_buffer = device.new_buffer(ARGUMENT_BUFFER_SIZE, MTLResourceOptions::StorageModeShared);
    tracing::info!("Created argument buffer");

    // Create heaps for resource allocation.
    // Use Shared storage so the CPU can write via replace_region() / contents().
    // IMPORTANT: CPU cache mode must match between heap and buffer allocation.
    let heap_size = INITIAL_HEAP_SIZE;
    let (heap_allocator, texture_heap) = create_heaps(&device, heap_size);

    let (argument_encoder, texture_encoder, storage_image_encoder, sampler_encoder) = create_argument_encoders(&device);

    let frame_table =
        super::frame_table::MetalFrameTable::init(device.as_ref(), argument_buffer.as_ref(), argument_encoder.as_ref());

    let handle = state.next_device_handle;
    state.next_device_handle += 1;

    tracing::info!(
        "Created Metal device {} for adapter {} ({})",
        handle,
        adapter_id,
        device.name(),
    );

    let ld = Arc::new(LogicalDevice {
        device,
        command_queue,
        heap_allocator: Mutex::new(heap_allocator),
        texture_heap: Mutex::new(texture_heap),
        argument_buffer,
        argument_encoder,
        texture_encoder,
        storage_image_encoder,
        sampler_encoder,
        frame_table: Mutex::new(frame_table),
        descriptors: Arc::new(Mutex::new(DescriptorRegistry::new())),
        timeline_next: Arc::new(AtomicU64::new(1)),
        timeline_scheduled_max: AtomicU64::new(0),
        retired_floor: AtomicU64::new(0),
        deletion_queue: Mutex::new(DeletionQueue::new()),
        queue_lock: Arc::new(Mutex::new(())),
        submission_worker: Arc::new(crate::backend::submission_worker::SubmissionWorker::new(
            crate::backend::submission_worker::SUBMISSION_QUEUE_CAPACITY,
        )),
    });

    super::frame_table::init_device(ld.as_ref());
    state.devices.insert(handle, ld);

    Ok(handle)
}

/// Create the buffer and texture heap allocators for a logical device.
fn create_heaps(device: &MTLDevice, heap_size: u64) -> (HeapAllocator, TextureHeapAllocator) {
    tracing::info!("Creating buffer heap allocator...");
    let buffer_heap_desc = HeapDescriptor::new();
    buffer_heap_desc.set_size(heap_size);
    buffer_heap_desc.set_storage_mode(MTLStorageMode::Shared);
    buffer_heap_desc.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);
    buffer_heap_desc.set_heap_type(MTLHeapType::Automatic);
    // Tracked (not Untracked): switching to Untracked was investigated as a
    // potential perf win (avoiding Metal's implicit hazard-tracking overhead)
    // but turned out to break frame pipelining in practice.
    //
    // The explicit synchronisation Untracked requires is:
    //   • MTLFence at blit↔compute encoder boundaries within a command buffer
    //   • MTLSharedEvent waits between standalone blit CBs (buffer writes/clears)
    //     and the following compute CB, to ensure cache coherency
    //   • A per-resource "written buffers" annotation that lets consecutive
    //     compute CBs skip the event wait when they share no written resources
    //
    // All Goldy integration tests pass with that scheme, but Ekrano still
    // regresses (~185 FPS vs ~200 FPS Tracked baseline) for two reasons:
    //   1. Intra-graph partition waits: emit_partitioned_commands splits large
    //      graphs into two CBs submitted back-to-back; the second CB must wait
    //      for the first via MTLSharedEvent (MTLFence can't cross CB boundaries),
    //      which serialises consecutive partitions and eliminates GPU pipelining.
    //      Metal's hardware hazard tracking avoids this cost entirely.
    //   2. Encoder-split overhead: every ResourceBarrier forces end_encoding +
    //      new_compute_command_encoder + use_heaps_for_compute, which is paid
    //      once per wave rather than once per CB submission.
    //
    // The net result is that the synchronisation overhead for Untracked exceeds
    // the savings from bypassing implicit hazard tracking for this workload.
    // Revisit if Apple exposes a lighter cross-CB barrier primitive, or if
    // Ekrano moves to single-CB submission for its graphs.
    buffer_heap_desc.set_hazard_tracking_mode(MTLHazardTrackingMode::Tracked);
    let buffer_heap = device.new_heap(&buffer_heap_desc);
    let heap_allocator = HeapAllocator::new(device.clone(), buffer_heap, heap_size);
    tracing::info!("Created buffer heap allocator (primary={}MB)", heap_size / 1024 / 1024);

    tracing::info!("Creating texture heap...");
    let texture_heap_desc = HeapDescriptor::new();
    texture_heap_desc.set_size(heap_size);
    texture_heap_desc.set_storage_mode(MTLStorageMode::Shared);
    texture_heap_desc.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);
    texture_heap_desc.set_heap_type(MTLHeapType::Automatic);
    // Tracked: same rationale as the buffer heap above. Textures are an even
    // harder case for Untracked — render-target hazards (coarse→fine, ping-pong
    // filter layers, swapchain copies) span multiple command buffers, and
    // MTLSharedEvent ordering alone did not restore correctness there.
    texture_heap_desc.set_hazard_tracking_mode(MTLHazardTrackingMode::Tracked);
    let texture_heap_raw = device.new_heap(&texture_heap_desc);
    let texture_heap = TextureHeapAllocator::new(device.clone(), texture_heap_raw, heap_size);
    tracing::info!("Created texture heap (size={}MB)", heap_size / 1024 / 1024);

    (heap_allocator, texture_heap)
}

/// Create the four argument encoders used for buffers, sampled textures, storage images,
/// and samplers.
pub(super) fn create_argument_encoders(
    device: &MTLDevice,
) -> (
    mtl::ArgumentEncoder,
    mtl::ArgumentEncoder,
    mtl::ArgumentEncoder,
    mtl::ArgumentEncoder,
) {
    // Encoder for raw buffer pointers (RWStructuredBuffer / StructuredBuffer).
    let buffer_arg_desc = mtl::ArgumentDescriptor::new();
    buffer_arg_desc.set_index(0);
    buffer_arg_desc.set_data_type(mtl::MTLDataType::Pointer);
    buffer_arg_desc.set_access(mtl::MTLArgumentAccess::ReadWrite);
    let argument_encoder = device.new_argument_encoder(mtl::Array::from_slice(&[buffer_arg_desc]));
    tracing::info!(
        "Created buffer ArgumentEncoder (encoded_length={})",
        argument_encoder.encoded_length()
    );

    // Encoder for sampled textures (read-only `Texture2D` / `SampledSpatial`).
    let texture_arg_desc = mtl::ArgumentDescriptor::new();
    texture_arg_desc.set_index(0);
    texture_arg_desc.set_data_type(mtl::MTLDataType::Texture);
    texture_arg_desc.set_texture_type(mtl::MTLTextureType::D2);
    texture_arg_desc.set_access(mtl::MTLArgumentAccess::ReadOnly);
    let texture_encoder = device.new_argument_encoder(mtl::Array::from_slice(&[texture_arg_desc]));
    tracing::info!(
        "Created texture ArgumentEncoder (encoded_length={})",
        texture_encoder.encoded_length()
    );

    // Encoder for storage images (read-write `RWTexture2D` / `DirectSpatial`).
    // Must be ReadWrite — a ReadOnly encoder causes a GPU page fault on the first
    // compute shader write dispatch.
    let storage_image_arg_desc = mtl::ArgumentDescriptor::new();
    storage_image_arg_desc.set_index(0);
    storage_image_arg_desc.set_data_type(mtl::MTLDataType::Texture);
    storage_image_arg_desc.set_texture_type(mtl::MTLTextureType::D2);
    storage_image_arg_desc.set_access(mtl::MTLArgumentAccess::ReadWrite);
    let storage_image_encoder = device.new_argument_encoder(mtl::Array::from_slice(&[storage_image_arg_desc]));
    tracing::info!(
        "Created storage image ArgumentEncoder (encoded_length={})",
        storage_image_encoder.encoded_length()
    );

    // Encoder for samplers.  Samplers are read-only by nature; the access field
    // is ignored by Metal for sampler descriptors but we set ReadOnly for clarity.
    let sampler_arg_desc = mtl::ArgumentDescriptor::new();
    sampler_arg_desc.set_index(0);
    sampler_arg_desc.set_data_type(mtl::MTLDataType::Sampler);
    sampler_arg_desc.set_access(mtl::MTLArgumentAccess::ReadOnly);
    let sampler_encoder = device.new_argument_encoder(mtl::Array::from_slice(&[sampler_arg_desc]));
    let sampler_stride = sampler_encoder.encoded_length();
    tracing::info!("Created sampler ArgumentEncoder (encoded_length={})", sampler_stride);

    // Every resource category in the argument buffer is laid out as
    // MAX_RESOURCES_PER_CATEGORY × 8 bytes.  ARGUMENT_BUFFER_SIZE is derived
    // from that assumption.  Fail loudly at device creation if this GPU returns
    // a different stride so that sampler offsets never silently diverge.
    assert_eq!(
        sampler_stride, 8,
        "Metal sampler argument encoder reported encoded_length={sampler_stride}, \
         expected 8 (MTLResourceID size on Apple Silicon / Intel / AMD). \
         ARGUMENT_BUFFER_SIZE and the sampler slot layout assume an 8-byte stride; \
         please update ARGUMENT_BUFFER_SIZE and GPU_RESOURCE_STRIDE to match."
    );

    (
        argument_encoder,
        texture_encoder,
        storage_image_encoder,
        sampler_encoder,
    )
}

/// Destroy a logical device and clean up resources owned by it.
pub(super) fn destroy(state: &mut MetalState, device_handle: DeviceHandle) {
    if let Some(ld) = state.devices.remove(&device_handle) {
        let _ = ld.submission_worker.flush();
        ld.deletion_queue.lock().unwrap().flush_all();
        state.buffers.retain(|_, b| b.device_handle != device_handle);
        state.shaders.retain(|_, s| s.device_handle != device_handle);
        state.pipelines.retain(|_, p| p.device_handle != device_handle);
        state.compute_pipelines.retain(|_, p| p.device_handle != device_handle);
        state.render_targets.retain(|_, t| t.device_handle != device_handle);
        state.surfaces.retain(|_, s| s.device_handle != device_handle);
        state.textures.retain(|_, t| t.device_handle != device_handle);
        state.samplers.retain(|_, s| s.device_handle != device_handle);

        tracing::info!("Destroyed Metal device {}", device_handle);
    }
}

/// Check if a device handle is valid.
pub(super) fn is_valid(state: &MetalState, device_handle: DeviceHandle) -> bool {
    state.devices.contains_key(&device_handle)
}
