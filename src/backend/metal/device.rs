//! Device management logic.

use super::super::DeviceHandle;
use super::staging::{StagingBelt, TextureStagingPool, DEFAULT_STAGING_CHUNK_SIZE};
use super::types::{
    DeletionQueue, HeapAllocator, LogicalDevice, MetalState, ResourceRegistry,
    TextureHeapAllocator, TimelineWaiter, ARGUMENT_BUFFER_SIZE,
};
use crate::backend::{AdapterInfo, BackendType, DeviceType};
use ::metal as mtl;
use anyhow::{Context, Result};

/// Initial heap size for both the buffer and texture heaps.
///
/// TODO(#heap-config): This is a hardcoded starting point. It should be tuned
/// based on `MTLDevice.recommendedMaxWorkingSetSize` or per-device heuristics.
/// For a 16 GB M-series Mac the primary heap could be bumped to 256 MB to reduce
/// the likelihood of overflow allocations. See types.rs `MAX_HEAP_SIZE` for the
/// upper bound.
const INITIAL_HEAP_SIZE: u64 = 64 * 1024 * 1024;
use mtl::{
    Device as MTLDevice, HeapDescriptor, MTLCPUCacheMode, MTLHazardTrackingMode, MTLHeapType,
    MTLResourceOptions, MTLStorageMode,
};

/// Enumerate available Metal devices/adapters.
pub(super) fn enumerate() -> Vec<AdapterInfo> {
    let devices = MTLDevice::all();
    devices
        .iter()
        .enumerate()
        .map(|(idx, device)| {
            let name = device.name().to_string();
            let device_type = if device.is_low_power() {
                DeviceType::IntegratedGpu
            } else {
                DeviceType::DiscreteGpu
            };

            AdapterInfo {
                id: idx as u32,
                name,
                vendor: "Apple".to_string(),
                backend: BackendType::Metal,
                device_type,
            }
        })
        .collect()
}

/// Create a logical device from an adapter ID.
pub(super) fn create(state: &mut MetalState, adapter_id: u32) -> Result<DeviceHandle> {
    let devices = MTLDevice::all();
    let device = devices
        .get(adapter_id as usize)
        .cloned()
        .or_else(MTLDevice::system_default)
        .context("No Metal device available")?;

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
    let argument_buffer =
        device.new_buffer(ARGUMENT_BUFFER_SIZE, MTLResourceOptions::StorageModeShared);
    tracing::info!("Created argument buffer");

    // Create heaps for resource allocation.
    // Use Shared storage so the CPU can write via replace_region() / contents().
    // IMPORTANT: CPU cache mode must match between heap and buffer allocation.
    let heap_size = INITIAL_HEAP_SIZE;
    let (heap_allocator, texture_heap) = create_heaps(&device, heap_size);

    let (argument_encoder, texture_encoder, storage_image_encoder) =
        create_argument_encoders(&device);

    let handle = state.next_device_handle;
    state.next_device_handle += 1;

    tracing::info!(
        "Created Metal device {} for adapter {} ({})",
        handle,
        adapter_id,
        device.name(),
    );

    let timeline_event = device.new_shared_event();
    let signal_queue = std::sync::Arc::new(crate::signal::SignalQueue::new());
    let timeline_waiter = TimelineWaiter::new_with_signals(std::sync::Arc::clone(&signal_queue));

    state.devices.insert(
        handle,
        LogicalDevice {
            device,
            command_queue,
            heap_allocator,
            texture_heap,
            argument_buffer,
            argument_encoder,
            texture_encoder,
            storage_image_encoder,
            resource_registry: ResourceRegistry::new(),
            timeline_event,
            timeline_waiter,
            signal_queue,
            pending_swapchain_returns: Mutex::new(Vec::new()),
            timeline_next: 1,
            timeline_scheduled_max: 0,
            deletion_queue: DeletionQueue::new(),
            last_committed_timeline: None,
            staging_belt: StagingBelt::new(DEFAULT_STAGING_CHUNK_SIZE),
            texture_staging_pool: TextureStagingPool::new(),
            in_flight_command_buffers: std::collections::VecDeque::new(),
        },
    );

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
    buffer_heap_desc.set_hazard_tracking_mode(MTLHazardTrackingMode::Tracked);
    let buffer_heap = device.new_heap(&buffer_heap_desc);
    let heap_allocator = HeapAllocator::new(device.clone(), buffer_heap, heap_size);
    tracing::info!(
        "Created buffer heap allocator (primary={}MB)",
        heap_size / 1024 / 1024
    );

    tracing::info!("Creating texture heap...");
    let texture_heap_desc = HeapDescriptor::new();
    texture_heap_desc.set_size(heap_size);
    texture_heap_desc.set_storage_mode(MTLStorageMode::Shared);
    texture_heap_desc.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);
    texture_heap_desc.set_heap_type(MTLHeapType::Automatic);
    texture_heap_desc.set_hazard_tracking_mode(MTLHazardTrackingMode::Tracked);
    let texture_heap_raw = device.new_heap(&texture_heap_desc);
    let texture_heap = TextureHeapAllocator::new(device.clone(), texture_heap_raw, heap_size);
    tracing::info!("Created texture heap (size={}MB)", heap_size / 1024 / 1024);

    (heap_allocator, texture_heap)
}

/// Create the three argument encoders used for buffers, sampled textures, and storage images.
fn create_argument_encoders(
    device: &MTLDevice,
) -> (
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
    let storage_image_encoder =
        device.new_argument_encoder(mtl::Array::from_slice(&[storage_image_arg_desc]));
    tracing::info!(
        "Created storage image ArgumentEncoder (encoded_length={})",
        storage_image_encoder.encoded_length()
    );

    (argument_encoder, texture_encoder, storage_image_encoder)
}

/// Destroy a logical device and clean up resources owned by it.
pub(super) fn destroy(state: &mut MetalState, device_handle: DeviceHandle) {
    if let Some(mut ld) = state.devices.remove(&device_handle) {
        ld.staging_belt.destroy_all();
        ld.texture_staging_pool.destroy_all();
        ld.deletion_queue.flush_all();
        state
            .buffers
            .retain(|_, b| b.device_handle != device_handle);
        state
            .shaders
            .retain(|_, s| s.device_handle != device_handle);
        state
            .pipelines
            .retain(|_, p| p.device_handle != device_handle);
        state
            .compute_pipelines
            .retain(|_, p| p.device_handle != device_handle);
        state
            .render_targets
            .retain(|_, t| t.device_handle != device_handle);
        state
            .surfaces
            .retain(|_, s| s.device_handle != device_handle);
        state
            .textures
            .retain(|_, t| t.device_handle != device_handle);
        state
            .samplers
            .retain(|_, s| s.device_handle != device_handle);

        tracing::info!("Destroyed Metal device {}", device_handle);
    }
}

/// Check if a device handle is valid.
pub(super) fn is_valid(state: &MetalState, device_handle: DeviceHandle) -> bool {
    state.devices.contains_key(&device_handle)
}
