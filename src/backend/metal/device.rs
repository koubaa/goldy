//! Device management logic.

use super::super::DeviceHandle;
use super::types::{LogicalDevice, MetalState, ResourceRegistry, ARGUMENT_BUFFER_SIZE};
use crate::backend::{AdapterInfo, BackendType, DeviceType};
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{
    Device as MTLDevice, HeapDescriptor, MTLCPUCacheMode, MTLHeapType, MTLResourceOptions,
    MTLStorageMode,
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
#[allow(clippy::too_many_lines)]
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
    // TODO(#heap-config): Heap sizes are intentionally hardcoded for now.
    let heap_size: u64 = 64 * 1024 * 1024; // 64 MB per heap

    tracing::info!("Creating buffer heap...");
    let buffer_heap_desc = HeapDescriptor::new();
    buffer_heap_desc.set_size(heap_size);
    buffer_heap_desc.set_storage_mode(MTLStorageMode::Shared);
    buffer_heap_desc.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);
    buffer_heap_desc.set_heap_type(MTLHeapType::Automatic);
    let buffer_heap = device.new_heap(&buffer_heap_desc);
    tracing::info!("Created buffer heap (size={}MB)", heap_size / 1024 / 1024);

    tracing::info!("Creating texture heap...");
    let texture_heap_desc = HeapDescriptor::new();
    texture_heap_desc.set_size(heap_size);
    texture_heap_desc.set_storage_mode(MTLStorageMode::Shared);
    texture_heap_desc.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);
    texture_heap_desc.set_heap_type(MTLHeapType::Automatic);
    let texture_heap = device.new_heap(&texture_heap_desc);
    tracing::info!("Created texture heap (size={}MB)", heap_size / 1024 / 1024);

    // Create ArgumentEncoder for encoding buffers into the argument buffer
    let buffer_arg_desc = mtl::ArgumentDescriptor::new();
    buffer_arg_desc.set_index(0);
    buffer_arg_desc.set_data_type(mtl::MTLDataType::Pointer);
    buffer_arg_desc.set_access(mtl::MTLArgumentAccess::ReadWrite);
    let argument_encoder = device.new_argument_encoder(mtl::Array::from_slice(&[buffer_arg_desc]));
    tracing::info!(
        "Created buffer ArgumentEncoder (encoded_length={})",
        argument_encoder.encoded_length()
    );

    // Create ArgumentEncoder for encoding textures
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

    let handle = state.next_device_handle;
    state.next_device_handle += 1;

    tracing::info!(
        "Created Metal device {} for adapter {} ({})",
        handle,
        adapter_id,
        device.name(),
    );

    state.devices.insert(
        handle,
        LogicalDevice {
            device,
            command_queue,
            buffer_heap,
            texture_heap,
            argument_buffer,
            argument_encoder,
            texture_encoder,
            resource_registry: ResourceRegistry::new(),
            heap_buffer_count: 0,
            heap_texture_count: 0,
        },
    );

    Ok(handle)
}

/// Destroy a logical device and clean up resources owned by it.
pub(super) fn destroy(state: &mut MetalState, device_handle: DeviceHandle) {
    if state.devices.remove(&device_handle).is_some() {
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
