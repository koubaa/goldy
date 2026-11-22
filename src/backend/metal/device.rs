//! Runtime management logic.

use super::super::DeviceHandle;
use super::types::{
    DeletionQueue, DescriptorRegistry, HeapAllocator, LogicalDevice, MetalAdapterInfo, MetalState,
    TextureHeapAllocator, ARGUMENT_BUFFER_SIZE,
};
use crate::backend::metal_storage::{HeapStorage, TextureStorage};
use crate::backend::{AdapterInfo, BackendType};
use crate::types::DeviceType;
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
use mtl::{Device as MTLDevice, MTLResourceOptions, MTLStorageMode};

/// Enumerate Metal adapters that meet the hardware floor.
///
/// Unsupported GPUs (Polaris, the Intel CI paravirtual device, anything
/// without Metal 3 or Tier 2) are omitted so adapter selection falls through
/// to a GPU that can run.
pub(super) fn enumerate(adapters: &[MetalAdapterInfo]) -> Vec<AdapterInfo> {
    adapters
        .iter()
        .filter(|entry| device_facts(&entry.device).meets_floor())
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
pub(super) fn adapter_capabilities(
    adapters: &[MetalAdapterInfo],
    adapter_id: u32,
) -> crate::runtime::RuntimeCapabilities {
    let mut caps = crate::runtime::RuntimeCapabilities {
        has_zero_copy_storage_readback: true,
        buffer_resize_cost: crate::types::BufferResizeCost::Constant,
        buffer_page_size: 16 * 1024,
        buffer_decommit_supported: true,
        host_sidecar_on_submit_worker: true,
        // One command buffer per partition. Encode overlap is opt-in
        // (`RuntimeDescriptor::compute_partition_split`): a command-buffer boundary
        // waits on the shared event, which helps a latency-bound decode and costs
        // a renderer that already keeps the GPU busy.
        compute_partition_split: crate::runtime::ComputePartitionSplit::None,
        // Fuse upload blits with the following compute partition so Scheme matches
        // Classic's single-CB structure and avoids an extra commit + event wait.
        fuse_upload_with_compute_partitions: true,
        ..crate::runtime::RuntimeCapabilities::default()
    };
    if let Some(entry) = adapters.iter().find(|a| a.adapter_id == adapter_id) {
        let device = &entry.device;
        let facts = device_facts(device);
        let plan = facts.memory_features().plan();
        if plan.discrete {
            // Private parcels are not demand-paged, and madvise does not apply.
            // CPU_READABLE buffers stay Shared, so zero-copy readback remains available.
            caps.buffer_resize_cost = crate::types::BufferResizeCost::Copy;
            caps.buffer_decommit_supported = false;
            caps.buffer_page_size = host_page_size();
        }
        let rt = device.supports_raytracing();
        caps.ray_query = rt;
        // Metal has no SBT / TraceRays; inline RayQuery is the RT path.
        caps.ray_tracing_pipelines = false;
        caps.mesh_shaders = facts.mesh_shaders();
        caps.amplification_shaders = facts.mesh_shaders();
    }
    caps
}

fn host_page_size() -> u64 {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page > 0 {
            return page as u64;
        }
    }
    16 * 1024
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

    let facts = device_facts(&device);
    if let Some(reason) = facts.floor_rejection() {
        anyhow::bail!(
            "{reason} ({}; apple_family={}, metal3={}, tier2={}, rw_texture_tier2={}, os_major={}). \
             Intel and AMD Mac support is best-effort and untested on hardware.",
            device.name(),
            facts.apple_family,
            facts.metal3,
            facts.argument_buffers_tier2,
            facts.read_write_texture_tier2,
            facts.os_major,
        );
    }

    let features = facts.memory_features();
    let memory = features.plan();
    tracing::info!(
        name = %device.name(),
        family = features.family_summary(),
        unified = features.has_unified_memory,
        apple_family = features.apple_family,
        mac2 = features.mac2_family,
        heap = ?memory.heap,
        host_buffers = ?memory.host_visible_buffer,
        textures = ?memory.texture,
        "Metal storage plan"
    );

    // CPU encodes into `argument_buffer`. On Apple silicon the GPU reads that
    // same Shared buffer. Discrete placement keeps a Private copy and blits
    // before each encoder (`stage_bindless_uploads`).
    let argument_buffer = device.new_buffer(ARGUMENT_BUFFER_SIZE, MTLResourceOptions::StorageModeShared);
    let argument_device = if memory.discrete {
        Some(device.new_buffer(ARGUMENT_BUFFER_SIZE, MTLResourceOptions::StorageModePrivate))
    } else {
        None
    };
    tracing::info!("Created argument buffer");

    // Heaps are Shared only for Apple-family GPUs (`supportsFamily(Apple1)`).
    // Intel iGPUs report `hasUnifiedMemory` and still abort `newHeap:` with
    // "Shared storage mode disallowed". Those GPUs, and AMD discrete Macs, get
    // Private heaps. Host-visible buffers there are standalone Shared
    // allocations (see `allocate_mtl_storage_buffer`). CPU cache mode on the
    // heap matches buffer allocations that come out of it.
    let heap_size = INITIAL_HEAP_SIZE;
    crate::backend::metal_storage::debug_refuse_shared_resource(
        memory.discrete,
        matches!(memory.heap, HeapStorage::Shared),
        "MTLHeap",
    );
    let (heap_allocator, texture_heap) = create_heaps(&device, heap_size, memory.heap);

    let (argument_encoder, texture_encoder, storage_image_encoder, sampler_encoder, accel_encoder) =
        create_argument_encoders(&device)?;

    let frame_table = super::frame_table::MetalFrameTable::init(
        device.as_ref(),
        argument_buffer.as_ref(),
        argument_encoder.as_ref(),
        memory.discrete,
    );

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
        memory,
        heap_allocator: Mutex::new(heap_allocator),
        texture_heap: Mutex::new(texture_heap),
        argument_buffer,
        argument_device,
        argument_encoder,
        texture_encoder,
        storage_image_encoder,
        sampler_encoder,
        accel_encoder,
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

fn device_facts(device: &MTLDevice) -> crate::backend::metal_storage::MetalDeviceFacts {
    use crate::backend::metal_storage::MetalDeviceFacts;
    MetalDeviceFacts {
        has_unified_memory: device.has_unified_memory(),
        apple_family: device.supports_family(mtl::MTLGPUFamily::Apple1),
        apple7: device.supports_family(mtl::MTLGPUFamily::Apple7),
        mac2_family: device.supports_family(mtl::MTLGPUFamily::Mac2),
        metal3: device.supports_family(mtl::MTLGPUFamily::Metal3),
        argument_buffers_tier2: device.argument_buffers_support() == mtl::MTLArgumentBuffersTier::Tier2,
        read_write_texture_tier2: device.read_write_texture_support() == mtl::MTLReadWriteTextureTier::Tier2,
        os_major: host_os_major(),
        paravirtual: device.name().to_lowercase().contains("paravirtual"),
    }
}

fn host_os_major() -> u32 {
    static MAJOR: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MAJOR.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("sw_vers").arg("-productVersion").output() {
                if let Ok(text) = std::str::from_utf8(&output.stdout) {
                    if let Some(major) = text.trim().split('.').next().and_then(|part| part.parse().ok()) {
                        return major;
                    }
                }
            }
            0
        }
        #[cfg(not(target_os = "macos"))]
        {
            17
        }
    })
}

fn heap_mtl_storage(storage: HeapStorage) -> MTLStorageMode {
    match storage {
        HeapStorage::Shared => MTLStorageMode::Shared,
        HeapStorage::Private => MTLStorageMode::Private,
    }
}

pub(super) fn texture_mtl_storage(storage: TextureStorage) -> MTLStorageMode {
    match storage {
        TextureStorage::Shared => MTLStorageMode::Shared,
        TextureStorage::Private => MTLStorageMode::Private,
    }
}

/// Create the buffer and texture heap allocators for a logical device.
///
/// Both heaps use the same storage mode. Tracked (not Untracked): switching to
/// Untracked was investigated as a potential perf win but broke frame
/// pipelining. The explicit synchronisation Untracked requires (fences at
/// blit/compute boundaries, shared-event waits, a written-buffer annotation)
/// costs more than implicit hazard tracking for this workload. Textures are a
/// harder case still — render-target hazards span multiple command buffers.
fn create_heaps(device: &MTLDevice, heap_size: u64, storage: HeapStorage) -> (HeapAllocator, TextureHeapAllocator) {
    let mode = heap_mtl_storage(storage);
    tracing::info!("Creating buffer heap allocator ({:?})...", mode);
    let buffer_heap = super::types::new_tracked_heap(device, heap_size, mode);
    let heap_allocator = HeapAllocator::new(device.clone(), buffer_heap, heap_size);
    tracing::info!("Created buffer heap allocator (primary={}MB)", heap_size / 1024 / 1024);

    tracing::info!("Creating texture heap ({:?})...", mode);
    let texture_heap_raw = super::types::new_tracked_heap(device, heap_size, mode);
    let texture_heap = TextureHeapAllocator::new(device.clone(), texture_heap_raw, heap_size);
    tracing::info!("Created texture heap (size={}MB)", heap_size / 1024 / 1024);

    (heap_allocator, texture_heap)
}

/// Create the four argument encoders used for buffers, sampled textures, storage images,
/// and samplers.
pub(super) fn create_argument_encoders(
    device: &MTLDevice,
) -> Result<(
    mtl::ArgumentEncoder,
    mtl::ArgumentEncoder,
    mtl::ArgumentEncoder,
    mtl::ArgumentEncoder,
    Option<mtl::ArgumentEncoder>,
)> {
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
    require_encoder_stride("buffer", argument_encoder.encoded_length())?;
    require_encoder_stride("texture", texture_encoder.encoded_length())?;
    require_encoder_stride("storage image", storage_image_encoder.encoded_length())?;
    require_encoder_stride("sampler", sampler_stride)?;

    let accel_encoder = if device.supports_raytracing() {
        let accel_arg_desc = mtl::ArgumentDescriptor::new();
        accel_arg_desc.set_index(0);
        accel_arg_desc.set_data_type(mtl::MTLDataType::InstanceAccelerationStructure);
        accel_arg_desc.set_access(mtl::MTLArgumentAccess::ReadOnly);
        let accel_encoder = device.new_argument_encoder(mtl::Array::from_slice(&[accel_arg_desc]));
        require_encoder_stride("acceleration structure", accel_encoder.encoded_length())?;
        tracing::info!(
            "Created accel ArgumentEncoder (encoded_length={})",
            accel_encoder.encoded_length()
        );
        Some(accel_encoder)
    } else {
        tracing::info!("Skipping accel ArgumentEncoder (device has no ray tracing)");
        None
    };

    Ok((
        argument_encoder,
        texture_encoder,
        storage_image_encoder,
        sampler_encoder,
        accel_encoder,
    ))
}

fn require_encoder_stride(name: &str, encoded_length: u64) -> Result<()> {
    anyhow::ensure!(
        encoded_length == 8,
        "Metal {name} argument encoder reported encoded_length={encoded_length}, expected 8. \
         ARGUMENT_BUFFER_SIZE assumes an 8-byte stride (macOS 13+ C-layout argument buffers)."
    );
    Ok(())
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
