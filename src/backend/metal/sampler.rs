//! Sampler management logic.

use super::super::{DeviceHandle, SamplerHandle};
use super::types::{MetalState, ResourceRegistry, ARGUMENT_BUFFER_SIZE};
use super::utils::{address_mode_to_mtl, compare_to_mtl, filter_to_mtl, mipmap_mode_to_mtl};
use ::metal as mtl;
use anyhow::{Context, Result};

/// Create a sampler.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    desc: &crate::types::SamplerDesc,
) -> Result<SamplerHandle> {
    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    let handle = state.next_sampler_handle;
    state.next_sampler_handle += 1;

    let descriptor = mtl::SamplerDescriptor::new();
    descriptor.set_min_filter(filter_to_mtl(desc.min_filter));
    descriptor.set_mag_filter(filter_to_mtl(desc.mag_filter));
    descriptor.set_mip_filter(mipmap_mode_to_mtl(desc.mipmap_filter));
    descriptor.set_address_mode_s(address_mode_to_mtl(desc.address_mode_u));
    descriptor.set_address_mode_t(address_mode_to_mtl(desc.address_mode_v));
    descriptor.set_address_mode_r(address_mode_to_mtl(desc.address_mode_w));
    descriptor.set_max_anisotropy(desc.max_anisotropy as u64);
    descriptor.set_lod_min_clamp(desc.lod_min_clamp);
    descriptor.set_lod_max_clamp(desc.lod_max_clamp);
    descriptor.set_support_argument_buffers(true);

    if let Some(compare) = desc.compare {
        descriptor.set_compare_function(compare_to_mtl(compare));
    }

    let sampler = logical_device.device.new_sampler(&descriptor);

    let index = logical_device.resource_registry.register_sampler(handle);
    let encoding_index = ResourceRegistry::sampler_global_index(index);
    tracing::debug!(
        "Registered sampler {} at bindless local={} global={}",
        handle,
        index,
        encoding_index
    );

    let encoded_length = logical_device.sampler_encoder.encoded_length();
    let offset = (encoding_index as u64) * encoded_length;
    if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
        logical_device
            .sampler_encoder
            .set_argument_buffer(&logical_device.argument_buffer, offset);
        logical_device
            .sampler_encoder
            .set_sampler_state(0, &sampler);
        tracing::trace!(
            "Encoded sampler {} at arg buffer offset {} (stride={})",
            handle,
            offset,
            encoded_length,
        );
    } else {
        tracing::error!(
            "Sampler {handle}: argument buffer overflow — \
             offset {offset} + encoded_length {encoded_length} \
             exceeds ARGUMENT_BUFFER_SIZE {ARGUMENT_BUFFER_SIZE}; \
             sampler will not be encoded and shaders may see a stale binding"
        );
    }

    state.samplers.insert(
        handle,
        super::types::SamplerState_ {
            device_handle,
            sampler,
            arg_buffer_index: index,
        },
    );

    tracing::debug!("Created sampler (handle={})", handle);
    Ok(handle)
}

/// Destroy a sampler.
pub(super) fn destroy(state: &mut MetalState, sampler_handle: SamplerHandle) {
    if let Some(sampler) = state.samplers.remove(&sampler_handle) {
        if let Some(device) = state.devices.get_mut(&sampler.device_handle) {
            device.resource_registry.unregister_sampler(sampler_handle);
            let barrier = device
                .timeline_scheduled_max
                .load(std::sync::atomic::Ordering::Relaxed);
            device.deletion_queue.queue(
                barrier,
                super::types::PendingDeletion::Sampler {
                    sampler: sampler.sampler,
                },
            );
        }
    }
}

/// Get the bindless index for a sampler.
pub(super) fn bindless_index(state: &MetalState, sampler_handle: SamplerHandle) -> Option<u32> {
    state
        .samplers
        .get(&sampler_handle)
        .map(|s| s.arg_buffer_index)
}
