//! Sampler management logic.

use super::types::{self, SamplerState, SharedSamplerTable};
use super::utils::{address_mode_to_vk, compare_to_vk, filter_to_vk, mipmap_mode_to_vk};
use super::{DeviceHandle, SamplerHandle};
use crate::types::SamplerDesc;
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

/// Create a sampler with the given description.
pub(super) fn create(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    samplers: &SharedSamplerTable,
    device_handle: DeviceHandle,
    desc: &SamplerDesc,
) -> Result<SamplerHandle> {
    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;

    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(filter_to_vk(desc.mag_filter))
        .min_filter(filter_to_vk(desc.min_filter))
        .mipmap_mode(mipmap_mode_to_vk(desc.mipmap_filter))
        .address_mode_u(address_mode_to_vk(desc.address_mode_u))
        .address_mode_v(address_mode_to_vk(desc.address_mode_v))
        .address_mode_w(address_mode_to_vk(desc.address_mode_w))
        .mip_lod_bias(0.0)
        .anisotropy_enable(desc.max_anisotropy > 1.0)
        .max_anisotropy(desc.max_anisotropy)
        .compare_enable(desc.compare.is_some())
        .compare_op(desc.compare.map(compare_to_vk).unwrap_or(vk::CompareOp::ALWAYS))
        .min_lod(desc.lod_min_clamp)
        .max_lod(desc.lod_max_clamp)
        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
        .unnormalized_coordinates(false);

    let sampler =
        unsafe { logical_device.device.create_sampler(&sampler_info, None) }.context("Failed to create sampler")?;

    let bindless_descriptor_set = logical_device.bindless_descriptor_set;

    let handle = samplers.write().unwrap().alloc_handle();

    let bindless_index = {
        let logical_device = devices.get(&device_handle).unwrap();
        let index = logical_device
            .descriptors
            .lock()
            .unwrap()
            .resource_registry
            .register_sampler(handle);

        // Update the global descriptor set with this sampler
        if let Some(descriptor_set) = bindless_descriptor_set {
            let sampler_info = vk::DescriptorImageInfo::default().sampler(sampler);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(types::bindless_bindings::SAMPLERS)
                .dst_array_element(index)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(std::slice::from_ref(&sampler_info));

            unsafe {
                logical_device
                    .device
                    .update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }

            tracing::trace!("Registered sampler {} at bindless index {}", handle, index);
        }

        Some(index)
    };

    samplers.write().unwrap().entries.insert(
        handle,
        SamplerState {
            device_handle,
            sampler,
            bindless_index,
        },
    );

    tracing::debug!("Created sampler (handle={})", handle);
    Ok(handle)
}

/// Destroy a sampler, unregistering it from bindless and cleaning up GPU resources.
pub(super) fn destroy(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    samplers: &SharedSamplerTable,
    sampler_handle: SamplerHandle,
) {
    if let Some(sampler) = samplers.write().unwrap().entries.remove(&sampler_handle) {
        if let Some(logical_device) = devices.get(&sampler.device_handle) {
            // Defer reclamation: sampler slot must not be reused until all
            // in-flight submissions that referenced it have retired.
            logical_device
                .descriptors
                .lock()
                .unwrap()
                .reclaim_sampler_slots(sampler_handle);

            unsafe {
                logical_device.device_wait_idle_locked().ok();
                logical_device.device.destroy_sampler(sampler.sampler, None);
            }
        }
    }
}

/// Get the bindless descriptor index for a sampler, if any.
pub(super) fn bindless_index(samplers: &SharedSamplerTable, sampler_handle: SamplerHandle) -> Option<u32> {
    samplers
        .read()
        .unwrap()
        .entries
        .get(&sampler_handle)
        .and_then(|s| s.bindless_index)
}
