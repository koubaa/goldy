//! Sampler management operations.

use super::types::{Dx12State, SamplerState};
use super::utils;
use super::{DeviceHandle, SamplerHandle};
use crate::types::SamplerDesc;
use anyhow::{Context, Result};
use windows::Win32::Graphics::Direct3D12::*;

/// Create a sampler.
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    desc: &SamplerDesc,
) -> Result<SamplerHandle> {
    let handle = state.next_sampler_handle;
    state.next_sampler_handle += 1;

    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("Invalid device handle")?;

    let sampler_offset = logical_device
        .ledger
        .lock()
        .unwrap()
        .resource_registry
        .register_sampler(handle);

    let sampler_desc = D3D12_SAMPLER_DESC {
        Filter: utils::filter_to_d3d12(desc.min_filter, desc.mag_filter, desc.mipmap_filter),
        AddressU: utils::address_mode_to_d3d12(desc.address_mode_u),
        AddressV: utils::address_mode_to_d3d12(desc.address_mode_v),
        AddressW: utils::address_mode_to_d3d12(desc.address_mode_w),
        MipLODBias: 0.0,
        MaxAnisotropy: desc.max_anisotropy as u32,
        ComparisonFunc: desc
            .compare
            .map(utils::compare_to_d3d12)
            .unwrap_or(D3D12_COMPARISON_FUNC_ALWAYS),
        BorderColor: [0.0, 0.0, 0.0, 0.0],
        MinLOD: desc.lod_min_clamp,
        MaxLOD: desc.lod_max_clamp,
    };

    let sampler_cpu_handle = unsafe {
        let mut handle = logical_device
            .sampler_heap
            .GetCPUDescriptorHandleForHeapStart();
        handle.ptr += (sampler_offset * logical_device.sampler_descriptor_size) as usize;
        handle
    };
    unsafe {
        logical_device
            .device
            .CreateSampler(&sampler_desc, sampler_cpu_handle);
    }

    state.samplers.insert(
        handle,
        SamplerState {
            device_handle,
            sampler_offset,
            desc: desc.clone(),
            bindless_offset: Some(sampler_offset), // Sampler offset is the bindless offset
        },
    );

    tracing::debug!("Created sampler (handle={})", handle);
    Ok(handle)
}

/// Destroy a sampler.
pub(super) fn destroy(state: &mut Dx12State, sampler_handle: SamplerHandle) {
    if let Some(sampler) = state.samplers.remove(&sampler_handle) {
        if let Some(ld) = state.devices.get_mut(&sampler.device_handle) {
            ld.ledger
                .lock()
                .unwrap()
                .reclaim_sampler_slots(sampler_handle);
        }
    }
}

/// Get the bindless index for a sampler.
pub(super) fn bindless_index(state: &Dx12State, sampler_handle: SamplerHandle) -> Option<u32> {
    state
        .samplers
        .get(&sampler_handle)
        .and_then(|s| s.bindless_offset)
}
