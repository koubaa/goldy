//! Device management logic.

use super::types::{self, DxgiAdapterInfo, LogicalDevice};
use super::{utils, DeviceHandle, Dx12State};
use crate::backend::{AdapterInfo, BackendType};
use anyhow::{Context, Result};
use windows::Win32::Graphics::{
    Direct3D::*,
    Direct3D12::*,
    Dxgi::*,
};

/// Enumerate available adapters.
pub(super) fn enumerate(adapters: &[DxgiAdapterInfo]) -> Vec<AdapterInfo> {
    adapters
        .iter()
        .map(|adapter| {
            let name = String::from_utf16_lossy(&adapter.desc.Description)
                .trim_end_matches('\0')
                .to_string();
            let flags = DXGI_ADAPTER_FLAG(adapter.desc.Flags as i32);
            let device_type = utils::device_type_from_flags(flags);
            let vendor = utils::vendor_name(adapter.desc.VendorId);

            AdapterInfo {
                id: adapter.adapter_id,
                name,
                vendor: vendor.to_string(),
                backend: BackendType::Dx12,
                device_type,
            }
        })
        .collect()
}

/// Create a logical device from an adapter ID.
#[allow(clippy::too_many_lines)]
pub(super) fn create(state: &mut Dx12State, adapter_id: u32) -> Result<DeviceHandle> {
    let adapter = state
        .adapters
        .iter()
        .find(|a| a.adapter_id == adapter_id)
        .context("Invalid adapter ID")?;

    // Create D3D12 device
    let mut device: Option<ID3D12Device> = None;
    unsafe { D3D12CreateDevice(&adapter.adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }
        .context("Failed to create D3D12 device")?;

    let device = device.context("D3D12CreateDevice returned null")?;

    // Create command queue
    let queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
        Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
        NodeMask: 0,
    };

    let command_queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }
        .context("Failed to create command queue")?;

    // Create command allocator
    let command_allocator: ID3D12CommandAllocator =
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .context("Failed to create command allocator")?;

    // Create RTV descriptor heap
    let rtv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: 256, // Should be enough for most cases
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };

    let rtv_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&rtv_heap_desc) }
        .context("Failed to create RTV heap")?;

    let rtv_descriptor_size =
        unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) };

    // Create DSV descriptor heap
    let dsv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
        NumDescriptors: 256,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };

    let dsv_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&dsv_heap_desc) }
        .context("Failed to create DSV heap")?;

    let dsv_descriptor_size =
        unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) };

    // Create CBV/SRV/UAV descriptor heap (large for bindless rendering)
    let cbv_srv_uav_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        NumDescriptors: 16384, // Large heap for bindless resource access
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        NodeMask: 0,
    };

    let cbv_srv_uav_heap: ID3D12DescriptorHeap =
        unsafe { device.CreateDescriptorHeap(&cbv_srv_uav_heap_desc) }
            .context("Failed to create CBV/SRV/UAV heap")?;

    let cbv_srv_uav_descriptor_size = unsafe {
        device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    };

    // Create sampler descriptor heap (large for bindless rendering)
    let sampler_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        NumDescriptors: 2048, // Large heap for bindless sampler access
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        NodeMask: 0,
    };

    let sampler_heap: ID3D12DescriptorHeap =
        unsafe { device.CreateDescriptorHeap(&sampler_heap_desc) }
            .context("Failed to create sampler heap")?;

    let sampler_descriptor_size =
        unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) };

    // Bindless rendering enabled via Slang's direct DXIL output (SM 6.6)
    // Shaders must use ResourceDescriptorHeap[index] with root constants for indices
    let bindless_enabled = true;
    tracing::info!("DX12 bindless enabled (SM 6.6 via Slang DXIL)");

    // Create shared bindless root signature
    // This will be reused by all graphics and compute pipelines
    let bindless_root_signature = if bindless_enabled {
        let root_constants = D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: types::MAX_ROOT_CONSTANT_INDICES as u32,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };

        let root_params = [root_constants];

        let desc1 = D3D12_ROOT_SIGNATURE_DESC1 {
            NumParameters: 1,
            pParameters: root_params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
        };

        let versioned_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 { Desc_1_1: desc1 },
        };

        let mut signature_blob: Option<ID3DBlob> = None;
        let mut error_blob: Option<ID3DBlob> = None;

        unsafe {
            D3D12SerializeVersionedRootSignature(
                &versioned_desc,
                &mut signature_blob,
                Some(&mut error_blob),
            )
        }
        .context("Failed to serialize shared bindless root signature")?;

        let blob = signature_blob.context("Root signature serialization produced no output")?;
        let root_sig: ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    blob.GetBufferPointer() as *const u8,
                    blob.GetBufferSize(),
                ),
            )
        }
        .context("Failed to create shared bindless root signature")?;

        tracing::debug!("Created shared bindless root signature");
        Some(root_sig)
    } else {
        None
    };

    // Create fence
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .context("Failed to create fence")?;

    let handle = state.next_device_handle;
    state.next_device_handle += 1;

    state.devices.insert(
        handle,
        LogicalDevice {
            device,
            adapter_id,
            command_queue,
            command_allocator,
            rtv_heap,
            rtv_descriptor_size,
            dsv_heap,
            dsv_descriptor_size,
            cbv_srv_uav_heap,
            cbv_srv_uav_descriptor_size,
            sampler_heap,
            sampler_descriptor_size,
            fence,
            fence_value: 1,
            bindless_enabled,
            bindless_root_signature,
            resource_registry: types::ResourceRegistry::new(),
        },
    );

    tracing::info!(
        "Created DX12 device {} for adapter {} [bindless={}]",
        handle,
        adapter_id,
        bindless_enabled
    );
    Ok(handle)
}

/// Destroy a logical device.
#[allow(dead_code)] // Will be used when mod.rs destroy_device is refactored in Phase 8
pub(super) fn destroy(state: &mut Dx12State, device_handle: DeviceHandle) {
    if let Some(logical_device) = state.devices.remove(&device_handle) {
        // Wait for GPU to finish
        let fence_value = logical_device.fence_value;
        let _ = unsafe { logical_device.command_queue.Signal(&logical_device.fence, fence_value) };
        // Note: no event handle - we're just doing a simple GPU wait
        
        tracing::info!("Destroyed DX12 device {}", device_handle);
    }
}
