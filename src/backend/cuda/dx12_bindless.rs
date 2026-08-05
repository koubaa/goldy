//! Companion DX12 bindless heaps, root signature, and frame-table for CUDA raster.
//!
//! CUDA registry slots are the DX12 descriptor indices: the same `u32` identifies a
//! resource for PTX launch args and for `ResourceDescriptorHeap[index]` in DXIL.
//! Protocol slots 0/1 are reserved for the device-level frame-table selector/table.

#![cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]

use super::dx12_companion::Dx12Companion;
use crate::backend::shared::TOTAL_PUSH_BYTES;
use crate::frame_table::{
    staging_row_payload_byte_offset, staging_selector_byte_offset, FRAME_TABLE_MAX_ROWS, FRAME_TABLE_ROW_STRIDE,
    FRAME_TABLE_STAGING_BYTES, FRAME_TABLE_TABLE_U32S, FRAME_TABLE_USER_SLOT_BASE,
};
use crate::types::{AddressMode, BindlessSlotKind, FilterMode, SamplerDesc, TextureFormat};
use anyhow::{bail, Context as _, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub(super) const MAX_BINDLESS_CBV_SRV_UAV: u32 = 16384;
pub(super) const MAX_BINDLESS_SAMPLERS: u32 = 2048;

/// First user-resource slot (0/1 reserved for frame-table protocol).
pub(super) const USER_SLOT_BASE: u32 = FRAME_TABLE_USER_SLOT_BASE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DescriptorCategory {
    BufferSrv,
    BufferUav,
    BufferCbv,
    TextureSrv,
    TextureUav,
    #[allow(dead_code)] // sampler heap uses `occupy_sampler`, not resource `occupy`
    Sampler,
}

impl DescriptorCategory {
    pub fn slot_kind(self) -> BindlessSlotKind {
        match self {
            Self::BufferSrv | Self::TextureSrv => BindlessSlotKind::ReadOnlySrv,
            Self::BufferUav | Self::TextureUav => BindlessSlotKind::StorageUav,
            Self::BufferCbv => BindlessSlotKind::UniformCbv,
            Self::Sampler => BindlessSlotKind::ReadOnlySrv,
        }
    }
}

/// Occupancy + category metadata for a fixed-index companion descriptor heap.
pub(super) struct BindlessRegistry {
    /// Slot → category currently written (resource heap).
    resource_slots: HashMap<u32, DescriptorCategory>,
    /// Sampler-heap occupancy.
    sampler_slots: HashMap<u32, ()>,
    /// Deferred reclaim: (slot, is_sampler, retire_at_companion_fence).
    pending_reclaim: Vec<(u32, bool, u64)>,
}

impl BindlessRegistry {
    pub fn new() -> Self {
        Self {
            resource_slots: HashMap::new(),
            sampler_slots: HashMap::new(),
            pending_reclaim: Vec::new(),
        }
    }

    pub fn occupy(&mut self, slot: u32, category: DescriptorCategory) -> Result<()> {
        if slot >= MAX_BINDLESS_CBV_SRV_UAV {
            bail!("CUDA/DX12 bindless: resource slot {slot} exceeds heap size {MAX_BINDLESS_CBV_SRV_UAV}");
        }
        if let Some(prev) = self.resource_slots.insert(slot, category) {
            if prev != category {
                tracing::debug!(
                    slot,
                    ?prev,
                    ?category,
                    "CUDA/DX12 bindless: overwriting descriptor category"
                );
            }
        }
        Ok(())
    }

    pub fn occupy_sampler(&mut self, slot: u32) -> Result<()> {
        if slot >= MAX_BINDLESS_SAMPLERS {
            bail!("CUDA/DX12 bindless: sampler slot {slot} exceeds heap size {MAX_BINDLESS_SAMPLERS}");
        }
        self.sampler_slots.insert(slot, ());
        Ok(())
    }

    pub fn category(&self, slot: u32) -> Option<DescriptorCategory> {
        self.resource_slots.get(&slot).copied()
    }

    pub fn slot_kind(&self, slot: u32) -> Option<BindlessSlotKind> {
        self.category(slot).map(|c| c.slot_kind())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn has_resource(&self, slot: u32) -> bool {
        self.resource_slots.contains_key(&slot)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn has_sampler(&self, slot: u32) -> bool {
        self.sampler_slots.contains_key(&slot)
    }

    /// Queue slot for reclaim after companion fence `retire_at`.
    pub fn defer_reclaim(&mut self, slot: u32, is_sampler: bool, retire_at: u64) {
        if is_sampler {
            self.sampler_slots.remove(&slot);
        } else {
            self.resource_slots.remove(&slot);
        }
        self.pending_reclaim.push((slot, is_sampler, retire_at));
    }

    pub fn drain_reclaimed(&mut self, completed: u64) {
        self.pending_reclaim.retain(|(_, _, retire_at)| *retire_at > completed);
    }
}

/// Shader-visible heaps + bindless root signature owned by the companion.
pub(super) struct BindlessHeaps {
    pub cbv_srv_uav_heap: ID3D12DescriptorHeap,
    pub cbv_srv_uav_descriptor_size: u32,
    pub sampler_heap: ID3D12DescriptorHeap,
    pub sampler_descriptor_size: u32,
    pub root_signature: ID3D12RootSignature,
    pub registry: std::sync::Mutex<BindlessRegistry>,
}

// SAFETY: COM heaps used under Goldy's backend lock.
unsafe impl Send for BindlessHeaps {}
unsafe impl Sync for BindlessHeaps {}

impl BindlessHeaps {
    pub fn create(device: &ID3D12Device) -> Result<Self> {
        let cbv_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: MAX_BINDLESS_CBV_SRV_UAV,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        let cbv_srv_uav_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&cbv_desc) }
            .context("CUDA/DX12: CreateDescriptorHeap(CBV/SRV/UAV) failed")?;
        let cbv_srv_uav_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };

        let sampler_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: MAX_BINDLESS_SAMPLERS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        let sampler_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&sampler_desc) }
            .context("CUDA/DX12: CreateDescriptorHeap(SAMPLER) failed")?;
        let sampler_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) };

        let root_signature = create_bindless_root_signature(device)?;

        Ok(Self {
            cbv_srv_uav_heap,
            cbv_srv_uav_descriptor_size,
            sampler_heap,
            sampler_descriptor_size,
            root_signature,
            registry: std::sync::Mutex::new(BindlessRegistry::new()),
        })
    }

    fn resource_cpu_handle(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let mut h = unsafe { self.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart() };
        h.ptr += (slot as usize) * self.cbv_srv_uav_descriptor_size as usize;
        h
    }

    fn sampler_cpu_handle(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let mut h = unsafe { self.sampler_heap.GetCPUDescriptorHandleForHeapStart() };
        h.ptr += (slot as usize) * self.sampler_descriptor_size as usize;
        h
    }

    pub fn write_buffer_srv(
        &self,
        device: &ID3D12Device,
        slot: u32,
        resource: &ID3D12Resource,
        num_elements: u32,
        stride: u32,
    ) -> Result<()> {
        self.registry
            .lock()
            .unwrap()
            .occupy(slot, DescriptorCategory::BufferSrv)?;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: num_elements.max(1),
                    StructureByteStride: stride.max(4),
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        unsafe {
            device.CreateShaderResourceView(resource, Some(&desc), self.resource_cpu_handle(slot));
        }
        Ok(())
    }

    pub fn write_buffer_uav(
        &self,
        device: &ID3D12Device,
        slot: u32,
        resource: &ID3D12Resource,
        num_elements: u32,
        stride: u32,
    ) -> Result<()> {
        self.registry
            .lock()
            .unwrap()
            .occupy(slot, DescriptorCategory::BufferUav)?;
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: num_elements.max(1),
                    StructureByteStride: stride.max(4),
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                },
            },
        };
        unsafe {
            device.CreateUnorderedAccessView(resource, None, Some(&desc), self.resource_cpu_handle(slot));
        }
        Ok(())
    }

    pub fn write_buffer_cbv(
        &self,
        device: &ID3D12Device,
        slot: u32,
        resource: &ID3D12Resource,
        size_bytes: u64,
    ) -> Result<()> {
        self.registry
            .lock()
            .unwrap()
            .occupy(slot, DescriptorCategory::BufferCbv)?;
        let aligned = (size_bytes + 255) & !255;
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: unsafe { resource.GetGPUVirtualAddress() },
            SizeInBytes: aligned as u32,
        };
        unsafe {
            device.CreateConstantBufferView(Some(&desc), self.resource_cpu_handle(slot));
        }
        Ok(())
    }

    pub fn write_texture_srv(
        &self,
        device: &ID3D12Device,
        slot: u32,
        resource: &ID3D12Resource,
        format: TextureFormat,
    ) -> Result<()> {
        self.registry
            .lock()
            .unwrap()
            .occupy(slot, DescriptorCategory::TextureSrv)?;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: texture_format_to_dxgi(format)?,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        unsafe {
            device.CreateShaderResourceView(resource, Some(&desc), self.resource_cpu_handle(slot));
        }
        Ok(())
    }

    pub fn write_texture_uav(
        &self,
        device: &ID3D12Device,
        slot: u32,
        resource: &ID3D12Resource,
        format: TextureFormat,
    ) -> Result<()> {
        self.registry
            .lock()
            .unwrap()
            .occupy(slot, DescriptorCategory::TextureUav)?;
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: texture_format_to_dxgi(format)?,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        unsafe {
            device.CreateUnorderedAccessView(Some(resource), None, Some(&desc), self.resource_cpu_handle(slot));
        }
        Ok(())
    }

    pub fn write_sampler(&self, device: &ID3D12Device, slot: u32, desc: &SamplerDesc) -> Result<()> {
        self.registry.lock().unwrap().occupy_sampler(slot)?;
        let sampler_desc = sampler_desc_to_d3d12(desc);
        unsafe {
            device.CreateSampler(&sampler_desc, self.sampler_cpu_handle(slot));
        }
        Ok(())
    }

    pub fn set_descriptor_heaps(&self, list: &ID3D12GraphicsCommandList) {
        let heaps: [Option<ID3D12DescriptorHeap>; 2] =
            [Some(self.cbv_srv_uav_heap.clone()), Some(self.sampler_heap.clone())];
        unsafe { list.SetDescriptorHeaps(&heaps) };
    }

    pub fn defer_reclaim_resource(&self, slot: u32, retire_at: u64) {
        self.registry.lock().unwrap().defer_reclaim(slot, false, retire_at);
    }

    pub fn defer_reclaim_sampler(&self, slot: u32, retire_at: u64) {
        self.registry.lock().unwrap().defer_reclaim(slot, true, retire_at);
    }

    pub fn drain_reclaimed(&self, completed: u64) {
        self.registry.lock().unwrap().drain_reclaimed(completed);
    }
}

/// Device-level frame table for companion raster (single FIFO DIRECT queue).
pub(super) struct CompanionFrameTable {
    pub selector: ID3D12Resource,
    pub device_table: ID3D12Resource,
    pub staging: ID3D12Resource,
    pub staging_mapped: *mut u8,
    pub selector_slot: u32,
    pub table_slot: u32,
    /// Last companion fence that consumed each row (0 = free).
    pub last_fence_for_row: [AtomicU64; FRAME_TABLE_MAX_ROWS as usize],
    pub next_row: AtomicU64,
}

// SAFETY: resources used under backend lock; mapped pointer is companion-owned.
unsafe impl Send for CompanionFrameTable {}
unsafe impl Sync for CompanionFrameTable {}

impl CompanionFrameTable {
    pub fn create(device: &ID3D12Device, bindless: &BindlessHeaps) -> Result<Self> {
        let selector = create_default_u32_buffer(device, 1, "cuda_ft_selector")?;
        let device_table = create_default_u32_buffer(device, FRAME_TABLE_TABLE_U32S as u32, "cuda_ft_table")?;
        let (staging, staging_mapped) = create_upload_staging(device)?;

        // Protocol slots 0/1 hold selector / table UAVs (Metal-style fixed indices).
        bindless.write_buffer_uav(device, 0, &selector, 1, 4)?;
        bindless.write_buffer_uav(device, 1, &device_table, FRAME_TABLE_TABLE_U32S as u32, 4)?;

        Ok(Self {
            selector,
            device_table,
            staging,
            staging_mapped,
            selector_slot: 0,
            table_slot: 1,
            last_fence_for_row: std::array::from_fn(|_| AtomicU64::new(0)),
            next_row: AtomicU64::new(0),
        })
    }

    /// Write staging + record GPU copies. Returns the chosen row index.
    pub fn record_prologue(
        &self,
        companion: &Dx12Companion,
        list: &ID3D12GraphicsCommandList,
        data: &[u32],
    ) -> Result<u32> {
        let completed = unsafe { companion.fence.GetCompletedValue() };
        let start = (self.next_row.fetch_add(1, Ordering::AcqRel) % u64::from(FRAME_TABLE_MAX_ROWS)) as u32;
        let mut row = start;
        for _ in 0..FRAME_TABLE_MAX_ROWS {
            let prev = self.last_fence_for_row[row as usize].load(Ordering::Acquire);
            if prev <= completed {
                break;
            }
            row = (row + 1) % FRAME_TABLE_MAX_ROWS;
        }
        let prev = self.last_fence_for_row[row as usize].load(Ordering::Acquire);
        if prev > completed {
            companion.cpu_wait(prev)?;
        }

        // Pack selector word + row payload into staging (same layout as DX12).
        let staging =
            unsafe { std::slice::from_raw_parts_mut(self.staging_mapped, FRAME_TABLE_STAGING_BYTES as usize) };
        let sel_off = staging_selector_byte_offset(row) as usize;
        staging[sel_off..sel_off + 4].copy_from_slice(&row.to_le_bytes());
        let payload_off = staging_row_payload_byte_offset(row) as usize;
        let copy_u32s = data.len().min(FRAME_TABLE_ROW_STRIDE as usize);
        for (i, &word) in data.iter().take(copy_u32s).enumerate() {
            let o = payload_off + i * 4;
            staging[o..o + 4].copy_from_slice(&word.to_le_bytes());
        }

        let row_bytes = u64::from(FRAME_TABLE_ROW_STRIDE) * 4;
        let dest_offset = u64::from(row) * row_bytes;
        let src_payload = staging_row_payload_byte_offset(row);
        let src_selector = staging_selector_byte_offset(row);

        // Selector/table are created in UAV and returned to UAV after every prologue so
        // retained command lists can replay the same barriers safely.
        transition_buffer(
            list,
            &self.staging,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        transition_buffer(
            list,
            &self.device_table,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        transition_buffer(
            list,
            &self.selector,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        unsafe {
            list.CopyBufferRegion(
                &self.device_table,
                dest_offset,
                &self.staging,
                src_payload,
                (copy_u32s * 4) as u64,
            );
            list.CopyBufferRegion(&self.selector, 0, &self.staging, src_selector, 4);
        }

        transition_buffer(
            list,
            &self.device_table,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        transition_buffer(
            list,
            &self.selector,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        transition_buffer(
            list,
            &self.staging,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );

        Ok(row)
    }

    pub fn mark_row_submitted(&self, row: u32, fence: u64) {
        self.last_fence_for_row[row as usize].store(fence, Ordering::Release);
    }
}

fn create_bindless_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature> {
    let root_constants = D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: 0,
                Num32BitValues: (TOTAL_PUSH_BYTES / 4) as u32,
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
    let versioned = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 { Desc_1_1: desc1 },
    };
    let mut sig_blob: Option<ID3DBlob> = None;
    let mut sig_err: Option<ID3DBlob> = None;
    unsafe { D3D12SerializeVersionedRootSignature(&versioned, &mut sig_blob, Some(&mut sig_err)) }
        .context("CUDA/DX12: serialize bindless root signature")?;
    if let Some(err) = sig_err {
        let msg = unsafe {
            let ptr = err.GetBufferPointer() as *const u8;
            let len = err.GetBufferSize();
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len))
        };
        bail!("CUDA/DX12: bindless root signature error: {msg}");
    }
    let sig_blob = sig_blob.context("CUDA/DX12: null bindless root signature blob")?;
    let root_signature: ID3D12RootSignature = unsafe {
        device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(sig_blob.GetBufferPointer() as *const u8, sig_blob.GetBufferSize()),
        )
    }
    .context("CUDA/DX12: CreateRootSignature(bindless)")?;
    Ok(root_signature)
}

fn create_default_u32_buffer(device: &ID3D12Device, num_u32s: u32, name: &str) -> Result<ID3D12Resource> {
    let size = (u64::from(num_u32s) * 4).max(256);
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut resource,
        )
    }
    .with_context(|| format!("CUDA/DX12: CreateCommittedResource({name})"))?;
    let resource = resource.with_context(|| format!("CUDA/DX12: null {name}"))?;
    let _ = unsafe { resource.SetName(&windows::core::HSTRING::from(name)) };
    Ok(resource)
}

fn create_upload_staging(device: &ID3D12Device) -> Result<(ID3D12Resource, *mut u8)> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: FRAME_TABLE_STAGING_BYTES,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut resource,
        )
    }
    .context("CUDA/DX12: CreateCommittedResource(frame-table staging)")?;
    let resource = resource.context("CUDA/DX12: null staging")?;
    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe { resource.Map(0, None, Some(&mut mapped)) }.context("CUDA/DX12: Map frame-table staging")?;
    Ok((resource, mapped as *mut u8))
}

fn transition_buffer(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    if before == after {
        return;
    }
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    unsafe { list.ResourceBarrier(&[barrier]) };
}

pub(super) fn texture_format_to_dxgi(format: TextureFormat) -> Result<DXGI_FORMAT> {
    Ok(match format {
        TextureFormat::R8Unorm => DXGI_FORMAT_R8_UNORM,
        TextureFormat::Rg8Unorm => DXGI_FORMAT_R8G8_UNORM,
        TextureFormat::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        TextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        TextureFormat::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        TextureFormat::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
    })
}

fn filter_to_d3d12(min: FilterMode, mag: FilterMode, mip: FilterMode) -> D3D12_FILTER {
    use FilterMode::*;
    match (min, mag, mip) {
        (Nearest, Nearest, Nearest) => D3D12_FILTER_MIN_MAG_MIP_POINT,
        (Nearest, Nearest, Linear) => D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR,
        (Nearest, Linear, Nearest) => D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT,
        (Nearest, Linear, Linear) => D3D12_FILTER_MIN_POINT_MAG_MIP_LINEAR,
        (Linear, Nearest, Nearest) => D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT,
        (Linear, Nearest, Linear) => D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
        (Linear, Linear, Nearest) => D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
        (Linear, Linear, Linear) => D3D12_FILTER_MIN_MAG_MIP_LINEAR,
    }
}

fn address_to_d3d12(mode: AddressMode) -> D3D12_TEXTURE_ADDRESS_MODE {
    match mode {
        AddressMode::ClampToEdge => D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressMode::Repeat => D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressMode::MirrorRepeat => D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
    }
}

fn sampler_desc_to_d3d12(desc: &SamplerDesc) -> D3D12_SAMPLER_DESC {
    D3D12_SAMPLER_DESC {
        Filter: filter_to_d3d12(desc.min_filter, desc.mag_filter, desc.mipmap_filter),
        AddressU: address_to_d3d12(desc.address_mode_u),
        AddressV: address_to_d3d12(desc.address_mode_v),
        AddressW: address_to_d3d12(desc.address_mode_w),
        MipLODBias: 0.0,
        MaxAnisotropy: desc.max_anisotropy as u32,
        ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        BorderColor: [0.0, 0.0, 0.0, 0.0],
        MinLOD: desc.lod_min_clamp,
        MaxLOD: desc.lod_max_clamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_occupies_and_reclaims() {
        let mut reg = BindlessRegistry::new();
        assert!(reg.occupy(USER_SLOT_BASE, DescriptorCategory::BufferSrv).is_ok());
        assert!(reg.has_resource(USER_SLOT_BASE));
        assert_eq!(reg.slot_kind(USER_SLOT_BASE), Some(BindlessSlotKind::ReadOnlySrv));
        reg.defer_reclaim(USER_SLOT_BASE, false, 10);
        assert!(!reg.has_resource(USER_SLOT_BASE));
        reg.drain_reclaimed(9);
        assert_eq!(reg.pending_reclaim.len(), 1);
        reg.drain_reclaimed(10);
        assert!(reg.pending_reclaim.is_empty());
    }

    #[test]
    fn protocol_slots_are_below_user_base() {
        assert_eq!(USER_SLOT_BASE, 2);
    }

    #[test]
    fn category_maps_to_slot_kind() {
        assert_eq!(DescriptorCategory::BufferSrv.slot_kind(), BindlessSlotKind::ReadOnlySrv);
        assert_eq!(DescriptorCategory::BufferUav.slot_kind(), BindlessSlotKind::StorageUav);
        assert_eq!(
            DescriptorCategory::TextureSrv.slot_kind(),
            BindlessSlotKind::ReadOnlySrv
        );
        assert_eq!(DescriptorCategory::TextureUav.slot_kind(), BindlessSlotKind::StorageUav);
        assert_eq!(DescriptorCategory::BufferCbv.slot_kind(), BindlessSlotKind::UniformCbv);
        assert_eq!(DescriptorCategory::Sampler.slot_kind(), BindlessSlotKind::ReadOnlySrv);
    }

    #[test]
    fn registry_occupies_sampler() {
        let mut reg = BindlessRegistry::new();
        assert!(reg.occupy_sampler(0).is_ok());
        assert!(reg.has_sampler(0));
        reg.defer_reclaim(0, true, 5);
        assert!(!reg.has_sampler(0));
    }

    #[test]
    fn occupy_rejects_out_of_range_slot() {
        let mut reg = BindlessRegistry::new();
        let err = reg
            .occupy(MAX_BINDLESS_CBV_SRV_UAV, DescriptorCategory::BufferSrv)
            .expect_err("slot at heap size must fail");
        assert!(err.to_string().contains("exceeds heap size"), "unexpected error: {err}");
    }
}
