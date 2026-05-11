//! Per-device staging buffers for deferred `GpuCommand::WriteBuffer` uploads on DX12.

use super::super::shared::{BeltChunk as BeltChunkTrait, StagingBeltCore};
use super::types::LogicalDevice;
use anyhow::{Context, Result};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

pub(super) const DEFAULT_STAGING_CHUNK_SIZE: u64 = super::super::shared::DEFAULT_STAGING_CHUNK_SIZE;

struct Dx12BeltChunk {
    resource: ID3D12Resource,
    capacity: u64,
    offset: u64,
    mapped: usize,
}

impl BeltChunkTrait for Dx12BeltChunk {
    fn capacity(&self) -> u64 {
        self.capacity
    }
    fn offset(&self) -> u64 {
        self.offset
    }
    fn offset_mut(&mut self) -> &mut u64 {
        &mut self.offset
    }
    fn mapped_ptr(&self) -> *mut u8 {
        self.mapped as *mut u8
    }
}

impl Dx12BeltChunk {
    fn destroy(self) {
        // Explicitly unmap before the COM smart pointer releases the resource.
        unsafe { self.resource.Unmap(0, None) };
    }
}

pub(super) struct StagingBelt {
    core: StagingBeltCore<Dx12BeltChunk>,
    /// Standalone UPLOAD resources (e.g. texture copy staging) until fence completes.
    standalone_in_flight: Vec<(u64, Vec<ID3D12Resource>)>,
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            core: StagingBeltCore::new(chunk_size),
            standalone_in_flight: Vec::new(),
        }
    }

    /// Call at the beginning of [`super::compute::submit`].
    pub fn reclaim(&mut self, fence: &ID3D12Fence) -> Result<()> {
        let completed = unsafe { fence.GetCompletedValue() };
        let mut i = 0;
        while i < self.core.in_flight.len() {
            let token = self.core.in_flight[i].0;
            if completed >= token {
                let (_, mut chunks) = self.core.in_flight.remove(i);
                for ch in &mut chunks {
                    ch.reset();
                }
                self.core.free.extend(chunks);
            } else {
                i += 1;
            }
        }
        let mut j = 0;
        while j < self.standalone_in_flight.len() {
            let token = self.standalone_in_flight[j].0;
            if completed >= token {
                self.standalone_in_flight.remove(j);
                // Resources dropped here; GPU has finished the submission that used them.
            } else {
                j += 1;
            }
        }
        Ok(())
    }

    /// Copy `data` into belt memory; return Upload resource + source offset for `CopyBufferRegion`.
    pub fn write(
        &mut self,
        logical_device: &LogicalDevice,
        data: &[u8],
    ) -> Result<(ID3D12Resource, u64)> {
        let (idx, start) = self
            .core
            .write(data, |size| allocate_chunk(logical_device, size))?;
        Ok((self.core.active[idx].resource.clone(), start))
    }

    pub fn finish(&mut self, fence_token: u64) {
        self.core.finish(fence_token);
    }

    /// Retain standalone upload buffers (texture staging, etc.) until `fence_token` completes.
    pub fn defer_standalone_resources(&mut self, fence_token: u64, resources: Vec<ID3D12Resource>) {
        if resources.is_empty() {
            return;
        }
        self.standalone_in_flight.push((fence_token, resources));
    }

    /// Drop all free chunks whose capacity exceeds `chunk_size`.
    ///
    /// Safe to call at any time: `free_chunks` only holds chunks whose GPU fence has
    /// already signaled, so no GPU wait is needed.
    pub fn trim(&mut self) {
        self.core.trim_free(|ch| ch.destroy());
    }

    pub unsafe fn destroy_all(&mut self) {
        self.core.destroy_all(|ch| ch.destroy());
        self.standalone_in_flight.clear();
    }
}

fn allocate_chunk(logical_device: &LogicalDevice, size: u64) -> Result<Dx12BeltChunk> {
    let upload_heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
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
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };

    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        logical_device.device.CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut resource,
        )
    }
    .context("StagingBelt: CreateCommittedResource failed")?;
    let resource = resource.context("StagingBelt: null upload resource")?;

    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
    let no_read = D3D12_RANGE { Begin: 0, End: 0 };
    unsafe { resource.Map(0, Some(&no_read), Some(&mut mapped)) }
        .context("StagingBelt: Map failed")?;

    Ok(Dx12BeltChunk {
        resource,
        capacity: size,
        offset: 0,
        mapped: mapped as usize,
    })
}
