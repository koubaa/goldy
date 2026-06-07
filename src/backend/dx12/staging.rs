//! Per-device staging buffers for deferred `GpuCommand::WriteBuffer` uploads on DX12.

use super::super::shared::{BeltChunk as BeltChunkTrait, StagingBeltCore};
use super::types::LogicalDevice;
use crate::tracy_zone;
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
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            core: StagingBeltCore::new(chunk_size),
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
        Ok(())
    }

    /// Copy `data` into belt memory; return Upload resource + source offset for `CopyBufferRegion`.
    pub fn write(&mut self, logical_device: &LogicalDevice, data: &[u8]) -> Result<(ID3D12Resource, u64)> {
        let (idx, start) = self.core.write(data, |size| allocate_chunk(logical_device, size))?;
        Ok((self.core.active[idx].resource.clone(), start))
    }

    pub fn finish(&mut self, fence_token: u64) {
        self.core.finish(fence_token);
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
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
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
    unsafe { resource.Map(0, Some(&no_read), Some(&mut mapped)) }.context("StagingBelt: Map failed")?;

    Ok(Dx12BeltChunk {
        resource,
        capacity: size,
        offset: 0,
        mapped: mapped as usize,
    })
}

// ── TextureStagingPool ────────────────────────────────────────────────────────

/// A permanently-mapped, pre-allocated staging buffer for a single texture upload.
///
/// Unlike `StagingBelt` chunks (bump-allocated), each entry corresponds to one
/// texture region and is returned to the pool as a whole unit.
pub(super) struct TextureStagingEntry {
    pub resource: ID3D12Resource,
    /// Allocated byte capacity of this entry.
    pub capacity: u64,
    /// Permanently-mapped CPU pointer into `resource`, stored as `usize` for `Send`.
    mapped: usize,
}

impl TextureStagingEntry {
    /// Returns the permanently-mapped write pointer for this entry.
    pub fn mapped_ptr(&self) -> *mut u8 {
        self.mapped as *mut u8
    }

    /// Destroy this entry's DX12 resources. Caller must ensure the GPU is idle
    /// with respect to any in-flight copy commands that referenced this buffer.
    pub(super) unsafe fn destroy(self) {
        // Explicitly unmap before the COM smart pointer releases the resource.
        unsafe { self.resource.Unmap(0, None) };
    }
}

// SAFETY: the raw pointer in `mapped` is owned exclusively by this entry and
// only accessed by the CPU owner between acquire and release.
unsafe impl Send for TextureStagingEntry {}
unsafe impl Sync for TextureStagingEntry {}

/// Timeline-gated free-list pool for texture-upload staging buffers.
///
/// Eliminates per-frame `CreateCommittedResource` / resource drop calls by recycling
/// entries whose GPU copy timeline has advanced past their release point.
pub(super) struct TextureStagingPool {
    free: Vec<TextureStagingEntry>,
    in_flight: Vec<(u64, Vec<TextureStagingEntry>)>,
}

impl TextureStagingPool {
    pub fn new() -> Self {
        Self {
            free: Vec::new(),
            in_flight: Vec::new(),
        }
    }

    /// Acquire a staging entry with at least `size` bytes of capacity.
    ///
    /// Returns a recycled free entry on a pool hit. On a miss, allocates a new
    /// permanently-mapped entry. The entry's mapped memory is ready for `memcpy`.
    pub fn acquire(&mut self, logical_device: &LogicalDevice, size: u64) -> Result<TextureStagingEntry> {
        if let Some(pos) = self.free.iter().rposition(|e| e.capacity >= size) {
            let _tz = tracy_zone!("dx12.texture_staging.acquire.hit");
            return Ok(self.free.swap_remove(pos));
        }
        let _tz = tracy_zone!("dx12.texture_staging.acquire.miss");
        allocate_texture_staging_entry(logical_device, size)
    }

    /// Tag `entries` with `fence_token` and move them to in-flight.
    ///
    /// Entries become available for reuse once `reclaim(completed)` is called
    /// with `completed >= fence_token`.
    pub fn release(&mut self, fence_token: u64, entries: Vec<TextureStagingEntry>) {
        if !entries.is_empty() {
            let _tz = tracy_zone!("dx12.texture_staging.release");
            self.in_flight.push((fence_token, entries));
        }
    }

    /// Move entries whose fence has completed back to the free list.
    pub fn reclaim(&mut self, completed_fence_value: u64) {
        let _tz = tracy_zone!("dx12.texture_staging.reclaim");
        let mut i = 0;
        while i < self.in_flight.len() {
            if self.in_flight[i].0 <= completed_fence_value {
                let (_, entries) = self.in_flight.swap_remove(i);
                self.free.extend(entries);
            } else {
                i += 1;
            }
        }
    }

    /// Destroy all free and in-flight entries unconditionally.
    ///
    /// # Safety
    /// Must only be called when the device is idle — all GPU copy commands
    /// referencing in-flight entries must have completed.
    pub unsafe fn destroy_all(&mut self) {
        for entry in self.free.drain(..) {
            entry.destroy();
        }
        for (_, entries) in self.in_flight.drain(..) {
            for entry in entries {
                entry.destroy();
            }
        }
    }
}

fn allocate_texture_staging_entry(logical_device: &LogicalDevice, size: u64) -> Result<TextureStagingEntry> {
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
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
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
    .context("TextureStagingPool: CreateCommittedResource failed")?;
    let resource = resource.context("TextureStagingPool: null upload resource")?;

    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
    let no_read = D3D12_RANGE { Begin: 0, End: 0 };
    unsafe { resource.Map(0, Some(&no_read), Some(&mut mapped)) }.context("TextureStagingPool: Map failed")?;

    Ok(TextureStagingEntry {
        resource,
        capacity: size,
        mapped: mapped as usize,
    })
}
