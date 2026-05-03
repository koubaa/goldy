//! Per-device staging buffers for deferred `ComputeCommand::WriteBuffer` uploads on DX12.

use super::types::LogicalDevice;
use anyhow::{Context, Result};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

pub(super) const DEFAULT_STAGING_CHUNK_SIZE: u64 = 256 * 1024;

const COPY_ALIGN: u64 = 256;

struct BeltChunk {
    resource: ID3D12Resource,
    capacity: u64,
    offset: u64,
    mapped: usize,
}

impl BeltChunk {
    fn reset(&mut self) {
        self.offset = 0;
    }

    fn destroy(self) {
        // Explicitly unmap before the COM smart pointer releases the resource.
        unsafe { self.resource.Unmap(0, None) };
    }
}

pub(super) struct StagingBelt {
    free_chunks: Vec<BeltChunk>,
    active_chunks: Vec<BeltChunk>,
    /// Chunks in use until `fence.CompletedValue >= token`.
    in_flight: Vec<(u64, Vec<BeltChunk>)>,
    chunk_size: u64,
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            free_chunks: Vec::new(),
            active_chunks: Vec::new(),
            in_flight: Vec::new(),
            chunk_size,
        }
    }

    /// Call at the beginning of [`super::compute::submit`].
    pub fn reclaim(&mut self, fence: &ID3D12Fence) -> Result<()> {
        let completed = unsafe { fence.GetCompletedValue() };
        let mut i = 0;
        while i < self.in_flight.len() {
            let token = self.in_flight[i].0;
            if completed >= token {
                let (_, mut chunks) = self.in_flight.remove(i);
                for ch in &mut chunks {
                    ch.reset();
                }
                self.free_chunks.extend(chunks);
            } else {
                i += 1;
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
        if data.is_empty() {
            anyhow::bail!("StagingBelt::write: empty data");
        }
        let len = data.len() as u64;

        if let Some(ch) = self.active_chunks.last_mut() {
            let start = align_up(ch.offset, COPY_ALIGN);
            if start + len <= ch.capacity {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        (ch.mapped as *mut u8).add(start as usize),
                        data.len(),
                    );
                }
                ch.offset = start + len;
                return Ok((ch.resource.clone(), start));
            }
        }

        let alloc_size = self.chunk_size.max(align_up(len, COPY_ALIGN));

        // Linear scan from the back (most-recently-freed first) to avoid the
        // push-then-immediately-pop infinite loop the old pattern had when every
        // free chunk was smaller than `len`.
        let mut chunk =
            if let Some(pos) = self.free_chunks.iter().rposition(|c| c.capacity >= len) {
                let mut c = self.free_chunks.swap_remove(pos);
                c.reset();
                c
            } else {
                allocate_chunk(logical_device, alloc_size)?
            };

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (chunk.mapped as *mut u8).add(0usize),
                data.len(),
            );
        }
        chunk.offset = len;
        let res = chunk.resource.clone();
        self.active_chunks.push(chunk);
        Ok((res, 0))
    }

    pub fn finish(&mut self, fence_token: u64) {
        if self.active_chunks.is_empty() {
            return;
        }
        self.in_flight
            .push((fence_token, std::mem::take(&mut self.active_chunks)));
    }

    /// Drop all free chunks whose capacity exceeds `chunk_size`.
    ///
    /// Safe to call at any time: `free_chunks` only holds chunks whose GPU fence has
    /// already signaled, so no GPU wait is needed.
    pub fn trim(&mut self) {
        let chunk_size = self.chunk_size;
        let mut i = 0;
        while i < self.free_chunks.len() {
            if self.free_chunks[i].capacity > chunk_size {
                self.free_chunks.swap_remove(i).destroy();
            } else {
                i += 1;
            }
        }
    }

    pub unsafe fn destroy_all(&mut self) {
        for ch in self.free_chunks.drain(..) {
            ch.destroy();
        }
        for ch in self.active_chunks.drain(..) {
            ch.destroy();
        }
        for (_, mut vec) in self.in_flight.drain(..) {
            for ch in vec.drain(..) {
                ch.destroy();
            }
        }
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

fn allocate_chunk(logical_device: &LogicalDevice, size: u64) -> Result<BeltChunk> {
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

    Ok(BeltChunk {
        resource,
        capacity: size,
        offset: 0,
        mapped: mapped as usize,
    })
}
