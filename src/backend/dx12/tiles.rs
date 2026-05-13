//! Tile heap pool and [`ID3D12CommandQueue::UpdateTileMappings`] helpers for reserved buffers.

use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// Standard buffer virtual tile size (64 KiB) when tiled resources are supported.
pub const BUFFER_TILE_BYTES: u32 = 65536;

const TILES_PER_CHUNK: u32 = 256;

#[inline]
fn heap_ptr(h: &ID3D12Heap) -> *mut core::ffi::c_void {
    h.as_raw()
}

#[derive(Debug)]
struct HeapChunk {
    heap: ID3D12Heap,
    tile_stride: u64,
    num_tiles: u32,
    free: Vec<u32>,
}

/// Sub-allocates 64 KiB tiles from [`ID3D12Heap`] chunks for reserved buffer mappings.
pub(crate) struct TileHeapPool {
    chunks: Vec<HeapChunk>,
}

impl TileHeapPool {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    fn push_chunk(device: &ID3D12Device10) -> Result<HeapChunk> {
        let size = BUFFER_TILE_BYTES as u64 * TILES_PER_CHUNK as u64;
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: size,
            Properties: heap_props,
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        };
        let mut heap = None;
        unsafe { device.CreateHeap(&desc, &mut heap) }.context("TileHeapPool CreateHeap")?;
        let heap = heap.context("CreateHeap returned null")?;
        // Tile 0 returned to caller; remaining indices go to the free list.
        let mut free: Vec<u32> = (1..TILES_PER_CHUNK).collect();
        free.reverse();
        Ok(HeapChunk {
            heap,
            tile_stride: BUFFER_TILE_BYTES as u64,
            num_tiles: TILES_PER_CHUNK,
            free,
        })
    }

    pub fn alloc_tile(&mut self, device: &ID3D12Device10) -> Result<(ID3D12Heap, u64)> {
        for ch in &mut self.chunks {
            if let Some(idx) = ch.free.pop() {
                let offset = u64::from(idx).saturating_mul(ch.tile_stride);
                return Ok((ch.heap.clone(), offset));
            }
        }
        let ch = Self::push_chunk(device)?;
        let heap = ch.heap.clone();
        self.chunks.push(ch);
        Ok((heap, 0))
    }

    pub fn free_tile(&mut self, heap: &ID3D12Heap, offset: u64) {
        let hp = heap_ptr(heap);
        for ch in &mut self.chunks {
            if heap_ptr(&ch.heap) == hp {
                debug_assert!(offset % ch.tile_stride == 0);
                let idx = u32::try_from(offset / ch.tile_stride).unwrap_or(0);
                debug_assert!(idx < ch.num_tiles);
                ch.free.push(idx);
                return;
            }
        }
        tracing::warn!("TileHeapPool::free_tile: unknown heap (leak?)");
    }
}

pub(crate) fn align_reserved_cap(cap: u64) -> u64 {
    let b = u64::from(BUFFER_TILE_BYTES);
    cap.div_ceil(b) * b
}

pub(crate) fn num_tiles_for_bytes(size: u64) -> u32 {
    if size == 0 {
        return 0;
    }
    u32::try_from(size.div_ceil(u64::from(BUFFER_TILE_BYTES))).unwrap_or(u32::MAX)
}

pub(crate) fn tiles_needed_for_logical_size(size: u64) -> u32 {
    num_tiles_for_bytes(size)
}

/// [`UpdateTileMappings`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandqueue-updatetilemappings)—map contiguous resource tiles to heap offsets.
pub(crate) fn map_tile_run(
    queue: &ID3D12CommandQueue,
    resource: &ID3D12Resource,
    start_tile: u32,
    num_tiles: u32,
    heap: &ID3D12Heap,
    heap_start_offset: u64,
) -> Result<()> {
    if num_tiles == 0 {
        return Ok(());
    }
    let coord = D3D12_TILED_RESOURCE_COORDINATE {
        X: start_tile,
        Y: 0,
        Z: 0,
        Subresource: 0,
    };
    let region = D3D12_TILE_REGION_SIZE {
        NumTiles: num_tiles,
        UseBox: false.into(),
        Width: 0,
        Height: 0,
        Depth: 0,
    };
    let range_flag = D3D12_TILE_RANGE_FLAGS::default();
    let heap_off = u32::try_from(heap_start_offset)
        .map_err(|_| anyhow::anyhow!("heap offset exceeds UINT"))?;
    unsafe {
        queue.UpdateTileMappings(
            resource,
            1,
            Some(&coord),
            Some(&region),
            heap,
            1,
            Some(&range_flag),
            Some(&heap_off),
            Some(&num_tiles),
            D3D12_TILE_MAPPING_FLAGS::default(),
        );
    }
    Ok(())
}

/// Unmap a contiguous run (NULL heap mappings).
pub(crate) fn unmap_tile_run(
    queue: &ID3D12CommandQueue,
    resource: &ID3D12Resource,
    start_tile: u32,
    num_tiles: u32,
) -> Result<()> {
    if num_tiles == 0 {
        return Ok(());
    }
    let coord = D3D12_TILED_RESOURCE_COORDINATE {
        X: start_tile,
        Y: 0,
        Z: 0,
        Subresource: 0,
    };
    let region = D3D12_TILE_REGION_SIZE {
        NumTiles: num_tiles,
        UseBox: false.into(),
        Width: 0,
        Height: 0,
        Depth: 0,
    };
    let range_flag = D3D12_TILE_RANGE_FLAG_NULL;
    unsafe {
        queue.UpdateTileMappings(
            resource,
            1,
            Some(&coord),
            Some(&region),
            None,
            1,
            Some(&range_flag),
            None,
            Some(&num_tiles),
            D3D12_TILE_MAPPING_FLAGS::default(),
        );
    }
    Ok(())
}

/// Unmap every mapped tile and return heap slots to `pool` (if any).
pub(crate) fn teardown_reserved_mappings(
    queue: &ID3D12CommandQueue,
    pool: &mut Option<TileHeapPool>,
    resource: &ID3D12Resource,
    tiles: &[Option<(ID3D12Heap, u64)>],
) {
    for (i, slot) in tiles.iter().enumerate() {
        if let Some((heap, off)) = slot {
            let _ = unmap_tile_run(queue, resource, i as u32, 1);
            if let Some(p) = pool.as_mut() {
                p.free_tile(heap, *off);
            }
        }
    }
}
