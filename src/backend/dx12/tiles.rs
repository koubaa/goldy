//! Tile heap pool and [`ID3D12CommandQueue::UpdateTileMappings`] helpers for reserved buffers.
//!
//! ## Ordering vs command lists
//!
//! `UpdateTileMappings` is executed on the [`ID3D12CommandQueue`] directly (not inside a command
//! list). On a given queue, the D3D12 runtime orders these calls relative to
//! [`ID3D12CommandQueue::ExecuteCommandLists`]: mappings become visible to subsequent executes on
//! **that same queue**. Goldy uses a single `DIRECT` queue per device, so tile map/unmap done
//! during reserved-buffer creation, `set_logical_size`, or `hint_unused_above` in the `buffer`
//! module is visible to the next compute/graphics submission without extra barriers. When a
//! reserved buffer is migrated to a committed resource on resize, the copy command list is recorded
//! and executed on that queue while source tiles are still mapped; teardown of the old reserved
//! resource is deferred until the fence passes.

use std::ffi::c_void;

use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// Standard buffer virtual tile size (64 KiB) when tiled resources are supported.
pub const BUFFER_TILE_BYTES: u32 = 65536;

const TILES_PER_CHUNK: u32 = 256;

#[inline]
pub(crate) fn heap_ptr(h: &ID3D12Heap) -> *mut c_void {
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

/// Map many resource tiles in as few `UpdateTileMappings` calls as practical.
///
/// D3D12 allows only one `ID3D12Heap` per call. Within each heap group, consecutive resource tiles
/// whose heap allocations are consecutive 64 KiB tiles are coalesced into a single
/// [`map_tile_run`] (`NumTiles` > 1).
pub(crate) fn map_tiles_batched(
    queue: &ID3D12CommandQueue,
    resource: &ID3D12Resource,
    mappings: &[(u32, ID3D12Heap, u64)],
) -> Result<()> {
    if mappings.is_empty() {
        return Ok(());
    }

    let mut groups: Vec<(ID3D12Heap, Vec<(u32, u64)>)> = Vec::new();
    for (tile, heap, off) in mappings {
        let p = heap_ptr(heap);
        if let Some((_, vec)) = groups
            .iter_mut()
            .find(|(gheap, _)| heap_ptr(gheap) == p)
        {
            vec.push((*tile, *off));
        } else {
            groups.push((heap.clone(), vec![(*tile, *off)]));
        }
    }

    for (heap, mut tiles) in groups {
        tiles.sort_by_key(|(t, _)| *t);
        let mut i = 0usize;
        while i < tiles.len() {
            let (t0, byte0) = tiles[i];
            let heap_tile0 = u32::try_from(byte0 / u64::from(BUFFER_TILE_BYTES))
                .map_err(|_| anyhow::anyhow!("heap tile index"))?;
            let mut run_len = 1u32;
            let mut expect_t = t0.saturating_add(1);
            let mut expect_heap_tile = heap_tile0.saturating_add(1);
            let mut j = i + 1;
            while j < tiles.len() {
                let (tj, bytej) = tiles[j];
                let ht = u32::try_from(bytej / u64::from(BUFFER_TILE_BYTES))
                    .map_err(|_| anyhow::anyhow!("heap tile index"))?;
                if tj == expect_t && ht == expect_heap_tile {
                    run_len = run_len.saturating_add(1);
                    expect_t = expect_t.saturating_add(1);
                    expect_heap_tile = expect_heap_tile.saturating_add(1);
                    j += 1;
                } else {
                    break;
                }
            }
            map_tile_run(queue, resource, t0, run_len, &heap, byte0)?;
            i = j;
        }
    }
    Ok(())
}

/// [`UpdateTileMappings`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandqueue-updatetilemappings)—map contiguous resource tiles to heap offsets.
///
/// `heap_start_offset` is the byte offset into `heap` from [`TileHeapPool`]. D3D12
/// `pHeapRangeStartOffsets` uses **tile indices** (offset ÷ 64 KiB) into that heap.
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
    let range_flag = D3D12_TILE_RANGE_FLAG_NONE;
    let heap_range_start = u32::try_from(heap_start_offset / u64::from(BUFFER_TILE_BYTES))
        .map_err(|_| anyhow::anyhow!("heap tile index"))?;

    unsafe {
        queue.UpdateTileMappings(
            resource,
            1,
            Some(&coord),
            Some(&region),
            heap,
            1,
            Some(&range_flag),
            Some(&heap_range_start),
            Some(&num_tiles),
            D3D12_TILE_MAPPING_FLAG_NONE,
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
///
/// Coalesces adjacent mapped tiles into fewer `UpdateTileMappings` calls.
pub(crate) fn teardown_reserved_mappings(
    queue: &ID3D12CommandQueue,
    pool: &mut Option<TileHeapPool>,
    resource: &ID3D12Resource,
    tiles: &[Option<(ID3D12Heap, u64)>],
) {
    let mut i = 0usize;
    while i < tiles.len() {
        while i < tiles.len() && tiles[i].is_none() {
            i += 1;
        }
        if i >= tiles.len() {
            break;
        }
        let run_start = i;
        while i < tiles.len() && tiles[i].is_some() {
            i += 1;
        }
        let num_tiles = (i - run_start) as u32;
        let _ = unmap_tile_run(queue, resource, run_start as u32, num_tiles);
        if let Some(p) = pool.as_mut() {
            for j in run_start..i {
                if let Some((heap, off)) = &tiles[j] {
                    p.free_tile(heap, *off);
                }
            }
        }
    }
}
