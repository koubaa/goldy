# Transient Allocation

Rendering pipelines allocate many short-lived GPU buffers each frame — scratch storage, per-pass intermediates, vertex streams. After submission these allocations are dead until the GPU finishes, at which point the memory can be recycled. How and when that recycling happens determines whether the CPU can overlap with the GPU or must wait.

Goldy's `TransientAllocator` trait defines a pluggable strategy for this pattern. Consumers call three methods per frame (`begin_frame` → `alloc` → `end_frame`) and the strategy handles everything else: growth, synchronization, reclamation.

## Choosing a Strategy

Two strategies are built in. More can be added by implementing the `TransientAllocator` trait.

| Strategy | Pipeline depth | Memory | Best for |
|----------|---------------|--------|----------|
| `BumpReset` | 1 (serialized) | Lowest | Minimum-memory baseline, depth=1 workloads, diagnostic isolation |
| `Heap` | Pipelined | Peak ≈ live working set | Default for overlapping CPU encoding with GPU execution |

The strategy is selected at construction time via `TransientAllocatorStrategy`. The default is `Heap` — the pipelined free-list strategy.

## API Overview

### Construction

```rust
use goldy::{TransientAllocatorStrategy, TransientAllocatorConfig};

// Explicit strategy
let strategy = TransientAllocatorStrategy::Heap;
let mut allocator = strategy.create(&device, TransientAllocatorConfig {
    initial_size: 4 * 1024 * 1024,
    alignment: 256,
    flags: BufferFlags::GPU_ONLY,
})?;

// Or use the default strategy (Heap)
let mut allocator = TransientAllocatorStrategy::default().create(
    &device,
    TransientAllocatorConfig::default(),
)?;
```

### Per-frame lifecycle

```rust
// 1. Begin frame — reclaim completed ranges, grow if needed
allocator.begin_frame(&device, estimated_frame_bytes)?;

// 2. Allocate — returns a BufferView with its own bindless descriptor
let tiles = allocator.alloc(&device, tile_bytes, Some(tile_stride))?;
let segments = allocator.alloc(&device, seg_bytes, Some(seg_stride))?;

// ... build and submit GPU work ...
let timeline = device.submit(&graph)?;

// 3. End frame — stamp mid-frame frees with this epoch
allocator.end_frame(timeline);
```

### Diagnostics

```rust
allocator.name();            // "bump_reset" or "heap"
allocator.capacity();        // total GPU bytes held
allocator.used_this_frame(); // bytes allocated so far this frame
```

## How the Strategies Work

### BumpReset

The minimum-memory strategy. A single `BufferPool` backs all allocations. At the start of each frame, `begin_frame` blocks until the previous frame's GPU epoch has been signaled, then resets the bump pointer to zero.

```
Frame N:   [  CPU encode  ]────submit────►[  GPU execute  ]
Frame N+1:                      wait ◄────┘[  CPU encode  ]──►...
```

This is equivalent to a per-thread arena allocator with synchronous reset — the CPU cannot begin recording frame N+1 until frame N's GPU work completes. It uses the absolute minimum memory (one pool, no free-list bookkeeping), making it ideal for pipeline depth = 1, profiling baselines, and isolating reclamation-timing bugs.

### Heap

A single monolithic buffer with a best-fit free list and coalescing. Mid-frame `free()` calls return sub-allocations to a deferred-free queue keyed by GPU timeline; once the GPU retires past that epoch, ranges merge back into the free list for reuse within the same or subsequent frames.

```
Frame N:   [  CPU encode  ]──submit──►[  GPU execute  ]
Frame N+1: [  CPU encode  ]──submit──►     [  GPU execute  ]
Frame N+2: [  CPU encode  ]──...           reclaim N's ranges ◄─┘
```

Peak memory is proportional to the peak *live* working set, not the sum of all allocations. `end_frame` may arrive after the next `begin_frame` on the surface path — mid-frame frees with `epoch=None` are stamped in `end_frame`.

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `initial_size` | 64 KiB | Backing storage allocated on first frame. The allocator grows reactively and auto-shrinks after warmup. |
| `alignment` | 256 | Sub-allocation alignment (must be power of two). 256 covers all known `minStorageBufferOffsetAlignment` values. |
| `flags` | `GPU_ONLY` | `BufferFlags` applied to backing storage |

## Implementing a Custom Strategy

Implement the `TransientAllocator` trait:

```rust
pub trait TransientAllocator: Send {
    fn begin_frame(&mut self, device: &Device, hint_size: u64) -> Result<()>;
    fn alloc(&mut self, device: &Device, size: u64, element_stride: Option<u32>) -> Result<BufferView>;
    fn end_frame(&mut self, epoch: TimelineValue);
    fn capacity(&self) -> u64;
    fn name(&self) -> &'static str;

    // Optional overrides with defaults
    fn used_this_frame(&self) -> u64 { 0 }
    fn hint_unused_above(&mut self, _offset: u64) {}
    fn free(&mut self, _offset: u64, _size: u64, _epoch: Option<TimelineValue>) {}
    fn clear(&mut self) {}
}
```

Possible future strategies:

- **PerNameRecycle** — per-`(name, size_class)` buffer pool modeled after Vello's `ResourcePool`. Trades the single-address-range property for simpler reasoning about per-buffer lifetimes.
- **BackendNative** — delegate to Metal's `makeAliasable` placement heaps, Vulkan sparse rebind, or DX12 `UpdateTileMappings` for zero-copy region recycling at the driver level.
- **DebugSequential** — fresh `Buffer` per allocation, no reuse. Catches use-after-free hazards at the cost of allocation overhead.

## Relationship to Other Pooling Types

| Type | Scope | Lifecycle |
|------|-------|-----------|
| Scattered bump arena (`BufferPool`) | Internal sub-allocation | Used by `TransientAllocator` strategies |
| `FrameOrchestrator<T>` | Frame-slot ring with typed cleanup payloads | Epoch-aware, depth-capped, callback-driven |
| `TransientPool` | Context-owned lease recycle bins | Epoch-gated; deeds/leases are the client API |
| **`TransientAllocator`** | **Pluggable per-frame bump allocation** | **Epoch-aware, strategy-selectable** |

`TransientAllocator` *uses* the scattered bump arena internally (each strategy is backed by one), but adds lifecycle management that the bump cursor alone does not provide.
