# Transient Allocation

Rendering pipelines allocate many short-lived GPU buffers each frame — scratch storage, per-pass intermediates, vertex streams. After submission these allocations are dead until the GPU finishes, at which point the memory can be recycled. How and when that recycling happens determines whether the CPU can overlap with the GPU or must wait.

Goldy's `TransientAllocator` trait defines a pluggable strategy for this pattern. Consumers call three methods per frame (`begin_frame` → `alloc` → `end_frame`) and the strategy handles everything else: growth, synchronization, reclamation.

## Choosing a Strategy

Two strategies are built in. More can be added by implementing the `TransientAllocator` trait.

| Strategy | Pipeline depth | Memory | Best for |
|----------|---------------|--------|----------|
| `BumpReset` | 1 (serialized) | Lowest | Debugging, validation, single-frame-at-a-time workloads |
| `EpochRegions` | Adaptive (up to `max_regions`) | Moderate | Pipelined rendering, overlapping CPU encoding with GPU execution |

The strategy is selected at construction time via `TransientAllocatorStrategy`. The default is `EpochRegions` — the pipelined strategy.

## API Overview

### Construction

```rust
use goldy::{TransientAllocatorStrategy, TransientAllocatorConfig};

// Explicit strategy
let strategy = TransientAllocatorStrategy::EpochRegions;
let mut allocator = strategy.create(&device, TransientAllocatorConfig {
    initial_size: 4 * 1024 * 1024,
    expected_max: 64 * 1024 * 1024,
    min_region_size: 4 * 1024 * 1024,
    max_regions: 4,
    alignment: 256,
    flags: BufferFlags::GPU_ONLY,
})?;

// Or use the default strategy (EpochRegions)
let mut allocator = TransientAllocatorStrategy::default().create(
    &device,
    TransientAllocatorConfig::default(),
)?;
```

### Per-frame lifecycle

```rust
// 1. Begin frame — reclaim completed regions, grow if needed
allocator.begin_frame(&device, estimated_frame_bytes)?;

// 2. Allocate — returns a BufferView with its own bindless descriptor
let tiles = allocator.alloc(&device, tile_bytes, Some(tile_stride))?;
let segments = allocator.alloc(&device, seg_bytes, Some(seg_stride))?;

// ... build and submit GPU work ...
let timeline = device.submit(&graph)?;

// 3. End frame — tag all active regions with this epoch
allocator.end_frame(timeline);
```

### Diagnostics

```rust
allocator.name();            // "bump_reset" or "epoch_regions"
allocator.capacity();        // total GPU bytes held
allocator.used_this_frame(); // bytes allocated so far this frame
```

## How the Strategies Work

### BumpReset

The simplest correct strategy. A single `BufferPool` backs all allocations. At the start of each frame, `begin_frame` blocks until the previous frame's GPU epoch has been signaled, then resets the bump pointer to zero.

```
Frame N:   [  CPU encode  ]────submit────►[  GPU execute  ]
Frame N+1:                      wait ◄────┘[  CPU encode  ]──►...
```

This is equivalent to a per-thread arena allocator with synchronous `free` — the CPU cannot begin recording frame N+1 until frame N's GPU work completes. It uses the absolute minimum memory (one pool, no copies), making it ideal for profiling baselines and for workloads where the CPU is not the bottleneck.

### EpochRegions

Multiple bump regions, each tagged with a `TimelineValue` epoch. Regions transition through four states:

```
   ┌─────┐   begin_frame    ┌────────┐   end_frame    ┌─────────┐
   │Empty│ ────────────────► │ Active │ ─────────────► │ Retired │
   └─────┘                   └────────┘                └────┬────┘
      ▲                          │ begin_frame               │
      │                          │ (deferred end_frame)      │
      │                          ▼                           │
      │                      ┌─────────┐  end_frame          │
      │                      │ Pending │ ───────────────────►│
      │                      └─────────┘                     │
      │              gpu_progress() >= epoch                 │
      └──────────────────────────────────────────────────────┘
```

- **Empty** — bump pointer at zero, ready for allocation.
- **Active** — being allocated from this frame. Multiple regions may be active if a single frame spills beyond one region's capacity.
- **Pending** — frame finished recording but the epoch is not yet known (surface-presentation path where `end_frame` is deferred until after `Frame::present`). Promoted to Retired when the deferred `end_frame` supplies the epoch.
- **Retired** — written by an earlier frame, waiting for the GPU to catch up. Becomes Empty once `device.gpu_progress() >= epoch`.

`begin_frame` promotes completed retirees without blocking. Only when the region cap (`max_regions`) is hit and no retirees are complete does the strategy fall back to a synchronous wait — this is the "safety valve" that prevents unbounded memory growth.

```
Frame N:   [  CPU encode  ]──submit──►[  GPU execute  ]
Frame N+1: [  CPU encode  ]──submit──►     [  GPU execute  ]
Frame N+2: [  CPU encode  ]──...           reclaim N's regions ◄─┘
```

The closest CPU-side analogue is epoch-based reclamation (EBR), the same pattern used by crossbeam's `Collector`, the Linux kernel's RCU, and JVM region-based garbage collectors like G1/ZGC.

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `initial_size` | 64 KiB | Backing storage allocated on first frame |
| `expected_max` | 16 MiB | Capacity hint for backends that pre-reserve virtual address range (e.g. Metal placement heaps) |
| `min_region_size` | 4 MiB | Minimum bytes per region for `EpochRegions`. Smaller = finer reclamation granularity, more regions. |
| `max_regions` | 3 | Pipeline depth cap for `EpochRegions`. `BumpReset` ignores this. |
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
| `BufferPool` | Manual sub-allocation from one buffer | Caller manages reset timing |
| `FrameOrchestrator<T>` | Frame-slot ring with typed cleanup payloads | Epoch-aware, depth-capped, callback-driven |
| `TexturePool` | Acquire/release cache for textures | Keyed recycling, no sub-allocation |
| **`TransientAllocator`** | **Pluggable per-frame bump allocation** | **Epoch-aware, strategy-selectable** |

`TransientAllocator` *uses* `BufferPool` internally (each strategy/region is backed by one), but adds lifecycle management that `BufferPool` alone does not provide.
