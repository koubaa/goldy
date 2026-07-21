# VRAM Allocator

All GPU buffer and texture allocations route through the device's installed `VramAllocator`:

1. **Pool parcels** — `RetainedPool` / `TransientPool` → `Device::alloc_buffer` / `alloc_texture`
2. **Standalone buffers** — `Device::alloc_buffer` / `alloc_buffer_with_*`

The `VramAllocator` trait sits **below** those paths, providing a single customization point for where GPU memory comes from. The pools control *when* to reclaim; the VRAM allocator controls *how* to allocate.

## Architecture

```text
┌────────────────────────────────────────────────┐
│  Consumers
│  ┌─────────────┐  ┌──────────┐  ┌───────────┐ │
│  │ TransientPool│  │RetainedPool│ │ Direct   │ │
│  │ (leases)     │  │ (deeds)    │ │ alloc_*  │ │
│  └──────┬───────┘  └────┬───────┘ └────┬─────┘ │
│         │               │              │       │
│  ┌──────▼───────────────▼──────────────▼──────┐│
│  │           VramAllocator trait               ││
│  │  alloc_buffer / alloc_texture / budget      ││
│  └──────────────────┬──────────────────────────┘│
│                     │                           │
│  ┌──────────────────▼──────────────────────────┐│
│  │           GpuBackend (Metal/Vulkan/DX12)    ││
│  └─────────────────────────────────────────────┘│
└────────────────────────────────────────────────┘
```

## Using the Default Allocator

Every `Device` ships with a `DefaultVramAllocator` that delegates directly to the backend with zero overhead. You can allocate through it explicitly via convenience methods:

```rust
let buf = device.alloc_buffer(size, BufferKind::Scattered, None, BufferFlags::empty())?;
let tex = device.alloc_texture(width, height, format, access, flags)?;
```

Every `device.alloc_*` call attaches an allocator **deed** and calls `VramAllocator::notify_freed` on drop. Borrowing sub-parcels such as `BufferView` never account.

Goldy's pooling paths (`RetainedPool`, `TransientPool`) all route through the device's allocator automatically.

## Allocation Policy (Tracking and Budget)

Install a [`BudgetPolicy`](../../src/allocation_policy.rs) on the default allocator to track live GPU bytes and optionally enforce a cap:

```rust
use goldy::BudgetPolicy;
use std::sync::Arc;

let policy = Arc::new(BudgetPolicy::with_budget(512 * 1024 * 1024)); // 512 MiB budget
device.set_allocation_policy(policy.clone())?;

// All allocations through `device` now update the policy counters.
println!("GPU memory in use: {} bytes", policy.allocated_bytes());
```

Use `BudgetPolicy::new()` when you only need telemetry without a hard budget.

## Installing a Custom Allocator

Use `Device::with_vram_allocator` to create a device handle that routes all allocations through a custom [`VramAllocator`] implementation (for example, a backend-native heap strategy):

```rust
use goldy::vram_allocator::{DefaultVramAllocator, VramAllocator};
use std::sync::Arc;

let custom = Arc::new(DefaultVramAllocator::new());
let aliased = device.with_vram_allocator(custom);
```

The original device handle is unaffected — only resources created through the new handle go through the custom allocator.

## Built-in Allocators

| Component | Tracking | Budget | Use case |
|-----------|----------|--------|----------|
| `DefaultVramAllocator` | No | No | Zero-overhead passthrough (default) |
| `BudgetPolicy` on default allocator | Yes | Optional | Memory telemetry, budget enforcement |

## Implementing a Custom Allocator

Implement the `VramAllocator` trait to intercept all GPU memory allocation:

```rust
use goldy::vram_allocator::VramAllocator;
use goldy::buffer::Buffer;
use goldy::texture::Texture;
use goldy::device::Device;
use goldy::types::*;

struct MyAllocator { /* ... */ }

impl VramAllocator for MyAllocator {
    fn alloc_buffer(
        &self,
        device: &Device,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> anyhow::Result<Buffer> {
        // Custom allocation logic here.
        // Fall back to the default:
        Buffer::new_with_stride_and_flags(device, size, access, element_stride, flags)
    }

    fn alloc_texture(
        &self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> anyhow::Result<Texture> {
        Texture::new(device, width, height, format, access, flags)
    }

    fn notify_freed(
        &self,
        reserved: u64,
        _committed: u64,
        _kind: goldy::vram_allocator::ParcelType,
    ) {
        // Decrement your tracked bytes (reserved is the parcel's reserved backing size).
        let _ = reserved;
    }

    fn name(&self) -> &'static str { "my-allocator" }
}
```

All trait methods have default implementations that delegate to the standard constructors, so you only need to override the methods you care about. Parcels allocated via `Device::alloc_buffer` / `alloc_texture` notify your allocator automatically on drop when you install it with `with_vram_allocator`.

## Relationship to pools

The pools and the VRAM allocator are complementary:

- **`RetainedPool` / `TransientPool`** control *recycling policy*: when deeds and leases may be reissued after GPU retirement.
- **`VramAllocator`** controls *allocation source*: where the backing memory comes from, budget enforcement, unified telemetry.

Both pools allocate through `Device::alloc_*`, so a custom VRAM allocator automatically intercepts retained and transient parcels.

## Future Directions

The `VramAllocator` trait is designed as a foundation for backend-native memory strategies:

- **Metal `makeAliasable` placement heaps** — alias transient buffers and textures into the same physical pages when their lifetimes don't overlap.
- **Vulkan VMA integration** — delegate to the Vulkan Memory Allocator for sub-allocation and defragmentation.
- **Adaptive budgeting** — dynamically adjust pipeline depth based on available GPU memory instead of using a hardcoded constant.
- **Defragmentation** — move allocations and update bindless descriptors atomically.

See [issue #142](https://github.com/koubaa/goldy/issues/142) for the full design discussion.
