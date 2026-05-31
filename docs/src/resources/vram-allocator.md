# VRAM Allocator

Goldy has three independent GPU memory allocation paths:

1. **Transient sub-allocations** — `TransientAllocator` → `BufferPool` → `Buffer::new`. Pluggable recycling policy, but no control over *where* memory comes from.
2. **Standalone buffers** — consumers call `Buffer::new` directly for bump readback, staging, indirect dispatch, etc.
3. **Textures** — `TexturePool` → `Texture::new`.

The `VramAllocator` trait sits **below** all three, providing a single customization point for where GPU memory comes from. The transient allocator controls *when* to reclaim; the VRAM allocator controls *how* to allocate.

## Architecture

```text
┌────────────────────────────────────────────────┐
│  Consumers (ekrano, user code)                 │
│  ┌─────────────┐  ┌──────────┐  ┌───────────┐ │
│  │ TransientAlloc│ │ Buffer   │  │ Texture   │ │
│  │ (recycling)  │  │ alloc    │  │ alloc     │ │
│  └──────┬───────┘  └────┬─────┘  └─────┬─────┘ │
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
// These go through the device's VramAllocator (parcel carries a deed; Drop notifies the allocator):
let buf = device.alloc_buffer(size, DataAccess::Scattered, None, BufferFlags::empty())?;
let tex = device.alloc_texture(width, height, format, access, flags)?;

// These bypass the allocator (direct backend call; no deed, no allocator accounting on Drop):
let buf = Buffer::new(&device, size, DataAccess::Scattered)?;
let tex = Texture::new(&device, width, height, format, access, flags)?;
```

Only allocations through `device.alloc_*` attach an allocator **deed** and call `VramAllocator::notify_freed` on drop. Borrowing sub-parcels such as `BufferView` never account.

Goldy's built-in pooling systems (`TexturePool`, `BufferPool`, ekrano's `ResourcePool`) all route through the device's allocator automatically.

## Installing a Custom Allocator

Use `Device::with_vram_allocator` to create a device handle that routes all allocations through your allocator:

```rust
use goldy::vram_allocator::{TrackingVramAllocator, DefaultVramAllocator};
use std::sync::Arc;

let tracking = Arc::new(TrackingVramAllocator::with_budget(
    Arc::new(DefaultVramAllocator),
    512 * 1024 * 1024, // 512 MiB budget
));

let device = device.with_vram_allocator(tracking.clone());

// All allocations through `device` now go through the tracking allocator.
// Query live bytes at any time:
println!("GPU memory in use: {} bytes", tracking.allocated_bytes());
```

The original device handle is unaffected — only resources created through the new handle go through the custom allocator.

## Built-in Allocators

| Allocator | Tracking | Budget | Use case |
|-----------|----------|--------|----------|
| `DefaultVramAllocator` | No | No | Zero-overhead passthrough (default) |
| `TrackingVramAllocator` | Yes | Optional | Memory telemetry, budget enforcement |

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
        access: DataAccess,
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
        access: SpatialAccess,
        flags: TextureFlags,
    ) -> anyhow::Result<Texture> {
        Texture::new(device, width, height, format, access, flags)
    }

    fn notify_freed(
        &self,
        reserved: u64,
        _committed: u64,
        _kind: goldy::vram_allocator::ParcelKind,
    ) {
        // Decrement your tracked bytes (reserved is the parcel's reserved backing size).
        let _ = reserved;
    }

    fn name(&self) -> &'static str { "my-allocator" }
}
```

All trait methods have default implementations that delegate to the standard constructors, so you only need to override the methods you care about. Parcels allocated via `Device::alloc_buffer` / `alloc_texture` notify your allocator automatically on drop when you install it with `with_vram_allocator`.

## Relationship to TransientAllocator

The two traits are complementary:

- **`TransientAllocator`** controls *recycling policy*: when to reclaim per-frame scratch memory, how many frames can overlap, how to handle epoch-based reclamation.
- **`VramAllocator`** controls *allocation source*: where the backing memory comes from, budget enforcement, unified telemetry.

`TransientAllocator` implementations create their backing `BufferPool` regions through the device's `VramAllocator`. This means a custom VRAM allocator automatically intercepts transient allocations too.

## Future Directions

The `VramAllocator` trait is designed as a foundation for backend-native memory strategies:

- **Metal `makeAliasable` placement heaps** — alias transient buffers and textures into the same physical pages when their lifetimes don't overlap.
- **Vulkan VMA integration** — delegate to the Vulkan Memory Allocator for sub-allocation and defragmentation.
- **Adaptive budgeting** — dynamically adjust pipeline depth based on available GPU memory instead of using a hardcoded constant.
- **Defragmentation** — move allocations and update bindless descriptors atomically.

See [issue #142](https://github.com/koubaa/goldy/issues/142) for the full design discussion.
