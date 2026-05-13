# Pooling and Sub-Allocation

GPU resource allocation is expensive. Creating many small buffers or textures each frame produces allocation overhead, descriptor churn, and VRAM fragmentation. Goldy provides three pooling types to amortize these costs.

## BufferPool

`BufferPool` sub-allocates typed regions from a single large `DataAccess::Scattered` backing buffer. Each region gets its own bindless descriptor, so shaders see independent zero-based buffers.

### Creating a Pool

```rust
use goldy::BufferPool;

let mut pool = BufferPool::new(&device, 1024 * 1024)?; // 1 MB pool
```

The backing buffer uses `DataAccess::Scattered` and a default sub-allocation alignment of 256 bytes (satisfies `minStorageBufferOffsetAlignment` on all known Vulkan/DX12 hardware).

For custom alignment:

```rust
let mut pool = BufferPool::with_alignment(&device, total_size, 512)?;
```

### Allocating Regions

Typed allocation — stride is inferred from `T`:

```rust
let tiles: BufferView = pool.alloc::<[u32; 2]>(1024)?;    // 1024 elements
let segments: BufferView = pool.alloc::<[f32; 6]>(4096)?;  // 4096 elements
```

Allocate and fill in one call:

```rust
let data = vec![[1.0f32, 0.0, 0.0]; 100];
let view: BufferView = pool.alloc_with_data(&data)?;
```

Raw byte allocation with explicit stride:

```rust
let view = pool.alloc_bytes(4096, Some(16))?;
```

Each allocation is aligned to satisfy both the pool alignment (256) and `offset % element_stride == 0` (required by DX12 `StructuredBuffer` views).

### Using Allocated Views

Every `BufferView` from a pool has its own bindless descriptor. Bind it like any buffer:

```rust
let tile_handle = tiles.bindless_handle().unwrap();
pass.bind_resources_typed(&[tile_handle]);

// Or as a vertex/index buffer
pass.set_vertex_buffer(0, &tiles);
```

Write data into a view:

```rust
view.write_data(&new_data)?;
```

### Sizing a Pool

Use `BufferPool::padded_size` to compute the exact byte capacity needed for a known set of allocations, including alignment padding:

```rust
let size = BufferPool::padded_size(&[
    (1024, std::mem::size_of::<[u32; 2]>()),  // tiles
    (4096, std::mem::size_of::<[f32; 6]>()),  // segments
    (512,  std::mem::size_of::<u32>()),        // indices
]);
let mut pool = BufferPool::new(&device, size)?;
```

### Resetting

`reset()` moves the bump pointer back to zero without invalidating existing views. Use for frame-to-frame reuse when previous views are no longer in flight.

```rust
pool.reset();
```

### Pool Queries

```rust
pool.used();             // bytes currently allocated
pool.capacity();         // total pool size
pool.remaining();        // bytes available
pool.backing_buffer();   // reference to the underlying Buffer
```

## BufferPoolRing

`BufferPoolRing` is a fixed-size ring of `BufferPool`s for double- (or N-) buffered rendering. Each frame advances to the next slot, and the pool that was active N frames ago is safe to reset because its GPU work has completed.

### Usage

```rust
use goldy::BufferPoolRing;

let mut ring = BufferPoolRing::<2>::new(); // double-buffered

// Each frame:
ring.advance();
ring.prepare(&device, needed_bytes)?;

if ring.take_clear_flag() {
    // New backing buffer was allocated — zero-fill it
    let pool = ring.current_mut().unwrap();
    pool.backing_buffer().clear(&device, 0, pool.capacity())?;
}

let pool = ring.current_mut().unwrap();
let view = pool.alloc::<[f32; 4]>(256)?;
```

### How It Works

1. **`advance()`** — rotates to the next pool slot (call once at frame start)
2. **`prepare(device, size)`** — ensures the current slot has at least `size` bytes. Resets the pool if large enough, or allocates a new one if not. Sets a clear flag when a new allocation occurs.
3. **`take_clear_flag()`** — returns `true` exactly once after `prepare` allocates a new backing buffer. Issue a `clear_buffer` for the backing when this fires.
4. **`current_mut()`** / **`current()`** — access the current frame's pool

### Bounded Prepare

`prepare_bounded` adds an optional upper bound. If the current pool exceeds `max_size`, it is reallocated at `size`, enabling hysteresis-based shrinking:

```rust
ring.prepare_bounded(&device, needed_size, Some(max_size))?;
```

### Cleanup

```rust
ring.clear(); // drop all pools and reset state
```

## TexturePool

`TexturePool` caches released textures for reuse, avoiding repeated GPU allocation and deallocation. This is particularly valuable on DX12 where texture allocation involves descriptor heap management.

### Creating a Pool

```rust
use goldy::{TexturePool, TexturePoolConfig};

let mut pool = TexturePool::new(TexturePoolConfig {
    max_per_key: 4, // keep up to 4 textures per (width, height, format, access, flags) key
});

// Or use defaults (max_per_key = 8)
let mut pool = TexturePool::default();
```

### Acquire and Release

```rust
use goldy::{SpatialAccess, TextureFormat, TextureFlags};

// Acquire — returns a pooled texture if available, otherwise creates a new one
let texture = pool.acquire(
    &device,
    1920, 1080,
    TextureFormat::Rgba16Float,
    SpatialAccess::Direct,
    TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
)?;

// ... use the texture for this frame's work ...

// Release — return to pool after GPU work completes
pool.release(texture);
```

Borrowed textures (`texture.borrow()`) are silently dropped on release and not pooled.

### Pool Key

Textures are keyed by `(width, height, format, access, flags)`. Acquiring a texture only matches exact keys — a 128×128 texture will not be returned for a 256×256 request.

### Eviction

When a key already holds `max_per_key` entries, additional releases are dropped (destroyed) immediately.

### Stats and Cleanup

```rust
let stats = pool.stats();
println!("{} textures pooled, ~{} bytes", stats.entries, stats.estimated_bytes);

pool.clear(); // drop all pooled textures, free GPU memory
```

## When to Use Pooling

| Scenario | Recommendation |
|----------|---------------|
| Many small storage buffers with similar lifetime | `BufferPool` — one allocation, many views |
| Per-frame uniform/storage data that changes every frame | `BufferPoolRing` — ring-buffered pools, safe reset each frame |
| Transient render targets or compute textures | `TexturePool` — acquire/release cycle avoids allocation churn |
| Long-lived buffers (mesh data, static textures) | Individual `Buffer` / `Texture` — pooling adds no benefit |
| Uniform buffer updated once at startup | Individual `Buffer` — no per-frame reuse needed |

## Sub-Allocation Patterns

### Static Geometry Pool

Pack all static mesh data into one `BufferPool` at load time:

```rust
let size = BufferPool::padded_size(&[
    (vertex_count, std::mem::size_of::<Vertex>()),
    (index_count, std::mem::size_of::<u32>()),
]);
let mut pool = BufferPool::new(&device, size)?;

let vertices = pool.alloc_with_data(&vertex_data)?;
let indices = pool.alloc_with_data(&index_data)?;
```

### Per-Frame Dynamic Data

Use `BufferPoolRing` for data that changes every frame:

```rust
let mut ring = BufferPoolRing::<2>::new();

// In the render loop:
ring.advance();
ring.prepare(&device, frame_data_size)?;

let pool = ring.current_mut().unwrap();
let uniforms = pool.alloc_with_data(&[camera_data])?;
let instances = pool.alloc_with_data(&instance_transforms)?;
```

### Transient Compute Textures

Pool intermediate textures in a multi-pass compute pipeline:

```rust
let mut tex_pool = TexturePool::default();

// Each frame:
let temp = tex_pool.acquire(&device, w, h, fmt, SpatialAccess::Direct, flags)?;
// ... compute pass writes to temp ...
// ... next pass reads from temp ...
tex_pool.release(temp); // return for reuse next frame
```
