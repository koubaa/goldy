# Buffers

`Buffer` is a GPU memory allocation for storing typed data — uniforms, vertex data, index data, compute storage, or anything a shader needs to read or write.

## Creating buffers (recommended)

For application-owned GPU memory, use [`RetainedPool`](retained-pool.md) and bind the returned [`Parcel`](retained-pool.md) in a scheme (`with_parcel`, `set_vertex_buffer`, [`MemoryExchange`](../compute/timeline.md) deposits). All Rust, Python, FFI, and .NET examples use this path.

```rust
use goldy::{BufferFlags, BufferKind, RetainedPool};

let mut pool = RetainedPool::new(device.clone());
let vertices = [/* Vertex2D ... */];
let vertex_parcel = pool.acquire_buffer_with_data(&vertices, BufferKind::Scattered)?;

// Uninitialized storage (e.g. a uniform updated each frame via MemoryExchange deposit):
let uniform = pool.acquire_buffer_sized::<MyUniforms>(1, BufferKind::Broadcast, BufferFlags::empty())?;
```

See [`retained-pool.md`](retained-pool.md) for textures, mosaics, and release.

### With Raw Bytes

When the data is naturally `&[u8]`, pass an explicit element stride to `acquire_buffer`:

```rust
use goldy::{BufferFlags, BufferKind, RetainedPool};

let mut pool = RetainedPool::new(device.clone());

// Stride defaults to 1 when omitted (byte-addressable)
let parcel = pool.acquire_buffer(
    raw_bytes.len() as u64,
    BufferKind::Scattered,
    None,
    BufferFlags::empty(),
    Some(&raw_bytes),
)?;

// Explicit stride for structured buffer views
let parcel = pool.acquire_buffer(
    raw_bytes.len() as u64,
    BufferKind::Scattered,
    Some(16),
    BufferFlags::empty(),
    Some(&raw_bytes),
)?;

// With flags (e.g. CPU_READABLE)
let parcel = pool.acquire_buffer(
    raw_bytes.len() as u64,
    BufferKind::Scattered,
    Some(16),
    BufferFlags::CPU_READABLE,
    Some(&raw_bytes),
)?;
```

### Empty Buffer

```rust
let parcel = pool.acquire_buffer(
    4096,
    BufferKind::Scattered,
    None,
    BufferFlags::empty(),
    None,
)?;

// With a specific element stride
let parcel = pool.acquire_buffer(
    4096,
    BufferKind::Scattered,
    Some(64),
    BufferFlags::empty(),
    None,
)?;
```

## Low-level `Device::alloc_*` (crate-internal)

The runtime routes standalone allocations through [`VramAllocator`](vram-allocator.md) via
crate-internal `Device::alloc_buffer` helpers. Application code should not call these;
use `RetainedPool` above.

## Data Access Patterns

The access pattern describes how shader threads access the buffer. This drives hardware optimizations and determines the bindless descriptor category.

```rust
pub enum BufferKind {
    Scattered, // default — any thread, any address, read/write
    Broadcast, // all threads read the same address
}
```

| Pattern | Shader Mapping | Use When |
|---------|---------------|----------|
| `Scattered` | `StructuredBuffer<T>`, `RWStructuredBuffer<T>` | General storage: particles, meshes, compute I/O |
| `Broadcast` | `ConstantBuffer` / uniform buffer | Uniform data: transforms, time, settings |

For read-only input buffers that don't need write access, create with `BufferKind::Scattered` and access through `goldy_buf_ro<T>` in the shader. This enables hardware read-cache optimizations without requiring a separate access pattern.

## BufferFlags

```rust
bitflags! {
    pub struct BufferFlags: u32 {
        const COPY_SRC      = 1 << 0;
        const COPY_DST      = 1 << 1;
        const CPU_READABLE  = 1 << 2;
        const CPU_WRITABLE  = 1 << 4;
    }
}
```

| Flag | Purpose |
|------|---------|
| `COPY_SRC` | Buffer can be a copy source |
| `COPY_DST` | Buffer can be a copy destination |
| `CPU_READABLE` | Medium hint for host-visible storage. Prefer [`MemoryExchange::bind_withdraw`](../compute/timeline.md) for observation. Not a public host-read API. |
| `CPU_WRITABLE` | Host-mapped staging for deposits / upload copies. Prefer [`MemoryExchange::bind_deposit_buffer`](../compute/timeline.md) for application uploads. |

Query `DeviceCapabilities::has_zero_copy_storage_readback` to detect whether withdraw staging can elide a GPU copy on the current backend.

## Writing Data

Prefer [`MemoryExchange::bind_deposit_buffer`](../compute/timeline.md) for CPU→GPU uploads. Direct host writes on `CPU_WRITABLE` staging parcels remain for deposit/staging internals:

### Raw bytes

```rust
buffer.write(offset, &bytes)?;
```

### Typed data

```rust
buffer.write_data(offset, &[1.0f32, 2.0, 3.0])?;
```

Both methods write at a byte offset from the start of the buffer.

## Reading Data

Use a memory exchange withdraw bound into a scheme:

```rust
let memory = MemoryExchange::new(&ctx);
let withdraw = memory.bind_withdraw(&mut scheme, buffer.whole())?;
let mut submission = scheme.submit()?;
let bytes = withdraw.claim(&mut submission)?.consume()?;
```

## Clearing

Zero-fill a region of the buffer:

```rust
buffer.clear(&device, offset, size)?;
```

## Bindless Descriptors

Every buffer with `Scattered` or `Broadcast` access is registered in the global bindless descriptor set. Retrieve the index to pass to shaders:

```rust
// Typed handle (preferred) — carries ResourceCategory for validation
let handle = buffer.handle(ResourceAccess::Read).unwrap();

// Raw index
let index = buffer.resource_index(ResourceAccess::Read).unwrap();

// Read-only SRV index (separate from UAV on DX12; same on Vulkan/Metal)
let srv_handle = buffer.handle(ResourceAccess::Read).unwrap();
```

## BufferView

A `BufferView` is a sub-region of an existing `Buffer` with its own bindless descriptor. The shader sees the sub-region as a zero-based buffer.

### Creating Views

```rust
// Raw byte view — offset, size, optional element stride
let view = buffer.create_view(1024, 512, Some(16))?;

// Typed view — first element index, element count
let view = buffer.create_typed_view::<[f32; 4]>(0, 256)?;
```

### Using Views

Views implement `BufferSource`, so they work anywhere a `Buffer` does — `set_vertex_buffer`, `set_index_buffer`, `write_data`, `clear`, and bindless binding:

```rust
let view_handle = view.handle(ResourceAccess::Read).unwrap();
pass.set_vertex_buffer(0, &view);
```

### Lifetime

Dropping a `BufferView` unregisters its descriptor but does not free the parent buffer's memory. Multiple views of the same buffer can exist simultaneously.

## StructuredBufferElement

The `StructuredBufferElement` trait marks types safe for `RetainedPool::acquire_buffer_with_data`.
It is implemented for common multi-byte primitives (`u16`, `u32`, `f32`, `f64`, etc.), fixed-size arrays of those types, and `#[repr(C)]` structs via `#[derive(goldy_derive::StructuredBufferElement)]`.

**Not implemented for `u8`/`i8`** — passing `&[u8]` would set stride to 1, which almost never matches the shader's expected struct stride. Use `RetainedPool::acquire_buffer` with an explicit element stride for raw bytes.

## Matrix Convention

Goldy uses **column-major** matrix layout in uniform/constant buffers across all backends. Rust math libraries (glam, nalgebra, ultraviolet) already store matrices column-major, so upload directly without transposing:

```rust
let uniforms = MyUniforms {
    projection: proj.to_cols_array_2d(),
    modelview: view.to_cols_array_2d(),
};
buffer.write_data(0, &[uniforms])?;
```

Goldy sets `SLANG_MATRIX_LAYOUT_COLUMN_MAJOR` at the Slang session level, so DX12, Vulkan, and Metal all interpret `float4x4` the same way.
