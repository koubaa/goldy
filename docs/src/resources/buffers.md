# Buffers

`Buffer` is a GPU memory allocation for storing typed data — uniforms, vertex data, index data, compute storage, or anything a shader needs to read or write.

## Creating Buffers

### With Typed Data

`Device::alloc_buffer_with_data` allocates through the installed [`VramAllocator`](vram-allocator.md) and uploads an initial slice. The element stride is inferred from `T`, which is critical for correct `StructuredBuffer` views on DX12.

```rust
use goldy::DataAccess;

let positions = vec![[0.0f32, 1.0, 0.0], [1.0, 0.0, 0.0]];
let buffer = device.alloc_buffer_with_data(&positions, DataAccess::Scattered)?;
```

**Type matters.** Passing `&[u8]` (e.g. from `bytemuck::bytes_of`) sets the element stride to 1 byte, while shaders usually expect a larger struct stride. Use a typed slice or `alloc_buffer_with_bytes_stride` instead.

### With Raw Bytes

When the data is naturally `&[u8]`:

```rust
// Stride defaults to 1 (byte-addressable)
let buffer = device.alloc_buffer_with_bytes(&raw_bytes, DataAccess::Scattered)?;

// Explicit stride for structured buffer views
let buffer = device.alloc_buffer_with_bytes_stride(&raw_bytes, DataAccess::Scattered, 16)?;

// With flags (e.g. CPU_READABLE)
let buffer = device.alloc_buffer_with_bytes_stride_and_flags(
    &raw_bytes,
    DataAccess::Scattered,
    16,
    BufferFlags::CPU_READABLE,
)?;
```

### Empty Buffer

```rust
use goldy::{BufferFlags, DataAccess};

let buffer = device.alloc_buffer(4096, DataAccess::Scattered, None, BufferFlags::empty())?;

// With a specific element stride
let buffer = device.alloc_buffer(4096, DataAccess::Scattered, Some(64), BufferFlags::empty())?;
```

## Data Access Patterns

The access pattern describes how shader threads access the buffer. This drives hardware optimizations and determines the bindless descriptor category.

```rust
pub enum DataAccess {
    Scattered, // default — any thread, any address, read/write
    Broadcast, // all threads read the same address
}
```

| Pattern | Shader Mapping | Use When |
|---------|---------------|----------|
| `Scattered` | `StructuredBuffer<T>`, `RWStructuredBuffer<T>` | General storage: particles, meshes, compute I/O |
| `Broadcast` | `ConstantBuffer` / uniform buffer | Uniform data: transforms, time, settings |

For read-only input buffers that don't need write access, create with `DataAccess::Scattered` and access through `goldy_buf_ro<T>` in the shader. This enables hardware read-cache optimizations without requiring a separate access pattern.

## BufferFlags

```rust
bitflags! {
    pub struct BufferFlags: u32 {
        const COPY_SRC      = 1 << 0;
        const COPY_DST      = 1 << 1;
        const CPU_READABLE  = 1 << 2;
    }
}
```

| Flag | Purpose |
|------|---------|
| `COPY_SRC` | Buffer can be a copy source |
| `COPY_DST` | Buffer can be a copy destination |
| `CPU_READABLE` | Optimize for readback. On Vulkan/Metal, `read_to_cpu` is a direct memcpy from host-visible memory. On DX12, it performs a GPU copy into a READBACK heap and waits. |

Query `DeviceCapabilities::has_zero_copy_storage_readback` to detect whether readback is zero-copy on the current backend.

## Writing Data

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

Read buffer contents back to the CPU. The buffer should have been created with `BufferFlags::CPU_READABLE` for optimal performance.

```rust
let mut output = vec![0u8; buffer.size() as usize];
buffer.read_to_cpu(&device, &mut output)?;
```

## Clearing

Zero-fill a region of the buffer:

```rust
buffer.clear(&device, offset, size)?;
```

## Bindless Descriptors

Every buffer with `Scattered` or `Broadcast` access is registered in the global bindless descriptor set. Retrieve the index to pass to shaders:

```rust
// Typed handle (preferred) — carries BindlessCategory for validation
let handle = buffer.bindless_handle().unwrap();

// Raw index
let index = buffer.bindless_index().unwrap();

// Read-only SRV index (separate from UAV on DX12; same on Vulkan/Metal)
let srv_handle = buffer.bindless_srv_handle().unwrap();
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

Views implement `BufferSource`, so they work anywhere a `Buffer` does — `set_vertex_buffer`, `set_index_buffer`, `write_data`, `read_to_cpu`, `clear`, and bindless binding:

```rust
let view_handle = view.bindless_handle().unwrap();
pass.set_vertex_buffer(0, &view);
```

### Lifetime

Dropping a `BufferView` unregisters its descriptor but does not free the parent buffer's memory. Multiple views of the same buffer can exist simultaneously.

## StructuredBufferElement

The `StructuredBufferElement` trait marks types safe for `Device::alloc_buffer_with_data` and `BufferPool::alloc_with_data`. It is implemented for common multi-byte primitives (`u16`, `u32`, `f32`, `f64`, etc.), fixed-size arrays of those types, and `#[repr(C)]` structs via `#[derive(goldy_derive::StructuredBufferElement)]`.

**Not implemented for `u8`/`i8`** — passing `&[u8]` would set stride to 1, which almost never matches the shader's expected struct stride. Use `device.alloc_buffer_with_bytes_stride` for raw bytes.

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
