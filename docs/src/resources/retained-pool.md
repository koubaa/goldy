# RetainedPool and Parcel

[`RetainedPool`](../../src/retained_pool.rs) is the public door for **retained** GPU memory — buffers and textures the application holds across frames. It returns opaque [`Parcel`](../../src/parcel.rs) values; bind them in the task graph instead of passing raw `Buffer` handles.

## Quick start

```rust
use goldy::{BufferKind, RetainedPool, TaskGraph};

let mut pool = RetainedPool::new(device.clone());

// Typed upload (stride inferred from T):
let vertices = [/* ... */];
let vb = pool.acquire_buffer_with_data(&vertices, BufferKind::Scattered)?;

// Raw bytes with explicit stride:
let parcel = pool.acquire_buffer(
    raw_bytes.len() as u64,
    BufferKind::Scattered,
    Some(16),
    BufferFlags::empty(),
    Some(&raw_bytes),
)?;

// Uninitialized buffer (rewrite each frame with write_parcel):
let uniform = pool.acquire_buffer_sized::<MyUniforms>(1, BufferKind::Broadcast, BufferFlags::empty())?;

// Texture with initial pixels:
let tex = pool.acquire_texture(w, h, format, access, flags, Some(&pixels))?;

// Mosaic: multiple sub-views in one backing allocation (ping-pong, level geometry):
let mut mosaic = pool.mosaic();
mosaic.emplace::<u32>(&grid_a);
mosaic.emplace::<u32>(&grid_b);
let cells = mosaic.build()?;
```

## Task graph binding

```rust
frame_graph.write_parcel(&uniform, 0, bytemuck::bytes_of(&data).to_vec())?;

let mut pass = frame_graph.render_pass("draw", &rt);
pass.bind_parcel_mut(&vb, NodeAccess::Read);
pass.set_vertex_buffer(0, &vb);
pass.draw(0..3, 0..1);
```

For mosaic parcels, use `parcel.view(slot)` to bind a sub-range.

## Release

Call `pool.release(&ctx, parcel)` when resizing or tearing down. While held, parcels need no epoch polling — the runtime stamps them at submit.

## Bindings

| Language | Types | Acquire |
|----------|-------|---------|
| Rust | `RetainedPool`, `Parcel` | `acquire_buffer`, `acquire_buffer_with_data`, `acquire_buffer_sized`, `mosaic` |
| Python | `goldy.RetainedPool`, `goldy.Parcel` | `acquire_buffer(numpy_array, kind)` |
| C# | `RetainedPool`, `Parcel` | `AcquireBuffer<T>(data, kind)` |
| C / ffi-client | `GoldyRetainedPool`, `GoldyParcel` | `goldy_retained_pool_acquire_buffer`, mosaic builder |

All examples under `goldy/examples/`, `python/examples/`, `dotnet/Goldy.Examples/`, and `ffi-client/examples/` use this API.
