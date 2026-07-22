# RetainedPool, Buffer, and Parcel

[`RetainedPool`](../../src/retained_pool.rs) is the public door for **retained** GPU memory. Acquire returns a [`Buffer`](../../src/parcel.rs) (possibly partitioned) or a texture [`Parcel`](../../src/parcel.rs). **Bind parcels**, not raw aggregates — each parcel is one bindable unit (whole buffer, buffer range, or texture).

## Quick start

```rust
use goldy::{BufferKind, BufferFlags, RetainedPool, field, Init, NodeAccess, Scheme};

let mut pool = RetainedPool::new(device.clone());

// Single-unit buffer (derefs to whole parcel):
let vertices = [/* ... */];
let vb = pool.acquire_buffer_with_data(&vertices, BufferKind::Scattered)?;

// Raw bytes with explicit stride:
let uniform_buf = pool.acquire_buffer(
    raw_bytes.len() as u64,
    BufferKind::Scattered,
    Some(16),
    BufferFlags::empty(),
    Some(&raw_bytes),
)?;

// Uninitialized buffer (rewrite each frame with write_parcel):
let uniform = pool.acquire_buffer_sized::<MyUniforms>(1, BufferKind::Broadcast, BufferFlags::empty())?;

// Texture parcel:
let tex = pool.acquire_texture(w, h, format, access, flags, Some(&pixels))?;

// Partitioned record (ping-pong, level geometry):
let cells = pool.acquire_record([
    field("a", Init::data(&grid_a)),
    field("b", Init::zeros::<u32>(n)),
])?;
```

## Scheme binding

```rust
let mut upload = Scheme::new(&ctx);
upload.write_parcel(&*uniform, 0, bytemuck::bytes_of(&data).to_vec())?;
upload.submit()?;

let mut pass = scheme.render_pass("draw", &rt);
pass.with_parcel(&*vb, NodeAccess::Read);
pass.set_vertex_buffer(0, &*vb);
pass.draw(0..3, 0..1);

// Partitioned buffer: bind one field/range
pass.with_parcel(&cells["a"], NodeAccess::Read);

// Geometry bound via BufferSource only — register dependency without descriptor:
pass.with_buffer_dependency(&geometry, NodeAccess::Read);
```

Binding a multi-unit `Buffer` as one descriptor panics; index into fields instead.

## Release

Call `pool.release(&ctx, hold)` when resizing or tearing down. While held, buffers need no epoch polling — the runtime stamps each parcel at submit.

## Bindings

| Language | Types | Acquire |
|----------|-------|---------|
| Rust | `RetainedPool`, `Buffer`, `Parcel` | `acquire_buffer*`, `acquire_record`, `acquire_texture` |
| Python | `goldy.RetainedPool`, `goldy.Buffer`, `goldy.Parcel` | `acquire_buffer`, `acquire_record`, `acquire_texture` |
| C# | `RetainedPool`, `Buffer`, `Parcel`, `RecordBuilder` | `AcquireBuffer`, `Record()`, `AcquireTexture` |
| C / ffi-client | `GoldyRetainedPool`, `GoldyBuffer`, `GoldyParcel` | `goldy_retained_pool_acquire_buffer`, `goldy_record_builder_*` |

All examples under `goldy/examples/`, `python/examples/`, `dotnet/Goldy.Examples/`, and `ffi-client/examples/` use this API.
