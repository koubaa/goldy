# Runtime-Owned Memory

[`Runtime`](../../src/runtime.rs) backs **retained** GPU memory — the same way a CPU program allocates from the process heap without asking for a separate heap handle. Acquire returns a [`Buffer`](../../src/parcel.rs) (possibly partitioned) or a texture [`Parcel`](../../src/parcel.rs). **Bind parcels**, not raw aggregates — each parcel is one bindable unit (whole buffer, buffer range, or texture).

[`Context::release_buffer`](../../src/retained_pool.rs) / [`Context::release_texture`](../../src/retained_pool.rs) park a held parcel in that context's transient pool for epoch-gated reuse. There is no public `RetainedPool` type: acquisition is inherent on `Runtime`.

## Quick start

```rust
use goldy::{BufferKind, BufferFlags, MemoryExchange, field, Init, NodeAccess, Scheme};

// Single-unit buffer (derefs to whole parcel):
let vertices = [/* ... */];
let vb = runtime.acquire_buffer_with_data(&vertices, BufferKind::Scattered)?;

// Raw bytes with explicit stride:
let uniform_buf = runtime.acquire_buffer(
    raw_bytes.len() as u64,
    BufferKind::Scattered,
    Some(16),
    BufferFlags::empty(),
    Some(&raw_bytes),
)?;

// Uninitialized buffer (rewrite each frame with a MemoryExchange deposit):
let uniform = runtime.acquire_buffer_sized::<MyUniforms>(1, BufferKind::Broadcast, BufferFlags::empty())?;

// Texture parcel:
let tex = runtime.acquire_texture(w, h, format, access, flags, Some(&pixels))?;

// Partitioned record (ping-pong, level geometry):
let cells = runtime.acquire_record([
    field("a", Init::data(&grid_a)),
    field("b", Init::zeros::<u32>(n)),
])?;
```

## Scheme binding

```rust
let memory = MemoryExchange::new(&ctx);
let mut upload = Scheme::new(&ctx);
let deposit = memory.bind_deposit(&mut upload, goldy::DepositTarget::buffer_elements::<MyUniforms>(&*uniform, 1))?;
(&deposit << &data)?;
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

Call `ctx.release_buffer(buffer)` / `ctx.release_texture(texture)` when resizing or tearing down. While held, buffers need no epoch polling — the runtime stamps each parcel at submit.

## Bindings

| Language | Types | Acquire |
|----------|-------|---------|
| Rust | `Runtime`, `Buffer`, `Parcel` | `acquire_buffer*`, `acquire_record`, `acquire_texture` |
| Python | `goldy.Runtime`, `goldy.Buffer`, `goldy.Parcel` | `Runtime.acquire_buffer`, `Runtime.acquire_texture` |
| C# | `Runtime`, `Buffer`, `Parcel`, `RecordBuilder` | `Runtime.AcquireBuffer`, `Runtime.AcquireTexture` |
| C / ffi-client | `GoldyRuntime`, `GoldyBuffer`, `GoldyParcel` | `goldy_runtime_acquire_*` |

All examples under `examples/` acquire from `Runtime` directly. See the [bindings](../bindings/python.md) section for language-specific guides.
