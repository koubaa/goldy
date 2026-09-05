# instancing

GPU-driven instancing: a compute pass updates per-instance transforms in a storage buffer,
and one instanced draw renders them all. The per-instance layout lives in
`examples/instance2d.rs` and mirrors `QuadInstance` in the Slang shaders.

<video src="../assets/examples/instancing.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example instancing
```

## What it demonstrates

- Compute-updated instance data consumed by a raster pass
- Matching host and shader struct layouts
- One draw call for many objects

## Source

`examples/instancing.rs`:

```rust,noplayground
{{#include ../../../examples/instancing.rs}}
```

The example pulls in `examples/common.rs`, `examples/instance2d.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/instancing_update.slang`:

```slang
{{#include ../../../shaders/instancing_update.slang}}
```

`shaders/instancing_render.slang`:

```slang
{{#include ../../../shaders/instancing_render.slang}}
```
