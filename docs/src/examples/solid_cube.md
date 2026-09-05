# solid_cube

A solid 3D cube with per-face colours, indexed geometry, and a depth attachment on the
scheme-leased render target.

<video src="../assets/examples/solid_cube.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example solid_cube
```

## What it demonstrates

- Indexed draws
- Depth attachment on a leased render target
- CPU-side model/view/projection transforms uploaded per frame

## Source

`examples/solid_cube.rs`:

```rust,noplayground
{{#include ../../../examples/solid_cube.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/vertex_color_2d.slang`:

```slang
{{#include ../../../shaders/vertex_color_2d.slang}}
```
