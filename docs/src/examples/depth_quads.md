# depth_quads

Two fullscreen quads whose depths cross periodically. Because depth testing decides
visibility, the picture is independent of the order the quads are drawn in — which is exactly
what the animation makes visible.

<video src="../assets/examples/depth_quads.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example depth_quads
```

## What it demonstrates

- `Scheme::lease_render_target` with a depth attachment
- Depth-stencil state on a render pipeline
- Draw-order independence

## Source

`examples/depth_quads.rs`:

```rust,noplayground
{{#include ../../../examples/depth_quads.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/depth_test.slang`:

```slang
{{#include ../../../shaders/depth_test.slang}}
```
