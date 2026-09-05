# triangle

The smallest complete Goldy program: three coloured vertices in a retained
[`Scheme`](../programming-model/parcels.md), rendered into a scheme-leased offscreen
render target, then handed to the swapchain through a `SurfaceExchange` transaction.
Every other windowed example is a variation on this skeleton.

```bash
cargo run --features examples --example triangle
```

## What it demonstrates

- `RetainedPool::acquire_buffer_with_data` for a static vertex buffer
- `Scheme::render_pass` recorded once and resubmitted every frame
- `SurfaceExchange::bind_render_target` plus `Transaction::claim` / `Claim::consume` to present
- Pipeline and scheme rebuild on window resize

## Source

`examples/triangle.rs`:

```rust,noplayground
{{#include ../../../examples/triangle.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/vertex_color_2d.slang`:

```slang
{{#include ../../../shaders/vertex_color_2d.slang}}
```
