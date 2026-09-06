# compute_to_surface

Rendering with no `RenderPipeline` at all. The compute shader writes the swapchain drawable
obtained from `SurfaceExchange::bind_destination`, and the frame is settled by claiming the
present transaction. This is the shortest path from a dispatch to the screen.

<video src="../assets/examples/compute_to_surface.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example compute_to_surface
```

## What it demonstrates

- `SurfaceExchange::bind_destination` — present-on-scheme
- `Transaction::claim` and `Claim::consume` settlement
- `DirectSpatial<T>` storage-texture writes to a drawable

## Notes

On the WebGPU backend the swapchain image cannot be bound as storage, so present falls back
to a copy or blit path automatically. See
[Backend Architecture](../backends/overview.md) for the `GOLDY_WEBGPU_PRESENT` override.

## Source

`examples/compute_to_surface.rs`:

```rust,noplayground
{{#include ../../../examples/compute_to_surface.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

The Slang source is inline in the example above.
