# tunnel

A flight through an endless tunnel, produced by converting screen coordinates to polar
coordinates and scrolling a procedural texture along them.

<video src="../assets/examples/tunnel.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example tunnel
```

## What it demonstrates

- Polar-coordinate screen-space effects
- Vertex-less fullscreen rendering

## Source

`examples/tunnel.rs`:

```rust,noplayground
{{#include ../../../examples/tunnel.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/tunnel.slang`:

```slang
{{#include ../../../shaders/tunnel.slang}}
```
