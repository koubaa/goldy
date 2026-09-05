# waveform

An animated waveform visualizer built from a `LINE_STRIP` whose vertices are recomputed on
the CPU and uploaded each frame.

<video src="../assets/examples/waveform.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example waveform
```

## What it demonstrates

- `LINE_STRIP` topology
- Per-frame vertex buffer deposits

## Source

`examples/waveform.rs`:

```rust,noplayground
{{#include ../../../examples/waveform.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/vertex_color_2d.slang`:

```slang
{{#include ../../../shaders/vertex_color_2d.slang}}
```
