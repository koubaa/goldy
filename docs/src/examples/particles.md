# particles

A rain and snow particle system. The compute pass integrates positions and wraps particles
at the window edges; pressing `Space` swaps the simulation mode at runtime.

<video src="../assets/examples/particles.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example particles
```

## What it demonstrates

- Runtime switching between simulation modes
- Compute dispatch feeding an instanced raster pass

## Controls

| Key | Action |
|-----|--------|
| `Space` | Toggle between rain and snow |
| `Escape` | Exit |

## Source

`examples/particles.rs`:

```rust,noplayground
{{#include ../../../examples/particles.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/rain_snow_update.slang`:

```slang
{{#include ../../../shaders/rain_snow_update.slang}}
```

`shaders/rain_snow_render.slang`:

```slang
{{#include ../../../shaders/rain_snow_render.slang}}
```
