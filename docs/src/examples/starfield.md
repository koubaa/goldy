# starfield

A 3D starfield flying towards the viewer. A compute pass advances and recycles stars; the
raster pass scales each point by its depth.

```bash
cargo run --features examples --example starfield
```

## What it demonstrates

- Compute-driven particle recycling
- Depth-scaled point rendering

## Source

`examples/starfield.rs`:

```rust,noplayground
{{#include ../../../examples/starfield.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/starfield_update.slang`:

```slang
{{#include ../../../shaders/starfield_update.slang}}
```

`shaders/starfield_render.slang`:

```slang
{{#include ../../../shaders/starfield_render.slang}}
```
