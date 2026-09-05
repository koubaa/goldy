# compute_particles

A compute shader integrates particle positions in place, and a graphics pass draws them as
instanced quads from the same buffer. Both nodes live in one retained scheme, so Goldy
derives the compute-to-raster barrier from the declared parcel accesses.

<video src="../assets/examples/compute_particles.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example compute_particles
```

## What it demonstrates

- Compute and render nodes in a single retained scheme
- Read/write parcel access driving automatic hazard tracking
- Instanced draws sourced from compute output

## Source

`examples/compute_particles.rs`:

```rust,noplayground
{{#include ../../../examples/compute_particles.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/particle_update.slang`:

```slang
{{#include ../../../shaders/particle_update.slang}}
```

`shaders/particle_render.slang`:

```slang
{{#include ../../../shaders/particle_render.slang}}
```
