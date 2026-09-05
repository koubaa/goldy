# game_of_life

Conway's Game of Life on the GPU. Both cell grids live in a single retained record buffer as
fields `"a"` and `"b"`, so ping-pong is a sub-view swap rather than two separate parcels.
Each simulation step runs an ephemeral compute scheme; the display scheme is re-recorded
when the active field flips.

<video src="../assets/examples/game_of_life.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example game_of_life
```

## What it demonstrates

- Sub-views of one retained mosaic parcel for ping-pong state
- Mixing ephemeral compute schemes with a retained display scheme
- Compute → render → present in one frame

## Source

`examples/game_of_life.rs`:

```rust,noplayground
{{#include ../../../examples/game_of_life.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/game_of_life.slang`:

```slang
{{#include ../../../shaders/game_of_life.slang}}
```

`shaders/game_of_life_render.slang`:

```slang
{{#include ../../../shaders/game_of_life_render.slang}}
```
