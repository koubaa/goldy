# bouncing_lines

Line segments bouncing off the window edges. A compute pass integrates the simple physics;
a `LINE_LIST` pipeline draws the result.

<video src="../assets/examples/bouncing_lines.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example bouncing_lines
```

## What it demonstrates

- `LINE_LIST` topology
- Compute dispatch feeding a raster pass in one retained scheme

## Source

`examples/bouncing_lines.rs`:

```rust,noplayground
{{#include ../../../examples/bouncing_lines.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/bouncing_lines_update.slang`:

```slang
{{#include ../../../shaders/bouncing_lines_update.slang}}
```

`shaders/bouncing_lines_render.slang`:

```slang
{{#include ../../../shaders/bouncing_lines_render.slang}}
```
