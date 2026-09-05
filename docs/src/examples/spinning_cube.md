# spinning_cube

A wireframe cube drawn with `LINE_LIST` topology and a hand-rolled 3D projection, which
keeps the example free of any matrix library.

```bash
cargo run --features examples --example spinning_cube
```

## What it demonstrates

- Line primitives in a render pipeline
- Per-frame vertex uploads through a deposit transaction

## Source

`examples/spinning_cube.rs`:

```rust,noplayground
{{#include ../../../examples/spinning_cube.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/vertex_color_2d.slang`:

```slang
{{#include ../../../shaders/vertex_color_2d.slang}}
```
