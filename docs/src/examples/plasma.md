# plasma

The classic demoscene plasma effect: layered sine fields evaluated per pixel in the fragment
shader, with time as the only input.

```bash
cargo run --features examples --example plasma
```

## What it demonstrates

- Vertex-less fullscreen fragment effect
- Single time uniform driving the whole image

## Source

`examples/plasma.rs`:

```rust,noplayground
{{#include ../../../examples/plasma.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/plasma.slang`:

```slang
{{#include ../../../shaders/plasma.slang}}
```
