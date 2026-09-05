# gradient

An animated full-screen gradient driven by a single time uniform. Rendering is vertex-less:
the vertex stage synthesizes a fullscreen triangle from `SV_VertexID`, which is the
Goldy-native way to write screen-space effects.

<video src="../assets/examples/gradient.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example gradient
```

## What it demonstrates

- Vertex-less fullscreen rendering
- Per-frame uniform updates through a deposit transaction
- `LayoutCheckable` host/shader struct layout validation

## Notes

Set `GOLDY_VALIDATE_LAYOUTS=1` to have the example cross-check its uniform struct layout
against Slang reflection:

```bash
GOLDY_VALIDATE_LAYOUTS=1 cargo run --features examples --example gradient
```

## Source

`examples/gradient.rs`:

```rust,noplayground
{{#include ../../../examples/gradient.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/gradient.slang`:

```slang
{{#include ../../../shaders/gradient.slang}}
```
