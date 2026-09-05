# checkerboard

A procedural checkerboard whose UVs are distorted over time in the fragment shader — no
texture and no vertex buffer, just a time uniform and arithmetic.

```bash
cargo run --features examples --example checkerboard
```

## What it demonstrates

- Procedural fragment-shader texturing
- `ShaderModule::from_slang_with_options` for compile-time options
- Retained scheme with offscreen render pass and copy-to-present

## Notes

Set `GOLDY_VALIDATE_LAYOUTS=1` to validate the uniform layout against Slang reflection.

## Source

`examples/checkerboard.rs`:

```rust,noplayground
{{#include ../../../examples/checkerboard.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/checkerboard.slang`:

```slang
{{#include ../../../shaders/checkerboard.slang}}
```
