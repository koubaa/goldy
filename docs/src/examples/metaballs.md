# metaballs

Organic blobs rendered by summing an inverse-square field from several moving centres and
thresholding it in the fragment shader.

```bash
cargo run --features examples --example metaballs
```

## What it demonstrates

- Scalar-field rendering in a fragment shader
- Animated uniform arrays

## Source

`examples/metaballs.rs`:

```rust,noplayground
{{#include ../../../examples/metaballs.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/metaballs.slang`:

```slang
{{#include ../../../shaders/metaballs.slang}}
```
