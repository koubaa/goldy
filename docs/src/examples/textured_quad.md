# textured_quad

A procedurally generated checkerboard texture sampled onto a quad. The vertex stage declares
no bindless resources at all; only the fragment stage takes `tex` and `smp`, and Goldy binds
them by pipeline name.

```bash
cargo run --features examples --example textured_quad
```

## What it demonstrates

- Stage-local resource declarations
- Named draw bindings (`tex`, `smp`)
- Automatic payload linking of `FullscreenVarying` between stages
- Texture upload through a deposit transaction

## Source

`examples/textured_quad.rs`:

```rust,noplayground
{{#include ../../../examples/textured_quad.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

The Slang source is inline in the example above.
