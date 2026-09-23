# tensor_algebra

Headless dense tensor algebra: fill a vector, then a tensor GEMV through the same
retained scheme. There is no window and no capture clip.

```bash
cargo run --example tensor_algebra
```

CUDA compute-only (no `graphics`):

```bash
cargo run --no-default-features --features cuda,tensor --example tensor_algebra
```

## What it demonstrates

- `TensorKernels` / `TensorRecorder` recording into an ordinary `Scheme`
- Checked views and semantic matmul (cuBLAS / MPS / stdlib)
- Host observation through `MemoryExchange`

## Source

`examples/tensor_algebra.rs`:

```rust,noplayground
{{#include ../../../examples/tensor_algebra.rs}}
```
