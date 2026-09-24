# Matrix Multiply

`Scheme::matmul` records a **semantic** GEMM/GEMV. The task graph schedules it from
buffer bindings like any other node. The backend chooses an implementation on the
first submit and retains that plan.

The tensor front end (`goldy/tensor`, on by default) derives `m`/`n`/`k`, transpose
flags, offsets, and leading dimensions from checked [`TensorView`](./tensor.md)s and
records the same node. `MatMulView` remains the low-level escape hatch.

```rust
scheme
    .matmul("q_projection", MatMulDesc::gemv(dim, dim))
    .a(&weights, MatMulView::offset(wq))
    .b(&xb, MatMulView::packed())
    .out(&q, MatMulView::packed())
    .record();
```

The contract is row-major `C[m, n] = alpha * op(A)[m, k] @ op(B)[k, n] + beta * C`.
The first slice is FP32 with `alpha = 1`, `beta = 0` (the stdlib fallback requires
those epilogue values; native libraries honor other alpha/beta).

## Implementation choice

| Backend | Default | Override |
|---------|---------|----------|
| CUDA | Goldy `gemv_f32` for GEMV, cuBLAS `cublasSgemm` otherwise | `GOLDY_MATMUL=library` / `fallback` |
| Metal | Metal Performance Shaders | `GOLDY_MATMUL=fallback` |
| Vulkan, DX12, WebGPU, CPU | Goldy stdlib kernels | — |

A GEMV here is `n = 1`, no transposes, `alpha = 1`, `beta = 0`. On CUDA, cuBLAS
`sgemv` picks a split-K kernel plus a separate reduction for decode-sized matrices
(a few hundred rows and columns). Goldy's `gemv_f32` reduces each row with one
32-lane group in a single pass, which is faster there and matches cuBLAS on large
bandwidth-bound shapes. `GOLDY_MATMUL=library` restores cuBLAS for every shape.

There is no public `prepare()`. Each stdlib pipeline is compiled on first submit
by the first node that needs it. Subsequent clean submits reuse the realized
command list / CUDA graph.

Custom leading dimensions are honored by native libraries and by `gemv_f32`. The
general stdlib GEMM kernel requires packed row-major storage (`lda`/`ldb`/`ldc`
derived from `m`/`n`/`k` and the transpose flags).
