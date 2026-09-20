# Matrix Multiply

`Scheme::matmul` records a **semantic** GEMM/GEMV. The task graph schedules it from
buffer bindings like any other node. The backend chooses an implementation on the
first submit and retains that plan.

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
| CUDA | cuBLAS (`cublasSgemv` when `n = 1`, otherwise `cublasSgemm`) | `GOLDY_MATMUL=fallback` |
| Metal | Metal Performance Shaders | `GOLDY_MATMUL=fallback` |
| Vulkan, DX12, WebGPU, CPU | Goldy stdlib kernel | — |

There is no public `prepare()`. The stdlib pipeline is compiled on first submit
when the backend has no native library (or when fallback is forced). Subsequent
clean submits reuse the realized command list / CUDA graph.

Custom leading dimensions are honored by native libraries. The stdlib kernel
requires packed row-major storage (`lda`/`ldb`/`ldc` derived from `m`/`n`/`k` and
the transpose flags).
