# Tensor Algebra

Goldy's `tensor` feature is a batteries-included **dense tensor algebra** over parcels.
It is not an ML framework: there is no autograd, no `nn.Module`, no optimizer, and no
Llama-specific operators. A future neural-network library can compete with `torch.nn`
by building on this layer.

```mermaid
flowchart TD
  App["Application or model"] --> Tensor["Goldy tensor layer"]
  Tensor --> Scheme["Goldy Scheme and parcels"]
  Scheme --> Backend["CUDA, Metal, Vulkan, DX12, WebGPU, CPU"]
  NN["Future NN library"] --> Tensor
```

Enable it with the `tensor` Cargo feature (on by default; independent of `graphics`):

```toml
goldy = { version = "0.3", features = ["tensor"] }
# CUDA compute-only, no graphics:
# goldy = { version = "0.3", default-features = false, features = ["cuda", "tensor"] }
```

## What the layer owns

- Concrete ranks 0..=4, dtypes `F32` / `U32` / `I32`
- Packed row-major storage plus positive/zero-stride views
- Checked `narrow`, `reshape`, `permute`, `transpose`, `broadcast_to`
- Shape inference, output allocation, and recording into the **same** [`Scheme`](../programming-model/parcels.md)
- Portable elementwise / reduction / gather / scatter kernels plus a tensor front-end for [semantic MatMul](./matmul.md)

## What it does not own

Autograd, parameters, optimizers, model formats, RMSNorm, RoPE, attention, activations
as NN layers, KV-cache policy, tokenization, symbolic shapes, or negative strides.

Low-level `Scheme`, `Buffer`, `Parcel`, and `MatMulView` APIs remain escape hatches.

## Views are lenses

A [`Tensor`] is a buffer parcel plus layout metadata. A [`TensorView`] never mints a new
ownership identity. Binding a view:

1. Claims a conservative byte envelope (`BufferRange`, or the parent buffer) so overlapping
   aliases stay visible to Goldy's hazard analysis.
2. Uses the **parent** bindless slot so shaders see the whole buffer plus element offsets.

Read-only broadcast views (zero stride on an expanded axis) share that envelope; they are
not independent identities. Write layouts require a positive stride on every axis with
`shape > 1`.

Host updates and observations still go through [`MemoryExchange`](./settlement.md).

## Recording

[`TensorContext`] prepares portable kernels and retains tiny layout parcels. Keep it alive
for as long as recorded schemes exist. [`TensorRecorder`] borrows the context and a mutable
`Scheme`:

```rust,no_run
# use goldy::{Instance, RequestAdapterOptions, RuntimeDescriptor, Scheme, Tensor, TensorContext, TensorShape};
# fn main() -> Result<(), goldy::GoldyError> {
# let runtime = Instance::new().unwrap().request_adapter(&Default::default()).unwrap().request_runtime(&Default::default()).unwrap();
# let ctx = runtime.create_context().unwrap();
let mut tensors = TensorContext::new(&runtime)?;
let a = Tensor::from_f32(&runtime, TensorShape::vector(4), &[1.0, 2.0, 3.0, 4.0])?;
let b = Tensor::from_f32(&runtime, TensorShape::vector(1), &[10.0])?;
let mut scheme = Scheme::new(&ctx);
let c = tensors.recorder(&mut scheme).add("add", a.view(), b.view())?;
# let _ = c;
# Ok(())
# }
```

Allocating methods (`add`, `matmul`, `sum`, …) create packed outputs. `_into` forms
(`add_into`, `matmul_into`, `cast_into`, `fill`) use caller storage, including in-place
work when the write layout is legal.

Custom `#[goldy::compute]` kernels bind `Tensor` / `TensorView` like any other parcel and
can dispatch by logical extent:

```rust,ignore
kernel.record(&mut scheme, "rope", &q_view, &k_view, &step, ...).over_tensor(&q_view);
```

## Operations

| Family | Ops | Notes |
|--------|-----|-------|
| Construction | `zeros`, `from_f32` / `from_u32` / `from_i32`, `fill`, `copy`, `contiguous`, `cast` | Per-op dtype support |
| Unary | `neg`, `abs`, `exp`, `log`, `sqrt`, `reciprocal` | F32 |
| Binary | `add` / `sub` / `mul` / `div` / `min` / `max` and `*_scalar` | F32, NumPy-style broadcasting |
| Reductions | `sum`, `max_reduce`, `min_reduce`, `mean` | One axis; squeezed by default |
| Gather / scatter | `gather`; `scatter` with [`ScatterMode`] | Modes are explicit: `UniqueWrite`, `Add`, `Min`, `Max` |
| Matmul | rank-1/2 and batched rank-3 | Rank-2 packed cases lower to `Scheme::matmul` (cuBLAS / MPS / stdlib) |
| Softmax | composition of max / sub / exp / sum / div | Convenience only; not fused attention |

Scatter `Add` / `Min` / `Max` are defined (a single thread walks colliding indices). They are
**not** idempotent on scheme replay if the destination is the accumulator; unique-write and
pure functions of the inputs are.

## Bindings

The `tensor` Cargo feature is passed through to `goldy-ffi`, `goldy-ffi-client`, and
`goldy-py`. C (`goldy.h`) and C++ (`goldy.hpp`) expose acquire / add / matmul / fill.
Python copies NumPy arrays into tensors on acquire and reads results back through
[`MemoryExchange`](../programming-model/parcels.md) on `tensor.parcel()` — there is no
arbitrary zero-copy host view of GPU storage.

## llama3.goldy

[`llama3.goldy`](https://github.com/koubaa/llama3.goldy) is the proving consumer: activations
and checkpoint weights are tensors, static GEMVs use tensor matmul, and custom RMSNorm / RoPE /
attention / SwiGLU kernels bind tensor views. Dynamic decode state stays a `DecodeStep` deposit.
