# Rust Compute Kernels

Goldy can lower a **restricted Rust GPU dialect** into canonical `[goldy_compute]`
Slang at compile time, then prepare and record through the normal Scheme path.

This is the initial design for issue #78. It is **not** arbitrary Rust, a second
runtime compiler, or CUDA `<<<>>>` syntax. Slang remains the runtime backend
compiler; the proc-macro is an AOT frontend that produces structured
`KernelDef` metadata, a retained structured definition, and typed `record` helpers.

To **step the same kernel on the CPU** without a handwritten Rust twin, see
[CPU host-callable shaders](../debugging/cpu-host-callable.md) (issue #292).

## Quick example

```rust
use goldy::gpu;

#[goldy::compute(workgroup_size = [256, 1, 1])]
fn saxpy(x: &[f32], y: &mut [f32], a: f32) {
    let i = gpu::global_id().x;
    if i < y.len() {
        y[i] = a * x[i] + y[i];
    }
}

// Host:
let kernel = saxpy::Kernel::prepare(&device)?;
kernel
    .record(&mut scheme, "saxpy", &x, &y, a)
    .over_1d(n);
// or exact grid counts:
kernel
    .record(&mut scheme, "saxpy", &x, &y, a)
    .groups([n.div_ceil(256), 1, 1]);
```

`prepare` compiles (or hits the shader cache) once. `record` only appends Scheme
topology — it does not launch into a stream. Use `use goldy::gpu;` (or
`goldy::gpu::global_id()`) for builtins.

## Signature mapping

Rust GPU-dialect types use the same names as `shaders/goldy_exp/access.slang`
(`BufRO`, `Scattered`, `DirectSpatial`, `ThreadId`, …).

| Rust parameter | Slang / Scheme |
|---|---|
| `&[T]` / `gpu::BufRO<T>` | `BufRO<T>`, `NodeAccess::Read` |
| `&mut [T]` | `Scattered<T>`, `NodeAccess::ReadWrite` |
| `gpu::Scattered<T>` | `Scattered<T>`, `NodeAccess::Write` |
| `gpu::Tensor<T>` | parent `BufRO<T>` + packed layout, `NodeAccess::Read` |
| `gpu::TensorMut<T>` | parent `Scattered<T>` + packed layout, `NodeAccess::ReadWrite` |
| `gpu::TensorWrite<T>` | parent `Scattered<T>` + packed layout, `NodeAccess::Write` |
| `gpu::Uniform<T>` | broadcast resource, `NodeAccess::Read` |
| `gpu::DirectSpatial<gpu::Float4>` | `DirectSpatial<float4>`, `NodeAccess::Write` (swapchain lease or texture) |
| `u32` / `i32` / `f32` / `bool` | typed scalar push words (no manual `to_bits`) |

Hidden builtins (appended to the Slang signature when used):

| Rust | Slang |
|---|---|
| `gpu::global_id()` | `ThreadId` |
| `gpu::local_id()` | `GroupThreadId` |
| `gpu::workgroup_id()` | `GroupId` |
| `gpu::workgroup_barrier()` | `GroupMemoryBarrierWithGroupSync` |
| `let mut s = gpu::workgroup_array::<T, N>()` | file-scope `groupshared T s[N]` |
| `gpu::workgroup_sum::<N>(val, scratch)` | tree-reduce sum; every lane gets the total |
| `gpu::workgroup_max::<N>(val, scratch)` | tree-reduce max; every lane gets the max |
| `gpu::workgroup_softmax_in_place::<N>(buf, base, count, scratch)` | in-place softmax over `buf[base .. base+count)` |

`workgroup_size` is fixed on the attribute / `KernelDef`. `.groups` / `.over_*`
only control the grid. A different workgroup size is a different pipeline.

Workgroup arrays are a **fixed** size known at compile time (not dynamic shared
memory). Declare them at the kernel top level, then index them like a buffer.

`workgroup_sum` / `workgroup_max` / `workgroup_softmax_in_place` are 1D
collectives. `N` must be a power of two (typically `workgroup_size.x`). They
return the reduced value to **every** lane and include a trailing barrier, so
the result is immediately usable. Softmax writes `buf[base + t]` for
`t < count`; unused lanes contribute identity (`-1e30` / `0`). All threads in
the workgroup must execute the call (no divergent branches around it).
`workgroup_sum`/`workgroup_max` must be a `let` or simple assignment, not nested
in a larger expression. Omit `::<N>` to use `workgroup_size.x`. When `buf` is a
tensor parameter, softmax indexes through the view (logical `base + t`).

## Logical tensors vs physical buffers

`gpu::Tensor<T>` / `TensorMut<T>` / `TensorWrite<T>` bind a [`TensorView`](../compute/tensor.md):
the shader receives the **parent** parcel plus a scheme-owned packed metadata
parcel (`GoldyTensorLayout` per tensor, one buffer for the dispatch). Indexing
is logical-view-relative:

- `view[i]` delinearizes `i` through rank/shape, then applies offset and strides
- `view.len()` is the logical `numel`
- `view.dim(axis)` and `view.rank()` read checked layout facts

Ordinary `&[T]` / `&mut [T]` / `gpu::Scattered<T>` stay the physical-index escape
hatch: `buf[i]` is an element index in the parent buffer, and `.len()` is the
buffer length. Tensor `record` methods take `TensorView` arguments, validate
dtype, writeability, and optional shape contracts, and return `Result` because
packing the layout (or a contract miss) can fail.

### Shape contracts

Annotate tensor parameters with `#[tensor(shape = [...])]` to fix rank and to
require equal extents at record time. Dimensions may be `_` (any extent), an
integer literal (exact), or an identifier (symbolic equality within one `record`
call):

```rust
#[goldy::compute(workgroup_size = [256, 1, 1])]
fn rmsnorm(
    #[tensor(shape = [dim])] x: gpu::Tensor<f32>,
    #[tensor(shape = [dim])] weight: gpu::Tensor<f32>,
    #[tensor(shape = [dim])] out: gpu::TensorWrite<f32>,
) { /* ... */ }
```

Unannotated tensor parameters keep today's any-shape behavior. The generated
`record` method checks every tensor argument against the contract **before**
binding parcels or appending GraphIR. Failures name the kernel, parameter, axis,
expected spec, and actual shape. Shader parameter order, the 48-byte
`GoldyTensorLayout` ABI, and `KERNEL_ABI_VERSION` are unchanged.

Relationships that are not dimension equality — for example query-head /
KV-head divisibility — stay explicit kernel or domain checks, not part of this
DSL.

Goldy only has eight user scalar words, so layouts are **not** push constants.
The metadata parcel is interned on the scheme, read-only in GraphIR, and does
not need an external `TensorKernels` keepalive.

## Architecture

```text
Rust kernel
    │
    ▼
goldy_derive::compute
    ├── syn AST validation (GPU dialect)
    ├── goldy_shader_ir ShaderKernel (retained as definition())
    ├── canonical [goldy_compute] Slang
    └── KernelDef / KernelParam ABI
    │
    ▼
Kernel::prepare(device)
    └── existing ShaderModule + ComputePipeline + cache
    │
    ▼
typed record() → SchemeNodeBuilder bindings in declaration order
```

### Retained definitions

Each generated module exposes `definition()`, the structured `ShaderKernel` the
canonical source was lowered from, and a prepared kernel keeps it in
`KernelDef::definition`. Hand-authored Slang has no definition and is opaque to
composition.

`goldy::kernel::ir` lowers a definition in two steps: `lower_body` turns its
statements into Slang against an entry's builtins and tensor slots, and
`assemble_virtual_entry` wraps one or more lowered bodies in a single
`[goldy_compute]` entry. The standalone source is the one-body case.
`ShaderKernel::namespaced` renames locals and workgroup arrays, and
`rename_symbols` maps formal parameters, so several definitions can share one
entry. Composition does not change the Slang generated for standalone kernels.

### Fusing invocations

`invoke(args..)` on a prepared (non-tensor) kernel returns a builder. Its grid
methods (`over_1d`, `over_2d`, `over_3d` or `groups`) produce an `Invocation`,
which holds a dispatch as a value. `Invocation::record` records it alone.
`FusedKernel::prepare` composes a sequence of invocations into one compute
pipeline, and `FusedKernel::record` records that sequence as one dispatch node:

```rust
let stages = [
    scale.invoke(&x, &t, n, 2.0).over_1d(n),
    bias.invoke(&t, &y, n, 1.0).over_1d(n),
];
let fused = goldy::FusedKernel::prepare(&device, &stages)?;
fused.record(&mut scheme, "scale+bias", &stages)?;
```

Each constituent becomes a helper function that the fused entry calls in order,
so `return` and locals stay per stage. Every constituent still loads and stores
its parcels, so `t` above ends in the same state as after the unfused pair.
Arguments that are the same parcel share one binding. This keeps a parcel that
one stage writes and a later stage reads coherent within the dispatch.

Composition is conservative. `prepare` returns `FusionError::Rejected` with a
`FusionRejection` reason when any of these hold:

- a stage has no retained definition;
- a stage binds tensors;
- stages differ in workgroup size or grid;
- two different arguments overlap in memory and one of them is written;
- a parcel written by one stage is read or written by another at anything other
  than the stage's own thread index;
- the fused entry exceeds the portable binding or workgroup-memory limits.

Record the invocations unfused in that case. The fused pipeline depends on which
arguments are the same parcel, not on the parcels themselves, so one
`FusedKernel` can record any invocation sequence with the same shape.

Raw hand-written `[goldy_compute]` shaders continue to work. Simple sources can
also be parsed into the same `KernelDef` shape via
`goldy::slang::try_kernel_def_from_source`, and wrappers can be emitted from ABI
metadata with `emit_wrapper_from_kernel_def` so both paths share frame-table /
PushLayout lowering.

## Supported dialect (MVP)

Allowed: scalar arithmetic/comparisons, `let` / `let mut`, assignment,
field/index access, `if`/`else`, `while`, `for i in 0..n`, casts, selected math
intrinsics (`abs`/`min`/`max`/`floor`/`ceil`/`sqrt`/`sin`/`cos`/`exp`/`pow`/`length`),
vector constructors (`gpu::float2`/`float3`/`float4`), buffer `.len()`, tensor
`.len()` / `.dim(axis)` / `.rank()`, `return`,
workgroup shared arrays + barriers, workgroup sum/max/softmax collectives, and the ID builtins above.

`#[goldy::gpu]` structs may be passed as `&[T]` uniforms; `prepare` prepends
the generated Slang struct.

Rejected with span diagnostics: allocation, iterators/closures, traits/dyn,
recursion, async, panics, arbitrary std calls, `usize`/`isize`, references
except resource parameters, and unsupported patterns.

Element types for buffer slices are currently `u32` / `i32` / `f32` / `bool`, or
a `#[goldy::gpu]` struct for read-only `&[T]`.

## Diagnostics and dumps

1. Proc-macro errors at Rust compile time for unsupported syntax.
2. Slang / pipeline errors during `Kernel::prepare`.
3. Backend errors after target compilation.

Set `GOLDY_DUMP_RUST_KERNELS=1` (or a directory path) to dump canonical Slang and
ABI metadata at prepare time.

## Out of scope (later)

Graphics stages, `GpuType` derive, CUDA scalar parity polish, dynamic shared
memory, specialization, and broad Rust compatibility belong to later phases /
the wider goldy-jit roadmap.
