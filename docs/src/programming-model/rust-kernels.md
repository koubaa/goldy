# Rust Compute Kernels

Goldy can lower a **restricted Rust GPU dialect** into canonical `[goldy_compute]`
Slang at compile time, then prepare and record through the normal Scheme path.

This is the initial design for issue #78. It is **not** arbitrary Rust, a second
runtime compiler, or CUDA `<<<>>>` syntax. Slang remains the runtime backend
compiler; the proc-macro is an AOT frontend that produces structured
`KernelDef` metadata and typed `record` helpers.

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
in a larger expression. Omit `::<N>` to use `workgroup_size.x`.

## Architecture

```text
Rust kernel
    │
    ▼
goldy_derive::compute
    ├── syn AST validation (GPU dialect)
    ├── goldy_shader_ir
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

Raw hand-written `[goldy_compute]` shaders continue to work. Simple sources can
also be parsed into the same `KernelDef` shape via
`goldy::slang::try_kernel_def_from_source`, and wrappers can be emitted from ABI
metadata with `emit_wrapper_from_kernel_def` so both paths share frame-table /
PushLayout lowering.

## Supported dialect (MVP)

Allowed: scalar arithmetic/comparisons, `let` / `let mut`, assignment,
field/index access, `if`/`else`, `while`, `for i in 0..n`, casts, selected math
intrinsics (`abs`/`min`/`max`/`floor`/`ceil`/`sqrt`/`sin`/`cos`/`exp`/`pow`/`length`),
vector constructors (`gpu::float2`/`float3`/`float4`), buffer `.len()`, `return`,
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
