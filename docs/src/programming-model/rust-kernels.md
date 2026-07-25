# Rust Compute Kernels

Goldy can lower a **restricted Rust GPU dialect** into canonical `[goldy_compute]`
Slang at compile time, then prepare and record through the normal Scheme path.

This is the initial design for issue #78. It is **not** arbitrary Rust, a second
runtime compiler, or CUDA `<<<>>>` syntax. Slang remains the runtime backend
compiler; the proc-macro is an AOT frontend that produces structured
`KernelDef` metadata and typed `record` helpers.

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

| Rust parameter | Slang / Scheme |
|---|---|
| `&[T]` | `BufRO<T>`, `NodeAccess::Read` |
| `&mut [T]` | `Scattered<T>`, `NodeAccess::ReadWrite` |
| `gpu::Out<T>` | `Scattered<T>`, `NodeAccess::Write` |
| `gpu::Uniform<T>` | broadcast resource, `NodeAccess::Read` |
| `u32` / `i32` / `f32` / `bool` | typed scalar push words (no manual `to_bits`) |

Hidden builtins (appended to the Slang signature when used):

| Rust | Slang |
|---|---|
| `gpu::global_id()` | `ThreadId` |
| `gpu::local_id()` | `GroupThreadId` |
| `gpu::workgroup_id()` | `GroupId` |

`workgroup_size` is fixed on the attribute / `KernelDef`. `.groups` / `.over_*`
only control the grid. A different workgroup size is a different pipeline.

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
intrinsics (`abs`/`min`/`max`/`floor`/`ceil`/`sqrt`), buffer `.len()`, `return`,
and the ID builtins above.

Rejected with span diagnostics: allocation, iterators/closures, traits/dyn,
recursion, async, panics, arbitrary std calls, `usize`/`isize`, references
except resource parameters, and unsupported patterns.

Element types for buffer slices are currently `u32` / `i32` / `f32` / `bool`.

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
