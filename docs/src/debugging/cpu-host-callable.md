# CPU host-callable shaders (debug)

Goldy can JIT the **same Slang compute kernels** it runs on GPU and execute them
on the host via Slang `SLANG_SHADER_HOST_CALLABLE` (`getEntryPointHostCallable`).
This is an **opt-in debug path** so you can step a stage in a CPU debugger
without maintaining a second handwritten Rust implementation.

It is **not** a production CPU renderer, not a scheme/submit backend, and not a
replacement for Vulkan / DX12 / Metal / CUDA.

## When to use it

- Stepping a `#[goldy::compute]` / `[goldy_compute]` kernel in a native debugger
- Checking buffer math on host slices before wiring GPU parcels
- Replacing deleted CPU twins in clients (for example Ekrano) with the real Slang

## How to run a kernel

```rust
use goldy::cpu_shaders::{self, CpuBinding};
use goldy::slang::SlangCompiler;

let compiler = SlangCompiler::new()?;
let kernel = cpu_shaders::compile_kernel(&compiler, &kernel_def, &["shaders"])?;
let mut data: Vec<u32> = (0..64).collect();
kernel.dispatch_1d(64, &mut [CpuBinding::u32s(&mut data)])?;
```

`cpu_shaders::compile` accepts `[goldy_compute]` source (or raw
`[shader("compute")]` after you pack bindings yourself). The CPU wrapper keeps
`BufRO` / `Scattered` as typed `uniform` entry-point parameters instead of
Goldy bindless slot indices.

Set `GOLDY_CPU_SHADERS=1` when you want the documented env gate (reserved for a
future `Device` debug option). The compile APIs above are already opt-in; GPU
paths ignore the variable.

Host-callable JIT uses vendored `slang-llvm` next to `libslang`. No extra C++
toolchain is required when that library is present. Do **not** set Slang
`SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM` with the current vendored Slang:
`getEntryPointHostCallable` SIGSEGVs. Goldy omits that flag.

## What lowers

| Type | CPU ABI |
|------|---------|
| `BufRO<T>`, `Scattered<T>` (`T` = `uint` / `int` / `float` / `bool`) | `{ T* data; size_t count }` |
| Scalar `uint` / `int` / `float` / `bool` | 4-byte word |
| `ThreadId`, `GroupThreadId`, `GroupId` | `SV_DispatchThreadID` / `SV_GroupThreadID` / `SV_GroupID` |
| `goldy_buf_len(buf)` | `GetDimensions` on the CPU structured buffer |

Workgroups run serially through the Slang CPU prelude (`ComputeVaryingInput`
start/end group IDs).

## What does not lower yet

| Type | Notes |
|------|--------|
| Broadcast / `gpu::Uniform<T>` / constant-buffer structs | Needs a CPU constant-buffer view |
| `ByteAddress` | CPU prelude has byte-address types; Goldy ABI packing is not wired |
| `Interpolated<T>` (sampled textures) | No software texture path |
| `DirectSpatial<T>` (storage images) | No software texture path |
| `Filter` / samplers | Texture-only |
| `[goldy_vertex]` / `[goldy_fragment]` | Compute only |
| Goldy bindless frame table / parcels | Dispatch is on host slices, not scheme submit |

Fine rasterization stays GPU-only until textures work. `Interlocked` / groupshared
behavior follows the Slang CPU prelude (typically mutex or sequential atomics)
and is not a substitute for GPU memory-model testing.

## Related

- [Debugging and Observability](overview.md)
- [Rust Compute Kernels](../programming-model/rust-kernels.md)
- [Environment Variables](../appendix/environment-variables.md)
