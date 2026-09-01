# Debugging Tips

## Metal Backend Issues

### VRAM / heap diagnostics (`RUST_LOG` + `goldy::diag::*`)

Heap allocation events, submit summaries, and periodic residency snapshots use dedicated `tracing` targets (Metal backend only for now). Enable with `RUST_LOG`, for example:

```bash
RUST_LOG=goldy::diag::alloc=info,goldy::diag::submit=info,goldy::diag::mem=info \
  cargo run --example triangle
```

- `goldy::diag::alloc` — primary full, overflow heap creation, compact, reset
- `goldy::diag::submit` — dispatch count and pipeline labels per submit (pre-scans commands; only runs when this target is enabled)
- `goldy::diag::mem` — `metal-alloc` snapshot every N submits (`GOLDY_MEM_CADENCE`, default 60)

### Runtime shader validation (`MTL_SHADER_VALIDATION`)

When **GPU API validation** is on, Goldy sets **`MTL_SHADER_VALIDATION=1`** before the first Metal device is created, **only if** `MTL_SHADER_VALIDATION` is not already set. That opts into Apple’s runtime shader validation (see Apple’s Metal debugging documentation for other allowed values). GPU API validation includes **`GOLDY_VALIDATION=1`/`true`/`yes`**, the **`api`** token, or **`all`** (see [Unified validation](#unified-validation-goldy_validation)). Like Vulkan, this is applied at backend initialization; avoid touching Metal before `Instance::new()` if you rely on Goldy to set the variable.

```bash
GOLDY_VALIDATION=1 cargo run --example triangle
GOLDY_VALIDATION=api cargo run --example triangle
GOLDY_VALIDATION=all cargo run --example triangle
```

### Shader Compiles but Uniforms Don't Update (Static Animation)

If using `bind_resources()` with a Metal shader that uses `ParameterBlock`:

1. **Check pipeline has ParameterBlock layouts**: The Metal backend needs reflection data to populate the argument buffer. Enable debug logging:
   ```bash
   RUST_LOG=goldy::backend::metal=trace cargo run --example myexample
   ```
   Look for: `"Allocated bindless argument buffer"` and `"BindResources: Wrote GPU address"`.

2. **Verify argument buffer binding**: Check logs for `"BindResources: Bound ParameterBlock argument buffer at slot X"`. If missing, the buffer isn't being bound to the shader.

3. **Ensure buffer is heap-allocated**: Bindless buffers must be allocated from the Metal heap. Check for `"Encoded buffer N at arg buffer offset"` during buffer creation.

## DX12 Backend Issues

### Bindless Buffers Show Wrong Data (All Zeros, Garbage)

When using `bind_resources()` with storage buffers on DX12, the shader may read incorrect data. This is often caused by SRV/UAV descriptor mismatch:

**Background**: DX12 requires different descriptor types for read vs write access:
- `StructuredBuffer<T>` (read-only) → needs **SRV** (Shader Resource View)
- `RWStructuredBuffer<T>` (read-write) → needs **UAV** (Unordered Access View)

Goldy creates both SRV and UAV descriptors for storage buffers, stored as:
- `bindless_offset` → UAV index
- `bindless_srv_offset` → SRV index

**Current behavior for `bind_resources()`**:
- **Render shaders**: Always use SRV offsets (render shaders only read)
- **Compute shaders**: First buffer uses SRV (read input), subsequent buffers use UAV (write outputs)

This matches the common ping-pong pattern (e.g., Game of Life: read from buffer A, write to buffer B).

### Python/FFI Buffers with Wrong Element Stride

If a `StructuredBuffer<uint>` reads garbage on DX12 but works on Vulkan, check the buffer's element stride. DX12's structured buffer views require the correct `StructureByteStride`:

- `uint` / `int` / `float` → stride = 4
- `uint2` / `float2` → stride = 8
- Raw bytes → stride = 1 (but incompatible with `StructuredBuffer`)

Python buffers automatically detect stride from numpy dtype. If using raw bytes, ensure you're not accessing them as a typed `StructuredBuffer` in the shader.

## Vulkan Backend Issues

### GPU validation (Khronos validation layer)

Vulkan API misuse is easiest to catch with the Khronos validation layer. CI GPU jobs set `GOLDY_VALIDATION=all` and `GOLDY_VALIDATION_FATAL=1`, and restore `VK_LAYER_PATH` on lavapipe so ERROR messages fail the suite. Locally:

1. **`GOLDY_VALIDATION`** — short form **`1`/`true`/`yes`** turns on **GPU API validation only** (Khronos layer + `VK_EXT_debug_utils` here; Metal shader validation on macOS). Token lists are also supported (see [Unified validation](#unified-validation-goldy_validation) below). Requires validation layers on the machine (e.g. [Vulkan SDK](https://vulkan.lunarg.com/sdk/home), or on Debian/Ubuntu often `vulkan-validationlayers`).

   ```bash
   GOLDY_VALIDATION=1 cargo test --features vulkan
   GOLDY_VALIDATION=api cargo test --features vulkan
   GOLDY_VALIDATION=1 cargo run --example triangle
   ```

2. **Loader-only** — set `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` (and ensure the loader can find the layer). Goldy detects that substring and enables the same instance extensions and layer list as GPU API validation.

Khronos messages are captured with a `VK_EXT_debug_utils` messenger (target `goldy::validation`). They do **not** fail tests or `Result` calls by themselves: `vk*` can still return success. To fail:

```bash
GOLDY_VALIDATION_FATAL=1 GOLDY_VALIDATION=api cargo test --features vulkan
GOLDY_VALIDATION_FATAL=1 GOLDY_VALIDATION=all cargo test --features vulkan
```

ERROR-severity messages then surface as `Err` from submit/wait/create and panic on backend drop (so `cargo test` fails even if the last call returned `Ok`).

### Unified validation (`GOLDY_VALIDATION`)

| Input | Effect |
|-------|--------|
| `GOLDY_VALIDATION=layout` | Layout / stride checks only (no built-in API validation hooks unless combined). |
| `GOLDY_VALIDATION=layout,api` or `layout api` or `layout;api` | Layout plus graphics API validation (separators: comma, semicolon, or whitespace). |
| `GOLDY_VALIDATION=timeline` | WSI timeline invariants: post-wait semaphore counter check in Vulkan `acquire()`. |
| `GOLDY_VALIDATION=scheme` or `readback` | Retained-scheme grant readback invariants (frame/grant pairing, staging pool checks). |
| `GOLDY_VALIDATION=all` | Layout + API + timeline + scheme + host_access (same as listing those tokens). |
| `GOLDY_VALIDATION_FATAL=1` | Separate switch: with GPU API validation on, Vulkan ERROR messages fail Goldy `Result` calls and panic on backend drop. |
| `GOLDY_VALIDATION=1` / `true` / `yes` | **GPU API only** — does **not** enable layout checks (keeps dispatch-time layout work off unless you opt in). |

The **`api`** token selects Goldy’s graphics-API validation path (Vulkan validation layer + `VK_EXT_debug_utils` where built; Metal `MTL_SHADER_VALIDATION` when applicable). For Vulkan you can still set **`VK_INSTANCE_LAYERS`**, **`VK_LAYER_PATH`**, etc. yourself if you prefer the loader directly. **`GOLDY_VALIDATE_LAYOUTS=1`** still works unchanged and is equivalent to enabling the layout family only.

**When these take effect:** You do not need a special “wrapper process” or argv-time setup. Variables must be set **before the first Goldy call that initializes the backend** — in practice, before `Instance::new()` (or the FFI/Python equivalent that creates the instance). That is earlier than “first device” in the abstract API sense, but it is still just “before GPU init in this process,” not necessarily before `main` if nothing touches the GPU earlier. If another crate or static initializer touches Vulkan/Metal first, set env at the very start of `main` (or in the test harness `#[init]`).

**Developer experience:** Prefer the usual shell form `GOLDY_VALIDATION=1 cargo test …` / `cargo run …` (same pattern as `GOLDY_BACKEND`, `RUST_LOG`). That works for humans, copy-paste docs, and agents that already run `cargo test` per `AGENTS.md` without learning a repo-specific script.

### Shader Not Working (Static Output, No Animation)

Dump the SPIR-V using `GOLDY_DUMP_SHADERS` (see [Inspecting Compiled Shader Assembly](#inspecting-compiled-shader-assembly)) and check:

1. **Push Constants Not Generated**: Verify the SPIR-V contains `OpVariable ... PushConstant`. If it shows `Uniform` storage class instead, the shader isn't correctly declaring push constants.

2. **Descriptor Set/Binding Mismatch**: Check `OpDecorate` lines for `Binding` and `DescriptorSet`. Expected bindings:
   - Binding 0: Storage buffers
   - Binding 1: Uniform buffers  
   - Binding 2: Sampled images
   - Binding 3: Samplers

### Slang Preprocessor Issues

See [shaders/README.md](shaders/README.md#preprocessor-defines) for Slang-specific preprocessor behavior that can cause cross-platform issues.

## Rust vs Slang struct layout validation

Wrong `#[repr(C)]` layouts for uniforms or structured-buffer types often show up as subtle bugs (garbage values, misaligned reads). Goldy can compare your Rust layout to Slang’s reflection on the **same** shader compile that emits SPIR-V / DXIL / MSL—no second compile.

### Enabling validation

Set **`GOLDY_VALIDATE_LAYOUTS`** to a truthy value, or use the unified form **`GOLDY_VALIDATION=layout`**, **`GOLDY_VALIDATION=layout,api`**, or **`GOLDY_VALIDATION=all`**, before creating the device or compiling shaders:

| Value   | Effect        |
|---------|---------------|
| (unset) | No validation |
| `1`     | Validate      |
| `true`  | Validate      |
| `yes`   | Validate      |

```bash
GOLDY_VALIDATE_LAYOUTS=1 cargo run --example gradient --release
GOLDY_VALIDATION=layout cargo run --example gradient --release
GOLDY_VALIDATION=all cargo run --example gradient --release
```

If a layout check fails, compilation returns an error describing size / field offset / name mismatches.

### In application code

1. Match the Rust struct name to the Slang `struct` name you want checked (reflection uses `FindTypeByName`).
2. Add **`#[derive(LayoutCheckable)]`** (re-exported from the `goldy` crate).
3. Pass **`&[YourStruct::LAYOUT_CHECK]`** as the last argument to **`ShaderModule::from_slang_with_options`** (other `from_slang*` helpers pass empty checks).

When the env var is off, those checks are skipped and `from_slang_with_options` behaves like a normal compile path.

The **`gradient`** and **`checkerboard`** examples demonstrate this with `TimeUniforms` vs `struct TimeUniforms` in the shader sources.

Standalone reflection without shader creation remains available via **`Device::reflect_struct`** and **`SlangCompiler::reflect_struct_layout`**.

## Buffer stride validation (push constants)

A common footgun is uploading uniform or structured data with the wrong **element stride** (for example passing `bytemuck::bytes_of(&uniforms)` into `Buffer::with_data`, which infers `T = u8` and stride 1). That can work on Vulkan but yield silent wrong reads on Direct3D 12 structured-buffer views.

`GOLDY_VALIDATE_LAYOUTS` covers this too. When enabled, Goldy validates at **dispatch / draw time** that each buffer bound via `bind_resources_typed` has the same element stride the shader expects for `goldy_buf_ro<T>`, `goldy_scattered<T>`, `goldy_broadcast<T>`, and `goldy_byte_address` (byte stride 1).

```bash
GOLDY_VALIDATE_LAYOUTS=1 cargo run --example compute_to_surface
```

This is **off by default** (no per-dispatch cost). Turn it on when results look wrong after a binding or buffer upload change, or when debugging cross-backend differences.

The check compares the buffer’s recorded stride (from `Buffer::with_data`, `with_bytes_stride`, `new_with_stride`, etc.) against Slang’s reflected size of `T` for each `goldy_*<T>(slot)` call. If reflection cannot resolve a type name, that slot is skipped with a warning in the `goldy::slang` tracing target.

## Inspecting Compiled Shader Assembly

When a shader produces unexpected results, inspecting the compiled bytecode can reveal codegen issues that aren't visible in the source. This is useful when:
- Push constants/uniforms show wrong values
- Resource bindings don't work as expected
- Shader logic appears correct but output is wrong

### Dumping Compiled Shaders

Set the `GOLDY_DUMP_SHADERS` environment variable to a directory path:

```bash
GOLDY_DUMP_SHADERS=/tmp/shaders cargo run --example game_of_life
```

This writes compiled bytecode for each shader entry point:
- `{entry}_dx12.dxil` - DirectX 12 (DXIL)
- `{entry}_vulkan.spv` - Vulkan (SPIR-V)
- `{idx}_{entry}.metal` - Metal (MSL)
- `{entry}_h{handle}_{spec}_cuda.cu` - CUDA C++ generated by Slang
- `{entry}_h{handle}_{spec}_cuda.ptx` - PTX loaded by the CUDA driver (`cuModuleLoadData`)
- `goldy_apply_dispatch_shape.cu` / `.ptx` - CUDA graph indirect-dispatch updater (NVRTC)

CUDA `{spec}` is `id` for the identity kernel, or `f4rgba8` (joined with `-` if several slots) for `DirectSpatial<float4>` ↔ `Rgba8Unorm` specialization.

### Disassembling DXIL

Use `dxc` (DirectX Shader Compiler) to disassemble:

```bash
dxc -dumpbin cs_main_dx12.dxil > cs_main.txt
```

Key things to check:
- **Buffer Definitions**: Verify struct layouts and sizes match expectations
- **cbufferLoadLegacy**: Each call loads a 16-byte register; check `regIndex` values
- **extractvalue**: Which component (0-3) is extracted from loaded data

### Disassembling SPIR-V

Use `spirv-dis` from the Vulkan SDK:

```bash
spirv-dis cs_main_vulkan.spv > cs_main.txt
```

Key things to check:
- **OpVariable storage class**: Push constants should use `PushConstant`, not `Uniform`
- **OpAccessChain**: Array/struct element access - verify indices are correct
- **OpCompositeExtract**: Component extraction from vectors/structs

## Runtime Logging

For debugging resource binding issues, add temporary logging:

```rust
tracing::debug!(
    "Buffer {} at bindless index {} (storage={})",
    handle, index, is_storage
);
```

Enable with `RUST_LOG=goldy=debug cargo run`.

## FFI / Language Bindings Issues

### "undefined identifier 'DescriptorHandle'" in .NET/C# or other bindings

This error occurs when the FFI loads the wrong Slang DLL. Goldy requires Slang 2026.13+ with SM 6.6 bindless support (`DescriptorHandle` intrinsic).

**Cause**: The Slang library search falls back to an older `slang.dll` from the Vulkan SDK instead of the bundled `slang-compiler.dll`.

**Automatic Slang**: The Rust crate embeds Slang at compile time (`goldy/build.rs`) and
extracts it on first shader compile. Python dev installs (`pip install -e ".[dev]"`)
use that path — no `build-slang.py` required.

Release / redistribution layouts copy Slang next to native libs:
- **.NET**: `build-native.ps1` / `build-native.sh` → `runtimes/{rid}/native/`
- **Python wheels**: `build-slang.py` before `maturin build` (CI only)
- **C++**: CMake installs Slang alongside `goldy_ffi`

If you still encounter this error, rebuild the native extension or set `GOLDY_SLANG_PATH`.

**Manual Override**: To use a custom Slang version, set the environment variable:
```bash
export GOLDY_SLANG_PATH=/path/to/slang-compiler.dll
```

**Slang loader search order** (see `goldy/src/slang/loader.rs`):
1. `GOLDY_SLANG_PATH` environment variable
2. Same directory as the running executable (wheel / FFI layout)
3. Cache directory (extracted from bytes embedded at compile time)

The required Slang version and file list are defined in `slang/manifest.json`.