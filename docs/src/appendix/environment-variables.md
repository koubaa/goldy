# Environment Variables

Goldy reads several environment variables at runtime for backend selection, validation, debugging, and Slang configuration.

## General

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `GOLDY_BACKEND` | `vulkan`, `vk`, `dx12`, `d3d12`, `directx`, `metal`, `mtl`, `cuda`, `webgpu`, `wgpu`, `cpu` | Platform default (macOS → Metal, Windows → DX12, Linux → Vulkan) | Override backend selection at runtime. Shipped: Vulkan, DX12, Metal. In progress: CUDA, WebGPU. `cpu` is a compute-only host-callable JIT path (never a platform default). |
| `GOLDY_SLANG_PATH` | File path | *(not set)* | Override the path to the Slang shared library (`slang.dll` / `libslang.dylib` / `libslang.so`). Bypasses the default search order (vendored next to executable → extracted from embedded). |
| `GOLDY_FFI_PATH` | File path | *(not set)* | Full path to the `goldy_ffi` shared library (`goldy_ffi.dll` / `libgoldy_ffi.so` / `libgoldy_ffi.dylib`). Used by `goldy-ffi-client` for runtime library loading. |

## Validation

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `GOLDY_VALIDATION` | Comma/semicolon/whitespace-separated list: `api`, `layout`, `layouts`, `all`; or `1` / `true` / `yes` | *(not set)* | Enable validation categories. `api` enables GPU API validation (Vulkan validation layers, Metal shader validation, CUDA Driver diagnostics: PTX JIT logs, eager stream sync, launch-limit checks; WebGPU/wgpu validation error scopes on shader/PSO create and bind groups). `layout` enables Rust/Slang struct layout and buffer stride checks. `all` enables both (plus timeline/scheme where applicable). The shorthand `1` / `true` / `yes` enables **GPU API only** (layout stays opt-in). Deep CUDA memory/race checking is **not** covered — use external `compute-sanitizer`. |
| `GOLDY_VALIDATE_LAYOUTS` | `1`, `true`, `yes` | *(not set)* | Legacy toggle for layout validation only. Equivalent to `GOLDY_VALIDATION=layout`. |

### Validation Examples

```bash
# GPU API validation (Vulkan layers, Metal shader validation, CUDA Driver diagnostics)
GOLDY_VALIDATION=api cargo run --example triangle

# Layout + stride checks only
GOLDY_VALIDATION=layout cargo run --example triangle

# Everything
GOLDY_VALIDATION=all cargo run --example triangle

# Shorthand for GPU API only
GOLDY_VALIDATION=1 cargo run --example triangle

# CUDA with API validation (JIT logs, eager sync, launch limits)
GOLDY_BACKEND=cuda GOLDY_VALIDATION=api cargo test --features cuda
```

## DX12-Specific

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `GOLDY_DX12_DEBUG` | `1`, `true` | On in debug builds | Enable the D3D12 debug layer. On by default in debug builds; set explicitly for release builds. |
| `GOLDY_DX12_NO_DEBUG` | `1`, `true` | *(not set)* | Force-disable the D3D12 debug layer even in debug builds. Useful to avoid debug-layer crashes in parallel test threads. |
| `GOLDY_DX12_GBV` | `1`, `true` | *(not set)* | Enable D3D12 GPU-Based Validation. Catches UAV/SRV descriptor mismatches, resource state errors, and out-of-bounds access on the GPU timeline. Very slow — use for targeted debugging only. |
| `GOLDY_DX12_FORCE_WARP` | `1`, `true` | *(not set)* | Force the DX12 backend to use the WARP software rasterizer, even when hardware GPUs are present. Use for headless CI or reproducing WARP-specific rendering bugs. |
| `GOLDY_DX12_ALLOW_WARP` | `1`, `true` | *(not set)* | Allow the WARP adapter to appear in device enumeration. Without this or `GOLDY_DX12_FORCE_WARP`, WARP is hidden. |

## Debugging

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `GOLDY_DUMP_SHADERS` | Directory path | *(not set)* | Dump compiled shaders to the specified directory at compile time. Vulkan: `{entry}_h{handle}_vulkan.spv`. DX12: `{entry}_h{handle}_dx12.dxil`. Metal: `{idx}_{entry}.metal`. CUDA: `{entry}_h{handle}_{spec}_cuda.cu` (Slang CUDA C++) and `.ptx` (NVRTC/Slang PTX loaded by the CUDA driver), plus `goldy_apply_dispatch_shape.{cu,ptx}` for the graph updater. |
| `GOLDY_DUMP_RUST_KERNELS` | `1` / `true` / directory path | *(not set)* | Dump canonical `[goldy_compute]` Slang and structured ABI metadata produced by `#[goldy::compute]` during `Kernel::prepare`. `1`/`true` writes under the process temp dir (`goldy_rust_kernels/`). |
| `GOLDY_CPU_SHADERS` | `1`, `true`, `yes` | *(not set)* | Documented gate for the standalone `goldy::cpu_shaders` APIs. GPU backends ignore this. Scheme submit uses `GOLDY_BACKEND=cpu` instead. |
| `GOLDY_GPU_PROFILE` | Any non-empty value; optional `chrome[=path]` | *(not set)* | Enable GPU timestamp profiling logs. On Vulkan/DX12, records per-dispatch GPU durations. On Metal, records command-buffer GPU duration. `chrome` / `chrome=/path.json` also writes a Perfetto Chrome-trace JSON file. Disables retained CB reuse while active. |
| `GOLDY_SHADER_TIMING` | `1` / any value other than `0` | *(not set)* | Print stderr wall-clock breakdown of Slang cache lookup, search-path hashing, and PSO create during shader compile |
| `GOLDY_METAL_CAPTURE` | `1` / path / `path,skip=N,frames=M` | *(not set)* | **Metal only.** Opt-in programmatic `MTLCaptureManager` GPU capture for Xcode Metal Debugger. `1`/`true`/`yes` captures to Developer Tools; a path writes a `.gputrace`. Optional `skip=N` (default 60) skips warm-up submits; `frames=M` (default 1) captures M submits. Automatically sets `METAL_CAPTURE_ENABLED=1` if unset. Open the `.gputrace` in Xcode → Performance to inspect register pressure, occupancy, and per-line shader costs. |
| `GOLDY_API_LOG` | File path | *(not set)* | **Metal only.** Append NDJSON Metal API call traces (dispatches, encoder open/close, commits) to the given file. |
| `GOLDY_API_LOG_SYNC` | `1` | *(not set)* | Force synchronous `GOLDY_API_LOG` writes (for tests). |

### Metal capture example

```bash
# Warm up 120 submits, then write one .gputrace for Xcode Metal Debugger
GOLDY_METAL_CAPTURE=/tmp/capture-tiger.gputrace,skip=120,frames=1 \
  target/release/with_winit_bin --timeout-secs 12 --no-vsync

# Open /tmp/capture-tiger.gputrace in Xcode → click Performance → select fine_area
```

## Interop with System Variables

Goldy also respects these non-Goldy environment variables:

| Variable | Backend | Description |
|----------|---------|-------------|
| `VK_INSTANCE_LAYERS` | Vulkan | If set to include `VK_LAYER_KHRONOS_validation`, Goldy enables Vulkan validation regardless of `GOLDY_VALIDATION`. |
| `VK_LAYER_PATH` | Vulkan | Standard Vulkan loader variable for locating validation layer manifests. |
| `MTL_SHADER_VALIDATION` | Metal | When `GOLDY_VALIDATION` enables API validation and this variable is unset, Goldy sets it to `1` before creating the first Metal device. If you set it yourself, Goldy does not override it. |
| `METAL_CAPTURE_ENABLED` | Metal | Required for programmatic GPU capture outside Xcode. Goldy sets this to `1` automatically when `GOLDY_METAL_CAPTURE` is set (if unset). |
| `CUDA_LAUNCH_BLOCKING` | CUDA | When `GOLDY_VALIDATION` enables API validation and this variable is unset, Goldy sets it to `1` before CUDA driver init. If you set it yourself, Goldy does not override it. Forces synchronous kernel launches so errors surface at the launch site. |
| `WGPU_BACKEND` | WebGPU | wgpu instance backend mask (`vulkan`, `metal`, `dx12`, `gl`, …). Goldy passes this through `wgpu::InstanceDescriptor::from_env_or_default()`. CI uses `vulkan` on Linux, `metal` on macOS, and `dx12` on Windows. |
| `WGPU_FORCE_FALLBACK_ADAPTER` | WebGPU | When set (`1`/`true`), wgpu prefers a software adapter (WARP on Windows). Used in Windows CI because hosted runners have no discrete GPU. |
