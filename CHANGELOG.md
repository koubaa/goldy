# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`MeshPipeline` / `dispatch_mesh`** — mesh (+ optional amplification) graphics
  pipelines on Vulkan (`VK_EXT_mesh_shader`) and DX12 (mesh tier 1). Record with
  `SchemeRenderPassBuilder::set_mesh_pipeline` and `dispatch_mesh`. Skip when
  `DeviceCapabilities::mesh_shaders` is false. Metal / WebGPU / CUDA are not wired yet.

- **`DeviceCapabilities` RT / mesh bits** — `ray_query`, `ray_tracing_pipelines`,
  `mesh_shaders`, and `amplification_shaders` report adapter hardware (Vulkan
  extensions + features, DXR / mesh tiers, Metal `supportsRaytracing` / GPU
  family). Slang compile plumbing accepts RT and mesh stages (`rgen_main`,
  `mesh_main`, …); reflection maps `RaytracingAccelerationStructure` to
  `ResourceKind::AccelerationStructure`.

- **`RayTracingPipeline` / `Scheme::trace_rays`** — one raygen, miss, and triangle
  closest-hit group with a backend-owned shader-binding table. Vulkan
  `vkCmdTraceRaysKHR` and DX12 `DispatchRays` (DIRECT queue). Bind raygen
  resources like compute (`Accel`, `Scattered`, `DispatchRaysIndex`). Skip when
  `DeviceCapabilities::ray_tracing_pipelines` is false.

- **`GOLDY_VALIDATION=host_access`** — page-protect CPU-backend parcel storage
  (page-aligned, unshared mapping + guard page). Stray host pointers fault
  outside legal CPU windows (upload, dispatch, withdraw). Included in `all`.
  First slice; native mapped staging can use the same allocator later.
- **`GOLDY_VALIDATION_FATAL=1`** — Vulkan Khronos ERROR messages captured by a
  debug-utils messenger fail Goldy `Result` calls and panic on backend drop.
  Independent of `GOLDY_VALIDATION` (`all` does not imply it). Vulkan lavapipe
  CI jobs set `GOLDY_VALIDATION=all` and `GOLDY_VALIDATION_FATAL=1` (and restore
  `VK_LAYER_PATH`) so a Khronos ERROR fails the suite. Metal and DX12 jobs do
  not set `GOLDY_VALIDATION_FATAL` (it is Vulkan-only).

### Fixed

- WebGPU bind-group cache keys include the exclusive pipeline identity so
  structurally identical layouts from different PSOs are not reused.
- Tight (`src_row_pitch == 0`) `copy_buffer_to_texture_parcel` resubmits are
  not asserted as retention hits on WebGPU; they are standalone partitions.

- Game of Life render shader no longer uses `fwidth` (Slang Metal fragment target
  rejects derivative builtins). Restores screenshot tests and the Python headless example.
- Python Game of Life examples pass `VertexBufferLayout.empty()` for the
  `VertexId` fullscreen pass. The Python `RenderPipelineDesc` default is still
  Vertex2D, which made `draw_fullscreen()` hit Khronos
  `VUID-vkCmdDraw-None-04007` under `GOLDY_VALIDATION_FATAL=1`.

- **Rust compute kernels (issue #78, initial design)** — `#[goldy::compute]` proc-macro
  lowers a restricted GPU dialect to canonical `[goldy_compute]` Slang plus structured
  `KernelDef` / `KernelParam` ABI metadata (`goldy_shader_ir`). Host API:
  `Kernel::prepare` (lazy compile/cache) and typed `record(...).over_1d` / `.groups`.
- **CPU host-callable shaders (issue #292, initial debug path)** — `ShaderTarget::HostCallable`
  compiles the same Slang compute kernels via `getEntryPointHostCallable` and runs them
  on host buffers (`goldy::cpu_shaders`). Opt-in; not a production backend. See
  [CPU host-callable shaders](docs/src/debugging/cpu-host-callable.md).
- **CPU compute backend (`GOLDY_BACKEND=cpu`)** — compute-only device that JITs
  `[goldy_compute]` kernels and executes scheme submits on host parcels. Textures,
  samplers, and vertex/fragment shaders are rejected. Never a platform default.
- Shared `KernelAbi` bridge for virtual-main: `try_kernel_def_from_source`,
  `emit_wrapper_from_kernel_def` so Rust and raw Slang paths share frame-table wrappers.
- `goldy_buf_len` helper for portable buffer `.len()` lowering on SPIR-V/DX12.
- Docs: [Rust Compute Kernels](docs/src/programming-model/rust-kernels.md);
  `GOLDY_DUMP_RUST_KERNELS` dump env var.
- CUDA+DX12 present scratch is a depth-3 ring independent of the DXGI image, with
  separate ready (CUDA-produced) and recycle (DX12-produced) fences so compute N+1
  does not wait present-copy N. Documented as an interop staging tradeoff until
  CUDA/DX12 sync APIs improve.

## [0.2.0] - 2026-07-23

Fondaco Machine rewrite. This is a **breaking** release relative to 0.1.0 — the
imperative command-encoder API is gone. Schemes, parcels, and exchanges are the
public programming model.

### Added

- **Scheme-first API** — retained dependency graphs with ownership-derived ordering
- **Exchanges** — `SurfaceExchange` (present), `MemoryExchange` (withdraw/deposit)
- **Parcels** — stable handles for buffers, textures, and related GPU data
- **Contexts, retained/transient pools, VRAM allocator** — shared device-scoped pools
- **Compute-to-surface** — compute shaders write swapchain drawables directly
- **Growable buffers** — `Buffer::resize_to` with stable handles
- **Virtual entry points** — `[goldy_compute]`, `[goldy_vertex]`, `[goldy_fragment]` with `goldy_exp`
- **Metal backend** (macOS) — native Metal Tier 2+, not MoltenVK
- **DX12 Enhanced Barriers** baseline; Vulkan raised to **1.4+**
- **Language bindings** — Python (PyPI), .NET (NuGet), C++ (FFI)
- **21 Rust examples** — triangle through multi-window and headless workflows
- **`goldy_derive`** — `LayoutCheckable`, `StructuredBufferElement`

### Changed

- Submission settlement uses claim/`consume`/`discard` and `wait_until_settled`
- Imperative CPU readback paths replaced by `MemoryExchange` withdraw/deposit
- Slang compiler remains embedded at build time (override with `GOLDY_SLANG_PATH`)

### Experimental

- CUDA and WebGPU backends (feature-gated prototypes; not production-ready)

### Platforms

- Windows x86_64 (DX12 default, Vulkan optional)
- Linux x86_64 (Vulkan; Wayland surfaces)
- macOS aarch64 (Metal)

### Packaging

- crates.io packages exclude binding trees (`python/`, `ffi/`, `dotnet/`, `cpp/`, …)
- Publish order: `goldy_shader_ir`, then `goldy_derive`, then `goldy`

## [0.1.0] - 2026-01-11

Initial release of Goldy, a modern GPU library for Rust.

### Added

- **Vulkan 1.3+ backend** — dynamic rendering and related modern features
- **DX12 backend** (Windows)
- **Slang shader compilation** — compile Slang to SPIR-V at runtime
- **Shader library system** — reusable modules with `import` support
- **Built-in `goldy_exp` library** — experimental shader utilities
- **Surface rendering** — window/swapchain via `raw-window-handle`
- **Render targets** — off-screen rendering with CPU readback
- **Compute pipelines**
- **Bind groups** — descriptor-set abstraction for uniforms and textures
- **18 examples** — triangle through compute particles and Game of Life

### Platforms

- Windows x86_64 (Vulkan, DX12)
- Linux x86_64 (Vulkan)
- macOS aarch64 (Vulkan via MoltenVK)

### Dependencies

- Slang compiler (auto-downloaded during build, or via Vulkan SDK)
- Vulkan 1.3+ capable GPU (2018+ recommended)
