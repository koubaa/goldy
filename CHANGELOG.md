# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Params-only scheme dirtiness** — `Scheme::set_node_pipeline`,
  `set_node_dispatch`, and `set_node_param` mark a scheme params-dirty instead
  of structurally dirty. The next submit recomputes partition fingerprints and
  re-records only partitions whose baked payload changed; other retained
  partitions resubmit. The schedule cache keys on bindings only; emitted
  command lists key on a separate emission fingerprint so a pipeline swap
  cannot reuse a stale `SetPipeline`. Structural mutations (new nodes,
  bindings) still drop all retained command lists.

- **`NodeId`** — finalizing a dispatch (`dispatch`, `dispatch_shape_parcel`)
  returns the recorded node's identity, which is what `set_node_pipeline`,
  `set_node_dispatch`, and `set_node_param` now take instead of a raw index.
  Ids carry their originating scheme, so a node id from another scheme is
  rejected rather than silently addressing an unrelated node. Nodes are only
  appended, so an id keeps pointing at the same dispatch site for the life of
  the scheme and can key per-site history across frames.

- **`ReplayStats::clean_submits`** — submissions that found the scheme clean
  (no structural, params, or topology dirtiness). Unlike `resubmit_hits`, it
  reflects scheme state rather than backend command-list retention, so it is
  present and meaningful on every backend including Metal and WebGPU.

- **`ShaderModule::variant`** — frontend-retained source, search paths, and
  preprocessor defines. `variant(extra_defines)` merges/overrides defines and
  allocates a new backend shader handle without a per-backend trait method.
  Post-virtual-main source is cached once per module (`OnceLock`).

- **Bakeable scalar params** — every generated compute wrapper reads each scalar
  `with_param` slot through a macro that defaults to that backend's own read
  expression (`_GOLDY_SPEC_<ENTRY>_UW<slot>`, named by
  `slang::virtual_main::scalar_specialization_macro`; the push-constant word on
  Vulkan/DX12/Metal, the user-params uniform field on WebGPU, the kernel
  argument on CUDA). Defining it to a `u32` wire-word literal bakes the value
  into the compiled program, so the shader compiler sees a constant instead of a
  load, with no change to the shader source and no change to the parameter
  layout — unbaked slots keep their positions and the recorded command list is
  untouched, so a baked pipeline can be swapped onto an already-recorded
  dispatch node. Note that baking must not change the pipeline's *binding*
  layout: WebGPU derives bind group layouts from compiled WGSL usage, so baking
  every scalar of an entry point leaves the user-params uniform unreferenced and
  drops the binding. This is the primitive behind
  [shader specialization prediction](docs/src/design/shader-specialization.md).

- **Unlocked compute Slang compile** — `ComputePipeline::new` runs Slang
  outside `device.inner.backend.lock()` on Vulkan and DX12, then seeds the
  stage cache before PSO create (still under the lock). Mock/CPU/Metal/WebGPU/CUDA
  keep in-lock compile until they implement `seed_compute_stage`. No
  `PipelineFactory` yet.

- **`MeshPipeline` / `dispatch_mesh`** — mesh (+ optional amplification) graphics
  pipelines on Vulkan (`VK_EXT_mesh_shader`), DX12 (mesh tier 1), and Metal
  (`MTLMeshRenderPipelineDescriptor` / `drawMeshThreadgroups`). Record with
  `SchemeRenderPassBuilder::set_mesh_pipeline` and `dispatch_mesh`. Skip when
  `DeviceCapabilities::mesh_shaders` is false. WebGPU / CUDA are not wired.

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
  `DeviceCapabilities::ray_tracing_pipelines` is false. Metal exposes
  `ray_query` only (no SBT / `TraceRays`).

- **Graph validation (`GoldyError::Validation`)** — `Scheme::submit` always checks
  dependency cycles, mesh vs vertex command mix-ups, BLAS/TLAS misuse, and
  `BufferFlags::ACCEL_INPUT` on BLAS geometry, with `hint:` text. `GOLDY_VALIDATION=scheme`
  (alias `graph`) also requires Accel builds in the same scheme before RayQuery / TraceRays.
  Examples: `mesh_triangle`, `ray_query`.

- **CPU dispatches in `Scheme`** — `Scheme::cpu_node(label)` records a serial,
  stateless host function as a scheme node. The function's parameter list is
  its virtual main: one `&[T]` / `&mut [T]` (`T: bytemuck::Pod`) per parcel
  bound with `with_parcel` / `with_lease`, followed by `u32` / `i32` / `f32` /
  `bool` scalars from `with_param`. Host visibility is a property of the node:
  every binding is staged through a device→host readback copy before the call
  and a host→device upload copy after it (`Overwrite` skips the download and
  arrives zeroed), so parcels keep their device-resident allocation on every
  backend. A CPU dispatch is a full pipeline drain and never retains; GPU
  partitions around it retain as before. New `goldy::cpu_dispatch` module
  (`CpuArg`, `CpuMain`), `SchemeCpuNodeBuilder`, and
  `NodeKind::CpuDispatch`. Textures are not accepted as CPU parameters yet.

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
