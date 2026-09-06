# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Static shader validation** (`GOLDY_SHADER_VALIDATION`) — opt-in static
  checks over Slang's front-end IR, run at shader compile time. A separate
  variable from `GOLDY_VALIDATION` and *not* implied by `GOLDY_VALIDATION=all`:
  these cost a second compile plus a whole-program analysis per shader and
  report "not proven" rather than invariant violations. Value grammar:
  `all`, a check name (`bounds`), `-name` to exclude (`all,-bounds`). The
  reader and rule-agnostic pieces (RIFF/fossil container, instruction tree,
  linking, debug info, CFG/dominators) live in `goldy::slang::ir`; each check
  is a rule under `goldy::slang::shader_validation`. Public entry points:
  `SlangCompiler::validate_shader(.., ShaderChecks)` →
  `ShaderValidationReport`, and `shader_validation::validate` for
  `.slang-module` bytes produced elsewhere.
- **Bounds check** (`GOLDY_SHADER_VALIDATION=bounds`) — the first rule: an
  interval analysis over Slang's front-end IR that reports
  every dynamic index into a fixed-length array / vector / matrix it cannot
  prove to satisfy `0 <= index < length`, with the Slang source location, the
  call path from the entry point, and a note on what the index depends on
  (`SV_VertexID`, `WaveGetLaneCount()`, a buffer load, a float conversion,
  ...). Interprocedural: helpers, generics (per specialization, including
  `let N : int` array lengths), interface dispatch through witness tables and
  `out`/`inout` parameters are analyzed in their calling context, across
  `import`ed modules (`goldy_exp`). Understands dominating guards
  (`if (i >= 0)`, `&&` conjunctions), clamps (`max`/`min`/`&`/`%`), workgroup
  scan patterns, padded-dispatch early-outs and counted loops; flags the eager
  `select(cond, arr[i], x)` form. Runs regardless of the shader target and
  optimization level. Warnings only; never fails a compile. Report type:
  `BoundsReport` (`ShaderValidationReport::bounds`). See
  `docs/src/design/shader-bounds-analysis.md` for the integration decision
  (Slang IR vs. SPIR-V), the corpus evaluation over `shaders/`, and known
  limitations.

- **Examples in the book** — every `[[example]]` target now has a page under
  Examples in the mdBook, with its description, run command, controls, a
  recording of it running, and full Rust plus Slang source inlined from
  `examples/` and `shaders/` via mdBook includes. The gallery becomes the index
  and its two phantom entries (`headless_triangle`, `scheme_screenshot`) are
  gone.

- **`scripts/record_example_captures.sh`** — builds the examples, runs each one
  on a virtual X11 display, and grabs the window with ffmpeg into
  `docs/src/assets/examples/*.webm`. Defaults to the WebGPU backend, the only
  one whose surface path reaches X11 on Linux.

- **Yielding scripts** — a `[goldy_compute]` shader may suspend a lane with
  `$yield(continuation, payload, state)` and resume it in a `[goldy_resume]`
  function with the result of a host or GPU *handler*. Payload types are
  declared with `[goldy_petition(Result = BufRO<E>)]`; continuations receive a
  `Resolved<E>` window into a runtime-owned result arena (`is_null()` on
  rejection). On the host, `ComputePipeline::new` compiles the continuation
  entry points alongside the prologue, `SchemeNodeBuilder::yield_point(name,
  YieldPoint)` binds a handler per continuation (`YieldPoint::cpu` closure over
  `Petition` + `Promised<E>`, or `YieldPoint::node` compute dispatch),
  `Backpressure::{Stall, Drop}` picks the overflow policy, and
  `Scheme::yield_stats(node)` reports per-submission counters. The dispatch is
  recorded as a single host-driven node that runs the yield/service/resume
  rounds as sub-schemes on the same context (double-buffered mailboxes, so a
  continuation may yield to itself). See
  [Yielding Scripts](docs/src/programming-model/yielding-scripts.md).

- **`Interlocked*` on the CPU backend** — `goldy_exp`'s `InterlockedAdd` /
  `Or` / `Xor` / `Min` / `Max` / `Exchange` compile for the host-callable
  target as plain read-modify-write (the CPU backend runs lanes serially).

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

- **Shader specialization prediction** — retained schemes now specialize their
  own compute dispatches. Every dispatch node with `with_param` scalars gets a
  per-site predictor that counts, per slot, the clean submits during which the
  wire word held its value. After 2 such submits a variant with the stable slots
  baked (via the macros above) is compiled on a worker thread; after 10 it is
  bound on the node as a params-only re-record. `set_node_param` on a baked slot
  puts the node back on the caller's pipeline in the same call, so no submit
  ever runs a program whose baked value disagrees with the frame. A slot that
  invalidates a compile or a promotion needs a longer streak before it is baked
  again, so facts that flip every few frames stay dynamic while their neighbours
  specialize; three failed compiles pin a site to its universal pipeline.
  Variants live in a bounded per-scheme LRU and are reused across demotions.
  Output is byte-identical either way; the visible effects are
  `ReplayStats::specialization_warms` / `specialization_promotions` /
  `specialization_demotions` (new fields — exhaustive struct literals must add
  them), `Scheme::node_is_specialized`, and one extra record per promotion or
  demotion. On by default; `GOLDY_SPECIALIZATION=0` turns it off
  (`test_support::SpecializationOverride` pins it for tests). Backends whose
  compute pipeline layouts do not follow the shader signature decline through a
  new internal `GpuBackend::compute_pipeline_layout_follows_signature`: WebGPU
  (wgpu auto layouts drop bindings a baked variant stops reading) and CPU (no
  bake macros) return `false` and never see the predictor run. Design:
  [shader specialization prediction](docs/src/design/shader-specialization.md).

- **Shader provenance** — `ShaderModule` keeps its compile inputs (source,
  search paths, defines, optimization level, layout checks) in a shared
  `ShaderProvenance` with a process-unique id, and every `ComputePipeline`
  carries an `Arc` to its module's provenance, so the runtime can compile a
  variant of the program a dispatch runs after the caller has dropped the
  module. `ShaderModule::variant` is now a thin wrapper over it.

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

### Changed

- **Docs: what baking compiles** — the shader-specialization design note now
  states that predicted variants are a full Slang + driver recompile (not a
  constant patch), that Slang's default opt level leaves dead blocks in SPIR-V
  while the driver DCE's them, and when that is worth expecting. Scalar params,
  Slang defines, "no permutation systems", compute pipelines, and
  `GOLDY_SPECIALIZATION` point at the same distinction.

### Fixed

- **Slang diagnostics pointed one or two lines early in virtual-main shaders.**
  Stripping `[goldy_*]` / `[numthreads]` / `[outputtopology]` from the user
  function dropped their newlines, shifting every line after them relative to
  the `#line 1` directive. The removed spans now keep their newlines.

- **`mandelbrot` ignored the run limit** — the example sets
  `ControlFlow::Wait`, so with no input it idled past `GOLDY_EXAMPLE_TIMEOUT` /
  `EXAMPLE_TIMEOUT` forever and hung `run_all_examples.sh`. It now polls when a
  run limit is set.
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
