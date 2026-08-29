# Backend Architecture

Goldy ships three GPU backends today, each implemented natively against the platform
graphics API — no translation layers (like MoltenVK) are involved. Two additional
backends are in active development; a Tenstorrent backend is planned.

| Backend | Status | API Level | Platforms | Rust Crate |
|---------|--------|-----------|-----------|------------|
| Vulkan | Shipped | 1.4+ | Windows, Linux | `ash` |
| DX12 | Shipped | Direct3D 12 | Windows | `windows` + `gpu-allocator` |
| Metal | Shipped | Tier 2+ | macOS | `metal` |
| CUDA | In progress | CUDA Driver API | NVIDIA GPUs | `cudarc` |
| WebGPU | In progress | WebGPU (via wgpu) | Cross-platform | `wgpu` |
| Tenstorrent | Planned | TT-Metalium / TT-MLIR | Tenstorrent accelerators | — |

## Native Implementations

Each backend maps Goldy concepts directly to the most natural primitives of
its target API:

```
┌─────────────────────────────────────────────────────────────┐
│                    Goldy Core API                           │
│                                                             │
│   Device, Buffer, Texture, Pipeline, Scheme, ...          │
└─────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Vulkan 1.4+   │    │ Metal 2+      │    │ DX12          │
│               │    │               │    │               │
│ • ash crate   │    │ • metal-rs    │    │ • windows-rs  │
│ • Dynamic     │    │ • Argument    │    │ • Root        │
│   rendering   │    │   buffers     │    │   signatures  │
│ • Descriptor  │    │ • Native      │    │ • Descriptor  │
│   indexing    │    │   hazard      │    │   heaps       │
│ • Buffer      │    │   tracking    │    │               │
│   device addr │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
```

Translation layers introduce overhead from API mismatches, incompatible
synchronization models, and extra validation. Native backends can leverage
each API's strengths directly — for example, Metal's built-in hazard
tracking, or Vulkan's descriptor indexing for bindless rendering.

## Backend Selection

### Default Selection

Goldy selects the platform-preferred backend automatically:

| Platform | Default Backend |
|----------|-----------------|
| macOS | Metal |
| Windows | DX12 |
| Linux | Vulkan |

### Runtime Override — `GOLDY_BACKEND`

Override the backend at runtime with the `GOLDY_BACKEND` environment variable:

```bash
GOLDY_BACKEND=vulkan cargo run --example triangle
GOLDY_BACKEND=dx12   cargo run --example triangle
```

Accepted values (case-insensitive):

| Value | Backend | Status |
|-------|---------|--------|
| `vulkan`, `vk` | Vulkan | Shipped |
| `dx12`, `d3d12`, `directx` | DX12 | Shipped |
| `metal`, `mtl` | Metal | Shipped |
| `cuda` | CUDA | In progress |
| `webgpu`, `wgpu` | WebGPU | In progress |

An unrecognized value produces a clear error listing the valid options.

### Programmatic Selection

Query the active backend at runtime:

```rust
let instance = Instance::new()?;
println!("Backend: {:?}", instance.backend_type());
// Prints: Backend: Dx12   (on Windows)
// Prints: Backend: Vulkan (on Linux)
// Prints: Backend: Metal  (on macOS)
```

### Compile-Time Selection (Feature Flags)

You can also restrict which backends are compiled in via Cargo features.
This excludes both the code *and* the dependencies of unselected backends:

```bash
cargo build --no-default-features --features vulkan
```

See [Conditional Compilation](conditional-compilation.md) for details on
feature flags, dependency exclusion, and CI setup.

## Adapter Enumeration

After creating an `Instance`, enumerate available GPU adapters to inspect
what hardware is present:

```rust
let instance = Instance::new()?;
let adapters = instance.enumerate_adapters();

for adapter in &adapters {
    println!("{}: {} ({})", adapter.id(), adapter.name(), adapter.vendor());
    println!("  Type: {:?}", adapter.device_type());
}
```

### `DeviceType`

Each adapter reports a `DeviceType`:

| Variant | Meaning |
|---------|---------|
| `DiscreteGpu` | Dedicated graphics card with its own VRAM |
| `IntegratedGpu` | GPU integrated into the CPU (shared memory) |
| `Cpu` | Software renderer (e.g. WARP on DX12, lavapipe on Vulkan) |
| `Other` | Unknown or unrecognized device class |

### Creating a Device

Request a device with a preferred `DeviceType`. If no adapter matches,
Goldy falls back to the first available adapter:

```rust
let device = instance
    .request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        ..Default::default()
    })?
    .request_device(&DeviceDescriptor::default())?;

// Or target a specific adapter by ID:
let device = instance.create_device_for_adapter(adapter.id())?;
```

## Backend Capabilities

### Device Capabilities

Query format preferences and backend-specific capabilities after creating
a device:

```rust
let caps = device.capabilities();

println!("Surface format:     {:?}", caps.preferred_surface_format);
println!("Render target fmt:  {:?}", caps.preferred_render_target_format);
println!("Zero-copy readback: {}", caps.has_zero_copy_storage_readback);
```

| Capability | Vulkan | DX12 | Metal |
|------------|--------|------|-------|
| Zero-copy CPU storage readback | Yes | No (requires GPU copy to readback heap) | Yes |
| Preferred surface format | `Bgra8UnormSrgb` | `Bgra8UnormSrgb` | `Bgra8UnormSrgb` |

### Vulkan Backend

The Vulkan backend requires Vulkan 1.4+ and uses:

- **Dynamic rendering** (`VK_KHR_dynamic_rendering`) — no `VkRenderPass` or `VkFramebuffer` objects
- **Descriptor indexing** — bindless resource access by index in shaders
- **Buffer device address** — 64-bit GPU pointers for direct memory access in shaders

### DX12 Backend

The DX12 backend uses the `windows` crate and provides:

- **Root signatures** for resource binding
- **Descriptor heaps** for efficient bindless resource management
- **Shader compilation** via Slang to DXIL
- **WARP** software rasterizer for headless/CI use (`GOLDY_DX12_FORCE_WARP=1`)
- **GPU-Based Validation** for deep debugging (`GOLDY_DX12_GBV=1`)

### Metal Backend

The Metal backend uses the `metal` crate (native Metal, not MoltenVK):

- **Argument buffers** for bindless resource binding
- **Native hazard tracking** — Metal tracks resource hazards automatically
- Shader compilation via Slang to Metal Shading Language

### CUDA Backend (in progress)

Compute-focused prototype targeting NVIDIA GPUs via the CUDA Driver API (**CUDA 13.1+**
required for device-updatable graph nodes). Slang compiles to PTX; dispatches use the
CUDA launch model. The `cuda` feature does not imply `graphics`. Buffer schemes,
uploads/readbacks, timelines, **indirect dispatch**, and **2D textures/samplers**
(CUDA arrays + texture/surface objects) work.

**Windows presentation:** when `cuda`, `graphics`, and `dx12` are all enabled, each
CUDA device opens a LUID-matched DX12 companion. Surface frames expose an
`Rgba8Unorm` shared scratch texture from a **depth-3 staging ring** independent of
the DXGI swapchain image. Typical schemes (including Ekrano) write a CUDA-owned
staging texture then `CopyTexture` into that imported scratch — the same
local-then-copy pattern as native DX12 — before present's same-format `CopyResource`
onto the `R8G8B8A8_UNORM` DXGI swapchain. CUDA signals a **ready** fence; DX12 waits
it, copies, then signals a **recycle** fence. CUDA waits recycle only when wrapping
the ring, so compute N+1 does not serialize behind present-copy N. Adapter mismatch,
WARP, and linked-node adapters fail at device creation. A first-slice raster path is
also available under the same feature gate: offscreen `Rgba32Float` and `Rgba8Unorm` render targets, indexed and non-indexed
point/line/triangle pipelines (Slang → DXIL), bindless render bindings, optional
DX12-only depth attachments / depth-stencil PSOs / `ClearDepth`, and
`CopyRenderTarget` into present scratch / CUDA textures. Depth is not CUDA-imported
(compute cannot sample it yet); stencil ops remain off. Vulkan interop is not
supported.

Enable with the `cuda` Cargo feature (`--no-default-features --features cuda`
auto-selects CUDA; in default builds use `GOLDY_BACKEND=cuda`):

```bash
cargo test --no-default-features --features cuda --test scheme_compute_integration
# Windows presentation:
GOLDY_BACKEND=cuda cargo run --example compute_to_surface --features examples
```

Texture notes for CUDA:

- Sampled formats: `R8Unorm`, `Rg8Unorm`, `Rgba8Unorm`, `Rgba8UnormSrgb`,
  `Rgba16Float`, `Rgba32Float`. **BGRA is rejected** (no matching CUDA array swizzle).
- Writable shader access (`DirectSpatial<T>`) supports size-matched pairs and Goldy’s
  typed-UAV emulation for convertible pairs:
  - `DirectSpatial<float4>` ↔ `Rgba32Float` (identity surface store)
  - `DirectSpatial<float4>` ↔ `Rgba8Unorm` (lazy PTX specialization: pack/unpack view over
    `uint8_t4`, DX12-style `round(saturate(x)*255)` on store). Partitions that launch this
    specialized variant stay on stream-replay segments between CUDA graph islands
    (or use full op-list retention when no graph-safe island remains).
  - `DirectSpatial<half4>` ↔ `Rgba16Float`
  - `DirectSpatial<uint8_t4>` ↔ `Rgba8Unorm` (Slang has no `uchar4` alias)
  - Upload/copy/readback of other supported sampled formats still works.
- Surfaces expose `Rgba8Unorm` imported scratch from a depth-3 ring (CUDA+DX12
  interop tradeoff: extra staging textures so compute does not reuse frame N's
  scratch in N+1). Prefer writing a CUDA-owned
  `Rgba8Unorm` texture (or render target) and exporting with `CopyTexture`; direct
  launches onto imported scratch remain supported but are costlier under WDDM.
  The DXGI swapchain is matching `R8G8B8A8_UNORM` so present is a single
  `CopyResource`.
- `DeviceCapabilities` on CUDA advertise `preferred_surface_format = Rgba8Unorm` and
  `preferred_render_target_format = Rgba8Unorm` (no BGRA in supported lists).
- CUDA has no separate sampler object — filtering is baked into each `CUtexObject`.
  A dispatch may use at most one distinct `Filter` configuration; additional distinct
  samplers are rejected.

Retainable partitions are split into alternating CUDA graph islands (contiguous
graph-safe kernel launches) and stream-replayed boundary segments (clears, copies,
format-specialized launches, present exports). Islands are captured on first submit
and relaunched on clean resubmits; stream segments re-execute between them on the
same CUDA stream. Indirect dispatches in graph islands use CUDA 13.1
device-updatable kernel nodes: an in-graph updater reads the GPU-resident
`DispatchShape` and updates the consumer node's grid (or disables it for a zero /
oversized shape). Uploads and other fully graph-unsafe partitions stay on the
stream command-replay path, where indirect grids are resolved with a worker-side DtoH
before `cuLaunchKernel`. Dynamic waits and completion events remain outside the
captured graph. Stream capture is skipped when `CUDA_LAUNCH_BLOCKING` is set
(including under `GOLDY_VALIDATION=api`).

With `GOLDY_VALIDATION=api` (or `all`), the CUDA backend enables Driver diagnostics: PTX JIT error/info logs on module load, host-side launch-limit checks, StructuredBuffer ABI checks, and per-op stream synchronize with labeled errors. It may set `CUDA_LAUNCH_BLOCKING=1` when unset. Deep memory/race checking still requires external [`compute-sanitizer`](https://docs.nvidia.com/compute-sanitizer/), not `GOLDY_VALIDATION`.

### WebGPU Backend (in progress)

Cross-platform prototype built on [wgpu](https://github.com/gfx-rs/wgpu). Intended for broader
portability and browser-adjacent targets. Enable with the `webgpu` Cargo feature. Not yet at
parity with the shipped Vulkan/DX12/Metal backends.

Compute buffers, scalar uniforms, **indirect dispatch**, and **2D textures/samplers** work.
Submit is **non-blocking**: the context timeline advances from wgpu's
`on_submitted_work_done` callback (pumped by `Device::poll`). Host waits
(`Context::wait_until`, withdraw) block on the submission index, not on submit
itself. Resources bind as a single `@group(0)` in shader-parameter order (no bindless heap). Texture
notes:

- Sampled formats: `R8Unorm`, `Rg8Unorm`, `Rgba8Unorm`, `Rgba8UnormSrgb`, `Bgra8Unorm`,
  `Bgra8UnormSrgb`, `Rgba16Float`, `Rgba32Float` (subject to adapter format features).
- `DirectSpatial<T>` storage textures follow WGSL: the shader type encodes the format.
  `DirectSpatial<float4>` therefore pairs with `Rgba32Float` (unlike native backends, which
  often write `float4` into `Rgba8Unorm` UAVs). sRGB and BGRA are rejected for storage.
- Uploads use `queue.write_texture`. Texture withdraw staging uses WebGPU's 256-byte row
  pitch; `query_texture_copy_footprint` reports the padded layout.
- Surfaces (`graphics`): create/configure and `begin_frame` acquire a wgpu drawable
  and return an `Rgba8Unorm` storage scratch (swapchain images cannot be UAVs).
  Present copies scratch → swapchain on the same queue, then `SurfaceTexture::present`.
  The copy waits before returning so the single scratch is not overwritten in flight.
  Graphics pipelines and render targets remain unsupported.
- Compute sampling must use `SampleLevel` (WGSL has no implicit derivatives in compute).

```bash
cargo test --no-default-features --features webgpu --lib backend::webgpu
```

### Tenstorrent Backend (planned)

**Torus** is a planned Fondaco runtime for Tenstorrent Tensix hardware. No implementation ships in Goldy today.

## The `GpuBackend` Trait

All backends implement the `GpuBackend` trait, which defines the full
interface for device management, resource creation, shader compilation,
pipeline management, rendering, and compute dispatch:

```rust
pub trait GpuBackend: Send + Sync {
    fn backend_type(&self) -> BackendType;
    fn enumerate_adapters(&self) -> Vec<AdapterInfo>;
    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle>;
    fn create_buffer(&mut self, device: DeviceHandle, ...) -> Result<BufferHandle>;
    fn create_shader_with_paths(&mut self, device: DeviceHandle, ...) -> Result<ShaderHandle>;
    fn create_pipeline(&mut self, device: DeviceHandle, ...) -> Result<PipelineHandle>;
    // ... rendering, compute, surface, texture, sampler, timeline ...
}
```

Resources are identified by opaque `u64` handles (`DeviceHandle`,
`BufferHandle`, `ShaderHandle`, etc.) that each backend maps to native
API objects internally.
