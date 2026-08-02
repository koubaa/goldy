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
`Rgba32Float` shared scratch texture (CUDA writes via `DirectSpatial<float4>`);
present blits to the BGRA8 swapchain on the DX12 DIRECT queue using a shared
D3D12 fence imported as a CUDA external semaphore. This is zero-copy between CUDA
and DX12; a GPU blit into the non-shareable swapchain image remains. Adapter
mismatch, WARP, and linked-node adapters fail at device creation. A first-slice
raster path is also available under the same feature gate: offscreen
`Rgba32Float` render targets, non-indexed point/line/triangle pipelines (Slang → DXIL), and
`CopyRenderTarget` into present scratch / CUDA textures. Depth, indexed draws,
and bindless render bindings are not in this slice. Vulkan interop is not
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
- Writable shader access (`DirectSpatial<T>`) initially requires a storage-compatible
  format: `DirectSpatial<float4>` ↔ `Rgba32Float` only. Upload/copy/readback of other
  supported formats still works.
- CUDA has no separate sampler object — filtering is baked into each `CUtexObject`.
  A dispatch may use at most one distinct `Filter` configuration; additional distinct
  samplers are rejected.

Retainable kernel-only partitions are captured into CUDA graphs on first submit and
relaunched on clean resubmits. Indirect dispatches in those partitions use CUDA 13.1
device-updatable kernel nodes: an in-graph updater reads the GPU-resident
`DispatchShape` and updates the consumer node's grid (or disables it for a zero /
oversized shape). Uploads, clears, and copies (including texture copies) stay on the
stream command-replay path, where indirect grids are resolved with a worker-side DtoH
before `cuLaunchKernel`. Dynamic waits and completion events remain outside the
captured graph. Stream capture is skipped when `CUDA_LAUNCH_BLOCKING` is set
(including under `GOLDY_VALIDATION=api`).

With `GOLDY_VALIDATION=api` (or `all`), the CUDA backend enables Driver diagnostics: PTX JIT error/info logs on module load, host-side launch-limit checks, StructuredBuffer ABI checks, and per-op stream synchronize with labeled errors. It may set `CUDA_LAUNCH_BLOCKING=1` when unset. Deep memory/race checking still requires external [`compute-sanitizer`](https://docs.nvidia.com/compute-sanitizer/), not `GOLDY_VALIDATION`.

### WebGPU Backend (in progress)

Cross-platform prototype built on [wgpu](https://github.com/gfx-rs/wgpu). Intended for broader portability and browser-adjacent targets. Enable with the `webgpu` Cargo feature. Not yet at parity with the shipped Vulkan/DX12/Metal backends.

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
