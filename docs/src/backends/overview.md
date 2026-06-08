# Backend Architecture

Goldy supports three GPU backends, each implemented natively against the platform
graphics API — no translation layers (like MoltenVK) are involved.

| Backend | API Level | Platforms | Rust Crate |
|---------|-----------|-----------|------------|
| Vulkan | 1.4+ | Windows, Linux | `ash` |
| DX12 | Direct3D 12 | Windows | `windows` + `gpu-allocator` |
| Metal | Tier 2+ | macOS, iOS | `metal` |

## Native Implementations

Each backend maps Goldy concepts directly to the most natural primitives of
its target API:

```
┌─────────────────────────────────────────────────────────────┐
│                    Goldy Core API                           │
│                                                             │
│   Device, Buffer, Texture, Pipeline, TaskGraph, ...       │
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

| Value | Backend |
|-------|---------|
| `vulkan`, `vk` | Vulkan |
| `dx12`, `d3d12`, `directx` | DX12 |
| `metal`, `mtl` | Metal |

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
let device = instance.create_device(DeviceType::DiscreteGpu)?;

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
