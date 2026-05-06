# Backend Architecture

Goldy uses a backend abstraction that allows different GPU APIs while maintaining a unified interface.

## Current Backends

| Backend | Status | Platforms |
|---------|--------|-----------|
| Vulkan |  Implemented | Windows, Linux |
| DX12 |  Implemented | Windows |
| Metal | Implemented | macOS, iOS |

## Backend Independence

Each backend uses **native idioms**, not translation:

```
┌─────────────────────────────────────────────────────────────┐
│                    Goldy Core API                              │
│                                                              │
│   Device, Buffer, Texture, Pipeline, CommandEncoder, ...    │
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

## Vulkan Backend

The Vulkan backend uses modern Vulkan 1.4+ features:

### Dynamic Rendering

No render pass objects:

```rust
// Traditional Vulkan requires:
// 1. VkRenderPass
// 2. VkFramebuffer

// Goldy uses dynamic rendering (VK_KHR_dynamic_rendering):
// Just specify attachments at draw time
```

### Descriptor Indexing

Bindless resource access:

```rust
// Traditional Vulkan:
// 1. Create descriptor set layout
// 2. Create descriptor pool
// 3. Allocate descriptor sets
// 4. Update descriptor sets
// 5. Bind descriptor sets

// Goldy uses descriptor indexing:
// Access resources by index in shader
```

### Buffer Device Address

Direct memory access in shaders:

```rust
// Traditional: buffer bindings
// Goldy: 64-bit pointers
```

## DX12 Backend

The DX12 backend provides native Windows support using the `windows` crate:

- **Root signatures** for resource binding
- **Descriptor heaps** for efficient resource management
- **Shader compilation** via Slang → DXIL

## Backend Trait

```rust
pub trait GpuBackend: Send + Sync {
    // Instance/device management
    fn backend_type(&self) -> BackendType;
    fn enumerate_adapters(&self) -> Vec<AdapterDesc>;
    fn create_device(&mut self, adapter_idx: u32) -> Result<DeviceId>;
    
    // Resources
    fn create_buffer(&mut self, device: DeviceId, size: u64, access: DataAccess) -> Result<BufferId>;
    fn write_buffer(&mut self, buffer: BufferId, data: &[u8]) -> Result<()>;
    fn destroy_buffer(&mut self, buffer: BufferId);
    
    // Shaders
    fn create_shader(&mut self, device: DeviceId, spirv: &[u32]) -> Result<ShaderId>;
    fn destroy_shader(&mut self, shader: ShaderId);
    
    // Pipelines
    fn create_pipeline(&mut self, desc: &PipelineDesc) -> Result<PipelineId>;
    fn destroy_pipeline(&mut self, pipeline: PipelineId);
    
    // Rendering
    fn begin_frame(&mut self, device: DeviceId, width: u32, height: u32) -> Result<FrameId>;
    fn execute_commands(&mut self, frame: FrameId, commands: &[RenderCommand]) -> Result<()>;
    fn end_frame(&mut self, frame: FrameId) -> Result<Vec<u8>>;
}
```

## Resource IDs

Resources are identified by opaque IDs:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u32);

// etc.
```

The backend maps these to native handles internally.

## Backend Selection

### Default Selection

By default, Goldy selects the platform-preferred backend:

| Platform | Default Backend |
|----------|-----------------|
| Windows  | DX12            |
| Linux    | Vulkan          |
| macOS    | Metal           |

### Runtime Override

Override the backend at runtime with `GOLDY_BACKEND`:

```bash
# Valid values: vulkan (or vk), dx12 (or d3d12, directx), metal (or mtl)
GOLDY_BACKEND=vulkan cargo run --example triangle
GOLDY_BACKEND=dx12   cargo run --example triangle
```

### Cross-backend validation

| Variable | Values | Effect |
|----------|--------|--------|
| `GOLDY_VALIDATE_LAYOUTS` | `1`, `true`, `yes` | Rust vs Slang struct layout checks and dispatch-time buffer stride checks (unchanged). |
| `GOLDY_VALIDATION` | `1`, `true`, `yes` | **GPU API only:** Vulkan Khronos validation + `VK_EXT_debug_utils`; Metal sets `MTL_SHADER_VALIDATION=1` if unset. |
| `GOLDY_VALIDATION` | `layout`, `layouts` | Layout / stride family only. |
| `GOLDY_VALIDATION` | `api` | Graphics API validation (same as `1`; combine with `layout` using commas, semicolons, or spaces). |
| `GOLDY_VALIDATION` | `all` | Layout + GPU API (same as `layout,api`). |

Vulkan also enables the same instance path when **`VK_INSTANCE_LAYERS`** contains `VK_LAYER_KHRONOS_validation` (loader-driven workflow). For timing notes and more examples, see **[DEBUGGING.md](https://github.com/koubaa/goldy/blob/main/DEBUGGING.md)** in the repository.

### DX12: Additional Environment Variables

| Variable | Values | Purpose |
|---|---|---|
| `GOLDY_DX12_FORCE_WARP` | `1` | Use the DX12 WARP software rasterizer (see below) |
| `GOLDY_DX12_NO_DEBUG` | `1` | Disable the D3D12 debug layer (always off in release; set in parallel tests to avoid debug-layer crashes) |
| `GOLDY_DX12_DEBUG` | `1` | Force-enable the D3D12 debug layer even in release builds |
| `GOLDY_DX12_GBV` | `1` | Enable GPU-Based Validation (very slow, requires debug layer) |

### DX12 WARP — Software Rasterizer

[WARP](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/directx-warp) is
Microsoft's software implementation of D3D12. It is used on headless CI runners (no GPU)
and to reproduce GPU-driver or WARP-specific rendering bugs locally.

**Set `GOLDY_DX12_FORCE_WARP=1`** to run on WARP regardless of what hardware is present.
On Windows, DX12 is the default backend, so this is the only variable you need:

```bash
GOLDY_DX12_FORCE_WARP=1 cargo nextest run ...
```

After the first WARP device is created Goldy prints one stderr line:

```
[WARP] d3d10warp.dll loaded from: C:\WINDOWS\SYSTEM32\d3d10warp.dll
```


### Compile-Time Selection

You can also select backends at compile time using Cargo features. This excludes both the code and dependencies for unselected backends:

```bash
# Build with only Vulkan backend (excludes DX12 code and dependencies)
cargo build --no-default-features --features vulkan
```

For detailed information on feature flags, dependency exclusion, and CI setup, see [Conditional Compilation](conditional-compilation.md).

### Programmatic Selection

The backend can be queried at runtime:

```rust
let instance = Instance::new()?;
println!("Backend: {:?}", instance.backend_type());
// Prints: Backend: Dx12  (on Windows)
// Prints: Backend: Vulkan  (on Linux)
```

## Adding a New Backend

To add a backend (e.g., Metal):

1. Implement `GpuBackend` trait
2. Use native API (metal-rs)
3. Map Goldy concepts to native equivalents

```rust
pub struct MetalBackend {
    device: metal::Device,
    command_queue: metal::CommandQueue,
    // ...
}

impl GpuBackend for MetalBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Metal
    }
    
    fn create_buffer(&mut self, device: DeviceId, size: u64, access: DataAccess) -> Result<BufferId> {
        let options = access_to_metal_options(access);
        let buffer = self.device.new_buffer(size, options);
        // Store and return ID
    }
    
    // ... other methods
}
```

## Why Native Backends?

Translation layers (like MoltenVK) have overhead:

1. **API mismatch** - Vulkan concepts don't map 1:1 to Metal
2. **Synchronization** - Different hazard tracking models
3. **Descriptors** - Different binding models
4. **Validation** - Extra validation layer

Native backends can use each API's strengths directly.
