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
    fn create_buffer(&mut self, device: DeviceId, size: u64, usage: BufferUsage) -> Result<BufferId>;
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

You can override the default backend at runtime using the `GOLDY_BACKEND` environment variable:

```bash
# Use Vulkan on Windows (instead of DX12)
GOLDY_BACKEND=vulkan cargo run --example triangle

# Use DX12 explicitly
GOLDY_BACKEND=dx12 cargo run --example triangle

# Valid values: vulkan (or vk), dx12 (or d3d12), metal (or mtl)
```

This is useful for:
- Testing your app on different backends
- Working around driver bugs
- Debugging backend-specific issues

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
    
    fn create_buffer(&mut self, device: DeviceId, size: u64, usage: BufferUsage) -> Result<BufferId> {
        let options = usage_to_metal_options(usage);
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
