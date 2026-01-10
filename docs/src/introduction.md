# RAG: Rust Abstract GPU

**RAG** is a modern Rust GPU library that deliberately sheds legacy baggage. It targets only modern GPU APIs (Vulkan 1.4+, Metal 2+, DX12) and can therefore be significantly simpler than libraries that must maintain backward compatibility.

```rust
use rag::{Instance, DeviceType, Color, CommandEncoder, FrameOutput};

fn main() -> anyhow::Result<()> {
    // Create instance and device
    let instance = Instance::new()?;
    let device = instance.create_device(DeviceType::DiscreteGpu)?;
    
    // Create a frame and clear it
    let frame = FrameOutput::new(&device, 800, 600, TextureFormat::Rgba8Unorm);
    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::CORNFLOWER_BLUE);
    }
    
    let pixels = frame.render(encoder)?;
    Ok(())
}
```

## Key Features

| Attribute | Description |
|-----------|-------------|
| **Rust-native** | Idiomatic Rust API, not a wrapper around C APIs |
| **Modern-only** | Assumes Vulkan 1.4+, Metal 2+, DX12 baseline |
| **Legacy-free** | No OpenGL, no Vulkan <1.4, no OpenCL baggage |
| **Unified** | Graphics and compute in one API |
| **Fast-moving** | Not a standard—can iterate quickly |
| **WASI-ready** | Can be exposed to WASM guests via WIT |

## What RAG Is Not

- **Not a Vulkan wrapper**: Each backend is native (Metal uses Metal idioms, not translated Vulkan)
- **Not a WebGPU implementation**: Not bound by WebGPU spec committee
- **Not a compatibility layer**: Won't emulate missing features
- **Not a standard**: Can break things, move fast, be opinionated

## Quick Links

- [Getting Started](./getting-started/installation.md)
- [Examples](./examples/overview.md)
- [Design Philosophy](./design/motivation.md)
- [GitHub Repository](https://github.com/koubaa/rag)

## License

RAG is MIT licensed. See [License](./license.md) for details.

