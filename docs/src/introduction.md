<p align="center">
  <img src="assets/logo.jpeg" alt="Goldy Logo" width="600">
</p>

# Goldy: Modern GPU Library

**Goldy** is a modern Rust GPU library that deliberately sheds legacy baggage. It targets only modern GPU APIs (Vulkan 1.4+, DX12, Metal) and can therefore be significantly simpler than libraries that must maintain backward compatibility.

```rust
use goldy::{Instance, DeviceType, Color, CommandEncoder, Surface};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    // Create instance and device
    let instance = Instance::new()?;
    let device = Arc::new(instance.create_device(DeviceType::DiscreteGpu)?);
    
    // Create surface for zero-copy window presentation
    let surface = Surface::new(&device, &window)?;
    
    // Acquire frame and render
    let frame = surface.acquire()?;
    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::CORNFLOWER_BLUE);
    }
    
    frame.render(encoder)?;
    surface.present(frame)?;
    Ok(())
}
```

## Key Features

| Attribute | Description |
|-----------|-------------|
| **Rust-native** | Idiomatic Rust API, not a wrapper around C APIs |
| **Modern-only** | Assumes Vulkan 1.4+, DX12, Metal baseline |
| **Legacy-free** | No OpenGL, no Vulkan <1.4, no OpenCL baggage |
| **Unified** | Graphics and compute in one API |
| **Fast-moving** | Not a standard—can iterate quickly |

## What Goldy Is Not

- **Not a Vulkan wrapper**: Each backend is native (Metal uses Metal idioms, not translated Vulkan)
- **Not a WebGPU implementation**: Not bound by WebGPU spec committee
- **Not targeting web browsers**: Native GPU APIs only
- **Not a compatibility layer**: Won't emulate missing features
- **Not a standard**: Can break things, move fast, be opinionated

## Quick Links

- [Getting Started](./getting-started/installation.md)
- [Examples](./examples/overview.md)
- [Design Philosophy](./design/motivation.md)
- [GitHub Repository](https://github.com/koubaa/goldy)

## License

Goldy is dual-licensed under **LGPL-2.1-or-later** and a **commercial license**. See [License](./license.md) for details.
