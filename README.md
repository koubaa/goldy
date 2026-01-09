# RAG - Rust Abstract GPU

A modern GPU abstraction library for Rust that targets only modern graphics APIs.

## Philosophy

RAG deliberately sheds legacy baggage:

- **Modern-only**: Requires Vulkan 1.4+, Metal 2+, or DX12
- **No OpenGL**: No compatibility layer for legacy APIs
- **No Vulkan <1.4**: Assumes dynamic rendering, bindless descriptors
- **Fast-moving**: Not a standard, can iterate quickly

## Features

- Device enumeration and selection
- Buffer creation with automatic memory allocation
- WGSL shader compilation to native (SPIR-V, MSL, DXIL)
- Render pipeline with dynamic rendering
- Offscreen rendering with CPU readback

## Quick Start

```rust
use rag::{Instance, DeviceType};

fn main() -> anyhow::Result<()> {
    // Create instance and enumerate GPUs
    let instance = Instance::new()?;
    
    for adapter in instance.enumerate_adapters() {
        println!("{}: {} ({:?})", 
            adapter.id, 
            adapter.name, 
            adapter.device_type
        );
    }
    
    // Create device on discrete GPU
    let device = instance.create_device(DeviceType::DiscreteGpu)?;
    
    // ... render something ...
    
    Ok(())
}
```

## Examples

```bash
# List GPUs and clear to blue
cargo run --example clear

# Render a colored triangle
cargo run --example triangle

# Render a 7-segment clock display
cargo run --example clock
```

## Minimum Requirements

| Platform | Minimum API |
|----------|-------------|
| NVIDIA | Vulkan 1.3+ (RTX 2000 / GTX 1600) |
| AMD | Vulkan 1.3+ (RDNA 1 / RX 5000) |
| Intel | Vulkan 1.3+ (Xe / Arc) |
| Apple | Metal 2+ (M1 / A14) |

## License

MIT License - see [LICENSE](LICENSE) for details.

## LLM disclosure

Development was done with the help of AI tools.

