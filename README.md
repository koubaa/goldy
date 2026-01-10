# RAG: Rust Abstract GPU

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern Rust GPU library that deliberately sheds legacy baggage. RAG targets only modern GPU APIs (Vulkan 1.4+, Metal 2+, DX12) and can therefore be significantly simpler than libraries that must maintain backward compatibility.

## Quick Example

```rust
use rag::{Instance, DeviceType, Buffer, BufferUsage, Color, CommandEncoder, FrameOutput, TextureFormat};

fn main() -> anyhow::Result<()> {
    let instance = Instance::new()?;
    let device = instance.create_device(DeviceType::DiscreteGpu)?;
    
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

## Features

| Attribute | Description |
|-----------|-------------|
| **Rust-native** | Idiomatic Rust API, not a wrapper around C |
| **Modern-only** | Vulkan 1.4+, Metal 2+, DX12 baseline |
| **Slang shaders** | Single shader language for all backends |
| **Legacy-free** | No OpenGL, no Vulkan <1.4, no OpenCL |
| **Unified** | Graphics and compute in one API |
| **Fast-moving** | Not a standard—can iterate quickly |
| **WASI-ready** | Designed for sandboxed GPU access |

## Installation

```toml
[dependencies]
rag = "0.1"
```

## Documentation

📖 **[Full Documentation](https://koubaa.github.io/rag/)**

- [Getting Started](https://koubaa.github.io/rag/getting-started/installation.html)
- [Examples](https://koubaa.github.io/rag/examples/overview.html)
- [API Reference](https://koubaa.github.io/rag/reference/api.html)
- [Design Philosophy](https://koubaa.github.io/rag/design/motivation.html)

## Examples

Run the interactive examples:

```bash
cargo run --example triangle --release      # Basic triangle
cargo run --example digital_clock --release # 7-segment clock
cargo run --example plasma --release        # Demoscene plasma
cargo run --example mandelbrot --release    # Fractal explorer
cargo run --example starfield --release     # 3D starfield
cargo run --example particles --release     # Rain/snow
```

See [all 13 examples](https://koubaa.github.io/rag/examples/overview.html).

## Motivation

RAG is inspired by Sebastian Aaltonen's ["No Graphics API"](https://www.sebastianaaltonen.com/blog/no-graphics-api) vision of what's possible with modern GPU hardware. By targeting only modern GPUs (2018+), RAG can:

- Use dynamic rendering (no render pass objects)
- Use bindless descriptors (no descriptor sets)
- Assume coherent caches (simpler synchronization)
- Provide a dramatically simpler API

Read more in [Design Philosophy](https://koubaa.github.io/rag/design/motivation.html).

## Target Hardware

| Platform | Minimum |
|----------|---------|
| NVIDIA | RTX 2000 / GTX 1600 (2018+) |
| AMD | RDNA 1 / RX 5000 (2019+) |
| Intel | Xe / Alchemist (2022+) |
| Apple | M1 / A14 (2020+) |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Mohamed Koubaa

## Contributing

Contributions welcome! See [CONTRIBUTING](https://koubaa.github.io/rag/contributing.html).
