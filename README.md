<p align="center">
  <img src="assets/logo.jpeg" alt="Goldy Logo" width="600">
</p>

# Goldy: Modern GPU Library

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A modern Rust GPU library that deliberately sheds legacy baggage. Goldy targets only modern GPU APIs (Vulkan 1.4+, DX12, Metal Tier2+) and can therefore be significantly simpler than libraries that must maintain backward compatibility.

## Quick Example

```rust
use goldy::{Color, DeviceDescriptor, Instance, RequestAdapterOptions, Scheme, TargetLoad, TextureFormat};

fn main() -> anyhow::Result<()> {
    let instance = Instance::new()?;
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())?
        .request_device(&DeviceDescriptor::default())?;
    let ctx = device.create_context()?;

    let mut scheme = Scheme::new(&ctx);
    let rt = scheme.lease_render_target(800, 600, TextureFormat::Rgba8Unorm, None)?;
    let mut pass = scheme.render_pass("clear", &rt, TargetLoad::Clear(Color::CORNFLOWER_BLUE));
    pass.finish();
    scheme.submit()?;

    Ok(())
}
```

## Features

| Attribute | Description |
|-----------|-------------|
| **Rust-native** | Idiomatic Rust API, not a wrapper around C |
| **Modern-only** | Vulkan 1.4+, DX12, Metal Tier 2 |
| **Slang shaders** | Single shader language for all backends |
| **Unified** | Graphics and compute in one API |

## Installation

```toml
[dependencies]
goldy = "0.1"
```

### Slang Compiler

Goldy uses [Slang](https://github.com/shader-slang/slang) for shader compilation. The
Rust `build.rs` downloads (if needed) and **embeds** the pinned Slang version at
compile time; at runtime Goldy extracts and loads it automatically. Application
developers do not install Slang separately.

- Set `GOLDY_SLANG_PATH` only to override with a custom Slang build
- `slang/download.sh` is for **maintainers** bumping the pinned Slang version in
  `slang/manifest.json`, not for normal project setup

Release packaging for Python wheels and FFI redistributions is described in
[PACKAGING.md](PACKAGING.md). See [DEBUGGING.md](DEBUGGING.md) if shader
compilation fails at runtime.

Optional **Rust vs Slang struct layout checks** at shader compile time: set `GOLDY_VALIDATE_LAYOUTS=1` and pass `LayoutCheck` data from `#[derive(LayoutCheckable)]` into `ShaderModule::from_slang_with_options` (see [DEBUGGING.md](DEBUGGING.md) and the `gradient` / `checkerboard` examples).

## Documentation

📖 **[Full Documentation](https://koubaa.github.io/goldy/)**

- [Getting Started](https://koubaa.github.io/goldy/getting-started/installation.html)
- [Examples](https://koubaa.github.io/goldy/examples/overview.html)
- [API Reference](https://koubaa.github.io/goldy/reference/api.html)
- [Design Philosophy](https://koubaa.github.io/goldy/design/motivation.html)

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

### Selecting a Backend

By default, Goldy uses DX12 on Windows and Vulkan on Linux. Override with `GOLDY_BACKEND`:

```bash
# Run with Vulkan backend (on Windows)
GOLDY_BACKEND=vulkan cargo run --example triangle --release

# Run with DX12 backend
GOLDY_BACKEND=dx12 cargo run --example triangle --release
```

See [all examples](https://koubaa.github.io/goldy/examples/overview.html).

## Motivation

Goldy is inspired by Sebastian Aaltonen's ["No Graphics API"](https://www.sebastianaaltonen.com/blog/no-graphics-api) vision of what's possible with modern GPU hardware. By targeting only modern GPUs (2018+), Goldy can:

- Use dynamic rendering (no render pass objects)
- Use bindless descriptors (no descriptor sets)
- Assume coherent caches (simpler synchronization)
- Provide a dramatically simpler API

Goldy is also inspired by:
- Wayland's compositor architecture
- Ralph Levien's ["Requiem for piet-gpu-hal"](https://raphlinus.github.io/rust/gpu/2023/01/07/requiem-piet-gpu-hal.html)
- Slang's vision for unified shader language
- [WGPU](https://gfx-rs.github.io/2019/03/06/wgpu.html)
- This paper on GPU abstractions: https://www.kom.tu-darmstadt.de/papers/KCGS17.pdf

Read more in [Design Philosophy](https://koubaa.github.io/goldy/design/motivation.html).

## Target Hardware

| Platform | Minimum |
|----------|---------|
| NVIDIA | RTX 2000 / GTX 1600 (2018+) |
| AMD | RDNA 1 / RX 5000 (2019+) |
| Intel | Xe / Alchemist (2022+) |
| Apple | M1 / A14 (2020+) |

## Development

Before submitting a PR, run the CI checks locally:

```bash
cargo fmt --all -- --check
cargo clippy -- -D warnings
cargo test
```

## License

Goldy is licensed under the [MIT License](LICENSE). You may use, modify, and distribute Goldy in any project, including proprietary software, under the terms of the MIT License.

## Author

Mohamed Koubaa
