# Installation

## Requirements

- **Rust** stable (recent version recommended)
- A [supported GPU](../design/hardware.md)

## Adding Goldy to Your Project

```toml
[dependencies]
goldy = "0.1"
```

Or with cargo:

```bash
cargo add goldy
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `vulkan` | yes | Vulkan 1.4+ backend (Linux, Windows) |
| `dx12` | yes | DirectX 12 backend (Windows) |
| `metal` | yes | Metal Tier 2+ backend (macOS) |
| `cuda` | no | CUDA backend (in progress; NVIDIA compute) |
| `webgpu` | no | WebGPU backend (in progress; via wgpu) |
| `instrumentation` | yes | Structured tracing via `tracing-subscriber` (zero-cost when disabled) |

Platform-inappropriate features are no-ops — enabling `metal` on Linux or `dx12` on macOS compiles cleanly but does nothing.

To build with only specific backends:

```toml
[dependencies]
goldy = { version = "0.1", default-features = false, features = ["vulkan"] }
```

## Shader Toolchain

Goldy uses **Slang** as its shader language. The Rust `build.rs` downloads (if needed) and **embeds** the pinned Slang version at compile time; at runtime Goldy extracts and loads it automatically. Application developers do not install Slang separately.

Set `GOLDY_SLANG_PATH` only to override with a custom Slang build.

## Verifying Installation

```rust
use goldy::{DeviceDescriptor, Instance, RequestAdapterOptions};

fn main() -> anyhow::Result<()> {
    let instance = Instance::new()?;

    println!("Available GPUs:");
    for adapter in instance.enumerate_adapters() {
        println!("  {} ({:?})", adapter.name, adapter.device_type);
    }

    let device = instance
        .request_adapter(&RequestAdapterOptions::default())?
        .request_device(&DeviceDescriptor::default())?;
    println!("\nUsing: {}", device.adapter_info().name);

    Ok(())
}
```

```bash
cargo run
```

Expected output:

```
Available GPUs:
  NVIDIA GeForce RTX 4060 Ti (DiscreteGpu)
  Intel(R) UHD Graphics 770 (IntegratedGpu)

Using: NVIDIA GeForce RTX 4060 Ti
```

## Backend Selection

Goldy selects the best backend for your platform automatically:

| Platform | Default Backend |
|----------|-----------------|
| Windows  | DX12            |
| Linux    | Vulkan          |
| macOS    | Metal           |

Override at runtime with `GOLDY_BACKEND`:

```bash
GOLDY_BACKEND=vulkan cargo run
```

## Platform-Specific Setup

### Windows

DX12 is used by default and requires no additional setup. For the Vulkan backend, install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home). Ensure your GPU drivers are up to date.

### Linux

Install Vulkan development packages:

```bash
# Ubuntu/Debian
sudo apt install libvulkan-dev vulkan-tools

# Fedora
sudo dnf install vulkan-loader-devel vulkan-tools

# Arch
sudo pacman -S vulkan-icd-loader vulkan-tools
```

### macOS

Goldy uses the native Metal backend — no MoltenVK or Vulkan SDK needed. Ensure macOS 12+ and Xcode command-line tools are installed:

```bash
xcode-select --install
```

## Windowing (for examples)

Examples require the `examples` feature (winit is gated):

```bash
cargo run --features examples --example triangle --release
```

## Next Steps

- [Your First Triangle](./first-triangle.md) — draw a colored triangle
- [Your First Compute Shader](./first-compute.md) — write pixels from compute
