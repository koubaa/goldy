# Installation

## Requirements

- **Rust** 1.70 or later
- **Vulkan SDK** 1.4+ (for Vulkan backend)
- A [supported GPU](../design/hardware.md)

## Adding Goldy to Your Project

Add Goldy to your `Cargo.toml`:

```toml
[dependencies]
goldy = "0.1"
```

Or with cargo:

```bash
cargo add goldy
```

## Verifying Installation

Create a simple test program:

```rust
use goldy::{Instance, DeviceType};

fn main() -> anyhow::Result<()> {
    let instance = Instance::new()?;
    
    println!("Available GPUs:");
    for adapter in instance.enumerate_adapters() {
        println!("  {} ({:?})", adapter.name, adapter.device_type);
    }
    
    let device = instance.create_device(DeviceType::DiscreteGpu)?;
    println!("\nUsing: {}", device.adapter_info().name);
    
    Ok(())
}
```

Run it:

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

Goldy automatically selects the best backend for your platform:

| Platform | Default Backend |
|----------|-----------------|
| Windows  | DX12            |
| Linux    | Vulkan          |
| macOS    | Metal (planned) |

You can override this at runtime using the `GOLDY_BACKEND` environment variable:

```bash
# Use Vulkan on Windows (instead of DX12)
GOLDY_BACKEND=vulkan cargo run

# Valid values: vulkan, dx12, metal
```

## Platform-Specific Setup

### Windows

1. DX12 is used by default and requires no additional setup
2. For Vulkan backend: Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
3. Ensure your GPU drivers are up to date

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

Goldy's Metal backend is planned but not yet implemented. For now, use MoltenVK:

```bash
brew install molten-vk
```

## Windowing (for examples)

The examples use `winit` for windowing and Goldy's built-in Surface API for zero-copy GPU presentation:

```toml
[dev-dependencies]
winit = "0.30"
anyhow = "1.0"
```

## Next Steps

- [Your First Triangle](./first-triangle.md) - Draw something!
- [Understanding the API](./understanding-api.md) - Core concepts
- [Examples](../examples/overview.md) - See what's possible

