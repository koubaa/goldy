# Python Bindings Installation

Goldy provides Python bindings via PyO3, offering a Pythonic API for GPU programming with seamless NumPy integration.

## Requirements

- Python 3.9 or later
- NumPy 1.20 or later
- A compatible GPU with Vulkan 1.4+ or DX12 support

## Installation

### From PyPI (Recommended)

```bash
pip install goldy
```

### From Source

If you need to build from source (for development or unreleased features):

1. Install Rust toolchain: [rustup.rs](https://rustup.rs)
2. Install maturin: `pip install maturin`
3. Clone the repository and build:

```bash
git clone https://github.com/koubaa/goldy.git
cd goldy/python
maturin develop --release
```

## Verify Installation

```python
import goldy

# Check version
print(f"Goldy version: {goldy.__version__}")

# Create an instance and list available GPUs
instance = goldy.Instance()
adapters = instance.enumerate_adapters()

for adapter in adapters:
    print(f"  {adapter.name} ({adapter.device_type})")
```

## Optional Dependencies

For development and examples:

```bash
pip install goldy[dev]  # Installs pytest, pillow
```

For image output in examples:

```bash
pip install pillow
```

## Backend Selection

Goldy automatically selects the best backend for your platform:
- **Windows**: DX12 (default)
- **Linux**: Vulkan

Override with the `GOLDY_BACKEND` environment variable:

```python
import os
os.environ["GOLDY_BACKEND"] = "vulkan"  # Must be set before importing goldy

import goldy
instance = goldy.Instance()  # Will use Vulkan
```

Or from the command line:

```bash
GOLDY_BACKEND=vulkan python my_script.py
```

## Troubleshooting

### No GPU Backend Found

**On Windows (DX12 default):**
- Ensure you have Windows 10/11 with updated GPU drivers
- DX12 should work out of the box on modern systems

**On Linux (Vulkan default):**
```bash
sudo apt install vulkan-tools mesa-vulkan-drivers
vulkaninfo  # Verify Vulkan works
```

### Using Vulkan on Windows

If you want to use Vulkan instead of DX12:
1. Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
2. Set `GOLDY_BACKEND=vulkan` before running


