# Goldy Python Bindings

Python bindings for the [Goldy](https://github.com/koubaa/goldy) GPU library.

## Installation

```bash
pip install goldy
```

Or build from source:

```bash
cd goldy/python
pip install maturin

# For development (editable install)
maturin develop

# For wheel building (includes Slang libraries)
python build-slang.py  # Copy Slang libs to package
maturin build --release
```

## Quick Start

```python
import goldy
import numpy as np

# Create device
instance = goldy.Instance()
device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)

# Create vertex buffer with a triangle
vertices = np.array([
    # x, y, r, g, b, a
     0.0, -0.5, 1.0, 0.0, 0.0, 1.0,  # red
    -0.5,  0.5, 0.0, 1.0, 0.0, 1.0,  # green
     0.5,  0.5, 0.0, 0.0, 1.0, 1.0,  # blue
], dtype=np.float32)
buffer = goldy.Buffer(device, vertices, goldy.DataAccess.SCATTERED)

# Create shader and pipeline
shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
pipeline = goldy.RenderPipeline(device, shader, shader, goldy.RenderPipelineDesc())

# Render
target = goldy.RenderTarget(device, 800, 600, goldy.TextureFormat.RGBA8_UNORM)
encoder = goldy.CommandEncoder()
with encoder.begin_render_pass() as rp:
    rp.clear(goldy.Color(0.1, 0.1, 0.2, 1.0))
    rp.set_pipeline(pipeline)
    rp.set_vertex_buffer(0, buffer)
    rp.draw(range(3))

target.render(encoder)
pixels = target.read_to_cpu()  # Returns numpy.ndarray with shape (600, 800, 4)
```

## Examples

See the `examples/` directory for complete examples:

- **hello_triangle.py** - Basic triangle rendering
- **gradient.py** - Custom shader with goldy_exp library
- **adapter_info.py** - Print GPU adapter information
- **compute_demo.py** - Compute shader placeholder

Run an example:
```bash
cd goldy-py
maturin develop
python examples/hello_triangle.py
```

## Features

### NumPy Integration

Buffers accept numpy arrays directly:
```python
vertices = np.array([...], dtype=np.float32)
buffer = goldy.Buffer(device, vertices, goldy.DataAccess.SCATTERED)
```

Render targets return numpy arrays:
```python
pixels = target.read_to_cpu()  # Shape: (height, width, 4), dtype: uint8
```

### Context Managers

Pythonic API with `with` statements for render passes:
```python
with encoder.begin_render_pass() as rp:
    rp.clear(goldy.Color.RED)
    rp.set_pipeline(pipeline)
    rp.draw(range(3))
```

### Shader Libraries

Use the built-in `goldy_exp` library or register custom ones:
```python
# Built-in library
shader = goldy.ShaderModule.from_slang(device, '''
    import goldy_exp;
    
    [shader("fragment")]
    float4 fs_main(FullscreenVarying input) : SV_Target {
        return float4(rainbow(input.uv.x), 1.0);
    }
''')

# Custom library
device.register_library('mylib', '''
    module mylib;
    public float3 my_color() { return float3(1.0, 0.5, 0.0); }
''')
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Instance` | Entry point, enumerate adapters |
| `Device` | GPU device for creating resources |
| `Buffer` | GPU buffer for vertex/index/uniform data |
| `ShaderModule` | Compiled Slang shader |
| `RenderPipeline` | Complete render state |
| `RenderTarget` | Off-screen render target |
| `CommandEncoder` | Record render commands |
| `RenderPass` | Draw commands within a pass |

### Enums

| Enum | Values |
|------|--------|
| `DeviceType` | `DISCRETE_GPU`, `INTEGRATED_GPU`, `CPU`, `OTHER` |
| `TextureFormat` | `RGBA8_UNORM`, `BGRA8_UNORM`, `RGBA16_FLOAT`, ... |
| `DataAccess` | `SCATTERED` (storage), `BROADCAST` (uniform) |
| `PrimitiveTopology` | `TRIANGLE_LIST`, `LINE_LIST`, `POINT_LIST`, ... |

### Types

| Type | Description |
|------|-------------|
| `Color` | RGBA color (float or byte) |
| `VertexBufferLayout` | Vertex format description |
| `RenderPipelineDesc` | Pipeline configuration |
| `DepthStencilState` | Depth testing configuration |

## Testing

```bash
cd goldy-py
maturin develop
pytest tests/ -v
```

Tests are organized into:
- `test_types.py` - Type wrapper tests (no GPU required)
- `test_gpu.py` - GPU integration tests (skipped if no GPU)

## Requirements

- Python 3.9+
- NumPy 1.20+
- A compatible GPU (DX12 on Windows, Vulkan 1.4+ on Linux)

## Backend Selection

Goldy uses DX12 on Windows and Vulkan on Linux by default. Override with `GOLDY_BACKEND`:

```bash
# Use Vulkan on Windows
GOLDY_BACKEND=vulkan python examples/hello_triangle.py

# Use DX12 explicitly  
GOLDY_BACKEND=dx12 python examples/hello_triangle.py
```

Or set it in Python before importing goldy:

```python
import os
os.environ["GOLDY_BACKEND"] = "vulkan"
import goldy
```

## Publishing to PyPI

This package uses GitHub Actions with PyPI Trusted Publishers for automated releases.

### Creating a release

1. Update version in `pyproject.toml` and `python/goldy/__init__.py`
2. Commit and push changes
3. Create a git tag matching the version:
   ```bash
   git tag v0.1.1dev0
   git push origin v0.1.1dev0
   ```
4. create a release:
   `gh release create v0.1.1dev0 --title "v0.1.1dev0" --notes "..."`
5. The publish workflow will automatically build wheels and upload to PyPI

### Manual testing (TestPyPI)

For testing before a real release, you can configure a separate Trusted Publisher for TestPyPI at https://test.pypi.org/manage/account/publishing/ and modify the workflow to publish there first.

## License

MIT License
