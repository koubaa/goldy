# Goldy Python Bindings

Python bindings for the [Goldy](https://github.com/koubaa/goldy) GPU library.

## Installation

```bash
pip install goldy
```

Or build from source (development):

```bash
cd goldy/python
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate on cmd/PowerShell
pip install -e ".[dev]"
```

That editable install compiles the Rust extension via maturin. Slang is embedded at
compile time by Goldy's `build.rs` and extracted on first use — no separate Slang
install or `build-slang.py` step is needed for local development.

After changing Rust bindings (`python/src/*.rs`), rebuild with:

```bash
maturin develop
```

**Release wheels only** (maintainers / CI): copy Slang into the package tree before
`maturin build` so wheels ship DLLs alongside the module — see [PACKAGING.md](../PACKAGING.md).

## Quick Start

```python
import goldy
import numpy as np

# Create device
instance = goldy.Instance()
device = instance.request_adapter().request_device()

# Create vertex buffer with a triangle
vertices = np.array([
    # x, y, r, g, b, a
     0.0, -0.5, 1.0, 0.0, 0.0, 1.0,  # red
    -0.5,  0.5, 0.0, 1.0, 0.0, 1.0,  # green
     0.5,  0.5, 0.0, 0.0, 1.0, 1.0,  # blue
], dtype=np.float32)
retained_pool = goldy.RetainedPool(device)
vertex_parcel = retained_pool.acquire_buffer(vertices, goldy.BufferKind.SCATTERED)[0]

# Create shader and pipeline
shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
pipeline = goldy.RenderPipeline(device, shader, shader, goldy.RenderPipelineDesc())

# Graphics via Scheme (headless)
ctx = device.create_context()
scheme = goldy.Scheme(ctx)
rt = scheme.lease_render_target(100, 100, goldy.TextureFormat.RGBA8_UNORM)
with scheme.render_pass("clear", rt) as rp:
    rp.with_parcel(vertex_parcel, goldy.NodeAccess.READ)
    rp.clear(goldy.Color(0.1, 0.1, 0.2, 1.0))
    rp.set_pipeline(pipeline)
    rp.set_vertex_buffer_parcel(0, vertex_parcel)
    rp.draw(vertex_count=3)
scheme.copy_to_texture(rt, readback)
memory = goldy.MemoryExchange(ctx)
withdraw = memory.bind_withdraw_texture(scheme, readback)
submission = scheme.submit()
pixels = np.frombuffer(withdraw.claim(submission).consume(), dtype=np.uint8).reshape(100, 100, 4)
```

## Examples

See the `examples/` directory for complete examples:

- **triangle.py** / **triangle_headless.py** - Colored triangle via Scheme (headless readback)
- **triangle_window.py** - Windowed triangle via Scheme + present (requires GLFW)
- **game_of_life.py** - Hybrid compute + render scheme in a window (requires GLFW)
- **game_of_life_headless.py** - Headless Game of Life smoke test (CI / no display)
- **adapter_info.py** - Print GPU adapter information
- **compute_demo.py** - Standalone compute shader example

Run an example:
```bash
cd goldy/python
python examples/triangle.py
```

## Features

### NumPy Integration

Retained pools accept numpy arrays directly and return buffers (use `[0]` for a single-unit parcel):
```python
vertices = np.array([...], dtype=np.float32)
pool = goldy.RetainedPool(device)
parcel = pool.acquire_buffer(vertices, goldy.BufferKind.SCATTERED)[0]
```

Readback uses `MemoryExchange` withdraw (claim then consume):
```python
memory = goldy.MemoryExchange(ctx)
withdraw = memory.bind_withdraw_texture(scheme, texture)
submission = scheme.submit()
pixels = withdraw.claim(submission).consume()  # raw bytes; reshape as needed
```

### Context Managers

Pythonic API with `with` statements for scheme recording:
```python
with scheme.render_pass("main", rt) as rp:
    rp.with_parcel(parcel, goldy.NodeAccess.READ)
    rp.set_pipeline(pipeline)
    rp.draw(vertex_count=3)

scheme.node("update", pipeline).with_parcel(
    parcel, goldy.NodeAccess.READ_WRITE
).dispatch(4, 1, 1)
submission = scheme.submit()
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
| `RetainedPool` | Allocates retained GPU parcels |
| `Parcel` | Retained buffer or texture resource |
| `ShaderModule` | Compiled Slang shader |
| `RenderPipeline` | Complete render state |
| `SchemeRenderTargetLease` | Off-screen render target declared on a scheme |
| `Scheme` | Retained GPU dependency graph (render passes, compute, present) |
| `SchemeRenderPass` | Draw commands within a render-pass node |
| `SchemeComputeNode` | Record a compute dispatch node on a scheme |
| `NodeAccess` | Read/Write/ReadWrite for scheme dependency tracking |
| `ComputePipeline` | Compute shader pipeline |

### Enums

| Enum | Values |
|------|--------|
| `DeviceType` | `DISCRETE_GPU`, `INTEGRATED_GPU`, `CPU`, `OTHER` |
| `TextureFormat` | `RGBA8_UNORM`, `BGRA8_UNORM`, `RGBA16_FLOAT`, ... |
| `BufferKind` | `SCATTERED` (storage), `BROADCAST` (uniform) |
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
cd goldy/python
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
GOLDY_BACKEND=vulkan python examples/triangle.py

# Use DX12 explicitly  
GOLDY_BACKEND=dx12 python examples/triangle.py
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

If an upload partially fails (for example PyPI file-size limits), re-run **Publish to
PyPI** from GitHub Actions with **Upload built wheels to PyPI** checked. Already-uploaded
files are skipped (`skip-existing`).

### Manual testing (TestPyPI)

For testing before a real release, you can configure a separate Trusted Publisher for TestPyPI at https://test.pypi.org/manage/account/publishing/ and modify the workflow to publish there first.

## License

MIT License. See the [goldy repository](https://github.com/koubaa/goldy/blob/main/LICENSE) for the full text.
