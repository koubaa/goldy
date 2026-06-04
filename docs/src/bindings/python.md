# Python Bindings

Goldy provides Python bindings via PyO3, offering a Pythonic API for GPU programming with seamless NumPy integration.

## Installation

### From PyPI

```bash
pip install goldy
```

### From Source

```bash
git clone https://github.com/koubaa/goldy.git
cd goldy/python
pip install maturin
maturin develop --release
```

### Requirements

- Python 3.9+
- NumPy 1.20+
- A GPU with Vulkan 1.4+, DX12, or Metal Tier 2+ support

### Optional Dependencies

```bash
pip install goldy[dev]   # pytest, pillow
pip install pillow       # image output only
```

## Quick Start

```python
import goldy
import numpy as np
from PIL import Image

# Setup
instance = goldy.Instance()
device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
target = goldy.RenderTarget(device, 800, 600, goldy.TextureFormat.RGBA8_UNORM)

# Render
encoder = goldy.CommandEncoder()
with encoder.begin_render_pass() as rp:
    rp.clear(goldy.Color.CORNFLOWER_BLUE)
target.render(encoder)

# Read back as NumPy array and save
pixels = target.read_to_cpu()              # shape (600, 800, 4), dtype uint8
Image.fromarray(pixels, mode='RGBA').save('hello_goldy.png')
```

## NumPy Integration

### Creating GPU Buffers from Arrays

```python
vertices = np.array([
    # x, y, r, g, b, a
    0.0, -0.5, 1.0, 0.0, 0.0, 1.0,
    0.5,  0.5, 0.0, 1.0, 0.0, 1.0,
   -0.5,  0.5, 0.0, 0.0, 1.0, 1.0,
], dtype=np.float32)

buffer = goldy.Buffer(device, vertices, goldy.DataAccess.SCATTERED)
```

### Supported dtypes

| NumPy dtype    | Typical use case                        |
|----------------|-----------------------------------------|
| `np.float32`   | Vertex positions, colors, uniforms      |
| `np.float64`   | High-precision data                     |
| `np.uint32`    | Index buffers, compute data             |
| `np.int32`     | Signed integer data                     |
| `np.uint16`    | 16-bit index buffers                    |
| `np.uint8`     | Raw byte data                           |

### Reading Results Back to NumPy

Render target readback returns a NumPy array directly:

```python
pixels = target.read_to_cpu()
print(pixels.shape)   # (height, width, 4)
print(pixels.dtype)   # uint8
```

### Updating Buffers

```python
buffer = goldy.Buffer(device, np.zeros(256, dtype=np.float32), goldy.DataAccess.BROADCAST)

# Full update
buffer.write(0, np.random.rand(256).astype(np.float32))

# Partial update (starting at byte offset 64)
buffer.write(64, np.ones(32, dtype=np.float32))
```

### Performance Tips

- **Create once, update often** — avoid allocating new `Buffer` objects every frame. Use `buffer.write()` instead.
- **Use `np.float32`** — match the GPU's expected dtype to avoid an extra conversion.
- **Ensure contiguity** — sliced arrays may not be contiguous. Call `np.ascontiguousarray()` before uploading if needed.

## Compute Shaders

Goldy supports GPU compute from Python using Slang shaders.

### Basic Example

```python
import goldy
import numpy as np

instance = goldy.Instance()
device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)

data = np.arange(256, dtype=np.float32)
buffer = goldy.Buffer(device, data, goldy.DataAccess.SCATTERED)

SHADER = """
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<float> data, ThreadId id) {
    data[id.x] = data[id.x] * 2.0;
}
"""

shader = goldy.ShaderModule.from_slang(device, SHADER)
pipeline = goldy.ComputePipeline(device, shader)

encoder = goldy.ComputeEncoder()
with encoder.begin_compute_pass() as cp:
    cp.set_pipeline(pipeline)
    cp.bind_resources([buffer])
    cp.dispatch(4, 1, 1)      # 4 workgroups × 64 threads = 256 threads
encoder.dispatch(device)
```

### Ping-Pong Buffers

For iterative algorithms, alternate two buffers as input/output:

```python
buf_a = goldy.Buffer(device, initial_data, goldy.DataAccess.SCATTERED)
buf_b = goldy.Buffer(device, initial_data, goldy.DataAccess.SCATTERED)

use_a = True
for _ in range(100):
    encoder = goldy.ComputeEncoder()
    with encoder.begin_compute_pass() as cp:
        cp.set_pipeline(pipeline)
        cp.bind_resources([buf_a, buf_b] if use_a else [buf_b, buf_a])
        cp.dispatch(workgroups_x, workgroups_y, 1)
    encoder.dispatch(device)
    use_a = not use_a
```

### Combining Compute and Graphics

Use compute results directly in a subsequent render pass through shared storage buffers:

```python
# Compute pass
compute_encoder = goldy.ComputeEncoder()
with compute_encoder.begin_compute_pass() as cp:
    cp.set_pipeline(compute_pipeline)
    cp.bind_resources([buffer])
    cp.dispatch(workgroups, 1, 1)
compute_encoder.dispatch(device)

# Render pass — reads the same buffer
render_encoder = goldy.CommandEncoder()
with render_encoder.begin_render_pass() as rp:
    rp.set_pipeline(render_pipeline)
    rp.bind_resources([buffer])
    rp.draw(range(3))
target.render(render_encoder)
```

## Key Differences from Rust

| Aspect | Rust | Python |
|--------|------|--------|
| Instance creation | `Instance::new()?` | `goldy.Instance()` |
| Error handling | `Result<T, GoldyError>` | Raises `goldy.GoldyError` |
| Buffer data | `device.alloc_buffer_with_data( &[T], access)` | `goldy.Buffer(device, numpy_array, access)` |
| Render pass | `encoder.begin_render_pass()` returns struct | Context manager (`with ... as rp`) |
| Pixel readback | `target.read_to_cpu()` → `Vec<u8>` | `target.read_to_cpu()` → NumPy array `(H, W, 4)` |
| Resource lifetime | Explicit `Arc<Device>` ownership | Managed by Python GC via PyO3 |

## Backend Selection

Goldy auto-selects the best backend per platform (DX12 on Windows, Vulkan on Linux). Override with `GOLDY_BACKEND`:

```python
import os
os.environ["GOLDY_BACKEND"] = "vulkan"   # set before importing goldy

import goldy
instance = goldy.Instance()
```

## API Reference

### Core Classes

#### `Instance`

```python
instance = goldy.Instance()
instance.backend_type            # BackendType (Vulkan, DX12, Metal)
instance.enumerate_adapters()    # list of AdapterInfo
instance.create_device(type)     # Device
```

#### `Device`

```python
device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
device.is_valid()                # bool
```

#### `Buffer`

```python
buf = goldy.Buffer(device, data, access)    # data: numpy array or bytes
buf = goldy.Buffer.empty(device, size, access)
buf.size                                    # int (bytes)
buf.write(offset, data)                     # update contents
```

#### `RenderTarget`

```python
target = goldy.RenderTarget(device, width, height, format, depth_format=None)
target.width, target.height
target.format
target.has_depth
target.render(encoder)
target.read_to_cpu()       # numpy array (H, W, 4)
```

#### `ShaderModule`

```python
shader = goldy.ShaderModule.from_slang(device, slang_source)
```

#### `RenderPipeline`

```python
pipeline = goldy.RenderPipeline(device, vertex_shader, fragment_shader, desc)
```

#### `RenderPipelineDesc`

```python
desc = goldy.RenderPipelineDesc(
    vertex_layout=None,
    topology=goldy.PrimitiveTopology.TRIANGLE_LIST,
    target_format=goldy.TextureFormat.RGBA8_UNORM,
    depth_stencil=None,
)
```

#### `CommandEncoder` / `RenderPass`

```python
encoder = goldy.CommandEncoder()
with encoder.begin_render_pass() as rp:
    rp.clear(goldy.Color.BLACK)
    rp.set_pipeline(pipeline)
    rp.set_vertex_buffer(slot, buffer)
    rp.set_index_buffer(buffer, format)
    rp.bind_resources([buf1, buf2])
    rp.draw(vertices, instances=range(1))
    rp.draw_indexed(indices, base_vertex, instances)
```

### Compute Classes

#### `ComputePipeline`

```python
pipeline = goldy.ComputePipeline(device, shader)
```

#### `ComputeEncoder`

```python
encoder = goldy.ComputeEncoder()
with encoder.begin_compute_pass() as cp:
    cp.set_pipeline(pipeline)
    cp.bind_resources([buffer])
    cp.dispatch(wg_x, wg_y, wg_z)
encoder.dispatch(device)
```

### Enums

```python
# Device selection
goldy.DeviceType.DISCRETE_GPU | INTEGRATED_GPU | CPU | OTHER

# Texture formats
goldy.TextureFormat.RGBA8_UNORM | RGBA8_UNORM_SRGB | BGRA8_UNORM
                   | R8_UNORM | RG8_UNORM | RGBA16_FLOAT | RGBA32_FLOAT

# Buffer access patterns
goldy.DataAccess.SCATTERED    # any thread, any address (StructuredBuffer)
goldy.DataAccess.BROADCAST    # all threads same address (ConstantBuffer)

# Texture access patterns
goldy.SpatialAccess.INTERPOLATED   # hardware-filtered (Texture2D + sampler)
goldy.SpatialAccess.DIRECT         # direct indexing (RWTexture2D)

# Primitive topology
goldy.PrimitiveTopology.POINT_LIST | LINE_LIST | LINE_STRIP
                       | TRIANGLE_LIST | TRIANGLE_STRIP

# Index format
goldy.IndexFormat.UINT16 | UINT32
```

### Types

#### `Color`

```python
color = goldy.Color(r, g, b, a=1.0)       # floats 0-1
color = goldy.Color.from_rgb(255, 128, 0)  # bytes 0-255

# Predefined
goldy.Color.BLACK | WHITE | RED | GREEN | BLUE | CORNFLOWER_BLUE
```

#### `VertexBufferLayout`

```python
layout = goldy.VertexBufferLayout.vertex_2d()       # pos(2) + color(4)
layout = goldy.VertexBufferLayout.vertex_2d_uv()    # pos(2) + uv(2)
layout = goldy.VertexBufferLayout(stride, [
    goldy.VertexAttribute(location, format, offset),
])
```

#### `DepthStencilState`

```python
depth = goldy.DepthStencilState(
    format=goldy.DepthFormat.DEPTH32_FLOAT,
    depth_write_enabled=True,
    depth_compare=goldy.CompareFunction.LESS,
)
```

### Exceptions

All errors are raised as `goldy.GoldyError`:

```python
try:
    device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
except goldy.GoldyError as e:
    print(f"GPU error: {e}")
```
