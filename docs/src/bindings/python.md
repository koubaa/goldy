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
python -m venv .venv
source .venv/Scripts/activate   # platform-specific
pip install -e ".[dev]"
```

Slang is embedded when the extension is compiled; you do not run `build-slang.py` for
local development. Rebuild after editing `python/src/*.rs` with `maturin develop`.

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

# Graphics via TaskGraph (headless)
graph = goldy.TaskGraph()
with graph.render_pass("clear", target) as rp:
    rp.clear(goldy.Color.CORNFLOWER_BLUE)
graph.dispatch(device)
pixels = target.read_to_cpu()
```

## NumPy Integration

### Creating GPU Parcels from Arrays

```python
vertices = np.array([
    # x, y, r, g, b, a
    0.0, -0.5, 1.0, 0.0, 0.0, 1.0,
    0.5,  0.5, 0.0, 1.0, 0.0, 1.0,
   -0.5,  0.5, 0.0, 0.0, 1.0, 1.0,
], dtype=np.float32)

retained_pool = goldy.RetainedPool(device)
parcel = retained_pool.acquire_buffer(vertices, goldy.BufferKind.SCATTERED)
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
retained_pool = goldy.RetainedPool(device)
parcel = retained_pool.acquire_buffer(np.zeros(256, dtype=np.float32), goldy.BufferKind.BROADCAST)
graph = goldy.TaskGraph()

# Full update
graph.write_parcel(parcel, 0, np.random.rand(256).astype(np.float32).tobytes())

# Partial update (starting at byte offset 64)
graph.write_parcel(parcel, 64, np.ones(32, dtype=np.float32).tobytes())
```

### Performance Tips

- **Create once, update often** — avoid allocating new parcels every frame. Use `graph.write_parcel()` instead.
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
retained_pool = goldy.RetainedPool(device)
parcel = retained_pool.acquire_buffer(data, goldy.BufferKind.SCATTERED)

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

graph = goldy.TaskGraph()
idx = parcel.resource_index(goldy.ResourceAccess.WRITE)
with graph.compute_node("double", pipeline, workgroups=(4, 1, 1)) as node:
    node.with_parcel(parcel, goldy.NodeAccess.READ_WRITE)
    node.with_resource_slots([idx])   # 4 workgroups × 64 threads = 256 threads
graph.dispatch(device)
```

### Ping-Pong Buffers

For iterative algorithms, alternate two buffers as input/output:

```python
retained_pool = goldy.RetainedPool(device)
buf_a = retained_pool.acquire_buffer(initial_data, goldy.BufferKind.SCATTERED)
buf_b = retained_pool.acquire_buffer(initial_data, goldy.BufferKind.SCATTERED)

use_a = True
for _ in range(100):
    read_buf, write_buf = (buf_a, buf_b) if use_a else (buf_b, buf_a)
    read_idx = read_buf.resource_index(goldy.ResourceAccess.READ)
    write_idx = write_buf.resource_index(goldy.ResourceAccess.WRITE)
    graph = goldy.TaskGraph()
    with graph.compute_node("step", pipeline, workgroups=(workgroups_x, workgroups_y, 1)) as node:
        node.with_parcel(read_buf, goldy.NodeAccess.READ)
        node.with_parcel(write_buf, goldy.NodeAccess.WRITE)
        node.with_resource_slots([read_idx, write_idx])
    graph.dispatch(device)
    use_a = not use_a
```

### Combining Compute and Graphics

Standalone and hybrid compute workflows both use `TaskGraph` (see `python/examples/compute_demo.py`, `python/examples/game_of_life.py`, and `goldy/examples/game_of_life.rs`). Python exposes `render_pass` and `compute_node` on the same graph.

## Key Differences from Rust

| Aspect | Rust | Python |
|--------|------|--------|
| Instance creation | `Instance::new()?` | `goldy.Instance()` |
| Error handling | `Result<T, GoldyError>` | Raises `goldy.GoldyError` |
| Retained buffer | `retained_pool.acquire_buffer_with_data(&data, access)` | `retained_pool.acquire_buffer(numpy_array, access)` → `Parcel` |
| Render pass | `RenderPassBuilder` on `TaskGraph` | `with graph.render_pass(...) as rp:` |
| Compute node | `graph.node(...).dispatch(...)` | `with graph.compute_node(...) as node:` |
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

#### `RetainedPool` and `Parcel`

```python
pool = goldy.RetainedPool(device)
parcel = pool.acquire_buffer(data, access)  # data: numpy array or bytes
parcel.byte_size                            # int (bytes)
graph.write_parcel(parcel, offset, data)    # update contents through TaskGraph
```

#### `RenderTarget`

```python
target = goldy.RenderTarget(device, width, height, format, depth_format=None)
target.width, target.height
target.format
target.has_depth
target.read_to_cpu()       # numpy array (H, W, 4) — render via TaskGraph first
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

#### `TaskGraph` / `RenderPass`

```python
graph = goldy.TaskGraph()
graph.clear()

with graph.render_pass("main", scene_rt) as rp:
    rp.with_parcel(vertex_buffer, goldy.NodeAccess.READ)
    rp.clear(goldy.Color.BLACK)
    rp.set_pipeline(pipeline)
    rp.set_vertex_buffer_parcel(0, vertex_buffer)
    rp.draw(vertex_count=3)

# Headless
graph.dispatch(device)

# Windowed
swapchain = graph.declare_swapchain_output()
graph.copy_render_target_to_swapchain(scene_rt, swapchain)
frame = surface.acquire()
surface.submit_graph_to_frame(graph, frame)
surface.present(frame)
```

#### `ComputeNode` (on `TaskGraph`)

```python
with graph.compute_node("update", compute_pipeline, workgroups=(8, 8, 1)) as node:
    node.with_parcel(state_buf, goldy.NodeAccess.READ_WRITE)
    node.with_resource_slots([state_idx])
```

### Compute Classes

#### `ComputePipeline`

```python
pipeline = goldy.ComputePipeline(device, shader)
```

### Enums

```python
# Device selection
goldy.DeviceType.DISCRETE_GPU | INTEGRATED_GPU | CPU | OTHER

# Texture formats
goldy.TextureFormat.RGBA8_UNORM | RGBA8_UNORM_SRGB | BGRA8_UNORM
                   | R8_UNORM | RG8_UNORM | RGBA16_FLOAT | RGBA32_FLOAT

# Buffer access patterns
goldy.BufferKind.SCATTERED    # any thread, any address (StructuredBuffer)
goldy.BufferKind.BROADCAST    # all threads same address (ConstantBuffer)

# Texture access patterns
goldy.TextureKind.INTERPOLATED   # hardware-filtered (Texture2D + sampler)
goldy.TextureKind.DIRECT         # direct indexing (RWTexture2D)

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
