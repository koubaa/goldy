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
- A GPU with Vulkan 1.4+, DX12, or Metal Tier 2+ support (CUDA and WebGPU backends are in progress; Tenstorrent is planned)

### Optional Dependencies

```bash
pip install goldy[dev]   # pytest, pillow
pip install pillow       # image output only
```

## Quick Start

```python
import goldy
import numpy as np

instance = goldy.Instance()
device = instance.request_adapter().request_device()
ctx = device.create_context()

retained_pool = goldy.RetainedPool(device)
vertices = np.array([
    0.0, -0.5, 1.0, 0.0, 0.0, 1.0,
    -0.5,  0.5, 0.0, 1.0, 0.0, 1.0,
     0.5,  0.5, 0.0, 0.0, 1.0, 1.0,
], dtype=np.float32)
vertex_parcel = retained_pool.acquire_buffer(vertices, goldy.BufferKind.SCATTERED)[0]

shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
pipeline = goldy.RenderPipeline(device, shader, shader, goldy.RenderPipelineDesc())

readback = retained_pool.acquire_texture(
    100, 100, goldy.TextureFormat.RGBA8_UNORM,
    goldy.TextureKind.DIRECT, copy_src=True, copy_dst=True,
)

scheme = goldy.Scheme(ctx)
rt = scheme.lease_render_target(100, 100, goldy.TextureFormat.RGBA8_UNORM)
with scheme.render_pass("triangle", rt, goldy.TargetLoad.clear(goldy.Color(0.1, 0.1, 0.2, 1.0))) as rp:
    rp.with_parcel(vertex_parcel, goldy.NodeAccess.READ)
    rp.set_pipeline(pipeline)
    rp.set_vertex_buffer_parcel(0, vertex_parcel)
    rp.draw(vertex_count=3)

scheme.copy_to_texture(rt, readback)
memory = goldy.MemoryExchange(ctx)
withdraw = memory.bind_withdraw_texture(scheme, readback)
submission = scheme.submit()
pixels = np.frombuffer(withdraw.claim(submission).consume(), dtype=np.uint8).reshape(100, 100, 4)
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

Use `MemoryExchange.bind_withdraw` / `bind_withdraw_texture`, then claim and consume after submit:

```python
memory = goldy.MemoryExchange(ctx)
withdraw = memory.bind_withdraw(scheme, parcel)
submission = scheme.submit()
output = np.frombuffer(withdraw.claim(submission).consume(), dtype=np.float32)
```

### Performance Tips

- **Create once, update often** — avoid allocating new parcels every frame. Reuse retained buffers and update via upload schemes when needed.
- **Use `np.float32`** — match the GPU's expected dtype to avoid an extra conversion.
- **Ensure contiguity** — sliced arrays may not be contiguous. Call `np.ascontiguousarray()` before uploading if needed.

## Compute Shaders

Goldy supports GPU compute from Python using Slang shaders.

### Basic Example

```python
import goldy
import numpy as np

instance = goldy.Instance()
device = instance.request_adapter().request_device()
ctx = device.create_context()

data = np.arange(256, dtype=np.float32)
retained_pool = goldy.RetainedPool(device)
parcel = retained_pool.acquire_buffer(data, goldy.BufferKind.SCATTERED)[0]

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

scheme = goldy.Scheme(ctx)
scheme.node("double", pipeline).with_parcel(
    parcel, goldy.NodeAccess.READ_WRITE
).dispatch(4, 1, 1)
memory = goldy.MemoryExchange(ctx)
withdraw = memory.bind_withdraw(scheme, parcel)
submission = scheme.submit()
output = np.frombuffer(withdraw.claim(submission).consume(), dtype=np.float32)
```

### Ping-Pong Buffers

For iterative algorithms, alternate two buffer fields as input/output within one scheme (see `python/examples/game_of_life.py`).

### Combining Compute and Graphics

Hybrid compute + render workflows use a single `Scheme` with both compute nodes and render passes (see `python/examples/game_of_life.py` and `examples/game_of_life.rs`).

## Key Differences from Rust

| Aspect | Rust | Python |
|--------|------|--------|
| Instance creation | `Instance::new()?` | `goldy.Instance()` |
| Error handling | `Result<T, GoldyError>` | Raises `goldy.GoldyError` |
| Retained buffer | `retained_pool.acquire_buffer_with_data(&data, access)` | `retained_pool.acquire_buffer(numpy_array, access)` → `Parcel` |
| Render pass | `scheme.render_pass(...)` | `with scheme.render_pass(...) as rp:` |
| Compute node | `scheme.node(...).dispatch(...)` | `scheme.node(...).with_parcel(...).dispatch(...)` |
| Readback | `grant.consume(&submission)` | `grant.consume(submission)` |
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
instance.backend_type            # BackendType (Vulkan, DX12, Metal; CUDA and WebGPU in progress)
instance.enumerate_adapters()    # list of AdapterInfo
instance.request_adapter()       # Adapter
```

#### `Device` / `Context`

```python
device = instance.request_adapter().request_device()
ctx = device.create_context()
```

#### `RetainedPool` and `Parcel`

```python
pool = goldy.RetainedPool(device)
parcel = pool.acquire_buffer(data, access)  # data: numpy array or bytes
parcel.byte_size                            # int (bytes)
```

#### `Scheme`

```python
scheme = goldy.Scheme(ctx)
rt = scheme.lease_render_target(w, h, goldy.TextureFormat.RGBA8_UNORM)

with scheme.render_pass("main", rt, goldy.TargetLoad.clear(goldy.Color.BLACK)) as rp:
    rp.with_parcel(buf, goldy.NodeAccess.READ)
    rp.set_pipeline(pipeline)
    rp.draw(vertex_count=3)

scheme.node("update", compute_pipeline).with_parcel(
    buf, goldy.NodeAccess.READ_WRITE
).dispatch(wg_x, wg_y, 1)

surface = goldy.SurfaceExchange.from_glfw(ctx, window)
present = surface.bind_render_target(scheme, rt)
submission = scheme.submit()
present.claim(submission).consume()
```

#### `ShaderModule` / `RenderPipeline` / `ComputePipeline`

Standard pipeline construction — see `python/examples/triangle_headless.py`.

### Enums

```python
goldy.DeviceType.DISCRETE_GPU | INTEGRATED_GPU | CPU | OTHER
goldy.TextureFormat.RGBA8_UNORM | RGBA8_UNORM_SRGB | BGRA8_UNORM
goldy.BufferKind.SCATTERED | BROADCAST
goldy.NodeAccess.READ | WRITE | READ_WRITE | OVERWRITE
```

### Exceptions

All errors are raised as `goldy.GoldyError`:

```python
try:
    device = instance.request_adapter().request_device()
except goldy.GoldyError as e:
    print(f"GPU error: {e}")
```
