# Python API Reference

Complete reference for the Goldy Python bindings.

## Core Classes

### Instance

Creates a Goldy instance and manages GPU discovery.

```python
instance = goldy.Instance()
```

**Properties:**
- `backend_type` - The active backend (Vulkan, DX12, or Metal)

**Methods:**
- `enumerate_adapters()` - List available GPUs
- `create_device(device_type)` - Create a device on a compatible GPU

### Device

Represents a GPU and manages resources.

```python
device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
```

**Methods:**
- `is_valid()` - Check if device is usable

### Buffer

GPU memory buffer for vertex, index, uniform, or storage data.

```python
buffer = goldy.Buffer(device, data, usage)
```

**Constructor:**
- `device` - GPU device
- `data` - NumPy array or bytes
- `access` - `DataAccess` pattern (SCATTERED or BROADCAST)

**Properties:**
- `size` - Buffer size in bytes

**Methods:**
- `write(offset, data)` - Update buffer contents

**Static Methods:**
- `Buffer.empty(device, size, usage)` - Create uninitialized buffer

### RenderTarget

Off-screen render target for headless rendering.

```python
target = goldy.RenderTarget(device, width, height, format, depth_format=None)
```

**Properties:**
- `width`, `height` - Dimensions
- `format` - Texture format
- `has_depth` - Whether depth buffer is attached

**Methods:**
- `render(encoder)` - Execute render commands
- `read_to_cpu()` - Read pixels as NumPy array `(H, W, 4)`

### ShaderModule

Compiled shader from Slang source code.

```python
shader = goldy.ShaderModule.from_slang(device, source_code)
```

### RenderPipeline

Graphics pipeline for rendering.

```python
pipeline = goldy.RenderPipeline(device, vertex_shader, fragment_shader, desc)
```

### RenderPipelineDesc

Configuration for render pipeline creation.

```python
desc = goldy.RenderPipelineDesc(
    vertex_layout=None,           # VertexBufferLayout
    topology=PrimitiveTopology.TRIANGLE_LIST,
    target_format=TextureFormat.RGBA8_UNORM,
    depth_stencil=None,           # DepthStencilState
)
```

### CommandEncoder

Records GPU commands for rendering.

```python
encoder = goldy.CommandEncoder()
with encoder.begin_render_pass() as rp:
    rp.clear(goldy.Color.BLACK)
    rp.set_pipeline(pipeline)
    rp.draw(range(3))
```

### RenderPass

Active render pass for recording draw commands.

**Methods:**
- `clear(color)` - Clear color attachment
- `clear_depth(value)` - Clear depth attachment
- `set_pipeline(pipeline)` - Bind render pipeline
- `set_vertex_buffer(slot, buffer)` - Bind vertex buffer
- `set_index_buffer(buffer, format)` - Bind index buffer
- `bind_resources(buffers)` - Pass buffer indices to shaders
- `draw(vertices, instances=range(1))` - Draw vertices
- `draw_indexed(indices, base_vertex, instances)` - Draw indexed

## Compute Classes

### ComputePipeline

Pipeline for compute shaders.

```python
pipeline = goldy.ComputePipeline(device, shader)
```

### ComputeEncoder

Records compute commands.

```python
encoder = goldy.ComputeEncoder()
with encoder.begin_compute_pass() as cp:
    cp.set_pipeline(pipeline)
    cp.bind_resources([buffer])
    cp.dispatch(workgroups_x, workgroups_y, workgroups_z)
encoder.dispatch(device)
```

## Enums

### DeviceType

```python
goldy.DeviceType.DISCRETE_GPU      # Dedicated GPU
goldy.DeviceType.INTEGRATED_GPU    # Integrated graphics
goldy.DeviceType.CPU               # Software renderer
goldy.DeviceType.OTHER
```

### TextureFormat

```python
goldy.TextureFormat.RGBA8_UNORM
goldy.TextureFormat.RGBA8_UNORM_SRGB
goldy.TextureFormat.BGRA8_UNORM
goldy.TextureFormat.R8_UNORM
goldy.TextureFormat.RG8_UNORM
goldy.TextureFormat.RGBA16_FLOAT
goldy.TextureFormat.RGBA32_FLOAT
```

### DataAccess

Data access patterns for buffers:

```python
goldy.DataAccess.SCATTERED  # Any thread, any address (StructuredBuffer)
goldy.DataAccess.BROADCAST  # All threads same address (ConstantBuffer)
```

### SpatialAccess

Spatial access patterns for textures:

```python
goldy.SpatialAccess.INTERPOLATED  # Hardware filtering (Texture2D + sampler)
goldy.SpatialAccess.DIRECT        # Direct indexing (RWTexture2D)
```

### PrimitiveTopology

```python
goldy.PrimitiveTopology.POINT_LIST
goldy.PrimitiveTopology.LINE_LIST
goldy.PrimitiveTopology.LINE_STRIP
goldy.PrimitiveTopology.TRIANGLE_LIST
goldy.PrimitiveTopology.TRIANGLE_STRIP
```

### IndexFormat

```python
goldy.IndexFormat.UINT16
goldy.IndexFormat.UINT32
```

## Types

### Color

```python
# From floats (0-1 range)
color = goldy.Color(r, g, b, a=1.0)

# From bytes (0-255 range)
color = goldy.Color.from_rgb(255, 128, 0)

# Predefined colors
goldy.Color.BLACK
goldy.Color.WHITE
goldy.Color.RED
goldy.Color.GREEN
goldy.Color.BLUE
goldy.Color.CORNFLOWER_BLUE

# Convert
color.to_tuple()   # (r, g, b, a)
color.to_rgba8()   # [r8, g8, b8, a8]
```

### VertexBufferLayout

```python
# Predefined layouts
layout = goldy.VertexBufferLayout.vertex_2d()      # pos(2) + color(4)
layout = goldy.VertexBufferLayout.vertex_2d_uv()   # pos(2) + uv(2)

# Custom layout
layout = goldy.VertexBufferLayout(stride, [
    goldy.VertexAttribute(location, format, offset),
])
```

### DepthStencilState

```python
depth = goldy.DepthStencilState(
    format=goldy.DepthFormat.DEPTH32_FLOAT,
    depth_write_enabled=True,
    depth_compare=goldy.CompareFunction.LESS,
)
```

## Exceptions

### GoldyError

All Goldy errors are raised as `goldy.GoldyError`:

```python
try:
    device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
except goldy.GoldyError as e:
    print(f"GPU error: {e}")
```


