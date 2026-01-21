# NumPy Integration

Goldy's Python bindings are designed for seamless NumPy integration, enabling efficient data transfer between Python and the GPU.

## Buffer Creation from NumPy Arrays

Create GPU buffers directly from NumPy arrays:

```python
import goldy
import numpy as np

instance = goldy.Instance()
device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)

# Create vertex data as NumPy array
vertices = np.array([
    # x, y, r, g, b, a
    0.0, -0.5, 1.0, 0.0, 0.0, 1.0,   # Top vertex (red)
    0.5,  0.5, 0.0, 1.0, 0.0, 1.0,   # Bottom right (green)
   -0.5,  0.5, 0.0, 0.0, 1.0, 1.0,   # Bottom left (blue)
], dtype=np.float32)

# Create GPU buffer from the array
buffer = goldy.Buffer(device, vertices, goldy.DataAccess.SCATTERED)
print(f"Buffer size: {buffer.size} bytes")  # 72 bytes (18 floats * 4)
```

## Supported Data Types

Goldy accepts NumPy arrays of various types:

| NumPy dtype | Use case |
|------------|----------|
| `np.float32` | Vertex positions, colors, uniforms |
| `np.float64` | High-precision data |
| `np.uint32` | Index buffers, compute data |
| `np.int32` | Signed integer data |
| `np.uint16` | 16-bit index buffers |
| `np.uint8` | Raw byte data |

```python
# Index buffer example
indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
index_buffer = goldy.Buffer(device, indices, goldy.DataAccess.SCATTERED)

# Storage buffer for compute
data = np.arange(1024, dtype=np.float32)
storage_buffer = goldy.Buffer(device, data, goldy.DataAccess.SCATTERED)
```

## Reading Back to NumPy

Read GPU results directly into NumPy arrays:

```python
# Render target readback returns a NumPy array
target = goldy.RenderTarget(device, 800, 600, goldy.TextureFormat.RGBA8_UNORM)

# ... render commands ...

# Read pixels as NumPy array with shape (height, width, 4)
pixels = target.read_to_cpu()

print(f"Shape: {pixels.shape}")      # (600, 800, 4)
print(f"Dtype: {pixels.dtype}")      # uint8
print(f"Mean red: {pixels[:,:,0].mean():.2f}")
```

## Buffer Updates

Update buffer contents from NumPy arrays:

```python
# Create buffer with initial data
data = np.zeros(256, dtype=np.float32)
buffer = goldy.Buffer(device, data, goldy.DataAccess.BROADCAST)

# Update with new data
new_data = np.random.rand(256).astype(np.float32)
buffer.write(0, new_data)

# Partial update (starting at offset 64 bytes)
partial = np.ones(32, dtype=np.float32)
buffer.write(64, partial)
```

## Performance Tips

### Batch Operations

Minimize buffer creation in hot loops:

```python
# Bad: Creating buffer every frame
for frame in range(1000):
    buffer = goldy.Buffer(device, vertices, goldy.DataAccess.SCATTERED)  # Slow!

# Good: Create once, update as needed
buffer = goldy.Buffer(device, vertices, goldy.DataAccess.SCATTERED)
for frame in range(1000):
    if vertices_changed:
        buffer.write(0, vertices)
```

### Use Appropriate dtypes

Match your buffer's expected format:

```python
# Storage buffers typically use float32 or uint32
compute_data = np.array([...], dtype=np.float32)  # Good

# Avoid unnecessary type conversions
compute_data = np.array([...])  # Default float64
compute_data = compute_data.astype(np.float32)  # Extra conversion
```

### Contiguous Arrays

Ensure arrays are contiguous for optimal transfer:

```python
# Sliced arrays may not be contiguous
arr = np.arange(100, dtype=np.float32)
sliced = arr[::2]  # Not contiguous!

# Make contiguous if needed
contiguous = np.ascontiguousarray(sliced)
buffer = goldy.Buffer(device, contiguous, goldy.DataAccess.SCATTERED)
```


