# Quick Start

This guide walks through creating a simple render using Goldy's Python bindings.

## Basic Setup

```python
import goldy
import numpy as np

# Create an instance and select a GPU
instance = goldy.Instance()
device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)

print(f"Backend: {instance.backend_type}")
```

## Creating a Render Target

Render targets are off-screen textures you can draw to:

```python
# Create a 800x600 render target
target = goldy.RenderTarget(
    device, 
    800, 600, 
    goldy.TextureFormat.RGBA8_UNORM
)
```

## Recording Commands

Use a `CommandEncoder` to record GPU commands:

```python
encoder = goldy.CommandEncoder()

with encoder.begin_render_pass() as rp:
    # Clear to a nice blue color
    rp.clear(goldy.Color(0.1, 0.2, 0.4, 1.0))

# Execute the commands
target.render(encoder)
```

## Reading Pixels

Read the result back to CPU as a NumPy array:

```python
# Returns shape (height, width, 4) uint8 array
pixels = target.read_to_cpu()

print(f"Image shape: {pixels.shape}")  # (600, 800, 4)
print(f"First pixel: {pixels[0, 0]}")  # [26, 51, 102, 255]
```

## Saving to File

Use Pillow to save the image:

```python
from PIL import Image

img = Image.fromarray(pixels, mode='RGBA')
img.save('output.png')
```

## Complete Example

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

# Save
pixels = target.read_to_cpu()
Image.fromarray(pixels, mode='RGBA').save('hello_goldy.png')
print("Saved hello_goldy.png!")
```

## Next Steps

- [NumPy Integration](./numpy.md) - Zero-copy buffer sharing
- [Compute Shaders](./compute.md) - GPU computing with Python
- [API Reference](./api.md) - Full API documentation


