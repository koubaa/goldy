# Compute Shaders

Goldy supports GPU compute shaders for parallel data processing. This enables high-performance computing directly from Python.

## Basic Compute Example

```python
import goldy
import numpy as np

# Setup
instance = goldy.Instance()
device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)

# Create storage buffer with data
data = np.arange(256, dtype=np.float32)
buffer = goldy.Buffer(device, data, goldy.BufferUsage.STORAGE)

# Define compute shader (Slang)
SHADER = """
[[vk::binding(0, 0)]] RWStructuredBuffer<float> data;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    data[id.x] = data[id.x] * 2.0;  // Double each value
}
"""

# Create bind group layout and bind group
bind_layout = goldy.BindGroupLayout(device, [
    goldy.BindGroupLayoutBinding(
        0, goldy.ShaderStages.COMPUTE,
        goldy.BindingType.storage_buffer(read_only=False)
    ),
])

bind_group = goldy.BindGroup(device, bind_layout, [
    goldy.BufferBinding(0, buffer),
])

# Create compute pipeline
shader = goldy.ShaderModule.from_slang(device, SHADER)
pipeline = goldy.ComputePipeline(
    device, shader,
    goldy.ComputePipelineDesc([bind_layout])
)

# Dispatch compute work
encoder = goldy.ComputeEncoder()
with encoder.begin_compute_pass() as cp:
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bind_group)
    cp.dispatch(4, 1, 1)  # 4 workgroups × 64 threads = 256 threads

encoder.dispatch(device)
print("Compute shader executed!")
```

## Understanding Workgroups

Compute shaders run in workgroups of threads:

```slang
[numthreads(64, 1, 1)]  // 64 threads per workgroup
```

The total threads = workgroups × threads_per_group:

```python
# For 1024 elements with 64 threads per workgroup:
workgroups = (1024 + 63) // 64  # = 16 workgroups
cp.dispatch(workgroups, 1, 1)   # 16 × 64 = 1024 threads
```

## Ping-Pong Buffers

For iterative algorithms, use two buffers alternating as input/output:

```python
# Create two buffers
buffer_a = goldy.Buffer(device, initial_data, goldy.BufferUsage.STORAGE)
buffer_b = goldy.Buffer(device, initial_data, goldy.BufferUsage.STORAGE)

# Create bind groups for both directions
compute_layout = goldy.BindGroupLayout(device, [
    goldy.BindGroupLayoutBinding(0, goldy.ShaderStages.COMPUTE,
        goldy.BindingType.storage_buffer(read_only=True)),
    goldy.BindGroupLayoutBinding(1, goldy.ShaderStages.COMPUTE,
        goldy.BindingType.storage_buffer(read_only=False)),
])

# A → B
bind_a_to_b = goldy.BindGroup(device, compute_layout, [
    goldy.BufferBinding(0, buffer_a),  # Read
    goldy.BufferBinding(1, buffer_b),  # Write
])

# B → A
bind_b_to_a = goldy.BindGroup(device, compute_layout, [
    goldy.BufferBinding(0, buffer_b),  # Read
    goldy.BufferBinding(1, buffer_a),  # Write
])

# Iterate
use_a = True
for iteration in range(100):
    encoder = goldy.ComputeEncoder()
    with encoder.begin_compute_pass() as cp:
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind_a_to_b if use_a else bind_b_to_a)
        cp.dispatch(workgroups_x, workgroups_y, 1)
    encoder.dispatch(device)
    use_a = not use_a
```

## Game of Life Example

A complete example combining compute and graphics:

```python
import goldy
import numpy as np

GRID_SIZE = 128
COMPUTE_SHADER = f"""
static const uint SIZE = {GRID_SIZE};

[[vk::binding(0, 0)]] StructuredBuffer<uint> current;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> next;

uint getCell(int x, int y) {{
    x = (x + SIZE) % SIZE;
    y = (y + SIZE) % SIZE;
    return current[y * SIZE + x];
}}

uint countNeighbors(int x, int y) {{
    uint n = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            if (dx != 0 || dy != 0)
                n += getCell(x + dx, y + dy);
    return n;
}}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {{
    if (id.x >= SIZE || id.y >= SIZE) return;
    
    uint idx = id.y * SIZE + id.x;
    uint cell = current[idx];
    uint neighbors = countNeighbors(int(id.x), int(id.y));
    
    // Conway's rules
    next[idx] = (cell == 1) ? 
        ((neighbors == 2 || neighbors == 3) ? 1 : 0) :
        ((neighbors == 3) ? 1 : 0);
}}
"""

# Setup and run simulation...
# See python/examples/game_of_life.py for full example
```

## Combining Compute and Graphics

Use compute results in render passes via shared storage buffers:

```python
# Create storage buffer accessible from both compute and fragment shaders
buffer = goldy.Buffer(device, data, goldy.BufferUsage.STORAGE)

# Compute bind group
compute_bind = goldy.BindGroup(device, compute_layout, [
    goldy.BufferBinding(0, buffer),
])

# Render bind group (same buffer!)
render_layout = goldy.BindGroupLayout(device, [
    goldy.BindGroupLayoutBinding(0, goldy.ShaderStages.FRAGMENT,
        goldy.BindingType.storage_buffer(read_only=True)),
])
render_bind = goldy.BindGroup(device, render_layout, [
    goldy.BufferBinding(0, buffer),
])

# Run compute, then render
compute_encoder = goldy.ComputeEncoder()
# ... dispatch compute ...
compute_encoder.dispatch(device)

# Now use the results in rendering
render_encoder = goldy.CommandEncoder()
with render_encoder.begin_render_pass() as rp:
    rp.set_pipeline(render_pipeline)
    rp.set_bind_group(0, render_bind)  # Read compute results
    rp.draw(range(3))
target.render(render_encoder)
```


