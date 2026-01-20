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
buffer = goldy.Buffer(device, data, goldy.DataAccess.SCATTERED)

# Define compute shader (Slang)
SHADER = """
#include "goldy_exp.slang"

struct PushConstants { uint buffer_idx; };
[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    // Read, double, and write back
    float val = asfloat(g_StorageBuffers[pc.buffer_idx].Load(id.x * 4));
    g_StorageBuffers[pc.buffer_idx].Store(id.x * 4, asuint(val * 2.0));
}
"""

# Create compute pipeline
shader = goldy.ShaderModule.from_slang(device, SHADER)
pipeline = goldy.ComputePipeline(device, shader)

# Dispatch compute work
encoder = goldy.ComputeEncoder()
with encoder.begin_compute_pass() as cp:
    cp.set_pipeline(pipeline)
    cp.set_push_constants([buffer])  # Buffer index passed via push constants
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
buffer_a = goldy.Buffer(device, initial_data, goldy.DataAccess.SCATTERED)
buffer_b = goldy.Buffer(device, initial_data, goldy.DataAccess.SCATTERED)

# Iterate, swapping buffers each step
use_a = True
for iteration in range(100):
    encoder = goldy.ComputeEncoder()
    with encoder.begin_compute_pass() as cp:
        cp.set_pipeline(pipeline)
        if use_a:
            cp.set_push_constants([buffer_a, buffer_b])  # Read A, write B
        else:
            cp.set_push_constants([buffer_b, buffer_a])  # Read B, write A
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
#include "goldy_exp.slang"

static const uint SIZE = {GRID_SIZE};

struct PushConstants {{ uint current_idx; uint next_idx; }};
[[vk::push_constant]] PushConstants pc;

uint getCell(int x, int y) {{
    x = (x + SIZE) % SIZE;
    y = (y + SIZE) % SIZE;
    return g_StorageBuffers[pc.current_idx].Load((y * SIZE + x) * 4);
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
    uint cell = g_StorageBuffers[pc.current_idx].Load(idx * 4);
    uint neighbors = countNeighbors(int(id.x), int(id.y));
    
    // Conway's rules
    uint next = (cell == 1) ? 
        ((neighbors == 2 || neighbors == 3) ? 1 : 0) :
        ((neighbors == 3) ? 1 : 0);
    
    g_StorageBuffers[pc.next_idx].Store(idx * 4, next);
}}
"""

# Setup and run simulation...
# See python/examples/game_of_life.py for full example
```

## Combining Compute and Graphics

Use compute results in render passes via shared storage buffers:

```python
# Create storage buffer accessible from both compute and fragment shaders
buffer = goldy.Buffer(device, data, goldy.DataAccess.SCATTERED)

# Run compute
compute_encoder = goldy.ComputeEncoder()
with compute_encoder.begin_compute_pass() as cp:
    cp.set_pipeline(compute_pipeline)
    cp.set_push_constants([buffer])
    cp.dispatch(workgroups, 1, 1)
compute_encoder.dispatch(device)

# Now use the results in rendering
render_encoder = goldy.CommandEncoder()
with render_encoder.begin_render_pass() as rp:
    rp.set_pipeline(render_pipeline)
    rp.set_push_constants([buffer])  # Read compute results
    rp.draw(range(3))
target.render(render_encoder)
```

