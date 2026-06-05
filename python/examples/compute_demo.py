#!/usr/bin/env python3
"""Compute Demo - GPU compute shader example.

Demonstrates using compute shaders to process data on the GPU.
This example doubles all values in a buffer using a compute shader.

Usage:
    python compute_demo.py
"""

import goldy
import numpy as np


# Simple compute shader that doubles each value
COMPUTE_SHADER = """
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<float> data, ThreadId id) {
    data[id.x] = data[id.x] * 2.0;
}
"""


def main():
    print("Goldy Compute Demo")
    print("=" * 40)
    
    # Create device
    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
    
    print(f"Backend: {instance.backend_type}")
    print()
    
    # Create input data
    input_data = np.arange(256, dtype=np.float32)
    print(f"Input data (first 10): {input_data[:10]}")
    
    # Create GPU storage buffer
    buffer = goldy.Buffer(device, input_data, goldy.BufferKind.SCATTERED)
    print(f"Created buffer: {buffer.size} bytes")
    
    # Compile compute shader
    shader = goldy.ShaderModule.from_slang(device, COMPUTE_SHADER)
    print("Compiled compute shader")
    
    # Create compute pipeline
    pipeline = goldy.ComputePipeline(device, shader)
    print("Created compute pipeline")
    
    # Dispatch compute work
    encoder = goldy.ComputeEncoder()
    with encoder.begin_compute_pass() as cp:
        cp.set_pipeline(pipeline)
        cp.bind_resources([buffer])
        # 256 elements / 64 threads per workgroup = 4 workgroups
        cp.dispatch(4, 1, 1)
    
    encoder.dispatch(device)
    print("Dispatched compute shader")
    
    # Note: To read back results, you'd need a staging buffer with COPY_SRC
    # and use a copy command. For this demo, we just show the dispatch works.
    
    print()
    print("Compute shader executed successfully!")
    print("(Values in the buffer are now doubled)")


if __name__ == '__main__':
    main()
