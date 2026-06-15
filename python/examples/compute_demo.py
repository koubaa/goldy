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
    
    retained_pool = goldy.RetainedPool(device)
    parcel = retained_pool.acquire_buffer(input_data, goldy.BufferKind.SCATTERED)
    print(f"Created parcel: {parcel.byte_size} bytes")
    
    # Compile compute shader
    shader = goldy.ShaderModule.from_slang(device, COMPUTE_SHADER)
    print("Compiled compute shader")
    
    # Create compute pipeline
    pipeline = goldy.ComputePipeline(device, shader)
    print("Created compute pipeline")
    
    ctx = device.create_context()
    scheme = goldy.Scheme(ctx)
    scheme.node("double", pipeline).declare_parcel(
        parcel, goldy.NodeAccess.READ_WRITE, goldy.ResourceAccess.WRITE
    ).dispatch(4, 1, 1)
    grant = scheme.grant_read(parcel)
    frame = scheme.submit()
    output = np.frombuffer(grant.consume(frame), dtype=np.float32)
    print("Dispatched compute shader")

    expected = input_data * 2.0
    if not np.allclose(output, expected):
        raise RuntimeError("compute output mismatch")
    print(f"Output data (first 10): {output[:10]}")
    print()
    print("Compute shader executed successfully!")
    print("(Values in the buffer are now doubled)")


if __name__ == '__main__':
    main()
