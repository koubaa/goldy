#!/usr/bin/env python3
"""Gradient - Render a fullscreen gradient using a custom shader.

This example demonstrates:
- Writing custom Slang shaders
- Using the goldy_exp shader library
- Fullscreen quad rendering

Usage:
    python gradient.py
"""

import goldy
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# Custom gradient shader using goldy_exp library
GRADIENT_SHADER = """
import goldy_exp;

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    // Create a rainbow gradient based on horizontal position
    float3 color = rainbow(input.uv.x);
    
    // Add some vertical variation
    color *= 0.8 + 0.2 * sin(input.uv.y * 3.14159);
    
    return float4(color, 1.0);
}
"""


def main():
    print("Goldy Gradient Example")
    print("=" * 40)
    
    # Create device
    instance = goldy.Instance()
    device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
    
    # Check that goldy_exp library is available
    print(f"Available shader libraries: {device.list_libraries()}")
    
    # Create fullscreen quad vertices (position + uv)
    # Using Vertex2DUv layout: x, y, u, v
    vertices = np.array([
        # Triangle 1
        -1.0, -1.0,  0.0, 1.0,  # Bottom-left
         1.0, -1.0,  1.0, 1.0,  # Bottom-right
         1.0,  1.0,  1.0, 0.0,  # Top-right
        # Triangle 2
        -1.0, -1.0,  0.0, 1.0,  # Bottom-left
         1.0,  1.0,  1.0, 0.0,  # Top-right
        -1.0,  1.0,  0.0, 0.0,  # Top-left
    ], dtype=np.float32)
    
    vertex_buffer = goldy.Buffer(device, vertices, goldy.BufferUsage.VERTEX)
    
    # Compile custom shader
    shader = goldy.ShaderModule.from_slang(device, GRADIENT_SHADER)
    print("Compiled gradient shader")
    
    # Create pipeline with Vertex2DUv layout
    pipeline = goldy.RenderPipeline(
        device, shader, shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.vertex_2d_uv(),
            target_format=goldy.TextureFormat.RGBA8_UNORM,
        )
    )
    
    # Render
    width, height = 1920, 1080
    target = goldy.RenderTarget(device, width, height)
    
    encoder = goldy.CommandEncoder()
    with encoder.begin_render_pass() as rp:
        rp.clear(goldy.Color.BLACK)
        rp.set_pipeline(pipeline)
        rp.set_vertex_buffer(0, vertex_buffer)
        rp.draw(range(6))  # 6 vertices = 2 triangles = fullscreen quad
    
    target.render(encoder)
    
    # Save
    pixels = target.read_to_cpu()
    print(f"Rendered {width}x{height} gradient")
    
    if HAS_PIL:
        img = Image.fromarray(pixels, mode='RGBA')
        img.save('gradient.png')
        print("Saved: gradient.png")
    
    print("Done!")


if __name__ == '__main__':
    main()

