#!/usr/bin/env python3
"""Hello Triangle - Basic Goldy example.

This example renders a colored triangle to an image file.

Usage:
    python hello_triangle.py
"""

import goldy
import numpy as np

# Try to import PIL for saving the image
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Note: Install Pillow to save output as PNG: pip install pillow")


def main():
    print("Goldy Hello Triangle Example")
    print("=" * 40)
    
    # 1. Create instance and device
    instance = goldy.Instance()
    print(f"Backend: {instance.backend_type}")
    
    # List available adapters
    adapters = instance.enumerate_adapters()
    print(f"Found {len(adapters)} GPU adapter(s):")
    for adapter in adapters:
        print(f"  - {adapter.name} ({adapter.vendor})")
    
    # Create device on preferred GPU
    device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
    print(f"Using adapter ID: {device.adapter_id}")
    print()
    
    # 2. Create vertex buffer with a triangle
    # Each vertex: x, y, r, g, b, a (Vertex2D layout)
    vertices = np.array([
        # Position      Color (RGBA)
         0.0, -0.5,    1.0, 0.0, 0.0, 1.0,  # Top vertex (red)
        -0.5,  0.5,    0.0, 1.0, 0.0, 1.0,  # Bottom-left (green)
         0.5,  0.5,    0.0, 0.0, 1.0, 1.0,  # Bottom-right (blue)
    ], dtype=np.float32)
    
    vertex_buffer = goldy.Buffer(device, vertices, goldy.BufferUsage.VERTEX)
    print(f"Created vertex buffer: {vertex_buffer.size} bytes")
    
    # 3. Create shader and pipeline
    shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
    print("Compiled shader")
    
    pipeline = goldy.RenderPipeline(
        device, shader, shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.vertex_2d(),
            target_format=goldy.TextureFormat.RGBA8_UNORM,
        )
    )
    print("Created render pipeline")
    
    # 4. Create render target
    width, height = 800, 600
    target = goldy.RenderTarget(device, width, height, goldy.TextureFormat.RGBA8_UNORM)
    print(f"Created render target: {width}x{height}")
    print()
    
    # 5. Render the triangle
    print("Rendering...")
    encoder = goldy.CommandEncoder()
    with encoder.begin_render_pass() as rp:
        # Clear to dark blue
        rp.clear(goldy.Color(0.1, 0.1, 0.2, 1.0))
        # Draw the triangle
        rp.set_pipeline(pipeline)
        rp.set_vertex_buffer(0, vertex_buffer)
        rp.draw(range(3))  # 3 vertices = 1 triangle
    
    target.render(encoder)
    print("Render complete!")
    
    # 6. Read back and save
    pixels = target.read_to_cpu()
    print(f"Read {pixels.shape} pixels from GPU")
    
    if HAS_PIL:
        # Save as PNG
        img = Image.fromarray(pixels, mode='RGBA')
        img.save('hello_triangle.png')
        print("Saved: hello_triangle.png")
    else:
        # Just print some pixel values
        print(f"Center pixel: {pixels[height//2, width//2]}")
        print(f"Top-left pixel: {pixels[0, 0]}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

