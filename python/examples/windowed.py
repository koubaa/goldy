#!/usr/bin/env python3
"""Interactive Windowed Example - Real-time rendering with GLFW.

This example demonstrates:
1. Creating a window with GLFW
2. Using Surface for zero-copy presentation
3. Real-time render loop with input handling

Usage:
    pip install glfw
    python windowed.py

Controls:
    - ESC: Close window
    - Space: Toggle animation
    - R/G/B: Change clear color
"""

import goldy
import numpy as np
import time
import math

try:
    import glfw
except ImportError:
    print("This example requires GLFW. Install with: pip install glfw")
    print("You also need GLFW libraries installed on your system.")
    exit(1)


def main():
    print("Goldy Interactive Windowed Example")
    print("=" * 50)
    print("Controls: ESC=quit, Space=toggle animation, R/G/B=color")
    print()

    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # Configure window - NO_API means no OpenGL context (we use Vulkan)
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, True)

    # Create window
    width, height = 800, 600
    window = glfw.create_window(width, height, "Goldy - Press ESC to exit", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    # Create Goldy device and surface
    instance = goldy.Instance()
    device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
    print(f"Backend: {instance.backend_type}")

    surface = goldy.Surface.from_glfw(device, window)
    print(f"Surface: {surface.width}x{surface.height}")

    # Create vertex buffer with a triangle
    vertices = np.array([
        # Position      Color (RGBA)
         0.0, -0.5,    1.0, 0.0, 0.0, 1.0,  # Top (red)
        -0.5,  0.5,    0.0, 1.0, 0.0, 1.0,  # Bottom-left (green)
         0.5,  0.5,    0.0, 0.0, 1.0, 1.0,  # Bottom-right (blue)
    ], dtype=np.float32)
    vertex_buffer = goldy.Buffer(device, vertices, goldy.BufferUsage.VERTEX)

    # Create shader and pipeline
    shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
    pipeline = goldy.RenderPipeline(
        device, shader, shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.vertex_2d(),
            target_format=surface.format,
        )
    )

    # Animation state
    animate = True
    clear_color = goldy.Color(0.1, 0.1, 0.2, 1.0)
    frame_count = 0
    start_time = time.time()

    # Handle window resize
    def on_resize(win, w, h):
        nonlocal width, height
        if w > 0 and h > 0:
            width, height = w, h
            surface.resize(w, h)

    glfw.set_framebuffer_size_callback(window, on_resize)

    # Handle key input
    def on_key(win, key, scancode, action, mods):
        nonlocal animate, clear_color
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE:
                animate = not animate
                print(f"Animation: {'ON' if animate else 'OFF'}")
            elif key == glfw.KEY_R:
                clear_color = goldy.Color(0.3, 0.1, 0.1, 1.0)
            elif key == glfw.KEY_G:
                clear_color = goldy.Color(0.1, 0.3, 0.1, 1.0)
            elif key == glfw.KEY_B:
                clear_color = goldy.Color(0.1, 0.1, 0.3, 1.0)

    glfw.set_key_callback(window, on_key)

    print("\nRendering... (press ESC to exit)")

    # Main render loop
    while not glfw.window_should_close(window):
        # Poll events
        glfw.poll_events()

        # Animate vertices
        if animate:
            t = time.time() - start_time
            # Rotate triangle
            angle = t * 0.5
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            # Update vertex positions (rotate around center)
            new_vertices = np.array([
                # Rotated positions        Colors stay the same
                 sin_a * 0.5,  -cos_a * 0.5,    1.0, 0.0, 0.0, 1.0,
                -cos_a * 0.5 - sin_a * 0.5,  sin_a * 0.5 - cos_a * 0.5,    0.0, 1.0, 0.0, 1.0,
                 cos_a * 0.5 - sin_a * 0.5, -sin_a * 0.5 - cos_a * 0.5,    0.0, 0.0, 1.0, 1.0,
            ], dtype=np.float32)
            vertex_buffer.write(0, new_vertices)

        # Acquire frame
        frame = surface.acquire()

        # Record commands
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(clear_color)
            rp.set_pipeline(pipeline)
            rp.set_vertex_buffer(0, vertex_buffer)
            rp.draw(range(3))

        # Render and present
        frame.render(encoder)
        surface.present(frame)

        frame_count += 1

    # Cleanup
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\nRendered {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

    glfw.terminate()
    print("Done!")


if __name__ == '__main__':
    main()


