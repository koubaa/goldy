#!/usr/bin/env python3
"""Triangle example - render a colored triangle in an interactive window.

This example demonstrates the Surface API for zero-copy GPU presentation.

Usage:
    python windowed.py
"""

import goldy
import numpy as np
import math

import glfw


def main():
    print("Goldy Surface API Example")
    print("=" * 40)
    print("Rendering triangle with zero-copy GPU presentation")
    print("Press Escape or close window to exit")
    print()

    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # Configure window - NO_API means no OpenGL context (we use DX12/Vulkan/Metal)
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, True)

    # Create window
    width, height = 800, 600
    window = glfw.create_window(width, height, "Goldy - Animated Triangle (Surface API)", None, None)
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

    # Create shader and pipeline using surface's actual format
    shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
    pipeline = goldy.RenderPipeline(
        device, shader, shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.vertex_2d(),
            target_format=surface.format,
        )
    )

    # Animation state
    frame_count = 0

    # Handle window resize
    def on_resize(win, w, h):
        nonlocal width, height
        if w > 0 and h > 0:
            width, height = w, h
            surface.resize(w, h)

    glfw.set_framebuffer_size_callback(window, on_resize)

    # Handle key input
    def on_key(win, key, scancode, action, mods):
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, on_key)

    print("\nRendering...")

    # Main render loop
    while not glfw.window_should_close(window):
        # Poll events
        glfw.poll_events()

        # Animate background color
        t = math.sin(frame_count * 0.02) * 0.5 + 0.5
        bg_color = goldy.Color(
            0.1 + t * 0.1,
            0.1 + t * 0.05,
            0.2 + t * 0.1,
            1.0
        )

        # Acquire next frame from swapchain
        frame = surface.acquire()

        # Build render commands
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(bg_color)
            rp.set_pipeline(pipeline)
            rp.set_vertex_buffer(0, vertex_buffer)
            rp.draw(range(3))

        # Render to swapchain image (zero-copy - no CPU readback!)
        frame.render(encoder)

        # Present to screen
        surface.present(frame)

        frame_count += 1

    glfw.terminate()
    print("Done!")


if __name__ == '__main__':
    main()
