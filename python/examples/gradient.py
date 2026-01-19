#!/usr/bin/env python3
"""Gradient example - animated color gradient.

Demonstrates fragment shader with time-based animation using the Surface API.

Usage:
    python gradient.py
"""

import goldy
import numpy as np
import time
import os

import glfw


def load_shader(name):
    """Load shader from shared shaders directory."""
    shader_dir = os.path.join(os.path.dirname(__file__), "..", "..", "shaders")
    shader_path = os.path.join(shader_dir, name)
    with open(shader_path, "r") as f:
        return f.read()


def create_fullscreen_quad(time_offset):
    """Create fullscreen quad vertices with animated UV offset."""
    offset = time_offset * 0.1
    return np.array([
        # Position      UV
        -1.0, -1.0,    0.0 + offset, 1.0,
         1.0, -1.0,    1.0 + offset, 1.0,
         1.0,  1.0,    1.0 + offset, 0.0,
        -1.0, -1.0,    0.0 + offset, 1.0,
         1.0,  1.0,    1.0 + offset, 0.0,
        -1.0,  1.0,    0.0 + offset, 0.0,
    ], dtype=np.float32)


def main():
    print("Goldy Gradient Example - Press Escape to exit")

    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # Configure window
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, True)

    # Create window
    width, height = 800, 600
    window = glfw.create_window(width, height, "Goldy - Animated Gradient (Surface API)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    # Create Goldy device and surface
    instance = goldy.Instance()
    device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)

    surface = goldy.Surface.from_glfw(device, window)

    # Load and compile shader
    gradient_shader_src = load_shader("gradient.slang")
    shader = goldy.ShaderModule.from_slang(device, gradient_shader_src)

    # Create pipeline with custom vertex layout (position + uv)
    pipeline = goldy.RenderPipeline(
        device, shader, shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.vertex_2d_uv(),
            target_format=surface.format,
        )
    )

    # Create initial vertex buffer
    vertices = create_fullscreen_quad(0.0)
    vertex_buffer = goldy.Buffer(device, vertices, goldy.BufferUsage.VERTEX)

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
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, on_key)

    # Main render loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Update vertices with animated UV offset
        t = time.time() - start_time
        vertices = create_fullscreen_quad(t)
        vertex_buffer.write(0, vertices)

        # Acquire frame
        frame = surface.acquire()

        # Record commands
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(goldy.Color.BLACK)
            rp.set_pipeline(pipeline)
            rp.set_vertex_buffer(0, vertex_buffer)
            rp.draw(range(6))

        # Render and present
        frame.render(encoder)
        surface.present(frame)

    glfw.terminate()


if __name__ == '__main__':
    main()
