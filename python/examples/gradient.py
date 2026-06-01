#!/usr/bin/env python3
"""Gradient example - animated color gradient.

Uses vertex-less fullscreen triangle rendering (no vertex buffer needed).

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
    window = glfw.create_window(width, height, "Goldy - Animated Gradient", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    # Create Goldy device and surface
    instance = goldy.Instance()
    device = instance.request_adapter().request_device()

    surface = goldy.Surface.from_glfw(device, window)

    # Create uniform buffer for time
    uniform_buffer = goldy.Buffer.empty(
        device, 4, goldy.DataAccess.BROADCAST
    )

    # Load and compile shader from shared shaders directory
    gradient_shader_src = load_shader("gradient.slang")
    shader = goldy.ShaderModule.from_slang(device, gradient_shader_src)

    # Create pipeline - vertex-less (no vertex buffer needed)
    pipeline = goldy.RenderPipeline(
        device, shader, shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.empty(),
            target_format=surface.format,
        )
    )

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

        # Update uniform buffer with current time
        t = time.time() - start_time
        time_data = np.array([t], dtype=np.float32)
        uniform_buffer.write(0, time_data)

        # Acquire frame
        frame = surface.acquire()

        # Record commands
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(goldy.Color.BLACK)
            rp.set_pipeline(pipeline)
            # Bind resource slots
            rp.bind_resources([uniform_buffer])
            # Vertex-less fullscreen triangle: 3 vertices, no vertex buffer
            rp.draw(range(3))

        # Render and present
        frame.render(encoder)
        surface.present(frame)

    glfw.terminate()


if __name__ == '__main__':
    main()
