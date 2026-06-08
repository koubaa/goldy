#!/usr/bin/env python3
"""Triangle example — animated colored triangle in a window via TaskGraph.

Offscreen RenderTarget -> render_pass -> copy_render_target_to_swapchain -> present.

Usage:
    python triangle_window.py
"""

from __future__ import annotations

import math
import sys

import glfw
import goldy
import numpy as np


def make_scene_rt(device: goldy.Device, surface: goldy.Surface) -> goldy.RenderTarget:
    width = max(surface.width, 1)
    height = max(surface.height, 1)
    return goldy.RenderTarget(device, width, height, surface.format)


def main() -> int:
    print("Goldy Python Triangle Window (TaskGraph)")
    print("=" * 40)
    print("Press Escape or close the window to exit\n")

    if not glfw.init():
        print("Failed to initialize GLFW", file=sys.stderr)
        return 1

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    window = glfw.create_window(800, 600, "Goldy - Animated Triangle (Python)", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window", file=sys.stderr)
        return 1

    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
    surface = goldy.Surface.from_glfw(device, window)

    shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
    pipeline = goldy.RenderPipeline(
        device,
        shader,
        shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.vertex_2d(),
            target_format=surface.format,
        ),
    )

    vertices = np.array(
        [
            0.0,
            -0.5,
            1.0,
            0.0,
            0.0,
            1.0,
            -0.5,
            0.5,
            0.0,
            1.0,
            0.0,
            1.0,
            0.5,
            0.5,
            0.0,
            0.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )
    vertex_buffer = goldy.Buffer(device, vertices, goldy.BufferKind.SCATTERED)

    scene_rt = make_scene_rt(device, surface)
    frame_graph = goldy.TaskGraph()
    frame_count = 0

    try:
        while not glfw.window_should_close(window):
            fb_width, fb_height = glfw.get_framebuffer_size(window)
            if fb_width > 0 and fb_height > 0:
                if fb_width != surface.width or fb_height != surface.height:
                    surface.resize(fb_width, fb_height)
                    scene_rt = make_scene_rt(device, surface)

            t = math.sin(frame_count * 0.02) * 0.5 + 0.5
            bg = goldy.Color(0.1 + t * 0.1, 0.1 + t * 0.05, 0.2 + t * 0.1, 1.0)

            frame_graph.clear()
            with frame_graph.render_pass("triangle", scene_rt) as rp:
                (
                    rp.bind_buffer(vertex_buffer, goldy.NodeAccess.READ)
                    .clear(bg)
                    .set_pipeline(pipeline)
                    .set_vertex_buffer(0, vertex_buffer)
                    .draw(range(3))
                )

            swapchain = frame_graph.declare_swapchain_output()
            frame_graph.copy_render_target_to_swapchain(scene_rt, swapchain)

            frame = surface.acquire()
            surface.submit_graph_to_frame(frame_graph, frame)
            surface.present(frame)

            frame_count += 1
            glfw.poll_events()

            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
    finally:
        glfw.destroy_window(window)
        glfw.terminate()

    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
