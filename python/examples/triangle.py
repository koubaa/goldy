#!/usr/bin/env python3
"""Triangle example — animated colored triangle in a window via retained Scheme.

Offscreen render pass → copy_to_present → grant_present.

Requires: pip install glfw

Usage:
    python triangle.py
"""

from __future__ import annotations

import sys

import glfw
import goldy
import numpy as np


def record_scheme(
    scheme: goldy.Scheme,
    pipeline: goldy.RenderPipeline,
    vertex_parcel: goldy.Parcel,
    scene_rt: goldy.SchemeRenderTargetLease,
    screen: goldy.PresentLease,
    bg: goldy.Color,
) -> goldy.PresentGrant:
    with scheme.render_pass("triangle", scene_rt) as rp:
        (
            rp.bind_parcel(vertex_parcel, goldy.NodeAccess.READ)
            .clear(bg)
            .set_pipeline(pipeline)
            .set_vertex_buffer_parcel(0, vertex_parcel)
            .draw(range(3))
        )
    scheme.copy_to_present(scene_rt, screen)
    return scheme.grant_present(screen)


def main() -> int:
    print("Goldy Python Triangle Window (Scheme + Present)")
    print("=" * 40)
    print("Press Escape or close the window to exit\n")

    if not glfw.init():
        print("Failed to initialize GLFW", file=sys.stderr)
        return 1

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    window = glfw.create_window(800, 600, "Goldy - Triangle (Python / Scheme)", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window", file=sys.stderr)
        return 1

    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
    ctx = device.create_context()
    swapchain = goldy.SwapchainPool.from_glfw(ctx, window)
    screen = swapchain.lease()

    shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
    pipeline = goldy.RenderPipeline(
        device,
        shader,
        shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.vertex_2d(),
            target_format=swapchain.format,
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
    retained_pool = goldy.RetainedPool(device)
    vertex_parcel = retained_pool.acquire_buffer(vertices, goldy.BufferKind.SCATTERED)

    scheme = goldy.Scheme(ctx)
    scene_rt = scheme.lease_render_target(
        max(swapchain.width, 1),
        max(swapchain.height, 1),
        swapchain.format,
    )
    bg = goldy.Color(0.1, 0.1, 0.2, 1.0)
    present = record_scheme(scheme, pipeline, vertex_parcel, scene_rt, screen, bg)

    frame_count = 0

    try:
        while not glfw.window_should_close(window):
            fb_width, fb_height = glfw.get_framebuffer_size(window)
            if fb_width > 0 and fb_height > 0:
                if fb_width != swapchain.width or fb_height != swapchain.height:
                    swapchain.resize(fb_width, fb_height)
                    pipeline = goldy.RenderPipeline(
                        device,
                        shader,
                        shader,
                        goldy.RenderPipelineDesc(
                            vertex_layout=goldy.VertexBufferLayout.vertex_2d(),
                            target_format=swapchain.format,
                        ),
                    )
                    scheme = goldy.Scheme(ctx)
                    scene_rt = scheme.lease_render_target(
                        max(swapchain.width, 1),
                        max(swapchain.height, 1),
                        swapchain.format,
                    )
                    present = record_scheme(
                        scheme, pipeline, vertex_parcel, scene_rt, screen, bg
                    )

            submission = scheme.submit()
            present.consume(submission)

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
