#!/usr/bin/env python3
"""Conway's Game of Life — hybrid TaskGraph in a window (compute + render + blit).

Requires: pip install glfw

Usage:
    python game_of_life.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import glfw
import goldy
import numpy as np

GRID_WIDTH = 128
GRID_HEIGHT = 128
CELL_COUNT = GRID_WIDTH * GRID_HEIGHT
WORKGROUPS_X = (GRID_WIDTH + 7) // 8
WORKGROUPS_Y = (GRID_HEIGHT + 7) // 8

SHADERS_DIR = Path(__file__).resolve().parents[2] / "shaders"


def demo_frame_limit() -> int | None:
    raw = os.environ.get("GOLDY_DEMO_FRAMES", "").strip()
    if not raw:
        return None
    return max(1, int(raw))


def create_initial_state() -> np.ndarray:
    cells = np.zeros(CELL_COUNT, dtype=np.uint32)

    gun = [
        (1, 5),
        (1, 6),
        (2, 5),
        (2, 6),
        (11, 5),
        (11, 6),
        (11, 7),
        (12, 4),
        (12, 8),
        (13, 3),
        (13, 9),
        (14, 3),
        (14, 9),
        (15, 6),
        (16, 4),
        (16, 8),
        (17, 5),
        (17, 6),
        (17, 7),
        (18, 6),
        (21, 3),
        (21, 4),
        (21, 5),
        (22, 3),
        (22, 4),
        (22, 5),
        (23, 2),
        (23, 6),
        (25, 1),
        (25, 2),
        (25, 6),
        (25, 7),
        (35, 3),
        (35, 4),
        (36, 3),
        (36, 4),
    ]
    offset_x, offset_y = 10, 10
    for x, y in gun:
        px, py = x + offset_x, y + offset_y
        if px < GRID_WIDTH and py < GRID_HEIGHT:
            cells[py * GRID_WIDTH + px] = 1

    rng = 42
    for y in range(60, 100):
        for x in range(60, 100):
            rng = (rng * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
            if (rng >> 32) % 4 == 0:
                cells[y * GRID_WIDTH + x] = 1

    return cells


def make_scene_rt(device: goldy.Device, surface: goldy.Surface) -> goldy.RenderTarget:
    width = max(surface.width, 1)
    height = max(surface.height, 1)
    return goldy.RenderTarget(device, width, height, surface.format)


def main() -> int:
    print("Goldy Python Game of Life (TaskGraph)")
    print("=" * 40)
    print("Press Escape or close the window to exit\n")

    if not glfw.init():
        print("Failed to initialize GLFW", file=sys.stderr)
        return 1

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    window = glfw.create_window(800, 800, "Goldy - Game of Life (Python)", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window", file=sys.stderr)
        return 1

    compute_src = (SHADERS_DIR / "game_of_life.slang").read_text(encoding="utf-8")
    render_src = (SHADERS_DIR / "game_of_life_render.slang").read_text(encoding="utf-8")

    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
    surface = goldy.Surface.from_glfw(device, window)

    initial = create_initial_state()
    zeros = np.zeros(CELL_COUNT, dtype=np.uint32)
    buf_a = goldy.Buffer(device, initial, goldy.BufferKind.SCATTERED)
    buf_b = goldy.Buffer(device, zeros, goldy.BufferKind.SCATTERED)

    compute_shader = goldy.ShaderModule.from_slang(device, compute_src)
    render_shader = goldy.ShaderModule.from_slang(device, render_src)
    compute_pipeline = goldy.ComputePipeline(device, compute_shader)
    render_pipeline = goldy.RenderPipeline(
        device,
        render_shader,
        render_shader,
        goldy.RenderPipelineDesc(
            target_format=surface.format,
            topology=goldy.PrimitiveTopology.TRIANGLE_LIST,
        ),
    )

    scene_rt = make_scene_rt(device, surface)
    frame_graph = goldy.TaskGraph()
    use_buffer_a = True
    frame_count = 0
    last_update = time.monotonic()
    frame_limit = demo_frame_limit()
    start = time.monotonic()

    try:
        while not glfw.window_should_close(window):
            fb_width, fb_height = glfw.get_framebuffer_size(window)
            if fb_width > 0 and fb_height > 0:
                if fb_width != surface.width or fb_height != surface.height:
                    surface.resize(fb_width, fb_height)
                    scene_rt = make_scene_rt(device, surface)

            now = time.monotonic()
            should_update = (now - last_update) > 0.033

            frame_graph.clear()

            if should_update:
                last_update = now
                read_buf, write_buf = (buf_a, buf_b) if use_buffer_a else (buf_b, buf_a)
                read_idx = read_buf.resource_index(goldy.ResourceAccess.READ)
                write_idx = write_buf.resource_index(goldy.ResourceAccess.WRITE)
                with frame_graph.compute_node(
                    "game_of_life",
                    compute_pipeline,
                    workgroups=(WORKGROUPS_X, WORKGROUPS_Y, 1),
                ) as node:
                    (
                        node.bind_buffer(read_buf, goldy.NodeAccess.READ)
                        .bind_buffer(write_buf, goldy.NodeAccess.WRITE)
                        .bind_resources_raw([read_idx, write_idx])
                    )
                use_buffer_a = not use_buffer_a

            current_buf = buf_a if use_buffer_a else buf_b
            cells_idx = current_buf.resource_index(goldy.ResourceAccess.READ)

            with frame_graph.render_pass("game_of_life_render", scene_rt) as rp:
                (
                    rp.bind_buffer(current_buf, goldy.NodeAccess.READ)
                    .clear(goldy.Color.BLACK)
                    .set_pipeline(render_pipeline)
                    .bind_resource_index(cells_idx)
                    .draw_fullscreen()
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
            if frame_limit is not None and frame_count >= frame_limit:
                glfw.set_window_should_close(window, True)
    finally:
        elapsed = time.monotonic() - start
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        print(f"GOLDY_PERF: frames={frame_count} elapsed={elapsed:.2f}s avg_fps={fps:.1f}")
        glfw.destroy_window(window)
        glfw.terminate()

    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
