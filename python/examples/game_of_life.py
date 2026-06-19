#!/usr/bin/env python3
"""Conway's Game of Life — hybrid Scheme in a window (compute + render + present).

Ping-pong cell grids live in one retained record buffer (fields `"a"` / `"b"`).
Each simulation step runs an ephemeral compute scheme; the display scheme is
rebuilt when the active field flips.

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


def run_compute_step(
    ctx: goldy.Context,
    cells: goldy.Buffer,
    read_field: str,
    write_field: str,
    pipeline: goldy.ComputePipeline,
) -> None:
    scheme = goldy.Scheme(ctx)
    node = scheme.node("game_of_life", pipeline)
    (
        node.with_field(cells, read_field, goldy.NodeAccess.READ, goldy.ResourceAccess.READ_WRITE)
        .with_field(cells, write_field, goldy.NodeAccess.WRITE, goldy.ResourceAccess.WRITE)
        .dispatch(WORKGROUPS_X, WORKGROUPS_Y, 1)
    )
    scheme.submit()


def record_display_scheme(
    scheme: goldy.Scheme,
    cells: goldy.Buffer,
    current_field: str,
    render_pipeline: goldy.RenderPipeline,
    scene_rt: goldy.SchemeRenderTargetLease,
    screen: goldy.PresentLease,
) -> goldy.PresentGrant:
    with scheme.render_pass("game_of_life_render", scene_rt) as rp:
        (
            rp.with_field(cells, current_field, goldy.NodeAccess.READ)
            .clear(goldy.Color.BLACK)
            .set_pipeline(render_pipeline)
            .draw_fullscreen()
        )
    scheme.copy_to_present(scene_rt, screen)
    return scheme.grant_present(screen)


def rebuild_display_scheme(
    ctx: goldy.Context,
    swapchain: goldy.SwapchainPool,
    cells: goldy.Buffer,
    current_field: str,
    render_pipeline: goldy.RenderPipeline,
    screen: goldy.PresentLease,
) -> tuple[goldy.Scheme, goldy.SchemeRenderTargetLease, goldy.PresentGrant]:
    display_scheme = goldy.Scheme(ctx)
    scene_rt = display_scheme.lease_render_target(
        max(swapchain.width, 1),
        max(swapchain.height, 1),
        swapchain.format,
    )
    present = record_display_scheme(
        display_scheme, cells, current_field, render_pipeline, scene_rt, screen
    )
    return display_scheme, scene_rt, present


def main() -> int:
    print("Goldy Python Game of Life (Scheme + Present)")
    print("=" * 40)
    print("Press Escape or close the window to exit\n")

    if not glfw.init():
        print("Failed to initialize GLFW", file=sys.stderr)
        return 1

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    window = glfw.create_window(800, 800, "Goldy - Game of Life (Python / Scheme)", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window", file=sys.stderr)
        return 1

    compute_src = (SHADERS_DIR / "game_of_life.slang").read_text(encoding="utf-8")
    render_src = (SHADERS_DIR / "game_of_life_render.slang").read_text(encoding="utf-8")

    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
    ctx = device.create_context()
    swapchain = goldy.SwapchainPool.from_glfw(ctx, window)
    screen = swapchain.lease()

    initial = create_initial_state()
    retained_pool = goldy.RetainedPool(device)
    record = retained_pool.acquire_record()
    record.emplace_field("a", initial)
    record.emplace_field("b", initial)
    cells = record.build(retained_pool)

    compute_shader = goldy.ShaderModule.from_slang(device, compute_src)
    render_shader = goldy.ShaderModule.from_slang(device, render_src)
    compute_pipeline = goldy.ComputePipeline(device, compute_shader)
    render_pipeline = goldy.RenderPipeline(
        device,
        render_shader,
        render_shader,
        goldy.RenderPipelineDesc(
            target_format=swapchain.format,
            topology=goldy.PrimitiveTopology.TRIANGLE_LIST,
        ),
    )

    display_scheme = goldy.Scheme(ctx)
    scene_rt = display_scheme.lease_render_target(
        max(swapchain.width, 1),
        max(swapchain.height, 1),
        swapchain.format,
    )
    present = record_display_scheme(
        display_scheme, cells, "a", render_pipeline, scene_rt, screen
    )

    use_buffer_a = True
    frame_count = 0
    last_update = time.monotonic()
    frame_limit = demo_frame_limit()
    start = time.monotonic()

    try:
        while not glfw.window_should_close(window):
            fb_width, fb_height = glfw.get_framebuffer_size(window)
            if fb_width > 0 and fb_height > 0:
                if fb_width != swapchain.width or fb_height != swapchain.height:
                    swapchain.resize(fb_width, fb_height)
                    render_pipeline = goldy.RenderPipeline(
                        device,
                        render_shader,
                        render_shader,
                        goldy.RenderPipelineDesc(
                            target_format=swapchain.format,
                            topology=goldy.PrimitiveTopology.TRIANGLE_LIST,
                        ),
                    )
                    current_field = "a" if use_buffer_a else "b"
                    display_scheme, scene_rt, present = rebuild_display_scheme(
                        ctx,
                        swapchain,
                        cells,
                        current_field,
                        render_pipeline,
                        screen,
                    )

            now = time.monotonic()
            if (now - last_update) > 0.033:
                last_update = now
                read_field = "a" if use_buffer_a else "b"
                write_field = "b" if use_buffer_a else "a"
                run_compute_step(ctx, cells, read_field, write_field, compute_pipeline)
                use_buffer_a = not use_buffer_a
                current_field = "a" if use_buffer_a else "b"
                display_scheme, scene_rt, present = rebuild_display_scheme(
                    ctx,
                    swapchain,
                    cells,
                    current_field,
                    render_pipeline,
                    screen,
                )

            submission = display_scheme.submit()
            present.consume(submission)

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
