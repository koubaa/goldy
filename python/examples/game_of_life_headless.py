#!/usr/bin/env python3
"""Headless Game of Life — hybrid TaskGraph smoke test (compute + render + readback).

Mirrors `goldy/ffi-client/examples/game_of_life_headless.rs`. No GLFW/display required.

Usage:
    python game_of_life_headless.py
"""

from __future__ import annotations

from pathlib import Path

import goldy
import numpy as np

GRID_WIDTH = 128
GRID_HEIGHT = 128
CELL_COUNT = GRID_WIDTH * GRID_HEIGHT
WORKGROUPS_X = (GRID_WIDTH + 7) // 8
WORKGROUPS_Y = (GRID_HEIGHT + 7) // 8

SHADERS_DIR = Path(__file__).resolve().parents[2] / "shaders"


def initial_cells() -> np.ndarray:
    cells = np.zeros(CELL_COUNT, dtype=np.uint32)
    for y in (63, 64):
        for x in (63, 64):
            cells[y * GRID_WIDTH + x] = 1
    return cells


def count_live(cells: np.ndarray) -> int:
    return int(np.count_nonzero(cells == 1))


def main() -> int:
    print("Goldy Python Game of Life (headless TaskGraph)")
    print("=" * 40)

    compute_src = (SHADERS_DIR / "game_of_life.slang").read_text(encoding="utf-8")
    render_src = (SHADERS_DIR / "game_of_life_render.slang").read_text(encoding="utf-8")

    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
    print(f"Backend: {instance.backend_type}")

    initial = initial_cells()
    zeros = np.zeros(CELL_COUNT, dtype=np.uint32)
    retained_pool = goldy.RetainedPool(device)
    buf_a = retained_pool.acquire_buffer(initial, goldy.BufferKind.SCATTERED)
    buf_b = retained_pool.acquire_buffer(zeros, goldy.BufferKind.SCATTERED)

    compute_shader = goldy.ShaderModule.from_slang(device, compute_src)
    render_shader = goldy.ShaderModule.from_slang(device, render_src)
    compute_pipeline = goldy.ComputePipeline(device, compute_shader)
    render_pipeline = goldy.RenderPipeline(
        device,
        render_shader,
        render_shader,
        goldy.RenderPipelineDesc(
            target_format=goldy.TextureFormat.RGBA8_UNORM,
            topology=goldy.PrimitiveTopology.TRIANGLE_LIST,
        ),
    )

    target = goldy.RenderTarget(device, GRID_WIDTH, GRID_HEIGHT, goldy.TextureFormat.RGBA8_UNORM)

    read_idx = buf_a.resource_index(goldy.ResourceAccess.READ_WRITE)
    write_idx = buf_b.resource_index(goldy.ResourceAccess.WRITE)

    graph = goldy.TaskGraph()
    with graph.compute_node(
        "game_of_life",
        compute_pipeline,
        workgroups=(WORKGROUPS_X, WORKGROUPS_Y, 1),
    ) as node:
        (
            node.bind_parcel(buf_a, goldy.NodeAccess.READ)
            .bind_parcel(buf_b, goldy.NodeAccess.WRITE)
            .bind_resources_raw([read_idx, write_idx])
        )

    with graph.render_pass("game_of_life_render", target) as rp:
        (
            rp.bind_parcel(buf_b, goldy.NodeAccess.READ)
            .clear(goldy.Color.BLACK)
            .set_pipeline(render_pipeline)
            .bind_resource_index(buf_b.resource_index(goldy.ResourceAccess.READ))
            .draw_fullscreen()
        )

    graph.dispatch(device)

    cells_out = np.frombuffer(buf_b.read_to_cpu(device), dtype=np.uint32)
    assert cells_out.shape == (CELL_COUNT,)
    live = count_live(cells_out)
    assert live == 4, f"still-life block should remain 4 live cells, got {live}"

    pixels = target.read_to_cpu()
    cx = GRID_WIDTH // 2
    cy = GRID_HEIGHT // 2
    g = int(pixels[cy, cx, 1])
    assert g > 100, f"center pixel should show alive cells (g={g})"

    print(f"Simulation + render OK ({GRID_WIDTH}x{GRID_HEIGHT}, g={g} at center)")
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
