#!/usr/bin/env python3
"""Headless Game of Life — hybrid Scheme (compute + render + readback).

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

SLOT_A = 0
SLOT_B = 1

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
    print("Goldy Python Game of Life (headless Scheme)")
    print("=" * 40)

    compute_src = (SHADERS_DIR / "game_of_life.slang").read_text(encoding="utf-8")
    render_src = (SHADERS_DIR / "game_of_life_render.slang").read_text(encoding="utf-8")

    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
    ctx = device.create_context()
    print(f"Backend: {instance.backend_type}")

    initial = initial_cells()
    zeros = np.zeros(CELL_COUNT, dtype=np.uint32)
    retained_pool = goldy.RetainedPool(device)
    mosaic = retained_pool.mosaic()
    mosaic.emplace(initial)
    mosaic.emplace(zeros)
    cells = mosaic.build(retained_pool)

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

    readback = retained_pool.acquire_texture(
        GRID_WIDTH,
        GRID_HEIGHT,
        goldy.TextureFormat.RGBA8_UNORM,
        goldy.TextureKind.DIRECT,
        copy_src=True,
        copy_dst=True,
    )

    scheme = goldy.Scheme(ctx)
    node = scheme.node("game_of_life", compute_pipeline)
    (
        node.declare_parcel_view(cells, SLOT_A, goldy.NodeAccess.READ, goldy.ResourceAccess.READ_WRITE)
        .declare_parcel_view(cells, SLOT_B, goldy.NodeAccess.WRITE, goldy.ResourceAccess.WRITE)
        .dispatch(WORKGROUPS_X, WORKGROUPS_Y, 1)
    )

    rt = scheme.lease_render_target(GRID_WIDTH, GRID_HEIGHT, goldy.TextureFormat.RGBA8_UNORM)
    render_idx = cells.mosaic_view_resource_index(SLOT_B, goldy.ResourceAccess.READ_WRITE)
    with scheme.render_pass("game_of_life_render", rt) as rp:
        (
            rp.bind_parcel_view(cells, SLOT_B, goldy.NodeAccess.READ)
            .clear(goldy.Color.BLACK)
            .set_pipeline(render_pipeline)
            .bind_resource_index(render_idx)
            .draw_fullscreen()
        )

    scheme.copy_to_texture(rt, readback)
    grant = scheme.grant_read_texture(readback)
    submission = scheme.submit()
    pixels = grant.consume(submission)

    cells_out = np.frombuffer(
        cells.mosaic_view_read_to_cpu(SLOT_B, device), dtype=np.uint32
    )
    assert cells_out.shape == (CELL_COUNT,)
    live = count_live(cells_out)
    assert live == 4, f"still-life block should remain 4 live cells, got {live}"

    stride = GRID_WIDTH * 4
    cx = GRID_WIDTH // 2
    cy = GRID_HEIGHT // 2
    g = pixels[cy * stride + cx * 4 + 1]
    assert g > 100, f"center pixel should show alive cells (g={g})"

    print(f"Simulation + render OK ({GRID_WIDTH}x{GRID_HEIGHT}, g={g} at center)")
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
