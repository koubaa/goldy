#!/usr/bin/env python3
"""Headless Game of Life — hybrid Scheme (compute + render + readback).

Mirrors native scheme patterns: one retained record buffer (`"a"` / `"b"`), field
parcels bound via `with_field`.

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
    print("Goldy Python Game of Life (headless Scheme)")
    print("=" * 40)

    compute_src = (SHADERS_DIR / "game_of_life.slang").read_text(encoding="utf-8")
    render_src = (SHADERS_DIR / "game_of_life_render.slang").read_text(encoding="utf-8")

    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
    ctx = device.create_context()
    print(f"Backend: {instance.backend_type}")

    initial = initial_cells()
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
        node.with_field(cells, "a", goldy.NodeAccess.READ)
        .with_field(cells, "b", goldy.NodeAccess.OVERWRITE)
        .dispatch(WORKGROUPS_X, WORKGROUPS_Y, 1)
    )

    rt = scheme.lease_render_target(GRID_WIDTH, GRID_HEIGHT, goldy.TextureFormat.RGBA8_UNORM)
    with scheme.render_pass("game_of_life_render", rt, goldy.TargetLoad.discard()) as rp:
        (
            rp.with_field(cells, "b", goldy.NodeAccess.READ)
            .set_pipeline(render_pipeline)
            .draw_fullscreen()
        )

    scheme.copy_to_texture(rt, readback)
    memory = goldy.MemoryExchange(ctx)
    grant_tex = memory.bind_withdraw_texture(scheme, readback)
    grant_cells = memory.bind_withdraw(scheme, cells[1])
    submission = scheme.submit()
    pixels = grant_tex.claim(submission).consume()
    cells_out = np.frombuffer(grant_cells.claim(submission).consume(), dtype=np.uint32)
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
