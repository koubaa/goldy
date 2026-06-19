#!/usr/bin/env python3
"""Headless triangle — Scheme render pass + readback (CI / no display).

Usage:
    python triangle_headless.py
"""

import goldy
import numpy as np


def main():
    print("Goldy Python Triangle (headless Scheme)")
    print("=" * 40)

    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
    ctx = device.create_context()
    print(f"Backend: {instance.backend_type}")

    shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
    pipeline = goldy.RenderPipeline(
        device,
        shader,
        shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.vertex_2d(),
            target_format=goldy.TextureFormat.RGBA8_UNORM,
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
    vertex_parcel = retained_pool.acquire_buffer(vertices, goldy.BufferKind.SCATTERED)[0]
    readback = retained_pool.acquire_texture(
        100,
        100,
        goldy.TextureFormat.RGBA8_UNORM,
        goldy.TextureKind.DIRECT,
        copy_src=True,
        copy_dst=True,
    )

    scheme = goldy.Scheme(ctx)
    rt = scheme.lease_render_target(100, 100, goldy.TextureFormat.RGBA8_UNORM)
    with scheme.render_pass("triangle", rt) as rp:
        (
            rp.with_parcel(vertex_parcel, goldy.NodeAccess.READ)
            .clear(goldy.Color(0.1, 0.1, 0.2, 1.0))
            .set_pipeline(pipeline)
            .set_vertex_buffer_parcel(0, vertex_parcel)
            .draw(range(3))
        )

    scheme.copy_to_texture(rt, readback)
    grant = scheme.grant_read_texture(readback)
    submission = scheme.submit()
    pixels = np.frombuffer(grant.consume(submission), dtype=np.uint8).reshape(100, 100, 4)

    assert pixels.shape == (100, 100, 4)
    assert np.any(pixels[:, :, :3] > 0), "Triangle should write non-black pixels"
    print("Readback OK")
    print("Done!")


if __name__ == "__main__":
    main()
