#!/usr/bin/env python3
"""Headless triangle — TaskGraph clear + draw + readback (CI / no display).

Usage:
    python triangle_headless.py
"""

import goldy
import numpy as np


def main():
    print("Goldy Python Triangle (headless TaskGraph)")
    print("=" * 40)

    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
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
    vertex_buffer = goldy.Buffer(device, vertices, goldy.BufferKind.SCATTERED)

    target = goldy.RenderTarget(device, 100, 100, goldy.TextureFormat.RGBA8_UNORM)

    graph = goldy.TaskGraph()
    with graph.render_pass("triangle", target) as rp:
        (
            rp.bind_buffer(vertex_buffer, goldy.NodeAccess.READ)
            .clear(goldy.Color(0.1, 0.1, 0.2, 1.0))
            .set_pipeline(pipeline)
            .set_vertex_buffer(0, vertex_buffer)
            .draw(range(3))
        )

    graph.dispatch(device)
    pixels = target.read_to_cpu()

    assert pixels.shape == (100, 100, 4)
    assert np.any(pixels[:, :, :3] > 0), "Triangle should write non-black pixels"
    print("Readback OK")
    print("Done!")


if __name__ == "__main__":
    main()
