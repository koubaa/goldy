#!/usr/bin/env python3
"""Triangle example — render a colored triangle via TaskGraph (headless).

Renders to an offscreen RenderTarget, submits through TaskGraph, and verifies
readback. For windowed presentation, combine Surface.submit_graph_to_frame with
the same graph pattern (see goldy/examples/triangle.rs).

Usage:
    python triangle.py
"""

import goldy
import numpy as np


def main():
    print("Goldy Python Triangle (TaskGraph)")
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

    width, height = 100, 100
    target = goldy.RenderTarget(device, width, height, goldy.TextureFormat.RGBA8_UNORM)

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
    assert pixels.shape == (height, width, 4)
    assert pixels.dtype == np.uint8
    assert np.any(pixels[:, :, :3] > 0), "triangle should write non-black pixels"

    print(f"Rendered {width}x{height}, readback OK")
    print("Done!")


if __name__ == "__main__":
    main()
