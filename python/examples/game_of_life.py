#!/usr/bin/env python3
"""Conway's Game of Life - Compute + Graphics Example

This example demonstrates:
1. Compute shader running cellular automaton rules
2. Graphics shader rendering the grid
3. Ping-pong buffer technique for in-place updates

Usage:
    python game_of_life.py
"""

import goldy
import numpy as np
import time
import os
import glfw


GRID_WIDTH = 128
GRID_HEIGHT = 128
CELL_COUNT = GRID_WIDTH * GRID_HEIGHT


def load_shader(name):
    """Load shader from shared shaders directory."""
    shader_dir = os.path.join(os.path.dirname(__file__), "..", "..", "shaders")
    shader_path = os.path.join(shader_dir, name)
    with open(shader_path, "r") as f:
        return f.read()


def create_initial_state():
    """Create initial pattern (glider gun + some random cells)."""
    cells = np.zeros(CELL_COUNT, dtype=np.uint32)

    # Gosper Glider Gun (creates infinite gliders)
    gun = [
        (1, 5), (1, 6), (2, 5), (2, 6),
        (11, 5), (11, 6), (11, 7),
        (12, 4), (12, 8),
        (13, 3), (13, 9),
        (14, 3), (14, 9),
        (15, 6),
        (16, 4), (16, 8),
        (17, 5), (17, 6), (17, 7),
        (18, 6),
        (21, 3), (21, 4), (21, 5),
        (22, 3), (22, 4), (22, 5),
        (23, 2), (23, 6),
        (25, 1), (25, 2), (25, 6), (25, 7),
        (35, 3), (35, 4),
        (36, 3), (36, 4),
    ]

    # Place glider gun
    offset_x, offset_y = 10, 10
    for x, y in gun:
        px, py = x + offset_x, y + offset_y
        if px < GRID_WIDTH and py < GRID_HEIGHT:
            cells[py * GRID_WIDTH + px] = 1

    # Add some random cells in the lower right
    np.random.seed(42)
    for y in range(60, 100):
        for x in range(60, 100):
            if np.random.randint(4) == 0:
                cells[y * GRID_WIDTH + x] = 1

    return cells


def main():
    print("Game of Life - Press Escape to exit")

    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, True)

    window = glfw.create_window(800, 800, "Game of Life", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    # Create Goldy device and surface
    instance = goldy.Instance()
    device = instance.request_adapter().request_device()
    surface = goldy.Surface.from_glfw(device, window)

    # Load shaders
    compute_shader = goldy.ShaderModule.from_slang(device, load_shader("game_of_life.slang"))
    render_shader = goldy.ShaderModule.from_slang(device, load_shader("game_of_life_render.slang"))

    # Create ping-pong buffers
    initial_state = create_initial_state()
    alive_count = np.sum(initial_state)
    print(f"Initial alive cells: {alive_count} / {len(initial_state)}")
    buffer_a = goldy.Buffer(device, initial_state, goldy.BufferKind.SCATTERED)
    buffer_b = goldy.Buffer(device, initial_state, goldy.BufferKind.SCATTERED)

    # Create compute pipeline
    compute_pipeline = goldy.ComputePipeline(device, compute_shader)

    # Create render pipeline
    # Use empty vertex layout since the shader generates vertices via SV_VertexID
    render_pipeline = goldy.RenderPipeline(
        device, render_shader, render_shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.empty(),
            topology=goldy.PrimitiveTopology.TRIANGLE_LIST,
            target_format=surface.format,
        )
    )

    print(f"Initialized: {GRID_WIDTH}x{GRID_HEIGHT} grid")
    print("Features Gosper Glider Gun + random cells")

    use_buffer_a = True
    last_update = time.time()
    frame_count = 0
    
    # CI mode: exit after a few frames to avoid hanging
    ci_mode = os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true'
    max_frames = 10 if ci_mode else float('inf')

    # Handle window resize
    def on_resize(win, w, h):
        if w > 0 and h > 0:
            surface.resize(w, h)

    glfw.set_framebuffer_size_callback(window, on_resize)

    def on_key(win, key, scancode, action, mods):
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, on_key)

    # Main render loop
    while not glfw.window_should_close(window) and frame_count < max_frames:
        glfw.poll_events()

        # Update simulation ~30 times per second
        now = time.time()
        should_update = (now - last_update) > 0.033

        if should_update:
            last_update = now

            # Run compute pass with ping-pong buffers
            compute_encoder = goldy.ComputeEncoder()
            with compute_encoder.begin_compute_pass() as cp:
                cp.set_pipeline(compute_pipeline)

                # Bind resource slots
                # Order matters: [current_state, next_state] matching shader slots
                if use_buffer_a:
                    # A -> B: read from A, write to B
                    cp.bind_resources([buffer_a, buffer_b])
                else:
                    # B -> A: read from B, write to A
                    cp.bind_resources([buffer_b, buffer_a])

                # Dispatch workgroups (8x8 threads per group)
                workgroups_x = (GRID_WIDTH + 7) // 8
                workgroups_y = (GRID_HEIGHT + 7) // 8
                cp.dispatch(workgroups_x, workgroups_y, 1)

            compute_encoder.dispatch(device)

            # Toggle buffer for next frame
            use_buffer_a = not use_buffer_a

        # Render
        frame = surface.acquire()

        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(goldy.Color.BLACK)
            rp.set_pipeline(render_pipeline)

            # Read from the buffer that is now "current"
            if use_buffer_a:
                rp.bind_resources([buffer_a])
            else:
                rp.bind_resources([buffer_b])

            # Draw fullscreen triangle
            rp.draw(range(3))

        frame.render(encoder)
        surface.present(frame)
        
        frame_count += 1
        
    glfw.terminate()


if __name__ == '__main__':
    main()
