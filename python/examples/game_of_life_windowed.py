#!/usr/bin/env python3
"""Interactive Game of Life - Windowed version with real-time display.

This example demonstrates:
1. Compute + Graphics pipeline combination
2. Real-time windowed rendering with GLFW
3. Interactive controls

Usage:
    pip install glfw
    python game_of_life_windowed.py

Controls:
    - ESC: Close window
    - Space: Pause/resume simulation
    - R: Reset to initial state
    - C: Clear grid
    - Click: Toggle cells (when paused)
"""

import goldy
import numpy as np
import time
from pathlib import Path

try:
    import glfw
except ImportError:
    print("This example requires GLFW. Install with: pip install glfw")
    exit(1)

# Grid dimensions (must match shader constants)
GRID_WIDTH = 128
GRID_HEIGHT = 128
CELL_COUNT = GRID_WIDTH * GRID_HEIGHT

# Path to shader files
SHADER_DIR = Path(__file__).parent.parent.parent / "shaders"


def load_shader(name: str) -> str:
    return (SHADER_DIR / name).read_text()


def create_initial_state():
    """Create Gosper Glider Gun + random cells."""
    cells = np.zeros(CELL_COUNT, dtype=np.uint32)

    # Gosper Glider Gun
    gun = [
        (1, 5), (1, 6), (2, 5), (2, 6),
        (11, 5), (11, 6), (11, 7), (12, 4), (12, 8),
        (13, 3), (13, 9), (14, 3), (14, 9), (15, 6),
        (16, 4), (16, 8), (17, 5), (17, 6), (17, 7), (18, 6),
        (21, 3), (21, 4), (21, 5), (22, 3), (22, 4), (22, 5),
        (23, 2), (23, 6), (25, 1), (25, 2), (25, 6), (25, 7),
        (35, 3), (35, 4), (36, 3), (36, 4),
    ]

    for x, y in gun:
        px, py = x + 10, y + 10
        if px < GRID_WIDTH and py < GRID_HEIGHT:
            cells[py * GRID_WIDTH + px] = 1

    # Random cells
    np.random.seed(42)
    for y in range(60, 100):
        for x in range(60, 100):
            if np.random.randint(4) == 0 and x < GRID_WIDTH and y < GRID_HEIGHT:
                cells[y * GRID_WIDTH + x] = 1

    return cells


def main():
    print("Interactive Game of Life")
    print("=" * 50)
    print("Controls: ESC=quit, Space=pause, R=reset, C=clear")
    print()

    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, True)

    window = glfw.create_window(800, 800, "Game of Life - Press ESC to exit", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")

    # Create device and surface
    instance = goldy.Instance()
    device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
    surface = goldy.Surface.from_glfw(device, window)
    print(f"Backend: {instance.backend_type}")
    print(f"Grid: {GRID_WIDTH}x{GRID_HEIGHT}")

    # Create initial state
    initial_state = create_initial_state()

    # Create ping-pong buffers
    buffer_a = goldy.Buffer(device, initial_state, goldy.BufferUsage.STORAGE)
    buffer_b = goldy.Buffer(device, initial_state.copy(), goldy.BufferUsage.STORAGE)

    # === COMPUTE PIPELINE ===
    compute_bind_layout = goldy.BindGroupLayout(device, [
        goldy.BindGroupLayoutBinding(0, goldy.ShaderStages.COMPUTE,
            goldy.BindingType.storage_buffer(read_only=True)),
        goldy.BindGroupLayoutBinding(1, goldy.ShaderStages.COMPUTE,
            goldy.BindingType.storage_buffer(read_only=False)),
    ])

    compute_bind_group_a = goldy.BindGroup(device, compute_bind_layout, [
        goldy.BufferBinding(0, buffer_a),
        goldy.BufferBinding(1, buffer_b),
    ])
    compute_bind_group_b = goldy.BindGroup(device, compute_bind_layout, [
        goldy.BufferBinding(0, buffer_b),
        goldy.BufferBinding(1, buffer_a),
    ])

    compute_shader = goldy.ShaderModule.from_slang(device, load_shader("game_of_life.slang"))
    compute_pipeline = goldy.ComputePipeline(
        device, compute_shader,
        goldy.ComputePipelineDesc([compute_bind_layout])
    )

    # === RENDER PIPELINE ===
    render_bind_layout = goldy.BindGroupLayout(device, [
        goldy.BindGroupLayoutBinding(0, goldy.ShaderStages.FRAGMENT,
            goldy.BindingType.storage_buffer(read_only=True)),
    ])

    render_bind_group_a = goldy.BindGroup(device, render_bind_layout, [
        goldy.BufferBinding(0, buffer_a),
    ])
    render_bind_group_b = goldy.BindGroup(device, render_bind_layout, [
        goldy.BufferBinding(0, buffer_b),
    ])

    render_shader = goldy.ShaderModule.from_slang(device, load_shader("game_of_life_render.slang"))
    render_pipeline = goldy.RenderPipeline(
        device, render_shader, render_shader,
        goldy.RenderPipelineDesc(
            target_format=surface.format,
            bind_group_layouts=[render_bind_layout],
        )
    )

    # State
    use_buffer_a = True
    paused = False
    generation = 0
    last_update = time.time()
    update_interval = 1.0 / 30  # 30 updates per second

    # Handle resize
    def on_resize(win, w, h):
        if w > 0 and h > 0:
            surface.resize(w, h)

    glfw.set_framebuffer_size_callback(window, on_resize)

    # Handle keys
    def on_key(win, key, scancode, action, mods):
        nonlocal paused, use_buffer_a, generation
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE:
                paused = not paused
                print(f"{'Paused' if paused else 'Running'} at generation {generation}")
            elif key == glfw.KEY_R:
                # Reset to initial state
                buffer_a.write(0, create_initial_state())
                buffer_b.write(0, create_initial_state())
                use_buffer_a = True
                generation = 0
                print("Reset!")
            elif key == glfw.KEY_C:
                # Clear grid
                empty = np.zeros(CELL_COUNT, dtype=np.uint32)
                buffer_a.write(0, empty)
                buffer_b.write(0, empty)
                use_buffer_a = True
                generation = 0
                print("Cleared!")

    glfw.set_key_callback(window, on_key)

    print("\nRunning... (press ESC to exit)")
    frame_count = 0
    start_time = time.time()

    # Main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        now = time.time()

        # Update simulation (rate limited)
        if not paused and (now - last_update) >= update_interval:
            last_update = now

            # Run compute pass
            compute_encoder = goldy.ComputeEncoder()
            with compute_encoder.begin_compute_pass() as cp:
                cp.set_pipeline(compute_pipeline)
                cp.set_bind_group(0, compute_bind_group_a if use_buffer_a else compute_bind_group_b)
                cp.dispatch((GRID_WIDTH + 7) // 8, (GRID_HEIGHT + 7) // 8, 1)
            compute_encoder.dispatch(device)

            use_buffer_a = not use_buffer_a
            generation += 1

        # Render
        frame = surface.acquire()
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(goldy.Color.BLACK)
            rp.set_pipeline(render_pipeline)
            rp.set_bind_group(0, render_bind_group_a if use_buffer_a else render_bind_group_b)
            rp.draw(range(3))

        frame.render(encoder)
        surface.present(frame)
        frame_count += 1

    # Stats
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n{frame_count} frames, {generation} generations in {elapsed:.1f}s ({fps:.1f} FPS)")

    glfw.terminate()


if __name__ == '__main__':
    main()

