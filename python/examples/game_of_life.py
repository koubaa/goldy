#!/usr/bin/env python3
"""Conway's Game of Life - Compute + Graphics Example.

This example demonstrates:
1. Compute shader running cellular automaton rules
2. Graphics shader rendering the grid
3. Ping-pong buffer technique for in-place updates
4. Using both compute and graphics pipelines together

Usage:
    python game_of_life.py

The simulation runs for a set number of generations and saves frames as images.
"""

import goldy
import numpy as np
import time
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Note: Install Pillow for image output: pip install pillow")

# Grid dimensions (must match shader constants)
GRID_WIDTH = 128
GRID_HEIGHT = 128
CELL_COUNT = GRID_WIDTH * GRID_HEIGHT

# Path to shader files (shared with Rust examples)
SHADER_DIR = Path(__file__).parent.parent.parent / "shaders"


def load_shader(name: str) -> str:
    """Load a shader from the shared shaders directory."""
    path = SHADER_DIR / name
    return path.read_text()


def create_initial_state():
    """Create initial pattern with Gosper Glider Gun + random cells."""
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
        px = x + offset_x
        py = y + offset_y
        if px < GRID_WIDTH and py < GRID_HEIGHT:
            cells[py * GRID_WIDTH + px] = 1
    
    # Add some random cells in the lower right
    np.random.seed(42)
    for y in range(60, 100):
        for x in range(60, 100):
            if np.random.randint(4) == 0:
                if x < GRID_WIDTH and y < GRID_HEIGHT:
                    cells[y * GRID_WIDTH + x] = 1
    
    return cells


def main():
    print("Conway's Game of Life (Compute + Graphics)")
    print("=" * 50)
    
    # Create device
    instance = goldy.Instance()
    device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
    print(f"Backend: {instance.backend_type}")
    print(f"Grid: {GRID_WIDTH}x{GRID_HEIGHT} = {CELL_COUNT} cells")
    print()
    
    # Create initial state
    initial_state = create_initial_state()
    alive_count = np.sum(initial_state)
    print(f"Initial state: {alive_count} living cells")
    
    # Create ping-pong buffers
    buffer_a = goldy.Buffer(device, initial_state, goldy.BufferUsage.STORAGE)
    buffer_b = goldy.Buffer(device, initial_state, goldy.BufferUsage.STORAGE)
    print(f"Created buffers: {buffer_a.size} bytes each")
    
    # === COMPUTE PIPELINE ===
    
    # Compute bind group layout: read-only input, read-write output
    compute_bind_layout = goldy.BindGroupLayout(device, [
        goldy.BindGroupLayoutBinding(
            0, goldy.ShaderStages.COMPUTE,
            goldy.BindingType.storage_buffer(read_only=True)
        ),
        goldy.BindGroupLayoutBinding(
            1, goldy.ShaderStages.COMPUTE,
            goldy.BindingType.storage_buffer(read_only=False)
        ),
    ])
    
    # A -> B: read from A, write to B
    compute_bind_group_a = goldy.BindGroup(device, compute_bind_layout, [
        goldy.BufferBinding(0, buffer_a),
        goldy.BufferBinding(1, buffer_b),
    ])
    
    # B -> A: read from B, write to A
    compute_bind_group_b = goldy.BindGroup(device, compute_bind_layout, [
        goldy.BufferBinding(0, buffer_b),
        goldy.BufferBinding(1, buffer_a),
    ])
    
    # Load and compile compute shader from shared file
    compute_shader_src = load_shader("game_of_life.slang")
    compute_shader = goldy.ShaderModule.from_slang(device, compute_shader_src)
    print("Compiled compute shader (game_of_life.slang)")
    
    compute_pipeline = goldy.ComputePipeline(
        device, compute_shader,
        goldy.ComputePipelineDesc([compute_bind_layout])
    )
    print("Created compute pipeline")
    
    # === RENDER PIPELINE ===
    
    # Render bind group layout: read-only storage buffer
    render_bind_layout = goldy.BindGroupLayout(device, [
        goldy.BindGroupLayoutBinding(
            0, goldy.ShaderStages.FRAGMENT,
            goldy.BindingType.storage_buffer(read_only=True)
        ),
    ])
    
    # Bind groups for reading from A or B
    render_bind_group_a = goldy.BindGroup(device, render_bind_layout, [
        goldy.BufferBinding(0, buffer_a),
    ])
    render_bind_group_b = goldy.BindGroup(device, render_bind_layout, [
        goldy.BufferBinding(0, buffer_b),
    ])
    
    # Load and compile render shader from shared file
    render_shader_src = load_shader("game_of_life_render.slang")
    render_shader = goldy.ShaderModule.from_slang(device, render_shader_src)
    print("Compiled render shader (game_of_life_render.slang)")
    
    render_pipeline = goldy.RenderPipeline(
        device, render_shader, render_shader,
        goldy.RenderPipelineDesc(
            target_format=goldy.TextureFormat.RGBA8_UNORM,
            bind_group_layouts=[render_bind_layout],
        )
    )
    print("Created render pipeline")
    
    # Create render target
    target = goldy.RenderTarget(device, 512, 512, goldy.TextureFormat.RGBA8_UNORM)
    print(f"Created render target: {target.width}x{target.height}")
    print()
    
    # === SIMULATION LOOP ===
    
    num_generations = 100
    use_buffer_a = True
    save_every = 10
    
    print(f"Running {num_generations} generations...")
    start_time = time.time()
    
    for gen in range(num_generations):
        # === COMPUTE PASS: Update simulation ===
        compute_encoder = goldy.ComputeEncoder()
        with compute_encoder.begin_compute_pass() as cp:
            cp.set_pipeline(compute_pipeline)
            
            # Choose which bind group based on current buffer
            if use_buffer_a:
                cp.set_bind_group(0, compute_bind_group_a)  # A -> B
            else:
                cp.set_bind_group(0, compute_bind_group_b)  # B -> A
            
            # Dispatch workgroups (8x8 threads per group)
            workgroups_x = (GRID_WIDTH + 7) // 8
            workgroups_y = (GRID_HEIGHT + 7) // 8
            cp.dispatch(workgroups_x, workgroups_y, 1)
        
        compute_encoder.dispatch(device)
        
        # Toggle buffer for next frame
        use_buffer_a = not use_buffer_a
        
        # === RENDER PASS: Visualize the grid ===
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(goldy.Color.BLACK)
            rp.set_pipeline(render_pipeline)
            
            # Read from the buffer that was just written to
            if use_buffer_a:
                rp.set_bind_group(0, render_bind_group_a)
            else:
                rp.set_bind_group(0, render_bind_group_b)
            
            # Draw fullscreen triangle
            rp.draw(range(3))
        
        target.render(encoder)
        
        # Save frames periodically
        if HAS_PIL and gen % save_every == 0:
            pixels = target.read_to_cpu()
            img = Image.fromarray(pixels, mode='RGBA')
            filename = f'game_of_life_gen_{gen:04d}.png'
            img.save(filename)
            print(f"  Generation {gen}: saved {filename}")
    
    elapsed = time.time() - start_time
    fps = num_generations / elapsed
    
    print()
    print(f"Completed {num_generations} generations in {elapsed:.2f}s")
    print(f"Performance: {fps:.1f} generations/second")
    
    # Save final frame
    if HAS_PIL:
        pixels = target.read_to_cpu()
        img = Image.fromarray(pixels, mode='RGBA')
        img.save('game_of_life_final.png')
        print("Saved: game_of_life_final.png")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
