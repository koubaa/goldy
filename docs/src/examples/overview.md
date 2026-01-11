# Examples Overview

RAG includes 20 interactive examples demonstrating various GPU rendering techniques. Many are available as interactive WebGPU demos embedded in these docs!

<div class="rag-demo" data-canvas="gradient-canvas" data-demo="GradientDemo">
    <canvas id="gradient-canvas"></canvas>
</div>

*Animated gradient demo running via WebGPU.*

## Running Examples

```bash
cd rag
cargo run --example <name> --release
```

All examples support:
- **Escape** - Exit the application
- **Window resize** - Automatic adaptation

## Example Gallery

### Basic

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `triangle` | Colored triangle | Vertex buffers, basic pipeline |
| `gradient` | Animated gradient | Fragment shaders, UV coordinates |
| `window` | Triangle with animation | Surface API basics |

### Classic Demoscene

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `plasma` | Psychedelic plasma effect | Complex fragment math, time animation |
| `tunnel` | Flying through a tunnel | Polar coordinates, procedural textures |
| `metaballs` | Organic blob simulation | Distance fields, thresholding |
| `starfield` | 3D starfield flythrough | Particle rendering, depth simulation |

### Interactive

| Example | Description | Controls |
|---------|-------------|----------|
| `mandelbrot` | Fractal explorer | Arrows=pan, +/-=zoom, R=reset |
| `particles` | Rain/snow simulation | Space=toggle mode |
| `digital_clock` | 7-segment display | Space=pause, Click=color |

### Visual Effects

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `bouncing_lines` | Lines bouncing off walls | Line primitive, physics |
| `spinning_cube` | 3D wireframe cube | 3D projection, rotation matrices |
| `checkerboard` | Animated procedural texture | UV distortion, patterns |
| `waveform` | Audio waveform visualizer | Line strips, multiple draw calls |
| `instancing` | 400 rotating quads | Many objects, HSV colors |

### 3D Graphics

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `solid_cube` | Solid 3D cube | 3D rendering, depth buffer |
| `textured_quad` | Textured 2D quad | Texture loading, samplers |

### Compute

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `compute_particles` | GPU-accelerated particles | Compute shaders, storage buffers |
| `game_of_life` | Conway's Game of Life | Compute pipeline, ping-pong buffers |

### Advanced

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `multi_window` | Multiple windows | Multiple surfaces, window management |

## Source Code

All examples are in `rag/examples/`:

```
rag/examples/
├── triangle.rs         # Basic triangle
├── window.rs           # Surface API basics
├── digital_clock.rs    # 7-segment clock
├── gradient.rs         # Animated gradient
├── plasma.rs           # Plasma effect
├── tunnel.rs           # Tunnel effect
├── starfield.rs        # 3D starfield
├── mandelbrot.rs       # Fractal explorer
├── bouncing_lines.rs   # Physics lines
├── spinning_cube.rs    # 3D wireframe
├── metaballs.rs        # Blob effect
├── checkerboard.rs     # Procedural texture
├── instancing.rs       # Many quads
├── particles.rs        # Rain/snow
├── waveform.rs         # Audio visualizer
├── solid_cube.rs       # Solid 3D cube
├── textured_quad.rs    # Textured quad
├── compute_particles.rs# GPU particles
├── game_of_life.rs     # Cellular automaton
└── multi_window.rs     # Multiple windows
```

## Common Patterns

### Basic Window Setup

```rust
struct App {
    instance: Instance,
    device: Option<Arc<rag::Device>>,
    pipeline: Option<RenderPipeline>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
}
```

### Render Loop (Surface API)

```rust
fn render_frame(&mut self) -> anyhow::Result<()> {
    let surface = self.surface.as_ref().unwrap();
    
    // Acquire next swapchain image
    let frame = surface.acquire()?;
    
    // Record render commands
    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(background_color);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertices);
        pass.draw(0..count, 0..1);
    }
    
    // Render and present (zero-copy!)
    frame.render(encoder)?;
    surface.present(frame)?;
    
    Ok(())
}
```

### Resize Handling

```rust
fn handle_resize(&mut self, new_size: PhysicalSize<u32>) {
    if new_size.width > 0 && new_size.height > 0 {
        if let Some(surface) = &mut self.surface {
            let _ = surface.resize(new_size.width, new_size.height);
        }
    }
}
```

### Custom Shaders (Slang)

```slang
struct VertexOutput {
    float4 position : SV_Position;
    float2 uv;
};

[shader("vertex")]
VertexOutput vs_main(float2 pos : POSITION, float2 uv : TEXCOORD) {
    VertexOutput output;
    output.position = float4(pos, 0.0, 1.0);
    output.uv = uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return float4(input.uv, 0.5, 1.0);
}
```

## Next Steps

Pick an example that interests you:

- [Triangle](./triangle.md) - Start here
- [Digital Clock](./digital-clock.md) - More complex vertex generation
- [Plasma](./plasma.md) - Fragment shader effects
- [Mandelbrot](./mandelbrot.md) - Interactive exploration
