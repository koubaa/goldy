# Examples Overview

RAG includes 13 interactive examples demonstrating various GPU rendering techniques. Many are available as interactive WebGPU demos embedded in these docs!

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

## Source Code

All examples are in `rag/examples/`:

```
rag/examples/
├── triangle.rs        # Basic triangle
├── digital_clock.rs   # 7-segment clock
├── gradient.rs        # Animated gradient
├── plasma.rs          # Plasma effect
├── tunnel.rs          # Tunnel effect
├── starfield.rs       # 3D starfield
├── mandelbrot.rs      # Fractal explorer
├── bouncing_lines.rs  # Physics lines
├── spinning_cube.rs   # 3D wireframe
├── metaballs.rs       # Blob effect
├── checkerboard.rs    # Procedural texture
├── instancing.rs      # Many quads
├── particles.rs       # Rain/snow
└── waveform.rs        # Audio visualizer
```

## Common Patterns

### Basic Window Setup

```rust
struct App {
    instance: Instance,
    device: Option<rag::Device>,
    pipeline: Option<RenderPipeline>,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<...>>,
}
```

### Render Loop

```rust
fn render_frame(&mut self) -> anyhow::Result<()> {
    let frame = FrameOutput::new(&device, width, height, format);
    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(background_color);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertices);
        pass.draw(0..count, 0..1);
    }
    let output = frame.render(encoder)?;
    // Display output via softbuffer
}
```

### Custom Shaders

```rust
const MY_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.uv, 0.5, 1.0);
}
"#;
```

## Next Steps

Pick an example that interests you:

- [Triangle](./triangle.md) - Start here
- [Digital Clock](./digital-clock.md) - More complex vertex generation
- [Plasma](./plasma.md) - Fragment shader effects
- [Mandelbrot](./mandelbrot.md) - Interactive exploration

