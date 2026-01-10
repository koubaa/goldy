# Plasma Effect

Classic demoscene plasma effect using fragment shaders.

<div class="rag-demo" data-canvas="plasma-canvas" data-demo="PlasmaDemo">
    <canvas id="plasma-canvas"></canvas>
</div>

*Interactive demo running in your browser via WebGPU. Requires Chrome 113+, Edge 113+, or Firefox with WebGPU enabled.*

## Run It

```bash
cargo run --example plasma --release
```

## What It Demonstrates

- Custom fragment shader
- Time-based animation
- Passing time to shader via vertex attributes
- Mathematical functions in WGSL

## The Shader

The plasma effect is entirely computed in the fragment shader:

```wgsl
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv * 4.0;
    let t = in.time;
    
    // Classic plasma formula
    var v = sin(uv.x + t);
    v += sin(uv.y + t);
    v += sin(uv.x + uv.y + t);
    
    // Circular wave
    let cx = uv.x + 0.5 * sin(t / 3.0);
    let cy = uv.y + 0.5 * cos(t / 2.0);
    v += sin(sqrt(cx * cx + cy * cy + 1.0) + t);
    
    v = v / 2.0;
    
    // Color palette using phase-shifted sines
    let r = sin(v * 3.14159);
    let g = sin(v * 3.14159 + 2.094);  // +120°
    let b = sin(v * 3.14159 + 4.188);  // +240°
    
    return vec4<f32>(r * 0.5 + 0.5, g * 0.5 + 0.5, b * 0.5 + 0.5, 1.0);
}
```

## Key Techniques

### Passing Time to Shader

Without uniform buffers, we pass time through vertex attributes:

```rust
#[repr(C)]
struct PlasmaVertex {
    position: [f32; 2],
    uv: [f32; 2],
    time: f32,  // Same value for all vertices
}

fn create_quad(time: f32) -> [PlasmaVertex; 6] {
    [
        PlasmaVertex { position: [-1.0, -1.0], uv: [0.0, 1.0], time },
        PlasmaVertex { position: [1.0, -1.0], uv: [1.0, 1.0], time },
        // ...
    ]
}
```

### Full-Screen Quad

The plasma covers the entire screen using a quad from (-1,-1) to (1,1):

```rust
fn create_quad(time: f32) -> [PlasmaVertex; 6] {
    [
        PlasmaVertex { position: [-1.0, -1.0], uv: [0.0, 1.0], time },
        PlasmaVertex { position: [1.0, -1.0], uv: [1.0, 1.0], time },
        PlasmaVertex { position: [1.0, 1.0], uv: [1.0, 0.0], time },
        PlasmaVertex { position: [-1.0, -1.0], uv: [0.0, 1.0], time },
        PlasmaVertex { position: [1.0, 1.0], uv: [1.0, 0.0], time },
        PlasmaVertex { position: [-1.0, 1.0], uv: [0.0, 0.0], time },
    ]
}
```

### Color Cycling

The rainbow effect comes from phase-shifted sine waves:
- Red: `sin(v * π)`
- Green: `sin(v * π + 120°)`
- Blue: `sin(v * π + 240°)`

This creates smooth color transitions through the spectrum.

## Variations

### Different Plasma Formulas

```wgsl
// Interference pattern
v = sin(uv.x * 10.0) + sin(uv.y * 10.0);

// Spiral
let angle = atan2(uv.y - 0.5, uv.x - 0.5);
let dist = length(uv - 0.5);
v = sin(angle * 5.0 + dist * 10.0 + t);

// Ripples
v = sin(length(uv - 0.5) * 20.0 - t * 5.0);
```

### Different Color Palettes

```wgsl
// Fire palette
let r = v;
let g = v * v;
let b = v * v * v;

// Ocean palette
let r = 0.0;
let g = v * 0.5;
let b = 0.5 + v * 0.5;
```

## Full Source

See `rag/examples/plasma.rs` for the complete code.

