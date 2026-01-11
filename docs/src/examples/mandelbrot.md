# Mandelbrot Explorer

Mandelbrot fractal viewer with pan and zoom.

## Run It

```bash
cargo run --example mandelbrot --release
```

## Controls

| Key | Action |
|-----|--------|
| **Arrow keys** | Pan around |
| **+ / =** | Zoom in |
| **-** | Zoom out |
| **R** | Reset view |
| **Escape** | Exit |

## What It Demonstrates

- Complex math in fragment shader
- User input-driven parameter updates
- Per-pixel computation
- Iteration-based coloring

## The Algorithm

The Mandelbrot set is defined by iterating:

```
z(n+1) = z(n)² + c
```

Starting with z(0) = 0, for each point c in the complex plane. If |z| stays bounded (< 2) after many iterations, c is in the set.

## The Shader

```wgsl
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = in.complex_coord;
    var z = vec2<f32>(0.0, 0.0);
    var i: u32 = 0u;
    let max_iter: u32 = 256u;
    
    loop {
        if i >= max_iter { break; }
        if dot(z, z) > 4.0 { break; }  // |z|² > 4
        
        // z = z² + c (complex multiplication)
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        i = i + 1u;
    }
    
    if i >= max_iter {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);  // In set: black
    }
    
    // Outside set: color based on iteration count
    let t = f32(i) / f32(max_iter);
    let r = sin(t * 5.0) * 0.5 + 0.5;
    let g = sin(t * 7.0 + 1.0) * 0.5 + 0.5;
    let b = sin(t * 11.0 + 2.0) * 0.5 + 0.5;
    
    return vec4<f32>(r, g, b, 1.0);
}
```

## Key Techniques

### Complex Plane Mapping

UV coordinates are mapped to the complex plane with adjustable center and zoom:

```rust
// In vertex shader
out.complex_coord = center + (uv - 0.5) * 4.0 / zoom;
```

### Passing Parameters

Center and zoom are passed through vertex attributes:

```rust
struct MandelbrotVertex {
    position: [f32; 2],
    center: [f32; 2],  // Complex plane center
    zoom: f32,         // Zoom level
    uv: [f32; 2],
}
```

### Input Handling

```rust
match event.logical_key {
    Key::Named(NamedKey::ArrowUp) => self.center[1] += pan,
    Key::Named(NamedKey::ArrowDown) => self.center[1] -= pan,
    Key::Named(NamedKey::ArrowLeft) => self.center[0] -= pan,
    Key::Named(NamedKey::ArrowRight) => self.center[0] += pan,
    Key::Character(ref c) if c == "+" => self.zoom *= 1.5,
    Key::Character(ref c) if c == "-" => self.zoom /= 1.5,
    Key::Character(ref c) if c == "r" => {
        self.center = [-0.5, 0.0];
        self.zoom = 1.0;
    }
    // ...
}
```

## Interesting Locations

Try zooming into these coordinates:

| Location | Center | Zoom |
|----------|--------|------|
| Overview | (-0.5, 0) | 1x |
| Seahorse Valley | (-0.75, 0.1) | 50x |
| Elephant Valley | (0.275, 0.0) | 100x |
| Double Spiral | (-0.759, 0.126) | 1000x |

## Variations

### More Iterations

```wgsl
let max_iter: u32 = 1000u;  // More detail when zoomed
```

### Smooth Coloring

```wgsl
// Smooth iteration count
let log_zn = log(dot(z, z)) / 2.0;
let nu = log(log_zn / log(2.0)) / log(2.0);
let smooth_i = f32(i) + 1.0 - nu;
let t = smooth_i / f32(max_iter);
```

### Julia Sets

Change the iteration to use a fixed c:

```wgsl
let c = vec2<f32>(-0.7, 0.27015);  // Fixed c
var z = in.complex_coord;          // Start at pixel position

// Same iteration loop...
```

## Full Source

See `goldy/examples/mandelbrot.rs` for the complete code.

