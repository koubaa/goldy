# 3D Graphics Examples

Goldy includes several examples demonstrating 3D rendering concepts.

---

## Spinning Cube

A wireframe cube rotating in 3D space.

```bash
cargo run --example spinning_cube --release
```

### 3D Projection

Without a 3D pipeline, we do projection manually:

```rust
fn project(p: [f32; 3], fov: f32) -> [f32; 2] {
    let z = p[2] + 4.0;  // Distance from camera
    let scale = fov / z;  // Perspective divide
    [p[0] * scale, p[1] * scale]
}
```

### Rotation Matrices

```rust
fn rotate_y(p: [f32; 3], angle: f32) -> [f32; 3] {
    let (s, c) = (angle.sin(), angle.cos());
    [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]
}
```

---

## Starfield

Classic 3D starfield - flying forward through space.

```bash
cargo run --example starfield --release
```

### How It Works

Stars spawn at the center (far away) and expand outward as they get closer:

```rust
// z cycles from 1 (far) to 0 (near)
// radius increases as z decreases
let radius = max_radius * (1.0 - z);

// Star gets bigger and brighter as it approaches
let size = 0.005 + (1.0 - z) * 0.015;
```

---

## Tunnel

Classic demoscene tunnel effect with checkerboard texture.

```bash
cargo run --example tunnel --release
```

### Polar Coordinate Transform

The tunnel effect converts screen coordinates to polar coordinates:

```wgsl
let dist = length(uv);        // Distance from center
let angle = atan2(uv.y, uv.x); // Angle around center

// Convert to tunnel coordinates
let tunnel_depth = 1.0 / (dist + 0.1);
let tunnel_angle = angle / 3.14159 + time * 0.2;
```

---

## Building Your Own 3D

Goldy provides the primitives for 3D rendering:

1. **Vertex buffers** - Store transformed vertices
2. **Line/triangle primitives** - Draw edges or faces
3. **Custom shaders** - Do projection in vertex shader

For complex 3D, you might:
- Implement a matrix library
- Use `glam` or `cgmath` crates
- Build a proper 3D camera system

## Full Source

- `goldy/examples/spinning_cube.rs`
- `goldy/examples/starfield.rs`
- `goldy/examples/tunnel.rs`
