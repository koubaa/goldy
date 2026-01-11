# Particle Systems

Rain and snow simulation using CPU-driven particles.

## Run It

```bash
cargo run --example particles --release
```

## Controls

| Key | Action |
|-----|--------|
| **Space** | Toggle rain/snow |
| **Escape** | Exit |

## What It Demonstrates

- Many independent moving objects
- Per-particle physics
- Different behavior modes
- Dynamic vertex buffer updates

## Particle Structure

```rust
struct Particle {
    x: f32,      // Position
    y: f32,
    vx: f32,     // Velocity
    vy: f32,
    size: f32,   // Rendering size
    color: Color,
}
```

## Physics Update

Each frame, particles are updated:

```rust
impl Particle {
    fn update(&mut self, is_snow: bool) {
        self.x += self.vx;
        self.y += self.vy;
        
        // Snow drifts horizontally
        if is_snow {
            self.vx += (random() - 0.5) * 0.001;
            self.vx = self.vx.clamp(-0.01, 0.01);
        }

        // Respawn when off screen
        if self.y > 1.0 || self.x < -1.2 || self.x > 1.2 {
            *self = if is_snow { 
                Particle::new_snow() 
            } else { 
                Particle::new_rain() 
            };
        }
    }
}
```

## Rain vs Snow

### Rain
- Falls straight down (fast)
- Small, elongated drops
- Blue-white color

```rust
fn new_rain() -> Self {
    Self {
        x: random() * 2.0 - 1.0,
        y: -1.0 - random() * 0.5,  // Start above screen
        vx: (random() - 0.5) * 0.002,
        vy: 0.01 + random() * 0.02,  // Fast downward
        size: 0.002 + random() * 0.003,
        color: Color { r: 0.5, g: 0.6, b: 0.9, a: 0.8 },
    }
}
```

### Snow
- Falls slowly, drifts
- Larger, rounder flakes
- Pure white

```rust
fn new_snow() -> Self {
    Self {
        x: random() * 2.0 - 1.0,
        y: -1.0 - random() * 0.5,
        vx: (random() - 0.5) * 0.005,
        vy: 0.002 + random() * 0.005,  // Slow downward
        size: 0.003 + random() * 0.008,
        color: Color { r: 0.95, g: 0.95, b: 1.0, a: 0.9 },
    }
}
```

## Rendering

Each particle is rendered as a quad:

```rust
fn vertices(&self) -> [Vertex2D; 6] {
    let s = self.size;
    let c = self.color;
    // Elongated quad for rain drop
    [
        Vertex2D::new(self.x - s, self.y - s * 3.0, c),
        Vertex2D::new(self.x + s, self.y - s * 3.0, c),
        Vertex2D::new(self.x + s, self.y + s * 3.0, c),
        Vertex2D::new(self.x - s, self.y - s * 3.0, c),
        Vertex2D::new(self.x + s, self.y + s * 3.0, c),
        Vertex2D::new(self.x - s, self.y + s * 3.0, c),
    ]
}
```

## Render Loop

```rust
fn render_frame(&mut self) -> anyhow::Result<()> {
    // Update all particles
    for p in &mut self.particles {
        p.update(self.is_snow);
    }

    // Collect all vertices
    let mut vertices = Vec::with_capacity(NUM_PARTICLES * 6);
    for p in &self.particles {
        vertices.extend_from_slice(&p.vertices());
    }

    // Upload and draw
    let buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;
    // ... render as usual
}
```

## Performance Considerations

With 1000 particles, we're drawing 6000 vertices per frame. This is handled efficiently by:

1. **Batch rendering** - All particles in one draw call
2. **Buffer recreation** - Simple but effective for dynamic data
3. **GPU parallelism** - Vertex/fragment processing is parallel

For more particles (10,000+), consider:
- Instance rendering
- Compute shader particle updates
- GPU-driven culling

## Variations

### Confetti

```rust
fn new_confetti() -> Self {
    Self {
        vx: (random() - 0.5) * 0.02,
        vy: -0.005 + random() * 0.015,  // Some float up
        color: random_bright_color(),
        // ...
    }
}
```

### Fireworks

```rust
fn new_spark(origin_x: f32, origin_y: f32) -> Self {
    let angle = random() * 2.0 * PI;
    let speed = 0.01 + random() * 0.02;
    Self {
        x: origin_x,
        y: origin_y,
        vx: angle.cos() * speed,
        vy: angle.sin() * speed - 0.001,  // Gravity
        // ...
    }
}
```

## Full Source

See `goldy/examples/particles.rs` for the complete code.

