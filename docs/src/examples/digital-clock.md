# Digital Clock Example

A real-time 7-segment display clock.

## Run It

```bash
cargo run --example digital_clock --release
```

## Controls

| Key | Action |
|-----|--------|
| **Space** | Pause/resume timer |
| **Click** | Change color |
| **C** | Change color |
| **Escape** | Exit |

## What It Demonstrates

- Procedural geometry generation
- Real-time vertex buffer updates
- 7-segment display rendering
- User input handling
- Color cycling

## Key Concepts

### 7-Segment Display

Each digit is made of 7 segments that can be on or off:

```
 ─── (0)
│   │
(1) (2)
│   │
 ─── (3)
│   │
(4) (5)
│   │
 ─── (6)
```

Segment patterns for digits 0-9:

```rust
const SEGMENT_PATTERNS: [[bool; 7]; 10] = [
    [true, true, true, false, true, true, true],     // 0
    [false, false, true, false, false, true, false], // 1
    [true, false, true, true, true, false, true],    // 2
    // ... etc
];
```

### Vertex Generation

Each segment is a quad (2 triangles = 6 vertices):

```rust
fn quad_vertices(x: f32, y: f32, w: f32, h: f32, color: Color) -> [Vertex2D; 6] {
    [
        Vertex2D::new(x, y, color),
        Vertex2D::new(x + w, y, color),
        Vertex2D::new(x + w, y + h, color),
        Vertex2D::new(x, y, color),
        Vertex2D::new(x + w, y + h, color),
        Vertex2D::new(x, y + h, color),
    ]
}
```

### Time Display

```rust
let elapsed = self.elapsed_secs();
let hours = ((elapsed / 3600) % 100) as u8;
let minutes = ((elapsed % 3600) / 60) as u8;
let seconds = (elapsed % 60) as u8;

let digits: [u8; 8] = [
    hours / 10, hours % 10,
    10, // colon (special case)
    minutes / 10, minutes % 10,
    10, // colon
    seconds / 10, seconds % 10,
];
```

### Color Palette

```rust
const COLORS: [Color; 8] = [
    Color { r: 1.0, g: 0.1, b: 0.1, a: 1.0 },    // Red
    Color { r: 1.0, g: 0.65, b: 0.0, a: 1.0 },   // Orange
    Color { r: 1.0, g: 1.0, b: 0.0, a: 1.0 },    // Yellow
    Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 },    // Green
    Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 },    // Cyan
    Color { r: 0.0, g: 0.5, b: 1.0, a: 1.0 },    // Blue
    Color { r: 0.5, g: 0.0, b: 1.0, a: 1.0 },    // Purple
    Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 },    // Magenta
];
```

## Rendering Flow

1. Calculate current time → digit values
2. For each digit, check segment pattern
3. Generate quads for lit segments
4. Convert pixel coordinates to NDC
5. Upload all vertices to GPU
6. Draw in single call

```rust
fn render_frame(&mut self) -> anyhow::Result<()> {
    let vertices = generate_clock_vertices(elapsed, color, width, height);
    let vertex_buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;
    
    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(bg_color);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.draw(0..vertices.len() as u32, 0..1);
    }
    // ...
}
```

## Full Source

See `goldy/examples/digital_clock.rs` for the complete code.

