# Command Encoding

Commands are recorded into an encoder, then executed by rendering a frame.

## Command Encoder

```rust
use rag::CommandEncoder;

let mut encoder = CommandEncoder::new();
```

The encoder records GPU commands without executing them immediately.

## Render Pass

A render pass groups drawing commands:

```rust
let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    
    // Drawing commands go here
    pass.clear(Color::BLACK);
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &vertices);
    pass.draw(0..3, 0..1);
    
} // Pass ends when `pass` is dropped
```

## Render Pass Commands

### Clear

Clear the render target:

```rust
pass.clear(Color::BLACK);
pass.clear(Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 });
```

### Set Pipeline

Bind a render pipeline:

```rust
pass.set_pipeline(&pipeline);
```

### Set Vertex Buffer

Bind a vertex buffer to a slot:

```rust
pass.set_vertex_buffer(0, &buffer);  // Slot 0

// Multiple buffers (for instancing)
pass.set_vertex_buffer(0, &vertices);
pass.set_vertex_buffer(1, &instances);
```

### Draw

Draw primitives:

```rust
// draw(vertices, instances)
pass.draw(0..3, 0..1);      // 3 vertices, 1 instance
pass.draw(0..100, 0..1);    // 100 vertices, 1 instance
pass.draw(0..6, 0..10);     // 6 vertices, 10 instances
```

### Draw Indexed (future)

```rust
pass.set_index_buffer(&indices, IndexFormat::Uint16);
pass.draw_indexed(0..36, 0, 0..1);  // indices, base vertex, instances
```

## Command Order

Commands are executed in order within a pass:

```rust
{
    let mut pass = encoder.begin_render_pass();
    
    // 1. Clear first
    pass.clear(Color::BLACK);
    
    // 2. Draw background
    pass.set_pipeline(&bg_pipeline);
    pass.set_vertex_buffer(0, &bg_vertices);
    pass.draw(0..6, 0..1);
    
    // 3. Draw foreground (on top)
    pass.set_pipeline(&fg_pipeline);
    pass.set_vertex_buffer(0, &fg_vertices);
    pass.draw(0..triangle_count, 0..1);
}
```

## Multiple Draw Calls

You can issue multiple draws per pass:

```rust
{
    let mut pass = encoder.begin_render_pass();
    pass.clear(Color::BLACK);
    pass.set_pipeline(&pipeline);
    
    // Draw each object
    for (buffer, count) in objects {
        pass.set_vertex_buffer(0, buffer);
        pass.draw(0..count, 0..1);
    }
}
```

## Multiple Passes (future)

For effects requiring multiple render targets:

```rust
// Pass 1: Render scene to texture
{
    let mut pass = encoder.begin_render_pass_to(&scene_texture);
    pass.set_pipeline(&scene_pipeline);
    // ...
}

// Pass 2: Post-process
{
    let mut pass = encoder.begin_render_pass();
    pass.set_pipeline(&postprocess_pipeline);
    // Use scene_texture as input
    // ...
}
```

## Executing Commands

Commands are executed when you render to a surface or render target:

```rust
// For window display (zero-copy)
let frame = surface.acquire()?;
frame.render(encoder)?;
surface.present(frame)?;

// For headless/streaming (with optional CPU readback)
let target = RenderTarget::new(&device, width, height, format)?;
target.render(encoder)?;
let pixels = target.read_to_cpu()?;  // Only when needed
```

## Best Practices

### Batch Similar Draws

```rust
// Good: few state changes
pass.set_pipeline(&pipeline);
for object in &objects {
    pass.set_vertex_buffer(0, &object.vertices);
    pass.draw(0..object.count, 0..1);
}

// Less efficient: many state changes
for object in &objects {
    pass.set_pipeline(&object.pipeline);  // Changes every iteration
    pass.set_vertex_buffer(0, &object.vertices);
    pass.draw(0..object.count, 0..1);
}
```

### Minimize Clear Operations

```rust
// Clear once at start
pass.clear(Color::BLACK);

// Draw all objects
for object in &objects {
    pass.draw(...);
}
```

### Use Appropriate Draw Counts

```rust
// Good: draw what you need
pass.draw(0..actual_vertex_count, 0..1);

// Wasteful: drawing extra vertices
pass.draw(0..buffer_capacity, 0..1);
```

## Debug Tips

If nothing renders:

1. Check vertex buffer data
2. Check pipeline topology matches your data
3. Check shader locations match vertex layout
4. Ensure `draw()` range is correct
5. Check clear color isn't same as object color

