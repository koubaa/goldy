# Command Encoding

`CommandEncoder` records GPU rendering commands without executing them. It is completely lock-free and does not touch the GPU backend — you can create and fill encoders on any thread. The actual GPU work happens when you submit the commands through `Frame::render()` or `RenderTarget::render()`.

## Recording Commands

```rust
use goldy::{CommandEncoder, Color};

let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    pass.clear(Color::CORNFLOWER_BLUE);
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &vertices);
    pass.draw(0..3, 0..1);
} // pass ends when dropped

let commands = encoder.finish();
```

## Render Pass

A `RenderPass` is a borrow of the encoder that groups drawing commands. It begins with `begin_render_pass()` and ends when the `RenderPass` value is dropped.

```rust
let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    // all draw commands go here
}
```

Commands within a pass execute in recorded order.

### Clearing

Clear the color attachment, the depth buffer, or both:

```rust
pass.clear(Color::BLACK);
pass.clear_depth(1.0);  // standard depth clear (far plane)
pass.clear_depth(0.0);  // reverse-Z depth clear
```

### Setting the Pipeline

Bind the active `RenderPipeline`. You can switch pipelines within the same pass.

```rust
pass.set_pipeline(&scene_pipeline);
// ... draw scene ...

pass.set_pipeline(&ui_pipeline);
// ... draw UI ...
```

### Vertex and Index Buffers

Bind vertex data to a numbered slot. Both `Buffer` and `BufferView` are accepted — for pool-allocated views, the parent buffer and offset are resolved automatically.

```rust
pass.set_vertex_buffer(0, &vertex_buffer);

// With an explicit additional offset:
pass.set_vertex_buffer_offset(0, &vertex_buffer, byte_offset);
```

Bind an index buffer for indexed drawing:

```rust
use goldy::IndexFormat;

pass.set_index_buffer(&index_buffer, IndexFormat::Uint16);

// With an additional offset:
pass.set_index_buffer_offset(&index_buffer, byte_offset, IndexFormat::Uint32);
```

### Binding Resources

Goldy's bindless model passes resource indices to shaders through push constants. There are three binding methods:

**Typed handles** (preferred for new code) — each handle carries its `BindlessCategory`, enabling validation against shader reflection:

```rust
let tex = texture.bindless_handle().unwrap();
let samp = sampler.bindless_handle().unwrap();
pass.bind_resources_typed(&[tex, samp]);
```

**Buffer references** — extracts bindless indices from `Buffer` objects:

```rust
pass.bind_resources(&[&uniform_buffer, &data_buffer]);
```

**Raw indices** — for manual control or when mixing resource types:

```rust
let tex_idx = texture.bindless_index().unwrap();
let samp_idx = sampler.bindless_index().unwrap();
pass.bind_resources_raw(&[tex_idx, samp_idx]);
```

Raw indices can also carry user scalars alongside bindless indices:

```rust
pass.bind_resources_raw_with_user(
    &[buf_idx, tex_idx],  // bindless indices (region A)
    &[frame_number],      // user scalars (region B)
);
```

## Draw Calls

### draw

Draw non-indexed primitives:

```rust
// draw(vertex_range, instance_range)
pass.draw(0..3, 0..1);      // 3 vertices, 1 instance
pass.draw(0..6, 0..10);     // 6 vertices, 10 instances
pass.draw(100..106, 0..1);  // 6 vertices starting at vertex 100
```

### draw_indexed

Draw indexed primitives. Requires a prior `set_index_buffer()` call.

```rust
// draw_indexed(index_range, base_vertex, instance_range)
pass.draw_indexed(0..36, 0, 0..1);

// base_vertex is added to each index before vertex fetch
pass.draw_indexed(0..6, 1000, 0..1);

// negative base_vertex is allowed
pass.draw_indexed(0..3, -50, 0..1);
```

### draw_fullscreen

Draw a fullscreen triangle (3 vertices, no vertex buffer needed). Pair with `vs_fullscreen_triangle()` from `goldy_exp.vertex` or `fullscreen_position()`/`fullscreen_uv()` from `goldy_exp.primitives`.

```rust
pass.set_pipeline(&fullscreen_pipeline);
pass.bind_resources(&[&uniform_buffer]);
pass.draw_fullscreen();
```

This is more efficient than a fullscreen quad (3 vertices vs 6) and eliminates vertex buffer overhead entirely.

### draw_quads

Draw N instanced quads (6 vertices each, no vertex buffer needed). The shader reads per-instance data from a buffer and uses `quad_position()` from `goldy_exp.primitives` to generate vertex positions.

```rust
pass.set_pipeline(&instanced_pipeline);
pass.bind_resources(&[&instance_buffer]);
pass.draw_quads(400);  // draw 400 quads
```

## Submitting Commands

After recording, submit the encoder to a surface frame or render target:

```rust
// Surface presentation
let frame = surface.begin()?;
frame.render(encoder)?;
frame.present()?;

// Headless render target
target.render(encoder)?;
```

## Complete Example

```rust
let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();

    pass.clear(Color::BLACK);
    pass.clear_depth(1.0);

    // Draw opaque geometry
    pass.set_pipeline(&scene_pipeline);
    pass.set_vertex_buffer(0, &mesh_vertices);
    pass.set_index_buffer(&mesh_indices, IndexFormat::Uint32);
    pass.bind_resources(&[&camera_uniforms]);
    pass.draw_indexed(0..index_count, 0, 0..1);

    // Draw fullscreen post-process
    pass.set_pipeline(&post_pipeline);
    pass.bind_resources(&[&post_uniforms]);
    pass.draw_fullscreen();
}

let frame = surface.begin()?;
frame.render(encoder)?;
frame.present()?;
```

## Best Practices

- **Batch draws by pipeline.** Pipeline switches are cheap but not free. Group objects that share the same pipeline.
- **Clear once per pass.** Issue `clear()` at the start, then draw everything.
- **Use convenience methods.** `draw_fullscreen()` and `draw_quads()` avoid unnecessary vertex buffer allocations.
- **Encode on any thread.** `CommandEncoder` is lock-free; build command buffers in parallel if needed.
