# Your First Triangle

This tutorial draws a colored triangle in a window using Goldy's render pipeline and present-on-scheme API (`SurfaceExchange` + `Scheme` + `Transaction`).

See [`examples/triangle.rs`](../examples/triangle.md) for the full source.

## Recording the Scheme

Once at init (and again on resize), record a retained scheme: offscreen render pass → copy to surface via [`SurfaceExchange::bind_render_target`](../surfaces/overview.md).

```rust
use goldy::{
    shader::builtins, Buffer, BufferKind, Color, DeviceDescriptor, Instance, Lease, LeaseRenderTarget,
    NodeAccess, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool,
    Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, Transaction, Vertex2D,
};

fn record_scheme(
    scheme: &mut Scheme,
    surface: &SurfaceExchange,
    pipeline: &RenderPipeline,
    vertex_buffer: &Buffer,
    scene_rt: &Lease<LeaseRenderTarget>,
    bg_color: Color,
) -> anyhow::Result<Transaction> {
    let mut pass = scheme.render_pass("triangle", scene_rt, TargetLoad::Clear(bg_color));
    pass.with_parcel(vertex_buffer, NodeAccess::Read);
    pass.set_pipeline(pipeline);
    pass.set_vertex_buffer(0, vertex_buffer);
    pass.draw(0..3, 0..1);
    pass.finish();
    surface.bind_render_target(scheme, scene_rt).map_err(Into::into)
}
```

## Per-Frame Submit

Each frame submits the retained scheme and consumes the surface claim:

```rust
let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

## Walkthrough

### Instance, Device, and Context

```rust
let instance = Instance::new()?;
let device = Arc::new(
    instance
        .request_adapter(&RequestAdapterOptions::default())?
        .request_device(&DeviceDescriptor::default())?,
);
let ctx = device.create_context()?;
```

`Instance` discovers available GPUs. `create_context` opens the submission context used by `Scheme`.

### Vertex Buffer

```rust
let vertices = [
    Vertex2D::new(0.0, -0.5, Color::RED),
    Vertex2D::new(-0.5, 0.5, Color::GREEN),
    Vertex2D::new(0.5, 0.5, Color::BLUE),
];
let mut pool = RetainedPool::new(device.clone());
let vertex_buffer = pool.acquire_buffer_with_data(&vertices, BufferKind::Scattered)?;
```

`Vertex2D` is a built-in vertex type with position and color. Keep the pool alive for the buffer's lifetime.

### Shader and Pipeline

```rust
let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
let surface = SurfaceExchange::new(&ctx, window.as_ref(), SurfaceConfig::default())?;
let pipeline = RenderPipeline::new(
    &device,
    &shader,
    &shader,
    &RenderPipelineDesc {
        vertex_layout: Vertex2D::layout(),
        target_format: surface.format(),
        ..Default::default()
    },
)?;
```

`builtins::VERTEX_COLOR_2D` uses `[goldy_vertex]` and `[goldy_fragment]` virtual entry points from the `goldy_exp` library.

### Surface and Presentation

```rust
let mut scheme = Scheme::new(&ctx);
let scene_rt = scheme.lease_render_target(width, height, surface.format(), None)?;
let present = record_scheme(&mut scheme, &surface, &pipeline, &vertex_buffer, &scene_rt, bg_color)?;

// Each frame:
let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

`SurfaceExchange` manages the OS swapchain. Scene color is rendered to a scheme-leased offscreen target, copied to the drawable, and displayed when the claim is consumed. Rendering stays on the GPU — no CPU readback.

On resize, rebuild the scheme and transaction with the new dimensions (see `examples/triangle.rs`).

## Run It

```bash
cargo run --example triangle
```

You should see a window with a colored triangle on a dark blue background.

## Next Steps

- [Your First Compute Shader](./first-compute.md) — bypass the graphics pipeline entirely
- [Examples Gallery](../examples/gallery.md) — more complex demos
