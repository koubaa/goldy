# Goldy vs wgpu

Both Goldy and wgpu are Rust GPU libraries, but they serve different purposes and make different tradeoffs.

## Quick Comparison

| Aspect | wgpu | Goldy |
|--------|------|-----|
| **Primary Goal** | WebGPU implementation | Modern GPU simplicity |
| **Governance** | W3C WebGPU spec | Independent |
| **Browser Support** | Yes (via WebGPU) | No |
| **Minimum Hardware** | Wide compatibility | Modern only (2018+) |
| **API Complexity** | Medium | Low |
| **Iteration Speed** | Spec-bound | Fast |

## When to Use wgpu

Choose wgpu when you need:

- **Web deployment** - wgpu compiles to WebGPU for browsers
- **Maximum compatibility** - Supports older GPUs and drivers
- **Ecosystem** - Large community, many examples, good tooling
- **Stability** - Spec-driven API won't change unexpectedly
- **Production-ready** - Battle-tested in real applications

## When to Use Goldy

Choose Goldy when you need:

- **Simplicity** - Minimal API surface, fewer concepts
- **Modern features** - Assume bindless, dynamic rendering, etc.
- **Fast iteration** - API can evolve without committee approval
- **Native performance** - No abstraction layers for translation
- **WASI integration** - Designed for sandboxed GPU access

## Architecture Differences

### wgpu

```
Application
    │
    ▼
┌─────────────────┐
│     wgpu        │  ◄── WebGPU API
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│wgpu-  │ │wgpu-  │
│hal    │ │core   │
└───┬───┘ └───────┘
    │
    ▼
Vulkan/Metal/DX12/WebGPU
```

wgpu implements the WebGPU specification, which is designed as a lowest-common-denominator across all platforms including the web.

### Goldy

```
Application
    │
    ▼
┌─────────────────┐
│     Goldy       │  ◄── Native Rust API
└────────┬────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
  Vk  Metal  DX12
 1.4+   2+
```

Goldy talks directly to modern backend APIs without a translation layer. Each backend uses native idioms.

## Code Comparison

### Creating a Buffer

**wgpu:**
```rust
let buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("Vertex Buffer"),
    size: data.len() as u64,
    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,  // wgpu uses bitflags
    mapped_at_creation: false,
});
queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data));
```

**Goldy:**
```rust
let buffer = Buffer::with_data(&device, &data, DataAccess::Scattered)?;
```

### Render Pass

**wgpu:**
```rust
let mut encoder = device.create_command_encoder(&Default::default());
{
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, buffer.slice(..));
    pass.draw(0..3, 0..1);
}
queue.submit(std::iter::once(encoder.finish()));
```

**Goldy:**
```rust
let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    pass.clear(Color::BLACK);
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &buffer);
    pass.draw(0..3, 0..1);
}
let output = frame.render(encoder)?;
```

## Summary

```
                    Legacy Support    Speed    Simplicity
                    ──────────────    ─────    ──────────
wgpu                ████████          ██████   ██████
Goldy               ██                ████████ ████████
```

**Use wgpu** for production applications that need wide compatibility.

**Use Goldy** for new projects targeting modern hardware where simplicity matters.

Both are valid choices—pick the one that fits your constraints.

