# Rendering Outputs

`Surface` manages a swapchain for zero-copy GPU-to-display presentation. It wraps the platform window handle, acquires drawable textures each frame, and presents finished frames to the display.

## Creating a Surface

A `Surface` requires a `Device` and a window that implements `HasWindowHandle + HasDisplayHandle` (from the `raw-window-handle` crate).

```rust
use goldy::{Surface, SurfaceConfig, PresentMode, DepthFormat};

// Simplest form — Auto present mode, no depth buffer
let surface = Surface::new(&device, &window)?;

// With explicit configuration
let surface = Surface::new_with_config(&device, &window, SurfaceConfig {
    present_mode: PresentMode::Fifo,
    depth_format: Some(DepthFormat::Depth32Float),
})?;

```

Depth testing uses an offscreen `RenderTarget::new_with_depth` in the task graph, not the swapchain surface.

## SurfaceConfig

```rust
pub struct SurfaceConfig {
    pub present_mode: PresentMode,
    pub depth_format: Option<DepthFormat>,
}
```

| Field | Purpose | Default |
|-------|---------|---------|
| `present_mode` | Vsync strategy | `Auto` |
| `depth_format` | Depth buffer format, or `None` to disable | `None` |

## Present Modes

| Mode | Behavior | Backend Mapping |
|------|----------|-----------------|
| `Fifo` | Vsync — wait for display refresh. No tearing, capped at monitor Hz. | Metal `displaySyncEnabled=YES`, Vulkan `FIFO`, DX12 `Present(1)` |
| `Mailbox` | Triple-buffered — latest frame queued, older dropped. Low latency + no tearing. | Vulkan `MAILBOX`. Falls back to `Fifo` on Metal and some DX12 configurations. |
| `Immediate` | No sync, may tear. Maximum throughput for benchmarks. | Metal `displaySyncEnabled=NO`, Vulkan `IMMEDIATE`, DX12 `Present(0)` |
| `Auto` | Goldy chooses (`Mailbox` if available, then `Fifo`). | — |

Change the present mode at runtime without recreating the surface:

```rust
surface.set_present_mode(PresentMode::Immediate)?;
let current = surface.present_mode();
```

## Frame Acquisition Cycle

Each frame builds a task graph, blits an offscreen render target to the swapchain, then presents:

```rust
loop {
    frame_graph.clear();

    let mut pass = frame_graph.render_pass("main", &scene_rt);
    pass.bind_buffer_mut(&vertices, NodeAccess::Read);
    pass.clear(Color::CORNFLOWER_BLUE);
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &vertices);
    pass.draw(0..3, 0..1);
    pass.finish_recorded();

    let swapchain = frame_graph.declare_swapchain_output();
    frame_graph.copy_render_target_to_swapchain(&scene_rt, swapchain);

    let frame = surface.begin()?;
    let frame = surface.submit_graph_to_frame(&mut frame_graph, frame)?;
    frame.present()?;
}
```

`surface.acquire()` is a legacy alias for `surface.begin()`.

## Frame

`Frame` represents a single swapchain image bracket. It tracks whether the frame has been presented and auto-presents on drop if you forget.

### Frame Properties

```rust
let frame = surface.begin()?;

frame.width();   // frame dimensions (may differ from surface after resize)
frame.height();
```

### Graphics Path — submit_graph_to_frame

Record draw commands in `RenderPassBuilder` nodes, then submit the graph with the acquired frame:

```rust
let frame = surface.submit_graph_to_frame(&mut frame_graph, frame)?;
frame.present()?;
```

### Compute Path — Frame::submit_compute

For compute-to-surface workflows, access the frame's texture directly and submit a `TaskGraph`:

```rust
let frame = surface.begin()?;
let tex = frame.texture();  // the swapchain texture as a storage image

// Build a task graph that writes to tex...
frame.submit_compute(&task_graph)?;
frame.present()?;
```

`frame.texture()` returns a `&Texture` with `TextureKind::Direct`, suitable for binding as a storage image in compute shaders.

### Presenting

`frame.present()` consumes the `Frame`, submits all recorded work, and queues the image for display. It returns a `TimelineValue` that can be used with `Device::wait_until()`.

```rust
let timeline = frame.present()?;
```

If a `Frame` is dropped without calling `present()`, it auto-presents to avoid leaking the swapchain image. This is safe but wastes a frame.

## Surface Queries

```rust
surface.width();
surface.height();
surface.size();        // (width, height)
surface.format();      // TextureFormat of the swapchain images

// Validate that a pipeline's target format matches
surface.validate_pipeline_format(pipeline_format)?;
```

## Resize Handling

Call `resize()` when the window size changes. Zero-size dimensions are silently ignored (common during window minimize).

```rust
fn on_resize(surface: &mut Surface, width: u32, height: u32) -> Result<()> {
    surface.resize(width, height)?;
    Ok(())
}
```

## Texture Format

The swapchain format is chosen by the backend at surface creation (typically `Bgra8UnormSrgb`). Always use `surface.format()` when creating pipelines to ensure a match:

```rust
let desc = RenderPipelineDesc {
    target_format: surface.format(),
    ..Default::default()
};
```

## Frame Lifetime

`Frame` follows Rust ownership semantics:

- `begin()` acquires the swapchain image and returns a `Frame`
- `texture()` borrows the frame — valid until `present()` is called
- `present()` consumes the `Frame` — the borrow checker prevents use-after-present
- Dropping without presenting auto-presents (prevents swapchain deadlock)

```rust
let frame = surface.begin()?;
let tex = frame.texture();
// tex is valid here
frame.present()?;
// tex is now invalid — Rust prevents accessing it
```
