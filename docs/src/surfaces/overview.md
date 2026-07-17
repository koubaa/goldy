# Rendering Outputs

Goldy supports two presentation paths:

- **Present-on-scheme** (recommended) — [`SwapchainPool`](https://docs.rs/goldy/latest/goldy/struct.SwapchainPool.html) + [`PresentLease`](https://docs.rs/goldy/latest/goldy/struct.PresentLease.html) + [`PresentGrant`](https://docs.rs/goldy/latest/goldy/struct.PresentGrant.html). Record copy or compute-to-present once in a retained [`Scheme`](https://docs.rs/goldy/latest/goldy/struct.Scheme.html); submit each frame.
- **Legacy Surface API** — [`Surface`](https://docs.rs/goldy/latest/goldy/struct.Surface.html) acquire/present bracket (still used internally by `SwapchainPool`).

All windowed Rust examples use present-on-scheme.

## SwapchainPool and PresentLease

A `SwapchainPool` wraps the platform window and supplies drawable backings for stable present leases:

```rust
use goldy::{SwapchainPool, SurfaceConfig, PresentMode, DepthFormat};

let swapchain = SwapchainPool::new(&ctx, &window, 3)?;

// With explicit configuration
let swapchain = SwapchainPool::new_with_config(
    &ctx,
    &window,
    3,
    SurfaceConfig {
        present_mode: PresentMode::Fifo,
        depth_format: Some(DepthFormat::Depth32Float),
    },
)?;
let screen = swapchain.lease();
```

Depth testing uses an offscreen scheme-leased render target, not the swapchain drawable.

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

Change the present mode at runtime:

```rust
swapchain.set_present_mode(PresentMode::Immediate)?;
let current = swapchain.present_mode();
```

## Present-on-Scheme Frame Cycle

Record once at init (and on resize), submit each frame:

```rust
let mut pass = scheme.render_pass("main", &scene_rt);
pass.with_parcel(&vertices, NodeAccess::Read);
pass.clear(Color::CORNFLOWER_BLUE);
pass.set_pipeline(&pipeline);
pass.set_vertex_buffer(0, &vertices);
pass.draw(0..3, 0..1);
pass.finish();
scheme.copy_to_present(&scene_rt, &screen);
let present = scheme.grant_present(&screen);

// Each frame:
let submission = scheme.submit()?;
present.consume(&submission)?;
```

For pure compute-to-surface, bind the present lease in a compute node with `with_present(&screen)` instead of a render pass + copy.

## Legacy Surface API

`Surface` manages a swapchain for direct acquire/present workflows. It wraps the platform window handle and acquires drawable textures each frame.

```rust
let surface = Surface::new(&device, &window)?;
let frame = surface.begin()?;
// ... lower-level frame bracket ...
frame.present()?;
```

New applications should prefer `SwapchainPool` + `Scheme`. See [`examples/triangle.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/triangle.rs).

## Swapchain Queries

```rust
swapchain.width();
swapchain.height();
swapchain.size();        // (width, height)
swapchain.format();      // TextureFormat of the swapchain images
```

Always use `swapchain.format()` when creating pipelines to ensure a match:

```rust
let desc = RenderPipelineDesc {
    target_format: swapchain.format(),
    ..Default::default()
};
```

## Resize Handling

Call `resize()` when the window size changes. Zero-size dimensions are silently ignored (common during window minimize). Rebuild the scheme and present grant when dimensions change.

```rust
swapchain.resize(width, height)?;
// rebuild scheme + present grant
```

## Present Grant Lifetime

- Record `grant_present(&screen)` once when building the scheme.
- Each frame: `scheme.submit()` then `present.consume(&submission)`.
- Each submission may be consumed at most once per grant.

```rust
let submission = scheme.submit()?;
present.consume(&submission)?;
// submission consumed — do not reuse
```
