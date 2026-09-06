# Rendering Outputs

Windowed rendering uses **present-on-scheme**: [`SurfaceExchange`](https://docs.rs/goldy/latest/goldy/struct.SurfaceExchange.html) + [`Transaction`](https://docs.rs/goldy/latest/goldy/struct.Transaction.html) + [`Claim`](https://docs.rs/goldy/latest/goldy/struct.Claim.html). Record copy or compute-to-present once in a retained [`Scheme`](https://docs.rs/goldy/latest/goldy/struct.Scheme.html); submit each frame and settle the claim.

CPU compute (`GOLDY_BACKEND=cpu`) has no swapchain. Headless or foreign present
uses [`PixelExchange`](./pixel-exchange.md): the scheme writes a buffer pixmap and
consume blits into a [`PixelSink`](https://docs.rs/goldy/latest/goldy/trait.PixelSink.html).

All windowed Rust examples use the `SurfaceExchange` path.

## SurfaceExchange

A `SurfaceExchange` wraps the platform window and records how scheme output reaches the swapchain:

```rust
use goldy::{SurfaceExchange, SurfaceConfig, PresentMode, DepthFormat};

let surface = SurfaceExchange::new(&ctx, &window)?;

// With explicit configuration and in-flight depth
let surface = SurfaceExchange::new_with_depth(
    &ctx,
    &window,
    3,
    SurfaceConfig {
        present_mode: PresentMode::Fifo,
        depth_format: Some(DepthFormat::Depth32Float),
    },
)?;
```

Depth testing uses an offscreen scheme-leased render target, not the swapchain drawable.

### Bind helpers

| Method | Use |
|--------|-----|
| `bind_render_target(scheme, scene_rt)` | Offscreen render pass → surface copy |
| `bind(scheme, texture)` | Texture → surface copy |
| `bind_destination(scheme)` | Compute or other direct writes via `with_present(&lease)` |

Each bind returns a reusable [`Transaction`](https://docs.rs/goldy/latest/goldy/struct.Transaction.html). After `scheme.submit()`, extract the per-frame claim with `transaction.claim(&mut submission)?` and settle with `claim.consume()`.

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
surface.set_present_mode(PresentMode::Immediate)?;
let current = surface.present_mode();
```

## Present-on-Scheme Frame Cycle

Record once at init (and on resize), submit each frame:

```rust
let mut pass = scheme.render_pass("main", &scene_rt, TargetLoad::Clear(Color::CORNFLOWER_BLUE));
pass.with_parcel(&vertex_buffer, NodeAccess::Read);
pass.set_pipeline(&pipeline);
pass.set_vertex_buffer(0, &vertices);
pass.draw(0..3, 0..1);
pass.finish();
let present = surface.bind_render_target(&mut scheme, &scene_rt)?;

// Each frame:
let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

For pure compute-to-surface, use `bind_destination` and bind the returned lease in a compute node with `with_present(&lease)` instead of a render pass + copy.

## Surface Queries

```rust
surface.width();
surface.height();
surface.size();        // (width, height)
surface.format();      // TextureFormat of the swapchain images
```

Always use `surface.format()` when creating pipelines to ensure a match:

```rust
let desc = RenderPipelineDesc {
    target_format: surface.format(),
    ..Default::default()
};
```

## Resize Handling

Call `resize()` when the window size changes. Zero-size dimensions are silently
ignored (common during window minimize). Rebuild the scheme when
`surface.size()` changes.

`SurfaceExchange::resize` records the new extent immediately (and advances the
pool generation) but defers the DXGI/`ResizeBuffers` work until the next
drawable acquire. A burst of window-size events therefore only pays for one
structural rebuild per presented frame.

```rust
surface.resize(width, height)?;
// rebuild scheme + transaction using surface.size()
```

## Transaction Lifetime

- Record a bind (`bind_render_target`, `bind`, or `bind_destination`) once when building the scheme.
- Each frame: `scheme.submit()` then `transaction.claim(&mut submission)?.consume()?`.
- Each submission may be claimed at most once per transaction.

```rust
let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
// claim consumed — do not reuse this submission's claim slot
```
