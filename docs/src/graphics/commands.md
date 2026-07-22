# Scheme Rendering

Graphics commands are recorded through [`Scheme`](https://docs.rs/goldy/latest/goldy/struct.Scheme.html) render-pass nodes. The scheme declares resource dependencies (buffers, textures, parcels) and is retained across submissions.

Windowed apps render to an offscreen leased target, then present via [`SurfaceExchange`](../surfaces/overview.md). See [`examples/triangle.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/triangle.rs) for the canonical loop.

## Per-Frame Loop (windowed)

```rust
use goldy::{Color, NodeAccess, Scheme, SurfaceExchange, TargetLoad};

// Record once at init (and on resize):
let mut pass = scheme.render_pass("main", &scene_rt, TargetLoad::Clear(Color::CORNFLOWER_BLUE));
pass.with_parcel(&vertex_buffer, NodeAccess::Read);
pass.set_pipeline(&pipeline);
pass.set_vertex_buffer(0, &vertex_buffer);
pass.draw(0..3, 0..1);
pass.finish();
let present = surface.bind_render_target(&mut scheme, &scene_rt)?;

// Each frame:
let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

## Render Pass Builder

`render_pass(label, target, color_load)` returns a builder that records draw commands for one offscreen leased target. Color load is declared on the pass, not as a command-list clear:

- `TargetLoad::Load` — preserve prior color contents
- `TargetLoad::Clear(color)` — clear to a color at pass begin (private-inaugural)
- `TargetLoad::Discard` — prior color contents irrelevant; draws fully overwrite

### Depth clear

```rust
pass.clear_depth(1.0);
```

### Pipeline and Buffers

```rust
pass.set_pipeline(&pipeline);
pass.set_vertex_buffer(0, &vertices);
pass.set_index_buffer(&indices, IndexFormat::Uint16);
// Shader resources: with_parcel / with_shader_resources (samplers) before set_pipeline
```

### Drawing

```rust
pass.draw(0..3, 0..1);              // non-indexed
pass.draw_indexed(0..6, 0..1);      // indexed
pass.draw_fullscreen();             // 3-vertex fullscreen triangle
pass.draw_quads(4);                 // instanced quads
```

### Scheme Dependencies

Declare which resources the pass reads or writes so the runtime can track parcel lifetimes:

```rust
pass.with_parcel(&buf, NodeAccess::Read);
pass.with_texture(&tex, NodeAccess::Read);
```

Call `finish()` when done recording commands for this pass node.

## Offscreen-Only (tests, readback)

For headless rendering without a window, record a scheme, copy to a readback texture, and consume a read grant:

```rust
let mut scheme = Scheme::new(&ctx);
let rt = scheme.lease_render_target(800, 600, TextureFormat::Rgba8Unorm, None)?;
let mut pass = scheme.render_pass("clear", &rt, TargetLoad::Clear(Color::RED));
pass.finish();
scheme.copy_to_texture(&rt, &readback_texture);
let grant = scheme.grant_read_texture(&readback_texture);
let submission = scheme.submit()?;
let pixels = grant.consume(&submission)?;
```

## Hybrid Compute + Graphics

Put compute `node` dispatches and `render_pass` nodes in the **same** scheme, then submit once:

```rust
scheme.commit_write_parcel(&staging, 0, data)?;
scheme.node("sim", &compute_pipeline)
    .with_parcel(&state_buf, NodeAccess::ReadWrite)
    .dispatch(wg, 1, 1);
let mut pass = scheme.render_pass("draw", &scene_rt, TargetLoad::Discard);
// ...
let present = surface.bind_render_target(&mut scheme, &scene_rt)?;
let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

## Notes

Render-pass recording is scheme-owned: builders must call `finish()` (Rust) or drop/RAII finish (FFI bindings) before submitting.
