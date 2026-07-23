# Render Pass Nodes

Goldy has no command buffers and no command lists. A graphics draw is a **render pass node** inside a [`Scheme`](https://docs.rs/goldy/latest/goldy/struct.Scheme.html) — the same retained dependency graph that holds compute dispatches, copies, and present nodes. `scheme.render_pass(...)` returns a builder; what you call on that builder is *recorded* into the node, not executed immediately. Nothing touches the GPU until `scheme.submit()`.

This matters for how you think about the API: there is no "encoder" you open and close per frame. You build the graph once — typically at init and on resize — and resubmit it every frame. See [Settlement](../compute/settlement.md) for what happens after `submit()`, and [Pipelines](pipelines.md) for how `RenderPipeline` fits into a pass.

## Recording a Render Pass Node

`scheme.render_pass(label, target, color_load)` opens a builder bound to one leased render target:

```rust
use goldy::{Color, NodeAccess, Scheme, TargetLoad};

let mut pass = scheme.render_pass("triangle", &scene_rt, TargetLoad::Clear(Color::CORNFLOWER_BLUE));
pass.with_parcel(&vertex_buffer, NodeAccess::Read);
pass.set_pipeline(&pipeline);
pass.set_vertex_buffer(0, &vertex_buffer);
pass.draw(0..3, 0..1);
pass.finish();
```

`finish()` pushes the node into the scheme's graph. The builder cannot be reused after `finish()` — record a new pass for the next node.

### Color Load

Load behavior is a property of the pass node, not a separate clear call:

| Variant | Effect |
|---------|--------|
| `TargetLoad::Load` | Preserve prior color contents (the node reads the target) |
| `TargetLoad::Clear(color)` | Clear to `color` at pass start (private-inaugural — the node owns the target outright) |
| `TargetLoad::Discard` | Prior contents are irrelevant; the pass must fully overwrite every pixel |

This is a scheduling input, not cosmetic: `Clear`/`Discard` tell Goldy the pass does not depend on the target's previous contents, which affects how the runtime orders and aliases transient render targets across the scheme.

### Depth

```rust
pass.clear_depth(1.0);
```

Depth clear is declared the same way — as part of the node, before drawing.

## Declaring Dependencies

A render pass node participates in the scheme's dependency graph the same way a compute node does. Declare every parcel it reads or writes so Goldy can derive barriers and track parcel lifetimes:

```rust
pass.with_parcel(&vertex_buffer, NodeAccess::Read);
pass.with_parcel(&uniform_buf, NodeAccess::Read);
```

`with_parcel` also registers the parcel for typed bindless binding, in call order, the next time `set_pipeline` is called — so declare dependencies for a draw *before* calling `set_pipeline` for it.

For a `Buffer` you want to depend on without binding it as a shader resource (e.g. a geometry buffer accessed only through `set_vertex_buffer`/`set_index_buffer`), use `with_buffer_dependency` instead — it registers the dependency without claiming a bindless slot:

```rust
pass.with_buffer_dependency(&geometry, NodeAccess::Read);
```

## Pipeline, Buffers, and Drawing

```rust
pass.set_pipeline(&pipeline);
pass.set_vertex_buffer(0, &vertex_buffer);
pass.set_index_buffer(&indices, IndexFormat::Uint16);

pass.draw(0..3, 0..1);              // non-indexed: vertex range, instance range
pass.draw_indexed(0..6, 0..1);      // indexed: index range, instance range
pass.draw_fullscreen();             // shorthand for draw(0..3, 0..1)
```

`set_pipeline` binds a [`RenderPipeline`](pipelines.md) and, if any parcels were declared with `with_parcel` beforehand, resolves and binds their bindless handles for that pipeline's typed shader parameters. Calling `set_pipeline` again mid-pass starts a new binding scope for subsequent draws — declare each draw's parcels right before the `set_pipeline` call that will consume them.

For fullscreen or procedurally-generated geometry (no vertex buffer at all), skip `set_vertex_buffer` entirely and generate positions from `SV_VertexID` in the shader — see [Vertex Types and Layouts](vertices.md).

## Offscreen-Only (Tests, Readback)

Headless rendering — no window, no `SurfaceExchange` — records the same render pass node, then withdraws pixels through [`MemoryExchange`](../resources/retained-pool.md):

```rust
let memory = MemoryExchange::new(&ctx);
let mut scheme = Scheme::new(&ctx);
let rt = scheme.lease_render_target(800, 600, TextureFormat::Rgba8Unorm, None)?;

let mut pass = scheme.render_pass("clear", &rt, TargetLoad::Clear(Color::RED));
pass.finish();

scheme.copy_to_texture(&rt, &readback_texture);
let withdraw = memory.bind_withdraw(&mut scheme, &readback_texture)?;
let mut submission = scheme.submit()?;
let pixels = withdraw.claim(&mut submission)?.consume()?;
```

## Windowed Rendering

A windowed frame is the same render pass node, plus a present binding recorded once against the scheme via [`SurfaceExchange`](../surfaces/overview.md):

```rust
use goldy::{Color, NodeAccess, Scheme, SurfaceExchange, TargetLoad};

// Record once, at init and on resize:
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

The graph is recorded once; every frame just resubmits it and settles the present claim. See [`examples/triangle.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/triangle.rs) for the full loop, including resize handling (rebuild the scheme and transaction when the surface size changes).

## Compute and Graphics in One Scheme

Because render pass nodes and compute nodes live in the same graph, a hybrid frame is just multiple `node(...)` and `render_pass(...)` calls on one `Scheme`, submitted together:

```rust
let memory = MemoryExchange::new(&ctx);
let deposit = memory.bind_deposit_buffer(&mut scheme, &staging, data.len() as u64)?;
deposit.write(&mut scheme, 0, &data)?;

scheme.node("sim", &compute_pipeline)
    .with_parcel(&state_buf, NodeAccess::ReadWrite)
    .dispatch(wg, 1, 1);

let mut pass = scheme.render_pass("draw", &scene_rt, TargetLoad::Discard);
pass.with_parcel(&state_buf, NodeAccess::Read);
pass.set_pipeline(&pipeline);
pass.set_vertex_buffer(0, &vertex_buffer);
pass.draw(0..3, 0..1);
pass.finish();

let present = surface.bind_render_target(&mut scheme, &scene_rt)?;
let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

Goldy derives the ordering between the compute node and the render pass node from their declared parcel accesses — the simulation's write to `state_buf` is ordered before the pass's read, with no barrier authored by hand.

## Notes

- A render pass builder is single-use: call `finish()` (Rust) once recording is complete, before `scheme.submit()`. In the FFI bindings (C++, .NET, `goldy-ffi-client`), the equivalent is a RAII scope that finishes on drop or block exit.
- A pass node is scoped to one leased render target for its lifetime — draw into a different target by opening a new `render_pass(...)` node.
- Nothing in this page executes anything: recording is pure graph-building. Execution, barrier insertion, and transient aliasing all happen inside `scheme.submit()`.
