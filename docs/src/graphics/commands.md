# Task Graph Rendering

Graphics commands are recorded through [`RenderPassBuilder`](../../src/task_graph/graph.rs) nodes inside a [`TaskGraph`](../../src/task_graph/graph.rs). The graph declares resource dependencies (buffers, textures, parcels) and submits all GPU work in one dispatch.

Windowed apps render to an offscreen [`RenderTarget`](../surfaces/overview.md), blit to the swapchain, then present. See [`examples/triangle.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/triangle.rs) for the canonical loop.

## Per-Frame Loop (windowed)

```rust
use goldy::{Color, NodeAccess, RenderTarget, Surface, TaskGraph};

frame_graph.clear();

let mut pass = frame_graph.render_pass("main", &scene_rt);
pass.bind_buffer_mut(&vertex_buffer, NodeAccess::Read);
pass.clear(Color::CORNFLOWER_BLUE);
pass.set_pipeline(&pipeline);
pass.set_vertex_buffer(0, &vertex_buffer);
pass.draw(0..3, 0..1);
pass.finish_recorded();

let swapchain = frame_graph.declare_swapchain_output();
frame_graph.copy_render_target_to_swapchain(&scene_rt, swapchain);

let frame = surface.begin()?;
let frame = surface.submit_graph_to_frame(&mut frame_graph, frame)?;
frame.present()?;
```

## Render Pass Builder

`render_pass(label, target)` returns a builder that records draw commands for one offscreen target.

### Clearing

```rust
pass.clear(Color::BLACK);
pass.clear_depth(1.0);
```

### Pipeline and Buffers

```rust
pass.set_pipeline(&pipeline);
pass.set_vertex_buffer(0, &vertices);
pass.set_index_buffer(&indices, IndexFormat::Uint16);
pass.bind_resources(&[&uniforms, &textures]);
```

### Drawing

```rust
pass.draw(0..3, 0..1);              // non-indexed
pass.draw_indexed(0..6, 0..1);      // indexed
pass.draw_fullscreen();             // 3-vertex fullscreen triangle
pass.draw_quads(4);                 // instanced quads
```

### Graph Dependencies

Declare which resources the pass reads or writes so the runtime can track parcel lifetimes:

```rust
pass.bind_buffer_mut(&buf, NodeAccess::Read);
pass.bind_texture_mut(&tex, NodeAccess::Read);
pass.bind_parcel_mut(&parcel, NodeAccess::Write);
```

Call `finish_recorded()` when done recording commands for this pass node.

## Offscreen-Only (tests, readback)

For headless rendering without a window, dispatch the graph on a device context:

```rust
let ctx = device.create_context()?;
let mut graph = TaskGraph::new();
let mut pass = graph.render_pass("clear", &target);
pass.clear(Color::RED);
pass.finish_recorded();
graph.dispatch(&ctx)?;
let pixels = target.read_to_cpu()?;
```

## Hybrid Compute + Graphics

Put compute `dispatch` nodes and `render_pass` nodes in the **same** graph, then submit once:

```rust
frame_graph.write_buffer(&staging, &data);
frame_graph.dispatch("sim", &compute_pipeline, (wg, 1, 1));
// ... render_pass on scene_rt ...
frame_graph.copy_render_target_to_swapchain(&scene_rt, swapchain);
surface.submit_graph_to_frame(&mut frame_graph, frame)?;
```

## Notes

- Depth buffers live on the offscreen `RenderTarget` (`RenderTarget::new_with_depth`), not on the swapchain surface.
- Imperative graphics draw recording was removed; clients use `RenderPassBuilder::finish_recorded()`.
- Compute-only swapchain output (no raster) uses `SwapchainOutput` directly — see [Compute to Surface](../compute/compute-to-surface.md).
