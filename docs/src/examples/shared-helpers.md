# Shared Helpers

The windowed examples share a few modules. They are not registered as `[[example]]` targets;
each example pulls them in with `mod common;` and friends.

## `examples/common.rs`

Run limits (`GOLDY_EXAMPLE_TIMEOUT` / `EXAMPLE_TIMEOUT`), the trailing FPS window used by the
`GOLDY_PERF` line, hidden-window creation so the first frame is never a blank flash,
`FrameSink` (window present or a packed-RGBA dump for book clips), and
`render_pipeline` / `render_pipeline_for_surface`, which rebuild a pipeline against the
current colour-target format.

Set `GOLDY_EXAMPLE_CAPTURE` to a raw-RGBA output path to skip the window and write pixels that
`scripts/record_example_captures.sh` stitches with ffmpeg. Optional:
`GOLDY_EXAMPLE_CAPTURE_FRAMES` (default 75), `GOLDY_EXAMPLE_CAPTURE_FPS` (default 15),
`GOLDY_EXAMPLE_CAPTURE_WIDTH` / `HEIGHT` (default 640×480).

```rust,noplayground
{{#include ../../../examples/common.rs}}
```

## `examples/digital_clock_shared.rs`

Seven-segment digit geometry for [`digital_clock`](./digital_clock.md).

```rust,noplayground
{{#include ../../../examples/digital_clock_shared.rs}}
```

## `examples/instance2d.rs`

The per-instance struct for [`instancing`](./instancing.md), laid out to match `QuadInstance`
in `instancing_update.slang` and `instancing_render.slang`.

```rust,noplayground
{{#include ../../../examples/instance2d.rs}}
```
