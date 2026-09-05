# digital_clock

A seven-segment clock. Segment geometry is generated on the CPU in
`examples/digital_clock_shared.rs` and drawn as coloured triangles each frame.

<video src="../assets/examples/digital_clock.webm" autoplay loop muted playsinline
       width="640" style="max-width: 100%; border-radius: 4px;"></video>

```bash
cargo run --features examples --example digital_clock
```

## What it demonstrates

- Dynamic CPU-generated geometry
- Sharing helper modules between examples

## Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `C` | Cycle colour |
| `Escape` | Exit |

## Source

`examples/digital_clock.rs`:

```rust,noplayground
{{#include ../../../examples/digital_clock.rs}}
```

The example pulls in `examples/common.rs`, `examples/digital_clock_shared.rs` — see [Shared Helpers](./shared-helpers.md).

The Slang source is inline in the example above.
