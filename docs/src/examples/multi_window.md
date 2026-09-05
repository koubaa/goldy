# multi_window

Three windows — plasma, tunnel, and starfield — sharing one `Device`. Each window owns its
own `SurfaceExchange`, `Context`, and `Scheme`, which is the pattern for any multi-surface
application.

```bash
cargo run --features examples --example multi_window
```

## What it demonstrates

- Multiple surfaces on a single device
- One scheme and context per window
- Independent resize and close handling per window

## Controls

| Key | Action |
|-----|--------|
| `Space` | Toggle the focused window's effect modifier |
| `R` | Reset the focused window's effect |
| `Escape` | Close the focused window; the app exits with the last one |

## Source

`examples/multi_window.rs`:

```rust,noplayground
{{#include ../../../examples/multi_window.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/starfield.slang`:

```slang
{{#include ../../../shaders/starfield.slang}}
```
