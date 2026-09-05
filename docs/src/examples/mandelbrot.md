# mandelbrot

An interactive Mandelbrot explorer. Pan and zoom update a uniform that the fragment shader
uses as its complex-plane window, so navigation costs nothing but a deposit.

```bash
cargo run --features examples --example mandelbrot
```

## What it demonstrates

- Interactive uniform updates from keyboard input
- Iteration-count colouring in a fragment shader

## Controls

| Key | Action |
|-----|--------|
| Arrow keys | Pan |
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `R` | Reset view |
| `Escape` | Exit |

## Source

`examples/mandelbrot.rs`:

```rust,noplayground
{{#include ../../../examples/mandelbrot.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

## Shaders

`shaders/mandelbrot.slang`:

```slang
{{#include ../../../shaders/mandelbrot.slang}}
```
