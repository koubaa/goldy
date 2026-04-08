# Screenshot Reference Images

This directory contains reference PNG images for FLIP-based screenshot tests (`tests/screenshot_tests.rs`).

## Generating or updating all references

Run the dedicated tool (this is **not** part of `cargo test`):

```bash
cargo run --bin update-screenshots --features update-screenshots
```

From the `goldy` crate root. That overwrites the PNGs in this directory; commit only after visually verifying output.

## Requirements

- Format: RGBA PNG, 8-bit depth
- Dimensions: Must match the test's expected width/height
- Naming: Matches the test (e.g. `solid_red.png`, `rgb_triangle.png`)

## Updating reference images after a deliberate rendering change

1. Run `update-screenshots` as above (or adjust a single PNG by hand).
2. Run `cargo test --test screenshot_tests` and confirm all comparisons pass.

## Debugging Failed Tests

When a test fails, two debug images are generated:

- `{name}-actual.png` - What was actually rendered
- `{name}-diff.png` - FLIP difference map (magma colormap: blue=similar, red=different)
