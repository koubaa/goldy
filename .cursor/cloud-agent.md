# Cursor Cloud Agent Instructions

## Environment variables required for building and testing

All `cargo build`, `cargo test`, `cargo clippy`, and `cargo run` commands that touch the main `goldy` crate need these env vars:

```bash
export GOLDY_SLANG_PATH=/workspace/slang/bin/linux-x86_64/libslang.so
export LD_LIBRARY_PATH=/workspace/slang/bin/linux-x86_64
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export VK_LAYER_PATH=""
export CXX=g++-13
```

- `GOLDY_SLANG_PATH` — full path to the Slang `.so`; used by `build.rs` and at runtime.
- `LD_LIBRARY_PATH` — directory containing Slang shared libraries (needed at runtime).
- `VK_ICD_FILENAMES` — forces lavapipe (Mesa software Vulkan) as the only ICD; required since there is no hardware GPU.
- `VK_LAYER_PATH=""` — disables Vulkan validation layers for CI.
- `CXX=g++-13` — the `nv-flip-sys` dev-dependency requires C++ compilation; default `c++` (clang) can't find libstdc++ headers without this override.

## Running CI checks

Per the Development section in [AGENTS.md](../AGENTS.md), plus `--features vulkan` for the test step on Linux:

```bash
cargo fmt --all -- --check
cargo clippy --no-default-features -- -D warnings
cargo clippy -- -D warnings
cargo test --features vulkan
```

## Headless GPU rendering

The Cloud VM has no physical GPU. **Lavapipe** (Mesa's Vulkan 1.4 software renderer) provides the Vulkan backend. All unit tests, integration tests, and screenshot tests pass headlessly.

Interactive examples (Surface API / `winit`) require Wayland and will not work under Xvfb (X11 only). To test rendering output headlessly, use:

```bash
cargo run --bin update-screenshots --features update-screenshots
```

This renders triangles, Game of Life, and depth-occlusion scenes to PNG files in `tests/screenshots/`.
