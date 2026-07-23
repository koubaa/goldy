# Conditional Compilation

**Most users should use `GOLDY_BACKEND` for runtime switching** — see
[Backend Architecture](overview.md#runtime-override--goldy_backend).

Compile-time feature flags are useful when you need smaller binaries,
faster builds, or want to verify that each backend compiles independently
in CI.

## When to Use Compile-Time Features

Use `--no-default-features --features <backend>` when you need:

- **Smaller binaries** — exclude unused backend code
- **Faster builds** — skip compiling heavy backend dependencies
- **Missing SDK** — build on a system that lacks the Vulkan SDK or Windows SDK
- **CI matrix** — verify each backend compiles independently

## Feature Flags

Goldy defines one feature per backend plus an `instrumentation` feature:

```toml
[features]
default = ["vulkan", "metal", "dx12", "instrumentation"]
vulkan  = ["dep:ash"]
dx12    = ["dep:windows", "dep:gpu-allocator", "dep:windows-core"]
metal   = ["dep:metal", "dep:cocoa", "dep:objc", "dep:core-graphics-types",
           "dep:foreign-types", "dep:block"]

instrumentation = ["dep:tracing-subscriber"]
```

### Dependency Exclusion

Building with only one backend excludes both the **code** and the
**dependencies** for the others:

| Feature | Dependencies |
|---------|-------------|
| `vulkan` | `ash` |
| `dx12` | `windows`, `gpu-allocator`, `windows-core` |
| `metal` | `metal`, `cocoa`, `objc`, `core-graphics-types`, `foreign-types`, `block` |

```bash
# Default build on Windows — compiles Vulkan + DX12 dependencies
cargo build

# Vulkan-only build — downloads only ash
cargo build --no-default-features --features vulkan

# DX12-only build
cargo build --no-default-features --features dx12
```

This can significantly reduce build times and binary size.

## Platform-Specific Considerations

| Backend | Available On | Notes |
|---------|-------------|-------|
| `vulkan` | Windows, Linux (any platform with a Vulkan loader) | Broadest platform support |
| `dx12` | Windows only | Gated by `#[cfg(target_os = "windows")]` — the feature is ignored on other platforms |
| `metal` | macOS only | Gated by `#[cfg(target_os = "macos")]` — the feature is ignored on other platforms |

On macOS, the default backend is native Metal. Goldy does not require MoltenVK.

## Default Features

The `default` feature set enables all three backends plus instrumentation:

```toml
default = ["vulkan", "metal", "dx12", "instrumentation"]
```

To override, use `--no-default-features` and enable only what you need:

```bash
# Only Vulkan, no instrumentation
cargo build --no-default-features --features vulkan

# Vulkan + instrumentation
cargo build --no-default-features --features vulkan,instrumentation

# Metal-only on macOS
cargo build --no-default-features --features metal
```

## FFI and Python Feature Passthrough

The `goldy-ffi` and `goldy-py` crates propagate features to the core
`goldy` crate, so you can control backend selection in downstream builds:

```bash
# FFI bindings with only Vulkan backend
cargo build -p goldy-ffi --no-default-features --features vulkan

# Python bindings with only DX12 backend
cargo build -p goldy-py --no-default-features --features dx12
```

This is useful for creating platform-specific binary distributions.

## Cross-Compilation

When cross-compiling, keep in mind that platform-gated features are
silently ignored if the target platform doesn't match:

```bash
# Targeting macOS — dx12 feature is silently ignored, only metal + vulkan
# are active
cargo build --target aarch64-apple-darwin

# Targeting Windows — metal feature is silently ignored
cargo build --target x86_64-pc-windows-msvc --no-default-features --features dx12
```

For cross-compilation to work, you need the appropriate system SDKs
available. Vulkan is the most portable backend since the `ash` crate only
needs a Vulkan loader at runtime, not at compile time.

## CI Matrix Example

Verify each backend compiles independently in CI:

```yaml
# GitHub Actions
jobs:
  lint:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            features: vulkan
          - os: windows-latest
            features: vulkan
          - os: windows-latest
            features: dx12
          - os: macos-latest
            features: metal
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - run: cargo clippy --no-default-features --features ${{ matrix.features }} -- -D warnings
```

## Checking the Active Backend

At runtime, query which backend was selected:

```rust
let instance = Instance::new()?;
println!("Backend: {:?}", instance.backend_type());
```

If no backend feature is enabled for the current platform, `Instance::new()`
returns an error:

```
No GPU backend available - enable 'vulkan', 'dx12', or 'metal' feature
```
