# Conditional Compilation

**Most users should use `GOLDY_BACKEND`** for runtime switching—see [Backend Selection](backends.md#backend-selection).

## When to Use Compile-Time Features

Use `--no-default-features --features <backend>` when you need:

- **Smaller binaries** - Exclude unused backend code
- **Faster builds** - Skip compiling unused backend dependencies
- **Missing SDK** - Build on a system without the Vulkan SDK or Windows SDK
- **CI matrix** - Test that each backend compiles independently

## Feature Flags

Goldy's Cargo.toml defines these features:

```toml
[features]
default = ["vulkan", "dx12", "metal"]
vulkan = ["dep:ash"]
dx12 = ["dep:windows", "dep:gpu-allocator", "dep:windows-core"]
metal = ["dep:metal", "dep:cocoa", "dep:objc", ...]
```

### Dependency Exclusion

**Building with only one backend excludes both the code AND dependencies for other backends.**

| Feature | Dependencies Added |
|---------|-------------------|
| `vulkan` | `ash` |
| `dx12` | `windows`, `gpu-allocator`, `windows-core` |
| `metal` | `metal`, `cocoa`, `objc`, `core-graphics-types`, `foreign-types` |

For example, on Windows:

```bash
# Default build - compiles both Vulkan and DX12
cargo build
# Downloads: ash, windows (~200 modules), gpu-allocator, windows-core

# Vulkan-only build
cargo build --no-default-features --features vulkan
# Downloads: ash ONLY
# Does NOT compile or link any DX12 code or dependencies
```

This can significantly reduce build times and binary size.

## FFI and Python Feature Passthrough

The `goldy-ffi` and `goldy-py` crates propagate features to the core `goldy` crate:

```bash
# Build FFI bindings with only Vulkan backend
cargo build -p goldy-ffi --no-default-features --features vulkan

# Build Python bindings with only DX12 backend
cargo build -p goldy-py --no-default-features --features dx12
```

This is useful for creating platform-specific binary distributions.

## CI Matrix Example

To verify each backend compiles independently in CI:

```yaml
# GitHub Actions example
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

## Checking Available Backends

At runtime, you can query which backend is in use:

```rust
let instance = Instance::new()?;
println!("Backend: {:?}", instance.backend_type());
```
