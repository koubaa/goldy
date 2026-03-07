# Packaging and Slang Bundling

This document explains how Goldy packages its dependencies, particularly the Slang shader compiler, across all distribution channels.

## The Problem

Goldy requires a specific version of [Slang](https://github.com/shader-slang/slang) (currently **2026.4**) for SM 6.6 bindless support. Without bundling, users may encounter runtime failures because:

1. **Version mismatch**: The Vulkan SDK often includes an older Slang version that lacks required features like `DescriptorHandle`
2. **Missing libraries**: Slang consists of multiple DLLs (`slang-compiler.dll`, `slang.dll`, `slang-rt.dll`, etc.) that must all be present
3. **Search path issues**: The loader may find system-installed Slang before vendored copies

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     slang/manifest.json                                  │
│                   (Single Source of Truth)                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  version: "2026.4"                                           │   │
│  │  platforms:                                                      │   │
│  │    windows-x86_64: [slang-compiler.dll, slang.dll, ...]         │   │
│  │    linux-x86_64:   [libslang-compiler.so, libslang.so, ...]     │   │
│  │    macos-aarch64:  [libslang-compiler.dylib, ...]               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ build.rs  │   │ffi/build.rs│  │ binding   │
            │  (Rust)   │   │  (FFI)    │   │ scripts   │
            └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                  │               │               │
                  ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ crates.io │   │  target/  │   │ NuGet     │
            │  package  │   │   dir     │   │ PyPI      │
            │           │   │           │   │ vcpkg     │
            └───────────┘   └───────────┘   └───────────┘
```

## Manifest File

The [`slang/manifest.json`](slang/manifest.json) file defines:

- **version**: The required Slang version
- **download_url_template**: URL pattern for downloading releases
- **platforms**: Per-platform file lists with:
  - **files**: All library files needed at runtime
  - **primary**: The main entry point library for loader validation

When updating Slang, only this file needs to change (plus downloading new binaries).

## Platform-Specific Files

| Platform | Files |
|----------|-------|
| Windows x64 | `slang-compiler.dll`, `slang.dll`, `slang-rt.dll`, `slang-glslang.dll`, `slang-glsl-module.dll`, `slang-llvm.dll` |
| Linux x64 | `libslang-compiler.so`, `libslang.so`, `libslang-rt.so`, `libslang-glslang-*.so`, `libslang-glsl-module-*.so`, `libslang-llvm.so` |
| macOS arm64 | `libslang-compiler.dylib`, `libslang.dylib`, `libslang-rt.dylib`, `libslang-glslang.dylib`, `libslang-glsl-module.dylib`, `libslang-llvm.dylib` |

## Distribution Channels

### Rust Crate (crates.io)

**Build script**: [`build.rs`](build.rs)

The build script:
1. Reads `slang/manifest.json`
2. Checks for vendored binaries in `slang/bin/{platform}/`
3. Downloads from GitHub releases if missing
4. Copies all Slang libraries to `OUT_DIR`
5. Emits `cargo:rustc-link-search` for the loader

Users can override with `GOLDY_SLANG_PATH` environment variable.

### FFI Library (goldy-ffi)

**Build script**: [`ffi/build.rs`](ffi/build.rs)

Copies all Slang libraries alongside `goldy_ffi.dll` in the target directory. This ensures language bindings that load `goldy_ffi` can find Slang.

### .NET / NuGet

**Build scripts**: [`dotnet/build-native.ps1`](dotnet/build-native.ps1), [`dotnet/build-native.sh`](dotnet/build-native.sh)

**Package config**: [`dotnet/Goldy/Goldy.csproj`](dotnet/Goldy/Goldy.csproj)

The build scripts:
1. Build `goldy_ffi` for the current platform
2. Copy FFI library to `runtimes/{rid}/native/`
3. Read manifest and copy all Slang libraries to the same directory

The `.csproj` includes `slang*.dll` / `libslang*` in the NuGet package via the `runtimes/` content items.

### Python / PyPI

**Build script**: [`python/build-slang.py`](python/build-slang.py)

**Package config**: [`python/pyproject.toml`](python/pyproject.toml)

Before running `maturin build`:
1. Run `python build-slang.py` to copy Slang libraries to `python/goldy/`
2. The `pyproject.toml` `include` section picks up `*.dll`, `*.so`, `*.dylib`

### C++ / vcpkg

**Port file**: [`cpp/vcpkg/portfile.cmake`](cpp/vcpkg/portfile.cmake)

The portfile:
1. Downloads goldy source and pre-built binaries
2. Reads `slang/manifest.json` to get the file list
3. Installs Slang libraries alongside `goldy_ffi` in `bin/` (Windows) or `lib/` (Unix)

### C++ / Conan

**Recipe**: [`cpp/conan/conanfile.py`](cpp/conan/conanfile.py)

The `package()` method:
1. Copies `goldy_ffi` library
2. Reads manifest and copies all platform-appropriate Slang libraries
3. Uses `self.copy()` to include them in the Conan package

### C++ / CMake (local builds)

**CMake config**: [`cpp/CMakeLists.txt`](cpp/CMakeLists.txt)

The install rules:
1. Read manifest via `file(READ ... JSON)`
2. Install Slang libraries alongside the FFI library
3. The `goldyConfig.cmake` exports `GOLDY_SLANG_LIBRARIES` for downstream projects

## Updating Slang Version

To update to a new Slang version:

1. **Update manifest**:
   ```json
   {
     "version": "NEW.VERSION.HERE",
     ...
   }
   ```

2. **Download new binaries**:
   ```bash
   cd slang
   ./download.sh
   ```

3. **Verify file lists**: Check that `manifest.json` platform file lists match the downloaded files. Slang occasionally adds/removes libraries.

4. **Test all bindings**:
   ```bash
   # Rust
   cargo test
   
   # .NET
   cd dotnet && ./build-native.ps1 && dotnet test
   
   # Python
   cd python && python build-slang.py && maturin develop && pytest
   ```

5. **Update CHANGELOG.md** to note the Slang version bump

## Troubleshooting

See [DEBUGGING.md](DEBUGGING.md) for common issues related to Slang loading.

### Verifying Slang is Bundled

Check that Slang libraries are present alongside the FFI library:

```bash
# Windows
ls path/to/goldy_ffi.dll/../slang*.dll

# Linux
ls path/to/libgoldy_ffi.so/../libslang*.so

# macOS
ls path/to/libgoldy_ffi.dylib/../libslang*.dylib
```

### Force Re-download

Delete vendored binaries and rebuild:

```bash
rm -rf slang/bin/
cargo clean
cargo build
```

Or set `GOLDY_SLANG_PATH` to use a custom Slang installation.
