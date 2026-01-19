# Debugging Tips

## Metal Backend Issues

### Shader Compiles but Uniforms Don't Update (Static Animation)

If using `set_push_constants()` with a Metal shader that uses `ParameterBlock`:

1. **Check pipeline has ParameterBlock layouts**: The Metal backend needs reflection data to populate the argument buffer. Enable debug logging:
   ```bash
   RUST_LOG=goldy::backend::metal=trace cargo run --example myexample
   ```
   Look for: `"Allocated bindless argument buffer"` and `"SetPushConstants: Wrote GPU address"`.

2. **Verify argument buffer binding**: Check logs for `"SetPushConstants: Bound ParameterBlock argument buffer at slot X"`. If missing, the buffer isn't being bound to the shader.

3. **Ensure buffer is heap-allocated**: Bindless buffers must be allocated from the Metal heap. Check for `"Encoded buffer N at arg buffer offset"` during buffer creation.

## DX12 Backend Issues

### Bindless Buffers Show Wrong Data (All Zeros, Garbage)

When using `set_push_constants()` with storage buffers on DX12, the shader may read incorrect data. This is often caused by SRV/UAV descriptor mismatch:

**Background**: DX12 requires different descriptor types for read vs write access:
- `StructuredBuffer<T>` (read-only) → needs **SRV** (Shader Resource View)
- `RWStructuredBuffer<T>` (read-write) → needs **UAV** (Unordered Access View)

Goldy creates both SRV and UAV descriptors for storage buffers, stored as:
- `bindless_offset` → UAV index
- `bindless_srv_offset` → SRV index

**Current behavior for `set_push_constants()`**:
- **Render shaders**: Always use SRV offsets (render shaders only read)
- **Compute shaders**: First buffer uses SRV (read input), subsequent buffers use UAV (write outputs)

This matches the common ping-pong pattern (e.g., Game of Life: read from buffer A, write to buffer B).

### Python/FFI Buffers with Wrong Element Stride

If a `StructuredBuffer<uint>` reads garbage on DX12 but works on Vulkan, check the buffer's element stride. DX12's structured buffer views require the correct `StructureByteStride`:

- `uint` / `int` / `float` → stride = 4
- `uint2` / `float2` → stride = 8
- Raw bytes → stride = 1 (but incompatible with `StructuredBuffer`)

Python buffers automatically detect stride from numpy dtype. If using raw bytes, ensure you're not accessing them as a typed `StructuredBuffer` in the shader.

## Vulkan Backend Issues

### Shader Not Working (Static Output, No Animation)

When a shader works on DX12 but not Vulkan, check these common causes:

1. **SPIR-V Inspection**: Dump the generated SPIR-V and use `spirv-dis` to inspect:
   ```rust
   let spirv_bytes: &[u8] = bytemuck::cast_slice(spirv);
   std::fs::write("debug.spv", spirv_bytes).ok();
   ```
   Then: `spirv-dis debug.spv`

2. **Push Constants Not Generated**: Verify the SPIR-V contains `OpVariable ... PushConstant` for push constant data. If it shows `Uniform` or a different storage class, the shader isn't correctly declaring push constants.

3. **Descriptor Set/Binding Mismatch**: Check `OpDecorate` lines in SPIR-V for `Binding` and `DescriptorSet`. These must match what the backend expects:
   - Binding 0: Storage buffers
   - Binding 1: Uniform buffers  
   - Binding 2: Sampled images
   - Binding 3: Samplers

### Slang Preprocessor Issues

See [shaders/README.md](shaders/README.md#preprocessor-defines) for Slang-specific preprocessor behavior that can cause cross-platform issues.

## Runtime Logging

For debugging resource binding issues, add temporary logging:

```rust
tracing::debug!(
    "Buffer {} at bindless index {} (storage={})",
    handle, index, is_storage
);
```

Enable with `RUST_LOG=goldy=debug cargo run`.

## FFI / Language Bindings Issues

### "undefined identifier 'DescriptorHandle'" in .NET/C# or other bindings

This error occurs when the FFI loads the wrong Slang DLL. Goldy requires Slang 2025.24.3+ with SM 6.6 bindless support (`DescriptorHandle` intrinsic).

**Cause**: The Slang library search falls back to an older `slang.dll` from the Vulkan SDK instead of the bundled `slang-compiler.dll`.

**Automatic Bundling**: As of version 0.1.0, Goldy's FFI bindings automatically bundle Slang libraries:
- **.NET**: `build-native.ps1` / `build-native.sh` copies Slang to `runtimes/{rid}/native/`
- **Python**: `build-slang.py` copies Slang to the package before `maturin build`
- **C++**: CMake installs Slang alongside `goldy_ffi`

If you still encounter this error, the Slang libraries may not have been bundled correctly. Rebuild using the appropriate build script.

**Manual Override**: To use a custom Slang version, set the environment variable:
```bash
export GOLDY_SLANG_PATH=/path/to/slang-compiler.dll
```

**Slang Loader Search Order**:
1. `GOLDY_SLANG_PATH` environment variable
2. Same directory as executable (for FFI deployments)
3. `slang/bin/{platform}/` relative to executable
4. Downloaded by build.rs to `OUT_DIR` (Rust crate only)

The required Slang version and file list are defined in `slang/manifest.json`.