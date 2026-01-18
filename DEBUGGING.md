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
