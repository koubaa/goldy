// Cross-platform DescriptorHandle<T> support for Goldy
// Auto-included when you import goldy_exp.
//
// Slang's DescriptorHandle<T> provides cross-platform bindless resource access:
//   - DXIL: ResourceDescriptorHeap[index]
//   - Metal: Argument buffer pointer
//   - SPIRV: Custom routing (use the heap declarations below)
//
// Usage:
//   uniform DescriptorHandle<ConstantBuffer<MyData>> data;
//   float x = (*data).value;

implementing goldy_exp;

// =============================================================================
// SPIRV-specific: Vulkan descriptor heap declarations
//
// Organized by ACCESS PATTERN (what the hardware actually optimizes for):
//
//   0: SCATTERED      - Any thread, any address, read/write. No coherence.
//   1: BROADCAST      - All threads same address. Hardware broadcast cache.
//   2: INTERPOLATED   - Hardware filtering between neighbors (texture units).
//   3: DIRECT_SPATIAL - 2D/3D indexing without filtering. Read/write.
//   4: FILTER_CONFIG  - Not data. Settings for interpolated access.
//
// Must match Goldy's Vulkan backend (see vulkan/types.rs::bindless_bindings)
//
// Note: These are provided for advanced users who need direct heap access.
// Most shaders should use the preprocessor-based approach in other shaders
// (see checkerboard.slang for example).
// =============================================================================
