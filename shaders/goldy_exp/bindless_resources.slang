// Metal Tier 2 bindless resource infrastructure for Goldy
// Auto-included when you import goldy_exp.
//
// This module defines the ParameterBlock structure that matches Goldy's
// Metal backend argument buffer layout for Tier 2 (bindless) support.
//
// == Metal Tier 1 vs Tier 2 ==
//
// - Tier 1: Traditional bindings only, no argument buffer arrays
// - Tier 2: Full argument buffers with indexed resource access
//
// The __METAL_BINDLESS__ define is set when compiling for Tier 2 devices.
// Shaders can check this to conditionally use bindless features:
//
//   #if defined(__METAL__) && defined(__METAL_BINDLESS__)
//       // Use goldy_broadcast<T>(slot) - works on Tier 2
//   #elif defined(__METAL__)
//       // Use traditional bindings for Tier 1 fallback
//       ConstantBuffer<MyUniforms> uniforms : register(b0);
//   #endif
//
// == Access Pattern Organization ==
//
// Resources are organized by access pattern (same as Vulkan/DX12):
//   Slot 0: SCATTERED      - Storage buffers (RWStructuredBuffer)
//   Slot 1: BROADCAST      - Uniform buffers (ConstantBuffer)
//   Slot 2: INTERPOLATED   - Sampled textures (Texture2D)
//   Slot 3: DIRECT_SPATIAL - Storage images (RWTexture2D)
//   Slot 4: FILTER_CONFIG  - Samplers (SamplerState)

implementing goldy_exp;

// Maximum resources per access pattern category
static const uint GOLDY_MAX_RESOURCES = 64;

#if defined(__METAL__) && defined(__METAL_BINDLESS__)

// Metal Tier 2: ParameterBlock with typed resource arrays
// The arrays are typed to enable proper Metal code generation.
// Users access resources through goldy_broadcast<T>(), goldy_scattered<T>(), etc.

struct GoldyBindlessResources {
    // Slot 0: SCATTERED - read/write storage access (any thread, any address)
    // Uses uint as base element type - actual type is reinterpreted at access
    Array<RWStructuredBuffer<uint>, GOLDY_MAX_RESOURCES> storageBuffers;
    
    // Slot 1: BROADCAST - uniform/constant data (all threads same address)
    // Uses uint as base element type for flexibility
    Array<ConstantBuffer<uint>, GOLDY_MAX_RESOURCES> uniformBuffers;
    
    // Slot 2: INTERPOLATED - hardware-filtered texture reads
    Array<Texture2D<float4>, GOLDY_MAX_RESOURCES> textures;
    
    // Slot 3: DIRECT_SPATIAL - unfiltered read/write texture access
    Array<RWTexture2D<float4>, GOLDY_MAX_RESOURCES> storageImages;
    
    // Slot 4: FILTER_CONFIG - sampler states for texture filtering
    Array<SamplerState, GOLDY_MAX_RESOURCES> samplers;
};

// Global bindless resource block for Metal Tier 2
// Goldy's Metal backend binds this at buffer slot 0
ParameterBlock<GoldyBindlessResources> gGoldy;

#endif // __METAL__ && __METAL_BINDLESS__
