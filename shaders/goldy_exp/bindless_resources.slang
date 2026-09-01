// Metal bindless resource infrastructure for Goldy
// Auto-included when you import goldy_exp.
//
// This module defines the ParameterBlock structure that matches Goldy's
// Metal backend argument buffer layout. Argument Buffers Tier 2 is required.
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

// Maximum resources per access pattern category (must match MAX_RESOURCES_PER_CATEGORY in metal/types.rs)
static const uint GOLDY_MAX_RESOURCES = 4096;
#if defined(__METAL__)

// ParameterBlock with device pointers.
//
// Metal argument buffers store native device pointers when using
// ArgumentEncoder.setBuffer(). We declare these as uniform pointers which
// Slang compiles to Metal device pointers.
//
// Users access resources through goldy_broadcast<T>(), goldy_scattered<T>(), etc.

struct GoldyBindlessResources {
    // Slot 0: SCATTERED - device pointers to storage buffers
    // Both arrays use uint* for consistent 8-byte layout matching Rust backend
    uint* storageBuffers[GOLDY_MAX_RESOURCES];

    // Slot 1: BROADCAST - device pointers to uniform buffers
    // Typed as uint* for value reading (goldy_broadcast returns T, not T*)
    uint* uniformBuffers[GOLDY_MAX_RESOURCES];

    // Slot 2: INTERPOLATED - hardware-filtered texture reads
    Array<Texture2D<float4>, GOLDY_MAX_RESOURCES> textures;

    // Slot 3: DIRECT_SPATIAL - unfiltered read/write texture access
    Array<RWTexture2D<float4>, GOLDY_MAX_RESOURCES> storageImages;

    // Slot 4: FILTER_CONFIG - sampler states for texture filtering
    Array<SamplerState, GOLDY_MAX_RESOURCES> samplers;
    Array<RaytracingAccelerationStructure, GOLDY_MAX_RESOURCES> accelerationStructures;
};

// Global bindless resource block — bound at buffer slot 0 by the Metal backend
ParameterBlock<GoldyBindlessResources> gGoldy;

#endif // __METAL__

