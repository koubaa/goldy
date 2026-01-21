// Unified access pattern functions for Goldy
// Auto-included when you import goldy_exp.
//
// These functions provide cross-platform resource access organized by
// the underlying hardware access pattern:
//
//   goldy_broadcast<T>(slot)     - Uniform/constant data (all threads same address)
//   goldy_scattered<T>(slot)     - Storage buffers (any thread, any address)
//   goldy_interpolated<T>(slot)  - Hardware-filtered textures
//   goldy_filter(slot)           - Sampler states for texture filtering
//
// == Usage ==
//
//   import goldy_exp;
//   
//   struct MyUniforms { float time; };
//   
//   [shader("fragment")]
//   float4 fs_main(FullscreenVarying input) : SV_Target {
//       float t = goldy_broadcast<MyUniforms>(0).time;
//       return float4(t, 0, 0, 1);
//   }
//
// == Platform Support ==
//
// - SPIRV:  Routes to Goldy's custom binding layout (bindings 0-4)
// - DX12:   Uses DescriptorHandle which maps to ResourceDescriptorHeap
// - Metal Tier 2: Uses gGoldy ParameterBlock arrays
// - Metal Tier 1: Not supported - use traditional bindings with fallback
//
// == Metal Tier 1 Fallback ==
//
// For Metal Tier 1 (GitHub CI), use preprocessor fallback:
//
//   #if defined(__METAL__) && !defined(__METAL_BINDLESS__)
//   ConstantBuffer<MyUniforms> uniforms : register(b0);
//   #define UNIFORMS uniforms
//   #else
//   #define UNIFORMS goldy_broadcast<MyUniforms>(0)
//   #endif

implementing goldy_exp;

// =============================================================================
// SPIRV Implementation
// Uses Goldy's custom binding layout via getDescriptorFromHandle override
// =============================================================================
#ifdef __SPIRV__

/// Access a uniform/constant buffer by slot index (SPIRV).
public T goldy_broadcast<T>(uint slot)
{
    return g_GoldyUniformHeap[slot].asOpaqueDescriptor<ConstantBuffer<T>>();
}

/// Get a RWStructuredBuffer handle for scattered access (SPIRV).
public RWStructuredBuffer<T> goldy_scattered<T>(uint slot)
{
    return g_GoldyStorageHeap[slot].asOpaqueDescriptor<RWStructuredBuffer<T>>();
}

/// Access a 2D texture by slot index (SPIRV).
public Texture2D<T> goldy_interpolated<T : ITexelElement>(uint slot)
{
    return g_GoldyTextureHeap[slot].asOpaqueDescriptor<Texture2D<T>>();
}

/// Access a read-write 2D texture by slot index (SPIRV).
public RWTexture2D<T> goldy_direct_spatial<T : ITexelElement>(uint slot)
{
    return g_GoldyStorageImageHeap[slot].asOpaqueDescriptor<RWTexture2D<T>>();
}

/// Access a sampler state by slot index (SPIRV).
public SamplerState goldy_filter(uint slot)
{
    return g_GoldySamplerHeap[slot].asOpaqueDescriptor<SamplerState>();
}

// =============================================================================
// DX12 Implementation
// Uses DescriptorHandle<T> which Slang maps to ResourceDescriptorHeap
// =============================================================================
#elif defined(__DX12__)

/// Access a uniform/constant buffer by slot index (DX12).
public T goldy_broadcast<T>(uint slot)
{
    // Create a DescriptorHandle from the slot index
    DescriptorHandle<ConstantBuffer<T>> handle;
    handle = DescriptorHandle<ConstantBuffer<T>>(uint2(slot, 0));
    return *handle;
}

/// Get a RWStructuredBuffer handle for scattered access (DX12).
public RWStructuredBuffer<T> goldy_scattered<T>(uint slot)
{
    DescriptorHandle<RWStructuredBuffer<T>> handle;
    handle = DescriptorHandle<RWStructuredBuffer<T>>(uint2(slot, 0));
    return *handle;
}

/// Access a 2D texture by slot index (DX12).
public Texture2D<T> goldy_interpolated<T : ITexelElement>(uint slot)
{
    DescriptorHandle<Texture2D<T>> handle;
    handle = DescriptorHandle<Texture2D<T>>(uint2(slot, 0));
    return *handle;
}

/// Access a read-write 2D texture by slot index (DX12).
public RWTexture2D<T> goldy_direct_spatial<T : ITexelElement>(uint slot)
{
    DescriptorHandle<RWTexture2D<T>> handle;
    handle = DescriptorHandle<RWTexture2D<T>>(uint2(slot, 0));
    return *handle;
}

/// Access a sampler state by slot index (DX12).
public SamplerState goldy_filter(uint slot)
{
    DescriptorHandle<SamplerState> handle;
    handle = DescriptorHandle<SamplerState>(uint2(slot, 0));
    return *handle;
}

// =============================================================================
// Metal Tier 2 Implementation (Bindless)
// Uses ParameterBlock with device pointers
// =============================================================================
#elif defined(__METAL_BINDLESS__)

/// Access a uniform/constant buffer by slot index (Metal Tier 2).
/// Device pointers in gGoldy.uniformBuffers point directly to buffer data.
/// We use target_switch to handle Metal's pointer cast limitations.
public T goldy_broadcast<T>(uint slot)
{
    // Get the uint* device pointer from the argument buffer
    uint* basePtr = gGoldy.uniformBuffers[slot];
    
    // For Metal, we need to use the pointer directly without type casting
    // Read the data through the uint pointer and use reinterpret for the result
    __target_switch
    {
    case metal:
        // Metal: Read first sizeof(T)/4 uints and reinterpret as T
        // This works because both point to the same memory
        return reinterpret<T>(*basePtr);
    default:
        // Other targets: Direct pointer cast should work
        T* ptr = (T*)basePtr;
        return *ptr;
    }
}

/// Get a RWStructuredBuffer-like handle for scattered access (Metal Tier 2).
/// Note: Returns the first element - use goldy_scattered_at<T>(slot, idx) for indexed access.
public RWStructuredBuffer<T> goldy_scattered<T>(uint slot)
{
    // For Metal, we reinterpret the uint* as a buffer of the target type.
    // This should work for structured access patterns.
    uint* basePtr = gGoldy.storageBuffers[slot];
    return reinterpret<RWStructuredBuffer<T>>(basePtr);
}

/// Access a 2D texture by slot index (Metal Tier 2).
public Texture2D<T> goldy_interpolated<T : ITexelElement>(uint slot)
{
    // Textures don't need pointer conversion - use reinterpret for element type
    return reinterpret<Texture2D<T>>(gGoldy.textures[slot]);
}

/// Access a read-write 2D texture by slot index (Metal Tier 2).
public RWTexture2D<T> goldy_direct_spatial<T : ITexelElement>(uint slot)
{
    return reinterpret<RWTexture2D<T>>(gGoldy.storageImages[slot]);
}

/// Access a sampler state by slot index (Metal Tier 2).
public SamplerState goldy_filter(uint slot)
{
    return gGoldy.samplers[slot];
}

// =============================================================================
// Metal Tier 1 / Fallback
// =============================================================================
#else

// Metal Tier 1 does not support indexed bindless access.
// The access functions are intentionally NOT defined for Tier 1.
//
// Use traditional bindings with preprocessor fallback instead:
//
//   #if defined(__METAL__) && !defined(__METAL_BINDLESS__)
//   ConstantBuffer<MyUniforms> uniforms : register(b0);
//   #define MY_UNIFORM uniforms
//   #else
//   #define MY_UNIFORM goldy_broadcast<MyUniforms>(0)
//   #endif
//
// This ensures a clear compile error if someone tries to use goldy_broadcast()
// on Metal Tier 1, rather than silent failures.

#endif
