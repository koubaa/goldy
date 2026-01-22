// Unified access pattern functions for Goldy
// Auto-included when you import goldy_exp.
//
// These functions provide cross-platform resource access organized by
// the underlying hardware access pattern:
//
// == Static Access (compile-time slot indices) ==
//
//   goldy_broadcast<T>(slot)     - Uniform/constant data (all threads same address)
//   goldy_scattered<T>(slot)     - Storage buffers (any thread, any address)
//   goldy_interpolated<T>(slot)  - Hardware-filtered textures
//   goldy_filter(slot)           - Sampler states for texture filtering
//
// == Dynamic Access (runtime slot indices via push constants) ==
//
//   goldy_dyn_scattered<T>(n)    - Storage buffer from push constant slot n
//   goldy_dyn_broadcast<T>(n)    - Uniform buffer from push constant slot n
//
// Push constants are automatically declared when you import goldy_exp.
// No additional declarations (like GOLDY_DECLARE_DYNAMIC) are needed in user shaders.
//
// == Usage: Static Slots ==
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
// == Usage: Dynamic Slots (ping-pong, etc.) ==
//
//   import goldy_exp;
//   
//   [shader("compute")]
//   void cs_main(uint3 id : SV_DispatchThreadID) {
//       // Rust: set_push_constants(&[read_buf_slot, write_buf_slot])
//       RWStructuredBuffer<uint> current = goldy_dyn_scattered<uint>(0);
//       RWStructuredBuffer<uint> next = goldy_dyn_scattered<uint>(1);
//       next[id.x] = current[id.x] + 1;
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
// Dynamic Slot Infrastructure
// Standard push constant layout for runtime buffer selection (ping-pong, etc.)
// =============================================================================

/// Standard push constant structure for dynamic slot indices.
/// Provides 4 slots for buffer indices that can be swapped at runtime.
/// Automatically declared at module scope when you import goldy_exp.
public struct GoldyDynamicSlots {
    uint slots[4];  // Up to 4 dynamic buffer indices
};

// Module-level gGoldyDynamic for function compilation
#if defined(__SPIRV__)
// SPIRV: Push constants for runtime slot indices
[[vk::push_constant]] ConstantBuffer<GoldyDynamicSlots> gGoldyDynamic;
// Macro is no-op - gGoldyDynamic is already declared
#define GOLDY_DECLARE_DYNAMIC static const bool __goldy_dynamic_decl = true

#elif defined(__DX12__)
// DX12: Root constants in b0 space0 - functions read from this
cbuffer __GoldyDynamic : register(b0, space0) { uint __gGoldyDynamicSlots[4]; };
// Wrapper for function access
GoldyDynamicSlots __goldy_get_dynamic() {
    GoldyDynamicSlots result;
    result.slots[0] = __gGoldyDynamicSlots[0];
    result.slots[1] = __gGoldyDynamicSlots[1];
    result.slots[2] = __gGoldyDynamicSlots[2];
    result.slots[3] = __gGoldyDynamicSlots[3];
    return result;
}
#define gGoldyDynamic __goldy_get_dynamic()
// Macro is no-op - cbuffer is already declared
#define GOLDY_DECLARE_DYNAMIC static const bool __goldy_dynamic_decl = true

#elif defined(__METAL_BINDLESS__)
// Metal Tier 2: Push constants
[[vk::push_constant]] ConstantBuffer<GoldyDynamicSlots> gGoldyDynamic;
// Macro is no-op - gGoldyDynamic is already declared
#define GOLDY_DECLARE_DYNAMIC static const bool __goldy_dynamic_decl = true

#else
// Metal Tier 1: Static dummy (dynamic access not really supported)
// Zero-initialized by default
static GoldyDynamicSlots gGoldyDynamic;
// Macro is no-op - gGoldyDynamic is already declared
#define GOLDY_DECLARE_DYNAMIC static const bool __goldy_dynamic_decl = true
#endif

// =============================================================================
// SPIRV Implementation
// Uses Goldy's custom binding layout via getDescriptorFromHandle override
// =============================================================================
#ifdef __SPIRV__

/// Dynamic scattered access - reads slot index from gGoldyDynamic.slots[n].
/// Requires GOLDY_DECLARE_DYNAMIC in shader.
public RWStructuredBuffer<T> goldy_dyn_scattered<T>(uint n)
{
    return g_GoldyStorageHeap[gGoldyDynamic.slots[n]].asOpaqueDescriptor<RWStructuredBuffer<T>>();
}

/// Dynamic broadcast access - reads slot index from gGoldyDynamic.slots[n].
/// Requires GOLDY_DECLARE_DYNAMIC in shader.
public T goldy_dyn_broadcast<T>(uint n)
{
    return g_GoldyUniformHeap[gGoldyDynamic.slots[n]].asOpaqueDescriptor<ConstantBuffer<T>>();
}

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

/// Dynamic scattered access - reads slot index from gGoldyDynamic.slots[n].
/// Requires GOLDY_DECLARE_DYNAMIC in shader.
public RWStructuredBuffer<T> goldy_dyn_scattered<T>(uint n)
{
    uint slot = gGoldyDynamic.slots[n];
    DescriptorHandle<RWStructuredBuffer<T>> handle;
    handle = DescriptorHandle<RWStructuredBuffer<T>>(uint2(slot, 0));
    return *handle;
}

/// Dynamic broadcast access - reads slot index from gGoldyDynamic.slots[n].
/// Requires GOLDY_DECLARE_DYNAMIC in shader.
public T goldy_dyn_broadcast<T>(uint n)
{
    uint slot = gGoldyDynamic.slots[n];
    DescriptorHandle<ConstantBuffer<T>> handle;
    handle = DescriptorHandle<ConstantBuffer<T>>(uint2(slot, 0));
    return *handle;
}

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

/// Dynamic scattered read - reads element at index from dynamic slot n.
/// Requires GOLDY_DECLARE_DYNAMIC in shader.
public T goldy_dyn_scattered_get<T>(uint n, uint index)
{
    uint slot = gGoldyDynamic.slots[n];
    uint* basePtr = gGoldy.storageBuffers[slot];
    // Cast to T* and read element at index
    T* typedPtr = (T*)basePtr;
    return typedPtr[index];
}

/// Dynamic scattered write - writes element at index to dynamic slot n.
/// Requires GOLDY_DECLARE_DYNAMIC in shader.
public void goldy_dyn_scattered_set<T>(uint n, uint index, T value)
{
    uint slot = gGoldyDynamic.slots[n];
    uint* basePtr = gGoldy.storageBuffers[slot];
    // Cast to T* and write element at index
    T* typedPtr = (T*)basePtr;
    typedPtr[index] = value;
}

/// Legacy: Dynamic scattered access - Metal Tier 2 doesn't support returning buffer types.
/// Use goldy_dyn_scattered_get/set instead.
/// This version only works for reading element 0.
public T goldy_dyn_scattered<T>(uint n)
{
    return goldy_dyn_scattered_get<T>(n, 0);
}

/// Dynamic broadcast access - reads slot index from gGoldyDynamic.slots[n].
/// Requires GOLDY_DECLARE_DYNAMIC in shader.
public T goldy_dyn_broadcast<T>(uint n)
{
    uint slot = gGoldyDynamic.slots[n];
    uint* basePtr = gGoldy.uniformBuffers[slot];
    __target_switch
    {
    case metal:
        return reinterpret<T>(*basePtr);
    default:
        T* ptr = (T*)basePtr;
        return *ptr;
    }
}

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

/// Scattered read - reads element at index from slot.
public T goldy_scattered_get<T>(uint slot, uint index)
{
    uint* basePtr = gGoldy.storageBuffers[slot];
    // Cast to T* and read element at index
    T* typedPtr = (T*)basePtr;
    return typedPtr[index];
}

/// Scattered write - writes element at index to slot.
public void goldy_scattered_set<T>(uint slot, uint index, T value)
{
    uint* basePtr = gGoldy.storageBuffers[slot];
    // Cast to T* and write element at index
    T* typedPtr = (T*)basePtr;
    typedPtr[index] = value;
}

/// Legacy: Get buffer for scattered access - Metal Tier 2 doesn't support returning buffer types.
/// Use goldy_scattered_get/set instead.
/// This version only works for reading element 0.
public T goldy_scattered<T>(uint slot)
{
    return goldy_scattered_get<T>(slot, 0);
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
