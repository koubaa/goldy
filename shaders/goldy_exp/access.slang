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
// - Metal:  Uses gGoldy ParameterBlock arrays (Argument Buffers Tier 2 required)

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
// IMPORTANT: Use uint4 instead of uint[4] to ensure all values are in a single
// cbuffer register. Dynamic array indexing with uint[4] causes Slang to generate
// cbufferLoadLegacy calls with regIndex=n, treating each element as a separate
// 16-byte register, which reads out-of-bounds garbage for indices > 0.
cbuffer __GoldyDynamic : register(b0, space0) { uint4 __gGoldyDynamicSlots; };
// Direct access function that uses proper component indexing
uint __goldy_get_slot(uint n) {
    // Use explicit switch to ensure compiler generates correct component access
    switch (n) {
        case 0: return __gGoldyDynamicSlots.x;
        case 1: return __gGoldyDynamicSlots.y;
        case 2: return __gGoldyDynamicSlots.z;
        case 3: return __gGoldyDynamicSlots.w;
        default: return 0;
    }
}
// Wrapper for struct-based access (for compatibility with existing code)
GoldyDynamicSlots __goldy_get_dynamic() {
    GoldyDynamicSlots result;
    result.slots[0] = __gGoldyDynamicSlots.x;
    result.slots[1] = __gGoldyDynamicSlots.y;
    result.slots[2] = __gGoldyDynamicSlots.z;
    result.slots[3] = __gGoldyDynamicSlots.w;
    return result;
}
#define gGoldyDynamic __goldy_get_dynamic()
// Macro is no-op - cbuffer is already declared
#define GOLDY_DECLARE_DYNAMIC static const bool __goldy_dynamic_decl = true

#elif defined(__METAL__)
// Metal: Push constants for runtime slot indices
[[vk::push_constant]] ConstantBuffer<GoldyDynamicSlots> gGoldyDynamic;
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

/// Dynamic scattered access - reads slot index from push constants.
/// Uses __goldy_get_slot for proper cbuffer component access.
public RWStructuredBuffer<T> goldy_dyn_scattered<T>(uint n)
{
    uint slot = __goldy_get_slot(n);
    DescriptorHandle<RWStructuredBuffer<T>> handle;
    handle = DescriptorHandle<RWStructuredBuffer<T>>(uint2(slot, 0));
    return *handle;
}

/// Dynamic broadcast access - reads slot index from push constants.
/// Uses __goldy_get_slot for proper cbuffer component access.
public T goldy_dyn_broadcast<T>(uint n)
{
    uint slot = __goldy_get_slot(n);
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
// Metal Implementation (Bindless via Argument Buffers Tier 2)
// Uses ParameterBlock with device pointers
// =============================================================================
#elif defined(__METAL__)

// -----------------------------------------------------------------------------
// Public access functions - return typed device pointer
// Using Ptr<T, Access, AddressSpace> syntax for generic pointer return
// -----------------------------------------------------------------------------

/// Dynamic scattered access - returns typed pointer for read/write access.
/// Usage: goldy_dyn_scattered<Particle>(0)[idx]
public Ptr<T, Access.ReadWrite, AddressSpace.Device> goldy_dyn_scattered<T>(uint n)
{
    uint slot = gGoldyDynamic.slots[n];
    uint* basePtr = gGoldy.storageBuffers[slot];
    return (Ptr<T, Access.ReadWrite, AddressSpace.Device>)basePtr;
}

/// Static scattered access - returns typed pointer for read/write access.
/// Usage: goldy_scattered<Particle>(0)[idx]
public Ptr<T, Access.ReadWrite, AddressSpace.Device> goldy_scattered<T>(uint slot)
{
    uint* basePtr = gGoldy.storageBuffers[slot];
    return (Ptr<T, Access.ReadWrite, AddressSpace.Device>)basePtr;
}

/// Dynamic broadcast access - reads slot index from gGoldyDynamic.slots[n].
public T goldy_dyn_broadcast<T>(uint n)
{
    uint slot = gGoldyDynamic.slots[n];
    uint* basePtr = gGoldy.uniformBuffers[slot];
    return ((Ptr<T, Access.ReadWrite, AddressSpace.Device>)basePtr)[0];
}

/// Access a uniform/constant buffer by slot index (Metal).
/// Cast the device pointer to T* and index [0] to read sizeof(T) bytes.
/// reinterpret<T>(*basePtr) is WRONG: it only reads 4 bytes (one uint) and
/// zero-fills the rest, corrupting any T larger than 4 bytes.
public T goldy_broadcast<T>(uint slot)
{
    uint* basePtr = gGoldy.uniformBuffers[slot];
    return ((Ptr<T, Access.ReadWrite, AddressSpace.Device>)basePtr)[0];
}

/// Access a 2D texture by slot index (Metal).
public Texture2D<T> goldy_interpolated<T : ITexelElement>(uint slot)
{
    return reinterpret<Texture2D<T>>(gGoldy.textures[slot]);
}

/// Access a read-write 2D texture by slot index (Metal).
public RWTexture2D<T> goldy_direct_spatial<T : ITexelElement>(uint slot)
{
    return reinterpret<RWTexture2D<T>>(gGoldy.storageImages[slot]);
}

/// Access a sampler state by slot index (Metal).
public SamplerState goldy_filter(uint slot)
{
    return gGoldy.samplers[slot];
}

#endif
