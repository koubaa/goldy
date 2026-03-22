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
//       StorageBuffer<uint> current = goldy_dyn_scattered<uint>(0);
//       StorageBuffer<uint> next = goldy_dyn_scattered<uint>(1);
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
// Cross-platform buffer type aliases
//
// On SPIRV/DX12, RWStructuredBuffer<T> is the native type for storage buffers.
// On Metal, Slang cannot construct RWStructuredBuffer<T> from device pointers;
// the equivalent is Ptr<T, Access.ReadWrite, AddressSpace.Device> which supports
// the same subscript operator. StorageBuffer<T> normalizes this across platforms.
//
// Usage:
//   StorageBuffer<Particle> buf = goldy_dyn_scattered<Particle>(0);
//   buf[idx] = ...;
//
// Alternatively, use `var` for type inference:
//   var buf = goldy_dyn_scattered<Particle>(0);
// =============================================================================

#if defined(__METAL__)
public typealias StorageBuffer<T> = Ptr<T, Access.ReadWrite, AddressSpace.Device>;
// ByteAddressView is defined as a struct later in the Metal section.
#else
public typealias StorageBuffer<T> = RWStructuredBuffer<T>;
public typealias ByteAddressView = RWByteAddressBuffer;
#endif

// =============================================================================
// Dynamic Slot Infrastructure
// Standard push constant layout for runtime buffer selection (ping-pong, etc.)
// =============================================================================

/// Standard push constant structure for dynamic slot indices.
/// Provides 16 slots for buffer indices that can be swapped at runtime.
/// Automatically declared at module scope when you import goldy_exp.
public struct GoldyDynamicSlots {
    uint slots[16];  // Up to 16 dynamic buffer indices
};

// Module-level gGoldyDynamic for function compilation
#if defined(__SPIRV__)
// SPIRV: Push constants for runtime slot indices
[[vk::push_constant]] ConstantBuffer<GoldyDynamicSlots> gGoldyDynamic;
// Macro is no-op - gGoldyDynamic is already declared
#define GOLDY_DECLARE_DYNAMIC static const bool __goldy_dynamic_decl = true

#elif defined(__DX12__)
// DX12: Root constants in b0 space0 - functions read from this
// IMPORTANT: Use uint4 instead of uint[16] to ensure correct component access.
// Dynamic array indexing causes Slang to generate cbufferLoadLegacy with wrong
// register indexing. Four uint4s = 16 slots.
cbuffer __GoldyDynamic : register(b0, space0) {
    uint4 __gGoldyDynamicSlots0;  // slots 0-3
    uint4 __gGoldyDynamicSlots1;  // slots 4-7
    uint4 __gGoldyDynamicSlots2;  // slots 8-11
    uint4 __gGoldyDynamicSlots3;  // slots 12-15
};
// Direct access function that uses proper component indexing
uint __goldy_get_slot(uint n) {
    switch (n) {
        case 0: return __gGoldyDynamicSlots0.x;
        case 1: return __gGoldyDynamicSlots0.y;
        case 2: return __gGoldyDynamicSlots0.z;
        case 3: return __gGoldyDynamicSlots0.w;
        case 4: return __gGoldyDynamicSlots1.x;
        case 5: return __gGoldyDynamicSlots1.y;
        case 6: return __gGoldyDynamicSlots1.z;
        case 7: return __gGoldyDynamicSlots1.w;
        case 8: return __gGoldyDynamicSlots2.x;
        case 9: return __gGoldyDynamicSlots2.y;
        case 10: return __gGoldyDynamicSlots2.z;
        case 11: return __gGoldyDynamicSlots2.w;
        case 12: return __gGoldyDynamicSlots3.x;
        case 13: return __gGoldyDynamicSlots3.y;
        case 14: return __gGoldyDynamicSlots3.z;
        case 15: return __gGoldyDynamicSlots3.w;
        default: return 0;
    }
}
// Wrapper for struct-based access (for compatibility with existing code)
GoldyDynamicSlots __goldy_get_dynamic() {
    GoldyDynamicSlots result;
    result.slots[0] = __gGoldyDynamicSlots0.x;
    result.slots[1] = __gGoldyDynamicSlots0.y;
    result.slots[2] = __gGoldyDynamicSlots0.z;
    result.slots[3] = __gGoldyDynamicSlots0.w;
    result.slots[4] = __gGoldyDynamicSlots1.x;
    result.slots[5] = __gGoldyDynamicSlots1.y;
    result.slots[6] = __gGoldyDynamicSlots1.z;
    result.slots[7] = __gGoldyDynamicSlots1.w;
    result.slots[8] = __gGoldyDynamicSlots2.x;
    result.slots[9] = __gGoldyDynamicSlots2.y;
    result.slots[10] = __gGoldyDynamicSlots2.z;
    result.slots[11] = __gGoldyDynamicSlots2.w;
    result.slots[12] = __gGoldyDynamicSlots3.x;
    result.slots[13] = __gGoldyDynamicSlots3.y;
    result.slots[14] = __gGoldyDynamicSlots3.z;
    result.slots[15] = __gGoldyDynamicSlots3.w;
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
public StorageBuffer<T> goldy_dyn_scattered<T>(uint n)
{
    return g_GoldyStorageHeap[gGoldyDynamic.slots[n]].asOpaqueDescriptor<RWStructuredBuffer<T>>();
}

/// Dynamic read-only scattered access - reads slot index from gGoldyDynamic.slots[n].
/// On SPIRV, SSBOs are inherently read-write; returns RWStructuredBuffer for API
/// consistency with DX12 (which also returns RWStructuredBuffer to avoid SRV barriers).
public StorageBuffer<T> goldy_dyn_buf_ro<T>(uint n)
{
    return g_GoldyStorageHeap[gGoldyDynamic.slots[n]].asOpaqueDescriptor<RWStructuredBuffer<T>>();
}

/// Dynamic broadcast access - reads slot index from gGoldyDynamic.slots[n].
/// Requires GOLDY_DECLARE_DYNAMIC in shader.
public T goldy_dyn_broadcast<T>(uint n)
{
    return g_GoldyUniformHeap[gGoldyDynamic.slots[n]].asOpaqueDescriptor<ConstantBuffer<T>>();
}

/// Dynamic byte-address buffer access for InterlockedMin/Max (e.g. path_bboxes).
/// Same buffer as goldy_dyn_scattered but with byte-address view.
public ByteAddressView goldy_dyn_byte_address(uint n)
{
    return g_GoldyStorageHeap[gGoldyDynamic.slots[n]].asOpaqueDescriptor<RWByteAddressBuffer>();
}

/// Access a uniform/constant buffer by slot index (SPIRV).
public T goldy_broadcast<T>(uint slot)
{
    return g_GoldyUniformHeap[slot].asOpaqueDescriptor<ConstantBuffer<T>>();
}

/// Get a storage buffer handle for scattered access (SPIRV).
public StorageBuffer<T> goldy_scattered<T>(uint slot)
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

/// Dynamic direct spatial access - reads slot index from gGoldyDynamic.slots[n].
public RWTexture2D<T> goldy_dyn_direct_spatial<T : ITexelElement>(uint n)
{
    return g_GoldyStorageImageHeap[gGoldyDynamic.slots[n]].asOpaqueDescriptor<RWTexture2D<T>>();
}

/// Dynamic interpolated access - reads slot index from gGoldyDynamic.slots[n].
public Texture2D<T> goldy_dyn_interpolated<T : ITexelElement>(uint n)
{
    return g_GoldyTextureHeap[gGoldyDynamic.slots[n]].asOpaqueDescriptor<Texture2D<T>>();
}

/// Dynamic sampler access - reads slot index from gGoldyDynamic.slots[n].
public SamplerState goldy_dyn_filter(uint n)
{
    return g_GoldySamplerHeap[gGoldyDynamic.slots[n]].asOpaqueDescriptor<SamplerState>();
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
public StorageBuffer<T> goldy_dyn_scattered<T>(uint n)
{
    uint slot = __goldy_get_slot(n);
    DescriptorHandle<RWStructuredBuffer<T>> handle;
    handle = DescriptorHandle<RWStructuredBuffer<T>>(uint2(slot, 0));
    return *handle;
}

/// Dynamic read-only scattered access - reads slot index from push constants.
/// On DX12, uses RWStructuredBuffer<T> (UAV) to avoid resource state transitions.
/// Buffers start in D3D12_RESOURCE_STATE_UNORDERED_ACCESS; SRV would need a barrier.
/// Declare the result as StorageBuffer<T> and only read from it.
/// TODO: revisit with enhanced buffers
public StorageBuffer<T> goldy_dyn_buf_ro<T>(uint n)
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

/// Get a storage buffer handle for scattered access (DX12).
public StorageBuffer<T> goldy_scattered<T>(uint slot)
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

/// Dynamic direct spatial access - reads slot index from push constants.
public RWTexture2D<T> goldy_dyn_direct_spatial<T : ITexelElement>(uint n)
{
    uint slot = __goldy_get_slot(n);
    DescriptorHandle<RWTexture2D<T>> handle;
    handle = DescriptorHandle<RWTexture2D<T>>(uint2(slot, 0));
    return *handle;
}

/// Dynamic interpolated access - reads slot index from push constants.
public Texture2D<T> goldy_dyn_interpolated<T : ITexelElement>(uint n)
{
    uint slot = __goldy_get_slot(n);
    DescriptorHandle<Texture2D<T>> handle;
    handle = DescriptorHandle<Texture2D<T>>(uint2(slot, 0));
    return *handle;
}

/// Dynamic sampler access - reads slot index from push constants.
public SamplerState goldy_dyn_filter(uint n)
{
    uint slot = __goldy_get_slot(n);
    DescriptorHandle<SamplerState> handle;
    handle = DescriptorHandle<SamplerState>(uint2(slot, 0));
    return *handle;
}

/// Dynamic byte-address buffer access for InterlockedMin/Max (e.g. path_bboxes).
public ByteAddressView goldy_dyn_byte_address(uint n)
{
    uint slot = __goldy_get_slot(n);
    DescriptorHandle<RWByteAddressBuffer> handle;
    handle = DescriptorHandle<RWByteAddressBuffer>(uint2(slot, 0));
    return *handle;
}

// =============================================================================
// Metal Implementation (Bindless via Argument Buffers Tier 2)
// Uses ParameterBlock with device pointers
// =============================================================================
#elif defined(__METAL__)

// -----------------------------------------------------------------------------
// Public access functions
// Return RWStructuredBuffer<T> on all platforms for a unified API.
// On Metal, reinterpret device pointers as RWStructuredBuffer<T> so that
// callers can assign the result to an RWStructuredBuffer<T> local variable
// (the pattern used by ekrano shaders) without a type mismatch.
// -----------------------------------------------------------------------------

/// Dynamic scattered access - read/write storage buffer from push constant slot n.
public StorageBuffer<T> goldy_dyn_scattered<T>(uint n)
{
    uint slot = gGoldyDynamic.slots[n];
    uint* basePtr = gGoldy.storageBuffers[slot];
    return (Ptr<T, Access.ReadWrite, AddressSpace.Device>)basePtr;
}

/// Dynamic read-only scattered access from push constant slot n.
/// Returns StorageBuffer<T> for API consistency (read-only intent is caller's contract).
public StorageBuffer<T> goldy_dyn_buf_ro<T>(uint n)
{
    uint slot = gGoldyDynamic.slots[n];
    uint* basePtr = gGoldy.storageBuffers[slot];
    return (Ptr<T, Access.ReadWrite, AddressSpace.Device>)basePtr;
}

/// Static scattered access - read/write storage buffer by compile-time slot index.
public StorageBuffer<T> goldy_scattered<T>(uint slot)
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

/// Dynamic direct spatial access - reads slot index from gGoldyDynamic.slots[n].
public RWTexture2D<T> goldy_dyn_direct_spatial<T : ITexelElement>(uint n)
{
    uint slot = gGoldyDynamic.slots[n];
    return reinterpret<RWTexture2D<T>>(gGoldy.storageImages[slot]);
}

/// Dynamic interpolated access - reads slot index from gGoldyDynamic.slots[n].
public Texture2D<T> goldy_dyn_interpolated<T : ITexelElement>(uint n)
{
    uint slot = gGoldyDynamic.slots[n];
    return reinterpret<Texture2D<T>>(gGoldy.textures[slot]);
}

/// Dynamic sampler access - reads slot index from gGoldyDynamic.slots[n].
public SamplerState goldy_dyn_filter(uint n)
{
    uint slot = gGoldyDynamic.slots[n];
    return gGoldy.samplers[slot];
}

// Free-function wrappers for atomic intrinsics (avoids name shadowing inside
// ByteAddressView methods where InterlockedMin would resolve to `this.InterlockedMin`).
void __goldy_atomic_min(StorageBuffer<uint> buf, uint word_idx, uint value, out uint original) {
    InterlockedMin(buf[word_idx], value, original);
}
void __goldy_atomic_max(StorageBuffer<uint> buf, uint word_idx, uint value, out uint original) {
    InterlockedMax(buf[word_idx], value, original);
}
void __goldy_atomic_cmpxchg(StorageBuffer<uint> buf, uint word_idx, uint compare, uint value, out uint original) {
    InterlockedCompareExchange(buf[word_idx], compare, value, original);
}

/// Wrapper providing RWByteAddressBuffer-compatible methods over a device uint pointer.
/// On Metal, Slang's `reinterpret<RWByteAddressBuffer>(ptr)` triggers a
/// "type cannot be packed for dynamic dispatch" error when any shader in the
/// same compilation unit uses interface constraints. This struct avoids that by
/// implementing the byte-address API on top of a typed device pointer.
public struct ByteAddressView {
    StorageBuffer<uint> _ptr;

    [ForceInline]
    public uint Load(uint addr) { return _ptr[addr >> 2u]; }

    [ForceInline]
    public void Store(uint addr, uint value) { _ptr[addr >> 2u] = value; }

    [ForceInline]
    public void InterlockedMin(uint addr, uint value, out uint original_value) {
        __goldy_atomic_min(_ptr, addr >> 2u, value, original_value);
    }

    [ForceInline]
    public void InterlockedMax(uint addr, uint value, out uint original_value) {
        __goldy_atomic_max(_ptr, addr >> 2u, value, original_value);
    }

    [ForceInline]
    public void InterlockedCompareExchange(uint addr, uint compare, uint value, out uint original_value) {
        __goldy_atomic_cmpxchg(_ptr, addr >> 2u, compare, value, original_value);
    }
};

/// Dynamic byte-address buffer access for InterlockedMin/Max on Metal.
public ByteAddressView goldy_dyn_byte_address(uint n)
{
    ByteAddressView v;
    v._ptr = goldy_dyn_scattered<uint>(n);
    return v;
}

#endif
