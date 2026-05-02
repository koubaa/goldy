// Unified access pattern functions for Goldy
// Auto-included when you import goldy_exp.
//
// Resources are passed as `uniform` entry-point parameters — plain `uint` slot
// indices for descriptors, or any scalar type for immediate values. Slang maps
// these to push constants (SPIR-V / DX12) or buffer-arg 1 (Metal) automatically.
//
// == Static Access (compile-time slot indices from parameter) ==
//
//   goldy_broadcast<T>(slot)     - Uniform/constant data (all threads same address)
//   goldy_scattered<T>(slot)     - Storage buffers (any thread, any address)
//   goldy_buf_ro<T>(slot)        - Read-only storage buffers (hardware read cache)
//   goldy_byte_address(slot)     - Byte-address buffer (for atomics)
//   goldy_interpolated<T>(slot)  - Hardware-filtered textures
//   goldy_direct_spatial<T>(slot)- Read-write 2D textures
//   goldy_filter(slot)           - Sampler states for texture filtering
//
// == Usage ==
//
//   import goldy_exp;
//
//   struct MyUniforms { float time; };
//
//   [shader("fragment")]
//   float4 fs_main(uniform uint uniforms_slot,
//                  FullscreenVarying input) : SV_Target {
//       // Rust: pass.set_push_constants_raw(&[uniforms_buf.bindless_index()])
//       float t = goldy_broadcast<MyUniforms>(uniforms_slot).time;
//       return float4(t, 0, 0, 1);
//   }
//
// == Ping-pong / Dynamic Slot Selection ==
//
//   [shader("compute")]
//   void cs_main(uniform uint read_slot, uniform uint write_slot,
//                uint3 id : SV_DispatchThreadID) {
//       // Rust: set_push_constants_raw(&[read_buf.bindless_index(), write_buf.bindless_index()])
//       StorageBuffer<uint> current = goldy_scattered<uint>(read_slot);
//       StorageBuffer<uint> next = goldy_scattered<uint>(write_slot);
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
// StorageBuffer<T>  - Read/write scattered access (any thread, any address)
// ReadOnlyBuffer<T> - Read-only scattered access (hardware read cache path)
//
// On SPIRV/DX12, these map to RWStructuredBuffer<T> and StructuredBuffer<T>.
// On Metal, these map to device T* and const device T* respectively.
// Metal cannot construct RWStructuredBuffer<T> from device pointers, so these
// aliases normalize the types across platforms.
//
// Usage:
//   var buf = goldy_scattered<Particle>(slot);    // read/write
//   var input = goldy_buf_ro<Particle>(slot);     // read-only (hardware hint)
//   buf[idx] = input[idx] + 1;
// =============================================================================

#if defined(__METAL__)
public typealias StorageBuffer<T> = Ptr<T, Access.ReadWrite, AddressSpace.Device>;
public typealias ReadOnlyBuffer<T> = Ptr<T, Access.Read, AddressSpace.Device>;
// ByteAddressView is defined as a struct later in the Metal section.
#else
public typealias StorageBuffer<T> = RWStructuredBuffer<T>;
public typealias ReadOnlyBuffer<T> = StructuredBuffer<T>;
public typealias ByteAddressView = RWByteAddressBuffer;
#endif

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

/// Get a storage buffer handle for scattered access (SPIRV).
public StorageBuffer<T> goldy_scattered<T>(uint slot)
{
    return g_GoldyStorageHeap[slot].asOpaqueDescriptor<RWStructuredBuffer<T>>();
}

/// Get a read-only storage buffer handle (SPIRV).
/// Emits a NonWritable SSBO, enabling read cache optimizations.
public ReadOnlyBuffer<T> goldy_buf_ro<T>(uint slot)
{
    return g_GoldyStorageHeap[slot].asOpaqueDescriptor<StructuredBuffer<T>>();
}

/// Get a byte-address buffer handle (SPIRV).
/// Use for InterlockedMin/Max or other byte-addressed atomic operations.
public ByteAddressView goldy_byte_address(uint slot)
{
    return g_GoldyStorageHeap[slot].asOpaqueDescriptor<RWByteAddressBuffer>();
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

/// Get a read-only storage buffer handle (DX12).
/// Uses StructuredBuffer<T> (SRV descriptor), enabling read cache optimizations.
public ReadOnlyBuffer<T> goldy_buf_ro<T>(uint slot)
{
    DescriptorHandle<StructuredBuffer<T>> handle;
    handle = DescriptorHandle<StructuredBuffer<T>>(uint2(slot, 0));
    return *handle;
}

/// Get a byte-address buffer handle (DX12).
public ByteAddressView goldy_byte_address(uint slot)
{
    DescriptorHandle<RWByteAddressBuffer> handle;
    handle = DescriptorHandle<RWByteAddressBuffer>(uint2(slot, 0));
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
// Public access functions
// Return RWStructuredBuffer<T> on all platforms for a unified API.
// On Metal, reinterpret device pointers as RWStructuredBuffer<T> so that
// callers can assign the result to an RWStructuredBuffer<T> local variable.
// -----------------------------------------------------------------------------

/// Access a uniform/constant buffer by slot index (Metal).
/// Cast the device pointer to T* and index [0] to read sizeof(T) bytes.
/// reinterpret<T>(*basePtr) is WRONG: it only reads 4 bytes (one uint) and
/// zero-fills the rest, corrupting any T larger than 4 bytes.
public T goldy_broadcast<T>(uint slot)
{
    uint* basePtr = gGoldy.uniformBuffers[slot];
    return ((Ptr<T, Access.ReadWrite, AddressSpace.Device>)basePtr)[0];
}

/// Static scattered access - read/write storage buffer by slot index.
public StorageBuffer<T> goldy_scattered<T>(uint slot)
{
    uint* basePtr = gGoldy.storageBuffers[slot];
    return (Ptr<T, Access.ReadWrite, AddressSpace.Device>)basePtr;
}

/// Static read-only scattered access - const device pointer for read cache.
public ReadOnlyBuffer<T> goldy_buf_ro<T>(uint slot)
{
    uint* basePtr = gGoldy.storageBuffers[slot];
    return (Ptr<T, Access.Read, AddressSpace.Device>)basePtr;
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

/// Static byte-address buffer access by slot index (Metal).
public ByteAddressView goldy_byte_address(uint slot)
{
    ByteAddressView v;
    v._ptr = goldy_scattered<uint>(slot);
    return v;
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
