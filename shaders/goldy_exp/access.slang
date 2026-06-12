// Unified access pattern functions for Goldy
// Auto-included when you import goldy_exp.
//
// Resources are passed as `uniform` entry-point parameters — plain `uint` slot
// indices for descriptors, or any scalar type for immediate values. Slang maps
// these to push constants (SPIR-V / DX12) or buffer-arg 1 (Metal) automatically.
//
// =========================================================================
// == Preferred: [goldy_*] typed entry points (virtual main) ==
// =========================================================================
//
//   [goldy_compute]  / [goldy_vertex]  / [goldy_fragment]
//
//   Use the resource type aliases below as entry-point parameters. The Goldy
//   compiler resolves the slot index to a buffer/texture handle and passes it
//   directly — the parameter has the full native Slang type, so all built-in
//   methods (indexing, field writes, GetDimensions, Sample, etc.) work directly.
//
//   Resource type aliases (= their underlying Slang type):
//     Scattered<T>      = StorageBuffer<T>    data[i], data[i].field = v
//     BufRO<T>          = ReadOnlyBuffer<T>   data[i]
//     Interpolated<T>   = Texture2D<T>        tex.Sample(samp, uv)
//     DirectSpatial<T>  = RWTexture2D<T>      img[int2(x,y)], img.GetDimensions(w,h)
//     ByteAddress       = ByteAddressView     .Load/.Store/.Interlocked*
//     Filter            = SamplerState        tex.Sample(samp, uv)
//
//   Broadcast (constant buffer): declare the struct type directly as a parameter.
//   The codegen recognises any non-resource, non-SV struct as a constant-buffer
//   fetch and resolves it via goldy_broadcast<T>(slot) automatically.
//
//   System-value types (codegen maps these to SV_* semantics):
//     ThreadId          - SV_DispatchThreadID  (.x .y .z)
//     GroupThreadId     - SV_GroupThreadID     (.x .y .z)
//     GroupId           - SV_GroupID           (.x .y .z)
//     VertexId          - SV_VertexID          (.value)
//     InstanceId        - SV_InstanceID        (.value)
//     IsFrontFace       - SV_IsFrontFace       (.value)
//
//   Example:
//
//   [goldy_compute]
//   [numthreads(64, 1, 1)]
//   void cs_main(MyUniforms cfg, Scattered<uint> data, ThreadId id) {
//       data[id.x] = data[id.x] + cfg.base;
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
// The virtual-main resource type aliases below build on these:
//   Scattered<T>  = StorageBuffer<T>   (read/write, field-level writes supported)
//   BufRO<T>      = ReadOnlyBuffer<T>  (read-only, hardware read-cache hint)
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
// Frame-table protocol (record/execute index routing)
//
// Two bindless slots are reserved at device init:
//   GOLDY_FRAME_SELECTOR_SLOT — u32 row index for this submission
//   GOLDY_FRAME_TABLE_SLOT    — device-local u32 table (row * STRIDE + slot)
//
// Push constant `_rs0` / PushLayout._reserved[0] carries the dispatch base
// offset within the row.  Virtual-main preambles call goldy_frame_table_index.
// =============================================================================

static const uint GOLDY_FRAME_SELECTOR_SLOT = 0;
static const uint GOLDY_FRAME_TABLE_SLOT = 1;
static const uint GOLDY_FRAME_TABLE_ROW_STRIDE = 512;

/// Resolve bindless index `slot_k` for dispatch table base `dispatch_base`.
///
/// On DX12/Vulkan, `dispatch_base` is the within-row offset and a selector cell
/// (written by the GPU prologue copy) provides the current row.
///
/// On Metal, the CPU prologue encodes the absolute table offset directly into
/// `dispatch_base` (= row * ROW_STRIDE + within-row offset), so no selector
/// read is needed.  This avoids the shared-mutable-selector race that would
/// occur when multiple command buffers are in flight simultaneously.
public uint goldy_frame_table_index(uint dispatch_base, uint slot_k)
{
#if defined(__METAL__)
    return goldy_scattered<uint>(GOLDY_FRAME_TABLE_SLOT)[dispatch_base + slot_k];
#else
    uint sel = goldy_scattered<uint>(GOLDY_FRAME_SELECTOR_SLOT)[0];
    uint row_offset = sel * GOLDY_FRAME_TABLE_ROW_STRIDE + dispatch_base + slot_k;
    return goldy_scattered<uint>(GOLDY_FRAME_TABLE_SLOT)[row_offset];
#endif
}

// =============================================================================
// SPIRV Implementation
// Uses Goldy's custom binding layout via getDescriptorFromHandle override
// =============================================================================
#ifdef __SPIRV__

public T goldy_broadcast<T>(uint slot)
{
    return g_GoldyUniformHeap[slot].asOpaqueDescriptor<ConstantBuffer<T>>();
}

public StorageBuffer<T> goldy_scattered<T>(uint slot)
{
    return g_GoldyStorageHeap[slot].asOpaqueDescriptor<RWStructuredBuffer<T>>();
}

public ReadOnlyBuffer<T> goldy_buf_ro<T>(uint slot)
{
    return g_GoldyStorageHeap[slot].asOpaqueDescriptor<StructuredBuffer<T>>();
}

public ByteAddressView goldy_byte_address(uint slot)
{
    return g_GoldyStorageHeap[slot].asOpaqueDescriptor<RWByteAddressBuffer>();
}

public Texture2D<T> goldy_interpolated<T : ITexelElement>(uint slot)
{
    return g_GoldyTextureHeap[slot].asOpaqueDescriptor<Texture2D<T>>();
}

public RWTexture2D<T> goldy_direct_spatial<T : ITexelElement>(uint slot)
{
    return g_GoldyStorageImageHeap[slot].asOpaqueDescriptor<RWTexture2D<T>>();
}

public SamplerState goldy_filter(uint slot)
{
    return g_GoldySamplerHeap[slot].asOpaqueDescriptor<SamplerState>();
}

// =============================================================================
// DX12 Implementation
// Uses DescriptorHandle<T> which Slang maps to ResourceDescriptorHeap
// =============================================================================
#elif defined(__DX12__)

public T goldy_broadcast<T>(uint slot)
{
    DescriptorHandle<ConstantBuffer<T>> handle;
    handle = DescriptorHandle<ConstantBuffer<T>>(uint2(slot, 0));
    return *handle;
}

public StorageBuffer<T> goldy_scattered<T>(uint slot)
{
    DescriptorHandle<RWStructuredBuffer<T>> handle;
    handle = DescriptorHandle<RWStructuredBuffer<T>>(uint2(slot, 0));
    return *handle;
}

public ReadOnlyBuffer<T> goldy_buf_ro<T>(uint slot)
{
    DescriptorHandle<StructuredBuffer<T>> handle;
    handle = DescriptorHandle<StructuredBuffer<T>>(uint2(slot, 0));
    return *handle;
}

public ByteAddressView goldy_byte_address(uint slot)
{
    DescriptorHandle<RWByteAddressBuffer> handle;
    handle = DescriptorHandle<RWByteAddressBuffer>(uint2(slot, 0));
    return *handle;
}

public Texture2D<T> goldy_interpolated<T : ITexelElement>(uint slot)
{
    DescriptorHandle<Texture2D<T>> handle;
    handle = DescriptorHandle<Texture2D<T>>(uint2(slot, 0));
    return *handle;
}

public RWTexture2D<T> goldy_direct_spatial<T : ITexelElement>(uint slot)
{
    DescriptorHandle<RWTexture2D<T>> handle;
    handle = DescriptorHandle<RWTexture2D<T>>(uint2(slot, 0));
    return *handle;
}

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

public T goldy_broadcast<T>(uint slot)
{
    uint* basePtr = gGoldy.uniformBuffers[slot];
    return ((Ptr<T, Access.ReadWrite, AddressSpace.Device>)basePtr)[0];
}

public StorageBuffer<T> goldy_scattered<T>(uint slot)
{
    uint* basePtr = gGoldy.storageBuffers[slot];
    return (Ptr<T, Access.ReadWrite, AddressSpace.Device>)basePtr;
}

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

public ByteAddressView goldy_byte_address(uint slot)
{
    ByteAddressView v;
    v._ptr = goldy_scattered<uint>(slot);
    return v;
}

public Texture2D<T> goldy_interpolated<T : ITexelElement>(uint slot)
{
    return reinterpret<Texture2D<T>>(gGoldy.textures[slot]);
}

public RWTexture2D<T> goldy_direct_spatial<T : ITexelElement>(uint slot)
{
    return reinterpret<RWTexture2D<T>>(gGoldy.storageImages[slot]);
}

public SamplerState goldy_filter(uint slot)
{
    return gGoldy.samplers[slot];
}

#endif

// =============================================================================
// Resource type aliases — for use with [goldy_*] virtual entry points.
//
// These are plain typealiases for the underlying Slang buffer/texture types.
// The Goldy codegen resolves the push-constant slot index to a live handle
// (via goldy_scattered<T>(), goldy_buf_ro<T>(), etc.) and binds it to a local
// of the alias type before calling the user function. The user function receives
// the handle directly — no wrapper struct, no .buf()/.raw()/.tex() needed.
//
// For uniform/constant data: declare the struct type directly as a parameter in
// the virtual main signature. The codegen recognises any non-resource, non-SV
// struct as a broadcast and calls goldy_broadcast<T>(slot) automatically.
// =============================================================================

/// Read/write storage buffer (= StorageBuffer<T> = RWStructuredBuffer<T> on SPIRV/DX12).
/// All native indexing and field-level writes work directly: `data[i].field = v`.
public typealias Scattered<T> = StorageBuffer<T>;

/// Read-only storage buffer (= ReadOnlyBuffer<T> = StructuredBuffer<T> on SPIRV/DX12).
/// Index with `buf[i]` for element access.
public typealias BufRO<T> = ReadOnlyBuffer<T>;

/// Hardware-filtered 2D texture (= Texture2D<T>).
/// Use all native Texture2D methods: `.Sample(samp, uv)`, `.Load(loc)`, etc.
public typealias Interpolated<T : ITexelElement> = Texture2D<T>;

/// Read-write 2D texture (= RWTexture2D<T>).
/// Use all native methods: `img[int2(x,y)]`, `img.GetDimensions(w,h)`, etc.
public typealias DirectSpatial<T : ITexelElement> = RWTexture2D<T>;

/// Byte-address buffer (= ByteAddressView = RWByteAddressBuffer on SPIRV/DX12).
/// Use native methods: `.Load(addr)`, `.Store(addr, v)`, `.InterlockedMin(...)`, etc.
public typealias ByteAddress = ByteAddressView;

/// Sampler state (= SamplerState).
/// Pass directly to texture `.Sample(filter, uv)` calls.
public typealias Filter = SamplerState;

// =============================================================================
// System-value wrapper types — for use with [goldy_*] virtual entry points.
//
// Declare these as parameters in a [goldy_*] function to receive GPU-provided
// system values. The codegen maps each type to its SV_* semantic automatically.
// =============================================================================

/// SV_DispatchThreadID: global thread index in a compute dispatch.
/// Access individual components as `.x`, `.y`, `.z`, or swizzles `.xy`, `.xz`, `.yz`, `.xyz`.
public struct ThreadId {
    public uint x;
    public uint y;
    public uint z;
    public __init(uint3 v) { x = v.x; y = v.y; z = v.z; }
    public property uint2 xy { get { return uint2(x, y); } }
    public property uint2 xz { get { return uint2(x, z); } }
    public property uint2 yz { get { return uint2(y, z); } }
    public property uint3 xyz { get { return uint3(x, y, z); } }
}

/// SV_GroupThreadID: thread index within the current workgroup.
/// Access individual components as `.x`, `.y`, `.z`, or swizzles `.xy`, `.xz`, `.yz`, `.xyz`.
public struct GroupThreadId {
    public uint x;
    public uint y;
    public uint z;
    public __init(uint3 v) { x = v.x; y = v.y; z = v.z; }
    public property uint2 xy { get { return uint2(x, y); } }
    public property uint2 xz { get { return uint2(x, z); } }
    public property uint2 yz { get { return uint2(y, z); } }
    public property uint3 xyz { get { return uint3(x, y, z); } }
}

/// SV_GroupID: workgroup index in the dispatch grid.
/// Access individual components as `.x`, `.y`, `.z`, or swizzles `.xy`, `.xz`, `.yz`, `.xyz`.
public struct GroupId {
    public uint x;
    public uint y;
    public uint z;
    public __init(uint3 v) { x = v.x; y = v.y; z = v.z; }
    public property uint2 xy { get { return uint2(x, y); } }
    public property uint2 xz { get { return uint2(x, z); } }
    public property uint2 yz { get { return uint2(y, z); } }
    public property uint3 xyz { get { return uint3(x, y, z); } }
}

/// SV_VertexID: index of the current vertex being processed.
/// Read the raw index as `.value`.
public struct VertexId {
    public uint value;
    public __init(uint v) { value = v; }
}

/// SV_InstanceID: index of the current instance in an instanced draw.
/// Read the raw index as `.value`.
public struct InstanceId {
    public uint value;
    public __init(uint v) { value = v; }
}

/// SV_IsFrontFace: true if the current primitive is front-facing.
/// Read the flag as `.value`.
public struct IsFrontFace {
    public bool value;
    public __init(bool v) { value = v; }
}
