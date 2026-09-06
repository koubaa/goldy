// Portable interlocked (atomic) cells and operations.
// Auto-included when you import goldy_exp.
//
// WGSL only allows atomics on `Atomic<T>` in storage / workgroup memory. Native
// backends (DX12, Vulkan, Metal, CUDA) use ordinary `uint`/`int` plus HLSL
// `Interlocked*` intrinsics — including plain loads/stores, which must stay
// non-atomic on those targets.
//
// The serial CPU host-callable backend (`__CPU__`) has no atomic intrinsics;
// groups execute one after another there, so plain read-modify-write is exact.
//
// `Interlocked<T>` is a 4-byte cell. Operations are [ForceInline] so native
// compiles see `InterlockedAdd`/`Or`/`Xor`/`Min`/`Max`/`Exchange` on the scalar
// field. Do not copy a cell out of a buffer or groupshared array; pass the
// in-place location (`buf[i].field`, `sh[i]`).
//
// Call: InterlockedAdd(bump[0].lines, 1), InterlockedLoad(tile.backdrop), …

implementing goldy_exp;

#ifdef __WGSL__
public struct Interlocked<T : IAtomicable> {
    Atomic<T> _v;
}
#else
public struct Interlocked<T> {
    T _v;
}
#endif

[ForceInline]
public uint InterlockedLoad(__constref Interlocked<uint> dest) {
#ifdef __WGSL__
    return dest._v.load();
#else
    return dest._v;
#endif
}

[ForceInline]
public int InterlockedLoad(__constref Interlocked<int> dest) {
#ifdef __WGSL__
    return dest._v.load();
#else
    return dest._v;
#endif
}

[ForceInline]
public void InterlockedStore(inout Interlocked<uint> dest, uint value) {
#ifdef __WGSL__
    dest._v.store(value);
#else
    dest._v = value;
#endif
}

[ForceInline]
public void InterlockedStore(inout Interlocked<int> dest, int value) {
#ifdef __WGSL__
    dest._v.store(value);
#else
    dest._v = value;
#endif
}

[ForceInline]
public uint InterlockedAdd(inout Interlocked<uint> dest, uint value) {
#ifdef __WGSL__
    return dest._v.add(value);
#elif defined(__CPU__)
    uint orig = dest._v;
    dest._v = orig + value;
    return orig;
#else
    uint orig;
    InterlockedAdd(dest._v, value, orig);
    return orig;
#endif
}

[ForceInline]
public int InterlockedAdd(inout Interlocked<int> dest, int value) {
#ifdef __WGSL__
    return dest._v.add(value);
#elif defined(__CPU__)
    int orig = dest._v;
    dest._v = orig + value;
    return orig;
#else
    int orig;
    InterlockedAdd(dest._v, value, orig);
    return orig;
#endif
}

[ForceInline]
public void InterlockedOr(inout Interlocked<uint> dest, uint value) {
#ifdef __WGSL__
    dest._v.or(value);
#elif defined(__CPU__)
    dest._v = dest._v | value;
#else
    InterlockedOr(dest._v, value);
#endif
}

[ForceInline]
public void InterlockedXor(inout Interlocked<uint> dest, uint value) {
#ifdef __WGSL__
    dest._v.xor(value);
#elif defined(__CPU__)
    dest._v = dest._v ^ value;
#else
    InterlockedXor(dest._v, value);
#endif
}

[ForceInline]
public void InterlockedMin(inout Interlocked<uint> dest, uint value) {
#ifdef __WGSL__
    dest._v.min(value);
#elif defined(__CPU__)
    dest._v = min(dest._v, value);
#else
    InterlockedMin(dest._v, value);
#endif
}

[ForceInline]
public void InterlockedMin(inout Interlocked<int> dest, int value) {
#ifdef __WGSL__
    dest._v.min(value);
#elif defined(__CPU__)
    dest._v = min(dest._v, value);
#else
    InterlockedMin(dest._v, value);
#endif
}

[ForceInline]
public void InterlockedMax(inout Interlocked<uint> dest, uint value) {
#ifdef __WGSL__
    dest._v.max(value);
#elif defined(__CPU__)
    dest._v = max(dest._v, value);
#else
    InterlockedMax(dest._v, value);
#endif
}

[ForceInline]
public void InterlockedMax(inout Interlocked<int> dest, int value) {
#ifdef __WGSL__
    dest._v.max(value);
#elif defined(__CPU__)
    dest._v = max(dest._v, value);
#else
    InterlockedMax(dest._v, value);
#endif
}

[ForceInline]
public uint InterlockedExchange(inout Interlocked<uint> dest, uint value) {
#ifdef __WGSL__
    return dest._v.exchange(value);
#elif defined(__CPU__)
    uint orig = dest._v;
    dest._v = value;
    return orig;
#else
    uint orig;
    InterlockedExchange(dest._v, value, orig);
    return orig;
#endif
}

[ForceInline]
public int InterlockedExchange(inout Interlocked<int> dest, int value) {
#ifdef __WGSL__
    return dest._v.exchange(value);
#elif defined(__CPU__)
    int orig = dest._v;
    dest._v = value;
    return orig;
#else
    int orig;
    InterlockedExchange(dest._v, value, orig);
    return orig;
#endif
}

[ForceInline]
public void InterlockedExchange(inout Interlocked<uint> dest, uint value, out uint original) {
#ifdef __WGSL__
    original = dest._v.exchange(value);
#elif defined(__CPU__)
    original = dest._v;
    dest._v = value;
#else
    InterlockedExchange(dest._v, value, original);
#endif
}

[ForceInline]
public void InterlockedExchange(inout Interlocked<int> dest, int value, out int original) {
#ifdef __WGSL__
    original = dest._v.exchange(value);
#elif defined(__CPU__)
    original = dest._v;
    dest._v = value;
#else
    InterlockedExchange(dest._v, value, original);
#endif
}
