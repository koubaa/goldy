// Workgroup collective operations for parallel GPU algorithms (EXPERIMENTAL)
// Part of the goldy_exp module
//
// Provides generic workgroup-level primitives that replace hand-rolled
// scan/reduce loops. All functions are [ForceInline] to support groupshared
// parameters on DXIL (DXC rejects groupshared in function signatures).
//
// N must be a power of 2 in all functions below.

implementing goldy_exp;

/// Inclusive right-reduce: each thread accumulates values from higher-indexed
/// neighbors. After completion, thread 0 holds the total over the workgroup.
///
/// scratch[local_ix] contains the partial reduction from local_ix to N-1.
/// No trailing barrier — caller must sync before cross-lane reads from scratch.
[ForceInline]
public T workgroup_reduce<T : IMonoid, let N : int>(
    T val, uint local_ix, groupshared T scratch[N]
) {
    scratch[local_ix] = val;
    for (uint i = 0; i < firstbitlow(N); i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_ix + (1u << i) < N)
            val = val.combine(scratch[local_ix + (1u << i)]);
        GroupMemoryBarrierWithGroupSync();
        scratch[local_ix] = val;
    }
    return val;
}

/// Inclusive left-sweep prefix scan: each thread accumulates values from
/// lower-indexed neighbors. Thread k holds combine(v[0], ..., v[k]).
///
/// scratch[local_ix] contains the inclusive scan result.
/// No trailing barrier — caller must sync before cross-lane reads from scratch.
[ForceInline]
public T workgroup_inclusive_scan<T : IMonoid, let N : int>(
    T val, uint local_ix, groupshared T scratch[N]
) {
    scratch[local_ix] = val;
    for (uint i = 0; i < firstbitlow(N); i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_ix >= (1u << i))
            val = scratch[local_ix - (1u << i)].combine(val);
        GroupMemoryBarrierWithGroupSync();
        scratch[local_ix] = val;
    }
    return val;
}

/// Broadcast a value from thread 0 to all threads in the workgroup.
/// Includes a trailing barrier — the returned value is immediately usable.
[ForceInline]
public T workgroup_broadcast<T>(T value, uint local_ix, groupshared T slot[1]) {
    if (local_ix == 0) slot[0] = value;
    GroupMemoryBarrierWithGroupSync();
    return slot[0];
}

/// Branchless binary search over a shared-memory prefix sum array.
/// Returns the upper-bound index: the largest ix such that
/// prefix_sums[ix - 1] <= key (with ix = 0 if key < prefix_sums[0]).
///
/// This is ModernGPU's "load-balanced search" — the standard way to invert
/// a prefix sum to find which element owns a given position.
[ForceInline]
public uint workgroup_upper_bound<let LG_N : int>(
    uint key, groupshared uint prefix_sums[1 << LG_N]
) {
    uint ix = 0;
    for (uint i = 0; i < LG_N; i++) {
        uint probe = ix + ((1u << LG_N) / 2u >> i);
        if (key >= prefix_sums[probe - 1])
            ix = probe;
    }
    return ix;
}
