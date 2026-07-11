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

/// Map-then-reduce: apply IScannable.map to raw data, then workgroup_reduce.
///
/// Captures the pattern shared by pathtag_reduce (map = reduce_tag) and
/// draw_reduce (map = map_draw_tag): the same algorithm under different
/// (Map, Monoid) pairs.
[ForceInline]
public T mapped_workgroup_reduce<T : IMonoid, S : IScannable<T>, let N : int>(
    uint raw_data, uint local_ix, groupshared T scratch[N]
) {
    T mapped = S.map(raw_data);
    return workgroup_reduce<T, N>(mapped, local_ix, scratch);
}

/// Inclusive left-sweep prefix scan: each thread accumulates values from
/// lower-indexed neighbors. Thread k holds combine(v[0], ..., v[k]).
///
/// scratch[local_ix] contains the inclusive scan result.
/// No trailing barrier — caller must sync before cross-lane reads from scratch.
///
/// NOTE: Slang #10641 — cross-module [ForceInline] functions with groupshared
/// parameters may produce incorrect DXIL. Inline the scan loop at the call site
/// as a workaround (see ekrano coarse.slang, goldy compute_integration tests).
[ForceInline]
public T workgroup_inclusive_scan<T : IMonoid, let N : int>(
    T val, uint local_ix, groupshared T scratch[N]
) {
    scratch[local_ix] = val;
    for (uint i = 0; i < firstbitlow(N); i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_ix >= (1u << i)) {
            T other = scratch[local_ix - (1u << i)];
            val = other.combine(val);
        }
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

/// Inclusive prefix sum (additive monoid on uint) across `WG_SIZE_PARAM` threads using
/// wave intrinsics plus a tiny exclusive-prefix pass over per-wave totals on thread 0.
///
/// Requires `[numthreads(WG_SIZE_PARAM, 1, 1)]` (or equivalent dense mapping), subgroup
/// wave ops (SM6 / Vulkan subgroup arithmetic / Metal SIMD-groups), and
/// `WG_SIZE_PARAM % WaveGetLaneCount() == 0`.
///
/// `scratch_wave_totals` must hold at least `WG_SIZE_PARAM / min_subgroup_size` entries;
/// sizes <=32 cover `WG_SIZE_PARAM == 256` for subgroup sizes down to 8.
///
[ForceInline]
public uint workgroup_inclusive_scan_wave_uint_sum<let WG_SIZE_PARAM : int>(
    uint val,
    uint local_ix,
    groupshared uint scratch_wave_totals[32]
) {
    uint lc = WaveGetLaneCount();
    uint nw = WG_SIZE_PARAM / lc;
    uint wave_ix = local_ix / lc;

    uint inclusive_in_wave = WavePrefixSum(val) + val;
    uint wave_total = WaveActiveSum(val);

    if (WaveIsFirstLane())
        scratch_wave_totals[wave_ix] = wave_total;

    GroupMemoryBarrierWithGroupSync();

    if (local_ix == 0) {
        uint run = 0;
        for (uint i = 0; i < nw; i++) {
            uint s = scratch_wave_totals[i];
            scratch_wave_totals[i] = run;
            run += s;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    uint exclusive_between_waves = scratch_wave_totals[wave_ix];
    return exclusive_between_waves + inclusive_in_wave;
}
