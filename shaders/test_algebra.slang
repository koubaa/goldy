// Compile-only test for goldy_exp algebraic abstractions.
//
// Validates three critical Slang features across SPIRV / DXIL / Metal:
//   1. IMonoid interface definition and conformance via extension
//   2. Generic function with `groupshared T[N]` parameter (T generic, N compile-time int)
//   3. Monomorphization produces working code for multiple monoid types
//
// Run via: cargo test test_algebra_compiles

import goldy_exp;

// ---------------------------------------------------------------------------
// Test monoid: uint under addition
// ---------------------------------------------------------------------------
struct UintAdd {
    uint val;
};

extension UintAdd : IMonoid {
    static UintAdd identity() {
        UintAdd r;
        r.val = 0;
        return r;
    }
    UintAdd combine(UintAdd other) {
        UintAdd r;
        r.val = this.val + other.val;
        return r;
    }
}

// ---------------------------------------------------------------------------
// Test monoid: int2 under componentwise min
// ---------------------------------------------------------------------------
struct Int2Min {
    int2 val;
};

extension Int2Min : IMonoid {
    static Int2Min identity() {
        Int2Min r;
        r.val = int2(0x7fffffff, 0x7fffffff);
        return r;
    }
    Int2Min combine(Int2Min other) {
        Int2Min r;
        r.val = min(this.val, other.val);
        return r;
    }
}

// ---------------------------------------------------------------------------
// Gate test: generic function accepting groupshared T[N] parameter
// This is the critical backend validation — Slang must lower the groupshared
// array reference correctly for SPIRV, DXIL, and Metal.
// ---------------------------------------------------------------------------
[ForceInline]
T generic_groupshared_reduce<T : IMonoid, let N : int>(
    T val, uint local_ix, groupshared T scratch[N]
) {
    scratch[local_ix] = val;
    for (uint i = 0; i < firstbitlow(N); i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_ix + (1u << i) < N) {
            val = val.combine(scratch[local_ix + (1u << i)]);
        }
        GroupMemoryBarrierWithGroupSync();
        scratch[local_ix] = val;
    }
    return val;
}

// Left-sweep variant (prefix scan direction)
[ForceInline]
T generic_groupshared_scan<T : IMonoid, let N : int>(
    T val, uint local_ix, groupshared T scratch[N]
) {
    scratch[local_ix] = val;
    for (uint i = 0; i < firstbitlow(N); i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_ix >= (1u << i)) {
            val = scratch[local_ix - (1u << i)].combine(val);
        }
        GroupMemoryBarrierWithGroupSync();
        scratch[local_ix] = val;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Compute entry point exercising all of the above
// ---------------------------------------------------------------------------
static const uint WG_SIZE = 64;

groupshared UintAdd sh_uint[WG_SIZE];
groupshared Int2Min sh_int2[WG_SIZE];

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
    uint ix = local_id.x;

    // Exercise IMonoid identity
    UintAdd u = UintAdd.identity();
    u.val = ix;

    Int2Min m = Int2Min.identity();
    m.val = int2(int(ix), -int(ix));

    // Exercise generic_groupshared_reduce with UintAdd (right-sweep)
    UintAdd reduced = generic_groupshared_reduce<UintAdd, WG_SIZE>(u, ix, sh_uint);

    // Exercise generic_groupshared_scan with Int2Min (left-sweep)
    Int2Min scanned = generic_groupshared_scan<Int2Min, WG_SIZE>(m, ix, sh_int2);

    // Write results to prevent dead-code elimination
    if (ix == 0) {
        OUT[0] = reduced.val;
        OUT[1] = uint(scanned.val.x);
        OUT[2] = uint(scanned.val.y);
    }
}
