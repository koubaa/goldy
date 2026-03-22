// Compile-only test for IMonoid conformance extensions on ekrano types.
//
// Validates that TagMonoid, DrawMonoid, and Bic conform to IMonoid and can be
// used with generic functions accepting groupshared T[N] parameters.
//
// Run via: cargo test test_ekrano_monoids_compiles

import goldy_exp;
import ekrano_shared;

#define OUT goldy_dyn_scattered<uint>(0)

[ForceInline]
T test_reduce<T : IMonoid, let N : int>(
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

static const uint WG = 256;

groupshared TagMonoid  sh_tag[WG];
groupshared DrawMonoid sh_draw[WG];
groupshared Bic        sh_bic[WG];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 local_id : SV_GroupThreadID) {
    uint ix = local_id.x;

    // TagMonoid: identity + combine via IMonoid
    TagMonoid t = TagMonoid.identity();
    t.trans_ix = ix;
    t = test_reduce<TagMonoid, WG>(t, ix, sh_tag);

    // DrawMonoid: identity + combine via IMonoid
    DrawMonoid d = DrawMonoid.identity();
    d.path_ix = ix;
    d = test_reduce<DrawMonoid, WG>(d, ix, sh_draw);

    // Bic: identity + combine via IMonoid
    Bic b = Bic.identity();
    b.a = ix;
    b = test_reduce<Bic, WG>(b, ix, sh_bic);

    if (ix == 0) {
        OUT[0] = t.trans_ix;
        OUT[1] = d.path_ix;
        OUT[2] = b.a;
    }
}
