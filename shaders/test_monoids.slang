// Compile-only test for IMonoid conformance extensions on representative GPU types.
//
// Validates that struct types resembling a 2D renderer's TagMonoid, DrawMonoid,
// and Bic can conform to IMonoid and be used with generic groupshared T[N] functions.
//
// Self-contained: all types defined locally (no external dependency).
//
// Run via: cargo test test_monoids_compiles

import goldy_exp;

#define OUT goldy_dyn_scattered<uint>(0)

// --- Simplified 2D-renderer-like types ---

struct TagMonoid {
    uint trans_ix;
    uint pathseg_ix;
    uint pathseg_offset;
    uint style_ix;
    uint path_ix;
};

struct DrawMonoid {
    uint path_ix;
    uint clip_ix;
    uint scene_offset;
    uint info_offset;
};

struct Bic {
    uint a;
    uint b;
};

// --- IMonoid conformance ---

extension TagMonoid : IMonoid {
    static TagMonoid identity() {
        TagMonoid c;
        c.trans_ix = 0; c.pathseg_ix = 0; c.pathseg_offset = 0;
        c.style_ix = 0; c.path_ix = 0;
        return c;
    }
    TagMonoid combine(TagMonoid other) {
        TagMonoid c;
        c.trans_ix = this.trans_ix + other.trans_ix;
        c.pathseg_ix = this.pathseg_ix + other.pathseg_ix;
        c.pathseg_offset = this.pathseg_offset + other.pathseg_offset;
        c.style_ix = this.style_ix + other.style_ix;
        c.path_ix = this.path_ix + other.path_ix;
        return c;
    }
}

extension DrawMonoid : IMonoid {
    static DrawMonoid identity() {
        DrawMonoid c;
        c.path_ix = 0; c.clip_ix = 0; c.scene_offset = 0; c.info_offset = 0;
        return c;
    }
    DrawMonoid combine(DrawMonoid other) {
        DrawMonoid c;
        c.path_ix = this.path_ix + other.path_ix;
        c.clip_ix = this.clip_ix + other.clip_ix;
        c.scene_offset = this.scene_offset + other.scene_offset;
        c.info_offset = this.info_offset + other.info_offset;
        return c;
    }
}

extension Bic : IMonoid {
    static Bic identity() { Bic r; r.a = 0; r.b = 0; return r; }
    Bic combine(Bic other) {
        uint m = min(this.b, other.a);
        Bic r;
        r.a = this.a + other.a - m;
        r.b = this.b + other.b - m;
        return r;
    }
}

// --- Generic reduce using IMonoid ---

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

    TagMonoid t = TagMonoid.identity();
    t.trans_ix = ix;
    t = test_reduce<TagMonoid, WG>(t, ix, sh_tag);

    DrawMonoid d = DrawMonoid.identity();
    d.path_ix = ix;
    d = test_reduce<DrawMonoid, WG>(d, ix, sh_draw);

    Bic b = Bic.identity();
    b.a = ix;
    b = test_reduce<Bic, WG>(b, ix, sh_bic);

    if (ix == 0) {
        OUT[0] = t.trans_ix;
        OUT[1] = d.path_ix;
        OUT[2] = b.a;
    }
}
