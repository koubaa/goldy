// Compile-only test for workgroup collective operations.
//
// Validates that workgroup_reduce, workgroup_inclusive_scan, workgroup_broadcast,
// workgroup_upper_bound, IScannable, and mapped_workgroup_reduce compile with
// IMonoid types (including uint) and ekrano's TagMonoid/DrawMonoid/Bic/Bbox2D
// across SPIRV, DXIL, and Metal.
//
// Run via: cargo test test_collectives_compiles

import goldy_exp;
import ekrano_shared;

#define OUT goldy_dyn_scattered<uint>(0)

// --- IMonoid conformance for ekrano types ---

extension TagMonoid : IMonoid {
    static TagMonoid identity() { return tag_monoid_identity(); }
    TagMonoid combine(TagMonoid other) { return combine_tag_monoid(this, other); }
}

extension DrawMonoid : IMonoid {
    static DrawMonoid identity() { return draw_monoid_identity(); }
    DrawMonoid combine(DrawMonoid other) { return combine_draw_monoid(this, other); }
}

extension Bic : IMonoid {
    static Bic identity() { Bic r; r.a = 0; r.b = 0; return r; }
    Bic combine(Bic other) { return bic_combine(this, other); }
}

public struct Bbox2D {
    public float4 v;
};

extension Bbox2D : IMonoid {
    static Bbox2D identity() { Bbox2D r; r.v = float4(-1e9, -1e9, 1e9, 1e9); return r; }
    Bbox2D combine(Bbox2D other) { Bbox2D r; r.v = bbox_intersect(this.v, other.v); return r; }
}

// --- IScannable conformance for ekrano map functions ---

struct PathTagScanner : IScannable<TagMonoid> {
    static TagMonoid map(uint raw_input) { return reduce_tag(raw_input); }
};

struct DrawTagScanner : IScannable<DrawMonoid> {
    static DrawMonoid map(uint raw_input) { return map_draw_tag(raw_input); }
};

static const uint WG_SIZE = 256;
static const uint LG_WG_SIZE = 8;

groupshared TagMonoid  sh_tag[WG_SIZE];
groupshared DrawMonoid sh_draw[WG_SIZE];
groupshared Bic        sh_bic[WG_SIZE];
groupshared Bbox2D     sh_bbox[WG_SIZE];
groupshared uint       sh_uint[WG_SIZE];
groupshared uint       sh_broadcast[1];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 local_id : SV_GroupThreadID) {
    uint ix = local_id.x;

    // --- workgroup_reduce with TagMonoid ---
    TagMonoid t = TagMonoid.identity();
    t.trans_ix = ix;
    t = workgroup_reduce(t, ix, sh_tag);

    // --- workgroup_reduce with DrawMonoid ---
    DrawMonoid d = DrawMonoid.identity();
    d.path_ix = ix;
    d = workgroup_reduce(d, ix, sh_draw);

    // --- workgroup_reduce with Bic ---
    Bic b = Bic.identity();
    b.a = ix;
    b = workgroup_reduce(b, ix, sh_bic);

    // --- workgroup_inclusive_scan with TagMonoid ---
    TagMonoid ts = TagMonoid.identity();
    ts.pathseg_ix = ix;
    ts = workgroup_inclusive_scan(ts, ix, sh_tag);

    // --- workgroup_inclusive_scan with Bbox2D ---
    Bbox2D bbox;
    bbox.v = float4(float(ix), float(ix), float(ix + 1), float(ix + 1));
    bbox = workgroup_inclusive_scan(bbox, ix, sh_bbox);

    // --- workgroup_inclusive_scan with uint ---
    uint count = ix + 1;
    count = workgroup_inclusive_scan(count, ix, sh_uint);

    // --- workgroup_broadcast ---
    uint bcast = workgroup_broadcast(t.trans_ix, ix, sh_broadcast);

    // --- workgroup_upper_bound ---
    GroupMemoryBarrierWithGroupSync();
    uint found = workgroup_upper_bound<LG_WG_SIZE>(ix * 2, sh_uint);

    // --- mapped_workgroup_reduce with PathTagScanner (IScannable<TagMonoid>) ---
    TagMonoid mapped_tag = mapped_workgroup_reduce<TagMonoid, PathTagScanner>(ix, ix, sh_tag);

    // --- mapped_workgroup_reduce with DrawTagScanner (IScannable<DrawMonoid>) ---
    DrawMonoid mapped_draw = mapped_workgroup_reduce<DrawMonoid, DrawTagScanner>(ix, ix, sh_draw);

    // --- IScannable.map standalone ---
    TagMonoid from_scanner = PathTagScanner.map(0x01010101u);
    DrawMonoid from_draw_scanner = DrawTagScanner.map(DRAWTAG_FILL_COLOR);

    if (ix == 0) {
        OUT[0] = t.trans_ix;
        OUT[1] = d.path_ix;
        OUT[2] = b.a;
        OUT[3] = ts.pathseg_ix;
        OUT[4] = asuint(bbox.v.x);
        OUT[5] = count;
        OUT[6] = bcast;
        OUT[7] = found;
        OUT[8] = mapped_tag.trans_ix;
        OUT[9] = mapped_draw.path_ix;
        OUT[10] = from_scanner.path_ix;
        OUT[11] = from_draw_scanner.clip_ix;
    }
}
