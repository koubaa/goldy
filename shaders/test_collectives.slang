// Compile-only test for workgroup collective operations.
//
// Validates that workgroup_reduce, workgroup_inclusive_scan, workgroup_broadcast,
// workgroup_upper_bound, IScannable, and mapped_workgroup_reduce compile with
// IMonoid types (including uint) and representative multi-field struct monoids
// across SPIRV, DXIL, and Metal.
//
// Self-contained: all types defined locally (no ekrano dependency).
//
// Run via: cargo test test_collectives_compiles

import goldy_exp;

#define OUT goldy_dyn_scattered<uint>(0)

// --- Simplified ekrano-like types ---

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

struct Bbox2D {
    float4 v;
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

extension Bbox2D : IMonoid {
    static Bbox2D identity() { Bbox2D r; r.v = float4(-1e9, -1e9, 1e9, 1e9); return r; }
    Bbox2D combine(Bbox2D other) {
        Bbox2D r;
        r.v = float4(max(this.v.xy, other.v.xy), min(this.v.zw, other.v.zw));
        return r;
    }
}

// --- IScannable conformance ---

static const uint DRAWTAG_NOP = 0;
static const uint DRAWTAG_FILL_COLOR = 0x44;

TagMonoid reduce_tag(uint tag_word) {
    TagMonoid c;
    uint point_count = tag_word & 0x3030303u;
    c.pathseg_ix = countbits((point_count * 7u) & 0x4040404u);
    c.trans_ix = countbits(tag_word & (0x20 * 0x1010101u));
    uint n_points = point_count + ((tag_word >> 2u) & 0x1010101u);
    uint a = n_points + (n_points & (((tag_word >> 3u) & 0x1010101u) * 15u));
    a += a >> 8u;
    a += a >> 16u;
    c.pathseg_offset = a & 0xffu;
    c.path_ix = countbits(tag_word & (0x10 * 0x1010101u));
    c.style_ix = countbits(tag_word & (0x40 * 0x1010101u)) * 4;
    return c;
}

DrawMonoid map_draw_tag(uint tag_word) {
    DrawMonoid c;
    c.path_ix = (tag_word != DRAWTAG_NOP) ? 1u : 0u;
    c.clip_ix = tag_word & 1u;
    c.scene_offset = (tag_word >> 2u) & 0x07u;
    c.info_offset = (tag_word >> 6u) & 0x0fu;
    return c;
}

struct PathTagScanner : IScannable<TagMonoid> {
    static TagMonoid map(uint raw_input) { return reduce_tag(raw_input); }
};

struct DrawTagScanner : IScannable<DrawMonoid> {
    static DrawMonoid map(uint raw_input) { return map_draw_tag(raw_input); }
};

// --- Test shader ---

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
