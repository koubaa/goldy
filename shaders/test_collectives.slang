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
