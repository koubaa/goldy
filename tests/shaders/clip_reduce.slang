// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;
static const uint LG_WG_SIZE = 8;

groupshared Bic sh_bic[256];
groupshared uint sh_parent[256];
groupshared uint sh_path_ix[256];

// Slots: 0 = clip_inp, 1 = path_bboxes, 2 = reduced, 3 = clip_out

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID, uint3 wg_id : SV_GroupID) {
    ReadOnlyBuffer<ClipInp> clip_inp = goldy_dyn_buf_ro<ClipInp>(0);
    ReadOnlyBuffer<PathBbox> path_bboxes = goldy_dyn_buf_ro<PathBbox>(1);
    StorageBuffer<Bic> reduced = goldy_dyn_scattered<Bic>(2);
    StorageBuffer<ClipEl> clip_out = goldy_dyn_scattered<ClipEl>(3);

    int inp = clip_inp[global_id.x].path_ix;
    bool is_push = inp >= 0;
    Bic bic;
    bic.a = is_push ? 0 : 1;
    bic.b = is_push ? 1 : 0;

    sh_bic[local_id.x] = bic;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x + (1u << i) < WG_SIZE)
            bic = bic.combine(sh_bic[local_id.x + (1u << i)]);
        GroupMemoryBarrierWithGroupSync();
        sh_bic[local_id.x] = bic;
    }
    if (local_id.x == 0) {
        reduced[wg_id.x] = bic;
    }
    GroupMemoryBarrierWithGroupSync();
    uint size = sh_bic[0].b;
    bic.a = 0;
    bic.b = 0;
    if (local_id.x + 1 < WG_SIZE) {
        bic = sh_bic[local_id.x + 1];
    }
    if (is_push && bic.a == 0) {
        uint local_ix = size - bic.b - 1;
        sh_parent[local_ix] = local_id.x;
        sh_path_ix[local_ix] = uint(inp);
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_id.x < size) {
        uint path_ix = sh_path_ix[local_id.x];
        PathBbox path_bbox = path_bboxes[path_ix];
        uint parent_ix = sh_parent[local_id.x] + wg_id.x * WG_SIZE;
        float4 bbox = float4(float(path_bbox.x0), float(path_bbox.y0), float(path_bbox.x1), float(path_bbox.y1));
        ClipEl el;
        el.parent_ix = parent_ix;
        el._pad0 = 0;
        el._pad1 = 0;
        el._pad2 = 0;
        el.bbox = bbox;
        clip_out[global_id.x] = el;
    }
}
