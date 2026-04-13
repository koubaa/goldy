// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Tile allocation (and zeroing of tiles)

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;
static const uint LG_WG_SIZE = 8;
static const float SX = 1.0 / float(TILE_WIDTH);
static const float SY = 1.0 / float(TILE_HEIGHT);

groupshared uint sh_tile_count[256];
groupshared uint sh_previous_failed;

// Slots: 0=config, 1=scene, 2=draw_bboxes, 3=bump, 4=paths, 5=tiles

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID) {
    Config config = goldy_dyn_buf_ro<Config>(0)[0];
    ReadOnlyBuffer<uint> scene = goldy_dyn_buf_ro<uint>(1);
    ReadOnlyBuffer<float4> draw_bboxes = goldy_dyn_buf_ro<float4>(2);
    StorageBuffer<BumpAllocators> bump = goldy_dyn_scattered<BumpAllocators>(3);
    StorageBuffer<Path> paths = goldy_dyn_scattered<Path>(4);
    StorageBuffer<Tile> tiles = goldy_dyn_scattered<Tile>(5);

    if (local_id.x == 0) {
        uint failed_val = bump[0].failed & (STAGE_BINNING | STAGE_FLATTEN);
        sh_previous_failed = (failed_val != 0) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint failed = sh_previous_failed;
    if (failed != 0)
        return;

    uint drawobj_ix = global_id.x;
    uint drawtag = DRAWTAG_NOP;
    if (drawobj_ix < config.n_drawobj) {
        drawtag = scene[config.drawtag_base + drawobj_ix];
    }
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (drawtag != DRAWTAG_NOP && drawtag != DRAWTAG_END_CLIP) {
        float4 bbox = draw_bboxes[drawobj_ix];
        if (bbox.x < bbox.z && bbox.y < bbox.w) {
            x0 = int(floor(bbox.x * SX));
            y0 = int(floor(bbox.y * SY));
            x1 = int(ceil(bbox.z * SX));
            y1 = int(ceil(bbox.w * SY));
        }
    }
    uint ux0 = uint(clamp(x0, 0, int(config.width_in_tiles)));
    uint uy0 = uint(clamp(y0, 0, int(config.height_in_tiles)));
    uint ux1 = uint(clamp(x1, 0, int(config.width_in_tiles)));
    uint uy1 = uint(clamp(y1, 0, int(config.height_in_tiles)));
    uint tile_count = (ux1 - ux0) * (uy1 - uy0);
    sh_tile_count[local_id.x] = tile_count;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x >= (1u << i)) {
            uint other = sh_tile_count[local_id.x - (1u << i)];
            tile_count = other + tile_count;
        }
        GroupMemoryBarrierWithGroupSync();
        sh_tile_count[local_id.x] = tile_count;
    }
    uint total_tile_count = tile_count;
    if (local_id.x == WG_SIZE - 1) {
        uint count = sh_tile_count[WG_SIZE - 1];
        uint offset; InterlockedAdd(bump[0].tile, count, offset);
        if (offset + count > config.tiles_size) {
            offset = 0;
            InterlockedOr(bump[0].failed, STAGE_TILE_ALLOC);
        }
        paths[drawobj_ix].tiles = offset;
    }
    AllMemoryBarrierWithGroupSync();
    uint tile_offset = paths[drawobj_ix | (WG_SIZE - 1)].tiles;
    AllMemoryBarrierWithGroupSync();

    if (drawobj_ix < config.n_drawobj) {
        uint tile_subix = (local_id.x > 0) ? sh_tile_count[local_id.x - 1] : 0;
        Path path;
        path.bbox = uint4(ux0, uy0, ux1, uy1);
        path.tiles = tile_offset + tile_subix;
        path._pad0 = 0;
        path._pad1 = 0;
        path._pad2 = 0;
        paths[drawobj_ix] = path;
    }

    uint total_count = sh_tile_count[WG_SIZE - 1];
    for (uint i = local_id.x; i < total_count; i += WG_SIZE) {
        Tile t;
        t.backdrop = 0;
        t.segment_count_or_ix = 0;
        tiles[tile_offset + i] = t;
    }
}
