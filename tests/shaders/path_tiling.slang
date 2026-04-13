// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Write path segments.

import goldy_exp;
import ekrano_shared;

static const float EPSILON = 1e-6;

// Slots: 0=bump, 1=seg_counts, 2=lines, 3=paths, 4=tiles, 5=segments

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID) {
    StorageBuffer<BumpAllocators> bump = goldy_dyn_scattered<BumpAllocators>(0);
    ReadOnlyBuffer<SegmentCount> seg_counts = goldy_dyn_buf_ro<SegmentCount>(1);
    ReadOnlyBuffer<LineSoup> lines = goldy_dyn_buf_ro<LineSoup>(2);
    ReadOnlyBuffer<Path> paths = goldy_dyn_buf_ro<Path>(3);
    ReadOnlyBuffer<Tile> tiles = goldy_dyn_buf_ro<Tile>(4);
    StorageBuffer<Segment> segments = goldy_dyn_scattered<Segment>(5);

    uint n_segments = bump[0].seg_counts;
    if (global_id.x >= n_segments)
        return;

    SegmentCount seg_count = seg_counts[global_id.x];
    LineSoup line = lines[seg_count.line_ix];
    uint counts = seg_count.counts;
    uint seg_within_slice = counts >> 16;
    uint seg_within_line = counts & 0xffff;

    bool is_down = line.p1.y >= line.p0.y;
    float2 xy0 = is_down ? line.p0 : line.p1;
    float2 xy1 = is_down ? line.p1 : line.p0;
    float2 s0 = xy0 * TILE_SCALE;
    float2 s1 = xy1 * TILE_SCALE;

    TileDDA dda = TileDDA.from_segment(s0, s1);
    uint count = dda.count;
    float a = dda.a;
    float b = dda.b;
    float y0i = dda.y0;
    int x0i = int(dda.x0);
    float x_sign = dda.x_sign;
    bool is_positive_slope = dda.x_sign > 0.0;
    float z = floor(a * float(seg_within_line) + b);
    int x = x0i + int(x_sign * z);
    int y = int(y0i + float(seg_within_line) - z);

    Path path = paths[line.path_ix];
    int4 bbox = int4(asint(path.bbox.x), asint(path.bbox.y), asint(path.bbox.z), asint(path.bbox.w));
    int stride = bbox.z - bbox.x;
    int tile_ix = int(path.tiles) + (y - bbox.y) * stride + x - bbox.x;
    Tile tile = tiles[tile_ix];
    uint seg_start = ~tile.segment_count_or_ix;
    if (int(seg_start) < 0)
        return;

    float2 tile_xy = float2(float(x) * float(TILE_WIDTH), float(y) * float(TILE_HEIGHT));
    float2 tile_xy1 = tile_xy + float2(float(TILE_WIDTH), float(TILE_HEIGHT));

    if (seg_within_line > 0) {
        float z_prev = floor(a * (float(seg_within_line) - 1.0) + b);
        if (z == z_prev) {
            float xt = xy0.x + (xy1.x - xy0.x) * (tile_xy.y - xy0.y) / (xy1.y - xy0.y);
            xt = clamp(xt, tile_xy.x + 1e-3, tile_xy1.x);
            xy0 = float2(xt, tile_xy.y);
        } else {
            float x_clip = is_positive_slope ? tile_xy.x : tile_xy1.x;
            float yt = xy0.y + (xy1.y - xy0.y) * (x_clip - xy0.x) / (xy1.x - xy0.x);
            yt = clamp(yt, tile_xy.y + 1e-3, tile_xy1.y);
            xy0 = float2(x_clip, yt);
        }
    }
    if (seg_within_line < count - 1) {
        float z_next = floor(a * (float(seg_within_line) + 1.0) + b);
        if (z == z_next) {
            float xt = xy0.x + (xy1.x - xy0.x) * (tile_xy1.y - xy0.y) / (xy1.y - xy0.y);
            xt = clamp(xt, tile_xy.x + 1e-3, tile_xy1.x);
            xy1 = float2(xt, tile_xy1.y);
        } else {
            float x_clip = is_positive_slope ? tile_xy1.x : tile_xy.x;
            float yt = xy0.y + (xy1.y - xy0.y) * (x_clip - xy0.x) / (xy1.x - xy0.x);
            yt = clamp(yt, tile_xy.y + 1e-3, tile_xy1.y);
            xy1 = float2(x_clip, yt);
        }
    }
    float y_edge = 1e9;
    float2 p0 = xy0 - tile_xy;
    float2 p1 = xy1 - tile_xy;
    if (p0.x == 0.0) {
        if (p1.x == 0.0) {
            p0.x = EPSILON;
            if (p0.y == 0.0) {
                p1.x = EPSILON;
                p1.y = float(TILE_HEIGHT);
            } else {
                p1.x = 2.0 * EPSILON;
                p1.y = p0.y;
            }
        } else if (p0.y == 0.0) {
            p0.x = EPSILON;
        } else {
            y_edge = p0.y;
        }
    } else if (p1.x == 0.0) {
        if (p1.y == 0.0) {
            p1.x = EPSILON;
        } else {
            y_edge = p1.y;
        }
    }
    if (p0.x == floor(p0.x) && p0.x != 0.0)
        p0.x -= EPSILON;
    if (p1.x == floor(p1.x) && p1.x != 0.0)
        p1.x -= EPSILON;
    if (!is_down) {
        float2 tmp = p0;
        p0 = p1;
        p1 = tmp;
    }
    Segment seg;
    seg.point0 = p0;
    seg.point1 = p1;
    seg.y_edge = y_edge;
    seg._pad = 0;
    segments[seg_start + seg_within_slice] = seg;
}
