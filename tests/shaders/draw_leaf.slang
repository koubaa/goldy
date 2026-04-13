// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Finish prefix sum of drawtags, decode draw objects.

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;
static const uint LG_WG_SIZE = 8;

groupshared DrawMonoid sh_scratch[256];

// Slots: 0=config, 1=scene, 2=reduced, 3=path_bbox, 4=draw_monoid, 5=info, 6=clip_inp

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 local_id : SV_GroupThreadID, uint3 wg_id : SV_GroupID) {
    Config config = goldy_dyn_buf_ro<Config>(0)[0];
    ReadOnlyBuffer<uint> scene = goldy_dyn_buf_ro<uint>(1);
    ReadOnlyBuffer<DrawMonoid> reduced = goldy_dyn_buf_ro<DrawMonoid>(2);
    ReadOnlyBuffer<PathBbox> path_bbox = goldy_dyn_buf_ro<PathBbox>(3);
    StorageBuffer<DrawMonoid> draw_monoid = goldy_dyn_scattered<DrawMonoid>(4);
    StorageBuffer<uint> info = goldy_dyn_scattered<uint>(5);
    StorageBuffer<ClipInp> clip_inp = goldy_dyn_scattered<ClipInp>(6);

    DrawMonoid agg = DrawMonoid.identity();
    if (local_id.x < wg_id.x) {
        agg = reduced[local_id.x];
    }
    sh_scratch[local_id.x] = agg;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x + (1u << i) < WG_SIZE)
            agg = agg.combine(sh_scratch[local_id.x + (1u << i)]);
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[local_id.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();
    DrawMonoid prefix = sh_scratch[0];

    uint num_blocks_total = (config.n_drawobj + WG_SIZE - 1) / WG_SIZE;
    uint n_blocks_base = num_blocks_total / WG_SIZE;
    uint remainder = num_blocks_total % WG_SIZE;
    uint first_block = n_blocks_base * wg_id.x + min(wg_id.x, remainder);
    uint n_blocks = n_blocks_base + (wg_id.x < remainder ? 1u : 0u);
    uint block_start = first_block * WG_SIZE;
    uint blocks_end = block_start + n_blocks * WG_SIZE;

    while (block_start != blocks_end) {
        uint ix = block_start + local_id.x;
        uint tag_word = read_draw_tag_from_scene(scene, config, ix);
        agg = DrawTagScanner.map(tag_word);
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[local_id.x] = agg;
        for (uint i = 0; i < LG_WG_SIZE; i++) {
            GroupMemoryBarrierWithGroupSync();
            if (local_id.x >= (1u << i)) {
                DrawMonoid other = sh_scratch[local_id.x - (1u << i)];
                agg = other.combine(agg);
            }
            GroupMemoryBarrierWithGroupSync();
            sh_scratch[local_id.x] = agg;
        }
        DrawMonoid m = prefix;
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x > 0) {
            m = m + sh_scratch[local_id.x - 1];
        }
        if (ix < config.n_drawobj) {
            draw_monoid[ix] = m;
        }
        uint dd = config.drawdata_base + m.scene_offset;
        uint di = m.info_offset;

        if (tag_word == DRAWTAG_FILL_COLOR || tag_word == DRAWTAG_FILL_LIN_GRADIENT ||
            tag_word == DRAWTAG_FILL_RAD_GRADIENT || tag_word == DRAWTAG_FILL_SWEEP_GRADIENT ||
            tag_word == DRAWTAG_FILL_IMAGE || tag_word == DRAWTAG_BEGIN_CLIP || tag_word == DRAWTAG_BLURRED_ROUNDED_RECT) {
            PathBbox bbox = path_bbox[m.path_ix];
            Transform transform;
            transform.matrx = float4(0, 0, 0, 0);
            transform.translate = float2(0, 0);
            uint draw_flags = bbox.draw_flags;
            if (tag_word == DRAWTAG_FILL_LIN_GRADIENT || tag_word == DRAWTAG_FILL_RAD_GRADIENT ||
                tag_word == DRAWTAG_FILL_SWEEP_GRADIENT || tag_word == DRAWTAG_FILL_IMAGE ||
                tag_word == DRAWTAG_BLURRED_ROUNDED_RECT) {
                transform = read_transform(scene, config.transform_base, bbox.trans_ix);
            }

            if (tag_word == DRAWTAG_FILL_COLOR) {
                info[di] = draw_flags;
            } else if (tag_word == DRAWTAG_BEGIN_CLIP) {
                info[di] = draw_flags;
            } else if (tag_word == DRAWTAG_FILL_LIN_GRADIENT) {
                info[di] = draw_flags;
                float2 p0 = float2(asfloat(scene[dd + 1]), asfloat(scene[dd + 2]));
                float2 p1 = float2(asfloat(scene[dd + 3]), asfloat(scene[dd + 4]));
                p0 = transform * p0;
                p1 = transform * p1;
                float2 dxy = p1 - p0;
                float scale = 1.0 / dot(dxy, dxy);
                float2 line_xy = dxy * scale;
                float line_c = -dot(p0, line_xy);
                info[di + 1] = asuint(line_xy.x);
                info[di + 2] = asuint(line_xy.y);
                info[di + 3] = asuint(line_c);
            } else if (tag_word == DRAWTAG_FILL_RAD_GRADIENT) {
                static const float GRADIENT_EPSILON = 1.0 / float(1 << 12);
                info[di] = draw_flags;
                float2 p0 = float2(asfloat(scene[dd + 1]), asfloat(scene[dd + 2]));
                float2 p1 = float2(asfloat(scene[dd + 3]), asfloat(scene[dd + 4]));
                float r0 = asfloat(scene[dd + 5]);
                float r1 = asfloat(scene[dd + 6]);
                Transform user_to_gradient = transform_inverse(transform);
                Transform xform;
                xform.matrx = float4(0, 0, 0, 0);
                xform.translate = float2(0, 0);
                float focal_x = 0.0;
                float radius = 0.0;
                uint kind = 0;
                uint flags = 0;

                if (abs(r0 - r1) <= GRADIENT_EPSILON) {
                    kind = RAD_GRAD_KIND_STRIP;
                    float scaled = r0 / length(p1 - p0);
                    xform = two_point_to_unit_line(p0, p1) * user_to_gradient;
                    radius = scaled * scaled;
                } else {
                    kind = RAD_GRAD_KIND_CONE;
                    if (p0.x == p1.x && p0.y == p1.y) {
                        kind = RAD_GRAD_KIND_CIRCULAR;
                        p0 += float2(GRADIENT_EPSILON, GRADIENT_EPSILON);
                    }
                    if (r1 == 0.0) {
                        flags |= RAD_GRAD_SWAPPED;
                        float2 tmp_p = p0;
                        p0 = p1;
                        p1 = tmp_p;
                        float tmp_r = r0;
                        r0 = r1;
                        r1 = tmp_r;
                    }
                    focal_x = r0 / (r0 - r1);
                    float2 cf = (1.0 - focal_x) * p0 + focal_x * p1;
                    radius = r1 / length(p1 - cf);
                    Transform user_to_unit_line = two_point_to_unit_line(cf, p1) * user_to_gradient;
                    Transform user_to_scaled = user_to_unit_line;
                    if (abs(radius - 1.0) <= GRADIENT_EPSILON) {
                        kind = RAD_GRAD_KIND_FOCAL_ON_CIRCLE;
                        float scale = 0.5 * abs(1.0 - focal_x);
                        Transform scale_t;
                        scale_t.matrx = float4(scale, 0, 0, scale);
                        scale_t.translate = float2(0, 0);
                        user_to_scaled = scale_t * user_to_unit_line;
                    } else {
                        float a_val = radius * radius - 1.0;
                        float scale_ratio = abs(1.0 - focal_x) / a_val;
                        float scale_x = radius * scale_ratio;
                        float scale_y = sqrt(abs(a_val)) * scale_ratio;
                        Transform scale_t;
                        scale_t.matrx = float4(scale_x, 0, 0, scale_y);
                        scale_t.translate = float2(0, 0);
                        user_to_scaled = scale_t * user_to_unit_line;
                    }
                    xform = user_to_scaled;
                }
                info[di + 1] = asuint(xform.matrx.x);
                info[di + 2] = asuint(xform.matrx.y);
                info[di + 3] = asuint(xform.matrx.z);
                info[di + 4] = asuint(xform.matrx.w);
                info[di + 5] = asuint(xform.translate.x);
                info[di + 6] = asuint(xform.translate.y);
                info[di + 7] = asuint(focal_x);
                info[di + 8] = asuint(radius);
                info[di + 9] = (flags << 3) | kind;
            } else if (tag_word == DRAWTAG_FILL_SWEEP_GRADIENT) {
                info[di] = draw_flags;
                float2 p0 = float2(asfloat(scene[dd + 1]), asfloat(scene[dd + 2]));
                Transform ident;
                ident.matrx = float4(1, 0, 0, 1);
                ident.translate = p0;
                Transform xform = transform * ident;
                Transform inv = transform_inverse(xform);
                info[di + 1] = asuint(inv.matrx.x);
                info[di + 2] = asuint(inv.matrx.y);
                info[di + 3] = asuint(inv.matrx.z);
                info[di + 4] = asuint(inv.matrx.w);
                info[di + 5] = asuint(inv.translate.x);
                info[di + 6] = asuint(inv.translate.y);
                info[di + 7] = scene[dd + 3];
                info[di + 8] = scene[dd + 4];
            } else if (tag_word == DRAWTAG_FILL_IMAGE) {
                info[di] = draw_flags;
                Transform inv = transform_inverse(transform);
                info[di + 1] = asuint(inv.matrx.x);
                info[di + 2] = asuint(inv.matrx.y);
                info[di + 3] = asuint(inv.matrx.z);
                info[di + 4] = asuint(inv.matrx.w);
                info[di + 5] = asuint(inv.translate.x);
                info[di + 6] = asuint(inv.translate.y);
                info[di + 7] = scene[dd];
                info[di + 8] = scene[dd + 1];
                info[di + 9] = scene[dd + 2];
            } else if (tag_word == DRAWTAG_BLURRED_ROUNDED_RECT) {
                info[di] = draw_flags;
                Transform inv = transform_inverse(transform);
                info[di + 1] = asuint(inv.matrx.x);
                info[di + 2] = asuint(inv.matrx.y);
                info[di + 3] = asuint(inv.matrx.z);
                info[di + 4] = asuint(inv.matrx.w);
                info[di + 5] = asuint(inv.translate.x);
                info[di + 6] = asuint(inv.translate.y);
                info[di + 7] = scene[dd + 1];
                info[di + 8] = scene[dd + 2];
                info[di + 9] = scene[dd + 3];
                info[di + 10] = scene[dd + 4];
            }
        }

        if (tag_word == DRAWTAG_BEGIN_CLIP || tag_word == DRAWTAG_END_CLIP) {
            ClipInp clip_el;
            clip_el.ix = ix;
            clip_el.path_ix = (tag_word == DRAWTAG_BEGIN_CLIP) ? int(m.path_ix) : int(~ix);
            clip_inp[m.clip_ix] = clip_el;
        }

        block_start += WG_SIZE;
        prefix = prefix + sh_scratch[WG_SIZE - 1];
    }
}
