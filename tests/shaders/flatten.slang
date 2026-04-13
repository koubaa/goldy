// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Flatten curves to lines. Ported from original Vello flatten shader.
// Uses goldy_dyn_byte_address for path_bboxes (InterlockedMin/Max require RWByteAddressBuffer).

import goldy_exp;
import ekrano_shared;

// Local structs (flatten-specific)
struct CubicParams {
    float th0;
    float th1;
    float chord_len;
    float err;
};

struct EulerParams {
    float th0;
    float k0;
    float k1;
    float ch;
};

struct EulerSeg {
    float2 p0;
    float2 p1;
    EulerParams params;
};

struct PointDeriv {
    float2 point;
    float2 deriv;
};

struct PathTagData {
    uint tag_byte;
    TagMonoid monoid;
};

struct CubicPoints {
    float2 p0;
    float2 p1;
    float2 p2;
    float2 p3;
};

struct NeighboringSegment {
    bool do_join;
    float2 tangent;
};

// Constants
static const float D = 0.67;
static const float B = 0.39;
static const float DERIV_THRESH = 1e-6;
static const float DERIV_THRESH_SQUARED = DERIV_THRESH * DERIV_THRESH;
static const float DERIV_EPS = 1e-6;
static const float SUBDIV_LIMIT = 1.0 / 65536.0;
static const float K1_THRESH = 1e-3;
static const float DIST_THRESH = 1e-3;
static const float TANGENT_THRESH = 1e-6;
static const uint ESPC_ROBUST_NORMAL = 0;
static const uint ESPC_ROBUST_LOW_K1 = 1;
static const uint ESPC_ROBUST_LOW_DIST = 2;
static const float BREAK1 = 0.8;
static const float BREAK2 = 1.25;
static const float BREAK3 = 2.1;
static const float SIN_SCALE = 1.0976991822760038;
static const float QUAD_A1 = 0.6406;
static const float QUAD_B1 = -0.81;
static const float QUAD_C1 = 0.9148117935952064;
static const float QUAD_A2 = 0.5;
static const float QUAD_B2 = -0.156;
static const float QUAD_C2 = 0.16145779359520596;
static const float QUAD_W1 = 0.5 * QUAD_B1 / QUAD_A1;
static const float QUAD_V1 = 1.0 / QUAD_A1;
static const float QUAD_U1 = QUAD_W1 * QUAD_W1 - QUAD_C1 / QUAD_A1;
static const float QUAD_W2 = 0.5 * QUAD_B2 / QUAD_A2;
static const float QUAD_V2 = 1.0 / QUAD_A2;
static const float QUAD_U2 = QUAD_W2 * QUAD_W2 - QUAD_C2 / QUAD_A2;
static const float FRAC_PI_4 = 0.7853981633974483;
static const float CBRT_9_8 = 1.040041911525952;
static const uint PATH_BBOX_STRIDE = 6;  // x0,y0,x1,y1,draw_flags,trans_ix per path
static const uint FLATTEN_ARC_MAX_LINES = 512;

// Context passed to helpers (buffers + pathdata_base)
struct FlattenCtx {
    Config config;
    ReadOnlyBuffer<uint> scene;
    ReadOnlyBuffer<TagMonoid> tag_monoids;
    ByteAddressView path_bboxes;
    StorageBuffer<BumpAllocators> bump;
    StorageBuffer<LineSoup> lines;
    uint pathdata_base;
};

float approx_parabola_integral(float x) {
    return x * rsqrt(sqrt(1.0 - D + (D * D * D * D + 0.25 * x * x)));
}

float approx_parabola_inv_integral(float x) {
    return x * sqrt(1.0 - B + (B * B + 0.5 * x * x));
}

CubicParams cubic_from_points_derivs(float2 p0, float2 p1, float2 q0, float2 q1, float dt) {
    float2 chord = p1 - p0;
    float chord_squared = dot(chord, chord);
    float chord_len = sqrt(chord_squared);
    CubicParams r;
    if (chord_squared < DERIV_THRESH_SQUARED) {
        float chord_err = sqrt((9.0 / 32.0) * (dot(q0, q0) + dot(q1, q1))) * dt;
        r.th0 = 0.0; r.th1 = 0.0; r.chord_len = DERIV_THRESH; r.err = chord_err;
        return r;
    }
    float scale = dt / chord_squared;
    float2 h0 = float2(q0.x * chord.x + q0.y * chord.y, q0.y * chord.x - q0.x * chord.y);
    float th0 = atan2(h0.y, h0.x);
    float d0 = length(h0) * scale;
    float2 h1 = float2(q1.x * chord.x + q1.y * chord.y, q1.x * chord.y - q1.y * chord.x);
    float th1 = atan2(h1.y, h1.x);
    float d1 = length(h1) * scale;

    float err = 2.0;
    float cth0 = cos(th0);
    float cth1 = cos(th1);
    if (cth0 * cth1 >= 0.0) {
        float e0 = (2.0 / 3.0) / max(1.0 + cth0, 1e-9);
        float e1 = (2.0 / 3.0) / max(1.0 + cth1, 1e-9);
        float s0 = sin(th0);
        float s1 = sin(th1);
        float s01 = cth0 * s1 + cth1 * s0;
        float amin = 0.15 * (2.0 * e0 * s0 + 2.0 * e1 * s1 - e0 * e1 * s01);
        float a = 0.15 * (2.0 * d0 * s0 + 2.0 * d1 * s1 - d0 * d1 * s01);
        float aerr = abs(a - amin);
        float symm = abs(th0 + th1);
        float asymm = abs(th0 - th1);
        float dist = length(float2(d0 - e0, d1 - e1));
        float symm2 = symm * symm;
        float ctr = (4.625e-6 * symm * symm2 + 7.5e-3 * asymm) * symm2;
        float halo = (5e-3 * symm + 7e-2 * asymm) * dist;
        err = ctr + 1.55 * aerr + halo;
    }
    err *= chord_len;
    r.th0 = th0; r.th1 = th1; r.chord_len = chord_len; r.err = err;
    return r;
}

EulerParams es_params_from_angles(float th0, float th1) {
    float k0 = th0 + th1;
    float dth = th1 - th0;
    float d2 = dth * dth;
    float k2 = k0 * k0;
    float a = 6.0;
    a -= d2 * (1.0 / 70.0);
    a -= (d2 * d2) * (1.0 / 10780.0);
    a += (d2 * d2 * d2) * 2.769178184818219e-07;
    float b = -0.1 + d2 * (1.0 / 4200.0) + d2 * d2 * 1.6959677820260655e-05;
    float c = -1.0 / 1400.0 + d2 * 6.84915970574303e-05 - k2 * 7.936475029053326e-06;
    a += (b + c * k2) * k2;
    float k1 = dth * a;

    float ch = 1.0;
    ch -= d2 * (1.0 / 40.0);
    ch += (d2 * d2) * 0.00034226190482569864;
    ch -= (d2 * d2 * d2) * 1.9349474568904524e-06;
    float b_ = -1.0 / 24.0 + d2 * 0.0024702380951963226 - d2 * d2 * 3.7297408997537985e-05;
    float c_ = 1.0 / 1920.0 - d2 * 4.87350869747975e-05 - k2 * 3.1001936068463107e-06;
    ch += (b_ + c_ * k2) * k2;

    EulerParams r;
    r.th0 = th0; r.k0 = k0; r.k1 = k1; r.ch = ch;
    return r;
}

float es_params_eval_th(EulerParams params, float t) {
    return (params.k0 + 0.5 * params.k1 * (t - 1.0)) * t - params.th0;
}

float2 integ_euler_10(float k0, float k1) {
    float t1_1 = k0;
    float t1_2 = 0.5 * k1;
    float t2_2 = t1_1 * t1_1;
    float t2_3 = 2.0 * (t1_1 * t1_2);
    float t2_4 = t1_2 * t1_2;
    float t3_4 = t2_2 * t1_2 + t2_3 * t1_1;
    float t3_6 = t2_4 * t1_2;
    float t4_4 = t2_2 * t2_2;
    float t4_5 = 2.0 * (t2_2 * t2_3);
    float t4_6 = 2.0 * (t2_2 * t2_4) + t2_3 * t2_3;
    float t4_7 = 2.0 * (t2_3 * t2_4);
    float t4_8 = t2_4 * t2_4;
    float t5_6 = t4_4 * t1_2 + t4_5 * t1_1;
    float t5_8 = t4_6 * t1_2 + t4_7 * t1_1;
    float t6_6 = t4_4 * t2_2;
    float t6_7 = t4_4 * t2_3 + t4_5 * t2_2;
    float t6_8 = t4_4 * t2_4 + t4_5 * t2_3 + t4_6 * t2_2;
    float t7_8 = t6_6 * t1_2 + t6_7 * t1_1;
    float t8_8 = t6_6 * t2_2;
    float u = 1.0;
    u -= (1.0 / 24.0) * t2_2 + (1.0 / 160.0) * t2_4;
    u += (1.0 / 1920.0) * t4_4 + (1.0 / 10752.0) * t4_6 + (1.0 / 55296.0) * t4_8;
    u -= (1.0 / 322560.0) * t6_6 + (1.0 / 1658880.0) * t6_8;
    u += (1.0 / 92897280.0) * t8_8;
    float v = (1.0 / 12.0) * t1_2;
    v -= (1.0 / 480.0) * t3_4 + (1.0 / 2688.0) * t3_6;
    v += (1.0 / 53760.0) * t5_6 + (1.0 / 276480.0) * t5_8;
    v -= (1.0 / 11612160.0) * t7_8;
    return float2(u, v);
}

float2 es_params_eval(EulerParams params, float t) {
    float thm = es_params_eval_th(params, t * 0.5);
    float2 uv = integ_euler_10((params.k0 + params.k1 * (0.5 * t - 0.5)) * t, params.k1 * t * t);
    float scale = t / params.ch;
    float s = scale * sin(thm);
    float c = scale * cos(thm);
    float x = uv.x * c - uv.y * s;
    float y = -uv.y * c - uv.x * s;
    return float2(x, y);
}

float2 es_params_eval_with_offset(EulerParams params, float t, float offset) {
    float th = es_params_eval_th(params, t);
    float2 v = offset * float2(sin(th), cos(th));
    return es_params_eval(params, t) + v;
}

EulerSeg es_seg_from_params(float2 p0, float2 p1, EulerParams params) {
    EulerSeg r;
    r.p0 = p0; r.p1 = p1; r.params = params;
    return r;
}

float2 es_seg_eval_with_offset(EulerSeg es, float t, float normalized_offset) {
    float2 chord = es.p1 - es.p0;
    float2 xy = es_params_eval_with_offset(es.params, t, normalized_offset);
    return es.p0 + float2(chord.x * xy.x - chord.y * xy.y, chord.x * xy.y + chord.y * xy.x);
}

float pow_1_5_signed(float x) {
    return x * sqrt(abs(x));
}

float espc_int_approx(float x) {
    float y = abs(x);
    float a;
    if (y < BREAK1) {
        a = sin(SIN_SCALE * y) * (1.0 / SIN_SCALE);
    } else if (y < BREAK2) {
        a = (sqrt(8.0) / 3.0) * pow_1_5_signed(y - 1.0) + FRAC_PI_4;
    } else {
        float3 abc = (y < BREAK3) ? float3(QUAD_A1, QUAD_B1, QUAD_C1) : float3(QUAD_A2, QUAD_B2, QUAD_C2);
        a = (abc.x * y + abc.y) * y + abc.z;
    }
    return a * sign(x);
}

float espc_int_inv_approx(float x) {
    float y = abs(x);
    float a;
    if (y < 0.7010707591262915) {
        a = asin(y * SIN_SCALE) * (1.0 / SIN_SCALE);
    } else if (y < 0.903249293595206) {
        float b = y - FRAC_PI_4;
        float u = pow(abs(b), 2.0 / 3.0) * sign(b);
        a = u * CBRT_9_8 + 1.0;
    } else {
        float3 uvw = (y < 2.038857793595206) ? float3(QUAD_U1, QUAD_V1, QUAD_W1) : float3(QUAD_U2, QUAD_V2, QUAD_W2);
        a = sqrt(uvw.x + uvw.y * y) - uvw.z;
    }
    return a * sign(x);
}

PointDeriv eval_cubic_and_deriv(float2 p0, float2 p1, float2 p2, float2 p3, float t) {
    float m = 1.0 - t;
    float mm = m * m;
    float mt = m * t;
    float tt = t * t;
    float2 p = p0 * (mm * m) + (p1 * (3.0 * mm) + p2 * (3.0 * mt) + p3 * tt) * t;
    float2 q = (p1 - p0) * mm + (p2 - p1) * (2.0 * mt) + (p3 - p2) * tt;
    PointDeriv r;
    r.point = p; r.deriv = q;
    return r;
}

float2 cubic_start_tangent(float2 p0, float2 p1, float2 p2, float2 p3) {
    float EPS = 1e-12;
    float2 d01 = p1 - p0;
    float2 d02 = p2 - p0;
    float2 d03 = p3 - p0;
    if (dot(d01, d01) > EPS) return d01;
    if (dot(d02, d02) > EPS) return d02;
    return d03;
}

float2 cubic_end_tangent(float2 p0, float2 p1, float2 p2, float2 p3) {
    float EPS = 1e-12;
    float2 d23 = p3 - p2;
    float2 d13 = p3 - p1;
    float2 d03 = p3 - p0;
    if (dot(d23, d23) > EPS) return d23;
    if (dot(d13, d13) > EPS) return d13;
    return d03;
}

float2 read_f32_point(FlattenCtx ctx, uint ix) {
    float x = asfloat(ctx.scene[ctx.pathdata_base + ix]);
    float y = asfloat(ctx.scene[ctx.pathdata_base + ix + 1]);
    return float2(x, y);
}

float2 read_i16_point(FlattenCtx ctx, uint ix) {
    uint raw = ctx.scene[ctx.pathdata_base + ix];
    float x = float(int(raw << 16) >> 16);
    float y = float(int(raw) >> 16);
    return float2(x, y);
}

int round_down(float x) { return int(floor(x)); }
int round_up(float x) { return int(ceil(x)); }

PathTagData compute_tag_monoid(FlattenCtx ctx, uint ix) {
    uint tag_word = ctx.scene[ctx.config.pathtag_base + (ix >> 2)];
    uint shift = (ix & 3) * 8;
    TagMonoid tm = reduce_tag(tag_word & ((1u << shift) - 1u));
    tm = ctx.tag_monoids[ix >> 2] + tm;
    uint tag_byte = (tag_word >> shift) & 0xff;
    tm.trans_ix -= 1;
    tm.style_ix -= STYLE_SIZE_IN_WORDS;
    PathTagData r;
    r.tag_byte = tag_byte; r.monoid = tm;
    return r;
}

CubicPoints read_path_segment(FlattenCtx ctx, PathTagData tag, bool is_stroke) {
    float2 p0, p1, p2, p3;
    uint seg_type = tag.tag_byte & PATH_TAG_SEG_TYPE;
    uint pathseg_offset = tag.monoid.pathseg_offset;
    bool is_stroke_cap_marker = is_stroke && (tag.tag_byte & PATH_TAG_SUBPATH_END) != 0;

    if ((tag.tag_byte & PATH_TAG_F32) != 0) {
        p0 = read_f32_point(ctx, pathseg_offset);
        p1 = read_f32_point(ctx, pathseg_offset + 2);
        if (seg_type >= PATH_TAG_QUADTO) {
            p2 = read_f32_point(ctx, pathseg_offset + 4);
            if (seg_type == PATH_TAG_CUBICTO)
                p3 = read_f32_point(ctx, pathseg_offset + 6);
        }
    } else {
        p0 = read_i16_point(ctx, pathseg_offset);
        p1 = read_i16_point(ctx, pathseg_offset + 1);
        if (seg_type >= PATH_TAG_QUADTO) {
            p2 = read_i16_point(ctx, pathseg_offset + 2);
            if (seg_type == PATH_TAG_CUBICTO)
                p3 = read_i16_point(ctx, pathseg_offset + 3);
        }
    }

    if (is_stroke_cap_marker && seg_type == PATH_TAG_QUADTO) {
        p0 = p1;
        p1 = p2;
        seg_type = PATH_TAG_LINETO;
    }

    if (seg_type == PATH_TAG_LINETO) {
        p3 = p1;
        p2 = p3 + (1.0 / 3.0) * (p0 - p3);
        p1 = p0 + (1.0 / 3.0) * (p3 - p0);
    } else if (seg_type == PATH_TAG_QUADTO) {
        p3 = p2;
        p2 = p1 + (1.0 / 3.0) * (p2 - p1);
        p1 = p1 + (1.0 / 3.0) * (p0 - p1);
    }

    CubicPoints r;
    r.p0 = p0; r.p1 = p1; r.p2 = p2; r.p3 = p3;
    return r;
}

bool line_coords_finite(float2 p0, float2 p1) {
    return !(isnan(p0.x) || isnan(p0.y) || isnan(p1.x) || isnan(p1.y));
}

void write_line(FlattenCtx ctx, uint line_ix, uint path_ix, float2 p0, float2 p1, inout float4 bbox) {
    if (!line_coords_finite(p0, p1))
        return;
    bbox = float4(min(bbox.xy, min(p0, p1)), max(bbox.zw, max(p0, p1)));
    if (line_ix < ctx.config.lines_size) {
        LineSoup ls;
        ls.path_ix = path_ix;
        ls._pad = 0;
        ls.p0 = p0;
        ls.p1 = p1;
        ctx.lines[line_ix] = ls;
    }
}

void write_line_with_transform(FlattenCtx ctx, uint line_ix, uint path_ix, float2 p0, float2 p1, Transform t, inout float4 bbox) {
    float2 tp0 = t * p0;
    float2 tp1 = t * p1;
    write_line(ctx, line_ix, path_ix, tp0, tp1, bbox);
}

uint output_line(FlattenCtx ctx, uint path_ix, float2 p0, float2 p1, inout float4 bbox) {
    if (!line_coords_finite(p0, p1))
        return 0;
    uint line_ix = 0;
    InterlockedAdd(ctx.bump[0].lines, 1, line_ix);
    write_line(ctx, line_ix, path_ix, p0, p1, bbox);
    return line_ix;
}

void output_line_with_transform(FlattenCtx ctx, uint path_ix, float2 p0, float2 p1, Transform transform, inout float4 bbox) {
    float2 tp0 = transform * p0;
    float2 tp1 = transform * p1;
    // TEMP: count NaN lines in the tile field of bump allocator
    if (!line_coords_finite(tp0, tp1)) {
        InterlockedAdd(ctx.bump[0].tile, 1);
        return;
    }
    uint line_ix = 0;
    InterlockedAdd(ctx.bump[0].lines, 1, line_ix);
    write_line(ctx, line_ix, path_ix, tp0, tp1, bbox);
}

void output_two_lines_with_transform(FlattenCtx ctx, uint path_ix, float2 p00, float2 p01, float2 p10, float2 p11, Transform transform, inout float4 bbox) {
    float2 t00 = transform * p00;
    float2 t01 = transform * p01;
    float2 t10 = transform * p10;
    float2 t11 = transform * p11;
    if (!line_coords_finite(t00, t01) || !line_coords_finite(t10, t11))
        return;
    uint line_ix = 0;
    InterlockedAdd(ctx.bump[0].lines, 2, line_ix);
    write_line(ctx, line_ix, path_ix, t00, t01, bbox);
    write_line(ctx, line_ix + 1, path_ix, t10, t11, bbox);
}

NeighboringSegment read_neighboring_segment(FlattenCtx ctx, uint ix) {
    PathTagData tag = compute_tag_monoid(ctx, ix);
    CubicPoints pts = read_path_segment(ctx, tag, true);
    bool is_closed = (tag.tag_byte & PATH_TAG_SEG_TYPE) == PATH_TAG_LINETO;
    bool is_stroke_cap_marker = (tag.tag_byte & PATH_TAG_SUBPATH_END) != 0;
    bool do_join = !is_stroke_cap_marker || is_closed;
    float2 tangent = pts.p3 - pts.p0;
    if (!is_stroke_cap_marker)
        tangent = cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
    NeighboringSegment r;
    r.do_join = do_join;
    r.tangent = tangent;
    return r;
}

void flatten_euler(FlattenCtx ctx, CubicPoints cubic, uint path_ix, Transform local_to_device, float offset, float2 start_p, float2 end_p, inout float4 bbox) {
    float2 p0, p1, p2, p3;
    float scale;
    Transform transform;
    float2 t_start = start_p;
    float2 t_end = end_p;

    if (offset == 0.0) {
        p0 = local_to_device * cubic.p0;
        p1 = local_to_device * cubic.p1;
        p2 = local_to_device * cubic.p2;
        p3 = local_to_device * cubic.p3;
        scale = 1.0;
        transform = transform_identity();
        t_start = p0;
        t_end = p3;
    } else {
        p0 = cubic.p0; p1 = cubic.p1; p2 = cubic.p2; p3 = cubic.p3;
        transform = local_to_device;
        float4 mat = transform.matrx;
        scale = 0.5 * (length(float2(mat.x + mat.w, mat.y - mat.z)) + length(float2(mat.x - mat.w, mat.y + mat.z)));
    }

    if ((p0.x == p1.x && p0.y == p1.y) && (p0.x == p2.x && p0.y == p2.y) && (p0.x == p3.x && p0.y == p3.y))
        return;

    float tol = 0.25;
    uint t0_u = 0;
    float dt = 1.0;
    float2 last_p = p0;
    float2 last_q = p1 - p0;
    if (dot(last_q, last_q) < DERIV_THRESH_SQUARED)
        last_q = eval_cubic_and_deriv(p0, p1, p2, p3, DERIV_EPS).deriv;
    float last_t = 0.0;
    float2 lp0 = t_start;

    while (true) {
        float t0 = float(t0_u) * dt;
        if (t0 == 1.0) break;

        float t1 = t0 + dt;
        float2 this_p0 = last_p;
        float2 this_q0 = last_q;
        PointDeriv this_pq1 = eval_cubic_and_deriv(p0, p1, p2, p3, t1);
        if (dot(this_pq1.deriv, this_pq1.deriv) < DERIV_THRESH_SQUARED) {
            PointDeriv new_pq1 = eval_cubic_and_deriv(p0, p1, p2, p3, t1 - DERIV_EPS);
            this_pq1.deriv = new_pq1.deriv;
            if (t1 < 1.0) {
                this_pq1.point = new_pq1.point;
                t1 = t1 - DERIV_EPS;
            }
        }
        float actual_dt = t1 - last_t;
        CubicParams cubic_params = cubic_from_points_derivs(this_p0, this_pq1.point, this_q0, this_pq1.deriv, actual_dt);

        if (cubic_params.err * scale <= tol || dt <= SUBDIV_LIMIT) {
            EulerParams euler_params = es_params_from_angles(cubic_params.th0, cubic_params.th1);
            EulerSeg es = es_seg_from_params(this_p0, this_pq1.point, euler_params);
            float k0 = es.params.k0 - 0.5 * es.params.k1;
            float k1 = es.params.k1;
            float normalized_offset = offset / cubic_params.chord_len;
            // When the chord is negligible relative to the offset, the ESPC math
            // produces NaN. Emit a single straight-line segment instead.
            if (abs(normalized_offset) > 1e3) {
                float2 lp1 = (t1 == 1.0) ? t_end : es_seg_eval_with_offset(es, 1.0, 0.0);
                float2 l0 = (offset >= 0.0) ? lp0 : lp1;
                float2 l1 = (offset >= 0.0) ? lp1 : lp0;
                output_line_with_transform(ctx, path_ix, l0, l1, transform, bbox);
                lp0 = lp1;
                last_p = this_pq1.point;
                last_q = this_pq1.deriv;
                last_t = t1;
                t0_u += 1;
                uint shift = firstbitlow(t0_u);
                t0_u >>= shift;
                dt *= float(1u << shift);
                continue;
            }
            float dist_scaled = normalized_offset * es.params.ch;
            float scale_multiplier = sqrt(0.125 * scale * cubic_params.chord_len / (es.params.ch * tol));
            float a = 0.0, b = 0.0, integral = 0.0, int0 = 0.0, n_frac = 0.0;
            uint robust = ESPC_ROBUST_NORMAL;

            if (abs(k1) < K1_THRESH) {
                float k = es.params.k0;
                n_frac = sqrt(abs(k * (k * dist_scaled + 1.0)));
                robust = ESPC_ROBUST_LOW_K1;
            } else if (abs(dist_scaled) < DIST_THRESH) {
                a = k1; b = k0;
                int0 = pow_1_5_signed(b);
                integral = pow_1_5_signed(a + b) - int0;
                n_frac = (2.0 / 3.0) * integral / a;
                robust = ESPC_ROBUST_LOW_DIST;
            } else {
                a = -2.0 * dist_scaled * k1;
                b = -1.0 - 2.0 * dist_scaled * k0;
                int0 = espc_int_approx(b);
                integral = espc_int_approx(a + b) - int0;
                float k_peak = k0 - k1 * b / a;
                float integrand_peak = sqrt(abs(k_peak * (k_peak * dist_scaled + 1.0)));
                n_frac = integral * integrand_peak / a;
            }

            float n = clamp(ceil(n_frac * scale_multiplier), 1.0, 100.0);
            for (uint i = 0; i < uint(n); i++) {
                float2 lp1;
                if (i + 1 == uint(n) && t1 == 1.0) {
                    lp1 = t_end;
                } else {
                    float t = float(i + 1) / n;
                    float s = t;
                    if (robust != ESPC_ROBUST_LOW_K1) {
                        float u = integral * t + int0;
                        float inv = (robust == ESPC_ROBUST_LOW_DIST) ? (pow(abs(u), 2.0 / 3.0) * sign(u)) : espc_int_inv_approx(u);
                        s = (inv - b) / a;
                    }
                    lp1 = es_seg_eval_with_offset(es, s, normalized_offset);
                }
                float2 l0 = (offset >= 0.0) ? lp0 : lp1;
                float2 l1 = (offset >= 0.0) ? lp1 : lp0;
                output_line_with_transform(ctx, path_ix, l0, l1, transform, bbox);
                lp0 = lp1;
            }
            last_p = this_pq1.point;
            last_q = this_pq1.deriv;
            last_t = t1;
            t0_u += 1;
            uint shift = firstbitlow(t0_u);
            t0_u >>= shift;
            dt *= float(1u << shift);
        } else {
            t0_u = t0_u * 2;
            dt *= 0.5;
        }
    }
}

void flatten_arc(FlattenCtx ctx, uint path_ix, float2 begin, float2 end, float2 center, float angle, Transform transform, inout float4 bbox) {
    float2 p0 = transform * begin;
    float2 r = begin - center;
    float MIN_THETA = 0.0001;
    float tol = 0.25;
    float radius = max(tol, length(p0 - transform * center));
    float theta = max(MIN_THETA, 2.0 * acos(1.0 - tol / radius));
    uint n_lines = max(1u, uint(ceil(angle / theta)));
    n_lines = min(n_lines, FLATTEN_ARC_MAX_LINES);

    float c = cos(theta);
    float s = sin(theta);
    float2 r_next;
    r_next.x = r.x * c + r.y * s;
    r_next.y = r.y * c - r.x * s;

    float2 p_last = transform * end;
    if (!line_coords_finite(p0, p_last))
        return;

    uint line_ix = 0;
    InterlockedAdd(ctx.bump[0].lines, n_lines, line_ix);

    for (uint i = 0; i < n_lines - 1; i++) {
        r = r_next;
        r_next.x = r.x * c + r.y * s;
        r_next.y = r.y * c - r.x * s;
        float2 p1 = transform * (center + r);
        write_line(ctx, line_ix + i, path_ix, p0, p1, bbox);
        p0 = p1;
    }
    float2 p1 = transform * end;
    write_line(ctx, line_ix + n_lines - 1, path_ix, p0, p1, bbox);
}

void draw_cap(FlattenCtx ctx, uint path_ix, uint cap_style, float2 point, float2 cap0, float2 cap1, float2 offset_tangent, Transform transform, inout float4 bbox) {
    if (cap_style == STYLE_FLAGS_CAP_ROUND) {
        flatten_arc(ctx, path_ix, cap0, cap1, point, 3.1415927, transform, bbox);
        return;
    }
    float2 start = cap0;
    float2 end = cap1;
    bool is_square = (cap_style == STYLE_FLAGS_CAP_SQUARE);
    if (is_square) {
        float2 v = offset_tangent;
        float2 p0 = start + v;
        float2 p1 = end + v;
        float2 ts0 = transform * start;
        float2 ts1 = transform * p0;
        float2 te0 = transform * p1;
        float2 te1 = transform * end;
        float2 tm0 = transform * p0;
        float2 tm1 = transform * p1;
        if (!line_coords_finite(ts0, ts1) || !line_coords_finite(te0, te1) || !line_coords_finite(tm0, tm1))
            return;
        uint line_ix = 0;
        InterlockedAdd(ctx.bump[0].lines, 3u, line_ix);
        write_line(ctx, line_ix + 1, path_ix, ts0, ts1, bbox);
        write_line(ctx, line_ix + 2, path_ix, te0, te1, bbox);
        write_line(ctx, line_ix, path_ix, tm0, tm1, bbox);
        return;
    }
    output_line_with_transform(ctx, path_ix, start, end, transform, bbox);
}

void draw_join(FlattenCtx ctx, uint path_ix, uint style_flags, float2 p0, float2 tan_prev, float2 tan_next, float2 n_prev, float2 n_next, Transform transform, inout float4 bbox) {
    float2 front0 = p0 + n_prev;
    float2 front1 = p0 + n_next;
    float2 back0 = p0 - n_next;
    float2 back1 = p0 - n_prev;
    float cr = tan_prev.x * tan_next.y - tan_prev.y * tan_next.x;
    float d = dot(tan_prev, tan_next);

    uint join_bits = style_flags & STYLE_FLAGS_JOIN_MASK;
    if (join_bits == STYLE_FLAGS_JOIN_BEVEL) {
        output_two_lines_with_transform(ctx, path_ix, front0, front1, back0, back1, transform, bbox);
    } else if (join_bits == STYLE_FLAGS_JOIN_MITER) {
        float hypot_len = length(float2(cr, d));
        float miter_limit = f16tof32(style_flags & STYLE_MITER_LIMIT_MASK);
        bool did_miter_inner = false;
        if (2.0 * hypot_len < (hypot_len + d) * miter_limit * miter_limit && abs(cr) > TANGENT_THRESH * TANGENT_THRESH) {
            bool is_backside = cr > 0.0;
            float2 fp_last = is_backside ? back1 : front0;
            float2 fp_this = is_backside ? back0 : front1;
            float2 p = is_backside ? back0 : front0;
            float2 v = fp_this - fp_last;
            float h = (tan_prev.x * v.y - tan_prev.y * v.x) / cr;
            float2 miter_pt = fp_this - tan_next * h;
            float2 f0 = front0;
            float2 f1 = front1;
            float2 b0 = back0;
            float2 b1 = back1;
            if (is_backside)
                b0 = miter_pt;
            else
                f0 = miter_pt;
            float2 tp0 = transform * p;
            float2 tp1 = transform * miter_pt;
            float2 tf0 = transform * f0;
            float2 tf1 = transform * f1;
            float2 tb0 = transform * b0;
            float2 tb1 = transform * b1;
            if (line_coords_finite(tp0, tp1) && line_coords_finite(tf0, tf1) && line_coords_finite(tb0, tb1)) {
                uint tmp = 0;
                InterlockedAdd(ctx.bump[0].lines, 3, tmp);
                write_line(ctx, tmp, path_ix, tp0, tp1, bbox);
                write_line(ctx, tmp + 1, path_ix, tf0, tf1, bbox);
                write_line(ctx, tmp + 2, path_ix, tb0, tb1, bbox);
                did_miter_inner = true;
            }
        }
        if (!did_miter_inner) {
            float2 tf0 = transform * front0;
            float2 tf1 = transform * front1;
            float2 tb0 = transform * back0;
            float2 tb1 = transform * back1;
            if (!line_coords_finite(tf0, tf1) || !line_coords_finite(tb0, tb1))
                return;
            uint line_ix = 0;
            InterlockedAdd(ctx.bump[0].lines, 2, line_ix);
            write_line(ctx, line_ix, path_ix, tf0, tf1, bbox);
            write_line(ctx, line_ix + 1, path_ix, tb0, tb1, bbox);
        }
    } else if (join_bits == STYLE_FLAGS_JOIN_ROUND) {
        float2 arc0, arc1, other0, other1;
        if (cr > 0.0) {
            arc0 = back0; arc1 = back1; other0 = front0; other1 = front1;
        } else {
            arc0 = front0; arc1 = front1; other0 = back0; other1 = back1;
        }
        flatten_arc(ctx, path_ix, arc0, arc1, p0, abs(atan2(cr, d)), transform, bbox);
        output_line_with_transform(ctx, path_ix, other0, other1, transform, bbox);
    }
}

// ---- GPU-side dashing helpers ----

// Compute cumulative arc length from the start of the current subpath to
// the beginning of segment at index `ix`. Uses a backward walk to find the
// subpath start, then a forward walk to sum chord lengths.
float compute_subpath_arc(FlattenCtx ctx, uint ix, uint my_path_ix) {
    // Walk backward to find subpath start. Stop at:
    //  - PATH_TAG_PATH markers (draw object boundary)
    //  - Tags with SUBPATH_END set (end of previous subpath)
    uint start_ix = ix;
    const uint MAX_WALK = 512;
    for (uint i = 1; i <= MAX_WALK && ix >= i; i++) {
        uint prev_ix = ix - i;
        uint tw = ctx.scene[ctx.config.pathtag_base + (prev_ix >> 2)];
        uint sh = (prev_ix & 3) * 8;
        uint tb = (tw >> sh) & 0xFF;
        if ((tb & PATH_TAG_PATH) != 0) break;
        if ((tb & PATH_TAG_SUBPATH_END) != 0) break;
        start_ix = prev_ix;
    }

    // Forward walk: accumulate chord lengths of real (non-cap) segments before ix.
    float arc = 0.0;
    for (uint j = start_ix; j < ix; j++) {
        PathTagData ptag = compute_tag_monoid(ctx, j);
        uint seg = ptag.tag_byte & PATH_TAG_SEG_TYPE;
        if (seg == 0) continue;
        if ((ptag.tag_byte & PATH_TAG_SUBPATH_END) != 0) continue;
        CubicPoints p = read_path_segment(ctx, ptag, true);
        arc += distance(p.p0, p.p3);
    }
    return arc;
}

// Emit a single stroke dash sub-segment using the same flatten_euler path
// as normal strokes, ensuring identical rendering across backends.
void emit_dash_stroke(
    FlattenCtx ctx,
    uint path_ix,
    uint style_flags,
    float2 sp0,
    float2 sp3,
    float offset,
    Transform transform,
    inout float4 bbox
) {
    float2 tangent = sp3 - sp0;
    float tang_len = length(tangent);
    if (tang_len < 1e-12) return;
    float2 tn = tangent / tang_len;
    float2 offset_tangent = offset * tn;
    float2 n = offset_tangent.yx * float2(-1.0, 1.0);

    uint start_cap = (style_flags & STYLE_FLAGS_START_CAP_MASK) >> 2;
    uint end_cap = (style_flags & STYLE_FLAGS_END_CAP_MASK);
    draw_cap(ctx, path_ix, start_cap, sp0, sp0 - n, sp0 + n, -offset_tangent, transform, bbox);
    draw_cap(ctx, path_ix, end_cap, sp3, sp3 + n, sp3 - n, offset_tangent, transform, bbox);

    output_line_with_transform(ctx, path_ix, sp0 + n, sp3 + n, transform, bbox);
    output_line_with_transform(ctx, path_ix, sp3 - n, sp0 - n, transform, bbox);
}

// Process one segment of a dashed stroke: find visible intervals and emit them.
void process_dashed_segment(
    FlattenCtx ctx,
    uint path_ix,
    uint style_flags,
    CubicPoints pts,
    float offset,
    Transform transform,
    float cum_arc,
    float dash_offset_val,
    float dash_on,
    float dash_off,
    inout float4 bbox
) {
    float period = dash_on + dash_off;
    if (period < 1e-9) return;
    float seg_len = distance(pts.p0, pts.p3);
    if (seg_len < 1e-12) return;

    float seg_start = cum_arc;
    float seg_end = seg_start + seg_len;
    float pos = seg_start;

    const uint MAX_INTERVALS = 32;
    float snap_eps = period * 1e-5;
    for (uint di = 0; di < MAX_INTERVALS && pos < seg_end - 1e-6; di++) {
        float adjusted = pos + dash_offset_val;
        float in_period = fmod(adjusted, period);
        if (in_period < 0.0) in_period += period;
        if (in_period < snap_eps || period - in_period < snap_eps) in_period = 0.0;
        if (abs(in_period - dash_on) < snap_eps) in_period = dash_on;
        bool visible = in_period < dash_on;
        float to_next = visible ? (dash_on - in_period) : (period - in_period);
        to_next = max(to_next, 1e-6);
        float interval_end = min(pos + to_next, seg_end);

        if (visible) {
            float t0 = (pos - seg_start) / seg_len;
            float t1 = (interval_end - seg_start) / seg_len;
            t0 = clamp(t0, 0.0, 1.0);
            t1 = clamp(t1, 0.0, 1.0);
            float2 sp0 = lerp(pts.p0, pts.p3, t0);
            float2 sp3 = lerp(pts.p0, pts.p3, t1);
            emit_dash_stroke(ctx, path_ix, style_flags, sp0, sp3, offset, transform, bbox);
        }

        pos = interval_end;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID) {
    Config config = goldy_dyn_buf_ro<Config>(0)[0];
    ReadOnlyBuffer<uint> scene = goldy_dyn_buf_ro<uint>(1);
    ReadOnlyBuffer<TagMonoid> tag_monoids = goldy_dyn_buf_ro<TagMonoid>(2);
    ByteAddressView path_bboxes = goldy_dyn_byte_address(3);
    StorageBuffer<BumpAllocators> bump = goldy_dyn_scattered<BumpAllocators>(4);
    StorageBuffer<LineSoup> lines = goldy_dyn_scattered<LineSoup>(5);

    FlattenCtx ctx;
    ctx.config = config;
    ctx.scene = scene;
    ctx.tag_monoids = tag_monoids;
    ctx.path_bboxes = path_bboxes;
    ctx.bump = bump;
    ctx.lines = lines;
    ctx.pathdata_base = config.pathdata_base;

    uint ix = global_id.x + config.flatten_thread_base;
    float4 bbox = Bbox2D.empty().v;

    PathTagData tag = compute_tag_monoid(ctx, ix);
    uint path_ix = tag.monoid.path_ix;
    uint style_ix = tag.monoid.style_ix;
    uint trans_ix = tag.monoid.trans_ix;

    uint style_flags = scene[config.style_base + style_ix];
    uint draw_flags = ((style_flags & STYLE_FLAGS_FILL) == 0) ? 0u : DRAW_INFO_FLAGS_FILL_RULE_BIT;
    if ((tag.tag_byte & PATH_TAG_PATH) != 0) {
        uint base = path_ix * PATH_BBOX_STRIDE * 4;
        path_bboxes.Store(base + 16, draw_flags);
        path_bboxes.Store(base + 20, trans_ix);
    }

    uint seg_type = tag.tag_byte & PATH_TAG_SEG_TYPE;
    if (seg_type != 0) {
        bool is_stroke = (style_flags & STYLE_FLAGS_STYLE) != 0;
        Transform transform = read_transform(ctx.scene, config.transform_base, trans_ix);
        CubicPoints pts = read_path_segment(ctx, tag, is_stroke);

        if (is_stroke) {
            float linewidth = asfloat(scene[config.style_base + style_ix + 1]);
            float offset = 0.5 * linewidth;
            bool is_open = (tag.tag_byte & PATH_TAG_SEG_TYPE) != PATH_TAG_LINETO;
            bool is_stroke_cap_marker = (tag.tag_byte & PATH_TAG_SUBPATH_END) != 0;
            bool is_dashed = (style_flags & STYLE_FLAGS_DASHED) != 0;

            if (is_dashed && !is_stroke_cap_marker) {
                float dash_offset_val = asfloat(scene[config.style_base + style_ix + 2]);
                uint packed_dash = scene[config.style_base + style_ix + 3];
                float dash_on = f16tof32(packed_dash & 0xFFFF);
                float dash_off = f16tof32(packed_dash >> 16);

                float cum_arc = compute_subpath_arc(ctx, ix, path_ix);
                process_dashed_segment(
                    ctx, path_ix, style_flags, pts, offset, transform,
                    cum_arc, dash_offset_val, dash_on, dash_off, bbox
                );
            } else if (is_stroke_cap_marker) {
                if (is_open) {
                    if (is_dashed) {
                        // Dashed strokes: each dash already has its own caps
                        // from emit_dash_stroke, so no subpath-level cap needed.
                    } else {
                        float2 tangent = pts.p3 - pts.p0;
                        float2 offset_tangent = offset * normalize(tangent);
                        float2 n = offset_tangent.yx * float2(-1.0, 1.0);
                        draw_cap(ctx, path_ix, (style_flags & STYLE_FLAGS_START_CAP_MASK) >> 2, pts.p0, pts.p0 - n, pts.p0 + n, -offset_tangent, transform, bbox);
                    }
                }
            } else {
                NeighboringSegment neighbor = read_neighboring_segment(ctx, ix + 1);
                float2 tan_start = cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
                if (dot(tan_start, tan_start) < TANGENT_THRESH * TANGENT_THRESH) tan_start = float2(TANGENT_THRESH, 0.0);
                float2 tan_prev = cubic_end_tangent(pts.p0, pts.p1, pts.p2, pts.p3);
                if (dot(tan_prev, tan_prev) < TANGENT_THRESH * TANGENT_THRESH) tan_prev = float2(TANGENT_THRESH, 0.0);
                float2 tan_next = neighbor.tangent;
                if (dot(tan_next, tan_next) < TANGENT_THRESH * TANGENT_THRESH) tan_next = float2(TANGENT_THRESH, 0.0);
                float2 n_start = offset * normalize(float2(-tan_start.y, tan_start.x));
                float2 offset_tangent = offset * normalize(tan_prev);
                float2 n_prev = offset_tangent.yx * float2(-1.0, 1.0);
                float2 n_next = offset * normalize(tan_next).yx * float2(-1.0, 1.0);

                flatten_euler(ctx, pts, path_ix, transform, offset, pts.p0 + n_start, pts.p3 + n_prev, bbox);
                flatten_euler(ctx, pts, path_ix, transform, -offset, pts.p0 - n_start, pts.p3 - n_prev, bbox);

                if (neighbor.do_join) {
                    draw_join(ctx, path_ix, style_flags, pts.p3, tan_prev, tan_next, n_prev, n_next, transform, bbox);
                } else {
                    draw_cap(ctx, path_ix, (style_flags & STYLE_FLAGS_END_CAP_MASK), pts.p3, pts.p3 + n_prev, pts.p3 - n_prev, offset_tangent, transform, bbox);
                }
            }
        } else {
            flatten_euler(ctx, pts, path_ix, transform, 0.0, pts.p0, pts.p3, bbox);
        }

        if (bbox.z > bbox.x || bbox.w > bbox.y) {
            uint base = path_ix * PATH_BBOX_STRIDE * 4;
            signed_atomic_min(path_bboxes, base + 0, round_down(bbox.x));
            signed_atomic_min(path_bboxes, base + 4, round_down(bbox.y));
            signed_atomic_max(path_bboxes, base + 8, round_up(bbox.z));
            signed_atomic_max(path_bboxes, base + 12, round_up(bbox.w));
        }
    }
}
