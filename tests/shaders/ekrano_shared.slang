// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Shared type definitions for Ekrano Slang shaders.
// Import with: import ekrano_shared;
//
// Must be kept in sync with ekrano_encoding config and types.

module ekrano_shared;

import goldy_exp;

// === Config ===
// Sync with ConfigUniform in ekrano_encoding/src/config.rs
public struct Config {
    public uint width_in_tiles;

    public uint height_in_tiles;

    public uint target_width;

    public uint target_height;

    public uint base_color;

    public uint n_drawobj;

    public uint n_path;

    public uint n_clip;

    public uint bin_data_start;

    public uint pathtag_base;

    public uint pathdata_base;

    public uint drawtag_base;

    public uint drawdata_base;

    public uint transform_base;

    public uint style_base;

    public uint lines_size;

    public uint binning_size;

    public uint tiles_size;

    public uint seg_counts_size;

    public uint segments_size;

    public uint blend_size;

    public uint ptcl_size;

    public uint tile_y_offset;

    public uint flatten_thread_base;

};

// Tile geometry
public static const uint TILE_WIDTH = 16;
public static const uint TILE_HEIGHT = 16;
public static const uint N_TILE_X = 16;
public static const uint N_TILE_Y = 16;
public static const uint N_TILE = N_TILE_X * N_TILE_Y;
public static const float TILE_SCALE = 0.0625;
public static const uint BLEND_STACK_SPLIT = 4;

public static const float ONE_MINUS_ULP = 0.99999994;
public static const float ROBUST_EPSILON = 2e-7;

public uint span(float a, float b) {
    return uint(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

public struct TileDDA {
    public float a;
    public float b;
    public float x_sign;
    public float x0;
    public float y0;
    public uint count_x;
    public uint count;

    public static TileDDA from_segment(float2 s0, float2 s1) {
        TileDDA d;
        float dx = abs(s1.x - s0.x);
        float dy = s1.y - s0.y;
        float idxdy = 1.0 / (dx + dy);
        d.a = dx * idxdy;
        bool pos = s1.x >= s0.x;
        d.x_sign = pos ? 1.0 : -1.0;
        float xt0 = floor(s0.x * d.x_sign);
        float c = s0.x * d.x_sign - xt0;
        d.y0 = floor(s0.y);
        float ytop = (s0.y == s1.y) ? ceil(s0.y) : (d.y0 + 1.0);
        d.b = min((dy * c + dx * (ytop - s0.y)) * idxdy, ONE_MINUS_ULP);
        d.count_x = span(s0.x, s1.x) - 1;
        d.count = d.count_x + span(s0.y, s1.y);
        float err = floor(d.a * (float(d.count) - 1.0) + d.b) - float(d.count_x);
        if (err != 0.0) d.a -= ROBUST_EPSILON * sign(err);
        d.x0 = xt0 * d.x_sign + (pos ? 0.0 : -1.0);
        return d;
    }

    public int2 step(uint sub_ix) {
        float z = floor(a * float(sub_ix) + b);
        return int2(int(x0 + x_sign * z), int(y0 + float(sub_ix) - z));
    }
}

public struct Bbox2D {
    public float4 v;
    public static Bbox2D empty() { Bbox2D b; b.v = float4(1e31, 1e31, -1e31, -1e31); return b; }
    public static Bbox2D infinite() { Bbox2D b; b.v = float4(-1e9, -1e9, 1e9, 1e9); return b; }
    public static Bbox2D from_float4(float4 f) { Bbox2D b; b.v = f; return b; }
    public Bbox2D intersect(Bbox2D other) { Bbox2D b; b.v = float4(max(this.v.xy, other.v.xy), min(this.v.zw, other.v.zw)); return b; }
    public Bbox2D union_with(Bbox2D other) { Bbox2D b; b.v = float4(min(this.v.xy, other.v.xy), max(this.v.zw, other.v.zw)); return b; }
    public Bbox2D grow(float2 p) { Bbox2D b; b.v = float4(min(this.v.xy, p), max(this.v.zw, p)); return b; }
}

extension Bbox2D : IMonoid {
    static Bbox2D identity() { return Bbox2D.infinite(); }
    Bbox2D combine(Bbox2D other) { return this.intersect(other); }
}

// Radial gradient kinds
public static const uint RAD_GRAD_KIND_CIRCULAR = 1;
public static const uint RAD_GRAD_KIND_STRIP = 2;
public static const uint RAD_GRAD_KIND_FOCAL_ON_CIRCLE = 3;
public static const uint RAD_GRAD_KIND_CONE = 4;
public static const uint RAD_GRAD_SWAPPED = 1;

// === Pathtag ===
public struct TagMonoid {
    public uint trans_ix;

    public uint pathseg_ix;

    public uint pathseg_offset;

    public uint style_ix;

    public uint path_ix;

};

public static const uint PATH_TAG_SEG_TYPE = 3;
public static const uint PATH_TAG_LINETO = 1;
public static const uint PATH_TAG_QUADTO = 2;
public static const uint PATH_TAG_CUBICTO = 3;
public static const uint PATH_TAG_F32 = 8;
public static const uint PATH_TAG_TRANSFORM = 0x20;
public static const uint PATH_TAG_PATH = 0x10;
public static const uint PATH_TAG_SUBPATH_END = 4;
public static const uint PATH_TAG_STYLE = 0x40;
public static const uint STYLE_SIZE_IN_WORDS = 4;
public static const uint STYLE_FLAGS_STYLE = 0x80000000;
public static const uint STYLE_FLAGS_FILL = 0x40000000;
public static const uint STYLE_MITER_LIMIT_MASK = 0xFFFF;
public static const uint STYLE_FLAGS_START_CAP_MASK = 0x0C000000;
public static const uint STYLE_FLAGS_END_CAP_MASK = 0x03000000;
public static const uint STYLE_FLAGS_CAP_BUTT = 0;
public static const uint STYLE_FLAGS_CAP_SQUARE = 0x01000000;
public static const uint STYLE_FLAGS_CAP_ROUND = 0x02000000;
public static const uint STYLE_FLAGS_JOIN_MASK = 0x30000000;
public static const uint STYLE_FLAGS_JOIN_BEVEL = 0;
public static const uint STYLE_FLAGS_JOIN_MITER = 0x10000000;
public static const uint STYLE_FLAGS_JOIN_ROUND = 0x20000000;
public static const uint STYLE_FLAGS_DASHED = 0x00800000;

// === Drawtag ===
public struct DrawMonoid {
    public uint path_ix;

    public uint clip_ix;

    public uint scene_offset;

    public uint info_offset;

};

public static const uint DRAWTAG_NOP = 0;
public static const uint DRAWTAG_FILL_COLOR = 0x44;
public static const uint DRAWTAG_FILL_LIN_GRADIENT = 0x114;
public static const uint DRAWTAG_FILL_RAD_GRADIENT = 0x29c;
public static const uint DRAWTAG_FILL_SWEEP_GRADIENT = 0x254;
public static const uint DRAWTAG_FILL_IMAGE = 0x28C;
public static const uint DRAWTAG_BLURRED_ROUNDED_RECT = 0x2d4;
public static const uint DRAWTAG_BEGIN_CLIP = 0x49;
public static const uint DRAWTAG_END_CLIP = 0x21;
public static const uint DRAW_INFO_FLAGS_FILL_RULE_BIT = 1;

// === Tile ===
public struct Path {
    public uint4 bbox;

    public uint tiles;

    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
};

public struct Tile {
    public int backdrop;

    public uint segment_count_or_ix;

};

// Same layout as Tile, used with InterlockedAdd in path_count
public struct AtomicTile {
    public int backdrop;

    public uint segment_count_or_ix;

};

// For flatten stage - path bbox with atomic fields for InterlockedMin/Max
public struct AtomicPathBbox {
    public int x0;

    public int y0;

    public int x1;

    public int y1;

    public uint draw_flags;

    public uint trans_ix;

};

// === Segment ===
public struct Segment {
    public float2 point0;

    public float2 point1;

    public float y_edge;

    public uint _pad;
};

public struct LineSoup {
    public uint path_ix;

    public uint _pad;

    public float2 p0;

    public float2 p1;

};

public struct SegmentCount {
    public uint line_ix;

    public uint counts;

}

// === Bbox ===
public struct PathBbox {
    public int x0;

    public int y0;

    public int x1;

    public int y1;

    public uint draw_flags;

    public uint trans_ix;

};

// === Clip ===
public struct Bic {
    public uint a;

    public uint b;

};

public struct ClipInp {
    public uint ix;

    public int path_ix;

};

public struct ClipEl {
    public uint parent_ix;

    public uint _pad0;
    public uint _pad1;
    public uint _pad2;

    public float4 bbox;

};

// bbox_intersect from original Vello bbox shader
public float4 bbox_intersect(float4 a, float4 b) {
    return float4(max(a.xy, b.xy), min(a.zw, b.zw));
}

// bic_combine from original Vello clip shader
public Bic bic_combine(Bic x, Bic y) {
    uint m = min(x.b, y.a);
    Bic r;
    r.a = x.a + y.a - m;
    r.b = x.b + y.b - m;
    return r;
}

extension Bic : IMonoid {
    static Bic identity() { Bic b; b.a = 0; b.b = 0; return b; }
    Bic combine(Bic other) { return bic_combine(this, other); }
}

// === Bump ===
public static const uint STAGE_BINNING = 0x1;
public static const uint STAGE_TILE_ALLOC = 0x2;
public static const uint STAGE_FLATTEN = 0x4;
public static const uint STAGE_PATH_COUNT = 0x8;
public static const uint STAGE_COARSE = 0x10;

// Layout must match config.rs BumpAllocators. Fields are written atomically by other stages.
public struct BumpAllocators {
    public uint failed;

    public uint binning;

    public uint ptcl;

    public uint tile;

    public uint seg_counts;

    public uint segments;

    public uint blend;

    public uint lines;

};

// Must match Rust IndirectCount (count_x, count_y, count_z, pad0) = 16 bytes stride
public struct IndirectCount {
    public uint count_x;
    public uint count_y;
    public uint count_z;
    public uint pad0;
};

// Stage indices for multi-entry indirect dispatch buffer (Phase 1).
// Must match ekrano_encoding config.rs. Prefixed to avoid conflict with bump STAGE_* flags.
public static const uint INDIRECT_STAGE_PATHTAG_REDUCE = 0;
public static const uint INDIRECT_STAGE_PATHTAG_REDUCE2 = 1;
public static const uint INDIRECT_STAGE_PATHTAG_SCAN1 = 2;
public static const uint INDIRECT_STAGE_PATHTAG_SCAN = 3;
public static const uint INDIRECT_STAGE_PATHTAG_SCAN_LARGE = 4;
public static const uint INDIRECT_STAGE_BBOX_CLEAR = 5;
public static const uint INDIRECT_STAGE_FLATTEN = 6;
public static const uint INDIRECT_STAGE_DRAW_REDUCE = 7;
public static const uint INDIRECT_STAGE_DRAW_LEAF = 8;
public static const uint INDIRECT_STAGE_CLIP_REDUCE = 9;
public static const uint INDIRECT_STAGE_CLIP_LEAF = 10;
public static const uint INDIRECT_STAGE_BINNING = 11;
public static const uint INDIRECT_STAGE_TILE_ALLOC = 12;
public static const uint INDIRECT_STAGE_PATH_COUNT_SETUP = 13;
public static const uint INDIRECT_STAGE_PATH_COUNT = 14;
public static const uint INDIRECT_STAGE_BACKDROP = 15;
public static const uint INDIRECT_STAGE_COARSE = 16;
public static const uint INDIRECT_STAGE_PATH_TILING_SETUP = 17;
public static const uint INDIRECT_STAGE_PATH_TILING = 18;
public static const uint INDIRECT_STAGE_FINE = 19;

public static const uint N_INDIRECT_STAGES = 20;

// Layout must match ekrano_encoding WorkgroupCountsGpu.
public struct WorkgroupCountsGpu {
    public uint4 entries[N_INDIRECT_STAGES];
}

// === PTCL - command layout constants ===
public static const uint PTCL_INITIAL_ALLOC = 64;
public static const uint PTCL_INCREMENT = 256;
public static const uint PTCL_HEADROOM = 2;
public static const uint CMD_END = 0;
public static const uint CMD_FILL = 1;
public static const uint CMD_STROKE = 2;
public static const uint CMD_SOLID = 3;
public static const uint CMD_COLOR = 5;
public static const uint CMD_LIN_GRAD = 6;
public static const uint CMD_RAD_GRAD = 7;
public static const uint CMD_SWEEP_GRAD = 8;
public static const uint CMD_IMAGE = 9;
public static const uint CMD_BEGIN_CLIP = 10;
public static const uint CMD_END_CLIP = 11;
public static const uint CMD_JUMP = 12;
public static const uint CMD_BLUR_RECT = 13;

public struct CmdFill {
    public uint size_and_rule;

    public uint seg_data;

    public int backdrop;

};

public struct CmdStroke {
    public uint tile;

    public float half_width;

};

public struct CmdJump {
    public uint new_ix;

};

public struct CmdColor {
    public uint rgba_color;

};

public struct CmdEndClip {
    public uint blend;

    public float alpha;

};

// === Cubic ===
public struct Cubic {
    public float2 p0;

    public float2 p1;

    public float2 p2;

    public float2 p3;

    public float2 stroke;

    public uint path_ix;

    public uint flags;

};

public static const uint CUBIC_IS_STROKE = 1;

// === Transform ===
public struct Transform {
    public float4 matrx;

    public float2 translate;

};

public Transform operator*(Transform a, Transform b) { return transform_mul(a, b); }

public float2 operator*(Transform t, float2 p) {
    return t.matrx.xy * p.x + t.matrx.zw * p.y + t.translate;
}

public float2 transform_apply(Transform transform, float2 p) {
    return transform.matrx.xy * p.x + transform.matrx.zw * p.y + transform.translate;
}

public Transform transform_identity() {
    Transform r;
    r.matrx = float4(1.0, 0.0, 0.0, 1.0);
    r.translate = float2(0.0, 0.0);
    return r;
}

public Transform transform_inverse(Transform transform) {
    float inv_det = 1.0 / (transform.matrx.x * transform.matrx.w - transform.matrx.y * transform.matrx.z);
    float4 inv_mat = inv_det * float4(transform.matrx.w, -transform.matrx.y, -transform.matrx.z, transform.matrx.x);
    float2 inv_tr = float2(inv_mat.x * -transform.translate.x + inv_mat.z * -transform.translate.y,
                           inv_mat.y * -transform.translate.x + inv_mat.w * -transform.translate.y);
    Transform r;
    r.matrx = inv_mat;
    r.translate = inv_tr;
    return r;
}

public Transform transform_mul(Transform a, Transform b) {
    Transform r;
    r.matrx = float4(
        a.matrx.x * b.matrx.x + a.matrx.z * b.matrx.y,
        a.matrx.y * b.matrx.x + a.matrx.w * b.matrx.y,
        a.matrx.x * b.matrx.z + a.matrx.z * b.matrx.w,
        a.matrx.y * b.matrx.z + a.matrx.w * b.matrx.w
    );
    r.translate = a.matrx.xy * b.translate.x + a.matrx.zw * b.translate.y + a.translate;
    return r;
}

public Transform from_poly2(float2 p0, float2 p1) {
    Transform r;
    r.matrx = float4(p1.y - p0.y, p0.x - p1.x, p1.x - p0.x, p1.y - p0.y);
    r.translate = float2(p0.x, p0.y);
    return r;
}

public Transform two_point_to_unit_line(float2 p0, float2 p1) {
    Transform tmp1 = from_poly2(p0, p1);
    Transform inv = transform_inverse(tmp1);
    Transform tmp2 = from_poly2(float2(0.0, 0.0), float2(1.0, 0.0));
    return transform_mul(tmp2, inv);
}

// === BinHeader ===
public struct BinHeader {
    public uint element_count;

    public uint chunk_offset;

};

// === Pathtag helpers ===
public TagMonoid tag_monoid_identity() {
    TagMonoid c;
    c.trans_ix = 0; c.pathseg_ix = 0; c.pathseg_offset = 0; c.style_ix = 0; c.path_ix = 0;
    return c;
}

public TagMonoid operator+(TagMonoid a, TagMonoid b) {
    TagMonoid c;
    c.trans_ix = a.trans_ix + b.trans_ix;
    c.pathseg_ix = a.pathseg_ix + b.pathseg_ix;
    c.pathseg_offset = a.pathseg_offset + b.pathseg_offset;
    c.style_ix = a.style_ix + b.style_ix;
    c.path_ix = a.path_ix + b.path_ix;
    return c;
}

public TagMonoid combine_tag_monoid(TagMonoid a, TagMonoid b) { return a + b; }

extension TagMonoid : IMonoid {
    static TagMonoid identity() { return tag_monoid_identity(); }
    TagMonoid combine(TagMonoid other) { return this + other; }
}

public TagMonoid reduce_tag(uint tag_word) {
    TagMonoid c;
    uint point_count = tag_word & 0x3030303u;
    c.pathseg_ix = countbits((point_count * 7u) & 0x4040404u);
    c.trans_ix = countbits(tag_word & (PATH_TAG_TRANSFORM * 0x1010101u));
    uint n_points = point_count + ((tag_word >> 2u) & 0x1010101u);
    uint a = n_points + (n_points & (((tag_word >> 3u) & 0x1010101u) * 15u));
    a += a >> 8u;
    a += a >> 16u;
    c.pathseg_offset = a & 0xffu;
    c.path_ix = countbits(tag_word & (PATH_TAG_PATH * 0x1010101u));
    c.style_ix = countbits(tag_word & (PATH_TAG_STYLE * 0x1010101u)) * STYLE_SIZE_IN_WORDS;
    return c;
}

public struct PathTagScanner : IScannable<TagMonoid> {
    static TagMonoid map(uint raw_input) { return reduce_tag(raw_input); }
}

// === Drawtag helpers ===
public DrawMonoid draw_monoid_identity() {
    DrawMonoid c;
    c.path_ix = 0; c.clip_ix = 0; c.scene_offset = 0; c.info_offset = 0;
    return c;
}

public DrawMonoid operator+(DrawMonoid a, DrawMonoid b) {
    DrawMonoid c;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    c.scene_offset = a.scene_offset + b.scene_offset;
    c.info_offset = a.info_offset + b.info_offset;
    return c;
}

public DrawMonoid combine_draw_monoid(DrawMonoid a, DrawMonoid b) { return a + b; }

extension DrawMonoid : IMonoid {
    static DrawMonoid identity() { return draw_monoid_identity(); }
    DrawMonoid combine(DrawMonoid other) { return this + other; }
}

public DrawMonoid map_draw_tag(uint tag_word) {
    DrawMonoid c;
    c.path_ix = (tag_word != DRAWTAG_NOP) ? 1u : 0u;
    c.clip_ix = tag_word & 1u;
    c.scene_offset = (tag_word >> 2u) & 0x07u;
    c.info_offset = (tag_word >> 6u) & 0x0fu;
    return c;
}

public struct DrawTagScanner : IScannable<DrawMonoid> {
    static DrawMonoid map(uint raw_input) { return map_draw_tag(raw_input); }
}

// Read-only scene: use ReadOnlyBuffer when bound via goldy_dyn_buf_ro; StorageBuffer when RW binding (e.g. draw_reduce).
public uint read_draw_tag_from_scene(ReadOnlyBuffer<uint> scene, Config config, uint ix) {
    if (ix < config.n_drawobj) {
        return scene[config.drawtag_base + ix];
    }
    return DRAWTAG_NOP;
}

public uint read_draw_tag_from_scene(StorageBuffer<uint> scene, Config config, uint ix) {
    if (ix < config.n_drawobj) {
        return scene[config.drawtag_base + ix];
    }
    return DRAWTAG_NOP;
}

// === Blend - mix and compose mode constants ===
public static const uint MIX_NORMAL = 0;
public static const uint MIX_MULTIPLY = 1;
public static const uint MIX_SCREEN = 2;
public static const uint MIX_OVERLAY = 3;
public static const uint MIX_DARKEN = 4;
public static const uint MIX_LIGHTEN = 5;
public static const uint MIX_COLOR_DODGE = 6;
public static const uint MIX_COLOR_BURN = 7;
public static const uint MIX_HARD_LIGHT = 8;
public static const uint MIX_SOFT_LIGHT = 9;
public static const uint MIX_DIFFERENCE = 10;
public static const uint MIX_EXCLUSION = 11;
public static const uint MIX_HUE = 12;
public static const uint MIX_SATURATION = 13;
public static const uint MIX_COLOR = 14;
public static const uint MIX_LUMINOSITY = 15;
public static const uint MIX_CLIP = 128;

public static const uint COMPOSE_CLEAR = 0;
public static const uint COMPOSE_COPY = 1;
public static const uint COMPOSE_DEST = 2;
public static const uint COMPOSE_SRC_OVER = 3;
public static const uint COMPOSE_DEST_OVER = 4;
public static const uint COMPOSE_SRC_IN = 5;
public static const uint COMPOSE_DEST_IN = 6;
public static const uint COMPOSE_SRC_OUT = 7;
public static const uint COMPOSE_DEST_OUT = 8;
public static const uint COMPOSE_SRC_ATOP = 9;
public static const uint COMPOSE_DEST_ATOP = 10;
public static const uint COMPOSE_XOR = 11;
public static const uint COMPOSE_PLUS = 12;
public static const uint COMPOSE_PLUS_LIGHTER = 13;

// unpack4x8unorm / pack4x8unorm (portable helpers)
public float4 unpack4x8unorm(uint x) {
    return float4(
        float((x >> 0) & 0xFF) / 255.0,
        float((x >> 8) & 0xFF) / 255.0,
        float((x >> 16) & 0xFF) / 255.0,
        float((x >> 24) & 0xFF) / 255.0
    );
}
public uint pack4x8unorm(float4 v) {
    return (uint(saturate(v.x) * 255.0 + 0.5) << 0) |
           (uint(saturate(v.y) * 255.0 + 0.5) << 8) |
           (uint(saturate(v.z) * 255.0 + 0.5) << 16) |
           (uint(saturate(v.w) * 255.0 + 0.5) << 24);
}

// === PTCL structs ===
public struct CmdBlurRect {
    public uint rgba_color;

    public Transform xform;

    public float width;

    public float height;

    public float radius;

    public float std_dev;

};

public struct CmdLinGrad {
    public uint index;

    public uint extend_mode;

    public float line_x;

    public float line_y;

    public float line_c;

};

public struct CmdRadGrad {
    public uint index;

    public uint extend_mode;

    public Transform xform;

    public float focal_x;

    public float radius;

    public uint kind;

    public uint flags;

};

public struct CmdSweepGrad {
    public uint index;

    public uint extend_mode;

    public Transform xform;

    public float t0;

    public float t1;

};

public struct CmdImage {
    public Transform xform;

    public float2 atlas_offset;

    public float2 extents;

    public uint format;

    public uint x_extend_mode;

    public uint y_extend_mode;

    public uint quality;

    public float alpha;

    public uint alpha_type;

};

// === Blend functions ===
public float3 blend_screen(float3 cb, float3 cs) {
    return cb + cs - (cb * cs);
}
public float blend_color_dodge(float cb, float cs) {
    if (cb == 0.0) return 0.0;
    if (cs == 1.0) return 1.0;
    return min(1.0, cb / (1.0 - cs));
}
public float blend_color_burn(float cb, float cs) {
    if (cb == 1.0) return 1.0;
    if (cs == 0.0) return 0.0;
    return 1.0 - min(1.0, (1.0 - cb) / cs);
}
public float3 blend_hard_light(float3 cb, float3 cs) {
    return select(cs <= 0.5, cb * 2.0 * cs, blend_screen(cb, 2.0 * cs - 1.0));
}
public float3 blend_soft_light(float3 cb, float3 cs) {
    float3 d = select(cb <= 0.25, ((16.0 * cb - 12.0) * cb + 4.0) * cb, sqrt(cb));
    return select(cs <= 0.5, cb + (2.0 * cs - 1.0) * (d - cb), cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb));
}
public float blend_sat(float3 c) {
    return max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z));
}
public float blend_lum(float3 c) {
    return dot(c, float3(0.3, 0.59, 0.11));
}
public float blend_svg_lum(float3 c) {
    return dot(c, float3(0.2125, 0.7154, 0.0721));
}
public float3 blend_clip_color(float3 c) {
    float l = blend_lum(c);
    float n = min(c.x, min(c.y, c.z));
    float x = max(c.x, max(c.y, c.z));
    if (n < 0.0) c = l + (((c - l) * l) / (l - n));
    if (x > 1.0) c = l + (((c - l) * (1.0 - l)) / (x - l));
    return c;
}
public float3 blend_set_lum(float3 c, float l) {
    return blend_clip_color(c + (l - blend_lum(c)));
}
public float3 blend_set_sat(float3 c, float s) {
    float r = c.r, g = c.g, b = c.b;
    if (r <= g) {
        if (g <= b) { if (max(b,r)>min(b,r)) { g = ((g-min(b,r))*s)/(max(b,r)-min(b,r)); b = s; } else { g = b = 0; } r = 0; }
        else if (r <= b) { if (max(g,r)>min(g,r)) { b = ((b-min(g,r))*s)/(max(g,r)-min(g,r)); g = s; } else { b = g = 0; } r = 0; }
        else { if (max(g,b)>min(g,b)) { r = ((r-min(g,b))*s)/(max(g,b)-min(g,b)); g = s; } else { r = g = 0; } b = 0; }
    } else if (r <= b) { if (max(g,r)>min(g,r)) { b = ((b-min(g,r))*s)/(max(g,r)-min(g,r)); r = s; } else { b = r = 0; } g = 0; }
    else if (g <= b) { if (max(b,g)>min(b,g)) { r = ((r-min(b,g))*s)/(max(b,g)-min(b,g)); b = s; } else { r = b = 0; } g = 0; }
    else { if (max(b,r)>min(b,r)) { g = ((g-min(b,r))*s)/(max(b,r)-min(b,r)); b = s; } else { g = b = 0; } r = 0; }
    return float3(r, g, b);
}
public float3 blend_mix(float3 cb, float3 cs, uint mode) {
    if (mode == MIX_MULTIPLY) return cb * cs;
    if (mode == MIX_SCREEN) return blend_screen(cb, cs);
    if (mode == MIX_OVERLAY) return blend_hard_light(cs, cb);
    if (mode == MIX_DARKEN) return min(cb, cs);
    if (mode == MIX_LIGHTEN) return max(cb, cs);
    if (mode == MIX_COLOR_DODGE) return float3(blend_color_dodge(cb.x, cs.x), blend_color_dodge(cb.y, cs.y), blend_color_dodge(cb.z, cs.z));
    if (mode == MIX_COLOR_BURN) return float3(blend_color_burn(cb.x, cs.x), blend_color_burn(cb.y, cs.y), blend_color_burn(cb.z, cs.z));
    if (mode == MIX_HARD_LIGHT) return blend_hard_light(cb, cs);
    if (mode == MIX_SOFT_LIGHT) return blend_soft_light(cb, cs);
    if (mode == MIX_DIFFERENCE) return abs(cb - cs);
    if (mode == MIX_EXCLUSION) return cb + cs - 2.0 * cb * cs;
    if (mode == MIX_HUE) return blend_set_lum(blend_set_sat(cs, blend_sat(cb)), blend_lum(cb));
    if (mode == MIX_SATURATION) return blend_set_lum(blend_set_sat(cb, blend_sat(cs)), blend_lum(cb));
    if (mode == MIX_COLOR) return blend_set_lum(cs, blend_lum(cb));
    if (mode == MIX_LUMINOSITY) return blend_set_lum(cb, blend_lum(cs));
    return cs;
}
public float3 blend_unpremultiply(float4 color) {
    float inv_alpha = 1.0 / max(color.a, 1e-15);
    return color.rgb * inv_alpha;
}
public float4 blend_compose(float3 cb, float3 cs, float ab, float as_, uint compose_mode) {
    float fa = 0.0, fb = 0.0;
    if (compose_mode == COMPOSE_COPY) { fa = 1.0; fb = 0.0; }
    else if (compose_mode == COMPOSE_DEST) { fa = 0.0; fb = 1.0; }
    else if (compose_mode == COMPOSE_SRC_OVER) { fa = 1.0; fb = 1.0 - as_; }
    else if (compose_mode == COMPOSE_DEST_OVER) { fa = 1.0 - ab; fb = 1.0; }
    else if (compose_mode == COMPOSE_SRC_IN) { fa = ab; fb = 0.0; }
    else if (compose_mode == COMPOSE_DEST_IN) { fa = 0.0; fb = as_; }
    else if (compose_mode == COMPOSE_SRC_OUT) { fa = 1.0 - ab; fb = 0.0; }
    else if (compose_mode == COMPOSE_DEST_OUT) { fa = 0.0; fb = 1.0 - as_; }
    else if (compose_mode == COMPOSE_SRC_ATOP) { fa = ab; fb = 1.0 - as_; }
    else if (compose_mode == COMPOSE_DEST_ATOP) { fa = 1.0 - ab; fb = as_; }
    else if (compose_mode == COMPOSE_XOR) { fa = 1.0 - ab; fb = 1.0 - as_; }
    else if (compose_mode == COMPOSE_PLUS) { fa = 1.0; fb = 1.0; }
    else if (compose_mode == COMPOSE_PLUS_LIGHTER) {
        return min(float4(1.0, 1.0, 1.0, 1.0), float4(as_ * cs + ab * cb, as_ + ab));
    }
    float as_fa = as_ * fa;
    float ab_fb = ab * fb;
    float3 co = as_fa * cs + ab_fb * cb;
    return float4(co, min(as_fa + ab_fb, 1.0));
}
public float4 blend_mix_compose(float4 backdrop, float4 src, uint mode) {
    uint BLEND_DEFAULT = (MIX_NORMAL << 8) | COMPOSE_SRC_OVER;
    if ((mode & 0x7FFF) == BLEND_DEFAULT) return backdrop * (1.0 - src.a) + src;
    float3 cs = blend_unpremultiply(src);
    float3 cb = blend_unpremultiply(backdrop);
    uint mix_mode = mode >> 8;
    float3 mixed = blend_mix(cb, cs, mix_mode);
    cs = lerp(cs, mixed, backdrop.a);
    uint compose_mode = mode & 0xFF;
    if (compose_mode == COMPOSE_SRC_OVER)
        return float4(lerp(backdrop.rgb, cs, src.a), src.a + backdrop.a * (1.0 - src.a));
    return blend_compose(cb, cs, backdrop.a, src.a, compose_mode);
}
public float4 premul_alpha(float4 rgba) {
    return float4(rgba.rgb * rgba.a, rgba.a);
}

// === Fine shader helpers ===
public static const uint LUMINANCE_MASK_LAYER = 0x10000;
public static const uint IMAGE_QUALITY_LOW = 0;
public static const uint IMAGE_QUALITY_MEDIUM = 1;
public static const uint IMAGE_QUALITY_HIGH = 2;
public static const uint PIXEL_FORMAT_RGBA = 0;
public static const uint PIXEL_FORMAT_BGRA = 1;
public static const uint ALPHA = 0;
public static const uint PREMULTIPLIED_ALPHA = 1;
public static const uint EXTEND_PAD = 0;
public static const uint EXTEND_REPEAT = 1;
public static const uint EXTEND_REFLECT = 2;

public float3 unpremultiply(float4 color) { return blend_unpremultiply(color); }
public float svg_lum(float3 c) { return blend_svg_lum(c); }

public float4 maybe_premul_alpha(float4 pixel, uint alpha_type) {
    return (alpha_type == PREMULTIPLIED_ALPHA) ? pixel : premul_alpha(pixel);
}

public float extend_mode_normalized(float t, uint mode) {
    if (mode == EXTEND_PAD) return saturate(t);
    if (mode == EXTEND_REPEAT) return frac(t);
    return abs(t - 2.0 * round(0.5 * t));
}
public float extend_mode(float t, uint mode, float max_val) {
    if (mode == EXTEND_PAD) return clamp(t, 0.0, max_val);
    return extend_mode_normalized(t / max_val, mode) * max_val;
}

public float4 pixel_format(float4 pixel, uint format) {
    return (format == PIXEL_FORMAT_BGRA) ? pixel.bgra : pixel;
}

public static const uint GRADIENT_WIDTH = 512;
public static const uint PIXELS_PER_THREAD = 4;

public float4 premul_over(float4 bg, float4 fg) {
    return bg * (1.0 - fg.a) + fg;
}

public void composite_strip(inout float4 rgba[PIXELS_PER_THREAD], float4 fg, float area[PIXELS_PER_THREAD]) {
    for (uint i = 0; i < PIXELS_PER_THREAD; i++) {
        float4 fg_i = fg * area[i];
        rgba[i] = premul_over(rgba[i], fg_i);
    }
}

public float erf7(float x) {
    float y = clamp(x * 1.1283791671, -100.0, 100.0);
    float yy = y * y;
    float z = y + (0.24295 + (0.03395 + 0.0104 * yy) * yy) * (y * yy);
    return z / sqrt(1.0 + z * z);
}

public void signed_atomic_min(ByteAddressView buf, uint addr, int new_val) {
    uint orig;
    buf.InterlockedCompareExchange(addr, 0, 0, orig);
    while (new_val < int(orig)) {
        uint expected = orig;
        buf.InterlockedCompareExchange(addr, expected, uint(new_val), orig);
        if (orig == expected) break;
    }
}

public void signed_atomic_max(ByteAddressView buf, uint addr, int new_val) {
    uint orig;
    buf.InterlockedCompareExchange(addr, 0, 0, orig);
    while (new_val > int(orig)) {
        uint expected = orig;
        buf.InterlockedCompareExchange(addr, expected, uint(new_val), orig);
        if (orig == expected) break;
    }
}

public Transform read_transform(ReadOnlyBuffer<uint> scene, uint transform_base, uint ix) {
    uint base = transform_base + ix * 6;
    float c0 = asfloat(scene[base]);
    float c1 = asfloat(scene[base + 1]);
    float c2 = asfloat(scene[base + 2]);
    float c3 = asfloat(scene[base + 3]);
    float c4 = asfloat(scene[base + 4]);
    float c5 = asfloat(scene[base + 5]);
    Transform r;
    r.matrx = float4(c0, c1, c2, c3);
    r.translate = float2(c4, c5);
    return r;
}

public Transform read_transform(StorageBuffer<uint> scene, uint transform_base, uint ix) {
    uint base = transform_base + ix * 6;
    float c0 = asfloat(scene[base]);
    float c1 = asfloat(scene[base + 1]);
    float c2 = asfloat(scene[base + 2]);
    float c3 = asfloat(scene[base + 3]);
    float c4 = asfloat(scene[base + 4]);
    float c5 = asfloat(scene[base + 5]);
    Transform r;
    r.matrx = float4(c0, c1, c2, c3);
    r.translate = float2(c4, c5);
    return r;
}
