// Math utilities for Goldy shaders (EXPERIMENTAL)
// Part of the goldy_exp module

implementing goldy_exp;

// Constants
public static const float PI = 3.14159265359;
public static const float TAU = 6.28318530718;

// Hash functions for procedural generation
public float hash(float p) {
    return frac(sin(p * 127.1) * 43758.5453);
}

public float hash2(float2 p) {
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

// UV utilities
public float2 center_uv(float2 uv) {
    return (uv - 0.5) * 2.0;
}

public float2 scale_uv(float2 uv, float scale) {
    return uv * scale;
}

// Polar coordinate conversion
public float2 to_polar(float2 uv) {
    float dist = length(uv);
    float angle = atan2(uv.y, uv.x);
    return float2(dist, angle);
}

// Positive modulo — HLSL fmod returns negative for negative inputs
public float positive_mod(float x, float m) {
    return fmod(fmod(x, m) + m, m);
}

public float2 positive_mod(float2 x, float2 m) {
    return float2(
        fmod(fmod(x.x, m.x) + m.x, m.x),
        fmod(fmod(x.y, m.y) + m.y, m.y)
    );
}

// Smoothstep helper
public float smootherstep(float edge0, float edge1, float x) {
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

// ============================================================================
// 2D affine transforms (column vectors: p' = M * p + t)
// ============================================================================

/// Compact 2D affine: linear part in `matrx` as (m00, m01, m10, m11), translation in `translate`.
public struct Transform {
    public float4 matrx;
    public float2 translate;
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
    float2 inv_tr = float2(
        inv_mat.x * -transform.translate.x + inv_mat.z * -transform.translate.y,
        inv_mat.y * -transform.translate.x + inv_mat.w * -transform.translate.y
    );
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

/// Build transform from two points (Vello/Ekrano poly2 helper).
public Transform from_poly2(float2 p0, float2 p1) {
    Transform r;
    r.matrx = float4(p1.y - p0.y, p0.x - p1.x, p1.x - p0.x, p1.y - p0.y);
    r.translate = float2(p0.x, p0.y);
    return r;
}

/// Map the line through p0–p1 to the unit x-axis segment [0,1].
public Transform two_point_to_unit_line(float2 p0, float2 p1) {
    Transform tmp1 = from_poly2(p0, p1);
    Transform inv = transform_inverse(tmp1);
    Transform tmp2 = from_poly2(float2(0.0, 0.0), float2(1.0, 0.0));
    return transform_mul(tmp2, inv);
}

// ============================================================================
// Axis-aligned boxes (float4 as min.xy, max.zw)
// ============================================================================

public float4 bbox_intersect(float4 a, float4 b) {
    return float4(max(a.xy, b.xy), min(a.zw, b.zw));
}

// ============================================================================
// Extend modes for gradients / UV domains (0 = pad, 1 = repeat, 2 = reflect)
// ============================================================================

public float extend_mode_normalized(float t, uint mode) {
    if (mode == 0u) return saturate(t);
    if (mode == 1u) return frac(t);
    return abs(t - 2.0 * round(0.5 * t));
}

public float extend_mode(float t, uint mode, float max_val) {
    if (mode == 0u) return clamp(t, 0.0, max_val);
    return extend_mode_normalized(t / max_val, mode) * max_val;
}

// Fast erf approximation (degree-7 rational-style); used for blurred rounded rects (Ekrano fine).
public float erf7(float x) {
    float y = clamp(x * 1.1283791671, -100.0, 100.0);
    float yy = y * y;
    float z = y + (0.24295 + (0.03395 + 0.0104 * yy) * yy) * (y * yy);
    return z / sqrt(1.0 + z * z);
}
