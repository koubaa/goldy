// Compile-only test for goldy_exp math and primitives utilities.
//
// Exercises positive_mod, modelview_right, billboard_cylindrical_offset,
// hash, scale_uv, center_uv, to_polar, smootherstep, rainbow, palette, heat.
//
// Note: Transform, bbox_intersect, erf7, unpack4x8unorm, signed_atomic_*
// are defined in ekrano_shared (not goldy_exp) to avoid cross-module ambiguity.
//
// Run via: cargo test test_goldy_exp_math_compiles

import goldy_exp;

#define OUT goldy_dyn_scattered<float>(0)

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    // ── positive_mod ─────────────────────────────────────────────────────────
    OUT[0] = positive_mod(-1.0, 3.0);
    OUT[1] = positive_mod(2.5, 3.0);
    OUT[2] = positive_mod(0.0, 1.0);
    OUT[3] = positive_mod(-3.0, 3.0);

    float2 r2 = positive_mod(float2(-1.0, 2.5), float2(3.0, 3.0));
    OUT[4] = r2.x;
    OUT[5] = r2.y;

    // ── modelview_right ───────────────────────────────────────────────────────
    float4x4 m = float4x4(
        1, 0, 0, 0,
        5, 1, 0, 0,
        9, 0, 1, 0,
        0, 0, 0, 1
    );
    float3 right = modelview_right(m);
    OUT[6] = right.x;
    OUT[7] = right.y;
    OUT[8] = right.z;

    // ── billboard_cylindrical_offset ─────────────────────────────────────────
    float3 off = billboard_cylindrical_offset(
        float3(1.0, 2.0, 3.0),
        float3(1.0, 0.0, 0.0),
        5.0
    );
    OUT[9]  = off.x;
    OUT[10] = off.y;
    OUT[11] = off.z;

    // ── Hash / UV / Polar ────────────────────────────────────────────────────
    OUT[12] = hash(42.0);
    OUT[13] = hash2(float2(1.0, 2.0));

    float2 cuv = center_uv(float2(0.75, 0.25));
    OUT[14] = cuv.x;
    OUT[15] = cuv.y;

    float2 suv = scale_uv(float2(0.5, 0.5), 4.0);
    OUT[16] = suv.x;

    float2 pol = to_polar(float2(1.0, 0.0));
    OUT[17] = pol.x;

    OUT[18] = smootherstep(0.0, 1.0, 0.5);

    // ── Color utilities ──────────────────────────────────────────────────────
    float3 rb = rainbow(0.5);
    OUT[19] = rb.r;

    float3 pal = palette(0.5, float3(1.0, 1.0, 1.0), float3(0.0, 0.0, 0.0));
    OUT[20] = pal.r;

    float3 h = heat(0.5);
    OUT[21] = h.r;

    float3 hsv = hsv_to_rgb(float3(0.0, 1.0, 1.0));
    OUT[22] = hsv.r;

    OUT[23] = luminance(float3(1.0, 0.0, 0.0));
}
