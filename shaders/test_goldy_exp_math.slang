// Compile-only test for goldy_exp math and primitives utilities.
//
// This shader exercises positive_mod, modelview_right, and
// billboard_cylindrical_offset so that Slang-level type errors or syntax
// regressions are caught at compile time across all targets.
//
// Run via: cargo test test_goldy_exp_math_compiles

import goldy_exp;

#define OUT goldy_dyn_scattered<float>(0)

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    // ── positive_mod ─────────────────────────────────────────────────────────
    // negative dividend wraps into [0, m)
    OUT[0] = positive_mod(-1.0, 3.0);    // expected 2.0
    OUT[1] = positive_mod(2.5, 3.0);     // expected 2.5
    OUT[2] = positive_mod(0.0, 1.0);     // expected 0.0
    OUT[3] = positive_mod(-3.0, 3.0);    // expected 0.0

    // float2 overload
    float2 r2 = positive_mod(float2(-1.0, 2.5), float2(3.0, 3.0));
    OUT[4] = r2.x;  // expected 2.0
    OUT[5] = r2.y;  // expected 2.5

    // ── modelview_right ───────────────────────────────────────────────────────
    // Row-major construction: col 0 = (m[0][0], m[1][0], m[2][0]) = (1, 5, 9)
    float4x4 m = float4x4(
        1, 0, 0, 0,
        5, 1, 0, 0,
        9, 0, 1, 0,
        0, 0, 0, 1
    );
    float3 right = modelview_right(m);
    OUT[6] = right.x;  // expected 1.0
    OUT[7] = right.y;  // expected 5.0
    OUT[8] = right.z;  // expected 9.0

    // ── billboard_cylindrical_offset ─────────────────────────────────────────
    // center=(1,2,3), cam_right=(1,0,0), offset=5 → (6, 2, 3)
    float3 off = billboard_cylindrical_offset(
        float3(1.0, 2.0, 3.0),
        float3(1.0, 0.0, 0.0),
        5.0
    );
    OUT[9]  = off.x;  // expected 6.0
    OUT[10] = off.y;  // expected 2.0
    OUT[11] = off.z;  // expected 3.0
}
