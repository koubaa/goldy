// Compile-only test for goldy_exp math and primitives utilities.
//
// This shader exercises positive_mod, modelview_right, billboard_cylindrical_offset,
// erf7, and signed_atomic_* so that Slang-level type errors or syntax regressions
// are caught at compile time across all targets.
//
// Run via: cargo test test_goldy_exp_math_compiles

import goldy_exp;

#define OUT goldy_dyn_scattered<float>(0)
#define ATOM_BUF goldy_dyn_byte_address(1)

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

    // ── Transform / bbox / extend (upstreamed from Ekrano) ───────────────────
    Transform id = transform_identity();
    float2 tp = transform_apply(id, float2(3.0, 4.0));
    OUT[12] = tp.x;
    OUT[13] = tp.y;

    float4 bx = bbox_intersect(float4(0, 0, 10, 10), float4(5, 5, 15, 15));
    OUT[14] = bx.x;  // 5
    OUT[15] = bx.z;  // 10

    OUT[16] = extend_mode_normalized(1.25, 1u);  // repeat → frac 0.25
    OUT[17] = extend_mode(5.0, 1u, 4.0);       // repeat on [0,4] → 1.0

    float4 rgba = unpack4x8unorm(0xFFE0C080u);
    OUT[18] = rgba.r;  // 0x80 / 255
    OUT[19] = rgba.a;  // 0xFF / 255

    // ── erf7 ─────────────────────────────────────────────────────────────────
    OUT[20] = erf7(0.0);
    OUT[21] = erf7(1.0);

    // ── signed_atomic_* (compile + link path; values not asserted) ───────────
    ATOM_BUF.Store(0, 0x7fffffffu);
    signed_atomic_min(ATOM_BUF, 0, -5);
    signed_atomic_max(ATOM_BUF, 4, 10);
    OUT[22] = asfloat(ATOM_BUF.Load(0));
    OUT[23] = asfloat(ATOM_BUF.Load(4));
}
