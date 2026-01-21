// Test shader for unified access pattern functions
//
// This shader demonstrates:
// - SPIRV/DX12/Metal Tier 2: Uses goldy_broadcast<T>() for unified access
// - Metal Tier 1: Uses traditional bindings with fallback
//
// Run: cargo test --package goldy test_access_functions -- --nocapture

import goldy_exp;

struct TimeUniforms {
    float time;
};

// Cross-platform resource access using preprocessor fallback for Metal Tier 1
#if defined(__METAL__) && !defined(__METAL_BINDLESS__)
// Metal Tier 1: Traditional binding (GitHub CI compatibility)
ConstantBuffer<TimeUniforms> uniforms : register(b0);
#define TIME uniforms.time
#else
// SPIRV, DX12, Metal Tier 2: Unified access function
#define TIME goldy_broadcast<TimeUniforms>(0).time
#endif

[shader("vertex")]
FullscreenVarying vs_main(uint vertex_id : SV_VertexID) {
    return vs_fullscreen_triangle(vertex_id);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float t = TIME;
    
    float2 uv = scale_uv(input.uv, 4.0);
    
    // Classic plasma formula
    float v = sin(uv.x + t);
    v += sin(uv.y + t);
    v += sin(uv.x + uv.y + t);
    
    float cx = uv.x + 0.5 * sin(t / 3.0);
    float cy = uv.y + 0.5 * cos(t / 2.0);
    v += sin(sqrt(cx * cx + cy * cy + 1.0) + t);
    
    v = v / 2.0;
    
    return float4(rainbow(v), 1.0);
}
