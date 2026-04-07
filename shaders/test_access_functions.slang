// Test shader for unified access pattern functions
//
// Run: cargo test --package goldy test_access_functions -- --nocapture

import goldy_exp;

struct TimeUniforms {
    float time;
};

#define TIME goldy_dyn_broadcast<TimeUniforms>(0).time

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
