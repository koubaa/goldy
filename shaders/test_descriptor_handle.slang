// Test shader for DescriptorHandle<T> with Goldy's custom binding layout
//
// This shader demonstrates the cleanest cross-platform approach:
// - SPIRV/DX12: Uses DescriptorHandle<T> for automatic bindless access
// - Metal: Uses ConstantBuffer<T> with register() for explicit buffer binding
//
// The preprocessor is minimal: just the declaration and a macro for access.
//
// Run: cargo test --package goldy test_descriptor_handle -- --nocapture

import goldy_exp;

struct TimeUniforms {
    float time;
};

// Cross-platform resource declaration:
// - Metal needs ConstantBuffer + register() for [[buffer(N)]] attribute
// - SPIRV/DX12 can use DescriptorHandle for automatic bindless routing
#if defined(__METAL__)
ConstantBuffer<TimeUniforms> uniforms : register(b0);
#define UNIFORMS uniforms
#else
uniform ConstantBuffer<TimeUniforms>.Handle uniforms;
#define UNIFORMS (*uniforms)
#endif

[shader("vertex")]
FullscreenVarying vs_main(uint vertex_id : SV_VertexID) {
    return vs_fullscreen_triangle(vertex_id);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    // Unified access via macro
    float t = UNIFORMS.time;
    
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
