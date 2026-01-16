// Classic demoscene plasma effect
// Uses goldy_exp module for vertex format and color utilities
//
// ## Bindless Support
//
// When __BINDLESS__ is defined (Metal Argument Buffers Tier 2):
// - Uniforms are accessed via ParameterBlock for argument buffer support
// - Traditional binding syntax works as fallback

import goldy_exp;

// Uniform structure
struct TimeUniforms {
    float time;
};

#ifdef __BINDLESS__
// Bindless mode: Use ParameterBlock for Metal argument buffer support
struct PlasmaResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<PlasmaResources> gResources;
#define TIME gResources.uniforms.time
#else
// Traditional binding mode
[[vk::binding(0, 0)]]
ConstantBuffer<TimeUniforms> uniforms;
#define TIME uniforms.time
#endif

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = scale_uv(input.uv, 4.0);
    float t = TIME;
    
    // Classic plasma formula
    float v = sin(uv.x + t);
    v += sin(uv.y + t);
    v += sin(uv.x + uv.y + t);
    
    float cx = uv.x + 0.5 * sin(t / 3.0);
    float cy = uv.y + 0.5 * cos(t / 2.0);
    v += sin(sqrt(cx * cx + cy * cy + 1.0) + t);
    
    v = v / 2.0;
    
    // Use rainbow palette from goldy module
    return float4(rainbow(v), 1.0);
}
