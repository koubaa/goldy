// Classic demoscene plasma effect
// Uses goldy_exp module for vertex format and color utilities

import goldy_exp;

// Uniform buffer for time (updated each frame)
[[vk::binding(0, 0)]]
cbuffer Uniforms {
    float time;
};

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = scale_uv(input.uv, 4.0);
    float t = time;
    
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
