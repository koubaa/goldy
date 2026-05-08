// Animated checkerboard pattern
// Uses vertex-less fullscreen triangle and time uniform

import goldy_exp;

struct TimeUniforms {
    float time;
};

[goldy_vertex]
FullscreenVarying vs_main(VertexId vertex_id) {
    return vs_fullscreen_triangle(vertex_id.value);
}

[goldy_fragment]
float4 fs_main(TimeUniforms uniforms, FullscreenVarying input) : SV_Target {
    float t = uniforms.time;
    
    // Animated wave distortion
    float2 uv = input.uv;
    uv.x += sin(uv.y * 10.0 + t * 2.0) * 0.02;
    uv.y += cos(uv.x * 10.0 + t * 1.5) * 0.02;
    
    // Scale for checker pattern
    float scale = 8.0;
    float2 checker = floor(uv * scale);
    bool is_white = fmod(checker.x + checker.y, 2.0) == 0.0;
    
    // Animate colors
    float3 color1 = float3(
        0.2 + 0.1 * sin(t),
        0.1 + 0.1 * cos(t * 1.3),
        0.3 + 0.1 * sin(t * 0.7)
    );
    float3 color2 = float3(
        0.9 + 0.1 * cos(t * 0.8),
        0.85 + 0.1 * sin(t * 1.1),
        0.8 + 0.1 * cos(t)
    );
    
    // Add subtle gradient
    color1 += uv.y * 0.1;
    color2 -= uv.y * 0.1;
    
    if (is_white) {
        return float4(color2, 1.0);
    } else {
        return float4(color1, 1.0);
    }
}
