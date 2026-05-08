// Animated color gradient
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
    float2 uv = input.uv;
    float t = uniforms.time;
    
    // Smooth animated gradient using sine waves
    float3 color;
    color.r = 0.5 + 0.5 * sin(uv.x * 3.14159 + t * 0.7);
    color.g = 0.5 + 0.5 * sin(uv.y * 3.14159 + t * 0.5 + 2.0);
    color.b = 0.5 + 0.5 * sin((uv.x + uv.y) * 2.0 + t * 0.3 + 4.0);
    
    // Add subtle variation based on position
    color *= 0.8 + 0.2 * sin(uv.x * 6.28 + uv.y * 6.28 + t);
    
    return float4(color, 1.0);
}
