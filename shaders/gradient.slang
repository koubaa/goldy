// Animated color gradient
// Uses vertex-less fullscreen triangle and time uniform

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
    float2 uv = input.uv;
    float t = TIME;
    
    // Smooth animated gradient using sine waves
    float3 color;
    color.r = 0.5 + 0.5 * sin(uv.x * 3.14159 + t * 0.7);
    color.g = 0.5 + 0.5 * sin(uv.y * 3.14159 + t * 0.5 + 2.0);
    color.b = 0.5 + 0.5 * sin((uv.x + uv.y) * 2.0 + t * 0.3 + 4.0);
    
    // Add subtle variation based on position
    color *= 0.8 + 0.2 * sin(uv.x * 6.28 + uv.y * 6.28 + t);
    
    return float4(color, 1.0);
}
