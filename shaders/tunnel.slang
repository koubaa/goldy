// Classic demoscene tunnel effect
// Uses goldy_exp module for vertex format utilities

import goldy_exp;

struct TimeUniforms {
    float time;
};

#define TIME goldy_broadcast<TimeUniforms>(0).time

[shader("vertex")]
FullscreenVarying vs_main(uint vertex_id : SV_VertexID) {
    return vs_fullscreen_triangle(vertex_id);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = (input.uv - 0.5) * 2.0;
    float t = TIME;
    
    // Polar coordinates
    float dist = length(uv);
    float angle = atan2(uv.y, uv.x);
    
    // Tunnel coordinates
    float tunnel_depth = 1.0 / (dist + 0.1);
    float tunnel_angle = angle / 3.14159 + t * 0.2;
    
    // Animated texture coordinates
    float tx = tunnel_angle * 4.0;
    float ty = tunnel_depth - t * 2.0;
    
    // Checkerboard pattern
    float checker = floor(tx) + floor(ty);
    bool is_white = fmod(checker, 2.0) == 0.0;
    
    // Color based on depth and checker
    float depth_color = 1.0 - dist * 0.5;
    float3 color;
    
    if (is_white) {
        color = float3(0.8, 0.2, 0.4) * depth_color;
    } else {
        color = float3(0.2, 0.4, 0.8) * depth_color;
    }
    
    // Add glow at center
    color += float3(0.3, 0.5, 1.0) * (1.0 - dist) * (1.0 - dist);
    
    // Fog at edges
    color *= 1.0 - dist * 0.3;
    
    return float4(color, 1.0);
}
