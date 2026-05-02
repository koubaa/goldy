// Animated metaballs effect
// Uses goldy_exp module for vertex format utilities

import goldy_exp;

struct TimeUniforms {
    float time;
};

[shader("vertex")]
FullscreenVarying vs_main(uint vertex_id : SV_VertexID) {
    return vs_fullscreen_triangle(vertex_id);
}

float metaball(float2 p, float2 center, float radius) {
    float d = distance(p, center);
    return radius / (d * d + 0.001);
}

[shader("fragment")]
float4 fs_main(uniform uint uniforms_slot, FullscreenVarying input) : SV_Target {
    float2 uv = (input.uv - 0.5) * 2.0;
    float t = goldy_broadcast<TimeUniforms>(uniforms_slot).time;
    
    // Moving metaball centers
    float2 c1 = float2(sin(t * 1.1) * 0.5, cos(t * 0.9) * 0.5);
    float2 c2 = float2(sin(t * 0.8 + 1.0) * 0.6, cos(t * 1.2 + 2.0) * 0.4);
    float2 c3 = float2(sin(t * 1.3 + 2.0) * 0.4, cos(t * 0.7 + 1.0) * 0.6);
    float2 c4 = float2(cos(t * 0.9) * 0.5, sin(t * 1.1 + 3.0) * 0.5);
    float2 c5 = float2(cos(t * 1.0 + 1.5) * 0.3, sin(t * 0.8 + 0.5) * 0.7);
    
    // Sum metaball influences
    float v = 0.0;
    v += metaball(uv, c1, 0.15);
    v += metaball(uv, c2, 0.12);
    v += metaball(uv, c3, 0.18);
    v += metaball(uv, c4, 0.10);
    v += metaball(uv, c5, 0.14);
    
    // Threshold and color
    float threshold = 1.0;
    if (v > threshold) {
        float intensity = (v - threshold) / 2.0;
        float r = 0.2 + intensity * 0.3;
        float g = 0.5 + intensity * 0.4;
        float b = 0.8 + intensity * 0.2;
        return float4(r, g, b, 1.0);
    } else {
        float glow = v * 0.3;
        return float4(glow * 0.2, glow * 0.3, glow * 0.5, 1.0);
    }
}
