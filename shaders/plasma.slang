// Classic demoscene plasma effect
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
    float2 uv = scale_uv(input.uv, 4.0);
    float t = uniforms.time;
    
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
