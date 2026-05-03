// Interactive Mandelbrot fractal explorer
// Uses goldy_exp module for vertex format and color utilities

import goldy_exp;

struct ViewUniforms {
    float2 center;
    float zoom;
    float _padding;
};

[goldy_vertex]
FullscreenVarying vs_main(VertexId vertex_id) {
    FullscreenVarying output;
    
    float2 uv = float2(
        float((vertex_id.value << 1u) & 2u),
        float(vertex_id.value & 2u)
    );
    
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = uv;
    
    return output;
}

[goldy_fragment]
float4 fs_main(ViewUniforms view, FullscreenVarying input) : SV_Target {
    float2 c = view.center + (input.uv - 0.5) * 4.0 / view.zoom;
    float2 z = float2(0.0, 0.0);
    uint i = 0;
    const uint max_iter = 256;
    
    for (i = 0; i < max_iter; i++) {
        if (dot(z, z) > 4.0) break;
        z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    }
    
    if (i >= max_iter) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    
    float t = float(i) / float(max_iter);
    float3 color = palette(t, float3(5.0, 7.0, 11.0), float3(0.0, 1.0, 2.0));
    
    return float4(color, 1.0);
}
