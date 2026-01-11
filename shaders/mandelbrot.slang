// Interactive Mandelbrot fractal explorer
// Uses goldy_exp module for vertex format and color utilities

import goldy_exp;

// Uniform buffer for view parameters (updated on input)
[[vk::binding(0, 0)]]
cbuffer Uniforms {
    float2 center;
    float zoom;
    float _padding;  // Align to 16 bytes
};

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    // Map UV to complex plane around center with zoom
    float2 c = center + (input.uv - 0.5) * 4.0 / zoom;
    float2 z = float2(0.0, 0.0);
    uint i = 0;
    const uint max_iter = 256;
    
    for (i = 0; i < max_iter; i++) {
        if (dot(z, z) > 4.0) break;
        // z = z^2 + c
        z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    }
    
    if (i >= max_iter) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    
    // Use palette from goldy module for smooth coloring
    float t = float(i) / float(max_iter);
    float3 color = palette(t, float3(5.0, 7.0, 11.0), float3(0.0, 1.0, 2.0));
    
    return float4(color, 1.0);
}
