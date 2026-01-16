// Interactive Mandelbrot fractal explorer
// Uses goldy_exp module for vertex format and color utilities
//
// ## Bindless Support
//
// When __BINDLESS__ is defined (Metal Argument Buffers Tier 2):
// - Uniforms are accessed via ParameterBlock for argument buffer support
// - Traditional binding syntax works as fallback

import goldy_exp;

// Uniform structure for view parameters
struct ViewUniforms {
    float2 center;
    float zoom;
    float _padding;  // Align to 16 bytes
};

#ifdef __BINDLESS__
// Bindless mode: Use ParameterBlock for Metal argument buffer support
struct MandelbrotResources {
    ConstantBuffer<ViewUniforms> uniforms;
};
ParameterBlock<MandelbrotResources> gResources;
#define CENTER gResources.uniforms.center
#define ZOOM gResources.uniforms.zoom
#else
// Traditional binding mode
[[vk::binding(0, 0)]]
ConstantBuffer<ViewUniforms> uniforms;
#define CENTER uniforms.center
#define ZOOM uniforms.zoom
#endif

// Vertex-less fullscreen triangle - generates geometry from vertex index
[shader("vertex")]
FullscreenVarying vs_main(uint vertex_id : SV_VertexID) {
    FullscreenVarying output;
    
    // Generate UV from vertex index: (0,0), (2,0), (0,2)
    float2 uv = float2(
        float((vertex_id << 1u) & 2u),
        float(vertex_id & 2u)
    );
    
    // Position: UV [0,2] -> NDC [-1,3]
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = uv;
    
    return output;
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    // Map UV to complex plane around center with zoom
    float2 c = CENTER + (input.uv - 0.5) * 4.0 / ZOOM;
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
