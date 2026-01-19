// Interactive Mandelbrot fractal explorer
// Uses goldy_exp module for vertex format and color utilities
//
// ## Bindless Support
//
// This shader demonstrates cross-platform bindless resource access:
//
// - Metal (__BINDLESS__ + __METAL__): Uses ParameterBlock for argument buffers
// - Vulkan (__BINDLESS__ + __SPIRV__): Uses push constants + descriptor arrays
// - DX12 (__BINDLESS__ + __HLSL__): Uses root constants + ResourceDescriptorHeap
// - Traditional mode (no __BINDLESS__): Standard binding syntax

import goldy_exp;

// Uniform structure for view parameters
struct ViewUniforms {
    float2 center;
    float zoom;
    float _padding;  // Align to 16 bytes
};

#ifdef __BINDLESS__

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer support
struct MandelbrotResources {
    ConstantBuffer<ViewUniforms> uniforms;
};
ParameterBlock<MandelbrotResources> gResources;
#define CENTER gResources.uniforms.center
#define ZOOM gResources.uniforms.zoom

#elif defined(__SPIRV__)
// Vulkan: Use push constants for indices + descriptor array
import goldy_exp.bindless_indices;

// Global descriptor array of uniform buffers
[[vk::binding(1, 0)]] ConstantBuffer<ViewUniforms> g_UniformBuffers[];
// Note: Use getBindlessIndex() function, not BINDLESS_INDEX macro (macros don't export from modules)
#define CENTER g_UniformBuffers[getBindlessIndex(0)].center
#define ZOOM g_UniformBuffers[getBindlessIndex(0)].zoom

#else
// DX12: Bindless via root constants + ResourceDescriptorHeap
cbuffer BindlessIndices : register(b0) {
    uint uniformsIndex;
};
#define CENTER (*DescriptorHandle<ConstantBuffer<ViewUniforms>>(uint2(uniformsIndex, 0))).center
#define ZOOM (*DescriptorHandle<ConstantBuffer<ViewUniforms>>(uint2(uniformsIndex, 0))).zoom
#endif

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
