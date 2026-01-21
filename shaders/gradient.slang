// Animated color gradient
// Uses vertex-less fullscreen triangle and time uniform

import goldy_exp;

// Uniform structure
struct TimeUniforms {
    float time;
};

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer support
struct GradientResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<GradientResources> gResources;
#define TIME gResources.uniforms.time

#elif defined(__SPIRV__)
// Vulkan: Use push constants for indices + descriptor array
import goldy_exp.buffer_indices;

// Global descriptor array of uniform buffers
[[vk::binding(1, 0)]] ConstantBuffer<TimeUniforms> g_UniformBuffers[];
#define TIME g_UniformBuffers[getBufferIndex(0)].time

#else
// DX12: Root constants + ResourceDescriptorHeap
cbuffer BufferIndices : register(b0) {
    uint uniformsIndex;
};
#define TIME (*DescriptorHandle<ConstantBuffer<TimeUniforms>>(uint2(uniformsIndex, 0))).time
#endif

[shader("vertex")]
FullscreenVarying vs_main(uint vertex_id : SV_VertexID) {
    // Use vertex-less fullscreen triangle - no vertex buffer needed
    return vs_fullscreen_triangle(vertex_id);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = input.uv;
    float t = TIME;
    
    // Smooth animated gradient using sine waves (no discontinuities)
    float3 color;
    color.r = 0.5 + 0.5 * sin(uv.x * 3.14159 + t * 0.7);
    color.g = 0.5 + 0.5 * sin(uv.y * 3.14159 + t * 0.5 + 2.0);
    color.b = 0.5 + 0.5 * sin((uv.x + uv.y) * 2.0 + t * 0.3 + 4.0);
    
    // Add subtle variation based on position
    color *= 0.8 + 0.2 * sin(uv.x * 6.28 + uv.y * 6.28 + t);
    
    return float4(color, 1.0);
}
