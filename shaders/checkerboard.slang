// Animated checkerboard pattern
// Uses vertex-less fullscreen triangle and time uniform

import goldy_exp;

// Uniform structure
struct TimeUniforms {
    float time;
};

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer support
struct CheckerResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<CheckerResources> gResources;
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
    float t = TIME;
    
    // Animated wave distortion
    float2 uv = input.uv;
    uv.x += sin(uv.y * 10.0 + t * 2.0) * 0.02;
    uv.y += cos(uv.x * 10.0 + t * 1.5) * 0.02;
    
    // Scale for checker pattern
    float scale = 8.0;
    float2 checker = floor(uv * scale);
    bool is_white = fmod(checker.x + checker.y, 2.0) == 0.0;
    
    // Animate colors
    float3 color1 = float3(
        0.2 + 0.1 * sin(t),
        0.1 + 0.1 * cos(t * 1.3),
        0.3 + 0.1 * sin(t * 0.7)
    );
    float3 color2 = float3(
        0.9 + 0.1 * cos(t * 0.8),
        0.85 + 0.1 * sin(t * 1.1),
        0.8 + 0.1 * cos(t)
    );
    
    // Add subtle gradient
    color1 += uv.y * 0.1;
    color2 -= uv.y * 0.1;
    
    if (is_white) {
        return float4(color2, 1.0);
    } else {
        return float4(color1, 1.0);
    }
}
