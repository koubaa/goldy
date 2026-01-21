// Classic demoscene plasma effect
// Uses vertex-less fullscreen triangle and time uniform

import goldy_exp;

struct TimeUniforms {
    float time;
};

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer support
struct PlasmaResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<PlasmaResources> gResources;
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
    return vs_fullscreen_triangle(vertex_id);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = scale_uv(input.uv, 4.0);
    float t = TIME;
    
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
