// Classic demoscene plasma effect
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

// Uniform structure
struct TimeUniforms {
    float time;
};

#ifdef __BINDLESS__

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer support
struct PlasmaResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<PlasmaResources> gResources;
#define TIME gResources.uniforms.time

#elif defined(__SPIRV__)
// Vulkan: Use push constants for indices + descriptor array
import goldy_exp.bindless_indices;

// Global descriptor array of uniform buffers
[[vk::binding(1, 0)]] ConstantBuffer<TimeUniforms> g_UniformBuffers[];
#define TIME g_UniformBuffers[BINDLESS_INDEX(0)].time

#else
// DX12: Use root constants for indices (ResourceDescriptorHeap for SM 6.6+)
// For simplicity, fall back to traditional binding in this example
// Full DX12 SM 6.6 bindless would use ResourceDescriptorHeap[index]
[[vk::binding(0, 0)]]
ConstantBuffer<TimeUniforms> uniforms;
#define TIME uniforms.time
#endif

#else
// Traditional binding mode
[[vk::binding(0, 0)]]
ConstantBuffer<TimeUniforms> uniforms;
#define TIME uniforms.time
#endif

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
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
    
    // Use rainbow palette from goldy module
    return float4(rainbow(v), 1.0);
}
