// Animated metaballs effect
// Uses goldy_exp module for vertex format utilities
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
struct MetaballsResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<MetaballsResources> gResources;
#define TIME gResources.uniforms.time

#elif defined(__SPIRV__)
// Vulkan: Use push constants for indices + descriptor array
import goldy_exp.bindless_indices;

// Global descriptor array of uniform buffers
[[vk::binding(1, 0)]] ConstantBuffer<TimeUniforms> g_UniformBuffers[];
// Note: Use getBindlessIndex() function, not BINDLESS_INDEX macro (macros don't export from modules)
#define TIME g_UniformBuffers[getBindlessIndex(0)].time

#else
// DX12: Bindless via root constants + ResourceDescriptorHeap
cbuffer BindlessIndices : register(b0) {
    uint uniformsIndex;
};
#define TIME (*DescriptorHandle<ConstantBuffer<TimeUniforms>>(uint2(uniformsIndex, 0))).time
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

float metaball(float2 p, float2 center, float radius) {
    float d = distance(p, center);
    return radius / (d * d + 0.001);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = (input.uv - 0.5) * 2.0;
    float t = TIME;
    
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
        // Inside blob - gradient based on intensity
        float intensity = (v - threshold) / 2.0;
        float r = 0.2 + intensity * 0.3;
        float g = 0.5 + intensity * 0.4;
        float b = 0.8 + intensity * 0.2;
        return float4(r, g, b, 1.0);
    } else {
        // Outside - dark background with glow
        float glow = v * 0.3;
        return float4(glow * 0.2, glow * 0.3, glow * 0.5, 1.0);
    }
}
