// Classic demoscene tunnel effect
// Uses goldy_exp module for vertex format utilities
//
// ## Bindless Support
//
// This shader demonstrates cross-platform bindless resource access:
//
// - DX12 (__HLSL__): Root constants + ResourceDescriptorHeap
// - Vulkan (__SPIRV__): Push constants + global descriptor arrays (binding 1 = uniforms)
// - Metal: ParameterBlock for argument buffer
// - Traditional mode (no __BINDLESS__): Standard binding syntax

import goldy_exp;

// Uniform structure
struct TimeUniforms {
    float time;
};

#ifdef __BINDLESS__

// NOTE: Check __SPIRV__ FIRST because Slang auto-defines __HLSL__ for all targets
#if defined(__SPIRV__)
// Vulkan: Push constants for indices + global descriptor arrays
// Must match backend layout: binding 1 = uniform buffers array
import goldy_exp.bindless_indices;

[[vk::binding(1, 0)]] ConstantBuffer<TimeUniforms> g_UniformBuffers[];
// Note: Use getBindlessIndex() function, not BINDLESS_INDEX macro (macros don't export from modules)
#define TIME g_UniformBuffers[getBindlessIndex(0)].time

#elif defined(__HLSL__) || defined(__DX12__)
// DX12: Bindless via root constants + ResourceDescriptorHeap
cbuffer BindlessIndices : register(b0) {
    uint uniformsIndex;
};
#define TIME (*DescriptorHandle<ConstantBuffer<TimeUniforms>>(uint2(uniformsIndex, 0))).time

#else
// Metal: Use ParameterBlock for argument buffer
struct TunnelResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<TunnelResources> gResources;
#define TIME gResources.uniforms.time
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
    float2 uv = (input.uv - 0.5) * 2.0;
    float t = TIME;
    
    // Polar coordinates
    float dist = length(uv);
    float angle = atan2(uv.y, uv.x);
    
    // Tunnel coordinates
    float tunnel_depth = 1.0 / (dist + 0.1);
    float tunnel_angle = angle / 3.14159 + t * 0.2;
    
    // Animated texture coordinates
    float tx = tunnel_angle * 4.0;
    float ty = tunnel_depth - t * 2.0;
    
    // Checkerboard pattern
    float checker = floor(tx) + floor(ty);
    bool is_white = fmod(checker, 2.0) == 0.0;
    
    // Color based on depth and checker
    float depth_color = 1.0 - dist * 0.5;
    float3 color;
    
    if (is_white) {
        color = float3(0.8, 0.2, 0.4) * depth_color;
    } else {
        color = float3(0.2, 0.4, 0.8) * depth_color;
    }
    
    // Add glow at center
    color += float3(0.3, 0.5, 1.0) * (1.0 - dist) * (1.0 - dist);
    
    // Fog at edges
    color *= 1.0 - dist * 0.3;
    
    return float4(color, 1.0);
}
