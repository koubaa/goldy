// Instancing compute shader - updates rotation and color for each quad instance
// Demonstrates the Goldy-native pattern: compute shader updates data, graphics shader renders

import goldy_exp;

// Instance data matching goldy::Instance2D
struct QuadInstance {
    float2 position;
    float rotation;
    float scale;
    float4 color;
};

// Uniforms for animation
struct AnimParams {
    float time;
    float delta_time;
    uint total_instances;
    uint _pad;
};

// ============================================================================
// Platform-specific resource binding  
// ============================================================================

#if defined(__METAL__)
struct ComputeResources {
    RWStructuredBuffer<QuadInstance> instances;
    ConstantBuffer<AnimParams> params;
};
ParameterBlock<ComputeResources> gResources;
#define INSTANCES gResources.instances
#define PARAMS gResources.params

#elif defined(__SPIRV__)
import goldy_exp.buffer_indices;
[[vk::binding(0, 0)]] RWStructuredBuffer<QuadInstance> g_Storage[];
[[vk::binding(1, 0)]] ConstantBuffer<AnimParams> g_Uniforms[];
#define INSTANCES g_Storage[getBufferIndex(0)]
#define PARAMS g_Uniforms[getBufferIndex(1)]

#elif defined(__DX12__)
// DX12: Root constants + ResourceDescriptorHeap (matching proven pattern)
cbuffer BufferIndices : register(b0, space0) {
    uint instanceBufferIndex;
    uint paramsBufferIndex;
};
#define INSTANCES (*DescriptorHandle<RWStructuredBuffer<QuadInstance>>(uint2(instanceBufferIndex, 0)))
#define PARAMS (*DescriptorHandle<ConstantBuffer<AnimParams>>(uint2(paramsBufferIndex, 0)))

#endif

// ============================================================================
// Compute shader
// ============================================================================

float3 hsv_to_rgb(float h, float s, float v) {
    float4 K = float4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    float3 p = abs(frac(h.xxx + K.xyz) * 6.0 - K.www);
    return v * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= PARAMS.total_instances) return;
    
    QuadInstance inst = INSTANCES[idx];
    
    float phase = (float(idx) / float(PARAMS.total_instances)) * 6.28318;
    inst.rotation = PARAMS.time * 2.0 + phase;
    
    float hue = frac(float(idx) / float(PARAMS.total_instances) + PARAMS.time * 0.1);
    inst.color = float4(hsv_to_rgb(hue, 0.8, 0.9), 1.0);
    
    INSTANCES[idx] = inst;
}
