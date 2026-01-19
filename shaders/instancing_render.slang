// Instancing render shader - renders quads from instance buffer
// Demonstrates the Goldy-native pattern: no vertex buffer, reads from storage buffer

import goldy_exp;

// Instance data matching goldy::Instance2D
struct QuadInstance {
    float2 position;
    float rotation;
    float scale;
    float4 color;
};

// ============================================================================
// Platform-specific resource binding
// ============================================================================

#if defined(__METAL__)
struct RenderResources {
    StructuredBuffer<QuadInstance> instances;
};
ParameterBlock<RenderResources> gResources;
#define INSTANCES gResources.instances

#elif defined(__SPIRV__)
import goldy_exp.buffer_indices;
[[vk::binding(0, 0)]] StructuredBuffer<QuadInstance> g_Storage[];
#define INSTANCES g_Storage[getBufferIndex(0)]

#elif defined(__DX12__)
// DX12: Root constants + ResourceDescriptorHeap (matching proven pattern)
cbuffer BufferIndices : register(b0, space0) {
    uint instanceBufferIndex;
};
#define INSTANCES (*DescriptorHandle<StructuredBuffer<QuadInstance>>(uint2(instanceBufferIndex, 0)))

#endif

// ============================================================================
// Shaders
// ============================================================================

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VSOutput vs_main(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID) {
    VSOutput output;
    
    QuadInstance inst = INSTANCES[instance_id];
    float2 world_pos = quad_position_rotated(vertex_id, inst.position, inst.scale, inst.rotation);
    
    output.position = float4(world_pos, 0.0, 1.0);
    output.color = inst.color;
    
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    return input.color;
}
