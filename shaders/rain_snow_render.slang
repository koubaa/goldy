// Rain/Snow particle rendering shader
// Visualizes particles as elongated quads using instancing

struct Particle {
    float2 position;
    float2 velocity;
    float size;
    float _pad1;
    float _pad2;
    float _pad3;
};

struct ParticleParams {
    float is_snow;
    float frame;
    float _pad1;
    float _pad2;
};

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer
struct RenderResources {
    StructuredBuffer<Particle> particles;
    ConstantBuffer<ParticleParams> params;
};
ParameterBlock<RenderResources> gResources;
#define PARTICLES gResources.particles
#define PARAMS gResources.params

#elif defined(__SPIRV__)
// Vulkan: Push constants for indices + global descriptor arrays
import goldy_exp.buffer_indices;
[[vk::binding(0, 0)]] StructuredBuffer<Particle> g_StorageBuffers[];
[[vk::binding(1, 0)]] ConstantBuffer<ParticleParams> g_UniformBuffers[];
#define PARTICLES g_StorageBuffers[getBufferIndex(0)]
#define PARAMS g_UniformBuffers[getBufferIndex(1)]

#elif defined(__DX12__)
// DX12: Root constants + ResourceDescriptorHeap
cbuffer BufferIndices : register(b0, space0) {
    uint particleBufferIndex;
    uint paramsBufferIndex;
};
#define PARTICLES (*DescriptorHandle<StructuredBuffer<Particle>>(uint2(particleBufferIndex, 0)))
#define PARAMS (*DescriptorHandle<ConstantBuffer<ParticleParams>>(uint2(paramsBufferIndex, 0)))

#endif

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

// Quad vertices (two triangles) - elongated vertically for rain streaks
static const float2 quadVerts[6] = {
    float2(-1, -3), float2( 1, -3), float2( 1,  3),
    float2(-1, -3), float2( 1,  3), float2(-1,  3)
};

// Snow quad - more square
static const float2 snowQuadVerts[6] = {
    float2(-1, -1), float2( 1, -1), float2( 1,  1),
    float2(-1, -1), float2( 1,  1), float2(-1,  1)
};

[shader("vertex")]
VSOutput vs_main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    VSOutput output;
    
    Particle p = PARTICLES[instanceID];
    bool is_snow = PARAMS.is_snow > 0.5;
    
    float size = p.size;
    float2 localPos;
    
    if (is_snow) {
        // Snow: square particles
        localPos = snowQuadVerts[vertexID] * size;
    } else {
        // Rain: elongated streaks
        localPos = quadVerts[vertexID] * size;
    }
    
    output.position = float4(p.position + localPos, 0.0, 1.0);
    
    // Color based on mode
    if (is_snow) {
        // White/bluish snow
        float brightness = 0.7 + size * 30.0; // Bigger = brighter
        output.color = float4(0.95 * brightness, 0.95 * brightness, 1.0 * brightness, 0.8);
    } else {
        // Blue rain
        float brightness = 0.6 + size * 50.0;
        output.color = float4(0.5 * brightness, 0.6 * brightness, 0.9 * brightness, 0.7);
    }
    
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    return input.color;
}
