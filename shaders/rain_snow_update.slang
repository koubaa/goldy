// Rain/Snow particle compute shader
// Updates particle positions and velocities with mode-dependent physics

struct Particle {
    float2 position;
    float2 velocity;
    float size;
    float _pad1;
    float _pad2;
    float _pad3;
};

struct ParticleParams {
    float is_snow;    // 0.0 = rain, 1.0 = snow
    float frame;      // For randomness seeding
    float _pad1;
    float _pad2;
};

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer
struct ComputeResources {
    RWStructuredBuffer<Particle> particles;
    ConstantBuffer<ParticleParams> params;
};
ParameterBlock<ComputeResources> gResources;
#define PARTICLES gResources.particles
#define PARAMS gResources.params

#elif defined(__SPIRV__)
// Vulkan: Push constants for indices + global descriptor arrays
import goldy_exp.buffer_indices;
[[vk::binding(0, 0)]] RWStructuredBuffer<Particle> g_StorageBuffers[];
[[vk::binding(1, 0)]] ConstantBuffer<ParticleParams> g_UniformBuffers[];
#define PARTICLES g_StorageBuffers[getBufferIndex(0)]
#define PARAMS g_UniformBuffers[getBufferIndex(1)]

#elif defined(__DX12__)
// DX12: Root constants + ResourceDescriptorHeap
cbuffer BufferIndices : register(b0, space0) {
    uint particleBufferIndex;
    uint paramsBufferIndex;
};
#define PARTICLES (*DescriptorHandle<RWStructuredBuffer<Particle>>(uint2(particleBufferIndex, 0)))
#define PARAMS (*DescriptorHandle<ConstantBuffer<ParticleParams>>(uint2(paramsBufferIndex, 0)))

#endif

static const uint NUM_PARTICLES = 1000;

// Hash function for pseudo-random
float hash(float p) {
    return frac(sin(p * 127.1) * 43758.5453);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    
    if (idx >= NUM_PARTICLES) return;
    
    Particle p = PARTICLES[idx];
    bool is_snow = PARAMS.is_snow > 0.5;
    
    // Apply velocity directly (per-frame, like original CPU version)
    p.position += p.velocity;
    
    // Snow: add horizontal drift
    if (is_snow) {
        float seed = float(idx) + PARAMS.frame * 0.1;
        p.velocity.x += (hash(seed) - 0.5) * 0.001;
        p.velocity.x = clamp(p.velocity.x, -0.01, 0.01);
    }
    
    // Respawn when particle falls below screen or drifts too far
    if (p.position.y < -1.2 || p.position.x < -1.2 || p.position.x > 1.2) {
        float seed = float(idx) + PARAMS.frame * 0.01;
        
        // Reset to top with random x
        p.position.x = hash(seed) * 2.0 - 1.0;
        p.position.y = 1.0 + hash(seed + 100.0) * 0.5;
        
        if (is_snow) {
            // Snow: slower, more horizontal drift
            p.velocity.x = (hash(seed + 200.0) - 0.5) * 0.005;
            p.velocity.y = -(0.002 + hash(seed + 300.0) * 0.005);
            p.size = 0.003 + hash(seed + 400.0) * 0.008;
        } else {
            // Rain: faster, more vertical
            p.velocity.x = (hash(seed + 200.0) - 0.5) * 0.002;
            p.velocity.y = -(0.01 + hash(seed + 300.0) * 0.02);
            p.size = 0.002 + hash(seed + 400.0) * 0.003;
        }
    }
    
    PARTICLES[idx] = p;
}
