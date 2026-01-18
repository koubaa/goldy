// Particle simulation compute shader
// Updates particle positions and velocities

struct Particle {
    float2 position;
    float2 velocity;
};

#ifdef __BINDLESS__
#ifdef __DX12__
// DX12 bindless: Use Slang's DescriptorHandle which lowers to ResourceDescriptorHeap[idx]
// Root constants provide the descriptor index at b0
cbuffer BindlessIndices : register(b0, space0) {
    uint particleBufferIndex;
};

// Access via DescriptorHandle - Slang will lower to ResourceDescriptorHeap[idx] for DXIL
// Dereference with * operator to get the buffer
#define PARTICLES (*DescriptorHandle<RWStructuredBuffer<Particle>>(uint2(particleBufferIndex, 0)))
#else
// Metal/Vulkan bindless: Use ParameterBlock for argument buffer support
struct ComputeResources {
    RWStructuredBuffer<Particle> particles;
};

ParameterBlock<ComputeResources> gResources;

#define PARTICLES gResources.particles
#endif
#else
// Traditional binding mode
[[vk::binding(0, 0)]] RWStructuredBuffer<Particle> particles;

#define PARTICLES particles
#endif

// Use push constants or hardcoded values for simplicity
static const float deltaTime = 0.016; // ~60fps
static const float2 gravity = float2(0.0, -0.3);
static const float damping = 0.998;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    
    // Bounds check (assume max 65536 particles)
    if (idx >= 65536) return;
    
    Particle p = PARTICLES[idx];
    
    // Skip if particle is at origin with no velocity (uninitialized)
    if (p.position.x == 0.0 && p.position.y == 0.0 && 
        p.velocity.x == 0.0 && p.velocity.y == 0.0) {
        return;
    }
    
    // Apply gravity
    p.velocity += gravity * deltaTime;
    
    // Apply damping
    p.velocity *= damping;
    
    // Update position
    p.position += p.velocity * deltaTime;
    
    // Bounce off walls (normalized -1 to 1 space)
    if (p.position.x < -0.95) {
        p.velocity.x = abs(p.velocity.x) * 0.8;
        p.position.x = -0.95;
    }
    if (p.position.x > 0.95) {
        p.velocity.x = -abs(p.velocity.x) * 0.8;
        p.position.x = 0.95;
    }
    if (p.position.y < -0.95) {
        p.velocity.y = abs(p.velocity.y) * 0.8;
        p.position.y = -0.95;
    }
    if (p.position.y > 0.95) {
        p.velocity.y = -abs(p.velocity.y) * 0.8;
        p.position.y = 0.95;
    }
    
    PARTICLES[idx] = p;
}
