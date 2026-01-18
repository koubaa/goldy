// Starfield compute shader - updates star z-positions for 3D flying effect
// Stars move toward the camera (z decreases), respawn when they pass

// Star types
static const float STAR_TYPE_NORMAL = 0.0;
static const float STAR_TYPE_GALAXY = 1.0;
static const float STAR_TYPE_QUASAR = 2.0;
static const float STAR_TYPE_WHITE_DWARF = 3.0;

struct Star {
    float x;
    float y;
    float z;
    float star_type; // 0=normal, 1=galaxy, 2=quasar, 3=white dwarf
};

#ifdef __BINDLESS__
#ifdef __DX12__
cbuffer BindlessIndices : register(b0, space0) {
    uint starBufferIndex;
    uint paramsBufferIndex;
};

struct StarfieldParams {
    float speed;
    float frame;
    float _pad1;
    float _pad2;
};

#define STARS (*DescriptorHandle<RWStructuredBuffer<Star>>(uint2(starBufferIndex, 0)))
#define PARAMS (*DescriptorHandle<ConstantBuffer<StarfieldParams>>(uint2(paramsBufferIndex, 0)))
#else
// Metal/Vulkan bindless
struct StarfieldParams {
    float speed;
    float frame;
    float _pad1;
    float _pad2;
};

struct ComputeResources {
    RWStructuredBuffer<Star> stars;
    ConstantBuffer<StarfieldParams> params;
};

ParameterBlock<ComputeResources> gResources;

#define STARS gResources.stars
#define PARAMS gResources.params
#endif
#else
// Traditional binding mode
struct StarfieldParams {
    float speed;
    float frame;
    float _pad1;
    float _pad2;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Star> stars;
[[vk::binding(1, 0)]] ConstantBuffer<StarfieldParams> params;

#define STARS stars
#define PARAMS params
#endif

static const uint NUM_STARS = 500;

// Hash function for pseudo-random number generation
float hash(float p) {
    return frac(sin(p * 127.1) * 43758.5453);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    
    if (idx >= NUM_STARS) return;
    
    Star s = STARS[idx];
    
    // Move star toward camera (decrease z) - speed is per-frame like original
    s.z -= PARAMS.speed;
    
    // Respawn star when it passes the camera
    if (s.z <= 0.01) {
        // Use frame counter + index for varying randomness
        float seed = float(idx) + PARAMS.frame * 0.01;
        s.x = (hash(seed) - 0.5) * 2.0;
        s.y = (hash(seed + 100.0) - 0.5) * 2.0;
        s.z = 1.0;
        
        // Randomly assign new star type with probabilities:
        // 70% normal, 15% galaxy, 5% quasar, 10% white dwarf
        float type_roll = hash(seed + 200.0);
        if (type_roll < 0.70) {
            s.star_type = STAR_TYPE_NORMAL;
        } else if (type_roll < 0.85) {
            s.star_type = STAR_TYPE_GALAXY;
        } else if (type_roll < 0.90) {
            s.star_type = STAR_TYPE_QUASAR;
        } else {
            s.star_type = STAR_TYPE_WHITE_DWARF;
        }
    }
    
    STARS[idx] = s;
}
