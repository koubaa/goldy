// Rain/Snow particle compute shader
// Updates particle positions with mode-dependent physics

import goldy_exp;

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

// Dynamic buffer selection - Rust: set_push_constants(&[particles_slot, params_slot])
// Note: Push constant resources are declared in goldy_exp module; GOLDY_DECLARE_DYNAMIC is
// no longer needed (the resources are auto-declared per platform in access.slang).
#if defined(__METAL__) && !defined(__METAL_BINDLESS__)
// Metal Tier 1 fallback: traditional bindings
RWStructuredBuffer<Particle> particles : register(u0);
ConstantBuffer<ParticleParams> params : register(b0);
#define PARTICLES particles
#define PARAMS params
#else
// All other platforms: use unified dynamic access
#define PARTICLES goldy_dyn_scattered<Particle>(0)
#define PARAMS goldy_dyn_broadcast<ParticleParams>(1)
#endif

static const uint NUM_PARTICLES = 1000;

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
    
    p.position += p.velocity;
    
    if (is_snow) {
        float seed = float(idx) + PARAMS.frame * 0.1;
        p.velocity.x += (hash(seed) - 0.5) * 0.001;
        p.velocity.x = clamp(p.velocity.x, -0.01, 0.01);
    }
    
    if (p.position.y < -1.2 || p.position.x < -1.2 || p.position.x > 1.2) {
        float seed = float(idx) + PARAMS.frame * 0.01;
        
        p.position.x = hash(seed) * 2.0 - 1.0;
        p.position.y = 1.0 + hash(seed + 100.0) * 0.5;
        
        if (is_snow) {
            p.velocity.x = (hash(seed + 200.0) - 0.5) * 0.005;
            p.velocity.y = -(0.002 + hash(seed + 300.0) * 0.005);
            p.size = 0.003 + hash(seed + 400.0) * 0.008;
        } else {
            p.velocity.x = (hash(seed + 200.0) - 0.5) * 0.002;
            p.velocity.y = -(0.01 + hash(seed + 300.0) * 0.02);
            p.size = 0.002 + hash(seed + 400.0) * 0.003;
        }
    }
    
    PARTICLES[idx] = p;
}
