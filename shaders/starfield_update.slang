// Starfield compute shader - updates star z-positions for 3D flying effect

import goldy_exp;

static const float STAR_TYPE_NORMAL = 0.0;
static const float STAR_TYPE_GALAXY = 1.0;
static const float STAR_TYPE_QUASAR = 2.0;
static const float STAR_TYPE_WHITE_DWARF = 3.0;

struct Star {
    float x;
    float y;
    float z;
    float star_type;
};

struct StarfieldParams {
    float speed;
    float frame;
    float _pad1;
    float _pad2;
};

// Dynamic buffer selection - Rust: set_push_constants(&[stars_slot, params_slot])
#if defined(__METAL__) && !defined(__METAL_BINDLESS__)
// Metal Tier 1 fallback: traditional bindings
RWStructuredBuffer<Star> stars : register(u0);
ConstantBuffer<StarfieldParams> params : register(b0);
#define STARS stars
#define PARAMS params
#else
// All other platforms: use unified dynamic access
#define STARS goldy_dyn_scattered<Star>(0)
#define PARAMS goldy_dyn_broadcast<StarfieldParams>(1)
#endif

static const uint NUM_STARS = 500;

float hash(float p) {
    return frac(sin(p * 127.1) * 43758.5453);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    
    if (idx >= NUM_STARS) return;
    
    Star s = STARS[idx];
    
    s.z -= PARAMS.speed;
    
    if (s.z <= 0.01) {
        float seed = float(idx) + PARAMS.frame * 0.01;
        s.x = (hash(seed) - 0.5) * 2.0;
        s.y = (hash(seed + 100.0) - 0.5) * 2.0;
        s.z = 1.0;
        
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
