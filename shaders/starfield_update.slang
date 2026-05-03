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

static const uint NUM_STARS = 500;

float hash(float p) {
    return frac(sin(p * 127.1) * 43758.5453);
}

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<Star> STARS, StarfieldParams params, ThreadId id) {
    uint idx = id.x;
    
    if (idx >= NUM_STARS) return;
    
    Star s = STARS[idx];
    
    s.z -= params.speed;
    
    if (s.z <= 0.01) {
        float seed = float(idx) + params.frame * 0.01;
        s.x = (hash(seed) - 0.5) * 2.0;
        s.y = (hash(seed + 100.0) - 0.5) * 2.0;
        s.z = 1.0;
        
        float type_roll = hash(seed + 200.0);
        if (type_roll < 0.6) {
            s.star_type = STAR_TYPE_NORMAL;
        } else if (type_roll < 0.8) {
            s.star_type = STAR_TYPE_GALAXY;
        } else if (type_roll < 0.95) {
            s.star_type = STAR_TYPE_QUASAR;
        } else {
            s.star_type = STAR_TYPE_WHITE_DWARF;
        }
    }
    
    STARS[idx] = s;
}
