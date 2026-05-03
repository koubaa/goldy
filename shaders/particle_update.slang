// Particle simulation compute shader
// Updates particle positions and velocities

import goldy_exp;

struct Particle {
    float2 position;
    float2 velocity;
};

static const float deltaTime = 0.016;
static const float2 gravity = float2(0.0, -0.3);
static const float damping = 0.998;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<Particle> PARTICLES, ThreadId id) {
    uint idx = id.x;
    
    if (idx >= 65536) return;
    
    Particle p = PARTICLES[idx];
    
    if (p.position.x == 0.0 && p.position.y == 0.0 && 
        p.velocity.x == 0.0 && p.velocity.y == 0.0) {
        return;
    }
    
    p.velocity += gravity * deltaTime;
    p.velocity *= damping;
    p.position += p.velocity * deltaTime;
    
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
