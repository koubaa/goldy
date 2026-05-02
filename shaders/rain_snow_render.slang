// Rain/Snow particle rendering shader
// Visualizes particles as elongated quads using instancing

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

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

static const float2 quadVerts[6] = {
    float2(-1, -3), float2( 1, -3), float2( 1,  3),
    float2(-1, -3), float2( 1,  3), float2(-1,  3)
};

static const float2 snowQuadVerts[6] = {
    float2(-1, -1), float2( 1, -1), float2( 1,  1),
    float2(-1, -1), float2( 1,  1), float2(-1,  1)
};

[shader("vertex")]
VSOutput vs_main(uniform uint particles_slot, uniform uint params_slot,
                 uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    StorageBuffer<Particle> PARTICLES = goldy_scattered<Particle>(particles_slot);
    ParticleParams PARAMS = goldy_broadcast<ParticleParams>(params_slot);
    VSOutput output;
    
    Particle p = PARTICLES[instanceID];
    bool is_snow = PARAMS.is_snow > 0.5;
    
    float size = p.size;
    float2 localPos;
    
    if (is_snow) {
        localPos = snowQuadVerts[vertexID] * size;
    } else {
        localPos = quadVerts[vertexID] * size;
    }
    
    output.position = float4(p.position + localPos, 0.0, 1.0);
    
    if (is_snow) {
        float brightness = 0.7 + size * 30.0;
        output.color = float4(0.95 * brightness, 0.95 * brightness, 1.0 * brightness, 0.8);
    } else {
        float brightness = 0.6 + size * 50.0;
        output.color = float4(0.5 * brightness, 0.6 * brightness, 0.9 * brightness, 0.7);
    }
    
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    return input.color;
}
