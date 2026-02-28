// Particle rendering shader
// Visualizes particles as colored quads using instancing

import goldy_exp;

struct Particle {
    float2 position;
    float2 velocity;
};

// Dynamic buffer selection - Rust: set_push_constants(&[particles_slot])
#define PARTICLES goldy_dyn_scattered<Particle>(0)

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

static const float2 quadVerts[6] = {
    float2(-1, -1), float2( 1, -1), float2( 1,  1),
    float2(-1, -1), float2( 1,  1), float2(-1,  1)
};

[shader("vertex")]
VSOutput vs_main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    VSOutput output;
    
    Particle p = PARTICLES[instanceID];
    
    float size = 0.015;
    float2 localPos = quadVerts[vertexID] * size;
    
    output.position = float4(p.position + localPos, 0.0, 1.0);
    
    float speed = length(p.velocity);
    float angle = atan2(p.velocity.y, p.velocity.x);
    
    float h = (angle + 3.14159) / (2.0 * 3.14159);
    float3 rgb = abs(h * 6.0 - float3(3, 2, 4)) * float3(1, -1, -1) + float3(-1, 2, 2);
    rgb = clamp(rgb, 0.0, 1.0);
    
    float brightness = clamp(speed * 2.0 + 0.3, 0.3, 1.0);
    output.color = float4(rgb * brightness, 1.0);
    
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    return input.color;
}
