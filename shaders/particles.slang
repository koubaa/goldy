// Particle rain/snow demo
// Toggle between rain and snow modes via uniform
//
// ## Bindless Support
//
// When __BINDLESS__ is defined (Metal Argument Buffers Tier 2):
// - Uniforms are accessed via ParameterBlock for argument buffer support
// - Traditional binding syntax works as fallback

struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Uniform structure for particle parameters
struct ParticleParams {
    float time;
    float is_snow;  // 0.0 = rain, 1.0 = snow
    float _pad1;
    float _pad2;
};

#ifdef __BINDLESS__
// Bindless mode: Use ParameterBlock for Metal argument buffer support
struct ParticlesResources {
    ConstantBuffer<ParticleParams> params;
};
ParameterBlock<ParticlesResources> gResources;
#define TIME gResources.params.time
#define IS_SNOW gResources.params.is_snow
#else
// Traditional binding mode
[[vk::binding(0, 0)]]
ConstantBuffer<ParticleParams> params;
#define TIME params.time
#define IS_SNOW params.is_snow
#endif

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

float hash(float p) {
    return frac(sin(p * 127.1) * 43758.5453);
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = input.uv;
    bool snow = IS_SNOW > 0.5;
    
    float3 bg_color;
    float3 particle_color;
    float3 trail_color;
    
    if (snow) {
        bg_color = float3(0.02, 0.02, 0.04);
        particle_color = float3(0.9, 0.95, 1.0);
        trail_color = float3(0.5, 0.55, 0.7);
    } else {
        bg_color = float3(0.01, 0.01, 0.02);
        particle_color = float3(0.3, 0.6, 1.0);
        trail_color = float3(0.2, 0.4, 0.8);
    }
    
    float3 color = bg_color;
    
    for (int i = 0; i < 60; i++) {
        float fi = float(i);
        float x = hash(fi);
        
        // Snow drifts horizontally
        if (snow) {
            x += sin(TIME * 0.5 + fi * 0.3) * 0.05;
            x = frac(x);
        }
        
        // Snow is slower
        float base_speed = snow ? 0.12 : 0.3;
        float speed = base_speed + hash(fi + 100.0) * (snow ? 0.08 : 0.4);
        float phase = hash(fi + 200.0) * 10.0;
        
        float t = frac(TIME * speed + phase);
        float particle_y = t;
        
        float2 particle_pos = float2(x, particle_y);
        float dist = length(uv - particle_pos);
        
        // Snow has bigger particles
        float base_size = snow ? 0.004 : 0.002;
        float size = base_size + hash(fi + 300.0) * (snow ? 0.004 : 0.002);
        float brightness = size / (dist + 0.001);
        float fade = smoothstep(0.0, 0.15, t) * smoothstep(1.0, 0.85, t);
        
        color += particle_color * brightness * fade * 0.06;
        
        // Trail (shorter for snow)
        float trail_len = snow ? 0.03 : 0.08;
        float trail_top = particle_y - trail_len;
        
        if (uv.x > x - 0.003 && uv.x < x + 0.003 && uv.y > trail_top && uv.y < particle_y) {
            float trail_t = (uv.y - trail_top) / trail_len;
            color += trail_color * trail_t * fade * 0.2;
        }
    }
    
    return float4(color, 1.0);
}

