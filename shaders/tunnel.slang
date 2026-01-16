// Classic demoscene tunnel effect
// Uses uniform buffer for time animation
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

// Uniform structure
struct TimeUniforms {
    float time;
};

#ifdef __BINDLESS__
// Bindless mode: Use ParameterBlock for Metal argument buffer support
struct TunnelResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<TunnelResources> gResources;
#define TIME gResources.uniforms.time
#else
// Traditional binding mode
[[vk::binding(0, 0)]]
ConstantBuffer<TimeUniforms> uniforms;
#define TIME uniforms.time
#endif

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = (input.uv - 0.5) * 2.0;
    float t = TIME;
    
    // Polar coordinates
    float dist = length(uv);
    float angle = atan2(uv.y, uv.x);
    
    // Tunnel coordinates
    float tunnel_depth = 1.0 / (dist + 0.1);
    float tunnel_angle = angle / 3.14159 + t * 0.2;
    
    // Animated texture coordinates
    float tx = tunnel_angle * 4.0;
    float ty = tunnel_depth - t * 2.0;
    
    // Checkerboard pattern
    float checker = floor(tx) + floor(ty);
    bool is_white = fmod(checker, 2.0) == 0.0;
    
    // Color based on depth and checker
    float depth_color = 1.0 - dist * 0.5;
    float3 color;
    
    if (is_white) {
        color = float3(0.8, 0.2, 0.4) * depth_color;
    } else {
        color = float3(0.2, 0.4, 0.8) * depth_color;
    }
    
    // Add glow at center
    color += float3(0.3, 0.5, 1.0) * (1.0 - dist) * (1.0 - dist);
    
    // Fog at edges
    color *= 1.0 - dist * 0.3;
    
    return float4(color, 1.0);
}
