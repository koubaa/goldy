// Classic demoscene plasma effect
// Uses UV coordinates and time for animated plasma pattern

struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
    float time : TEXCOORD1;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float time : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    output.time = input.time;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = input.uv * 4.0;
    float t = input.time;
    
    // Classic plasma formula
    float v = sin(uv.x + t);
    v += sin(uv.y + t);
    v += sin(uv.x + uv.y + t);
    
    float cx = uv.x + 0.5 * sin(t / 3.0);
    float cy = uv.y + 0.5 * cos(t / 2.0);
    v += sin(sqrt(cx * cx + cy * cy + 1.0) + t);
    
    v = v / 2.0;
    
    // Color palette
    float r = sin(v * 3.14159);
    float g = sin(v * 3.14159 + 2.094);
    float b = sin(v * 3.14159 + 4.188);
    
    return float4(r * 0.5 + 0.5, g * 0.5 + 0.5, b * 0.5 + 0.5, 1.0);
}

