// Bouncing lines rendering shader
// Visualizes lines using instancing with LineList topology

import goldy_exp;

struct Line {
    float2 p1;
    float2 v1;
    float2 p2;
    float2 v2;
    uint color_index;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

// Dynamic buffer selection - Rust: set_push_constants(&[lines_slot])
#if defined(__METAL__) && !defined(__METAL_BINDLESS__)
// Metal Tier 1 fallback: traditional bindings
StructuredBuffer<Line> lines : register(t0);
#define LINES lines
#else
// All other platforms: use unified dynamic access
#define LINES goldy_dyn_scattered<Line>(0)
#endif

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

static const float4 colors[6] = {
    float4(1.0, 0.0, 0.0, 1.0),  // Red
    float4(0.0, 1.0, 0.0, 1.0),  // Green
    float4(0.0, 0.0, 1.0, 1.0),  // Blue
    float4(1.0, 1.0, 0.0, 1.0),  // Yellow
    float4(1.0, 0.0, 1.0, 1.0),  // Magenta
    float4(0.0, 1.0, 1.0, 1.0),  // Cyan
};

[shader("vertex")]
VSOutput vs_main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    VSOutput output;
    
    Line line = LINES[instanceID];
    
    float2 pos = (vertexID == 0) ? line.p1 : line.p2;
    
    output.position = float4(pos, 0.0, 1.0);
    output.color = colors[line.color_index % 6];
    
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    return input.color;
}
