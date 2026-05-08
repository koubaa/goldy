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

[goldy_vertex]
VSOutput vs_main(Scattered<Line> LINES, VertexId vertexID, InstanceId instanceID) {
    VSOutput output;
    
    Line line = LINES[instanceID.value];
    
    float2 pos = (vertexID.value == 0) ? line.p1 : line.p2;
    
    output.position = float4(pos, 0.0, 1.0);
    output.color = colors[line.color_index % 6];
    
    return output;
}

[goldy_fragment]
float4 fs_main(VSOutput input) : SV_Target {
    return input.color;
}
