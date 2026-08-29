// Game of Life rendering shader
// Renders the grid as a fullscreen quad with cell colors

import goldy_exp;

static const uint GRID_WIDTH = 128;
static const uint GRID_HEIGHT = 128;

struct VSOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

static const float2 positions[3] = {
    float2(-1, -1),
    float2( 3, -1),
    float2(-1,  3)
};

static const float2 uvs[3] = {
    float2(0, 1),
    float2(2, 1),
    float2(0, -1)
};

[goldy_vertex]
VSOutput vs_main(VertexId vertexID) {
    VSOutput output;
    output.position = float4(positions[vertexID.value], 0.0, 1.0);
    output.uv = uvs[vertexID.value];
    return output;
}

[goldy_fragment]
float4 fs_main(Scattered<uint> CELLS, VSOutput input) : SV_Target {
    float2 uv = input.uv;
    uv.y = 1.0 - uv.y;
    
    int x = int(uv.x * GRID_WIDTH);
    int y = int(uv.y * GRID_HEIGHT);
    
    x = clamp(x, 0, int(GRID_WIDTH) - 1);
    y = clamp(y, 0, int(GRID_HEIGHT) - 1);
    
    uint idx = y * GRID_WIDTH + x;
    uint cell = CELLS[idx];

    // ~1px lines on every cell edge. A fixed 0.05-of-cell threshold is often
    // thinner than a pixel, so many lines vanish or look chopped.
    float2 gridUV = float2(uv.x * float(GRID_WIDTH), uv.y * float(GRID_HEIGHT));
    float2 cellUV = frac(gridUV);
    float2 fw = max(fwidth(gridUV), float2(1e-5, 1e-5));
    float2 edge = min(cellUV, 1.0 - cellUV);
    float2 line = 1.0 - smoothstep(float2(0.0, 0.0), fw, edge);
    float gridLine = saturate(max(line.x, line.y)) * 0.2;
    
    if (cell == 1) {
        float3 alive = float3(0.2, 0.9, 0.3);
        return float4(alive + gridLine, 1.0);
    } else {
        float3 dead = float3(0.05, 0.08, 0.1);
        return float4(dead + gridLine, 1.0);
    }
}
