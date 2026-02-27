// Bouncing lines compute shader
// Updates line endpoint positions with wall bouncing physics

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
RWStructuredBuffer<Line> lines : register(u0);
#define LINES lines
#else
// All other platforms: use unified dynamic access
#define LINES goldy_dyn_scattered<Line>(0)
#endif

static const uint NUM_LINES = 20;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    
    if (idx >= NUM_LINES) return;
    
    Line line = LINES[idx];
    
    // Update positions
    line.p1 += line.v1;
    line.p2 += line.v2;
    
    // Bounce endpoint 1 off walls
    if (line.p1.x < -1.0 || line.p1.x > 1.0) {
        line.v1.x = -line.v1.x;
        line.p1.x = clamp(line.p1.x, -1.0, 1.0);
    }
    if (line.p1.y < -1.0 || line.p1.y > 1.0) {
        line.v1.y = -line.v1.y;
        line.p1.y = clamp(line.p1.y, -1.0, 1.0);
    }
    
    // Bounce endpoint 2 off walls
    if (line.p2.x < -1.0 || line.p2.x > 1.0) {
        line.v2.x = -line.v2.x;
        line.p2.x = clamp(line.p2.x, -1.0, 1.0);
    }
    if (line.p2.y < -1.0 || line.p2.y > 1.0) {
        line.v2.y = -line.v2.y;
        line.p2.y = clamp(line.p2.y, -1.0, 1.0);
    }
    
    LINES[idx] = line;
}
