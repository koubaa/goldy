// Conway's Game of Life compute shader
// Uses ping-pong buffers: reads from one, writes to the other
//
// ## Bindless Support
//
// When __BINDLESS__ is defined (Metal Argument Buffers Tier 2):
// - Buffers are accessed via ParameterBlock for argument buffer support
// - Traditional binding syntax works as fallback

// Grid dimensions (must match the example)
static const uint GRID_WIDTH = 128;
static const uint GRID_HEIGHT = 128;

#ifdef __BINDLESS__
// Bindless mode: Use ParameterBlock for Metal argument buffer support
struct GameOfLifeResources {
    StructuredBuffer<uint> currentState;
    RWStructuredBuffer<uint> nextState;
};
ParameterBlock<GameOfLifeResources> gResources;
#define CURRENT_STATE gResources.currentState
#define NEXT_STATE gResources.nextState
#else
// Traditional binding mode
[[vk::binding(0, 0)]] StructuredBuffer<uint> currentState;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> nextState;
#define CURRENT_STATE currentState
#define NEXT_STATE nextState
#endif

// Get cell state (1 = alive, 0 = dead)
uint getCell(int x, int y) {
    // Wrap around edges (toroidal grid)
    x = (x + GRID_WIDTH) % GRID_WIDTH;
    y = (y + GRID_HEIGHT) % GRID_HEIGHT;
    return CURRENT_STATE[y * GRID_WIDTH + x];
}

// Count living neighbors
uint countNeighbors(int x, int y) {
    uint count = 0;
    count += getCell(x - 1, y - 1);
    count += getCell(x,     y - 1);
    count += getCell(x + 1, y - 1);
    count += getCell(x - 1, y);
    count += getCell(x + 1, y);
    count += getCell(x - 1, y + 1);
    count += getCell(x,     y + 1);
    count += getCell(x + 1, y + 1);
    return count;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= GRID_WIDTH || id.y >= GRID_HEIGHT) return;
    
    uint idx = id.y * GRID_WIDTH + id.x;
    uint cell = CURRENT_STATE[idx];
    uint neighbors = countNeighbors(int(id.x), int(id.y));
    
    // Conway's rules:
    // - Living cell with 2-3 neighbors survives
    // - Dead cell with exactly 3 neighbors becomes alive
    // - All other cells die or stay dead
    uint newState = 0;
    if (cell == 1) {
        // Living cell
        newState = (neighbors == 2 || neighbors == 3) ? 1 : 0;
    } else {
        // Dead cell
        newState = (neighbors == 3) ? 1 : 0;
    }
    
    NEXT_STATE[idx] = newState;
}

