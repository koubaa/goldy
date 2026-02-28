// Conway's Game of Life compute shader
// Uses ping-pong buffers: reads from one, writes to the other

import goldy_exp;

static const uint GRID_WIDTH = 128;
static const uint GRID_HEIGHT = 128;

// Dynamic buffer selection for ping-pong pattern
// Rust side: set_push_constants(&[read_buffer_slot, write_buffer_slot])
#define CURRENT_STATE goldy_dyn_scattered<uint>(0)
#define NEXT_STATE goldy_dyn_scattered<uint>(1)

// Get cell state (1 = alive, 0 = dead)
uint getCell(int x, int y) {
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
    
    // Conway's rules
    uint newState = 0;
    if (cell == 1) {
        newState = (neighbors == 2 || neighbors == 3) ? 1 : 0;
    } else {
        newState = (neighbors == 3) ? 1 : 0;
    }
    
    NEXT_STATE[idx] = newState;
}
