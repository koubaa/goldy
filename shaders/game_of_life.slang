// Conway's Game of Life compute shader
// Uses ping-pong buffers: reads from one, writes to the other

import goldy_exp;

static const uint GRID_WIDTH = 128;
static const uint GRID_HEIGHT = 128;

uint getCell(StorageBuffer<uint> state, int x, int y) {
    x = (x + GRID_WIDTH) % GRID_WIDTH;
    y = (y + GRID_HEIGHT) % GRID_HEIGHT;
    return state[y * GRID_WIDTH + x];
}

uint countNeighbors(StorageBuffer<uint> state, int x, int y) {
    uint count = 0;
    count += getCell(state, x - 1, y - 1);
    count += getCell(state, x,     y - 1);
    count += getCell(state, x + 1, y - 1);
    count += getCell(state, x - 1, y);
    count += getCell(state, x + 1, y);
    count += getCell(state, x - 1, y + 1);
    count += getCell(state, x,     y + 1);
    count += getCell(state, x + 1, y + 1);
    return count;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uniform uint read_slot, uniform uint write_slot,
             uint3 id : SV_DispatchThreadID) {
    StorageBuffer<uint> CURRENT_STATE = goldy_scattered<uint>(read_slot);
    StorageBuffer<uint> NEXT_STATE = goldy_scattered<uint>(write_slot);

    if (id.x >= GRID_WIDTH || id.y >= GRID_HEIGHT) return;
    
    uint idx = id.y * GRID_WIDTH + id.x;
    uint cell = CURRENT_STATE[idx];
    uint neighbors = countNeighbors(CURRENT_STATE, int(id.x), int(id.y));
    
    // Conway's rules
    uint newState = 0;
    if (cell == 1) {
        newState = (neighbors == 2 || neighbors == 3) ? 1 : 0;
    } else {
        newState = (neighbors == 3) ? 1 : 0;
    }
    
    NEXT_STATE[idx] = newState;
}
