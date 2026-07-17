//! Game-of-life initial state helpers shared by scheme screenshot fixtures.

pub const GOL_GRID_WIDTH: u32 = 128;
pub const GOL_GRID_HEIGHT: u32 = 128;
pub const GOL_CELL_COUNT: u32 = GOL_GRID_WIDTH * GOL_GRID_HEIGHT;

pub fn create_gol_initial_state() -> Vec<u32> {
    let mut cells = vec![0u32; GOL_CELL_COUNT as usize];

    let gun = [
        (1, 5),
        (1, 6),
        (2, 5),
        (2, 6),
        (11, 5),
        (11, 6),
        (11, 7),
        (12, 4),
        (12, 8),
        (13, 3),
        (13, 9),
        (14, 3),
        (14, 9),
        (15, 6),
        (16, 4),
        (16, 8),
        (17, 5),
        (17, 6),
        (17, 7),
        (18, 6),
        (21, 3),
        (21, 4),
        (21, 5),
        (22, 3),
        (22, 4),
        (22, 5),
        (23, 2),
        (23, 6),
        (25, 1),
        (25, 2),
        (25, 6),
        (25, 7),
        (35, 3),
        (35, 4),
        (36, 3),
        (36, 4),
    ];

    let offset_x = 10;
    let offset_y = 10;
    for (x, y) in gun.iter() {
        let px = (x + offset_x) as u32;
        let py = (y + offset_y) as u32;
        if px < GOL_GRID_WIDTH && py < GOL_GRID_HEIGHT {
            cells[(py * GOL_GRID_WIDTH + px) as usize] = 1;
        }
    }

    let seed = 42u64;
    let mut rng = seed;
    for y in 60..100 {
        for x in 60..100 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32).is_multiple_of(4) {
                cells[(y * GOL_GRID_WIDTH + x) as usize] = 1;
            }
        }
    }

    cells
}
