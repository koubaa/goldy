// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Set up dispatch size for path count stage. Leaf shader.

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;

// Slots: 0 = bump (BumpAllocators), 1 = indirect (IndirectCount[N_INDIRECT_STAGES])

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main() {
    StorageBuffer<BumpAllocators> bump = goldy_dyn_scattered<BumpAllocators>(0);
    StorageBuffer<IndirectCount> indirect = goldy_dyn_scattered<IndirectCount>(1);

    if (bump[0].failed != 0) {
        indirect[INDIRECT_STAGE_PATH_COUNT].count_x = 0;
    } else {
        uint lines = bump[0].lines;
        indirect[INDIRECT_STAGE_PATH_COUNT].count_x = (lines + (WG_SIZE - 1)) / WG_SIZE;
    }
    indirect[INDIRECT_STAGE_PATH_COUNT].count_y = 1;
    indirect[INDIRECT_STAGE_PATH_COUNT].count_z = 1;
}
