// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Set up dispatch size for path tiling stage. Leaf shader.

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;

// Slots: 0 = bump (BumpAllocators), 1 = indirect (IndirectCount[N_INDIRECT_STAGES]), 2 = ptcl (uint array)

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main() {
    StorageBuffer<BumpAllocators> bump = goldy_dyn_scattered<BumpAllocators>(0);
    StorageBuffer<IndirectCount> indirect = goldy_dyn_scattered<IndirectCount>(1);
    StorageBuffer<uint> ptcl = goldy_dyn_scattered<uint>(2);

    if (bump[0].failed != 0) {
        indirect[INDIRECT_STAGE_PATH_TILING].count_x = 0;
        ptcl[0] = 0xFFFFFFFF;  // signal fine rasterizer that failure happened
    } else {
        uint segments = bump[0].seg_counts;
        indirect[INDIRECT_STAGE_PATH_TILING].count_x = (segments + (WG_SIZE - 1)) / WG_SIZE;
    }
    indirect[INDIRECT_STAGE_PATH_TILING].count_y = 1;
    indirect[INDIRECT_STAGE_PATH_TILING].count_z = 1;
}
