// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Phase 1: Copy uploaded workgroup counts into the indirect dispatch buffer.
// Runs before all other stages. Slots 14 (path_count) and 18 (path_tiling) are
// overwritten by path_count_setup and path_tiling_setup respectively.

import goldy_exp;
import ekrano_shared;

// Slots: 0 = WorkgroupCountsGpu (read-only), 1 = IndirectCount[] (write)

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main() {
    ReadOnlyBuffer<WorkgroupCountsGpu> wg = goldy_dyn_buf_ro<WorkgroupCountsGpu>(0);
    StorageBuffer<IndirectCount> indirect = goldy_dyn_scattered<IndirectCount>(1);

    WorkgroupCountsGpu counts = wg[0];

    for (uint i = 0; i < N_INDIRECT_STAGES; i++) {
        indirect[i].count_x = counts.entries[i].x;
        indirect[i].count_y = counts.entries[i].y;
        indirect[i].count_z = counts.entries[i].z;
    }
}
