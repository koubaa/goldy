// Compile-only test for goldy_exp Interlocked cells.
//
// Run via: cargo test test_goldy_exp_interlocked_compiles

import goldy_exp;

public struct Counters {
    public Interlocked<uint> u;
    public Interlocked<int> i;
}

groupshared Interlocked<uint> sh[64];

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<Counters> counters, ThreadId id) {
    InterlockedStore(sh[id.x], 0);
    GroupMemoryBarrierWithGroupSync();

    uint prev = InterlockedAdd(sh[id.x], 1);
    InterlockedOr(sh[id.x], 2);
    InterlockedXor(sh[id.x], 4);
    InterlockedMin(sh[id.x], 8);
    InterlockedMax(sh[id.x], 1);
    uint ex;
    InterlockedExchange(sh[id.x], prev, ex);

    InterlockedAdd(counters[0].u, 1);
    InterlockedOr(counters[0].u, 1);
    InterlockedStore(counters[0].u, InterlockedLoad(counters[0].u));

    InterlockedAdd(counters[0].i, 1);
    InterlockedMin(counters[0].i, 0);
    InterlockedMax(counters[0].i, 0);
    InterlockedStore(counters[0].i, InterlockedLoad(counters[0].i));
}
