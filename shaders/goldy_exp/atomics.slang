// Atomic helpers for Goldy shaders (EXPERIMENTAL)
// Part of the goldy_exp module
//
// RWByteAddressBuffer.InterlockedMin/Max use UNSIGNED comparison in HLSL/DXIL.
// For signed min/max (e.g. bbox aggregation with INT_MIN sentinels), use CAS loops.

implementing goldy_exp;

public void signed_atomic_min(RWByteAddressBuffer buf, uint addr, int new_val) {
    uint orig;
    buf.InterlockedCompareExchange(addr, 0, 0, orig);
    while (new_val < int(orig)) {
        uint expected = orig;
        buf.InterlockedCompareExchange(addr, expected, uint(new_val), orig);
        if (orig == expected) break;
    }
}

public void signed_atomic_max(RWByteAddressBuffer buf, uint addr, int new_val) {
    uint orig;
    buf.InterlockedCompareExchange(addr, 0, 0, orig);
    while (new_val > int(orig)) {
        uint expected = orig;
        buf.InterlockedCompareExchange(addr, expected, uint(new_val), orig);
        if (orig == expected) break;
    }
}
