// Yielding scripts: petitions, continuations, and resolved promises.
// Included from goldy_exp when GOLDY_YIELD is defined.
//
// A `[goldy_compute]` entry point may suspend a lane at a *yield point*:
//
//   [goldy_petition(Result = BufRO<BvhNode>)]
//   struct FetchBrick { uint brick_id; };
//
//   struct TraverseState { uint ray; uint node; };
//
//   [goldy_compute] [numthreads(64, 1, 1)]
//   void cs_main(Scattered<Ray> rays, BufRO<BvhNode> nodes, ThreadId tid) {
//       ...
//       $yield(cs_resume, FetchBrick { id }, TraverseState { tid.x, node });
//       return;
//   }
//
//   [goldy_resume] [numthreads(64, 1, 1)]
//   void cs_resume(Scattered<Ray> rays, Resolved<BvhNode> brick, TraverseState st, ThreadId tid) {
//       if (brick.is_null()) { ... } else { ... brick[0] ... }
//   }
//
// The Goldy compiler lowers `$yield` into an append onto a dispatch-private
// mailbox, and each `[goldy_resume]` function into its own compute entry point
// that reads one record per lane. `goldy_exp` includes this file when the
// compile defines `GOLDY_YIELD` (set automatically for yielding scripts and
// for shaders that call `goldy_resolve` / `goldy_reject`).

implementing goldy_exp;

/// Sentinel offset marking a rejected (null) promise.
static const uint GOLDY_RESOLVED_NULL = 0xFFFFFFFFu;

/// A promised buffer whose backing the handler provided: a read-only window of
/// `_len` elements starting at `_off` inside the yield point's result arena.
///
/// `is_null()` is true when the handler rejected the petition (or when the
/// arena ran out of room); indexing a null view is undefined.
public struct Resolved<T> {
    Scattered<T> _arena;
    uint _off;
    uint _len;

    [ForceInline]
    public __init(Scattered<T> arena, uint off, uint len) {
        _arena = arena;
        _off = off;
        _len = len;
    }

    /// True when the handler rejected this petition.
    [ForceInline]
    public bool is_null() { return _off == GOLDY_RESOLVED_NULL; }

    /// Element count of the resolved window (0 when null).
    [ForceInline]
    public uint len() { return is_null() ? 0u : _len; }

    public __subscript(uint i) -> T {
        [ForceInline]
        get { return _arena[_off + i]; }
    }
}

/// Resolution record written by a handler for one petition: `(offset, len)` in
/// arena elements, or `(GOLDY_RESOLVED_NULL, 0)` for a rejection.
///
/// GPU handlers (`Handler::node`) write one of these per petition; see
/// `goldy_resolve` / `goldy_reject`.
public typealias Resolution = uint2;

/// Fulfil petition `i` with `len` arena elements starting at `off`.
[ForceInline]
public void goldy_resolve(Scattered<Resolution> resolutions, uint i, uint off, uint len) {
    resolutions[i] = uint2(off, len);
}

/// Reject petition `i`: the continuation sees a null `Resolved<T>`.
[ForceInline]
public void goldy_reject(Scattered<Resolution> resolutions, uint i) {
    resolutions[i] = uint2(GOLDY_RESOLVED_NULL, 0u);
}
