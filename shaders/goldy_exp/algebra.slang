// Algebraic abstractions for parallel GPU algorithms (EXPERIMENTAL)
// Part of the goldy_exp module
//
// Provides:
//   IMonoid — interface for types with an associative binary operation and identity element.
//             The core abstraction for workgroup scan/reduce: any type conforming to IMonoid
//             can be used with generic parallel prefix operations.

implementing goldy_exp;

/// A monoid: an associative binary operation with an identity element.
///
/// Laws (not compiler-enforced, but algorithms depend on them):
///   combine(identity(), x) == x
///   combine(x, identity()) == x
///   combine(combine(x, y), z) == combine(x, combine(y, z))
///
/// Conformance example:
///   extension MyType : IMonoid {
///       static MyType identity() { MyType r; r.val = 0; return r; }
///       MyType combine(MyType other) { MyType r; r.val = this.val + other.val; return r; }
///   }
public interface IMonoid {
    static This identity();
    This combine(This other);
}
